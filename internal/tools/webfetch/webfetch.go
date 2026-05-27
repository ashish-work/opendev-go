// Package webfetch implements the web_fetch tool — the model's way to
// pull content from HTTP(S) URLs. First tool in the codebase that
// talks to the network.
//
// Output flow:
//
//  1. HTTP GET with capped read (10 MB raw) and capped redirects (5).
//  2. If format=text and content is HTML, run htmlToText for clean,
//     LLM-readable output (golang.org/x/net/html DOM parser, not
//     regex — handles malformed markup correctly).
//  3. Pipe through truncation.Truncate so long pages spill to disk
//     under ~/.opendev/tool-output/ and the model receives a preview
//     + read_file hint, exactly like big bash output.
//
// Safety surface: this is the first place the agent reaches outside
// the user's machine. URL scheme is restricted to http/https; no
// allowlist beyond that. A future permissions task can layer
// per-domain rules.
package webfetch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/tools/truncation"
)

// ToolName is the canonical name the model uses to invoke this tool.
const ToolName = "web_fetch"

// Tunable limits exposed as package vars so tests can shrink them.
var (
	// MaxRawBytes caps the raw HTTP body we read into memory before
	// any conversion. 10 MB is more than enough for any reasonable
	// text page and protects against pathological URLs (multi-GB
	// downloads, infinite streams) crashing the process.
	MaxRawBytes int64 = 10 * 1024 * 1024

	// DefaultTimeoutSec is used when the caller omits "timeout".
	DefaultTimeoutSec = 30

	// MaxTimeoutSec caps a caller-supplied "timeout" so the model
	// can't stall a turn for arbitrarily long.
	MaxTimeoutSec = 120

	// MaxRedirects caps redirect chains. The HTTP client returns an
	// error past this; we surface it as a failed fetch.
	MaxRedirects = 5

	// UserAgent identifies the agent in HTTP requests. Stable so
	// callers can write rate-limit rules against it.
	UserAgent = "opendev-go/v0.2"
)

// Tool implements tools.Tool for web_fetch. Stateless — concurrent
// calls are safe because http.Client is goroutine-safe and we
// construct a fresh one per call (cheap; pooling lives at the
// transport layer inside net/http).
type Tool struct{}

// New returns a ready-to-register Tool.
func New() *Tool { return &Tool{} }

// Compile-time guards.
var (
	_ tools.Tool        = (*Tool)(nil)
	_ tools.Categorized = (*Tool)(nil)
)

// Name implements tools.Tool.
func (t *Tool) Name() string { return ToolName }

// Category implements tools.Categorized.
func (t *Tool) Category() tools.Category { return tools.CategoryWeb }

// Description is the model's only authoritative source for web_fetch
// semantics; the system prompt no longer carries per-tool sections.
func (t *Tool) Description() string {
	return "Fetch the contents of an HTTP or HTTPS URL. Use this for reading " +
		"documentation pages, release notes, API specs, blog posts, or any " +
		"web content you need to inspect — preferred over `bash curl` because " +
		"it handles redirects, content-type detection, HTML extraction, and " +
		"output truncation as a single operation. " +
		"Parameters: url (required, must start with http:// or https://), " +
		"format (\"text\" default — HTML stripped to readable text; or " +
		"\"html\" for the raw response body), headers (optional object of " +
		"extra request headers — e.g. {\"Authorization\": \"Bearer ...\"}), " +
		"timeout (seconds; default 30, max 120). " +
		"GET only. Redirects are followed up to 5 deep. Raw response is " +
		"capped at 10 MB; output is then run through the standard truncation " +
		"flow so long pages spill to ~/.opendev/tool-output/ and you receive " +
		"a preview plus a read_file path for specific sections. " +
		"HTTP status codes 4xx/5xx return Success:false; the response body is " +
		"still included so you can see the error page. Do NOT retry the same " +
		"URL repeatedly on 4xx — the issue is on the server side or with the " +
		"URL itself; relay the failure to the user."
}

// Schema is the JSON Schema for the tool-call arguments.
func (t *Tool) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"url": {
				"type": "string",
				"description": "URL to fetch. Must start with http:// or https://."
			},
			"format": {
				"type": "string",
				"enum": ["text", "html"],
				"description": "Output format. text (default) strips HTML to clean readable text via a DOM parser; html returns the raw response body."
			},
			"headers": {
				"type": "object",
				"description": "Optional request headers as a flat string→string map (e.g. {\"Authorization\": \"Bearer ...\"}).",
				"additionalProperties": {"type": "string"}
			},
			"timeout": {
				"type": "integer",
				"description": "Request timeout in seconds. Default 30, capped at 120."
			}
		},
		"required": ["url"]
	}`)
}

type args struct {
	URL     string            `json:"url"`
	Format  string            `json:"format,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
	Timeout int               `json:"timeout,omitempty"`
}

// fetchResult bundles the bits we care about from the HTTP layer so
// the rest of Execute reads as one flow without re-extracting fields.
type fetchResult struct {
	status      int
	contentType string
	finalURL    string
	body        []byte
}

// Execute performs the HTTP GET, optionally converts HTML to text, and
// pipes the output through truncation. Tool-domain failures (bad URL,
// HTTP non-2xx) return ToolResult{Success:false}. Infrastructure
// failures (ctx cancellation, transport errors) surface as Go errors.
func (t *Tool) Execute(ctx context.Context, tctx tools.ToolContext, raw json.RawMessage) (tools.ToolResult, error) {
	var a args
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &a); err != nil {
			return failf("invalid arguments: %v", err), nil
		}
	}

	if a.URL == "" {
		return failf("url is required"), nil
	}
	if err := validateURL(a.URL); err != nil {
		return failf("%v", err), nil
	}

	format := strings.ToLower(a.Format)
	if format == "" {
		format = "text"
	}
	if format != "text" && format != "html" {
		return failf("format must be \"text\" or \"html\", got %q", a.Format), nil
	}

	timeoutSec := a.Timeout
	if timeoutSec <= 0 {
		timeoutSec = DefaultTimeoutSec
	}
	if timeoutSec > MaxTimeoutSec {
		timeoutSec = MaxTimeoutSec
	}
	timeout := time.Duration(timeoutSec) * time.Second

	if err := ctx.Err(); err != nil {
		return tools.ToolResult{}, err
	}

	fr, err := doFetch(ctx, a.URL, a.Headers, timeout)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return tools.ToolResult{}, err
		}
		return failf("fetch %s: %v", a.URL, err), nil
	}

	// Decide whether to extract text. Only does work for HTML
	// content; otherwise the bytes are returned verbatim.
	bodyStr := string(fr.body)
	if format == "text" && isHTMLContentType(fr.contentType) {
		extracted, err := htmlToText(strings.NewReader(bodyStr))
		if err == nil && strings.TrimSpace(extracted) != "" {
			bodyStr = extracted
		}
		// On error or empty extraction we keep the raw body — better
		// to give the model something than nothing.
	}

	trunc := truncation.Truncate(bodyStr, 0, 0, truncation.Head)

	meta := map[string]any{
		"status":         fr.status,
		"content_type":   fr.contentType,
		"final_url":      fr.finalURL,
		"bytes_received": len(fr.body),
		"format":         format,
		"truncated":      trunc.Truncated,
	}
	if trunc.OutputPath != "" {
		meta["output_path"] = trunc.OutputPath
	}

	if fr.status >= 400 {
		return tools.ToolResult{
			Success:  false,
			Output:   trunc.Content,
			Error:    fmt.Sprintf("HTTP %d from %s", fr.status, a.URL),
			Metadata: meta,
		}, nil
	}

	return tools.ToolResult{
		Success:  true,
		Output:   trunc.Content,
		Metadata: meta,
	}, nil
}

// doFetch performs the HTTP request and returns the captured bits.
// Separated from Execute so the test suite can exercise the request
// path against an httptest server without going through the full
// arg-parsing flow.
func doFetch(ctx context.Context, rawURL string, headers map[string]string, timeout time.Duration) (*fetchResult, error) {
	client := &http.Client{
		Timeout: timeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= MaxRedirects {
				return fmt.Errorf("stopped after %d redirects", MaxRedirects)
			}
			return nil
		},
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("User-Agent", UserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Cap raw bytes read so a multi-GB URL can't OOM us. LimitReader
	// hands EOF when the limit is hit; we don't distinguish "actually
	// done" from "we stopped early" in the body itself, but
	// bytes_received in metadata reflects what we actually got.
	body, err := io.ReadAll(io.LimitReader(resp.Body, MaxRawBytes))
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	finalURL := rawURL
	if resp.Request != nil && resp.Request.URL != nil {
		finalURL = resp.Request.URL.String()
	}

	return &fetchResult{
		status:      resp.StatusCode,
		contentType: resp.Header.Get("Content-Type"),
		finalURL:    finalURL,
		body:        body,
	}, nil
}

// validateURL accepts only http(s) URLs that parse cleanly. ftp://,
// file://, javascript:, missing scheme, etc. all reject — those are
// either out-of-scope for an HTTP fetcher or actively unsafe.
func validateURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("invalid url: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("unsupported url scheme %q (only http and https are allowed)", u.Scheme)
	}
	if u.Host == "" {
		return fmt.Errorf("url is missing a host")
	}
	return nil
}

// isHTMLContentType detects whether to run htmlToText. We're generous:
// anything with "html" or "xhtml" in the Content-Type substring counts.
// text/plain stays raw.
func isHTMLContentType(ct string) bool {
	ct = strings.ToLower(ct)
	return strings.Contains(ct, "html") || strings.Contains(ct, "xhtml")
}

func failf(format string, args ...any) tools.ToolResult {
	return tools.ToolResult{
		Success: false,
		Error:   fmt.Sprintf(format, args...),
	}
}
