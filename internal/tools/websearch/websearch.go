// Package websearch implements the web_search tool — the model's way
// to search the web without an API key. Hits DuckDuckGo's HTML
// interface (https://html.duckduckgo.com/html/), parses results out
// of the response, and returns (title, url, snippet) triples.
//
// Privacy + cost: no API key, no rate limit (gentle traffic), no
// per-query billing. DDG is the reasonable default here; if a user
// later wants Google/Brave/Kagi, those become sibling tools rather
// than parameters on this one.
//
// The HTTP layer in this file is intentionally NOT shared with
// web_fetch's HTTP code. Two HTTP-fetching tools is a small enough
// duplication that "wait for the third use case" is the right
// trade-off — extracting a shared helper today would generalize from
// a sample of two, which usually produces awkward abstractions. If
// the codebase grows a third HTTP-driven tool, that's the natural
// moment to extract.
package websearch

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
)

// ToolName is the canonical name the model uses to invoke this tool.
const ToolName = "web_search"

// Tunable settings exposed as package vars so tests can shrink limits.
var (
	// DefaultMaxResults is used when the caller omits max_results.
	DefaultMaxResults = 10

	// MaxMaxResults caps the caller-supplied max_results — the model
	// can't ask for thousands of hits even if it wants to. DDG itself
	// returns ~30 per page; going beyond that needs pagination which
	// isn't worth the complexity for this version.
	MaxMaxResults = 50

	// MaxBodyBytes caps the raw HTML body we read. DDG pages stay
	// well under this; the limit is defensive against a misbehaving
	// proxy or future DDG layout change that bloats the response.
	MaxBodyBytes int64 = 256 * 1024

	// HTTPTimeout caps the whole request. DDG usually responds in
	// <2s; 15s is a generous buffer.
	HTTPTimeout = 15 * time.Second

	// MaxRedirects caps redirect chains.
	MaxRedirects = 5

	// DefaultEndpoint is the DDG HTML search URL. The Tool struct's
	// Endpoint field overrides this for tests (httptest.NewServer).
	DefaultEndpoint = "https://html.duckduckgo.com/html/"

	// UserAgent is a real Chrome string. Without it DDG often serves
	// blank/minimal pages — it sniffs for bot traffic. The exact
	// value doesn't matter much as long as it looks like a browser.
	UserAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

// Tool implements tools.Tool for web_search. Stateless apart from the
// optional Endpoint override.
type Tool struct {
	// Endpoint overrides DefaultEndpoint. Empty uses the default.
	// Tests point this at an httptest.NewServer URL so we can verify
	// parsing without depending on live DDG.
	Endpoint string
}

// New returns a ready-to-register Tool that uses DefaultEndpoint.
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

// Description is the model's only authoritative source for web_search
// semantics; the system prompt no longer carries per-tool sections.
func (t *Tool) Description() string {
	return "Search the web via DuckDuckGo's HTML interface. Use this when you " +
		"need to find URLs to fetch with web_fetch, look up version numbers, " +
		"track down documentation, or generally discover what's online about " +
		"a topic. Privacy-respecting, no API key required. " +
		"Parameters: query (required, the search string — trim whitespace; " +
		"the tool refuses empty queries), max_results (default 10, max 50), " +
		"allowed_domains (only include results from these domains; matches " +
		"subdomains automatically — \"example.com\" admits \"docs.example.com\"), " +
		"blocked_domains (drop results from these domains; same subdomain " +
		"semantics). " +
		"Output is a numbered list of (title, url, snippet) — one per result. " +
		"The model's natural next move on any interesting result is web_fetch " +
		"on its URL. " +
		"Limitations: DuckDuckGo's HTML endpoint sometimes serves slightly " +
		"different result counts than its main interface, and very recent " +
		"news may not surface immediately. If a search returns zero results, " +
		"try alternative phrasings (the model has seen the equivalents) " +
		"rather than retrying the exact same query."
}

// Schema is the JSON Schema for the tool-call arguments.
func (t *Tool) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"query": {
				"type": "string",
				"description": "Search query string. Whitespace-trimmed; empty queries are rejected."
			},
			"max_results": {
				"type": "integer",
				"description": "Maximum results to return. Default 10, max 50."
			},
			"allowed_domains": {
				"type": "array",
				"items": {"type": "string"},
				"description": "Only include results from these domains. Subdomain match is automatic — \"example.com\" matches \"docs.example.com\"."
			},
			"blocked_domains": {
				"type": "array",
				"items": {"type": "string"},
				"description": "Drop results from these domains. Same subdomain semantics as allowed_domains."
			}
		},
		"required": ["query"]
	}`)
}

type args struct {
	Query          string   `json:"query"`
	MaxResults     int      `json:"max_results,omitempty"`
	AllowedDomains []string `json:"allowed_domains,omitempty"`
	BlockedDomains []string `json:"blocked_domains,omitempty"`
}

// Execute runs the search. Tool-domain failures (empty query, parse
// failure) return ToolResult{Success:false}. Infrastructure failures
// (ctx cancellation, transport errors, DDG non-2xx) surface as Go
// errors or Success:false depending on whether the model can act on
// them.
func (t *Tool) Execute(ctx context.Context, tctx tools.ToolContext, raw json.RawMessage) (tools.ToolResult, error) {
	var a args
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &a); err != nil {
			return failf("invalid arguments: %v", err), nil
		}
	}

	a.Query = strings.TrimSpace(a.Query)
	if a.Query == "" {
		return failf("query is required and must not be empty"), nil
	}

	maxResults := a.MaxResults
	if maxResults <= 0 {
		maxResults = DefaultMaxResults
	}
	if maxResults > MaxMaxResults {
		maxResults = MaxMaxResults
	}

	if err := ctx.Err(); err != nil {
		return tools.ToolResult{}, err
	}

	endpoint := t.Endpoint
	if endpoint == "" {
		endpoint = DefaultEndpoint
	}

	body, err := fetchDDG(ctx, endpoint, a.Query)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return tools.ToolResult{}, err
		}
		return failf("search request failed: %v", err), nil
	}

	results, err := parseResults(body)
	if err != nil {
		return failf("parse results: %v", err), nil
	}

	results = filterByDomain(results, a.AllowedDomains, a.BlockedDomains)
	if len(results) > maxResults {
		results = results[:maxResults]
	}

	output := formatResults(a.Query, results)

	meta := map[string]any{
		"query":        a.Query,
		"result_count": len(results),
		"results":      results,
	}

	return tools.ToolResult{
		Success:  true,
		Output:   output,
		Metadata: meta,
	}, nil
}

// fetchDDG performs the GET against DDG's HTML endpoint. Caller is
// responsible for context plumbing. Returns the raw body as a string;
// any non-2xx HTTP status surfaces as a Go error so Execute can
// distinguish "we got data" from "the search failed".
func fetchDDG(ctx context.Context, endpoint, query string) (string, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", fmt.Errorf("parse endpoint: %w", err)
	}
	q := u.Query()
	q.Set("q", query)
	u.RawQuery = q.Encode()

	client := &http.Client{
		Timeout: HTTPTimeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= MaxRedirects {
				return fmt.Errorf("stopped after %d redirects", MaxRedirects)
			}
			return nil
		},
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("User-Agent", UserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("DuckDuckGo returned HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, MaxBodyBytes))
	if err != nil {
		return "", fmt.Errorf("read body: %w", err)
	}
	return string(body), nil
}

// formatResults produces the numbered-list output the model sees.
// Plain text is enough — the model handles markdown, JSON, and prose
// equally well, and plain text is easier for the user to read in the
// REPL transcript.
func formatResults(query string, results []Result) string {
	if len(results) == 0 {
		return fmt.Sprintf("Search results for %q: (no results found)", query)
	}
	var b strings.Builder
	plural := "results"
	if len(results) == 1 {
		plural = "result"
	}
	fmt.Fprintf(&b, "Search results for %q (%d %s):\n\n", query, len(results), plural)
	for i, r := range results {
		fmt.Fprintf(&b, "%d. %s\n   %s\n", i+1, r.Title, r.URL)
		if r.Snippet != "" {
			fmt.Fprintf(&b, "   %s\n", r.Snippet)
		}
		if i < len(results)-1 {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func failf(format string, args ...any) tools.ToolResult {
	return tools.ToolResult{
		Success: false,
		Error:   fmt.Sprintf(format, args...),
	}
}
