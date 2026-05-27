package openai

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/ashishgupta/opendev-go/internal/provider"
)

// Compile-time assertion that *Client satisfies provider.Provider.
// If the Provider interface changes, this line fails to build and we
// catch the drift immediately rather than at the first runtime call.
var _ provider.Provider = (*Client)(nil)

// DefaultHTTPTimeout caps a single LLM call. LLM responses occasionally
// take 30+ seconds for long completions; 60s is a comfortable margin
// without letting truly hung requests pile up forever.
const DefaultHTTPTimeout = 60 * time.Second

// Client is the OpenAI Chat Completions HTTP transport. It composes
// Adapter (pure wire-format conversion, no I/O) with an http.Client
// (does the actual POST). Implements provider.Provider.
//
// Pointer-receiver methods because the type holds a *http.Client and
// callers occasionally tweak fields (BaseURL, APIKey) after construction.
type Client struct {
	// Adapter converts to/from OpenAI Chat Completions JSON. Exported
	// so callers can override BaseURL for proxies and tests.
	Adapter Adapter

	// APIKey is sent as the Bearer token. Must be non-empty for the
	// public OpenAI endpoint; some local servers ignore auth.
	APIKey string

	// HTTPClient is the transport. nil falls back to http.DefaultClient.
	// Override to add proxies, TLS config, retry transports, etc.
	HTTPClient *http.Client
}

// NewClient returns a Client targeting the public OpenAI API with a
// sensible default HTTP timeout. Override fields directly for custom
// proxies (Adapter.BaseURL) or longer/shorter timeouts (HTTPClient).
func NewClient(apiKey string) *Client {
	return &Client{
		Adapter:    New(),
		APIKey:     apiKey,
		HTTPClient: &http.Client{Timeout: DefaultHTTPTimeout},
	}
}

// Name implements provider.Provider.
func (c *Client) Name() string { return c.Adapter.Name() }

// Call implements provider.Provider. The flow is:
//
//  1. Adapter.BuildRequest(req) → JSON bytes.
//  2. POST to Adapter.ChatCompletionsURL() with Bearer auth.
//  3. On non-2xx, return *HTTPError carrying status + (truncated) body.
//  4. Adapter.ParseResponse(body) → provider.Response.
//
// ctx flows through http.NewRequestWithContext so timeouts and
// cancellations propagate to the transport.
func (c *Client) Call(ctx context.Context, req provider.Request) (provider.Response, error) {
	body, err := c.Adapter.BuildRequest(req)
	if err != nil {
		return provider.Response{}, fmt.Errorf("openai: build request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		c.Adapter.ChatCompletionsURL(),
		bytes.NewReader(body),
	)
	if err != nil {
		return provider.Response{}, fmt.Errorf("openai: new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return provider.Response{}, fmt.Errorf("openai: HTTP: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return provider.Response{}, fmt.Errorf("openai: read body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return provider.Response{}, &HTTPError{
			Status: resp.StatusCode,
			Body:   string(respBody),
		}
	}

	return c.Adapter.ParseResponse(respBody)
}

// HTTPError is returned by Call for any non-2xx response. Use errors.As
// to extract status code and body in callers that want to react
// differently to 401 (auth) vs 429 (rate limit) vs 5xx (transient).
type HTTPError struct {
	Status int
	Body   string
}

// Error implements the error interface. Body is truncated to keep log
// lines manageable when the server returns large HTML error pages.
func (e *HTTPError) Error() string {
	return fmt.Sprintf("openai: HTTP %d: %s", e.Status, truncate(e.Body, 500))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "...[truncated]"
}
