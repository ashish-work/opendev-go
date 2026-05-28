package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// newTestClient returns a Client wired to a test HTTP server. The
// server's URL replaces the OpenAI BaseURL so no network egress occurs.
func newTestClient(t *testing.T, handler http.HandlerFunc) (*Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	c := NewClient("test-key")
	c.Adapter.BaseURL = srv.URL
	return c, srv
}

func TestClientCall_RoundTrip(t *testing.T) {
	var (
		gotMethod      string
		gotPath        string
		gotAuth        string
		gotContentType string
		gotBody        []byte
	)

	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")
		gotBody, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-rt",
			"model": "gpt-4o",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "hi"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
		}`))
	})

	resp, err := c.Call(context.Background(), provider.Request{
		Model: "gpt-4o",
		Messages: []provider.Message{
			{Role: "user", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "hello"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}

	// Response shape
	if resp.Content != "hi" {
		t.Errorf("Content = %q, want %q", resp.Content, "hi")
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, "stop")
	}
	if resp.Usage.PromptTokens != 5 || resp.Usage.CompletionTokens != 1 {
		t.Errorf("Usage = %+v, want {Prompt:5 Completion:1}", resp.Usage)
	}

	// Request inspection
	if gotMethod != http.MethodPost {
		t.Errorf("method = %q, want POST", gotMethod)
	}
	if gotPath != "/chat/completions" {
		t.Errorf("path = %q, want /chat/completions", gotPath)
	}
	if gotAuth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want %q", gotAuth, "Bearer test-key")
	}
	if gotContentType != "application/json" {
		t.Errorf("Content-Type = %q, want %q", gotContentType, "application/json")
	}

	// Body should be valid JSON containing our model + user text.
	var parsed map[string]any
	if err := json.Unmarshal(gotBody, &parsed); err != nil {
		t.Fatalf("request body not JSON: %v\nbody: %s", err, gotBody)
	}
	if parsed["model"] != "gpt-4o" {
		t.Errorf(`body["model"] = %v, want "gpt-4o"`, parsed["model"])
	}
	if !strings.Contains(string(gotBody), `"hello"`) {
		t.Errorf("body missing user message text; got: %s", gotBody)
	}
}

func TestClientCall_HTTP401(t *testing.T) {
	c, _ := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"Invalid API key","code":"invalid_api_key"}}`))
	})

	_, err := c.Call(context.Background(), provider.Request{
		Model:    "gpt-4o",
		Messages: []provider.Message{{Role: "user"}},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	var httpErr *HTTPError
	if !errors.As(err, &httpErr) {
		t.Fatalf("err type = %T, want *HTTPError; err: %v", err, err)
	}
	if httpErr.Status != http.StatusUnauthorized {
		t.Errorf("Status = %d, want 401", httpErr.Status)
	}
	if !strings.Contains(httpErr.Body, "Invalid API key") {
		t.Errorf("Body missing expected text; got: %q", httpErr.Body)
	}
}

func TestClientCall_HTTP500_TruncatesLargeBody(t *testing.T) {
	bigBody := strings.Repeat("X", 5000) // larger than the 500-char truncation limit

	c, _ := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, bigBody)
	})

	_, err := c.Call(context.Background(), provider.Request{Model: "gpt-4o"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	var httpErr *HTTPError
	if !errors.As(err, &httpErr) {
		t.Fatalf("err type = %T, want *HTTPError", err)
	}
	if httpErr.Status != http.StatusInternalServerError {
		t.Errorf("Status = %d, want 500", httpErr.Status)
	}
	// Full body preserved on the struct...
	if len(httpErr.Body) != len(bigBody) {
		t.Errorf("Body length = %d, want %d (full body)", len(httpErr.Body), len(bigBody))
	}
	// ...but truncated in the formatted error message.
	msg := httpErr.Error()
	if !strings.Contains(msg, "[truncated]") {
		t.Errorf("Error() did not truncate large body; msg: %s", msg)
	}
	if len(msg) > 600 { // 500 chars + framing
		t.Errorf("Error() too long: %d chars", len(msg))
	}
}

func TestClientCall_ContextCancellation(t *testing.T) {
	// Handler sleeps; we cancel before it can respond.
	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"choices":[]}`))
		}
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancelled

	_, err := c.Call(ctx, provider.Request{
		Model:    "gpt-4o",
		Messages: []provider.Message{{Role: "user"}},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want wraps context.Canceled", err)
	}
}

func TestNewClientDefaults(t *testing.T) {
	c := NewClient("k")
	if c.APIKey != "k" {
		t.Errorf("APIKey = %q, want %q", c.APIKey, "k")
	}
	if c.HTTPClient == nil {
		t.Error("HTTPClient is nil, want a default *http.Client")
	}
	if c.HTTPClient.Timeout != DefaultHTTPTimeout {
		t.Errorf("Timeout = %v, want %v", c.HTTPClient.Timeout, DefaultHTTPTimeout)
	}
	if c.Adapter.BaseURL != DefaultBaseURL {
		t.Errorf("BaseURL = %q, want %q", c.Adapter.BaseURL, DefaultBaseURL)
	}
}

func TestClientCall_NilHTTPClientFallsBackToDefault(t *testing.T) {
	c, _ := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"choices": [{"index": 0,
				"message": {"role": "assistant", "content": "ok"},
				"finish_reason": "stop"}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
		}`))
	})
	c.HTTPClient = nil // exercise the fallback path

	resp, err := c.Call(context.Background(), provider.Request{
		Model:    "gpt-4o",
		Messages: []provider.Message{{Role: "user"}},
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want %q", resp.Content, "ok")
	}
}
