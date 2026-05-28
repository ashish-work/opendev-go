package anthropic

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
// server's URL replaces the Anthropic BaseURL so no network egress
// occurs.
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
		gotMethod  string
		gotPath    string
		gotAPIKey  string
		gotVersion string
		gotBody    []byte
	)
	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotAPIKey = r.Header.Get("x-api-key")
		gotVersion = r.Header.Get("anthropic-version")
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_rt","type":"message","role":"assistant","model":"claude-3-5",
			"stop_reason":"end_turn",
			"content":[{"type":"text","text":"hi"}],
			"usage":{"input_tokens":5,"output_tokens":1}
		}`))
	})

	resp, err := c.Call(context.Background(), provider.Request{
		Model: "claude-3-5-sonnet",
		Messages: []provider.Message{
			{Role: "user", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "hello"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}

	if resp.Content != "hi" {
		t.Errorf("Content = %q, want %q", resp.Content, "hi")
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Usage.PromptTokens != 5 || resp.Usage.CompletionTokens != 1 {
		t.Errorf("Usage = %+v", resp.Usage)
	}

	// Request inspection
	if gotMethod != http.MethodPost {
		t.Errorf("method = %q, want POST", gotMethod)
	}
	if gotPath != "/messages" {
		t.Errorf("path = %q, want /messages", gotPath)
	}
	if gotAPIKey != "test-key" {
		t.Errorf("x-api-key = %q, want test-key", gotAPIKey)
	}
	if gotVersion != DefaultAPIVersion {
		t.Errorf("anthropic-version = %q, want %q", gotVersion, DefaultAPIVersion)
	}

	var parsed map[string]any
	if err := json.Unmarshal(gotBody, &parsed); err != nil {
		t.Fatalf("body not JSON: %v\nbody: %s", err, gotBody)
	}
	if parsed["model"] != "claude-3-5-sonnet" {
		t.Errorf(`body["model"] = %v`, parsed["model"])
	}
	if !strings.Contains(string(gotBody), `"hello"`) {
		t.Errorf("body missing user text; got: %s", gotBody)
	}
}

func TestClientCall_NoAuthorizationBearerHeader(t *testing.T) {
	// Anthropic uses x-api-key, NOT Authorization: Bearer.
	// Pin that explicitly so a future copy-paste from openai/client.go
	// doesn't silently break authentication.
	var gotBearer string
	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotBearer = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`))
	})
	_, _ = c.Call(context.Background(), provider.Request{Model: "claude"})
	if gotBearer != "" {
		t.Errorf("Authorization header set to %q; Anthropic uses x-api-key only", gotBearer)
	}
}

func TestClientCall_CustomAPIVersionHeader(t *testing.T) {
	var gotVersion string
	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotVersion = r.Header.Get("anthropic-version")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`))
	})
	c.Adapter.APIVersion = "2099-12-31-beta"
	_, _ = c.Call(context.Background(), provider.Request{Model: "claude"})
	if gotVersion != "2099-12-31-beta" {
		t.Errorf("anthropic-version = %q, want override", gotVersion)
	}
}

func TestClientCall_HTTP401(t *testing.T) {
	c, _ := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"invalid key"}}`))
	})

	_, err := c.Call(context.Background(), provider.Request{
		Model:    "claude",
		Messages: []provider.Message{{Role: "user"}},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) {
		t.Fatalf("err type = %T, want *HTTPError", err)
	}
	if httpErr.Status != http.StatusUnauthorized {
		t.Errorf("Status = %d, want 401", httpErr.Status)
	}
	if !strings.Contains(httpErr.Body, "invalid key") {
		t.Errorf("Body missing expected text: %q", httpErr.Body)
	}
}

func TestClientCall_HTTP500_TruncatesLargeBody(t *testing.T) {
	bigBody := strings.Repeat("X", 5000)
	c, _ := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, bigBody)
	})

	_, err := c.Call(context.Background(), provider.Request{Model: "claude"})
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) {
		t.Fatalf("err type = %T, want *HTTPError", err)
	}
	if len(httpErr.Body) != len(bigBody) {
		t.Errorf("full body should be preserved on the struct: %d vs %d", len(httpErr.Body), len(bigBody))
	}
	msg := httpErr.Error()
	if !strings.Contains(msg, "[truncated]") {
		t.Errorf("Error() should truncate large body; got: %s", msg)
	}
}

func TestClientCall_ContextCancellation(t *testing.T) {
	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
			w.WriteHeader(http.StatusOK)
		}
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := c.Call(ctx, provider.Request{Model: "claude"})
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
		t.Errorf("APIKey = %q", c.APIKey)
	}
	if c.HTTPClient == nil || c.HTTPClient.Timeout != DefaultHTTPTimeout {
		t.Errorf("HTTPClient = %+v", c.HTTPClient)
	}
	if c.Adapter.BaseURL != DefaultBaseURL {
		t.Errorf("Adapter.BaseURL = %q, want default", c.Adapter.BaseURL)
	}
	if c.Adapter.APIVersion != DefaultAPIVersion {
		t.Errorf("Adapter.APIVersion = %q, want default", c.Adapter.APIVersion)
	}
}

func TestClientCall_NilHTTPClientFallsBackToDefault(t *testing.T) {
	c, _ := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`))
	})
	c.HTTPClient = nil
	resp, err := c.Call(context.Background(), provider.Request{Model: "claude"})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q", resp.Content)
	}
}

func TestClient_Name(t *testing.T) {
	if c := NewClient("k"); c.Name() != "anthropic" {
		t.Errorf("Name = %q, want anthropic", c.Name())
	}
}
