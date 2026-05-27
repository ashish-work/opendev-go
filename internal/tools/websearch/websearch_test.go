package websearch

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ashish-work/opendev-go/internal/tools"
)

// fixtureServer spins up an httptest server that returns the DDG
// fixture body for any request. Returns a *Tool already pointed at
// that server's URL — call srv.Close() in a defer in the test.
func fixtureServer(t *testing.T) (*Tool, *httptest.Server) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "ddg_results.html"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(data)
	}))
	return &Tool{Endpoint: srv.URL + "/"}, srv
}

func TestTool_Surface(t *testing.T) {
	tool := New()
	if tool.Name() != "web_search" {
		t.Errorf("Name() = %q, want web_search", tool.Name())
	}
	if tool.Category() != tools.CategoryWeb {
		t.Errorf("Category() = %v, want CategoryWeb", tool.Category())
	}
	var schema map[string]any
	if err := json.Unmarshal(tool.Schema(), &schema); err != nil {
		t.Fatalf("Schema returned invalid JSON: %v", err)
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema missing properties")
	}
	for _, key := range []string{"query", "max_results", "allowed_domains", "blocked_domains"} {
		if _, has := props[key]; !has {
			t.Errorf("schema missing property %q", key)
		}
	}
	required, _ := schema["required"].([]any)
	if len(required) != 1 || required[0] != "query" {
		t.Errorf("required = %v, want [query]", required)
	}
}

func TestTool_BasicSearch(t *testing.T) {
	tool, srv := fixtureServer(t)
	defer srv.Close()

	res, err := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"query": "go context cancellation",
	}))
	if err != nil {
		t.Fatalf("Execute err: %v", err)
	}
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	for _, want := range []string{
		"Search results for",
		"go context cancellation",
		"1. Effective Go",
		"go.dev/doc/effective_go",
		"5. golang/go: src/context/context.go",
	} {
		if !strings.Contains(res.Output, want) {
			t.Errorf("output missing %q:\n%s", want, res.Output)
		}
	}
	if res.Metadata["result_count"] != 5 {
		t.Errorf("result_count meta = %v, want 5", res.Metadata["result_count"])
	}
}

func TestTool_MaxResultsCaps(t *testing.T) {
	tool, srv := fixtureServer(t)
	defer srv.Close()

	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"query":       "anything",
		"max_results": 2,
	}))
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	if res.Metadata["result_count"] != 2 {
		t.Errorf("result_count = %v, want 2", res.Metadata["result_count"])
	}
	if strings.Contains(res.Output, "3.") {
		t.Errorf("max_results=2 should not produce a 3rd entry:\n%s", res.Output)
	}
}

func TestTool_MaxResultsClampsToMaxMax(t *testing.T) {
	tool, srv := fixtureServer(t)
	defer srv.Close()

	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"query":       "anything",
		"max_results": 9999, // clamp to MaxMaxResults; fixture has 5
	}))
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	if res.Metadata["result_count"] != 5 {
		t.Errorf("expected all 5 fixture results, got %v", res.Metadata["result_count"])
	}
}

func TestTool_AllowedDomains(t *testing.T) {
	tool, srv := fixtureServer(t)
	defer srv.Close()

	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"query":           "anything",
		"allowed_domains": []any{"go.dev"},
	}))
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	if res.Metadata["result_count"] != 2 { // go.dev + pkg.go.dev (subdomain match)
		t.Errorf("result_count = %v, want 2 (go.dev + pkg.go.dev)", res.Metadata["result_count"])
	}
	if strings.Contains(res.Output, "stackoverflow.com") {
		t.Errorf("stackoverflow should be filtered out:\n%s", res.Output)
	}
}

func TestTool_BlockedDomains(t *testing.T) {
	tool, srv := fixtureServer(t)
	defer srv.Close()

	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"query":           "anything",
		"blocked_domains": []any{"stackoverflow.com"},
	}))
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	if strings.Contains(res.Output, "stackoverflow.com") {
		t.Errorf("stackoverflow should be blocked:\n%s", res.Output)
	}
	if !strings.Contains(res.Output, "go.dev") {
		t.Errorf("non-blocked domains should pass through:\n%s", res.Output)
	}
}

func TestTool_MissingQuery(t *testing.T) {
	tool, srv := fixtureServer(t)
	defer srv.Close()

	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{}))
	if res.Success || !strings.Contains(res.Error, "query is required") {
		t.Errorf("expected query-required error, got success=%v err=%q", res.Success, res.Error)
	}
}

func TestTool_WhitespaceOnlyQuery(t *testing.T) {
	tool, srv := fixtureServer(t)
	defer srv.Close()

	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"query": "    \t  ",
	}))
	if res.Success || !strings.Contains(res.Error, "query is required") {
		t.Errorf("whitespace-only query should be rejected: success=%v err=%q", res.Success, res.Error)
	}
}

func TestTool_DDGNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	tool := &Tool{Endpoint: srv.URL + "/"}

	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"query": "anything",
	}))
	if res.Success {
		t.Fatalf("expected Success=false on DDG 503")
	}
	if !strings.Contains(res.Error, "HTTP 503") {
		t.Errorf("Error should mention HTTP 503: %q", res.Error)
	}
}

func TestTool_CancelledContext(t *testing.T) {
	tool, srv := fixtureServer(t)
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
		"query": "anything",
	}))
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestTool_TimeoutFires(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(500 * time.Millisecond)
		_, _ = w.Write([]byte("late"))
	}))
	defer srv.Close()
	tool := &Tool{Endpoint: srv.URL + "/"}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
		"query": "anything",
	}))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want context.DeadlineExceeded", err)
	}
}

func TestTool_QueryEncoded(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.URL.RawQuery
		_, _ = w.Write([]byte("<html></html>"))
	}))
	defer srv.Close()
	tool := &Tool{Endpoint: srv.URL + "/"}

	_, _ = tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"query": "go context cancellation",
	}))
	if !strings.Contains(got, "q=go+context+cancellation") &&
		!strings.Contains(got, "q=go%20context%20cancellation") {
		t.Errorf("expected encoded query in URL, got: %q", got)
	}
}

func TestTool_DefaultUserAgentSent(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("User-Agent")
		_, _ = w.Write([]byte("<html></html>"))
	}))
	defer srv.Close()
	tool := &Tool{Endpoint: srv.URL + "/"}

	_, _ = tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"query": "x",
	}))
	if got != UserAgent {
		t.Errorf("User-Agent = %q, want %q", got, UserAgent)
	}
	if !strings.Contains(got, "Chrome") {
		t.Errorf("UA should look like a browser (contain Chrome), got %q", got)
	}
}

func TestTool_NoResultsResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`<html><body><div class="error">No results</div></body></html>`))
	}))
	defer srv.Close()
	tool := &Tool{Endpoint: srv.URL + "/"}

	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"query": "asdfqwerzxcv",
	}))
	if !res.Success {
		t.Fatalf("Success=false (empty result should still succeed): %s", res.Error)
	}
	if !strings.Contains(res.Output, "no results found") {
		t.Errorf("expected no-results marker, got: %s", res.Output)
	}
	if res.Metadata["result_count"] != 0 {
		t.Errorf("result_count = %v, want 0", res.Metadata["result_count"])
	}
}

func TestTool_InvalidJSONArguments(t *testing.T) {
	tool := New()
	res, err := tool.Execute(context.Background(), tools.ToolContext{}, json.RawMessage(`{bad`))
	if err != nil {
		t.Fatalf("Execute err: %v", err)
	}
	if res.Success || !strings.Contains(res.Error, "invalid arguments") {
		t.Errorf("expected invalid-arguments error, got success=%v err=%q", res.Success, res.Error)
	}
}

func TestTool_OutputFormat(t *testing.T) {
	tool, srv := fixtureServer(t)
	defer srv.Close()

	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"query":       "anything",
		"max_results": 2,
	}))
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	// Output has a header line, then for each result: "N. Title",
	// "   URL", "   snippet". We don't pin exact line indices (blank
	// separators between results vary), just the shape.
	out := res.Output
	if !strings.Contains(out, "Search results for") {
		t.Errorf("missing header in output:\n%s", out)
	}
	if !strings.Contains(out, "\n1. ") {
		t.Errorf("missing numbered first entry:\n%s", out)
	}
	if !strings.Contains(out, "\n2. ") {
		t.Errorf("missing numbered second entry:\n%s", out)
	}
	// Each URL line should be 3-space-indented (per formatResults).
	if !strings.Contains(out, "\n   https://") {
		t.Errorf("expected indented URL lines:\n%s", out)
	}
}

// mkArgs marshals a map to JSON for the Execute call.
func mkArgs(t *testing.T, m map[string]any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return json.RawMessage(b)
}
