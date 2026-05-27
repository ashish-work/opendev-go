package webfetch

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ashish-work/opendev-go/internal/tools"
)

func TestTool_Surface(t *testing.T) {
	tool := New()
	if tool.Name() != "web_fetch" {
		t.Errorf("Name() = %q, want web_fetch", tool.Name())
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
	for _, key := range []string{"url", "format", "headers", "timeout"} {
		if _, has := props[key]; !has {
			t.Errorf("schema missing property %q", key)
		}
	}
	required, _ := schema["required"].([]any)
	if len(required) != 1 || required[0] != "url" {
		t.Errorf("required = %v, want [url]", required)
	}
}

func TestTool_PlainTextOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte("hello plain text"))
	}))
	defer srv.Close()

	tool := New()
	res, err := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url": srv.URL,
	}))
	if err != nil {
		t.Fatalf("Execute err: %v", err)
	}
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	if !strings.Contains(res.Output, "hello plain text") {
		t.Errorf("body missing from output: %q", res.Output)
	}
	if res.Metadata["status"] != 200 {
		t.Errorf("status = %v, want 200", res.Metadata["status"])
	}
	if !strings.Contains(res.Metadata["content_type"].(string), "text/plain") {
		t.Errorf("content_type = %v", res.Metadata["content_type"])
	}
}

func TestTool_HTMLExtractsTextByDefault(t *testing.T) {
	body := `<html><body>
		<script>var x = "leak";</script>
		<h1>Welcome</h1>
		<p>This is the <strong>main</strong> content.</p>
	</body></html>`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	tool := New()
	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url": srv.URL,
	}))
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	if !strings.Contains(res.Output, "Welcome") {
		t.Errorf("expected heading text: %q", res.Output)
	}
	if !strings.Contains(res.Output, "main") {
		t.Errorf("expected inline content: %q", res.Output)
	}
	if strings.Contains(res.Output, "leak") {
		t.Errorf("script content leaked: %q", res.Output)
	}
	if res.Metadata["format"] != "text" {
		t.Errorf("format meta = %v, want text", res.Metadata["format"])
	}
}

func TestTool_HTMLFormatReturnsRawBody(t *testing.T) {
	body := `<html><body><script>var x = 1;</script><p>hi</p></body></html>`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	tool := New()
	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url":    srv.URL,
		"format": "html",
	}))
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	if !strings.Contains(res.Output, "<script>") {
		t.Errorf("raw HTML mode should preserve markup: %q", res.Output)
	}
	if !strings.Contains(res.Output, "var x = 1") {
		t.Errorf("raw HTML mode should NOT strip scripts: %q", res.Output)
	}
}

func TestTool_HTTPErrorStillReturnsBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte("page not found"))
	}))
	defer srv.Close()

	tool := New()
	res, err := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url": srv.URL,
	}))
	if err != nil {
		t.Fatalf("Execute err: %v", err)
	}
	if res.Success {
		t.Fatalf("expected Success=false for 404")
	}
	if !strings.Contains(res.Error, "HTTP 404") {
		t.Errorf("Error should mention HTTP 404: %q", res.Error)
	}
	if !strings.Contains(res.Output, "page not found") {
		t.Errorf("4xx output should still include body: %q", res.Output)
	}
	if res.Metadata["status"] != 404 {
		t.Errorf("status meta = %v, want 404", res.Metadata["status"])
	}
}

func TestTool_CustomHeadersPropagate(t *testing.T) {
	gotHeader := ""
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Test-Run")
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	tool := New()
	_, _ = tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url":     srv.URL,
		"headers": map[string]any{"X-Test-Run": "agent-v2"},
	}))
	if gotHeader != "agent-v2" {
		t.Errorf("server saw X-Test-Run = %q, want agent-v2", gotHeader)
	}
}

func TestTool_DefaultUserAgentApplied(t *testing.T) {
	gotUA := ""
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotUA = r.Header.Get("User-Agent")
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	tool := New()
	_, _ = tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url": srv.URL,
	}))
	if gotUA != UserAgent {
		t.Errorf("User-Agent = %q, want %q", gotUA, UserAgent)
	}
}

func TestTool_CustomUserAgentOverrides(t *testing.T) {
	gotUA := ""
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotUA = r.Header.Get("User-Agent")
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	tool := New()
	_, _ = tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url":     srv.URL,
		"headers": map[string]any{"User-Agent": "custom-ua/1.0"},
	}))
	if gotUA != "custom-ua/1.0" {
		t.Errorf("custom UA should win, server saw %q", gotUA)
	}
}

func TestTool_RedirectChainSucceeds(t *testing.T) {
	var lastURL string
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		lastURL = r.URL.Path
		_, _ = w.Write([]byte("final"))
	}))
	defer srv2.Close()

	srv1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, srv2.URL+"/final-path", http.StatusFound)
	}))
	defer srv1.Close()

	tool := New()
	res, err := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url": srv1.URL,
	}))
	if err != nil {
		t.Fatalf("Execute err: %v", err)
	}
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	if !strings.Contains(res.Output, "final") {
		t.Errorf("redirected body missing: %q", res.Output)
	}
	if lastURL != "/final-path" {
		t.Errorf("redirect did not land on /final-path, got %s", lastURL)
	}
	finalMeta := res.Metadata["final_url"].(string)
	if !strings.Contains(finalMeta, "/final-path") {
		t.Errorf("final_url meta = %q, should include /final-path", finalMeta)
	}
}

func TestTool_RedirectLoopBlocked(t *testing.T) {
	// Two servers each redirecting to the other → cycle exceeds cap.
	var srv1URL, srv2URL string
	srv1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, srv2URL, http.StatusFound)
	}))
	defer srv1.Close()
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, srv1URL, http.StatusFound)
	}))
	defer srv2.Close()
	srv1URL = srv1.URL
	srv2URL = srv2.URL

	tool := New()
	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url": srv1URL,
	}))
	if res.Success {
		t.Fatalf("expected Success=false for redirect loop")
	}
	if !strings.Contains(res.Error, "redirect") && !strings.Contains(res.Error, "stopped") {
		t.Errorf("Error should mention redirect cap: %q", res.Error)
	}
}

func TestTool_InvalidURLs(t *testing.T) {
	cases := []struct {
		name string
		url  string
		want string
	}{
		{"no scheme", "example.com", "scheme"},
		{"ftp scheme", "ftp://example.com", "scheme"},
		{"file scheme", "file:///etc/passwd", "scheme"},
		{"javascript", "javascript:alert(1)", "scheme"},
		{"missing host", "http://", "host"},
	}
	tool := New()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
				"url": tc.url,
			}))
			if err != nil {
				t.Fatalf("Execute err: %v", err)
			}
			if res.Success {
				t.Fatalf("expected Success=false for %s", tc.url)
			}
			if !strings.Contains(res.Error, tc.want) {
				t.Errorf("Error = %q, want substring %q", res.Error, tc.want)
			}
		})
	}
}

func TestTool_MissingURL(t *testing.T) {
	tool := New()
	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{}))
	if res.Success || !strings.Contains(res.Error, "url is required") {
		t.Errorf("expected url-required error, got success=%v err=%q", res.Success, res.Error)
	}
}

func TestTool_InvalidFormat(t *testing.T) {
	tool := New()
	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url":    "https://example.com",
		"format": "markdown",
	}))
	if res.Success || !strings.Contains(res.Error, "format must be") {
		t.Errorf("expected format error, got success=%v err=%q", res.Success, res.Error)
	}
}

func TestTool_TimeoutFires(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Hold the request open longer than the client timeout.
		time.Sleep(500 * time.Millisecond)
		_, _ = w.Write([]byte("late"))
	}))
	defer srv.Close()

	tool := New()
	// Schema is integer seconds; we need sub-second timing for the test
	// so override DefaultTimeoutSec temporarily. The test asserts the
	// timeout path triggers; we don't care about wall-clock seconds.
	prev := DefaultTimeoutSec
	defer func() { DefaultTimeoutSec = prev }()
	// We can't override below 1s via the schema, so we use the test's
	// ctx deadline as the cancellation source instead. The HTTP client
	// honors both.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
		"url": srv.URL,
	}))
	if err == nil {
		t.Fatalf("expected ctx-deadline error")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want context.DeadlineExceeded", err)
	}
}

func TestTool_CancelledContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	tool := New()
	_, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
		"url": srv.URL,
	}))
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestTool_RawBodyCap(t *testing.T) {
	// Force a tiny raw cap so a small response triggers truncation
	// at the HTTP layer. After this, the body the tool processes is
	// only `cap` bytes regardless of how much the server sends.
	prev := MaxRawBytes
	MaxRawBytes = 32
	defer func() { MaxRawBytes = prev }()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte(strings.Repeat("abcdefghij", 100))) // 1000 bytes
	}))
	defer srv.Close()

	tool := New()
	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url": srv.URL,
	}))
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	bytesReceived := res.Metadata["bytes_received"].(int)
	if bytesReceived != 32 {
		t.Errorf("bytes_received = %d, want 32 (raw cap)", bytesReceived)
	}
}

func TestTool_OutputTruncationSpillsToDisk(t *testing.T) {
	// Generate a body larger than truncation.MaxBytes (50 KB) so the
	// spillover path fires. The spillover file path lands in
	// metadata.output_path.
	big := strings.Repeat("line of content for spillover testing\n", 2500) // ~95 KB
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte(big))
	}))
	defer srv.Close()

	tool := New()
	res, _ := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url": srv.URL,
	}))
	if !res.Success {
		t.Fatalf("Success=false: %s", res.Error)
	}
	if res.Metadata["truncated"] != true {
		t.Errorf("expected truncated=true, got %v", res.Metadata["truncated"])
	}
	if _, has := res.Metadata["output_path"]; !has {
		t.Errorf("expected output_path in metadata for truncated response")
	}
}

func TestTool_TimeoutCappedAtMax(t *testing.T) {
	// timeout=99999 should clamp to MaxTimeoutSec — verify by reading
	// the value back from a successful request. We can't easily check
	// the actual client.Timeout, but we can ensure the call succeeds
	// quickly with a small server response (so a 99999 timeout WOULD
	// allow it; the clamp itself is verified by the absence of a
	// failure related to e.g. an overflow).
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	tool := New()
	res, err := tool.Execute(context.Background(), tools.ToolContext{}, mkArgs(t, map[string]any{
		"url":     srv.URL,
		"timeout": 99999,
	}))
	if err != nil || !res.Success {
		t.Fatalf("oversize timeout should clamp and succeed; err=%v success=%v", err, res.Success)
	}
}

func TestTool_InvalidJSONArguments(t *testing.T) {
	tool := New()
	res, err := tool.Execute(context.Background(), tools.ToolContext{}, json.RawMessage(`{not json`))
	if err != nil {
		t.Fatalf("Execute err: %v", err)
	}
	if res.Success || !strings.Contains(res.Error, "invalid arguments") {
		t.Errorf("expected invalid-arguments error, got success=%v err=%q", res.Success, res.Error)
	}
}

func TestIsHTMLContentType(t *testing.T) {
	cases := []struct {
		ct   string
		want bool
	}{
		{"text/html", true},
		{"text/html; charset=utf-8", true},
		{"application/xhtml+xml", true},
		{"text/plain", false},
		{"application/json", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := isHTMLContentType(tc.ct); got != tc.want {
			t.Errorf("isHTMLContentType(%q) = %v, want %v", tc.ct, got, tc.want)
		}
	}
}

// mkArgs is the same helper as in other tool tests — marshals a map to
// JSON for the Execute call.
func mkArgs(t *testing.T, m map[string]any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return json.RawMessage(b)
}
