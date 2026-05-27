package websearch

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// loadFixture reads testdata/ddg_results.html. Failing here marks the
// test fatal because every parser test depends on the fixture.
func loadFixture(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "ddg_results.html"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return string(data)
}

func TestParseResults_AllFiveExtracted(t *testing.T) {
	results, err := parseResults(loadFixture(t))
	if err != nil {
		t.Fatalf("parseResults: %v", err)
	}
	if len(results) != 5 {
		t.Fatalf("got %d results, want 5", len(results))
	}
}

func TestParseResults_TitleSnippetURL(t *testing.T) {
	results, _ := parseResults(loadFixture(t))
	first := results[0]
	if first.Title != "Effective Go - The Go Programming Language" {
		t.Errorf("title = %q, want %q", first.Title,
			"Effective Go - The Go Programming Language")
	}
	if first.URL != "https://go.dev/doc/effective_go" {
		t.Errorf("url = %q, want https://go.dev/doc/effective_go", first.URL)
	}
	if !strings.Contains(first.Snippet, "writing clear, idiomatic Go code") {
		t.Errorf("snippet missing expected text: %q", first.Snippet)
	}
	if first.Domain != "go.dev" {
		t.Errorf("domain = %q, want go.dev", first.Domain)
	}
}

func TestParseResults_DecodesEntitiesAndStripsInlineTags(t *testing.T) {
	results, _ := parseResults(loadFixture(t))
	// Result 2 ("Stack Overflow") has <b> tags inside the title and snippet.
	r := results[1]
	if strings.Contains(r.Title, "<b>") {
		t.Errorf("inline <b> tags should be stripped from title: %q", r.Title)
	}
	if !strings.Contains(r.Title, "context cancellation") {
		t.Errorf("title content lost: %q", r.Title)
	}
	if strings.Contains(r.Snippet, "<b>") {
		t.Errorf("inline <b> tags should be stripped from snippet: %q", r.Snippet)
	}

	// Result 4 has &amp; in the title which should decode to &.
	r4 := results[3]
	if !strings.Contains(r4.Title, "patterns & best practices") {
		t.Errorf("entity decoding failed for &amp; in title: %q", r4.Title)
	}
	// And &lt; / &gt; in the snippet should decode to < / >.
	if !strings.Contains(r4.Snippet, "<goroutine>") {
		t.Errorf("entity decoding failed for &lt;goroutine&gt; in snippet: %q", r4.Snippet)
	}
}

func TestParseResults_UnwrapsDDGRedirects(t *testing.T) {
	results, _ := parseResults(loadFixture(t))
	for _, r := range results[:4] {
		if strings.Contains(r.URL, "duckduckgo.com/l/") {
			t.Errorf("redirect not unwrapped for %s: %q", r.Title, r.URL)
		}
		if strings.Contains(r.URL, "uddg=") {
			t.Errorf("uddg param leaked: %q", r.URL)
		}
	}
}

func TestParseResults_ProtocolRelativeURL(t *testing.T) {
	// Result 5 has a protocol-relative URL (//github.com/...) without
	// a DDG redirect wrapper. Should still resolve to https://.
	results, _ := parseResults(loadFixture(t))
	r := results[4]
	if !strings.HasPrefix(r.URL, "https://github.com/") {
		t.Errorf("protocol-relative URL not promoted: %q", r.URL)
	}
	if r.Domain != "github.com" {
		t.Errorf("domain extraction wrong: %q, want github.com", r.Domain)
	}
}

func TestParseResults_EmptyHTML(t *testing.T) {
	results, err := parseResults("")
	if err != nil {
		t.Fatalf("empty input should not error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("got %d results from empty HTML, want 0", len(results))
	}
}

func TestParseResults_NoResultsBlock(t *testing.T) {
	html := `<html><body><div class="error">No results</div></body></html>`
	results, _ := parseResults(html)
	if len(results) != 0 {
		t.Errorf("got %d results from no-result page, want 0", len(results))
	}
}

func TestParseResults_MalformedHTML(t *testing.T) {
	// Unclosed tags everywhere — DOM parser should still produce sensible
	// output rather than panicking.
	html := `<div class="result"><a class="result__a" href="//example.com/x">title<a class="result__snippet">snip`
	results, err := parseResults(html)
	if err != nil {
		t.Fatalf("malformed HTML errored: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results from malformed input, want 1", len(results))
	}
}

func TestUnwrapDDGRedirect(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		// Standard DDG redirect — protocol-relative wrapper with uddg
		{
			"//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc",
			"https://example.com/page",
		},
		// Already absolute with uddg
		{
			"https://duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev",
			"https://go.dev",
		},
		// Protocol-relative direct URL (no /l/ wrapper)
		{
			"//github.com/foo/bar",
			"https://github.com/foo/bar",
		},
		// Already absolute non-DDG URL — pass through
		{
			"https://go.dev/doc",
			"https://go.dev/doc",
		},
		// Malformed (no uddg param) — return as-is
		{
			"//duckduckgo.com/l/?other=value",
			"https://duckduckgo.com/l/?other=value",
		},
		// Empty
		{"", ""},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			if got := unwrapDDGRedirect(tt.in); got != tt.want {
				t.Errorf("unwrapDDGRedirect(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestExtractDomain(t *testing.T) {
	tests := []struct {
		url  string
		want string
	}{
		{"https://example.com/page", "example.com"},
		{"https://www.example.com/page", "example.com"},
		{"https://docs.example.com/page", "docs.example.com"},
		{"http://Example.COM/", "example.com"},
		{"https://example.com:8080/x", "example.com"},
		{"", ""},
		{"not a url", ""},
	}
	for _, tt := range tests {
		t.Run(tt.url, func(t *testing.T) {
			if got := extractDomain(tt.url); got != tt.want {
				t.Errorf("extractDomain(%q) = %q, want %q", tt.url, got, tt.want)
			}
		})
	}
}

func TestFilterByDomain_AllowOnly(t *testing.T) {
	in := []Result{
		{URL: "https://go.dev/x", Domain: "go.dev"},
		{URL: "https://stackoverflow.com/y", Domain: "stackoverflow.com"},
		{URL: "https://docs.go.dev/z", Domain: "docs.go.dev"},
	}
	out := filterByDomain(in, []string{"go.dev"}, nil)
	if len(out) != 2 {
		t.Fatalf("got %d results, want 2 (go.dev and docs.go.dev)", len(out))
	}
}

func TestFilterByDomain_BlockOnly(t *testing.T) {
	in := []Result{
		{URL: "https://go.dev/x", Domain: "go.dev"},
		{URL: "https://stackoverflow.com/y", Domain: "stackoverflow.com"},
		{URL: "https://meta.stackoverflow.com/z", Domain: "meta.stackoverflow.com"},
	}
	out := filterByDomain(in, nil, []string{"stackoverflow.com"})
	if len(out) != 1 || out[0].Domain != "go.dev" {
		t.Errorf("got %+v, want only go.dev", out)
	}
}

func TestFilterByDomain_AllowedAndBlocked(t *testing.T) {
	in := []Result{
		{URL: "https://go.dev/x", Domain: "go.dev"},
		{URL: "https://golang.org/y", Domain: "golang.org"},
		{URL: "https://example.com/z", Domain: "example.com"},
	}
	out := filterByDomain(in, []string{"go.dev", "golang.org"}, []string{"golang.org"})
	// allowed admits go.dev + golang.org; blocked removes golang.org;
	// final: just go.dev.
	if len(out) != 1 || out[0].Domain != "go.dev" {
		t.Errorf("got %+v, want only go.dev", out)
	}
}

func TestFilterByDomain_WWWStripped(t *testing.T) {
	in := []Result{
		{URL: "https://www.example.com/x", Domain: "example.com"},
	}
	// "www.example.com" in the filter should match "example.com".
	out := filterByDomain(in, []string{"www.example.com"}, nil)
	if len(out) != 1 {
		t.Errorf("www-prefix on filter pattern should match: got %+v", out)
	}
}

func TestFilterByDomain_NoFilters(t *testing.T) {
	in := []Result{
		{URL: "https://x.com", Domain: "x.com"},
		{URL: "https://y.com", Domain: "y.com"},
	}
	out := filterByDomain(in, nil, nil)
	if len(out) != 2 {
		t.Errorf("no filters should pass everything through")
	}
}
