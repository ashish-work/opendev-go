package websearch

import (
	"net/url"
	"strings"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

// Result is a single search hit returned to the model.
type Result struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet,omitempty"`
	Domain  string `json:"domain,omitempty"`
}

// parseResults walks a DuckDuckGo HTML response and extracts the
// (title, url, snippet) for every result block. Implementation note:
// we use golang.org/x/net/html as a DOM tokenizer rather than string-
// splitting on class names. The tokenizer handles DDG's occasionally
// malformed markup (unclosed tags, weird attribute quoting) without
// the parser silently breaking when DDG tweaks its templates.
//
// DDG's structure (stable for years): each result is a
//
//	<div class="result …">
//	  …
//	  <a class="result__a" href="<redirect-or-direct-url>">Title text</a>
//	  …
//	  <a class="result__snippet" …>Snippet text</a>
//	</div>
//
// We don't recurse into a result div once we find one — anchors inside
// the result are picked up by extractResult's local search, and
// avoiding the recursive descent saves O(depth) work per result.
func parseResults(body string) ([]Result, error) {
	doc, err := html.Parse(strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	var results []Result
	walkForResults(doc, &results)
	return results, nil
}

func walkForResults(n *html.Node, out *[]Result) {
	if n.Type == html.ElementNode && n.DataAtom == atom.Div && hasClass(n, "result") {
		if r, ok := extractResult(n); ok {
			*out = append(*out, r)
		}
		return
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		walkForResults(c, out)
	}
}

// extractResult locates the title and snippet anchors inside a result
// container. Returns ok=false when the title anchor or its href is
// missing — defensive against future DDG layout changes that we'd
// rather treat as "no result" than as garbage data.
func extractResult(container *html.Node) (Result, bool) {
	var r Result

	titleA := findFirst(container, func(n *html.Node) bool {
		return n.Type == html.ElementNode && n.DataAtom == atom.A && hasClass(n, "result__a")
	})
	if titleA == nil {
		return r, false
	}

	r.URL = unwrapDDGRedirect(getAttr(titleA, "href"))
	r.Title = strings.TrimSpace(textContent(titleA))

	snippetA := findFirst(container, func(n *html.Node) bool {
		return n.Type == html.ElementNode && n.DataAtom == atom.A && hasClass(n, "result__snippet")
	})
	if snippetA != nil {
		r.Snippet = strings.TrimSpace(textContent(snippetA))
	}

	if r.URL == "" || r.Title == "" {
		return r, false
	}
	r.Domain = extractDomain(r.URL)
	return r, true
}

// unwrapDDGRedirect peels off DDG's `//duckduckgo.com/l/?uddg=<target>`
// redirect wrapper to expose the actual destination URL. Also handles
// the rarer protocol-relative case (`//host/path`) by prepending https.
// Direct URLs pass through unchanged.
func unwrapDDGRedirect(raw string) string {
	if raw == "" {
		return ""
	}
	if strings.HasPrefix(raw, "//") {
		raw = "https:" + raw
	}
	if !strings.Contains(raw, "duckduckgo.com/l/") {
		return raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	target := u.Query().Get("uddg")
	if target == "" {
		return raw
	}
	return target
}

// extractDomain returns the lowercased hostname without a leading
// "www." prefix. Used for the allow/deny domain filter and surfaced
// in result metadata.
func extractDomain(rawURL string) string {
	if rawURL == "" {
		return ""
	}
	u, err := url.Parse(rawURL)
	if err != nil {
		return ""
	}
	host := strings.ToLower(u.Hostname())
	return strings.TrimPrefix(host, "www.")
}

// filterByDomain trims results by allowed/blocked domain lists.
// allowed acts as a whitelist (drop anything not on it); blocked acts
// as a blacklist (drop anything on it). Both lists are case-insensitive
// and treat `www.example.com` and `example.com` as equivalent.
// Subdomain matching is automatic: an allow rule of "example.com" also
// admits "docs.example.com" but not "fakeexample.com".
func filterByDomain(results []Result, allowed, blocked []string) []Result {
	if len(allowed) == 0 && len(blocked) == 0 {
		return results
	}
	out := make([]Result, 0, len(results))
	for _, r := range results {
		if r.Domain == "" {
			continue
		}
		if len(allowed) > 0 && !domainMatchesAny(r.Domain, allowed) {
			continue
		}
		if len(blocked) > 0 && domainMatchesAny(r.Domain, blocked) {
			continue
		}
		out = append(out, r)
	}
	return out
}

// domainMatchesAny returns true when domain equals or is a subdomain
// of any entry in patterns. Patterns are lowercased and stripped of a
// "www." prefix before comparison.
func domainMatchesAny(domain string, patterns []string) bool {
	for _, p := range patterns {
		clean := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(p)), "www.")
		if clean == "" {
			continue
		}
		if domain == clean || strings.HasSuffix(domain, "."+clean) {
			return true
		}
	}
	return false
}

// ---- DOM helpers ----

func hasClass(n *html.Node, want string) bool {
	for _, a := range n.Attr {
		if a.Key != "class" {
			continue
		}
		for _, c := range strings.Fields(a.Val) {
			if c == want {
				return true
			}
		}
	}
	return false
}

func getAttr(n *html.Node, key string) string {
	for _, a := range n.Attr {
		if a.Key == key {
			return a.Val
		}
	}
	return ""
}

func findFirst(n *html.Node, match func(*html.Node) bool) *html.Node {
	if match(n) {
		return n
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if found := findFirst(c, match); found != nil {
			return found
		}
	}
	return nil
}

// textContent collects all text nodes under n into a single
// whitespace-collapsed string. Used for both the title (where inline
// <b> highlighting is common) and the snippet (DDG tags matched
// terms with <b> wrappers).
func textContent(n *html.Node) string {
	var sb strings.Builder
	collectText(n, &sb)
	return strings.Join(strings.Fields(sb.String()), " ")
}

func collectText(n *html.Node, sb *strings.Builder) {
	if n.Type == html.TextNode {
		sb.WriteString(n.Data)
		sb.WriteString(" ")
		return
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		collectText(c, sb)
	}
}
