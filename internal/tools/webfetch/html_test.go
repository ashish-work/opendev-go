package webfetch

import (
	"strings"
	"testing"
)

func TestHtmlToText_PlainParagraphs(t *testing.T) {
	in := `<html><body><p>Hello world.</p><p>Second paragraph.</p></body></html>`
	got, err := htmlToText(strings.NewReader(in))
	if err != nil {
		t.Fatalf("htmlToText: %v", err)
	}
	if !strings.Contains(got, "Hello world.") || !strings.Contains(got, "Second paragraph.") {
		t.Errorf("missing paragraph text: %q", got)
	}
	if !strings.Contains(got, "\n\n") {
		t.Errorf("paragraphs should be separated by a blank line: %q", got)
	}
}

func TestHtmlToText_StripsScripts(t *testing.T) {
	in := `<html><body>
		<script>var leak = "should not appear";</script>
		<style>body { color: red }</style>
		<p>visible</p>
	</body></html>`
	got, _ := htmlToText(strings.NewReader(in))
	if strings.Contains(got, "leak") {
		t.Errorf("script content leaked into output: %q", got)
	}
	if strings.Contains(got, "color: red") {
		t.Errorf("style content leaked into output: %q", got)
	}
	if !strings.Contains(got, "visible") {
		t.Errorf("paragraph text missing: %q", got)
	}
}

func TestHtmlToText_DecodesEntities(t *testing.T) {
	in := `<p>5 &lt; 10 &amp;&amp; 10 &gt; 5</p>`
	got, _ := htmlToText(strings.NewReader(in))
	if !strings.Contains(got, "5 < 10 && 10 > 5") {
		t.Errorf("entities not decoded: %q", got)
	}
}

func TestHtmlToText_HeadingsSeparate(t *testing.T) {
	in := `<h1>Title</h1><p>body text</p><h2>Subtitle</h2><p>more</p>`
	got, _ := htmlToText(strings.NewReader(in))
	for _, want := range []string{"Title", "body text", "Subtitle", "more"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in: %q", want, got)
		}
	}
	// Title and body text should be on different lines.
	if !strings.Contains(got, "Title\n") && !strings.Contains(got, "Title\n\n") {
		t.Errorf("heading not separated from following text: %q", got)
	}
}

func TestHtmlToText_LinksUnwrap(t *testing.T) {
	// Anchor text should be preserved; the href URL is dropped.
	// This matches our "clean readable text" goal — the model can
	// always re-fetch a link by name if it needs the URL.
	in := `<p>See <a href="https://go.dev/doc">the docs</a> for more.</p>`
	got, _ := htmlToText(strings.NewReader(in))
	if !strings.Contains(got, "the docs") {
		t.Errorf("link text missing: %q", got)
	}
	if strings.Contains(got, "https://go.dev/doc") {
		t.Errorf("href should not appear in text output: %q", got)
	}
}

func TestHtmlToText_ListsAsLines(t *testing.T) {
	in := `<ul><li>first</li><li>second</li><li>third</li></ul>`
	got, _ := htmlToText(strings.NewReader(in))
	for _, item := range []string{"first", "second", "third"} {
		if !strings.Contains(got, item) {
			t.Errorf("list item %q missing: %q", item, got)
		}
	}
}

func TestHtmlToText_NestedDivsDoNotProduceBlankFloods(t *testing.T) {
	// Many sites wrap content in 5+ layers of <div>. Naive newline
	// insertion would produce huge runs of blank lines.
	in := `<div><div><div><p>actual content</p></div></div></div>`
	got, _ := htmlToText(strings.NewReader(in))
	// At most two consecutive newlines anywhere (one blank line).
	if strings.Contains(got, "\n\n\n") {
		t.Errorf("more than one blank line in output:\n%q", got)
	}
	if !strings.Contains(got, "actual content") {
		t.Errorf("content missing")
	}
}

func TestHtmlToText_MalformedHTML(t *testing.T) {
	// Unclosed tags, mismatched nesting — the DOM parser should still
	// produce reasonable text without erroring.
	in := `<p>first<div>second<p>third</p>`
	got, err := htmlToText(strings.NewReader(in))
	if err != nil {
		t.Fatalf("malformed HTML errored: %v", err)
	}
	for _, want := range []string{"first", "second", "third"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q from malformed input: %q", want, got)
		}
	}
}

func TestHtmlToText_NonHTMLLooksFine(t *testing.T) {
	// HTML parser tolerates non-HTML; should return the text content.
	in := `just a plain string with no tags`
	got, _ := htmlToText(strings.NewReader(in))
	if !strings.Contains(got, "just a plain string") {
		t.Errorf("plain string not preserved: %q", got)
	}
}

func TestHtmlToText_Empty(t *testing.T) {
	got, err := htmlToText(strings.NewReader(""))
	if err != nil {
		t.Fatalf("empty input errored: %v", err)
	}
	if got != "" {
		t.Errorf("empty input should produce empty output, got %q", got)
	}
}

func TestHtmlToText_TableCellsSpace(t *testing.T) {
	in := `<table><tr><td>cell1</td><td>cell2</td></tr><tr><td>cell3</td><td>cell4</td></tr></table>`
	got, _ := htmlToText(strings.NewReader(in))
	for _, want := range []string{"cell1", "cell2", "cell3", "cell4"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q: %q", want, got)
		}
	}
}

func TestHtmlToText_WhitespaceCollapsed(t *testing.T) {
	in := "<p>multiple     spaces\t\tbetween    words</p>"
	got, _ := htmlToText(strings.NewReader(in))
	if !strings.Contains(got, "multiple spaces between words") {
		t.Errorf("whitespace not collapsed: %q", got)
	}
}
