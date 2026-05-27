package webfetch

import (
	"io"
	"regexp"
	"strings"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

// skipTags are subtrees we drop entirely during text extraction — they
// contribute nothing readable for an LLM and would dilute the useful
// content. We walk INTO body elements and skip these at the boundary.
//
// atom.Atom values are the lowercase tag name as an enum; checking the
// node's DataAtom against this set is cheaper than a string compare.
var skipTags = map[atom.Atom]bool{
	atom.Script:   true,
	atom.Style:    true,
	atom.Head:     true,
	atom.Noscript: true,
	atom.Svg:      true,
	atom.Iframe:   true,
	atom.Template: true,
}

// blockTags get a newline boundary in the output so the result reads
// like prose rather than one long line. Inline elements (span, a, em,
// strong, code, etc.) join with spaces.
var blockTags = map[atom.Atom]bool{
	atom.P:          true,
	atom.Div:        true,
	atom.Section:    true,
	atom.Article:    true,
	atom.Header:     true,
	atom.Footer:     true,
	atom.Nav:        true,
	atom.Aside:      true,
	atom.Main:       true,
	atom.Li:         true,
	atom.Tr:         true,
	atom.Td:         true,
	atom.Th:         true,
	atom.Br:         true,
	atom.Hr:         true,
	atom.Pre:        true,
	atom.Blockquote: true,
	atom.H1:         true,
	atom.H2:         true,
	atom.H3:         true,
	atom.H4:         true,
	atom.H5:         true,
	atom.H6:         true,
	atom.Title:      true,
	atom.Form:       true,
	atom.Figure:     true,
	atom.Figcaption: true,
}

// majorBreaks get TWO newlines around them (a blank line in output).
// These are the elements that meaningfully separate "thoughts" in the
// rendered page.
var majorBreaks = map[atom.Atom]bool{
	atom.P:          true,
	atom.Pre:        true,
	atom.Hr:         true,
	atom.H1:         true,
	atom.H2:         true,
	atom.H3:         true,
	atom.H4:         true,
	atom.H5:         true,
	atom.H6:         true,
	atom.Blockquote: true,
	atom.Article:    true,
	atom.Section:    true,
}

// multiNewline collapses runs of 3+ newlines to exactly two, so the
// output has at most one blank line between paragraphs. Compiled once
// at package init; safe to use concurrently.
var multiNewline = regexp.MustCompile(`\n{3,}`)

// horizontalWhitespace collapses runs of spaces/tabs to a single space
// within a line. Applied per-line so it doesn't eat newlines.
var horizontalWhitespace = regexp.MustCompile(`[ \t]+`)

// htmlToText reads HTML from r and returns plain text suitable for an
// LLM to read. The general approach:
//
//  1. Parse via golang.org/x/net/html into a tree (handles malformed
//     HTML correctly — that's the whole point of using the parser).
//  2. Walk the tree, skipping subtrees we don't want (script/style/svg).
//  3. Emit text nodes verbatim (entities are already decoded by the
//     parser — &amp; → &).
//  4. Insert newlines around block elements, double newlines around
//     major-break elements.
//  5. Post-process: collapse horizontal whitespace, collapse 3+
//     consecutive newlines to two, trim.
//
// The parser tolerates broken HTML — that's the whole point of using a
// DOM parser instead of regex. If parsing returns an error (very rare
// — it returns errors only for I/O failures on the reader), we fall
// back to returning the input unchanged.
func htmlToText(r io.Reader) (string, error) {
	doc, err := html.Parse(r)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	walk(doc, &b)
	return cleanup(b.String()), nil
}

// walk recursively descends into n, emitting text and structural
// whitespace into b. Tag-driven control flow lives entirely here so
// the calling function stays a one-liner.
func walk(n *html.Node, b *strings.Builder) {
	if n.Type == html.ElementNode && skipTags[n.DataAtom] {
		return
	}

	if n.Type == html.TextNode {
		// Text nodes carry already-decoded entity content. We don't
		// pre-strip whitespace here — cleanup() handles that — so we
		// don't accidentally glue words from adjacent inline tags.
		b.WriteString(n.Data)
		return
	}

	if n.Type == html.ElementNode {
		// Emit pre-element separator for block tags. Major breaks get
		// a blank line; ordinary blocks just a newline.
		if majorBreaks[n.DataAtom] {
			b.WriteString("\n\n")
		} else if blockTags[n.DataAtom] {
			b.WriteString("\n")
		}
	}

	for c := n.FirstChild; c != nil; c = c.NextSibling {
		walk(c, b)
	}

	// Post-element separator mirrors the pre-element one so a sequence
	// of <p>s renders cleanly without explicit closing markers.
	if n.Type == html.ElementNode {
		if majorBreaks[n.DataAtom] {
			b.WriteString("\n\n")
		} else if blockTags[n.DataAtom] {
			b.WriteString("\n")
		}
	}
}

// cleanup tidies the assembled text. Two passes:
//
//  1. Per-line: collapse runs of spaces/tabs to a single space and
//     trim each line's edges.
//  2. Whole-string: collapse 3+ consecutive newlines to exactly two,
//     trim outer whitespace.
//
// This produces output that reads naturally — paragraphs separated by
// a blank line, no rivers of empty lines from nested empty divs.
func cleanup(s string) string {
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimSpace(horizontalWhitespace.ReplaceAllString(line, " "))
	}
	joined := strings.Join(lines, "\n")
	joined = multiNewline.ReplaceAllString(joined, "\n\n")
	return strings.TrimSpace(joined)
}
