package match

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// NormalizeLineEndings
// -----------------------------------------------------------------------------

func TestNormalizeLineEndings(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"no CR/LF passthrough", "hello", "hello"},
		{"LF unchanged", "a\nb\nc", "a\nb\nc"},
		{"CRLF to LF", "a\r\nb\r\nc", "a\nb\nc"},
		{"lone CR to LF", "a\rb\rc", "a\nb\nc"},
		{"mixed CRLF and CR", "a\r\nb\rc\r\n", "a\nb\nc\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := NormalizeLineEndings(tc.in); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Simple pass
// -----------------------------------------------------------------------------

func TestSimpleExactMatch(t *testing.T) {
	got, ok := Simple("hello world", "world")
	if !ok || got != "world" {
		t.Errorf("got (%q, %v), want (%q, true)", got, ok, "world")
	}
}

func TestSimpleNoMatch(t *testing.T) {
	_, ok := Simple("hello", "xyz")
	if ok {
		t.Error("ok = true, want false")
	}
}

func TestSimpleEmptyOldMatchesEverything(t *testing.T) {
	// strings.Contains returns true for empty needle. The Simple pass
	// reflects that — the empty-old guard lives in editfile.Execute, not
	// here, because pass functions are intentionally pure.
	got, ok := Simple("anything", "")
	if !ok || got != "" {
		t.Errorf("got (%q, %v), want (%q, true)", got, ok, "")
	}
}

// -----------------------------------------------------------------------------
// LineTrimmed pass
// -----------------------------------------------------------------------------

func TestLineTrimmedMatchesWithExtraIndent(t *testing.T) {
	original := "    fn foo() {\n        return 1\n    }\n"
	old := "fn foo() {\nreturn 1\n}"
	got, ok := LineTrimmed(original, old)
	if !ok {
		t.Fatalf("ok = false, want true")
	}
	// Returned span is the untrimmed slice of original (preserves indent).
	if !strings.Contains(original, got) {
		t.Errorf("Actual %q not a substring of original", got)
	}
}

func TestLineTrimmedRejectsAllWhitespacePattern(t *testing.T) {
	_, ok := LineTrimmed("anything\nat all\n", "   \n\t\t\n  ")
	if ok {
		t.Error("ok = true on all-whitespace old; want false")
	}
}

func TestLineTrimmedNoMatchWhenContentDiffers(t *testing.T) {
	_, ok := LineTrimmed("foo\nbar\nbaz\n", "qux\nquux\n")
	if ok {
		t.Error("ok = true, want false")
	}
}

// -----------------------------------------------------------------------------
// similarity / BlockAnchor pass
// -----------------------------------------------------------------------------

func TestSimilarity(t *testing.T) {
	cases := []struct {
		a, b string
		want float64
	}{
		{"", "", 1.0},
		{"abc", "", 0.0},
		{"abc", "abc", 1.0},
		{"abc", "xyz", 0.0},
	}
	for _, c := range cases {
		got := similarity(c.a, c.b)
		if got != c.want {
			t.Errorf("similarity(%q, %q) = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}

func TestBlockAnchorRejectsTooFewLines(t *testing.T) {
	// Two-line patterns are handled by Simple/LineTrimmed.
	_, ok := BlockAnchor("a\nb\n", "a\nb")
	if ok {
		t.Error("ok = true on 2-line old; want false (BlockAnchor needs ≥3 lines)")
	}
}

func TestBlockAnchorMatchesWithDriftedMiddle(t *testing.T) {
	original := "fn process() {\n    let result = compute(x);\n}\n"
	old := "fn process() {\n    let result = transform(x);\n}"
	got, ok := BlockAnchor(original, old)
	if !ok {
		t.Fatalf("ok = false, want true")
	}
	if !strings.Contains(original, got) {
		t.Errorf("Actual %q not a substring of original", got)
	}
}

func TestBlockAnchorFailsWhenAnchorsAbsent(t *testing.T) {
	_, ok := BlockAnchor("foo\nbar\nbaz\n", "BEGIN\n  body\nEND")
	if ok {
		t.Error("ok = true, want false")
	}
}

// -----------------------------------------------------------------------------
// WhitespaceNormalized pass
// -----------------------------------------------------------------------------

func TestWhitespaceNormalizedCollapsesSpaces(t *testing.T) {
	// File uses tabs + double spaces; old uses single spaces. Both
	// normalize to "x = 1 + 2".
	original := "    x  =\t1  +  2\n"
	old := "x = 1 + 2"
	got, ok := WhitespaceNormalized(original, old)
	if !ok {
		t.Fatalf("ok = false, want true")
	}
	if !strings.Contains(original, got) {
		t.Errorf("Actual %q not a substring of original", got)
	}
}

func TestWhitespaceNormalizedNoMatch(t *testing.T) {
	// "x=1" has no whitespace; "x = 1" does. wsNormalize cannot ADD
	// whitespace, so this pass can't reconcile them.
	_, ok := WhitespaceNormalized("x=1\n", "x = 1")
	if ok {
		t.Error("ok = true, want false")
	}
}

// -----------------------------------------------------------------------------
// IndentationFlexible pass
// -----------------------------------------------------------------------------

func TestIndentationFlexibleSkipsBlankLines(t *testing.T) {
	original := "    a\n\n    b\n\n    c\n"
	old := "a\nb\nc"
	got, ok := IndentationFlexible(original, old)
	if !ok {
		t.Fatalf("ok = false, want true")
	}
	// Returned span MUST include the blank lines (contiguous slice).
	if !strings.Contains(got, "\n\n") {
		t.Errorf("Actual %q does not contain blank lines from original", got)
	}
}

func TestIndentationFlexibleAbortsOnMismatch(t *testing.T) {
	_, ok := IndentationFlexible("a\nb\nc\n", "a\nXXX\nc")
	if ok {
		t.Error("ok = true, want false (greedy walk should abort)")
	}
}

func TestIndentationFlexibleRejectsAllBlankOld(t *testing.T) {
	_, ok := IndentationFlexible("anything\n", "   \n\t\t\n   ")
	if ok {
		t.Error("ok = true, want false (all-blank old)")
	}
}

// -----------------------------------------------------------------------------
// EscapeNormalized pass
// -----------------------------------------------------------------------------

func TestEscapeNormalizedLiteralBackslashN(t *testing.T) {
	original := "line one\nline two\n"
	old := `line one\nline two` // backslash + n, not real newline
	got, ok := EscapeNormalized(original, old)
	if !ok {
		t.Fatalf("ok = false, want true")
	}
	if got != "line one\nline two" {
		t.Errorf("Actual = %q, want %q", got, "line one\nline two")
	}
}

func TestEscapeNormalizedNoOpWhenNothingToUnescape(t *testing.T) {
	_, ok := EscapeNormalized("no escapes here\n", "no escapes here")
	if ok {
		t.Error("ok = true on identity unescape; want false")
	}
}

// -----------------------------------------------------------------------------
// TrimmedBoundary pass
// -----------------------------------------------------------------------------

func TestTrimmedBoundaryOuterWhitespaceOnly(t *testing.T) {
	got, ok := TrimmedBoundary("x = 1\n", "\n\n  x = 1  \n\n")
	if !ok {
		t.Fatalf("ok = false, want true")
	}
	if got != "x = 1" {
		t.Errorf("Actual = %q, want %q", got, "x = 1")
	}
}

func TestTrimmedBoundaryRejectsEmptyAfterTrim(t *testing.T) {
	_, ok := TrimmedBoundary("anything\n", "\n\n\n")
	if ok {
		t.Error("ok = true on all-whitespace old; want false (defensive empty-match guard)")
	}
}

func TestTrimmedBoundaryNoOpWhenAlreadyTrimmed(t *testing.T) {
	_, ok := TrimmedBoundary("x = 1\n", "x = 1")
	if ok {
		t.Error("ok = true, want false (trim was a no-op — Simple already handled this)")
	}
}

// -----------------------------------------------------------------------------
// ContextAware pass
// -----------------------------------------------------------------------------

func TestContextAwareSubstringAnchors(t *testing.T) {
	original := "    fn begin_processing() {\n        helper();\n    }\n"
	old := "fn begin\n  helper();\n}"
	got, ok := ContextAware(original, old)
	if !ok {
		t.Fatalf("ok = false, want true")
	}
	if !strings.Contains(original, got) {
		t.Errorf("Actual %q not a substring of original", got)
	}
}

func TestContextAwareRejectsSingleLineOld(t *testing.T) {
	_, ok := ContextAware("anything", "single line")
	if ok {
		t.Error("ok = true on single-line old; want false (need first+last anchors)")
	}
}

// -----------------------------------------------------------------------------
// MultiOccurrence pass
// -----------------------------------------------------------------------------

func TestMultiOccurrenceOuterTrimAndPerLineTrim(t *testing.T) {
	original := "    line1\n    line2\n    line3\n"
	old := "\n\n  line1\n  line2\n  line3\n\n"
	got, ok := MultiOccurrence(original, old)
	if !ok {
		t.Fatalf("ok = false, want true")
	}
	if !strings.Contains(original, got) {
		t.Errorf("Actual %q not a substring of original", got)
	}
}

func TestMultiOccurrenceRejectsEmptyAfterTrim(t *testing.T) {
	_, ok := MultiOccurrence("anything\n", "\n\n\n")
	if ok {
		t.Error("ok = true, want false")
	}
}

func TestAllNinePassesRegistered(t *testing.T) {
	want := []string{
		"simple", "line_trimmed", "block_anchor",
		"whitespace_normalized", "indentation_flexible",
		"escape_normalized", "trimmed_boundary",
		"context_aware", "multi_occurrence",
	}
	if len(DefaultPasses) != len(want) {
		t.Fatalf("len(DefaultPasses) = %d, want %d", len(DefaultPasses), len(want))
	}
	for i, p := range DefaultPasses {
		if p.Name != want[i] {
			t.Errorf("DefaultPasses[%d].Name = %q, want %q", i, p.Name, want[i])
		}
	}
}

// -----------------------------------------------------------------------------
// Find / FindWith dispatch
// -----------------------------------------------------------------------------

func TestFindUsesSimpleFirst(t *testing.T) {
	res, ok := Find("the cat sat on the mat", "cat sat")
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if res.PassName != "simple" {
		t.Errorf("PassName = %q, want %q", res.PassName, "simple")
	}
	if res.Actual != "cat sat" {
		t.Errorf("Actual = %q, want %q", res.Actual, "cat sat")
	}
}

func TestFindReturnsFalseWhenNothingMatches(t *testing.T) {
	_, ok := Find("hello world", "completely unrelated content here")
	if ok {
		t.Error("ok = true, want false")
	}
}

func TestFindNormalizesLineEndings(t *testing.T) {
	// Input has CRLF; old has LF. Match must still succeed.
	original := "line1\r\nline2\r\nline3\r\n"
	old := "line2"
	res, ok := Find(original, old)
	if !ok {
		t.Fatal("ok = false after CRLF normalization, want true")
	}
	if res.Actual != "line2" {
		t.Errorf("Actual = %q, want %q", res.Actual, "line2")
	}
}

func TestFindWithCustomPasses(t *testing.T) {
	// Verify FindWith respects the passes parameter. With only the
	// Simple pass available, a multi-line pattern whose interior
	// whitespace differs from the file should NOT match — Simple
	// requires byte-exact substring containment.
	passes := []Pass{{Name: "simple", Fn: Simple}}
	_, ok := FindWith(passes, "fn x() {\n\treturn 1\n}\n", "fn x() {\nreturn 1\n}")
	if ok {
		t.Error("ok = true with Simple-only passes; expected no match")
	}
}

// -----------------------------------------------------------------------------
// CountOccurrences
// -----------------------------------------------------------------------------

func TestCountOccurrences(t *testing.T) {
	cases := []struct {
		name     string
		haystack string
		needle   string
		want     int
	}{
		{"single match", "hello world", "world", 1},
		{"multiple matches", "abc abc abc", "abc", 3},
		{"no match", "abc", "xyz", 0},
		{"empty needle", "abc", "", 4}, // strings.Count semantics
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := CountOccurrences(tc.haystack, tc.needle); got != tc.want {
				t.Errorf("got %d, want %d", got, tc.want)
			}
		})
	}
}
