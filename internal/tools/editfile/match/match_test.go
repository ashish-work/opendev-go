package match

import (
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
