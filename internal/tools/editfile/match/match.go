// Package match implements the fuzzy-matching chain used by edit_file.
// Each pass is a pure function with the same signature:
//
//	func(original, old string) (actual string, ok bool)
//
// Passes are tried strictest-to-most-flexible; the first hit wins. The
// "actual" return is the bytes in `original` that matched, which may
// differ from `old` due to whitespace tolerance — we replace `actual`
// (not `old`) so the surrounding code keeps its real indentation.
//
// Passes are deliberately small and self-contained: each one targets a
// specific kind of input drift an LLM commonly produces (trailing
// whitespace, wrong indentation, escaped quotes, etc.). They are added
// incrementally so each one can be studied in isolation. This first
// commit lands the dispatch infrastructure plus the strictest pass
// (Simple). Subsequent commits add the rest.
package match

import "strings"

// Pass is one fuzzy-matching strategy. Pure function. Returns the
// matched substring from `original` plus ok=true on success.
type Pass struct {
	// Name is used in Result.PassName and in tests/logging.
	Name string

	// Fn is the matching function itself. Pure: no I/O, no shared
	// state, deterministic on its inputs.
	Fn func(original, old string) (string, bool)
}

// Result reports a successful match: the actual substring in `original`
// that should be replaced, plus the name of the pass that found it.
type Result struct {
	// Actual is the substring of `original` that matched. Use this
	// (NOT the caller's `old`) when computing replacements — fuzzy
	// passes may have whitespace-corrected the model's input.
	Actual string

	// PassName identifies which pass matched. "simple" means the
	// model's `old` was already exact; other names mean a fuzzy
	// pass had to correct for input drift.
	PassName string
}

// DefaultPasses is the canonical ordered chain, strictest to most
// flexible. The chain grows commit by commit as additional passes
// land — for now, only the Simple pass is wired up.
var DefaultPasses = []Pass{
	{Name: "simple", Fn: Simple},
}

// Find runs DefaultPasses on (original, old) and returns the first
// match. Wraps FindWith for the common case.
func Find(original, old string) (Result, bool) {
	return FindWith(DefaultPasses, original, old)
}

// FindWith runs the given passes in order and returns the first match.
// Exposed for tests that want to isolate a single pass, and for future
// callers that may want a custom chain (e.g. "exact-match only" mode).
// Both inputs have their line endings normalized to \n before any pass
// runs — this insulates every pass from CRLF / CR handling.
func FindWith(passes []Pass, original, old string) (Result, bool) {
	original = NormalizeLineEndings(original)
	old = NormalizeLineEndings(old)

	for _, p := range passes {
		if actual, ok := p.Fn(original, old); ok {
			return Result{Actual: actual, PassName: p.Name}, true
		}
	}
	return Result{}, false
}

// NormalizeLineEndings converts CRLF and lone CR to LF. Line-ending
// mismatch is the single biggest source of LLM edit failures, so we
// strip the variability before any pass runs.
func NormalizeLineEndings(s string) string {
	if !strings.ContainsAny(s, "\r") {
		return s
	}
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	return s
}

// CountOccurrences returns the number of (non-overlapping) occurrences
// of needle in haystack. Used by edit_file to decide whether a match
// is ambiguous (multi-match + !replace_all → error).
func CountOccurrences(haystack, needle string) int {
	return strings.Count(haystack, needle)
}
