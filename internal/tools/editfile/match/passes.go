package match

import "strings"

// -----------------------------------------------------------------------------
// Pass 1 — Simple: exact substring match
// -----------------------------------------------------------------------------

// Simple is the strictest pass: byte-for-byte substring match. Used as
// the baseline — when the model produces clean input, this hits first
// and no fuzzy work happens.
//
// Example (matches):
//
//	original: "hello world"
//	old:      "world"
//	→ ("world", true)
//
// Example (misses — even a single byte mismatch fails):
//
//	original: "fn foo() {\n\treturn 1\n}"   // tab-indented body
//	old:      "fn foo() {\nreturn 1\n}"     // no tab before "return"
//	→ ("", false)                           // LineTrimmed picks up the slack
func Simple(original, old string) (string, bool) {
	if strings.Contains(original, old) {
		return old, true
	}
	return "", false
}

// -----------------------------------------------------------------------------
// Pass 2 — LineTrimmed: per-line trim before comparing
// -----------------------------------------------------------------------------

// LineTrimmed handles the most common LLM edit failure: the model
// produced the right lines but with slightly different leading/trailing
// whitespace per line (extra indent, missing indent, trailing spaces).
//
// Strategy:
//  1. Split both inputs into lines.
//  2. Trim each line of `old`. Reject the call if every line trims to
//     empty (an all-whitespace pattern would match anywhere).
//  3. Walk `original` line by line. At each starting position, check
//     whether the next N original lines (trimmed) match the trimmed
//     `old` lines.
//  4. On hit, return the untrimmed slice of `original` — that's what
//     we'll actually replace, preserving the file's real indentation.
//
// Example (model dropped the indent):
//
//	original: "    fn foo() {\n        return 1\n    }\n"
//	old:      "fn foo() {\nreturn 1\n}"
//
//	After per-line trim:
//	  oldTrimmed:      ["fn foo() {", "return 1", "}"]
//	  originalTrimmed: ["fn foo() {", "return 1", "}", ""]
//	                    ^anchor matches oldTrimmed[0]
//
//	Next 3 trimmed file lines == oldTrimmed → match.
//	Return the UNTRIMMED slice from original (preserving real indent):
//	  → ("    fn foo() {\n        return 1\n    }", true)
//
// Example (rejected — all whitespace):
//
//	old: "   \n\t\t\n  "    // every line trims to empty
//	→ ("", false)           // would otherwise match at every position
//
// The final containment check (strings.Contains(original, actual)) is a
// safety guard — it should always pass since `actual` is built from
// `originalLines`, but we keep it defensively.
func LineTrimmed(original, old string) (string, bool) {
	oldLines := strings.Split(old, "\n")
	oldTrimmed := make([]string, len(oldLines))
	allEmpty := true
	for i, line := range oldLines {
		t := strings.TrimSpace(line)
		oldTrimmed[i] = t
		if t != "" {
			allEmpty = false
		}
	}
	if len(oldTrimmed) == 0 || allEmpty {
		return "", false
	}

	originalLines := strings.Split(original, "\n")

	for i := 0; i < len(originalLines); i++ {
		// Anchor on the first trimmed line.
		if strings.TrimSpace(originalLines[i]) != oldTrimmed[0] {
			continue
		}
		// Bounds: do the candidate N lines fit in original?
		if i+len(oldTrimmed) > len(originalLines) {
			continue
		}
		// All lines must match in trimmed form.
		allMatch := true
		for j, oldLn := range oldTrimmed {
			if strings.TrimSpace(originalLines[i+j]) != oldLn {
				allMatch = false
				break
			}
		}
		if !allMatch {
			continue
		}

		// Reconstruct the original (untrimmed) span — that's the
		// substring of `original` we'll replace.
		actual := strings.Join(originalLines[i:i+len(oldTrimmed)], "\n")
		if strings.Contains(original, actual) {
			return actual, true
		}
	}
	return "", false
}
