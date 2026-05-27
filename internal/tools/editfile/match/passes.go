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
