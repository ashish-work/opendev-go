package match

import (
	"regexp"
	"strings"
)

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

// -----------------------------------------------------------------------------
// Pass 3 — BlockAnchor: first/last lines anchor, middle scored by similarity
// -----------------------------------------------------------------------------

// blockAnchorSimilarityThreshold is the minimum LCS-based ratio
// required for a candidate to win when more than one (first, last)
// line pair was found. With exactly one candidate we accept any
// similarity — the anchors alone are enough disambiguation.
const blockAnchorSimilarityThreshold = 0.3

// BlockAnchor matches when the first and last lines of `old` (trimmed)
// pin down a region in `original`, even if the middle lines drifted.
// Useful when the model paraphrased the body but got the boundaries
// right (a surprisingly common LLM mode).
//
// Requires at least 3 lines (first + middle + last) — single- and
// two-line patterns are handled by Simple/LineTrimmed.
//
// Strategy:
//  1. Trim the first and last lines of `old` — these become the anchors.
//  2. Scan `original` for line pairs where line[i].trim() == firstAnchor
//     AND line[j].trim() == lastAnchor, with j sized roughly like
//     len(old) (search window is 2× the expected length).
//  3. For each candidate (i, j), score the middle by LCS-based
//     similarity vs `old`'s middle.
//  4. With one candidate, threshold drops to 0.0 (anchors alone are
//     enough). With multiple candidates, similarity ≥ 0.3 wins.
//
// Example (model paraphrased the middle):
//
//	original: "fn process() {\n    let result = compute(x);\n}\n"
//	old:      "fn process() {\n    let result = transform(x);\n}"
//	              ^anchor1                ^drift                  ^anchor2
//
//	First anchor "fn process() {" matches at line 0.
//	Last anchor "}" matches at line 2.
//	Middle similarity ≈ 0.85 (most chars overlap between "compute" and "transform").
//	Single candidate → threshold 0.0 → match wins.
//	→ ("fn process() {\n    let result = compute(x);\n}", true)
//
// Example (rejected — anchors absent):
//
//	original: "foo\nbar\nbaz\n"
//	old:      "BEGIN\n  body\nEND"
//	→ ("", false)    // no line trims to "BEGIN"
func BlockAnchor(original, old string) (string, bool) {
	oldLines := strings.Split(old, "\n")
	if len(oldLines) < 3 {
		return "", false
	}

	firstTrimmed := strings.TrimSpace(oldLines[0])
	lastTrimmed := strings.TrimSpace(oldLines[len(oldLines)-1])

	middleOld := make([]string, 0, len(oldLines)-2)
	for _, l := range oldLines[1 : len(oldLines)-1] {
		middleOld = append(middleOld, strings.TrimSpace(l))
	}

	originalLines := strings.Split(original, "\n")

	type candidate struct {
		start, end int
		sim        float64
	}
	var candidates []candidate

	for i := 0; i < len(originalLines); i++ {
		if strings.TrimSpace(originalLines[i]) != firstTrimmed {
			continue
		}
		// Search a window of 2× the expected length for the closing
		// anchor. 2× tolerates middle lines drifting modestly in count.
		windowEnd := i + len(oldLines)*2
		if windowEnd > len(originalLines) {
			windowEnd = len(originalLines)
		}
		for endIdx := i + len(oldLines) - 1; endIdx < windowEnd; endIdx++ {
			if endIdx >= len(originalLines) {
				break
			}
			if strings.TrimSpace(originalLines[endIdx]) != lastTrimmed {
				continue
			}
			middleOrig := make([]string, 0, endIdx-i-1)
			for _, l := range originalLines[i+1 : endIdx] {
				middleOrig = append(middleOrig, strings.TrimSpace(l))
			}

			var sim float64
			switch {
			case len(middleOld) == 0 && len(middleOrig) == 0:
				sim = 1.0
			case len(middleOld) == 0 || len(middleOrig) == 0:
				continue
			default:
				sim = similarity(strings.Join(middleOld, "\n"), strings.Join(middleOrig, "\n"))
			}
			candidates = append(candidates, candidate{start: i, end: endIdx, sim: sim})
		}
	}

	if len(candidates) == 0 {
		return "", false
	}

	threshold := blockAnchorSimilarityThreshold
	if len(candidates) == 1 {
		threshold = 0.0
	}

	bestIdx := 0
	for i := 1; i < len(candidates); i++ {
		if candidates[i].sim > candidates[bestIdx].sim {
			bestIdx = i
		}
	}
	if candidates[bestIdx].sim < threshold {
		return "", false
	}

	best := candidates[bestIdx]
	actual := strings.Join(originalLines[best.start:best.end+1], "\n")
	if strings.Contains(original, actual) {
		return actual, true
	}
	return "", false
}

// similarity returns a ratio in [0, 1] equal to 2*|LCS(a,b)| / (|a|+|b|),
// the same shape as Python's difflib.SequenceMatcher.ratio(). 1.0 means
// identical, 0.0 means no common substring. Operates on bytes (not
// runes) — fine because we apply it to trimmed code where ASCII
// dominates and a few multi-byte chars don't shift the score meaningfully.
//
// Examples:
//
//	similarity("",      "")      = 1.0   // both empty → identical
//	similarity("abc",   "")      = 0.0   // one empty → no overlap
//	similarity("abc",   "abc")   = 1.0   // identical
//	similarity("abc",   "xyz")   = 0.0   // LCS = "" → 0 / 6 = 0
//	similarity("abcdef","abcxyz")= 0.5   // LCS = "abc" (3) → 2*3 / (6+6) = 0.5
//	similarity("kitten","sitting")≈ 0.61 // LCS = "ittn" (4) → 8 / 13 ≈ 0.615
func similarity(a, b string) float64 {
	switch {
	case a == "" && b == "":
		return 1.0
	case a == "" || b == "":
		return 0.0
	}
	ab := []byte(a)
	bb := []byte(b)
	lcs := lcsLength(ab, bb)
	return 2.0 * float64(lcs) / float64(len(ab)+len(bb))
}

// lcsLength is the length of the longest common subsequence of a and b,
// computed with O(n) extra memory (two rolling rows instead of a full
// m×n table).
func lcsLength(a, b []byte) int {
	m, n := len(a), len(b)
	prev := make([]int, n+1)
	curr := make([]int, n+1)
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if a[i-1] == b[j-1] {
				curr[j] = prev[j-1] + 1
			} else if curr[j-1] > prev[j] {
				curr[j] = curr[j-1]
			} else {
				curr[j] = prev[j]
			}
		}
		prev, curr = curr, prev
		for k := range curr {
			curr[k] = 0
		}
	}
	best := 0
	for _, v := range prev {
		if v > best {
			best = v
		}
	}
	return best
}

// -----------------------------------------------------------------------------
// Pass 4 — WhitespaceNormalized: collapse \s+ to single space per line
// -----------------------------------------------------------------------------

// wsRe matches runs of whitespace within a single line. We split the
// input into lines first, so this never crosses newlines.
var wsRe = regexp.MustCompile(`\s+`)

// wsNormalize collapses each line's internal whitespace to single
// spaces and trims line edges. Preserves line breaks so we can compare
// line-by-line. The regex is compiled once at package init via
// regexp.MustCompile.
//
// Examples:
//
//	wsNormalize("x  =\t1  +  2")  = "x = 1 + 2"   // runs collapsed
//	wsNormalize("  hello  world ") = "hello world" // edges trimmed
//	wsNormalize("a\n  b  \nc")    = "a\nb\nc"     // per-line normalize
//	wsNormalize("x=1+2")           = "x=1+2"       // no whitespace → unchanged
func wsNormalize(s string) string {
	lines := strings.Split(s, "\n")
	for i, ln := range lines {
		lines[i] = strings.TrimSpace(wsRe.ReplaceAllString(ln, " "))
	}
	return strings.Join(lines, "\n")
}

// WhitespaceNormalized matches when the model collapsed (or expanded)
// whitespace runs in ways LineTrimmed can't recover from — e.g.,
// "if  x  ==  1" vs "if x == 1". Walks a sliding window around the
// expected line count and compares the normalized forms.
//
// IMPORTANT: this pass collapses RUNS of whitespace but cannot ADD or
// REMOVE whitespace where it doesn't exist on the other side. So:
//
//	"x=1"   vs "x = 1"   → MISMATCH  (one has no whitespace at all)
//	"x  =1" vs "x = 1"   → MATCH     (both have some whitespace, runs differ)
//	"a\tb"  vs "a   b"   → MATCH     (\t and "   " both collapse to " ")
//
// Strategy:
//  1. Normalize `old` once (collapse \s+ → " " per line, trim each).
//  2. For each starting line i in `original`, try window sizes from
//     ~oldLineCount to oldLineCount+2.
//  3. Normalize the candidate window; compare to the normalized `old`.
//  4. On match, return the candidate as it appeared in `original`
//     (un-normalized) so the replacement preserves real bytes.
func WhitespaceNormalized(original, old string) (string, bool) {
	normOld := wsNormalize(old)
	originalLines := strings.Split(original, "\n")
	oldLineCount := strings.Count(old, "\n") + 1

	for i := 0; i < len(originalLines); i++ {
		endMax := i + oldLineCount + 2
		if endMax > len(originalLines) {
			endMax = len(originalLines)
		}
		startJ := i + oldLineCount - 1
		for j := startJ; j <= endMax; j++ {
			if j > len(originalLines) || j < i {
				break
			}
			candidate := strings.Join(originalLines[i:j], "\n")
			if wsNormalize(candidate) == normOld && strings.Contains(original, candidate) {
				return candidate, true
			}
		}
	}
	return "", false
}
