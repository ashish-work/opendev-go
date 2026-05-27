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

// -----------------------------------------------------------------------------
// Pass 5 — IndentationFlexible: strip indent, skip blanks, greedy match
// -----------------------------------------------------------------------------

// IndentationFlexible handles patterns where the model dropped indents
// and possibly removed blank lines. Walks `original` line-by-line,
// skipping blanks, greedily matching the next non-blank line against
// the next non-blank line of `old`. On full match, returns the
// contiguous slice — INCLUDING any skipped blanks — so the replacement
// removes the whole region the model intended to delete/edit.
//
// Strategy:
//  1. Strip indent from every line of `old`; drop any blank lines.
//  2. For each start position in `original` where line.trim() ==
//     oldStripped[0], scan forward up to 3× the old length.
//  3. While scanning, skip blank lines; greedily match each non-blank
//     against the next oldStripped entry. Abort on a mismatch.
//  4. On a complete walk, return original[first_match .. last_match+1]
//     (CONTIGUOUS — includes any blank lines that were skipped during
//     matching).
//
// Example (file has blanks; old doesn't):
//
//	original: "    a\n\n    b\n\n    c\n"
//	old:      "a\nb\nc"
//
//	oldStripped: ["a", "b", "c"]
//
//	Scan from line 0:
//	  line 0 "    a"  trim="a"  match oldStripped[0]  ✓
//	  line 1 ""       trim=""   skip blank
//	  line 2 "    b"  trim="b"  match oldStripped[1]  ✓
//	  line 3 ""       trim=""   skip blank
//	  line 4 "    c"  trim="c"  match oldStripped[2]  ✓
//
//	Return slice [0..5] = "    a\n\n    b\n\n    c"
//	(includes the blank lines — they get replaced too)
//
// Example (rejected — middle mismatch):
//
//	original: "a\nb\nc\n"
//	old:      "a\nXXX\nc"
//	→ ("", false)    // greedy walk aborts on "b" ≠ "XXX"
func IndentationFlexible(original, old string) (string, bool) {
	oldStripped := make([]string, 0)
	for _, l := range strings.Split(old, "\n") {
		if t := strings.TrimSpace(l); t != "" {
			oldStripped = append(oldStripped, t)
		}
	}
	if len(oldStripped) == 0 {
		return "", false
	}

	originalLines := strings.Split(original, "\n")

	for i := 0; i < len(originalLines); i++ {
		if strings.TrimSpace(originalLines[i]) != oldStripped[0] {
			continue
		}

		matchedIndices := make([]int, 0, len(oldStripped))
		j := 0
		searchEnd := i + len(oldStripped)*3
		if searchEnd > len(originalLines) {
			searchEnd = len(originalLines)
		}
		for k := i; k < searchEnd; k++ {
			if j >= len(oldStripped) {
				break
			}
			trimmed := strings.TrimSpace(originalLines[k])
			if trimmed == "" {
				continue // skip blank lines
			}
			if trimmed == oldStripped[j] {
				matchedIndices = append(matchedIndices, k)
				j++
			} else {
				break // mismatch — abandon this anchor
			}
		}

		if j == len(oldStripped) && len(matchedIndices) > 0 {
			start := matchedIndices[0]
			end := matchedIndices[len(matchedIndices)-1] + 1
			actual := strings.Join(originalLines[start:end], "\n")
			if strings.Contains(original, actual) {
				return actual, true
			}
		}
	}
	return "", false
}

// -----------------------------------------------------------------------------
// Pass 6 — EscapeNormalized: turn literal escape sequences into real chars
// -----------------------------------------------------------------------------

// EscapeNormalized handles the case where the model produced LITERAL
// escape sequences ("\\n", "\\t", etc.) where the file has the actual
// characters. Happens when the model's input went through a JSON encode
// that double-escaped, or when the model literally typed `\n` thinking
// it would be interpreted as a newline.
//
// Strategy:
//  1. Unescape the well-known sequences: \n → newline, \t → tab,
//     \\ → \, \" → ", \' → '
//  2. If unescaping changed nothing, this pass has nothing to offer.
//  3. Otherwise, check whether the unescaped form is a substring of
//     `original`.
//
// Example:
//
//	original: "line one\nline two\n"   // real newline
//	old:      `line one\nline two`     // literal backslash-n
//
//	unescape(old) = "line one\nline two"  // real newline now
//	→ ("line one\nline two", true)
//
// Example (unescape was a no-op → skip):
//
//	old: "no escapes here"
//	→ ("", false)   // identity means nothing for this pass to add
func EscapeNormalized(original, old string) (string, bool) {
	unescaped := unescape(old)
	if unescaped == old {
		return "", false
	}
	if strings.Contains(original, unescaped) {
		return unescaped, true
	}
	return "", false
}

// unescape replaces the common LLM-emitted escape sequences with the
// characters they represent. Order matters: \n / \t go before \\ so
// that "\\n" (the model literally typing backslash+n) becomes a
// newline before the doubled-backslash pass runs.
//
// This is a known-incomplete unescaper — sequences like \xNN or \uNNNN
// aren't handled. v1 covers the cases LLMs hit in practice.
func unescape(s string) string {
	s = strings.ReplaceAll(s, `\n`, "\n")
	s = strings.ReplaceAll(s, `\t`, "\t")
	s = strings.ReplaceAll(s, `\\`, `\`)
	s = strings.ReplaceAll(s, `\"`, `"`)
	s = strings.ReplaceAll(s, `\'`, `'`)
	return s
}

// -----------------------------------------------------------------------------
// Pass 7 — TrimmedBoundary: drop outer whitespace + line-level boundary expand
// -----------------------------------------------------------------------------

// TrimmedBoundary handles `old` strings surrounded by extra whitespace
// at the very edges — leading/trailing newlines, padding spaces, or
// entire blank lines the model accidentally captured.
//
// Strategy (two-stage):
//  1. Trim the whole `old`. If unchanged, this pass is a no-op.
//  2. If the trimmed form appears verbatim in `original`, return it.
//  3. Otherwise, line-level boundary expansion: take the first
//     non-empty line content and the last non-empty line content of
//     `old`. Find an original line that CONTAINS the first content,
//     then a later original line that CONTAINS the last content. The
//     contiguous span is the candidate.
//
// Example (outer noise only):
//
//	original: "x = 1\n"
//	old:      "\n\n  x = 1  \n\n"
//
//	trimmed = "x = 1"
//	→ ("x = 1", true)        // appears in original verbatim
//
// Example (line-level expansion):
//
//	original: "    open()\n        body\n    close()\n"
//	old:      "\nopen()  \n  body\n  close()\n\n"
//
//	trim(old) != old (outer noise present)
//	trim(old) not in original (inner spacing differs)
//	first content "open()" — line 0 contains it.
//	last content "close()" — line 2 contains it.
//	→ ("    open()\n        body\n    close()", true)
//
// Example (rejected — empty boundaries):
//
//	old: "\n\n\n"
//	→ ("", false)   // first/last content lines are empty → can't anchor
func TrimmedBoundary(original, old string) (string, bool) {
	trimmed := strings.TrimSpace(old)
	if trimmed == "" {
		// Empty pattern would match everywhere via strings.Contains —
		// guard explicitly. Same defensive check as Execute's empty
		// old_string rejection in editfile.go.
		return "", false
	}
	if trimmed == old {
		return "", false // nothing to trim
	}
	if strings.Contains(original, trimmed) {
		return trimmed, true
	}

	// Line-level boundary expansion.
	oldLines := strings.Split(old, "\n")
	if len(oldLines) < 2 {
		return "", false
	}
	firstContent := strings.TrimSpace(oldLines[0])
	lastContent := strings.TrimSpace(oldLines[len(oldLines)-1])
	if firstContent == "" || lastContent == "" {
		return "", false
	}

	originalLines := strings.Split(original, "\n")
	for i := 0; i < len(originalLines); i++ {
		if !strings.Contains(originalLines[i], firstContent) {
			continue
		}
		end := i + len(oldLines) + 2
		if end > len(originalLines) {
			end = len(originalLines)
		}
		for j := i + 1; j < end; j++ {
			if j >= len(originalLines) {
				break
			}
			if !strings.Contains(originalLines[j], lastContent) {
				continue
			}
			candidate := strings.Join(originalLines[i:j+1], "\n")
			if strings.Contains(original, candidate) {
				return candidate, true
			}
		}
	}
	return "", false
}

// -----------------------------------------------------------------------------
// Pass 8 — ContextAware: substring anchors + whole-region similarity scoring
// -----------------------------------------------------------------------------

// contextAwareSimilarityThreshold is the minimum LCS-based ratio a
// ContextAware candidate must clear. Stricter than BlockAnchor's 0.3
// because ContextAware's anchors are looser (substring, not equality),
// so we want a higher whole-region match before accepting.
const contextAwareSimilarityThreshold = 0.5

// ContextAware uses the first and last non-empty lines of `old` as
// LOOSE anchors (substring match, not equality) and scores candidates
// by overall similarity. The fuzziest pass that still preserves the
// model's intended region rather than line-by-line shape.
//
// Strategy:
//  1. Find first/last non-empty lines of `old` (trimmed) — anchors.
//  2. Find every line in `original` that CONTAINS the first anchor
//     (substring; the file line can be a "superset" of the model's
//     line).
//  3. For each such start, scan forward up to 2× the old length
//     looking for the FIRST line containing the last anchor. (Not
//     exhaustive — bounds the algorithm at O(n·w).)
//  4. Score the resulting span against `old` by similarity.
//  5. Best span with similarity > 0.5 wins.
//
// Example:
//
//	original: "    fn begin_processing() {\n        helper();\n    }\n"
//	old:      "fn begin\n  helper();\n}"
//
//	first anchor "fn begin" — file line "    fn begin_processing() {"
//	  trimmed contains "fn begin"  ✓
//	last anchor "}" — file line "    }" trimmed contains "}"  ✓
//	span = file lines [0..2] joined
//	similarity(old, span) ≈ 0.7 → above 0.5 threshold → match
//	→ that span, true
//
// Example (rejected — too dissimilar):
//
//	original: "alpha\n  ... 50 lines of unrelated code ...\nbeta\n"
//	old:      "alpha\nsmall body\nbeta"
//	→ ("", false)   // similarity well under 0.5
func ContextAware(original, old string) (string, bool) {
	oldLines := strings.Split(old, "\n")
	if len(oldLines) < 2 {
		return "", false
	}

	originalLines := strings.Split(original, "\n")

	// First non-empty trimmed line of `old`.
	var firstCtx string
	for _, l := range oldLines {
		if t := strings.TrimSpace(l); t != "" {
			firstCtx = t
			break
		}
	}
	if firstCtx == "" {
		return "", false
	}

	// Last non-empty trimmed line of `old`.
	var lastCtx string
	for i := len(oldLines) - 1; i >= 0; i-- {
		if t := strings.TrimSpace(oldLines[i]); t != "" {
			lastCtx = t
			break
		}
	}
	if lastCtx == "" {
		return "", false
	}

	// Every position where a file line (trimmed) contains firstCtx.
	var starts []int
	for i, l := range originalLines {
		if strings.Contains(strings.TrimSpace(l), firstCtx) {
			starts = append(starts, i)
		}
	}
	if len(starts) == 0 {
		return "", false
	}

	var bestMatch string
	var bestSim float64

	for _, start := range starts {
		searchEnd := start + len(oldLines)*2
		if searchEnd > len(originalLines) {
			searchEnd = len(originalLines)
		}
		for end := start + 1; end < searchEnd; end++ {
			if strings.Contains(strings.TrimSpace(originalLines[end]), lastCtx) {
				candidate := strings.Join(originalLines[start:end+1], "\n")
				sim := similarity(strings.TrimSpace(old), strings.TrimSpace(candidate))
				if sim > bestSim && sim > contextAwareSimilarityThreshold {
					bestSim = sim
					bestMatch = candidate
				}
				break // first end-anchor per start only (bounds search)
			}
		}
	}

	if bestMatch != "" && strings.Contains(original, bestMatch) {
		return bestMatch, true
	}
	return "", false
}

// -----------------------------------------------------------------------------
// Pass 9 — MultiOccurrence: last-resort outer-trim + per-line trim compare
// -----------------------------------------------------------------------------

// MultiOccurrence is the last-resort fallback. Trim the WHOLE `old`
// (drops outer whitespace + blanks), split into lines, then do a
// strict line-by-line trimmed comparison against every starting
// position in `original`.
//
// This is essentially "LineTrimmed after outer trim" — useful when
// the model wrapped its `old` content in extra blank lines that
// LineTrimmed itself can't strip (LineTrimmed trims per-line, not
// outer).
//
// Strategy:
//  1. trim(old). Empty → skip.
//  2. Split into lines.
//  3. For each starting position i where i + N fits, compare each of
//     the next N trimmed file lines to the N trimmed old lines.
//  4. On match, return the UN-trimmed file slice.
//
// Example:
//
//	original: "    line1\n    line2\n    line3\n"
//	old:      "\n\n  line1\n  line2\n  line3\n\n"
//
//	trim(old) = "line1\n  line2\n  line3"
//	  (outer newlines + outer "  " gone, INTERIOR per-line indent stays)
//	per-line trimmed:        ["line1", "line2", "line3"]
//	per-line trimmed of file: ["line1", "line2", "line3", ""]
//	                                   ^ all 3 match at offset 0
//
//	→ ("    line1\n    line2\n    line3", true)
//
// Example (rejected — empty after outer trim):
//
//	old: "\n\n\n"
//	→ ("", false)
func MultiOccurrence(original, old string) (string, bool) {
	trimmed := strings.TrimSpace(old)
	if trimmed == "" {
		return "", false
	}

	originalLines := strings.Split(original, "\n")
	trimmedLines := strings.Split(trimmed, "\n")

	if len(trimmedLines) > len(originalLines) {
		return "", false
	}

	for i := 0; i <= len(originalLines)-len(trimmedLines); i++ {
		allMatch := true
		for k, ol := range trimmedLines {
			if strings.TrimSpace(originalLines[i+k]) != strings.TrimSpace(ol) {
				allMatch = false
				break
			}
		}
		if allMatch {
			candidate := strings.Join(originalLines[i:i+len(trimmedLines)], "\n")
			if strings.Contains(original, candidate) {
				return candidate, true
			}
		}
	}
	return "", false
}
