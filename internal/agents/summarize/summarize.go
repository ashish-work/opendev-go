// Package summarize is a rule-based library for producing one-line
// descriptions of tool output. It is a LIBRARY, not a wire-time filter
// — callers decide when to use it. The ReactLoop intentionally keeps
// raw tool output in history so the model can read it on the next
// turn; this package exists for a future history compactor that will
// swap raw → summary on OLD tool messages only when the whole
// conversation approaches the context-window limit.
//
// This is the "dual-store" architecture: spillover (in the truncation
// package) keeps the raw tool output accessible via read_file; this
// summarize library produces the short description that the future
// compactor would substitute for the raw history entry when context
// gets tight. Both ship; spillover is the eager bound-history strategy,
// summarize is held back for lazy compaction.
//
// Why rule-based instead of an LLM call:
//
//   - Free (no extra tokens spent on a model that just describes output)
//   - Fast (microseconds vs the round-trip)
//   - Deterministic (same input → same summary across runs / tests)
//
// Per-tool-name dispatch covers the three v1 tools (read_file, bash,
// edit_file) with tailored summaries. Everything else gets a generic
// line-and-char count.
package summarize

import (
	"fmt"
	"strings"
)

// DefaultThreshold is the byte cap below which a tool message flows
// into history verbatim. Above it, the rule-based summary kicks in.
// 4096 bytes is roughly 1024 tokens of English — small enough that a
// dozen normal tool calls fit in a typical context window; large enough
// that a "head this file" or "list these dirs" call stays raw.
const DefaultThreshold = 4096

// Result describes the summarized text destined for the history
// message, plus a flag indicating whether condensation actually fired.
type Result struct {
	// Text is the string that should go into the assistant-visible
	// tool-role message. Either the raw input passed through, or a
	// short tool-keyed description.
	Text string

	// Truncated is true when summarizeFor() ran (the raw text was
	// above the threshold). False means the text below is the input
	// verbatim. Useful for tests, logging, and future telemetry.
	Truncated bool
}

// Summarize uses DefaultThreshold. Common case.
func Summarize(toolName, text string) Result {
	return SummarizeWith(toolName, text, DefaultThreshold)
}

// SummarizeWith returns the text verbatim if its byte length is at or
// below threshold; otherwise a tool-name-keyed one-liner. Threshold is
// in BYTES (len(text)), not runes or tokens — len() is O(1) and the
// goal here is "stop history from ballooning," not exact token math.
//
// Examples:
//
//	SummarizeWith("read_file", "package main\n", 4096)
//	// -> {"package main\n", false}  (pass-through, way under threshold)
//
//	SummarizeWith("read_file", strings.Repeat("x\n", 5000), 4096)
//	// -> {"Read file (5000 lines, 10000 chars)", true}
//
//	SummarizeWith("bash", "lots of output...", 0)
//	// -> {"Command executed (N lines of output)", true}
//	// (threshold=0 forces summarization regardless of size — test helper)
//
// A pass-through preserves any "[ERROR] ..." prefix the caller added.
// A summarized result does NOT preserve that prefix — the rule-based
// one-liner replaces the whole text. For v1 this is acceptable because
// short errors (the typical case) stay under threshold and survive
// verbatim. Long errors (a tool dumping a 50KB stack trace) get
// replaced — which is the win, since the model didn't need 50KB.
func SummarizeWith(toolName, text string, threshold int) Result {
	if len(text) <= threshold {
		return Result{Text: text}
	}
	return Result{Text: summaryFor(toolName, text), Truncated: true}
}

// summaryFor dispatches on tool name. New tools get a one-line case
// added here; unknown names hit the default arm. Keep this function
// allocation-light — it runs on every long tool result.
func summaryFor(toolName, text string) string {
	lines := countLines(text)
	chars := len(text)

	switch toolName {
	case "read_file":
		return fmt.Sprintf("Read file (%d lines, %d chars)", lines, chars)
	case "bash":
		return fmt.Sprintf("Command executed (%d lines of output)", lines)
	case "edit_file":
		return "File edited successfully"
	default:
		return fmt.Sprintf("Success (%d lines, %d chars)", lines, chars)
	}
}

// countLines returns the number of textual lines. A trailing newline
// does NOT add an extra empty line, matching how most line-counting
// tools (wc -l, editors, etc.) describe file shape.
//
//	""        → 0
//	"a"       → 1
//	"a\n"     → 1
//	"a\nb"    → 2
//	"a\nb\n"  → 2
func countLines(s string) int {
	if s == "" {
		return 0
	}
	n := strings.Count(s, "\n")
	if !strings.HasSuffix(s, "\n") {
		n++
	}
	return n
}
