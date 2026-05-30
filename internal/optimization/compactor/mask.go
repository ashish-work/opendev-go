// Package compactor implements the concrete history-transformation
// stages that the optimization Level ladder dispatches to. Each
// stage is a pure function: history in, transformed history out,
// no I/O.
//
// Stages, in escalation order (added across Phase 5):
//
//   - MaskObservations  → this file. Replaces older role=tool
//     content with "[ref:<tool_call_id>]" markers while keeping
//     the most recent N raw.
//   - Prune             → later commit. Drops short tool outputs
//     and protects read_file / edit_file / write_file.
//   - SlidingWindow     → later commit. Keeps first + last 50,
//     replaces middle with a heuristic summary.
//   - FullCompact       → later commit. Calls the Compact LLM
//     slot to summarize the conversation middle.
//
// Composition rule: each stage is a self-contained transformation
// the safety phase may apply when the corresponding Level fires.
// Stages don't call each other; the loop decides which one runs.
package compactor

import (
	"strings"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// DefaultRecentKept is the number of most recent role=tool messages
// that stay unmasked when MaskObservations runs without a caller-
// specified count. 10 is the Phase 5 plan's starting value: enough
// fresh observations for the model to reason against, small enough
// that masking pays off on any moderately long session. Operators
// can tune by passing a different value to MaskObservations.
const DefaultRecentKept = 10

// markerPrefix is the unique prefix every masked tool message
// uses. The format "[ref:<id>]" is checked by isAlreadyMasked
// when MaskObservations is invoked on history that may already
// contain marker text — keeping the prefix in one place avoids
// drift between writer and reader sides of the idempotence
// check.
const markerPrefix = "[ref:"

// MaskObservations returns a new []provider.Message where older
// role=tool messages have their Content replaced with a single
// "[ref:<tool_call_id>]" text block. The most recent `recent`
// tool messages are left untouched.
//
// Behavior contract relied on by the safety phase:
//
//  1. Pairing preserved. Every masked tool message keeps its
//     ToolCallID and Name, so OpenAI's "assistant(tool_calls) →
//     tool" pairing and Anthropic's tool_use → tool_result pairing
//     both stay intact. Only Content changes.
//
//  2. Idempotent. Already-masked tool messages (Content begins
//     with "[ref:") are detected and skipped — re-masking is a
//     no-op. Safe to call from the safety phase on every iteration
//     without worrying about double-masking or marker recursion.
//
//  3. Immutable. The input slice is not mutated, AND the returned
//     slice is fully independent: every message's Content and
//     ToolCalls slices have their own backing arrays. A caller
//     that writes through the returned slice (`got[0].Content[0]
//     .Text = "..."`) cannot reach back into the input. Cloning
//     happens on every pass-through path, not just the masked
//     ones — see cloneMessage.
//
//  4. Defensive. Tool messages with empty ToolCallID are left raw
//     — there's no useful marker we can build without an ID, and
//     emitting "[ref:]" would surprise the model with no usable
//     handle to dereference. Non-tool roles (system/user/
//     assistant) pass through untouched.
//
//  5. Recency by tail walk. The tool messages counted as "recent"
//     are the LAST `recent` role=tool entries in history. Walking
//     once from end to start, the first `recent` tool messages
//     encountered (the youngest) stay raw; every older tool
//     message gets masked.
//
// recent ≤ 0 is treated as "mask every tool message" — useful
// when the caller has chosen a more aggressive level. recent
// greater than the count of tool messages in history is treated
// as "mask none" because the natural walk just runs out of
// candidates before hitting the masking branch.
//
// Pure function. No I/O, no allocation beyond the returned slice.
// Safe to call concurrently with itself; the input slice is read
// but never written.
func MaskObservations(history []provider.Message, recent int) []provider.Message {
	if len(history) == 0 {
		return history
	}
	if recent < 0 {
		recent = 0
	}

	// First pass: count tool messages and decide the cutoff
	// index — the position of the OLDEST tool message we want to
	// keep raw. Indices below cutoff get masked; indices at or
	// above cutoff stay raw.
	//
	// Walking from the end and decrementing a "remaining" counter
	// gives us the cutoff in one pass without allocating a
	// candidates slice.
	cutoff := -1 // every tool message before this index gets masked
	remaining := recent
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role != "tool" {
			continue
		}
		if remaining > 0 {
			remaining--
			cutoff = i
			continue
		}
		// We've already accounted for `recent` recent tool
		// messages. Everything older (lower index) than the
		// current i gets masked; cutoff stays at the youngest
		// kept-raw position. Break early; we don't need to
		// keep walking.
		break
	}

	// Second pass: build the output slice. Every message at
	// or after cutoff passes through unchanged; tool messages
	// before cutoff get masked.
	//
	// Even on the pass-through branches we clone the Message
	// so the returned slice is fully independent of history.
	// Go's `out[i] = msg` is a struct value copy, but Message
	// contains slice fields (Content, ToolCalls) whose headers
	// would share backing arrays with history's messages.
	// Without cloning, a caller that writes through the output
	// (`got[0].Content[0].Text = "..."`) would silently mutate
	// pc.History. Cloning is cheap (a 1-block Content slice +
	// a typically-empty ToolCalls slice) and the immutability
	// guarantee is worth the allocation.
	out := make([]provider.Message, len(history))
	for i, msg := range history {
		if msg.Role != "tool" {
			out[i] = cloneMessage(msg)
			continue
		}
		// Tool message; decide mask or keep.
		if cutoff != -1 && i >= cutoff {
			out[i] = cloneMessage(msg)
			continue
		}
		if msg.ToolCallID == "" {
			// No usable marker handle — leave raw rather than
			// emit "[ref:]". The model would have no way to
			// reference the call, defeating the purpose.
			out[i] = cloneMessage(msg)
			continue
		}
		if isAlreadyMasked(msg) {
			// Idempotent: don't re-wrap "[ref:c1]" as
			// "[ref:c1]" again. Pass through unchanged
			// (but still cloned for slice independence).
			out[i] = cloneMessage(msg)
			continue
		}
		out[i] = maskedCopy(msg)
	}
	return out
}

// cloneMessage returns a copy of m whose Content and ToolCalls
// slices have independent backing arrays. Used on every
// pass-through branch in MaskObservations so the returned slice
// is fully decoupled from the input — callers can mutate either
// side without spooky action at a distance.
//
// String fields (Role, ToolCallID, Name) are immutable in Go and
// share semantics for free. ContentBlock and ToolCall are
// struct value types; copying the outer slice is enough to make
// per-element writes safe. ToolCall.Arguments is a json.RawMessage
// ([]byte) whose underlying bytes are still shared after this
// copy — that's a deeper level of independence than any current
// caller needs, and the cost (per-tool-call json.RawMessage
// clone) would be wasted for the vastly common case of empty
// ToolCalls. If a future caller writes through arguments, this
// is the function to extend.
func cloneMessage(m provider.Message) provider.Message {
	out := m
	if len(m.Content) > 0 {
		out.Content = make([]provider.ContentBlock, len(m.Content))
		copy(out.Content, m.Content)
	}
	if len(m.ToolCalls) > 0 {
		out.ToolCalls = make([]provider.ToolCall, len(m.ToolCalls))
		copy(out.ToolCalls, m.ToolCalls)
	}
	return out
}

// maskedCopy returns a new Message identical to m except Content
// is a single text block containing the marker. Name and
// ToolCallID are preserved so the wire-format pairing stays
// intact.
func maskedCopy(m provider.Message) provider.Message {
	return provider.Message{
		Role:       m.Role,
		ToolCallID: m.ToolCallID,
		Name:       m.Name,
		Content: []provider.ContentBlock{
			{Kind: provider.ContentText, Text: markerFor(m.ToolCallID)},
		},
	}
}

// markerFor builds the masked-content text for a given
// tool_call_id. The format is documented in MaskObservations's
// godoc; centralised here so writer and reader paths can never
// drift.
func markerFor(id string) string {
	return markerPrefix + id + "]"
}

// isAlreadyMasked reports whether m's content already starts with
// the masked-marker prefix. Recognising masked messages keeps
// MaskObservations idempotent — the safety phase can call it on
// every iteration without re-wrapping prior markers.
//
// We check only the FIRST text block's prefix. Tool messages
// constructed by ToolResultMessage always have a single text
// block; a future change that emits multi-block tool messages
// would need to revisit this check.
func isAlreadyMasked(m provider.Message) bool {
	if len(m.Content) == 0 {
		return false
	}
	first := m.Content[0]
	if first.Kind != provider.ContentText {
		return false
	}
	return strings.HasPrefix(first.Text, markerPrefix)
}
