package compactor

import (
	"github.com/ashish-work/opendev-go/internal/provider"
)

// DefaultPruneMaxLen is the size threshold below which a tool
// output is considered "short" — low enough signal that, under
// context pressure, its content is worth dropping. 200 chars is
// the Phase 5 plan's starting value: a bash "exit 0", a one-line
// status, a tiny grep hit. Substantive outputs (file dumps,
// diffs, long command output) sit well above this and are left to
// the masking stage. Operators can tune by passing a different
// value to Prune.
const DefaultPruneMaxLen = 200

// markerPruned is the content a pruned tool message is collapsed
// to. Unlike the masking marker ("[ref:<id>]"), it carries no
// dereference handle on purpose: pruning targets LOW-SIGNAL
// output, so there is nothing worth re-requesting. The tool's
// identity still reaches the model via the preserved Name field
// on the message envelope, so a bare "[pruned]" is enough.
const markerPruned = "[pruned]"

// ProtectedTools lists the tool names whose outputs are NEVER
// pruned, regardless of size. These tools produce content the
// model commonly references by tool_call_id downstream — a file's
// contents, an edit's resulting diff — so silently dropping even
// a short instance risks breaking a later reference.
//
// Exported as a package var (not a constant slice — Go has none)
// so operators and tests can override the protected set. Treated
// as read-only by Prune: the slice is never mutated, only read,
// so reassigning it wholesale is the supported way to reconfigure.
var ProtectedTools = []string{"read_file", "edit_file", "write_file"}

// Prune returns a new []provider.Message where short, low-signal
// tool outputs are collapsed to a single "[pruned]" marker. It is
// the LevelPrune stage of the optimization ladder — the escalation
// past masking, applied when replacing only stale outputs (the
// masking stage) hasn't freed enough headroom.
//
// A role=tool message is pruned iff ALL of the following hold:
//
//  1. Its producing tool (Name) is not in ProtectedTools.
//  2. Its combined text length is in the open interval
//     (len("[pruned]"), maxLen) — short enough to be low signal,
//     but long enough that collapsing it actually saves tokens.
//     An output already at or below the marker's own length is
//     left raw: replacing it would only grow the message.
//  3. It is not already a masking marker ("[ref:..."). Those
//     carry a dereference handle the model may still use; pruning
//     them would strip the ID and lose the reference. Masked and
//     pruned messages are disjoint by construction.
//  4. It is not already pruned ("[pruned]"). Idempotence: the
//     safety phase may call Prune on every iteration.
//
// Selection is purely size- and tool-based; recency plays no part.
// That is deliberate — recency is the masking stage's axis, and
// keeping the two stages orthogonal makes each one's contract easy
// to reason about. In the loop, masking runs at the lower
// threshold band, so by the time Prune fires the bulky old outputs
// are already "[ref:...]" markers (skipped by rule 3) and what
// remains for Prune to act on is the residue of small outputs.
//
// Behavior contract (shared with MaskObservations):
//
//   - Pairing preserved. Only Content changes; Role, ToolCallID,
//     and Name survive, so OpenAI's assistant(tool_calls)->tool
//     and Anthropic's tool_use->tool_result pairings stay intact.
//   - Immutable. The input slice is never written, and the
//     returned slice is fully independent (every message cloned
//     via cloneMessage), so a caller may write through either side
//     without reaching the other.
//   - Defensive. Non-tool roles pass through untouched.
//
// maxLen <= 0 prunes nothing (every length fails rule 2's upper
// bound), a sane "disabled" default. Pure function: no I/O, safe
// to call concurrently with itself.
func Prune(history []provider.Message, maxLen int) []provider.Message {
	if len(history) == 0 {
		return history
	}

	protected := protectedSet()

	out := make([]provider.Message, len(history))
	for i, msg := range history {
		if !shouldPrune(msg, maxLen, protected) {
			out[i] = cloneMessage(msg)
			continue
		}
		out[i] = prunedCopy(msg)
	}
	return out
}

// shouldPrune applies the four-part predicate documented on Prune.
// Factored out so the decision reads as one expression and is unit-
// testable in isolation from the slice plumbing.
func shouldPrune(m provider.Message, maxLen int, protected map[string]struct{}) bool {
	if m.Role != "tool" {
		return false
	}
	if _, ok := protected[m.Name]; ok {
		return false
	}
	if isAlreadyMasked(m) || isAlreadyPruned(m) {
		return false
	}
	n := contentTextLen(m)
	// (len(marker), maxLen): short enough to be low signal, long
	// enough that collapsing it actually shrinks the message.
	return n > len(markerPruned) && n < maxLen
}

// prunedCopy returns a new Message identical to m except Content
// is a single "[pruned]" text block. ToolCallID and Name are
// preserved so the wire-format pairing — and the model's awareness
// of which tool ran — both survive.
func prunedCopy(m provider.Message) provider.Message {
	return provider.Message{
		Role:       m.Role,
		ToolCallID: m.ToolCallID,
		Name:       m.Name,
		Content: []provider.ContentBlock{
			{Kind: provider.ContentText, Text: markerPruned},
		},
	}
}

// isAlreadyPruned reports whether m's first text block is exactly
// the pruned marker. Keeps Prune idempotent: a second pass over
// already-pruned history is a no-op. Exact match (not prefix) is
// intentional — "[pruned]" carries no variable tail, so anything
// longer is genuine content that happens to start the same way.
func isAlreadyPruned(m provider.Message) bool {
	if len(m.Content) == 0 {
		return false
	}
	first := m.Content[0]
	if first.Kind != provider.ContentText {
		return false
	}
	return first.Text == markerPruned
}

// contentTextLen returns the total character count across all
// ContentText blocks in m. Non-text blocks contribute nothing —
// the size heuristic is about textual token cost, and a future
// image block carries its weight elsewhere. A single text block is
// the common case, so this is usually one len() call.
func contentTextLen(m provider.Message) int {
	n := 0
	for _, b := range m.Content {
		if b.Kind == provider.ContentText {
			n += len(b.Text)
		}
	}
	return n
}

// protectedSet builds a lookup set from the ProtectedTools slice.
// Built once per Prune call rather than kept as a package map so
// the configurable surface stays a simple slice and there is no
// shared mutable map to guard against concurrent reconfiguration.
func protectedSet() map[string]struct{} {
	set := make(map[string]struct{}, len(ProtectedTools))
	for _, name := range ProtectedTools {
		set[name] = struct{}{}
	}
	return set
}
