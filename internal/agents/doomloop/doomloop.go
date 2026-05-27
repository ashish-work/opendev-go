// Package doomloop detects when the ReAct loop is stuck — calling the
// same tool with the same arguments in a tight cycle and making no
// progress. The detector tracks a sliding window of recent tool-call
// fingerprints and flags repeating patterns of length 1, 2, or 3.
//
// Why this matters: an unguarded LLM agent can get into runaway states
// where every turn produces the same tool call, burning the iteration
// budget (and your money) without doing useful work. Detection +
// escalation breaks the cycle before it exhausts the iteration cap.
//
// Escalation walks three steps with each subsequent identical call
// after the threshold is reached:
//
//   - 1st detection → Redirect: inject a gentle nudge into history,
//     dispatch the tools anyway.
//   - 2nd detection → Notify: inject a stronger directive, still
//     dispatch.
//   - 3rd detection → ForceStop: halt the loop, surface ErrDoomLoop
//     to the caller, do NOT dispatch.
//
// The detector is stateful (sliding window + escalation counter) but
// safe for single-threaded use; the loop owns its detector per Run.
package doomloop

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strings"

	"github.com/ashishgupta/opendev-go/internal/provider"
)

const (
	// MaxCycleLen is the largest cycle length we check. A doom loop of
	// length > 3 is rare in practice and would slow detection.
	MaxCycleLen = 3

	// Threshold is how many times a cycle must repeat in the recent
	// window to count as a doom loop. 3 means [A,A,A] (1-step) or
	// [A,B,A,B,A,B] (2-step) etc.
	Threshold = 3

	// MaxRecent is the sliding-window size. 20 is enough to catch
	// 3-step cycles repeating 3 times (needs 9 entries) plus drift
	// tolerance.
	MaxRecent = 20
)

// Action is the detector's verdict for the latest tool-call batch.
// The loop interprets it and decides whether to nudge, log, or halt.
type Action int

const (
	// None — no doom loop detected. Continue normally.
	None Action = iota

	// Redirect — first detection. Inject a gentle nudge into history
	// and let the dispatch continue. The model gets another chance.
	Redirect

	// Notify — second detection. Inject a stronger directive — the
	// dispatch still runs, but the model is on notice.
	Notify

	// ForceStop — third detection. The loop halts; the caller should
	// surface this as an error (ErrDoomLoop) rather than dispatching
	// the tool calls. The detector's state is cleared so a subsequent
	// Run starts fresh.
	ForceStop
)

// String renders the action name for logs and tests.
func (a Action) String() string {
	switch a {
	case None:
		return "none"
	case Redirect:
		return "redirect"
	case Notify:
		return "notify"
	case ForceStop:
		return "force_stop"
	}
	return "unknown"
}

// Detector is the per-Run cycle finder. Zero value is NOT usable —
// use New(). Detector state is intentionally stateful (mutating
// methods); calls happen sequentially inside a single Run.
type Detector struct {
	// recent is the sliding window of fingerprints, FIFO. Cap MaxRecent.
	recent []string

	// nudgeCount tracks how many cycles we've ever flagged in this
	// detector's lifetime. Drives escalation: 1→Redirect, 2→Notify,
	// 3→ForceStop.
	nudgeCount int
}

// New returns an empty Detector. One Detector per ReactLoop.Run().
func New() *Detector {
	return &Detector{recent: make([]string, 0, MaxRecent)}
}

// NudgeCount reports how many doom-loop detections have fired so far.
// Useful for tests + logging.
func (d *Detector) NudgeCount() int { return d.nudgeCount }

// Reset clears the sliding window AND the nudge counter. Called
// automatically when ForceStop fires; can be called manually if the
// caller knows the conversation has fundamentally changed direction.
func (d *Detector) Reset() {
	d.recent = d.recent[:0]
	d.nudgeCount = 0
}

// Check appends fingerprints for the given tool calls and looks for
// a repeating cycle in the sliding window. Returns:
//
//   - action: what the loop should do (None/Redirect/Notify/ForceStop)
//   - warning: human-readable description of the cycle (empty when
//     action == None)
//   - recovery: hint text to inject into history alongside the warning
//     (empty for None and ForceStop)
//
// On ForceStop, the detector's sliding window is cleared but
// nudgeCount is preserved so callers can still inspect what happened.
func (d *Detector) Check(calls []provider.ToolCall) (Action, string, string) {
	for _, c := range calls {
		if len(d.recent) >= MaxRecent {
			d.recent = d.recent[1:]
		}
		d.recent = append(d.recent, fingerprint(c.Name, c.Arguments))
	}

	// Scan cycle lengths from shortest to longest; first hit wins.
	// Shorter cycles take fewer entries to qualify, so they naturally
	// win ties (e.g. AAAA matches both 1-step AAA and 2-step AAAA).
	for cycleLen := 1; cycleLen <= MaxCycleLen; cycleLen++ {
		required := cycleLen * Threshold
		if len(d.recent) < required {
			continue
		}
		segment := d.recent[len(d.recent)-required:]
		pattern := segment[:cycleLen]
		if !matchesCycle(segment, pattern, cycleLen) {
			continue
		}

		d.nudgeCount++
		warning := buildWarning(cycleLen, pattern)
		recovery := recoveryText(d.nudgeCount)
		action := actionFor(d.nudgeCount)
		if action == ForceStop {
			// Clear the window so a future Reset+Check on the same
			// detector behaves like a fresh start. nudgeCount stays
			// for caller introspection.
			d.recent = d.recent[:0]
		}
		return action, warning, recovery
	}

	return None, "", ""
}

// matchesCycle reports whether `segment` is `pattern` repeated
// (segment[i] == pattern[i % cycleLen]) for all i.
func matchesCycle(segment, pattern []string, cycleLen int) bool {
	for i, fp := range segment {
		if fp != pattern[i%cycleLen] {
			return false
		}
	}
	return true
}

// actionFor maps the running nudge count to an Action. 3+ all map
// to ForceStop — the caller is expected to halt before the next call.
func actionFor(nudgeCount int) Action {
	switch nudgeCount {
	case 1:
		return Redirect
	case 2:
		return Notify
	default:
		return ForceStop
	}
}

// recoveryText returns the hint that should accompany the warning in
// history. Hard-coded English for v1; future work can externalize
// these into a translatable reminders pack.
func recoveryText(nudgeCount int) string {
	switch nudgeCount {
	case 1:
		return "You're repeating the same action. Try a different approach: " +
			"reconsider the inputs, simplify the goal, or break the task into smaller steps."
	case 2:
		return "You've repeated this pattern multiple times. Step back and " +
			"question the approach itself. If the same tool keeps producing the " +
			"same result, the strategy is the problem — not the inputs."
	default:
		return ""
	}
}

// buildWarning produces the human-readable diagnostic. For a 1-step
// cycle it names the offending tool; for multi-step it lists the
// tools in cycle order.
func buildWarning(cycleLen int, pattern []string) string {
	if cycleLen == 1 {
		return fmt.Sprintf(
			"Tool `%s` has been called with the same arguments %d times in a row. The agent appears stuck.",
			toolNameOf(pattern[0]),
			Threshold,
		)
	}
	names := make([]string, len(pattern))
	for i, fp := range pattern {
		names[i] = toolNameOf(fp)
	}
	return fmt.Sprintf(
		"The agent is repeating a %d-step cycle (%s) %d times. It appears stuck.",
		cycleLen, strings.Join(names, " → "), Threshold,
	)
}

// fingerprint encodes a tool call as "name:hash16". Different args
// → different fingerprints. fnv-64a is fast, stdlib, and non-crypto
// (we don't need collision resistance, just consistency).
//
// Note: byte-equal JSON yields the same fingerprint; semantically-
// equal-but-byte-different JSON (different whitespace or key order)
// does NOT. Acceptable v1 simplification — most providers serialize
// the same call identically.
func fingerprint(name string, args json.RawMessage) string {
	h := fnv.New64a()
	h.Write([]byte(args))
	return fmt.Sprintf("%s:%016x", name, h.Sum64())
}

// toolNameOf splits "name:hash" and returns the name portion. Returns
// the input verbatim if there's no separator — defensive against
// malformed fingerprints.
func toolNameOf(fp string) string {
	if i := strings.IndexByte(fp, ':'); i >= 0 {
		return fp[:i]
	}
	return fp
}
