// Package optimization classifies how much pressure the agent's
// context window is under and names a staged response. v2 ships
// the classifier (Level + Check); subsequent commits add the
// concrete transformations (mask, prune, sliding-window, full
// LLM compact) keyed by Level.
//
// Design split — data plane vs control plane:
//
//   - This file is the data plane: pure functions, no I/O, no
//     side effects. Check(snapshot) maps a budget snapshot to a
//     Level and that's it. Safe to call from the safety phase
//     on every iteration without measurable overhead.
//
//   - The control plane lives next door in
//     internal/optimization/compactor/ (added by commits #26-29).
//     It reads the Level, applies the transformation, returns a
//     new history slice. The compactor never decides "when";
//     the agent loop does.
//
// Naming — why "optimization" and not "context":
//
//   - The Go stdlib already owns the import path identifier
//     "context". A local package with the same name would force
//     every consumer to rename one of the two on import — bad
//     ergonomics.
//
//   - "compaction" was the second candidate but it's too narrow;
//     the Healthy and Warning levels do not compact at all, and
//     future Levels (cache-tier hints, prefetch budgeting) could
//     add more non-compaction behavior. "optimization" is the
//     umbrella term that covers everything the safety phase
//     might do to keep the loop inside budget.
package optimization

import (
	"github.com/ashish-work/opendev-go/internal/budget"
)

// Level names a stage in the staged-optimization ladder. Each
// level corresponds to a contiguous band of context-window usage
// percent; the further up the ladder, the more aggressive the
// recovery action the loop will take.
//
// The numeric values are not part of the contract — code outside
// this package MUST switch on the named constants, not the
// underlying iota values. Adding a new level in the middle (e.g.
// a hypothetical LevelDedup between Mask and Prune) would shift
// every later value but the named identifiers stay stable.
type Level int

const (
	// LevelHealthy: usage < ThresholdWarning. The loop runs as
	// usual; no logging, no transformation. This is the only
	// level that costs nothing.
	LevelHealthy Level = iota

	// LevelWarning: usage in [ThresholdWarning, ThresholdMaskObservations).
	// Operators see a slog warning so they can spot pressure
	// building before any transformation kicks in. Useful as an
	// early-trigger for measurement during Phase 5 rollout.
	LevelWarning

	// LevelMaskObservations: usage in
	// [ThresholdMaskObservations, ThresholdPrune). The first
	// transformation. Older tool results get replaced by
	// "[ref:c_<id>]" markers in the outgoing request — the model
	// retains awareness via the IDs but stops paying full token
	// cost for stale outputs. Reversible: the original results
	// stay in pc.History so a follow-up call can re-include them
	// if needed.
	LevelMaskObservations

	// LevelPrune: usage in [ThresholdPrune, ThresholdAggressiveMask).
	// When masking isn't enough, drop short tool outputs (<200
	// chars) entirely. read_file / edit_file / write_file outputs
	// are protected because the model often references their
	// content by tool_call_id downstream.
	LevelPrune

	// LevelAggressiveMask: usage in
	// [ThresholdAggressiveMask, ThresholdFullCompact). Last-ditch
	// before the LLM-driven path. Mask EVERYTHING except the
	// most recent N tool results so the model has headroom to
	// respond.
	LevelAggressiveMask

	// LevelFullCompact: usage >= ThresholdFullCompact. Final
	// resort. Calls the Compact workflow slot to summarize the
	// conversation middle into a single assistant message. The
	// only level that costs an LLM call.
	LevelFullCompact
)

// Threshold constants are exported so:
//
//   - Tests can pin them as part of the contract.
//   - Operators tuning a single value have one diff, not a
//     hardcoded grep hunt.
//
// The values come from the v2 plan as starting points, not
// measured optimums. Tuning targets future Phase 5 sub-commits
// can move these without renaming the levels.
const (
	ThresholdWarning          = 0.70
	ThresholdMaskObservations = 0.80
	ThresholdPrune            = 0.85
	ThresholdAggressiveMask   = 0.90
	ThresholdFullCompact      = 0.99
)

// String returns a stable lowercase identifier for the level —
// snake_case to match the slog convention used elsewhere in the
// codebase (HookEvent, ToolCategory, etc.). Used in slog output
// and test assertions; never emitted to model context.
//
// Unknown Level values return "unknown" rather than panicking
// because the constant set may grow over time and we don't want
// a stale binary that received a future Level value over the
// wire to crash on log formatting.
func (l Level) String() string {
	switch l {
	case LevelHealthy:
		return "healthy"
	case LevelWarning:
		return "warning"
	case LevelMaskObservations:
		return "mask_observations"
	case LevelPrune:
		return "prune"
	case LevelAggressiveMask:
		return "aggressive_mask"
	case LevelFullCompact:
		return "full_compact"
	default:
		return "unknown"
	}
}

// Check classifies a budget snapshot into its corresponding
// optimization level.
//
// Input choice — UsagePct (Reported) vs Estimated:
//
//   - UsagePct = Reported / MaxContextTokens. Reported is the
//     prompt_tokens count the provider returned on the last
//     call: authoritative, billed.
//
//   - Estimated is the calibrator's heuristic count of what the
//     NEXT call would weigh. Useful for forward-looking
//     decisions but it can flicker between adjacent levels
//     mid-iteration because every history append changes it.
//
// Check uses UsagePct so the classification is stable per LLM
// call: it only changes when a response lands. Future
// compactor steps may want a forward-looking check based on
// Estimated; that's a separate function added when needed.
//
// Pure: same Snapshot in, same Level out. No I/O, no allocation
// (Level is an int). Safe to call from the safety phase on every
// iteration.
//
// A zero-value Snapshot (no API call yet — UsagePct == 0.0)
// returns LevelHealthy. That keeps the very first iteration
// from spuriously triggering optimization before there's any
// data.
func Check(snap budget.Snapshot) Level {
	pct := snap.UsagePct
	switch {
	case pct >= ThresholdFullCompact:
		return LevelFullCompact
	case pct >= ThresholdAggressiveMask:
		return LevelAggressiveMask
	case pct >= ThresholdPrune:
		return LevelPrune
	case pct >= ThresholdMaskObservations:
		return LevelMaskObservations
	case pct >= ThresholdWarning:
		return LevelWarning
	default:
		return LevelHealthy
	}
}
