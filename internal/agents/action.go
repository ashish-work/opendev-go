package agents

// LoopAction is the control-flow signal a phase function returns to
// the ReactLoop driver. It says one of two things:
//
//   - Continue: this phase did its work; move to the next phase
//     (or, if the last phase ran, to the next iteration).
//   - Return: exit the loop now and surface this Result + error to
//     the caller. Used for successful completion (Err = nil) and for
//     every failure path (ctx cancellation, max-iter, doom-loop, LLM
//     error, tool-exec error).
//
// Why a tagged struct rather than a sum-typed interface? Same reason
// ContentBlock, StreamEvent, and TurnKind work this way: single
// concrete type, switch on Kind, zero allocation per return. Readers
// who've learned the pattern once recognize it everywhere.
//
// This type is introduced ahead of the refactor that uses it (Phase 3
// commits #18-21 extract phase functions, each returning LoopAction).
// Keeping the type-introduction commit small means the larger refactors
// stay surgical replacements rather than entangled rewrites.

import "github.com/ashish-work/opendev-go/internal/cost"

// LoopActionKind tags the variant of a LoopAction. Iota-based int
// constants follow the same pattern as ContentKind and
// StreamEventKind — cheap to compare, exhaustive switches feel
// natural, easy to add a String method for logs.
type LoopActionKind int

const (
	// LoopActionContinue — proceed to the next phase, or to the next
	// iteration when the last phase ran. The Result and Err fields
	// are zero/nil.
	LoopActionContinue LoopActionKind = iota

	// LoopActionReturn — exit the loop. The Result field carries the
	// (possibly partial) state to surface to the caller; the Err
	// field carries the exit reason (nil = clean success, non-nil =
	// any failure including ctx cancellation, max-iter, etc.).
	LoopActionReturn
)

// String returns a stable lowercase identifier for the kind. Used in
// log lines and debug prints; matches the convention from
// StreamEventKind.
func (k LoopActionKind) String() string {
	switch k {
	case LoopActionContinue:
		return "continue"
	case LoopActionReturn:
		return "return"
	default:
		return "unknown"
	}
}

// LoopAction is one phase function's instruction back to the loop
// driver. Only the fields valid for the active Kind are populated;
// other fields are zero. Prefer the New* constructors over raw
// struct literals so the invariant is enforced at call sites.
//
// Tracker is included even on Continue because every phase that
// performs work (LLM call, tool dispatch) updates the running
// tracker, and the driver needs that update to flow back even when
// the loop continues. The cleanest way is to carry it on every
// LoopAction.
type LoopAction struct {
	// Kind selects which other fields are meaningful.
	Kind LoopActionKind

	// Result — valid for LoopActionReturn. Carries the loop's
	// outcome (Content, Messages, Budget, etc.) at the point of
	// exit. Always present on Return; ignored on Continue.
	Result Result

	// Err — valid for LoopActionReturn. nil means the loop is
	// returning successfully; non-nil means a failure exit (wrapped
	// with the existing sentinels: ErrLLM, ErrInterrupted,
	// ErrMaxIterations, ErrDoomLoop, ErrToolExec).
	Err error

	// Tracker — valid for both kinds. Phase functions that update
	// the cost tracker (the LLM-call phase, for example) write the
	// new tracker here; the driver picks it up and either passes it
	// to the next phase (Continue) or returns it (Return).
	Tracker cost.Tracker
}

// NewLoopActionContinue constructs a Continue action. Tracker is the
// updated cost tracker after this phase's work; pass the unchanged
// tracker when this phase did no work that affected cost.
func NewLoopActionContinue(tracker cost.Tracker) LoopAction {
	return LoopAction{
		Kind:    LoopActionContinue,
		Tracker: tracker,
	}
}

// NewLoopActionReturn constructs a Return action with the given
// Result, error, and tracker. err = nil signals a successful exit;
// non-nil signals failure. The driver returns all three to the
// outer caller (ReactLoop.Run / RunWithStream).
func NewLoopActionReturn(result Result, err error, tracker cost.Tracker) LoopAction {
	return LoopAction{
		Kind:    LoopActionReturn,
		Result:  result,
		Err:     err,
		Tracker: tracker,
	}
}
