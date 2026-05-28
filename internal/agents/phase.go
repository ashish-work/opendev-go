package agents

import (
	"github.com/ashish-work/opendev-go/internal/agents/doomloop"
	"github.com/ashish-work/opendev-go/internal/budget"
	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/tools"
)

// PhaseContext bundles the per-turn state that every phase function
// reads or mutates. After the Phase 3 extraction lands (commits #18
// through #21), each phase signature shrinks from
//
//	func phase(ctx, history, tracker, cal, detector, iter, tctx,
//	           sink, systemPrompt) (LoopAction, ...)
//
// to
//
//	func phase(ctx, pc *PhaseContext) LoopAction
//
// — same information, one parameter. The driver constructs one
// PhaseContext per turn at the top of RunWithStream, hands a pointer
// to each phase in sequence, and reads the mutated state between
// phases.
//
// Pointer-receiver semantics on every method. History lives behind a
// *[]Message so append calls propagate to the caller's slice (a plain
// []Message field would let appends produce a new slice header that
// the caller never sees). Tracker and Calibrator are plain value
// fields; phases reassign after each immutable update — the
// indirection isn't worth it for types whose update methods already
// return new values.
//
// What this type does NOT carry:
//   - The cancellable ctx.Context (Go idiom: ctx is always a
//     separate first argument).
//   - LoopAction control-flow helpers (Continue/Return); those live
//     on LoopAction itself.
//   - References to the *ReactLoop (the loop's config flows in via
//     constructor args; phases don't need the whole loop struct).
type PhaseContext struct {
	// History is a pointer to the running message log. Phases append
	// new messages by dereferencing via the AppendMessage helper:
	//   pc.AppendMessage(SystemMessage("..."))
	// The pointer is required because append may relocate the
	// backing array; assigning back through the pointer keeps the
	// caller's slice header in sync.
	History *[]provider.Message

	// Tracker is the running cost.Tracker. cost.Tracker is value-
	// typed and immutable — its update methods return new trackers.
	// Phases reassign:
	//   pc.Tracker, _ = pc.Tracker.RecordUsage(usage, pricing)
	// The new value is visible to subsequent phases through the
	// same *PhaseContext.
	Tracker cost.Tracker

	// Calibrator is the running budget.Calibrator. Same immutable-
	// value pattern as Tracker: phases reassign after Update calls:
	//   pc.Calibrator = pc.Calibrator.Update(apiPromptTokens, msgCount)
	Calibrator budget.Calibrator

	// Detector is the doom-loop sliding-window detector. Already a
	// pointer type with internal state; phases call Check directly.
	Detector *doomloop.Detector

	// Iter is the current iteration number. Set by the driver at the
	// top of each loop pass; read by the safety phase to enforce the
	// iteration cap and used by error-wrapping in interrupt paths.
	Iter int

	// ToolCtx is the immutable tool execution context (working
	// directory, etc.) passed to every tool dispatch. Read-only for
	// phases.
	ToolCtx tools.ToolContext

	// StreamSink is the per-turn streaming sink. The LLM-call phase
	// uses Provider.Stream when non-nil and Provider.Call when nil.
	// Same channel for every iteration of a turn.
	StreamSink chan<- provider.StreamEvent

	// SystemPrompt is the system prompt for this turn. Plumbed in so
	// Snapshot can call the calibrator without a separate parameter.
	// Stable across all iterations of one turn.
	SystemPrompt string

	// LastResponse is the most recent provider.Response from the
	// LLM-call phase. Populated when llmCallPhase returns
	// LoopActionContinue; consumed by the response-processing phase
	// that runs next. Zero between iterations and at the start of
	// each iteration before llmCallPhase runs.
	//
	// Lives here rather than on LoopAction because LoopAction is
	// control-flow-only and a Response field would force every
	// phase return to carry payload data. Per-iteration data
	// belongs on the per-iteration state bundle.
	LastResponse provider.Response
}

// NewPhaseContext constructs a PhaseContext. Called exactly once per
// turn at the top of the loop driver. Verbose argument list is the
// price of explicit construction; the function is called from one
// place and clarity at the call site is worth more than terseness at
// the constructor signature.
//
// history must be a non-nil pointer; nil would crash AppendMessage on
// first append. The caller typically does:
//
//	history := []provider.Message{SystemMessage(...), UserMessage(...)}
//	pc := NewPhaseContext(&history, ...)
//
// sink may be nil — the LLM-call phase treats nil as "use the non-
// streaming Call path."
func NewPhaseContext(
	history *[]provider.Message,
	tracker cost.Tracker,
	calibrator budget.Calibrator,
	detector *doomloop.Detector,
	toolCtx tools.ToolContext,
	sink chan<- provider.StreamEvent,
	systemPrompt string,
) *PhaseContext {
	return &PhaseContext{
		History:      history,
		Tracker:      tracker,
		Calibrator:   calibrator,
		Detector:     detector,
		ToolCtx:      toolCtx,
		StreamSink:   sink,
		SystemPrompt: systemPrompt,
	}
}

// AppendMessage adds one message to the underlying history slice and
// updates the caller's slice header through the pointer indirection.
// Wraps the noisy "*pc.History = append(*pc.History, msg)" idiom so
// phase functions read cleanly.
//
// Returns the new length, which is occasionally useful for phases
// that want to remember "the assistant message I just appended is at
// index N."
func (pc *PhaseContext) AppendMessage(msg provider.Message) int {
	*pc.History = append(*pc.History, msg)
	return len(*pc.History)
}

// Snapshot returns the current budget snapshot. Wraps the calibrator
// call so phases building Result.Budget on a return path don't need
// to thread Calibrator + History + SystemPrompt through the call.
//
// The snapshot is computed against the CURRENT history, so a phase
// that has just appended messages sees them reflected in the
// estimated count.
func (pc *PhaseContext) Snapshot() budget.Snapshot {
	return pc.Calibrator.Snapshot(*pc.History, pc.SystemPrompt)
}
