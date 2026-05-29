package agents

import (
	"context"

	"github.com/ashish-work/opendev-go/internal/agents/doomloop"
	"github.com/ashish-work/opendev-go/internal/budget"
	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/hooks"
	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/workflow"
)

// DefaultMaxIterations caps how many model→tool→model cycles one Run
// will take before giving up with ErrMaxIterations. A real session
// typically completes in 5-15 turns; 25 is the runaway guard.
const DefaultMaxIterations = 25

// Config holds the per-run knobs for a ReactLoop. Constructor fills in
// defaults for zero values so the zero Config is usable.
type Config struct {
	// Workflow names a model per slot (Execution/Thinking/Compact/
	// Critique/VLM). v1 only routes through SlotExecution; unset slots
	// transparently fall back to it. Execution.Model must be set or
	// the provider will reject the request.
	Workflow workflow.Config

	// MaxIterations is the loop cap. Zero falls back to DefaultMaxIterations.
	MaxIterations int

	// SystemPrompt is the leading system Message text. Empty falls
	// back to DefaultSystemPrompt.
	SystemPrompt string

	// WorkingDir is plumbed into every tool dispatch via ToolContext.
	// Empty means "no working dir" — tools that resolve relative paths
	// will treat paths as absolute.
	WorkingDir string

	// MaxContextTokens is the model's context-window cap used by the
	// budget calibrator. Zero disables usage-percent math but the
	// calibrator still tracks reported and estimated counts.
	MaxContextTokens int
}

// ReactLoop is the v1 single-phase agent loop. The flow each iteration:
//
//  1. Check ctx, exit if cancelled.
//  2. Build provider.Request from current history + tools.
//  3. Call the model (via LlmCaller, which updates the cost tracker).
//  4. Append assistant message to history.
//  5. If no tool calls: return Content as the final result.
//  6. Otherwise: dispatch each tool call sequentially, append results,
//     continue.
//
// A richer multi-phase loop (pre-check, thinking, critique, action,
// tool exec, post-processing) can be layered on later. v1 collapses
// all non-tool-exec phases into a single LLM call.
type ReactLoop struct {
	Caller   *LlmCaller
	Registry *tools.Registry
	Config   Config

	// Hooks dispatches lifecycle events to user-configured shell
	// commands at the points defined by hooks.HookEvent. nil is
	// valid and means "no hooks fire" — the loop behaves exactly
	// as it did before Phase 6 landed. Binary wiring (#35) sets
	// this field when settings.json registers any hooks.
	Hooks *hooks.Manager
}

// NewReactLoop wires the loop with defaults applied for zero fields in
// cfg. The returned loop is safe to reuse across multiple Run calls
// against the same provider+registry.
func NewReactLoop(caller *LlmCaller, registry *tools.Registry, cfg Config) *ReactLoop {
	if cfg.MaxIterations == 0 {
		cfg.MaxIterations = DefaultMaxIterations
	}
	if cfg.SystemPrompt == "" {
		cfg.SystemPrompt = DefaultSystemPrompt
	}
	return &ReactLoop{
		Caller:   caller,
		Registry: registry,
		Config:   cfg,
	}
}

// Run drives the loop to completion, returning the final Result, the
// final cost.Tracker (so callers can persist or display cost), and any
// error that exited the loop abnormally.
//
// Success paths: Result.Success == true, error == nil.
// Failure paths: error wraps ErrMaxIterations / ErrInterrupted /
// ErrLLM / ErrToolExec; the Result still carries the partial Messages
// history so callers can inspect what happened.
//
// Run is a thin wrapper around RunWithStream with a nil sink. Callers
// that don't need streaming use this; the streaming path costs nothing
// to skip.
func (l *ReactLoop) Run(ctx context.Context, userTask string) (Result, cost.Tracker, error) {
	return l.RunWithStream(ctx, userTask, nil)
}

// RunWithStream is Run plus a streaming sink. When sink is non-nil the
// per-iteration LLM call uses Provider.Stream and forwards each
// StreamEvent to sink as it arrives. When sink is nil the loop uses
// Provider.Call exactly as v1 did.
//
// Sink ownership: the caller creates and closes sink. The loop only
// writes to it; closing is the caller's job, not the loop's. This
// matches the contract LlmCaller.Stream documents and lets the TUI
// own per-turn channel lifetime cleanly.
//
// Multi-iteration semantics: a single turn may issue multiple LLM
// calls (model → tools → model → tools → final). Every iteration
// streams to the SAME sink. Consumers should treat each
// StreamEventDone as "this iteration's LLM call finished" — not "the
// turn finished." The turn boundary is when this function returns.
func (l *ReactLoop) RunWithStream(
	ctx context.Context,
	userTask string,
	sink chan<- provider.StreamEvent,
) (Result, cost.Tracker, error) {
	history := []provider.Message{
		SystemMessage(l.Config.SystemPrompt),
		UserMessage(userTask),
	}

	// One PhaseContext for the whole turn. Phases mutate it across
	// iterations — Tracker and Calibrator carry forward via
	// reassignment-after-immutable-update; History is a pointer
	// indirected through the local slice so appends across phases
	// and iterations all land in the same backing array.
	//
	// The per-iteration ephemeral fields (LastResponse, DoomLoop*)
	// are written before they're read every iteration: llm_call
	// writes LastResponse before process_response reads it;
	// process_response writes the DoomLoop* fields before
	// handle_completion reads them. Stale values from the previous
	// iteration are always overwritten, so no reset is needed at
	// the top of each iteration.
	pc := &PhaseContext{
		History:      &history,
		Tracker:      cost.Tracker{},
		Calibrator:   budget.New(l.Config.MaxContextTokens),
		Detector:     doomloop.New(),
		ToolCtx:      tools.ToolContext{WorkingDir: l.Config.WorkingDir},
		StreamSink:   sink,
		SystemPrompt: l.Config.SystemPrompt,
	}

	// Unbounded loop; safetyPhase enforces the iteration cap and
	// the inter-iteration ctx-cancel check. Each pass is a clean
	// orchestration of the five phases. Every phase has the same
	// signature, the same Continue/Return contract, and the same
	// Result/Err shape on exit.
	for iter := 1; ; iter++ {
		pc.Iter = iter

		if a := l.safetyPhase(ctx, pc); a.Kind == LoopActionReturn {
			return a.Result, a.Tracker, a.Err
		}
		if a := l.llmCallPhase(ctx, pc); a.Kind == LoopActionReturn {
			return a.Result, a.Tracker, a.Err
		}
		if a := l.processResponsePhase(ctx, pc); a.Kind == LoopActionReturn {
			return a.Result, a.Tracker, a.Err
		}
		if a := l.executeSequentialPhase(ctx, pc); a.Kind == LoopActionReturn {
			return a.Result, a.Tracker, a.Err
		}
		if a := l.handleCompletionPhase(ctx, pc); a.Kind == LoopActionReturn {
			return a.Result, a.Tracker, a.Err
		}
	}
}
