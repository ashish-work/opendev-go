package agents

import (
	"context"
	"fmt"

	"github.com/ashish-work/opendev-go/internal/agents/doomloop"
	"github.com/ashish-work/opendev-go/internal/budget"
	"github.com/ashish-work/opendev-go/internal/cost"
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
	tracker := cost.Tracker{}
	cal := budget.New(l.Config.MaxContextTokens)
	detector := doomloop.New()
	tctx := tools.ToolContext{WorkingDir: l.Config.WorkingDir}

	// snapshot builds the Result.Budget for any return path. The
	// `history` slice is captured by reference, so the snapshot
	// reflects whatever was in history at the point of return.
	snapshot := func() budget.Snapshot {
		return cal.Snapshot(history, l.Config.SystemPrompt)
	}

	// The loop runs unbounded; safetyPhase enforces the iteration
	// cap and the inter-iteration ctx-cancel check. PhaseContext is
	// constructed fresh each pass around the existing locals;
	// subsequent commits will migrate the body into its own phases
	// and the locals will gradually disappear.
	for iter := 1; ; iter++ {
		pc := &PhaseContext{
			History:      &history,
			Tracker:      tracker,
			Calibrator:   cal,
			Detector:     detector,
			Iter:         iter,
			ToolCtx:      tctx,
			StreamSink:   sink,
			SystemPrompt: l.Config.SystemPrompt,
		}
		if action := l.safetyPhase(ctx, pc); action.Kind == LoopActionReturn {
			return action.Result, action.Tracker, action.Err
		}

		// LLM-call phase: build request, dispatch via Call/Stream,
		// update the calibrator. The phase stashes the response on
		// pc.LastResponse and updates pc.Tracker + pc.Calibrator in
		// place; we sync back to the still-inline locals here. Once
		// process_response and execute_sequential land (#20/#21),
		// the locals disappear and pc becomes the single source of
		// truth.
		if action := l.llmCallPhase(ctx, pc); action.Kind == LoopActionReturn {
			return action.Result, action.Tracker, action.Err
		}
		tracker = pc.Tracker
		cal = pc.Calibrator
		resp := pc.LastResponse

		// process_response phase: makes the no-tool-call vs
		// tool-call decision, runs the doom-loop ForceStop check,
		// commits the assistant message to history on the dispatch
		// path. Stashes the doom-loop verdict on pc so the
		// still-inline Redirect / Notify path below can read it
		// without re-calling the detector.
		if action := l.processResponsePhase(ctx, pc); action.Kind == LoopActionReturn {
			return action.Result, action.Tracker, action.Err
		}
		action := pc.DoomLoopAction
		warning := pc.DoomLoopWarning
		recovery := pc.DoomLoopRecovery

		// Dispatch each tool call in order. Tool-domain failures
		// (Success: false ToolResult) flow into history as observations
		// the model will react to. Infrastructure errors (registry
		// invariants, ctx cancellation surfacing from the tool) bubble
		// out and end the loop.
		for _, call := range resp.ToolCalls {
			result, dispatchErr := l.Registry.Dispatch(
				ctx, tctx, call.Name, ensureJSON(call.Arguments),
			)
			if dispatchErr != nil {
				return Result{Messages: history, Budget: snapshot()}, tracker,
					fmt.Errorf("%w: %s: %v", ErrToolExec, call.Name, dispatchErr)
			}
			history = append(history, ToolResultMessage(call.ID, call.Name, result))
		}

		// Doom-loop Redirect/Notify: append the warning + recovery
		// hint AFTER all tool results. Inserting before would break
		// OpenAI's tool_call → tool_response pairing contract (the
		// REPL hit a 400 from exactly this ordering bug).
		if action == doomloop.Redirect || action == doomloop.Notify {
			history = append(history, SystemMessage(warning+"\n\n"+recovery))
		}
	}
}
