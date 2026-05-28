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

	for iter := 1; iter <= l.Config.MaxIterations; iter++ {
		// Ctx check at iteration top — catches cancellations between
		// turns without waiting for the next LLM round-trip.
		if err := ctx.Err(); err != nil {
			return Result{
					Messages:    history,
					Interrupted: true,
					Budget:      snapshot(),
				}, tracker,
				fmt.Errorf("%w: iter %d: %v", ErrInterrupted, iter, err)
		}

		// Snapshot the request-side message count BEFORE the call —
		// that's what apiPromptTokens will refer to once the response
		// comes back. (Assistant reply gets appended after.)
		msgCountAtRequest := len(history)

		req := provider.Request{
			Model:    l.Config.Workflow.Resolve(workflow.SlotExecution).Model,
			Messages: history,
			Tools:    SchemasFor(l.Registry),
		}

		var (
			resp       provider.Response
			newTracker cost.Tracker
			err        error
		)
		if sink != nil {
			resp, newTracker, err = l.Caller.Stream(ctx, req, tracker, sink)
		} else {
			resp, newTracker, err = l.Caller.Call(ctx, req, tracker)
		}
		tracker = newTracker
		if err != nil {
			// Context cancellation deserves ErrInterrupted (the agent-
			// layer sentinel for "user wants out") rather than ErrLLM,
			// which suggests an API failure. Either provider path
			// (Call or Stream) wraps ctx.Canceled in its error chain.
			if ctx.Err() != nil {
				return Result{
						Messages:    history,
						Interrupted: true,
						Budget:      snapshot(),
					}, tracker,
					fmt.Errorf("%w: iter %d: %v", ErrInterrupted, iter, err)
			}
			return Result{Messages: history, Budget: snapshot()}, tracker,
				fmt.Errorf("%w: %v", ErrLLM, err)
		}

		// Calibrate against the provider's authoritative count for the
		// messages it just saw.
		cal = cal.Update(resp.Usage.PromptTokens, msgCountAtRequest)

		// No tool calls = final answer. Append the assistant text and
		// exit. We do this BEFORE the tool-call path so we can hold
		// off on committing the assistant message until we've decided
		// whether we will dispatch its tool_calls — appending an
		// assistant message with tool_calls that don't get answered
		// breaks OpenAI's "assistant(tool_calls) must be immediately
		// followed by tool messages" contract.
		if len(resp.ToolCalls) == 0 {
			if resp.Content != "" {
				history = append(history, provider.Message{
					Role: "assistant",
					Content: []provider.ContentBlock{
						{Kind: provider.ContentText, Text: resp.Content},
					},
				})
			}
			return Result{
				Content:  resp.Content,
				Success:  true,
				Messages: history,
				Budget:   snapshot(),
			}, tracker, nil
		}

		// Doom-loop check happens BEFORE we commit the assistant
		// message to history. ForceStop discards the model's
		// tool_calls (we refuse to dispatch and we don't record an
		// assistant message whose tool_calls never get tool
		// responses) — only a system note explaining the halt.
		action, warning, recovery := detector.Check(resp.ToolCalls)
		if action == doomloop.ForceStop {
			history = append(history, SystemMessage(warning))
			return Result{Messages: history, Budget: snapshot()}, tracker,
				fmt.Errorf("%w: %s", ErrDoomLoop, warning)
		}

		// Commit the assistant message; we will dispatch its tools
		// and append their results immediately below.
		assistant := provider.Message{
			Role:      "assistant",
			ToolCalls: resp.ToolCalls,
		}
		if resp.Content != "" {
			assistant.Content = []provider.ContentBlock{
				{Kind: provider.ContentText, Text: resp.Content},
			}
		}
		history = append(history, assistant)

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

	// Loop hit its iteration cap. Return the partial Result so callers
	// can show the user what was happening up to the cap.
	return Result{Messages: history, Budget: snapshot()}, tracker,
		fmt.Errorf("%w (limit=%d)", ErrMaxIterations, l.Config.MaxIterations)
}
