package agents

import (
	"context"
	"fmt"

	"github.com/ashishgupta/opendev-go/internal/cost"
	"github.com/ashishgupta/opendev-go/internal/provider"
	"github.com/ashishgupta/opendev-go/internal/tools"
)

// DefaultMaxIterations caps how many model→tool→model cycles one Run
// will take before giving up with ErrMaxIterations. A real session
// typically completes in 5-15 turns; 25 is the runaway guard.
const DefaultMaxIterations = 25

// Config holds the per-run knobs for a ReactLoop. Constructor fills in
// defaults for zero values so the zero Config is usable.
type Config struct {
	// Model is the provider-specific identifier (e.g. "gpt-4o").
	// Passed through to provider.Request.Model unchanged.
	Model string

	// MaxIterations is the loop cap. Zero falls back to DefaultMaxIterations.
	MaxIterations int

	// SystemPrompt is the leading system Message text. Empty falls
	// back to DefaultSystemPrompt.
	SystemPrompt string

	// WorkingDir is plumbed into every tool dispatch via ToolContext.
	// Empty means "no working dir" — tools that resolve relative paths
	// will treat paths as absolute.
	WorkingDir string
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
func (l *ReactLoop) Run(ctx context.Context, userTask string) (Result, cost.Tracker, error) {
	history := []provider.Message{
		SystemMessage(l.Config.SystemPrompt),
		UserMessage(userTask),
	}
	tracker := cost.Tracker{}
	tctx := tools.ToolContext{WorkingDir: l.Config.WorkingDir}

	for iter := 1; iter <= l.Config.MaxIterations; iter++ {
		// Ctx check at iteration top — catches cancellations between
		// turns without waiting for the next LLM round-trip.
		if err := ctx.Err(); err != nil {
			return Result{
					Messages:    history,
					Interrupted: true,
				}, tracker,
				fmt.Errorf("%w: iter %d: %v", ErrInterrupted, iter, err)
		}

		req := provider.Request{
			Model:    l.Config.Model,
			Messages: history,
			Tools:    SchemasFor(l.Registry),
		}

		resp, newTracker, err := l.Caller.Call(ctx, req, tracker)
		tracker = newTracker
		if err != nil {
			return Result{Messages: history}, tracker,
				fmt.Errorf("%w: %v", ErrLLM, err)
		}

		// Append the assistant's response to history regardless of
		// whether it's a final answer or a tool-call turn.
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

		// No tool calls = final answer. Exit the loop with Success.
		if len(resp.ToolCalls) == 0 {
			return Result{
				Content:  resp.Content,
				Success:  true,
				Messages: history,
			}, tracker, nil
		}

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
				return Result{Messages: history}, tracker,
					fmt.Errorf("%w: %s: %v", ErrToolExec, call.Name, dispatchErr)
			}
			history = append(history, ToolResultMessage(call.ID, call.Name, result))
		}
	}

	// Loop hit its iteration cap. Return the partial Result so callers
	// can show the user what was happening up to the cap.
	return Result{Messages: history}, tracker,
		fmt.Errorf("%w (limit=%d)", ErrMaxIterations, l.Config.MaxIterations)
}
