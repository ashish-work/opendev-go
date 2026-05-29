package agents

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/ashish-work/opendev-go/internal/agents/doomloop"
	"github.com/ashish-work/opendev-go/internal/hooks"
	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/workflow"
)

// spawnSubagentToolName is the canonical tool name spawn_subagent
// registers as. Hardcoded here to avoid an import cycle with
// internal/tools/spawn (which imports internal/agents).
//
// Must match internal/tools/spawn.ToolName — checked via the
// homogeneous-batch detector below.
const spawnSubagentToolName = "spawn_subagent"

// MaxParallelSpawn caps the number of spawn_subagent calls that run
// concurrently when the model emits a homogeneous batch. The phase
// uses a buffered-channel semaphore of this size.
//
// Conservative against OpenAI Tier 1 TPM limits (4 × ~5K
// TPM/subagent stays inside the 30K cap for gpt-4o). Higher-tier
// users can edit this constant; a CLI flag is deferred until
// there's demand. Hardcoded so a single shared value is used by
// every binary; the tests verify the cap.
const MaxParallelSpawn = 4

// safetyPhase is the first phase of every iteration. It consolidates
// the two preconditions that can short-circuit the loop without an
// LLM call:
//
//   - Iteration cap. The loop driver runs an unbounded for; this
//     phase returns ErrMaxIterations once pc.Iter exceeds the
//     configured cap, replacing what used to be the loop condition.
//
//   - Context cancellation. Catches the user pressing Ctrl-C
//     between iterations (the LLM-call phase will catch it during
//     the call itself).
//
// Check order matters: max-iter first, then ctx. Reason: the original
// loop wrote
//
//	for iter := 1; iter <= MaxIterations; iter++ {
//	    if err := ctx.Err(); err != nil { return ErrInterrupted }
//	    // ...
//	}
//	return ErrMaxIterations
//
// — so when iter > cap, the loop condition failed before the ctx
// check ever ran. Reversing the order in the extracted phase would
// silently convert "cap exceeded with a cancelled context" from
// ErrMaxIterations to ErrInterrupted, a subtle behavior change.
// Preserve it by ordering checks to match the original control flow.
//
// Returns LoopActionContinue when both checks pass; the driver then
// runs the next phase in the iteration. Returns LoopActionReturn
// with the appropriate sentinel error and a full Result (Messages
// snapshot, Budget, Interrupted flag) otherwise. Result.Interrupted
// is set only on the ctx-cancel path — max-iter exits leave it
// false, matching the existing convention.
func (l *ReactLoop) safetyPhase(ctx context.Context, pc *PhaseContext) LoopAction {
	if pc.Iter > l.Config.MaxIterations {
		return NewLoopActionReturn(
			Result{
				Messages: *pc.History,
				Budget:   pc.Snapshot(),
			},
			fmt.Errorf("%w (limit=%d)", ErrMaxIterations, l.Config.MaxIterations),
			pc.Tracker,
		)
	}
	if err := ctx.Err(); err != nil {
		return NewLoopActionReturn(
			Result{
				Messages:    *pc.History,
				Interrupted: true,
				Budget:      pc.Snapshot(),
			},
			fmt.Errorf("%w: iter %d: %v", ErrInterrupted, pc.Iter, err),
			pc.Tracker,
		)
	}
	return NewLoopActionContinue(pc.Tracker)
}

// llmCallPhase runs one round of model-talking: build the request
// from current history + registered tools, dispatch via Stream when
// a sink is wired (pc.StreamSink != nil) or Call otherwise, update
// the budget calibrator against the API's authoritative
// prompt-token count, and stash the response on pc.LastResponse for
// the next phase to consume.
//
// Three return shapes:
//
//   - LoopActionContinue on success. pc.Tracker, pc.Calibrator, and
//     pc.LastResponse are updated; the driver runs process_response
//     next.
//
//   - LoopActionReturn carrying ErrInterrupted when ctx was
//     cancelled (the user pressed Ctrl-C during the call). Both
//     Call and Stream propagate ctx.Canceled through their error
//     chain; we detect it via ctx.Err() != nil so the agent layer
//     can report "user cancelled" instead of "API failed."
//     Result.Interrupted is set to true to match the safety phase's
//     convention.
//
//   - LoopActionReturn carrying ErrLLM for any other error. The
//     original error wraps via %v; the sentinel wraps via %w so
//     errors.Is keeps working.
//
// msgCountAtRequest is captured before the call returns because the
// assistant reply gets appended to history later (by
// process_response) and the calibrator wants the message count the
// API actually saw.
func (l *ReactLoop) llmCallPhase(ctx context.Context, pc *PhaseContext) LoopAction {
	msgCountAtRequest := len(*pc.History)

	exec := l.Config.Workflow.Resolve(workflow.SlotExecution)
	req := provider.Request{
		Model:           exec.Model,
		Messages:        l.buildRequestMessages(*pc.History),
		Tools:           SchemasFor(l.Registry),
		ReasoningEffort: exec.ReasoningEffort,
	}

	var (
		resp       provider.Response
		newTracker = pc.Tracker
		err        error
	)
	if pc.StreamSink != nil {
		resp, newTracker, err = l.Caller.Stream(ctx, req, pc.Tracker, pc.StreamSink)
	} else {
		resp, newTracker, err = l.Caller.Call(ctx, req, pc.Tracker)
	}
	pc.Tracker = newTracker

	if err != nil {
		// Context cancellation deserves ErrInterrupted (the agent-
		// layer sentinel for "user wants out") rather than ErrLLM,
		// which suggests an API failure. Either provider path (Call
		// or Stream) wraps ctx.Canceled in its error chain.
		if ctx.Err() != nil {
			return NewLoopActionReturn(
				Result{
					Messages:    *pc.History,
					Interrupted: true,
					Budget:      pc.Snapshot(),
				},
				fmt.Errorf("%w: iter %d: %v", ErrInterrupted, pc.Iter, err),
				pc.Tracker,
			)
		}
		return NewLoopActionReturn(
			Result{
				Messages: *pc.History,
				Budget:   pc.Snapshot(),
			},
			fmt.Errorf("%w: %v", ErrLLM, err),
			pc.Tracker,
		)
	}

	// Calibrate against the provider's authoritative count for the
	// messages it just saw. Done here (intrinsic to the LLM call)
	// rather than back in loop.go — splitting it across the phase
	// boundary would scatter the call's effects.
	pc.Calibrator = pc.Calibrator.Update(resp.Usage.PromptTokens, msgCountAtRequest)

	// Stash the response for process_response to consume.
	pc.LastResponse = resp

	return NewLoopActionContinue(pc.Tracker)
}

// todoSystemHeader prefaces the injected plan-of-record. The
// "persisted on disk" phrasing cues the model that the plan
// survives context compaction — it doesn't need to keep the plan
// alive in its own text. Surfaced as a constant so the prompt
// shape is grep-able from the test rig.
const todoSystemHeader = "Current plan-of-record (persisted on disk):\n\n"

// buildRequestMessages returns the Messages slice to send to the
// provider. When l.TodoStore is non-nil and the loaded state has
// at least one todo, a synthetic system message containing the
// rendered plan is appended at the end so the model sees current
// plan state without having to call `todo list`.
//
// The input slice is never mutated; the helper either returns
// history as-is (zero new allocations on the no-injection paths)
// or a fresh slice with the plan message appended. That keeps
// pc.History clean across iterations — every iteration's request
// sees fresh disk state, but the conversation log doesn't
// accumulate stale plan snapshots.
//
// On Store.Load error, the helper logs at debug and skips
// injection rather than failing the call. A flaky filesystem or
// a midway todo-tool write shouldn't break the agent loop — the
// model will see a slightly stale view (no plan that turn) and
// can recover.
//
// Position: appended at the END of history. Late position keeps
// the plan in recent-attention range. Both providers handle
// scattered system messages — OpenAI accepts them inline;
// Anthropic's extractSystem hoists them to its top-level system
// field.
//
// Known limitation: Anthropic's extractSystem joins all system
// messages into one cache_control'd block, so a plan that
// changes each turn invalidates the system-prompt cache hit.
// Proper fix is splitting static/dynamic system blocks with
// cache_control only on the static prefix; deferred until
// measurements show it matters.
func (l *ReactLoop) buildRequestMessages(history []provider.Message) []provider.Message {
	if l.TodoStore == nil {
		return history
	}
	state, err := l.TodoStore.Load()
	if err != nil {
		slog.Debug("agents: todo store load failed; skipping plan injection",
			"path", l.TodoStore.Path, "err", err)
		return history
	}
	if len(state.Todos) == 0 {
		return history
	}
	out := make([]provider.Message, 0, len(history)+1)
	out = append(out, history...)
	out = append(out, SystemMessage(todoSystemHeader+state.Render()))
	return out
}

// processResponsePhase makes the complete-vs-tool-call decision for
// one iteration. It reads pc.LastResponse and chooses one of three
// branches:
//
//  1. No tool calls — the model produced a final answer. Append the
//     assistant text to history (if any) and return Success.
//
//  2. Tool calls present, but the doom-loop detector says ForceStop
//     — the model is stuck repeating itself. Append a SystemMessage
//     with the warning and return ErrDoomLoop. CRITICAL: do NOT
//     append the assistant message in this branch. Its tool_calls
//     would orphan the corresponding tool-response slots and the
//     next request would fail OpenAI's "assistant(tool_calls) must
//     be immediately followed by tool messages" contract.
//
//  3. Tool calls present, detector says Continue / Redirect /
//     Notify — append the assistant message (text + tool_calls) and
//     hand off to the next phase, which will dispatch the tools.
//     The doom-loop verdict is stashed on PhaseContext so that next
//     phase can inject the warning + recovery hint AFTER tool
//     dispatch finishes (Redirect / Notify only).
//
// detector.Check is called exactly once per iteration. The detector
// mutates its sliding window on every call; double-Check'ing would
// double-fingerprint and break the 3-stage escalation. We stash the
// verdict on pc instead of recomputing.
func (l *ReactLoop) processResponsePhase(_ context.Context, pc *PhaseContext) LoopAction {
	resp := pc.LastResponse

	// Branch 1: no tool calls. Final answer.
	if len(resp.ToolCalls) == 0 {
		if resp.Content != "" {
			pc.AppendMessage(provider.Message{
				Role: "assistant",
				Content: []provider.ContentBlock{
					{Kind: provider.ContentText, Text: resp.Content},
				},
			})
		}
		return NewLoopActionReturn(
			Result{
				Content:  resp.Content,
				Success:  true,
				Messages: *pc.History,
				Budget:   pc.Snapshot(),
			},
			nil,
			pc.Tracker,
		)
	}

	// detector.Check once per iteration; verdict relayed via pc so
	// execute_sequential can act on Redirect / Notify after tool
	// dispatch finishes.
	action, warning, recovery := pc.Detector.Check(resp.ToolCalls)
	pc.DoomLoopAction = action
	pc.DoomLoopWarning = warning
	pc.DoomLoopRecovery = recovery

	// Branch 2: doom-loop ForceStop. We refuse to dispatch and we
	// don't record an assistant message whose tool_calls will never
	// get tool responses — only a system note explaining the halt.
	if action == doomloop.ForceStop {
		pc.AppendMessage(SystemMessage(warning))
		return NewLoopActionReturn(
			Result{
				Messages: *pc.History,
				Budget:   pc.Snapshot(),
			},
			fmt.Errorf("%w: %s", ErrDoomLoop, warning),
			pc.Tracker,
		)
	}

	// Branch 3: dispatch path. Commit the assistant message; the
	// next phase will dispatch its tools and append their results
	// immediately, satisfying the tool_call → tool_response pairing.
	assistant := provider.Message{
		Role:      "assistant",
		ToolCalls: resp.ToolCalls,
	}
	if resp.Content != "" {
		assistant.Content = []provider.ContentBlock{
			{Kind: provider.ContentText, Text: resp.Content},
		}
	}
	pc.AppendMessage(assistant)

	return NewLoopActionContinue(pc.Tracker)
}

// executeSequentialPhase dispatches the assistant's tool calls in
// order and appends a ToolResultMessage for each. Hooks intervene
// at three points around every call:
//
//   - PreToolUse fires before the dispatch. If it denies, the phase
//     appends a synthetic ToolResultMessage ("blocked by hook")
//     and skips dispatch — the synthetic message preserves
//     OpenAI's assistant(tool_calls) → tool_response pairing
//     contract. If it returns UpdatedInput, the rewritten JSON
//     replaces the model's original arguments before dispatch.
//
//   - PostToolUse fires after a successful dispatch.
//
//   - PostToolUseFailure fires after an infrastructure error
//     (Registry.Dispatch returned an error) before the phase bails
//     with ErrToolExec.
//
// AdditionalContext from any hook is appended as a SystemMessage
// AFTER the tool result — never between the assistant tool_call
// message and its paired tool_response, which would break the
// pairing (same shape as doom-loop Redirect/Notify in
// handle_completion).
//
// Two failure classes still flow through:
//
//   - Tool-domain failures (Success: false ToolResult) — flow into
//     history as observations the model will react to. The phase
//     does NOT return Return for these.
//   - Infrastructure failures (Registry.Dispatch returns an error)
//     — bubble out as ErrToolExec after PostToolUseFailure fires.
//
// Dispatched in order. v1 doesn't parallelize tool calls; Phase 7
// adds parallel dispatch for the homogeneous-spawn_subagent batch
// case but leaves the sequential path here as the default.
func (l *ReactLoop) executeSequentialPhase(ctx context.Context, pc *PhaseContext) LoopAction {
	calls := pc.LastResponse.ToolCalls

	// Parallel-spawn fast path: when the model emits a homogeneous
	// batch of spawn_subagent calls, dispatch them concurrently.
	// Other batch shapes (mixed, single call, non-spawn) fall
	// through to the sequential loop below.
	if isHomogeneousSpawnBatch(calls) {
		return l.executeParallelSpawnBatch(ctx, pc)
	}

	for _, call := range calls {
		outcome, action := l.executeOneCall(ctx, pc, call)
		if action.Kind == LoopActionReturn {
			return action
		}
		pc.AppendMessage(ToolResultMessage(outcome.Call.ID, outcome.Call.Name, outcome.Result))
		appendHookContext(pc, outcome.PreContext)
		appendHookContext(pc, outcome.PostContext)
	}
	return NewLoopActionContinue(pc.Tracker)
}

// oneCallOutcome packages everything one tool dispatch contributes
// to history: the tool result itself (becomes a ToolResultMessage)
// plus any AdditionalContext from PreToolUse / PostToolUse hooks
// (each becomes a separate SystemMessage AFTER the tool result).
//
// Returned by executeOneCall and consumed by both the sequential
// loop in executeSequentialPhase and the parallel dispatcher in
// executeParallelSpawnBatch. Centralising the shape keeps the
// hook-firing rules in one place.
type oneCallOutcome struct {
	// Call is the tool_call from the assistant message — needed
	// to construct ToolResultMessage(ID, Name, Result).
	Call provider.ToolCall

	// Result is what Registry.Dispatch (or a synthetic block)
	// produced.
	Result tools.ToolResult

	// PreContext is PreToolUse's AdditionalContext, appended as a
	// SystemMessage AFTER the tool result.
	PreContext string

	// PostContext is PostToolUse's AdditionalContext, appended as
	// a SystemMessage AFTER PreContext.
	PostContext string
}

// executeOneCall runs the hook → permissions → dispatch → hook
// chain for one tool call and returns either an outcome to append
// to history (with a Continue LoopAction) or a Return LoopAction
// for an infrastructure failure / ctx cancellation.
//
// Used by both the sequential loop and the parallel batch
// dispatcher — extracting the per-call logic keeps the
// hook-handling rules from drifting between the two paths.
//
// The function does NOT mutate pc.History; the caller is
// responsible for appending. This is what makes parallel dispatch
// safe: every goroutine produces an outcome locally, and the main
// goroutine drains the outcomes after wg.Wait.
func (l *ReactLoop) executeOneCall(ctx context.Context, pc *PhaseContext, call provider.ToolCall) (oneCallOutcome, LoopAction) {
	args := ensureJSON(call.Arguments)

	// 1. PreToolUse — gate, transform, or annotate.
	preResult, err := l.fireHook(ctx, hooks.HookEventPreToolUse, call.Name,
		hooks.PreToolUsePayload{Tool: call.Name, Args: args})
	if err != nil || ctx.Err() != nil {
		return oneCallOutcome{}, l.interruptedReturn(pc, hookCause(err, ctx))
	}
	if preResult.IsDeny() {
		return oneCallOutcome{
			Call: call,
			Result: tools.ToolResult{
				Output:  fmt.Sprintf("blocked by hook: %s", preResult.Reason),
				Success: false,
				Error:   "permission denied",
			},
			PreContext: preResult.AdditionalContext,
		}, NewLoopActionContinue(pc.Tracker)
	}
	if len(preResult.UpdatedInput) > 0 {
		args = preResult.UpdatedInput
	}

	// 2. Permissions check — consulted AFTER PreToolUse so a hook
	//    that rewrites args (sanitize, normalize, strip a forbidden
	//    prefix) gets to clean the input before the policy
	//    evaluates it. The deny path mirrors the hook-deny shape:
	//    Success:false, Error:"permission denied", and a Continue
	//    LoopAction so the model sees the rejection in its tool
	//    response and can pivot. This is not a hard stop; the
	//    agent learns from the deny just like it learns from any
	//    other tool failure.
	if decision := l.Permissions.Check(call.Name, string(args)); !decision.Allowed {
		return oneCallOutcome{
			Call: call,
			Result: tools.ToolResult{
				Output:  fmt.Sprintf("denied by policy: %s", decision.Reason),
				Success: false,
				Error:   "permission denied",
			},
			PreContext: preResult.AdditionalContext,
		}, NewLoopActionContinue(pc.Tracker)
	}

	// 4. Dispatch.
	result, dispatchErr := l.Registry.Dispatch(ctx, pc.ToolCtx, call.Name, args)
	if dispatchErr != nil {
		// 4a. PostToolUseFailure — fire but ignore the verdict
		// since we're bailing anyway.
		_, _ = l.fireHook(ctx, hooks.HookEventPostToolUseFailure, call.Name,
			hooks.PostToolUseFailurePayload{
				Tool:  call.Name,
				Error: dispatchErr.Error(),
			})
		return oneCallOutcome{}, NewLoopActionReturn(
			Result{
				Messages: *pc.History,
				Budget:   pc.Snapshot(),
			},
			fmt.Errorf("%w: %s: %v", ErrToolExec, call.Name, dispatchErr),
			pc.Tracker,
		)
	}

	// 4b. PostToolUse — collect annotation for the model.
	postResult, err := l.fireHook(ctx, hooks.HookEventPostToolUse, call.Name,
		hooks.PostToolUsePayload{
			Tool:    call.Name,
			Output:  result.Output,
			Success: result.Success,
		})
	if err != nil || ctx.Err() != nil {
		return oneCallOutcome{}, l.interruptedReturn(pc, hookCause(err, ctx))
	}

	return oneCallOutcome{
		Call:        call,
		Result:      result,
		PreContext:  preResult.AdditionalContext,
		PostContext: postResult.AdditionalContext,
	}, NewLoopActionContinue(pc.Tracker)
}

// interruptedReturn constructs the LoopActionReturn for the
// "user cancelled mid-call" path. Centralised so the formatting
// stays consistent.
func (l *ReactLoop) interruptedReturn(pc *PhaseContext, cause error) LoopAction {
	return NewLoopActionReturn(
		Result{
			Messages:    *pc.History,
			Interrupted: true,
			Budget:      pc.Snapshot(),
		},
		fmt.Errorf("%w: iter %d: %v", ErrInterrupted, pc.Iter, cause),
		pc.Tracker,
	)
}

// hookCause picks the right "why we're returning" cause for the
// interrupted path: prefer the hook's error when present, else
// fall back to ctx.Err() (the cancellation that propagated through
// a per-hook process kill).
func hookCause(hookErr error, ctx context.Context) error {
	if hookErr != nil {
		return hookErr
	}
	return ctx.Err()
}

// isHomogeneousSpawnBatch reports whether every tool call in the
// batch is spawn_subagent AND there's more than one of them. Single
// calls fall back to the sequential path because the goroutine +
// waitgroup + semaphore overhead would dominate the win.
//
// "Homogeneous spawn" is the obvious safe parallel case:
// spawn_subagent fires an isolated child loop with no shared
// mutable state between calls. Other tools (edit_file, bash) have
// ordering semantics that make parallel dispatch unsafe without
// per-tool analysis.
func isHomogeneousSpawnBatch(calls []provider.ToolCall) bool {
	if len(calls) < 2 {
		return false
	}
	for _, c := range calls {
		if c.Name != spawnSubagentToolName {
			return false
		}
	}
	return true
}

// executeParallelSpawnBatch dispatches a homogeneous batch of
// spawn_subagent calls concurrently. Each goroutine runs the same
// executeOneCall as the sequential path; results are collected into
// an indexed slice (declaration order) and appended to history
// after wg.Wait returns.
//
// Concurrency is capped at MaxParallelSpawn via a buffered-channel
// semaphore so a 10-call batch on a Tier 1 OpenAI key doesn't
// instantly blow through the TPM cap.
//
// Failure handling: if any goroutine produces a Return action
// (infrastructure error or interrupt), the dispatcher waits for
// all in-flight goroutines to drain (so no leak) and returns the
// first failing action. Successful outcomes from other goroutines
// are discarded — mirrors the sequential path's "bail on first
// error" behavior so the two paths are observationally
// equivalent.
//
// Ctx cancellation mid-batch is caught both via the per-call
// ctx checks inside executeOneCall AND via a final ctx.Err()
// check after wg.Wait, surfacing as ErrInterrupted.
func (l *ReactLoop) executeParallelSpawnBatch(ctx context.Context, pc *PhaseContext) LoopAction {
	calls := pc.LastResponse.ToolCalls
	outcomes := make([]oneCallOutcome, len(calls))
	actions := make([]LoopAction, len(calls))

	sem := make(chan struct{}, MaxParallelSpawn)
	var wg sync.WaitGroup

	for i, call := range calls {
		wg.Add(1)
		go func(idx int, c provider.ToolCall) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			outcome, action := l.executeOneCall(ctx, pc, c)
			outcomes[idx] = outcome
			actions[idx] = action
		}(i, call)
	}

	wg.Wait()

	// If ctx was cancelled mid-batch, return ErrInterrupted
	// regardless of whether the goroutines noticed. Belt-and-
	// braces: the per-call check inside executeOneCall already
	// catches most cases.
	if err := ctx.Err(); err != nil {
		return l.interruptedReturn(pc, err)
	}

	// Walk in declaration order so the FIRST failure (by tool_call
	// position) wins. Mirrors how the sequential path bails on the
	// first error it encounters.
	for _, action := range actions {
		if action.Kind == LoopActionReturn {
			return action
		}
	}

	// All successful — append in declaration order. Each outcome's
	// PreContext and PostContext become separate SystemMessages
	// AFTER the corresponding tool result, just like the sequential
	// path.
	for _, outcome := range outcomes {
		pc.AppendMessage(ToolResultMessage(outcome.Call.ID, outcome.Call.Name, outcome.Result))
		appendHookContext(pc, outcome.PreContext)
		appendHookContext(pc, outcome.PostContext)
	}

	return NewLoopActionContinue(pc.Tracker)
}

// fireHook is the nil-safe wrapper that phase code uses instead of
// calling l.Hooks.Fire directly. When l.Hooks is nil (no settings
// file, no hooks configured) it returns an empty FireResult so the
// phase reads the same shape whether or not hooks are wired.
//
// Errors from Manager.Fire propagate up — currently only ctx
// cancellation surfaces here (per-hook infrastructure errors are
// swallowed inside Fire and recorded on per-hook outcomes).
func (l *ReactLoop) fireHook(
	ctx context.Context,
	event hooks.HookEvent,
	identifier string,
	payload any,
) (*hooks.FireResult, error) {
	if l.Hooks == nil {
		return &hooks.FireResult{}, nil
	}
	return l.Hooks.Fire(ctx, event, identifier, payload)
}

// appendHookContext appends a hook's AdditionalContext as a system
// message when it's non-empty. Skips empty strings so a hook that
// chose not to comment doesn't leave a blank system message in
// history.
func appendHookContext(pc *PhaseContext, context string) {
	if context == "" {
		return
	}
	pc.AppendMessage(SystemMessage(context))
}

// handleCompletionPhase is the post-dispatch finalizer. Currently it
// has one job: on doom-loop Redirect or Notify, append the warning +
// recovery hint as a system message so the next LLM call sees it.
// The append happens AFTER all tool results because inserting before
// would break OpenAI's tool_call → tool_response pairing contract.
//
// This phase exists as a separate landing zone for hooks (Phase 6
// PostToolUse, Phase 7 SubagentStop) that fire after dispatch but
// before the next iteration's safety check. Splitting it out now
// keeps the hook integrations surgical when they land.
//
// ForceStop never reaches here — process_response would have
// returned LoopActionReturn before dispatch. The phase doesn't
// special-case it because the check would be unreachable.
func (l *ReactLoop) handleCompletionPhase(_ context.Context, pc *PhaseContext) LoopAction {
	if pc.DoomLoopAction == doomloop.Redirect || pc.DoomLoopAction == doomloop.Notify {
		pc.AppendMessage(SystemMessage(pc.DoomLoopWarning + "\n\n" + pc.DoomLoopRecovery))
	}
	return NewLoopActionContinue(pc.Tracker)
}
