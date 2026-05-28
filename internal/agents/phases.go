package agents

import (
	"context"
	"fmt"

	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/workflow"
)

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

	req := provider.Request{
		Model:    l.Config.Workflow.Resolve(workflow.SlotExecution).Model,
		Messages: *pc.History,
		Tools:    SchemasFor(l.Registry),
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
