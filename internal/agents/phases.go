package agents

import (
	"context"
	"fmt"
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
