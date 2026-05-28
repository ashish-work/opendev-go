package agents

import (
	"context"
	"errors"
	"testing"

	"github.com/ashish-work/opendev-go/internal/agents/doomloop"
	"github.com/ashish-work/opendev-go/internal/budget"
	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/workflow"
)

// newSafetyTestRig constructs a minimal *ReactLoop and *PhaseContext
// sized for safetyPhase tests. maxIter caps the loop; history is
// pre-populated with a system + user message so Snapshot has
// something to work with.
func newSafetyTestRig(t *testing.T, maxIter int) (*ReactLoop, *PhaseContext) {
	t.Helper()
	loop := &ReactLoop{
		Config: Config{
			MaxIterations: maxIter,
			SystemPrompt:  "system prompt",
		},
	}
	history := []provider.Message{
		SystemMessage("system prompt"),
		UserMessage("hello"),
	}
	pc := NewPhaseContext(
		&history,
		cost.Tracker{CallCount: 2, TotalInputTokens: 100},
		budget.New(128_000),
		doomloop.New(),
		tools.ToolContext{WorkingDir: "/tmp"},
		nil,
		"system prompt",
	)
	return loop, pc
}

func TestSafetyPhase_ContinueWhenIterUnderCap(t *testing.T) {
	loop, pc := newSafetyTestRig(t, 25)
	pc.Iter = 3

	action := loop.safetyPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Errorf("Kind = %s, want continue", action.Kind)
	}
	if action.Err != nil {
		t.Errorf("Continue action should have nil Err; got %v", action.Err)
	}
	// Tracker should pass through unchanged.
	if action.Tracker.CallCount != 2 {
		t.Errorf("Tracker.CallCount = %d, want 2 (unchanged)", action.Tracker.CallCount)
	}
}

func TestSafetyPhase_ContinueAtCap(t *testing.T) {
	// The cap is inclusive — matches the old "iter <= MaxIterations"
	// loop condition. The Nth iteration is allowed to run.
	loop, pc := newSafetyTestRig(t, 25)
	pc.Iter = 25

	action := loop.safetyPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Errorf("at Iter == cap, Kind = %s, want continue (cap is inclusive)", action.Kind)
	}
}

func TestSafetyPhase_ReturnsMaxIterationsWhenOverCap(t *testing.T) {
	loop, pc := newSafetyTestRig(t, 25)
	pc.Iter = 26

	action := loop.safetyPhase(context.Background(), pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return", action.Kind)
	}
	if !errors.Is(action.Err, ErrMaxIterations) {
		t.Errorf("Err = %v, want chain containing ErrMaxIterations", action.Err)
	}
	if action.Result.Interrupted {
		t.Errorf("max-iter exit should have Interrupted = false")
	}
	// History should be preserved on the Result snapshot.
	if got := len(action.Result.Messages); got != 2 {
		t.Errorf("Result.Messages len = %d, want 2", got)
	}
}

func TestSafetyPhase_ReturnsInterruptedOnCtxCancel(t *testing.T) {
	loop, pc := newSafetyTestRig(t, 25)
	pc.Iter = 5

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	action := loop.safetyPhase(ctx, pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return", action.Kind)
	}
	if !errors.Is(action.Err, ErrInterrupted) {
		t.Errorf("Err = %v, want chain containing ErrInterrupted", action.Err)
	}
	if !action.Result.Interrupted {
		t.Errorf("ctx-cancel exit should set Result.Interrupted = true")
	}
	// Error message should include the iter number for debuggability.
	if !contains(action.Err.Error(), "iter 5") {
		t.Errorf("error %q should mention iter 5", action.Err)
	}
}

func TestSafetyPhase_MaxIterWinsOverCtxCancel(t *testing.T) {
	// Both conditions hold: ctx cancelled AND iter > cap. The phase
	// must return ErrMaxIterations because the original code's
	// for-loop condition would have failed before the ctx check ran.
	loop, pc := newSafetyTestRig(t, 25)
	pc.Iter = 26

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	action := loop.safetyPhase(ctx, pc)
	if !errors.Is(action.Err, ErrMaxIterations) {
		t.Errorf("with both conditions, want ErrMaxIterations, got %v", action.Err)
	}
	if errors.Is(action.Err, ErrInterrupted) {
		t.Errorf("should not surface ErrInterrupted when max-iter also fires")
	}
}

func TestSafetyPhase_ResultBudgetIsCurrent(t *testing.T) {
	// Result.Budget should reflect the calibrator's snapshot at
	// decision time — not a zero value.
	loop, pc := newSafetyTestRig(t, 25)
	pc.Iter = 26
	pc.Calibrator = pc.Calibrator.Update(5000, 2)

	action := loop.safetyPhase(context.Background(), pc)
	if action.Result.Budget.Reported != 5000 {
		t.Errorf("Result.Budget.Reported = %d, want 5000 (current snapshot)",
			action.Result.Budget.Reported)
	}
}

func TestSafetyPhase_TrackerPassesThroughOnReturn(t *testing.T) {
	// The Return action should carry the tracker forward unchanged —
	// safety doesn't perform billed work.
	loop, pc := newSafetyTestRig(t, 25)
	pc.Iter = 26
	pc.Tracker.TotalCostUSD = 1.23
	pc.Tracker.CallCount = 9

	action := loop.safetyPhase(context.Background(), pc)
	if action.Tracker.TotalCostUSD != 1.23 || action.Tracker.CallCount != 9 {
		t.Errorf("Tracker not preserved on Return: %+v", action.Tracker)
	}
}

func TestSafetyPhase_IntegrationWithRun_MaxIter(t *testing.T) {
	// Smoke: the externally observable max-iter behavior is preserved
	// through the refactor. fakeProvider keeps yielding tool calls so
	// the loop never finishes; we expect ErrMaxIterations after the
	// configured cap.
	p := &fakeProvider{
		responses: []provider.Response{
			{ToolCalls: []provider.ToolCall{{ID: "c1", Name: "foo"}}},
			{ToolCalls: []provider.ToolCall{{ID: "c2", Name: "foo"}}},
			{ToolCalls: []provider.ToolCall{{ID: "c3", Name: "foo"}}},
		},
	}
	reg := tools.NewRegistry()
	if err := reg.Register(echoTool("foo")); err != nil {
		t.Fatalf("register: %v", err)
	}
	loop := NewReactLoop(NewLlmCaller(p, cost.Pricing{}), reg, Config{
		Workflow:      workflow.Config{Execution: workflow.SlotConfig{Model: "fake"}},
		MaxIterations: 2,
	})

	_, _, err := loop.Run(context.Background(), "go")
	if !errors.Is(err, ErrMaxIterations) {
		t.Errorf("err = %v, want ErrMaxIterations", err)
	}
}

func TestSafetyPhase_IntegrationWithRun_CtxCancel(t *testing.T) {
	// Pre-cancelled ctx surfaces as ErrInterrupted via safetyPhase
	// (the same way it did inline before this commit).
	p := &fakeProvider{responses: []provider.Response{{Content: "ok"}}}
	loop := newLoop(t, p, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, _, err := loop.Run(ctx, "hi")
	if !errors.Is(err, ErrInterrupted) {
		t.Errorf("err = %v, want ErrInterrupted via safetyPhase", err)
	}
}

// contains is a tiny strings.Contains alias kept local so this file
// doesn't import "strings" just for one check.
func contains(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
