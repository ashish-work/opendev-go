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

// newLLMTestRig wires a *ReactLoop around a fakeProvider with the
// given scripted responses and an empty tools registry. Returns the
// loop and a populated *PhaseContext ready for llmCallPhase calls.
//
// sink is plumbed onto pc.StreamSink — pass nil for the non-streaming
// path, a buffered channel for the streaming path.
func newLLMTestRig(t *testing.T, p *fakeProvider, sink chan<- provider.StreamEvent) (*ReactLoop, *PhaseContext) {
	t.Helper()
	loop := NewReactLoop(NewLlmCaller(p, cost.Pricing{}), tools.NewRegistry(), Config{
		Workflow:      workflow.Config{Execution: workflow.SlotConfig{Model: "fake-model"}},
		MaxIterations: 25,
		SystemPrompt:  "system prompt",
	})
	history := []provider.Message{
		SystemMessage("system prompt"),
		UserMessage("hello"),
	}
	pc := NewPhaseContext(
		&history,
		cost.Tracker{},
		budget.New(128_000),
		doomloop.New(),
		tools.ToolContext{},
		sink,
		"system prompt",
	)
	pc.Iter = 1
	return loop, pc
}

func TestLLMCallPhase_SuccessNonStreaming(t *testing.T) {
	p := &fakeProvider{
		responses: []provider.Response{{
			Content: "hello back",
			Usage:   provider.Usage{PromptTokens: 42, CompletionTokens: 5},
		}},
	}
	loop, pc := newLLMTestRig(t, p, nil)

	action := loop.llmCallPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if pc.LastResponse.Content != "hello back" {
		t.Errorf("LastResponse.Content = %q, want %q", pc.LastResponse.Content, "hello back")
	}
	if pc.Tracker.CallCount != 1 {
		t.Errorf("Tracker.CallCount = %d, want 1", pc.Tracker.CallCount)
	}
	if pc.Calibrator.Reported() != 42 {
		t.Errorf("Calibrator.Reported = %d, want 42 (from response)", pc.Calibrator.Reported())
	}
}

func TestLLMCallPhase_SuccessStreaming(t *testing.T) {
	p := &fakeProvider{
		responses: []provider.Response{{
			Content: "streamed",
			Usage:   provider.Usage{PromptTokens: 30, CompletionTokens: 2},
		}},
	}
	sink := make(chan provider.StreamEvent, 16)
	loop, pc := newLLMTestRig(t, p, sink)

	action := loop.llmCallPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if pc.LastResponse.Content != "streamed" {
		t.Errorf("LastResponse.Content = %q, want %q", pc.LastResponse.Content, "streamed")
	}
	// Streaming path should have forwarded events to the sink.
	close(sink)
	var sawTextDelta, sawDone bool
	for ev := range sink {
		switch ev.Kind {
		case provider.StreamEventTextDelta:
			sawTextDelta = true
		case provider.StreamEventDone:
			sawDone = true
		}
	}
	if !sawTextDelta || !sawDone {
		t.Errorf("expected TextDelta + Done events on streaming path; sawTextDelta=%v sawDone=%v",
			sawTextDelta, sawDone)
	}
}

func TestLLMCallPhase_LLMErrorReturnsErrLLM(t *testing.T) {
	p := &fakeProvider{
		responses: []provider.Response{{}}, // present, but errored below
		errors:    []error{errors.New("api timeout")},
	}
	loop, pc := newLLMTestRig(t, p, nil)

	action := loop.llmCallPhase(context.Background(), pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return", action.Kind)
	}
	if !errors.Is(action.Err, ErrLLM) {
		t.Errorf("Err = %v, want chain containing ErrLLM", action.Err)
	}
	if action.Result.Interrupted {
		t.Errorf("LLM-error path should not set Interrupted=true")
	}
}

func TestLLMCallPhase_CtxCancelReturnsErrInterrupted(t *testing.T) {
	// Real providers honor ctx by returning an error that wraps
	// ctx.Canceled when the context cancels mid-call. fakeProvider
	// is simpler — it just returns whatever we script — so we
	// simulate the situation by scripting an error AND pre-
	// cancelling ctx. llmCallPhase's ctx.Err() check then takes
	// the Interrupted branch.
	p := &fakeProvider{
		responses: []provider.Response{{}},
		errors:    []error{context.Canceled},
	}
	loop, pc := newLLMTestRig(t, p, nil)
	pc.Iter = 7

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	action := loop.llmCallPhase(ctx, pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return", action.Kind)
	}
	if !errors.Is(action.Err, ErrInterrupted) {
		t.Errorf("Err = %v, want chain containing ErrInterrupted", action.Err)
	}
	if !action.Result.Interrupted {
		t.Errorf("ctx-cancel path should set Interrupted=true")
	}
	if !contains(action.Err.Error(), "iter 7") {
		t.Errorf("error %q should mention iter number", action.Err)
	}
}

func TestLLMCallPhase_RequestShapeUsesWorkflowModelAndHistory(t *testing.T) {
	// Capture the request the caller sees. fakeProvider records it
	// on a slice; we inspect after the call.
	p := &fakeProvider{responses: []provider.Response{{Content: "ok"}}}
	loop, pc := newLLMTestRig(t, p, nil)

	_ = loop.llmCallPhase(context.Background(), pc)

	if len(p.requests) != 1 {
		t.Fatalf("requests captured = %d, want 1", len(p.requests))
	}
	req := p.requests[0]
	if req.Model != "fake-model" {
		t.Errorf("Request.Model = %q, want fake-model", req.Model)
	}
	if len(req.Messages) != 2 {
		t.Errorf("Request.Messages len = %d, want 2 (system + user)", len(req.Messages))
	}
}

func TestLLMCallPhase_CalibratorUsesMsgCountAtRequest(t *testing.T) {
	// Calibrator's Update receives len(history) at request time —
	// before any assistant append. With history = [system, user],
	// the count is 2.
	p := &fakeProvider{
		responses: []provider.Response{{
			Content: "ok",
			Usage:   provider.Usage{PromptTokens: 100},
		}},
	}
	loop, pc := newLLMTestRig(t, p, nil)

	_ = loop.llmCallPhase(context.Background(), pc)

	// pc.Calibrator should now report 100 (reported tokens from
	// response). The exact estimate isn't pinned (it depends on
	// budget's heuristic), but Reported is.
	if got := pc.Calibrator.Reported(); got != 100 {
		t.Errorf("Calibrator.Reported = %d, want 100", got)
	}
}

func TestLLMCallPhase_TrackerSurfacedEvenOnError(t *testing.T) {
	// On error paths the phase should still return the tracker so
	// the driver can carry session totals forward.
	p := &fakeProvider{
		responses: []provider.Response{{}},
		errors:    []error{errors.New("boom")},
	}
	loop, pc := newLLMTestRig(t, p, nil)
	pc.Tracker.CallCount = 5 // pretend we already had 5 calls in a session

	action := loop.llmCallPhase(context.Background(), pc)
	// fakeProvider returns the error without incrementing its calls
	// counter logic, but our pre-existing tracker state should be
	// reflected on the action (or unchanged in the error path).
	if action.Tracker.CallCount != 5 {
		t.Errorf("Tracker.CallCount = %d, want 5 (preserved on error)",
			action.Tracker.CallCount)
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
