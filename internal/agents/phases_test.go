package agents

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ashish-work/opendev-go/internal/agents/doomloop"
	"github.com/ashish-work/opendev-go/internal/budget"
	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/hooks"
	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/runtime/permissions"
	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/tools/todo"
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

func TestLLMCallPhase_PropagatesReasoningEffortFromSlot(t *testing.T) {
	// The phase reads SlotExecution's ReasoningEffort and stamps
	// it on the Request. End-to-end wiring check: configure the
	// slot, run one call, inspect the provider.Request the fake
	// provider captured.
	p := &fakeProvider{responses: []provider.Response{{Content: "ok"}}}
	loop := NewReactLoop(NewLlmCaller(p, cost.Pricing{}), tools.NewRegistry(), Config{
		Workflow: workflow.Config{
			Execution: workflow.SlotConfig{
				Model:           "fake-model",
				ReasoningEffort: provider.ReasoningEffortHigh,
			},
		},
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
		nil,
		"system prompt",
	)
	pc.Iter = 1

	_ = loop.llmCallPhase(context.Background(), pc)

	if len(p.requests) != 1 {
		t.Fatalf("requests captured = %d, want 1", len(p.requests))
	}
	if got := p.requests[0].ReasoningEffort; got != provider.ReasoningEffortHigh {
		t.Errorf("Request.ReasoningEffort = %q, want %q",
			got, provider.ReasoningEffortHigh)
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

// newProcessTestRig builds a *ReactLoop and *PhaseContext primed for
// processResponsePhase tests. resp is stashed on pc.LastResponse as
// processResponsePhase will read it.
func newProcessTestRig(t *testing.T, resp provider.Response) (*ReactLoop, *PhaseContext) {
	t.Helper()
	loop := &ReactLoop{
		Config: Config{MaxIterations: 25, SystemPrompt: "system prompt"},
	}
	history := []provider.Message{
		SystemMessage("system prompt"),
		UserMessage("hello"),
	}
	pc := NewPhaseContext(
		&history,
		cost.Tracker{CallCount: 3},
		budget.New(128_000),
		doomloop.New(),
		tools.ToolContext{},
		nil,
		"system prompt",
	)
	pc.Iter = 2
	pc.LastResponse = resp
	return loop, pc
}

func TestProcessResponsePhase_NoToolCallsReturnsSuccess(t *testing.T) {
	loop, pc := newProcessTestRig(t, provider.Response{Content: "all done"})

	action := loop.processResponsePhase(context.Background(), pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return", action.Kind)
	}
	if action.Err != nil {
		t.Errorf("Err = %v, want nil for success exit", action.Err)
	}
	if !action.Result.Success {
		t.Errorf("Result.Success = false, want true")
	}
	if action.Result.Content != "all done" {
		t.Errorf("Result.Content = %q, want %q", action.Result.Content, "all done")
	}
	// History should have grown: original 2 + appended assistant = 3.
	if got := len(*pc.History); got != 3 {
		t.Fatalf("history len = %d, want 3 (assistant appended)", got)
	}
	last := (*pc.History)[2]
	if last.Role != "assistant" || last.Content[0].Text != "all done" {
		t.Errorf("appended message = %+v, want assistant 'all done'", last)
	}
}

func TestProcessResponsePhase_NoToolCallsEmptyContentReturnsSuccessNoAppend(t *testing.T) {
	// Empty content + no tool calls is an edge case the original code
	// handles by NOT appending a phantom empty assistant message.
	loop, pc := newProcessTestRig(t, provider.Response{Content: ""})

	action := loop.processResponsePhase(context.Background(), pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return", action.Kind)
	}
	if !action.Result.Success {
		t.Errorf("Result.Success = false, want true")
	}
	if got := len(*pc.History); got != 2 {
		t.Errorf("history len = %d, want 2 (no phantom assistant on empty content)", got)
	}
}

func TestProcessResponsePhase_ToolCallsContinue(t *testing.T) {
	resp := provider.Response{
		Content: "I'll check the file.",
		ToolCalls: []provider.ToolCall{{
			ID: "c1", Name: "read_file",
			Arguments: json.RawMessage(`{"path":"x.go"}`),
		}},
	}
	loop, pc := newProcessTestRig(t, resp)

	action := loop.processResponsePhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue (dispatch path)", action.Kind)
	}

	// History: original 2 + assistant with text + tool_calls = 3.
	if got := len(*pc.History); got != 3 {
		t.Fatalf("history len = %d, want 3", got)
	}
	last := (*pc.History)[2]
	if last.Role != "assistant" {
		t.Errorf("appended role = %q, want assistant", last.Role)
	}
	if len(last.ToolCalls) != 1 || last.ToolCalls[0].ID != "c1" {
		t.Errorf("appended ToolCalls = %+v, want one with ID=c1", last.ToolCalls)
	}
	if len(last.Content) != 1 || last.Content[0].Text != "I'll check the file." {
		t.Errorf("appended Content = %+v, want text block", last.Content)
	}
}

func TestProcessResponsePhase_ToolCallsNoTextStillCommitsAssistant(t *testing.T) {
	// Model emits tool_calls with no text — the assistant message
	// must still be appended (with empty Content) so its tool_calls
	// pair with the upcoming tool_response messages.
	resp := provider.Response{
		ToolCalls: []provider.ToolCall{{ID: "c1", Name: "foo"}},
	}
	loop, pc := newProcessTestRig(t, resp)

	action := loop.processResponsePhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if got := len(*pc.History); got != 3 {
		t.Fatalf("history len = %d, want 3 (assistant appended even without text)", got)
	}
	last := (*pc.History)[2]
	if last.Role != "assistant" {
		t.Errorf("appended role = %q, want assistant", last.Role)
	}
	if len(last.Content) != 0 {
		t.Errorf("Content should be empty when resp.Content is empty; got %+v", last.Content)
	}
}

func TestProcessResponsePhase_ForceStopReturnsErrDoomLoop(t *testing.T) {
	// Drive the detector into ForceStop by repeating the same
	// fingerprint enough times. The detector escalates Redirect →
	// Notify → ForceStop, so three Checks of the same call set
	// reach ForceStop.
	resp := provider.Response{
		ToolCalls: []provider.ToolCall{{
			ID: "c1", Name: "foo",
			Arguments: json.RawMessage(`{}`),
		}},
	}
	loop, pc := newProcessTestRig(t, resp)
	// Pre-poison the detector so this iteration's Check is the one
	// that escalates to ForceStop. The detector escalates per
	// detected cycle: it takes 3 repeats to trigger the first
	// Redirect, a 4th to trigger Notify, and a 5th to trigger
	// ForceStop (nudgeCount = 3). So 4 priming Checks plus the
	// phase's own Check makes the phase's call the 5th.
	for i := 0; i < 4; i++ {
		pc.Detector.Check(resp.ToolCalls)
	}

	action := loop.processResponsePhase(context.Background(), pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return (ForceStop)", action.Kind)
	}
	if !errors.Is(action.Err, ErrDoomLoop) {
		t.Errorf("Err = %v, want chain containing ErrDoomLoop", action.Err)
	}
	// CRITICAL: history should have the SystemMessage warning, NOT
	// the assistant message (whose tool_calls would orphan tool
	// responses).
	if got := len(*pc.History); got != 3 {
		t.Fatalf("history len = %d, want 3 (system warning appended)", got)
	}
	last := (*pc.History)[2]
	if last.Role != "system" {
		t.Errorf("appended role = %q, want system (warning)", last.Role)
	}
	// No assistant message should appear after the user message.
	for i, m := range *pc.History {
		if m.Role == "assistant" {
			t.Errorf("unexpected assistant message at index %d on ForceStop path: %+v", i, m)
		}
	}
}

func TestProcessResponsePhase_StashesVerdictOnContinuePath(t *testing.T) {
	// On the dispatch path (not ForceStop), the verdict should be
	// stashed on pc so the still-inline Redirect / Notify post-
	// dispatch logic can consume it.
	resp := provider.Response{
		ToolCalls: []provider.ToolCall{{ID: "c1", Name: "foo"}},
	}
	loop, pc := newProcessTestRig(t, resp)

	_ = loop.processResponsePhase(context.Background(), pc)
	// First Check of a fresh detector produces None (no repeats
	// yet). Verdict fields are still populated — the still-inline
	// Redirect/Notify code below reads them.
	if pc.DoomLoopAction != doomloop.None {
		t.Errorf("DoomLoopAction = %v, want None (first check)", pc.DoomLoopAction)
	}
}

func TestProcessResponsePhase_TrackerPassesThrough(t *testing.T) {
	resp := provider.Response{Content: "done"}
	loop, pc := newProcessTestRig(t, resp)

	action := loop.processResponsePhase(context.Background(), pc)
	if action.Tracker.CallCount != 3 {
		t.Errorf("Tracker.CallCount = %d, want 3 (preserved through process_response)",
			action.Tracker.CallCount)
	}
}

// newExecuteTestRig builds a *ReactLoop with a tool registry plus
// the given tool, and a *PhaseContext primed with the given
// LastResponse so executeSequentialPhase has something to dispatch.
func newExecuteTestRig(t *testing.T, tool tools.Tool, lastResp provider.Response) (*ReactLoop, *PhaseContext) {
	t.Helper()
	reg := tools.NewRegistry()
	if tool != nil {
		if err := reg.Register(tool); err != nil {
			t.Fatalf("register tool: %v", err)
		}
	}
	loop := &ReactLoop{
		Registry: reg,
		Config:   Config{SystemPrompt: "system prompt"},
	}
	history := []provider.Message{
		SystemMessage("system prompt"),
		UserMessage("hello"),
	}
	pc := NewPhaseContext(
		&history,
		cost.Tracker{CallCount: 4},
		budget.New(0),
		doomloop.New(),
		tools.ToolContext{WorkingDir: "/tmp"},
		nil,
		"system prompt",
	)
	pc.Iter = 3
	pc.LastResponse = lastResp
	return loop, pc
}

func TestExecuteSequentialPhase_EmptyToolCallsContinues(t *testing.T) {
	loop, pc := newExecuteTestRig(t, nil, provider.Response{})

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Errorf("Kind = %s, want continue (no tools to dispatch)", action.Kind)
	}
	if got := len(*pc.History); got != 2 {
		t.Errorf("history len = %d, want 2 (unchanged)", got)
	}
}

func TestExecuteSequentialPhase_SingleSuccessfulDispatchAppendsToolResult(t *testing.T) {
	tool := &fakeTool{
		name:   "foo",
		desc:   "foo tool",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Output: "tool said hi", Success: true}, nil
		},
	}
	resp := provider.Response{
		ToolCalls: []provider.ToolCall{{ID: "c1", Name: "foo"}},
	}
	loop, pc := newExecuteTestRig(t, tool, resp)

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if got := len(*pc.History); got != 3 {
		t.Fatalf("history len = %d, want 3 (one tool result appended)", got)
	}
	last := (*pc.History)[2]
	if last.Role != "tool" {
		t.Errorf("appended role = %q, want tool", last.Role)
	}
	if last.ToolCallID != "c1" {
		t.Errorf("appended ToolCallID = %q, want c1", last.ToolCallID)
	}
	if last.Name != "foo" {
		t.Errorf("appended Name = %q, want foo", last.Name)
	}
	if tool.calls != 1 {
		t.Errorf("tool.calls = %d, want 1", tool.calls)
	}
}

func TestExecuteSequentialPhase_MultipleToolsDispatchedInOrder(t *testing.T) {
	dispatched := []string{}
	makeTool := func(name string) *fakeTool {
		return &fakeTool{
			name:   name,
			schema: json.RawMessage(`{"type":"object"}`),
			exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
				dispatched = append(dispatched, name)
				return tools.ToolResult{Output: name + "-result", Success: true}, nil
			},
		}
	}
	// Register both tools individually since the rig's helper only
	// takes one.
	reg := tools.NewRegistry()
	for _, n := range []string{"alpha", "beta"} {
		if err := reg.Register(makeTool(n)); err != nil {
			t.Fatalf("register %s: %v", n, err)
		}
	}
	loop := &ReactLoop{Registry: reg, Config: Config{SystemPrompt: "sys"}}
	history := []provider.Message{SystemMessage("sys"), UserMessage("hi")}
	pc := NewPhaseContext(&history, cost.Tracker{}, budget.New(0), doomloop.New(),
		tools.ToolContext{}, nil, "sys")
	pc.LastResponse = provider.Response{
		ToolCalls: []provider.ToolCall{
			{ID: "c1", Name: "alpha"},
			{ID: "c2", Name: "beta"},
		},
	}

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if len(dispatched) != 2 || dispatched[0] != "alpha" || dispatched[1] != "beta" {
		t.Errorf("dispatch order = %v, want [alpha beta]", dispatched)
	}
	if got := len(*pc.History); got != 4 {
		t.Fatalf("history len = %d, want 4 (two tool results appended)", got)
	}
	if (*pc.History)[2].ToolCallID != "c1" || (*pc.History)[3].ToolCallID != "c2" {
		t.Errorf("appended order wrong: [%s, %s]",
			(*pc.History)[2].ToolCallID, (*pc.History)[3].ToolCallID)
	}
}

func TestExecuteSequentialPhase_ToolDomainFailureIsObservationNotError(t *testing.T) {
	// Tool returns Success: false — that's a domain failure the
	// model should react to, NOT an infrastructure error. Phase
	// returns Continue; result still appended.
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Output: "", Error: "not found", Success: false}, nil
		},
	}
	resp := provider.Response{
		ToolCalls: []provider.ToolCall{{ID: "c1", Name: "foo"}},
	}
	loop, pc := newExecuteTestRig(t, tool, resp)

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Errorf("Kind = %s, want continue (Success:false is observation, not error)", action.Kind)
	}
	if got := len(*pc.History); got != 3 {
		t.Errorf("history len = %d, want 3 (failed result still appended)", got)
	}
}

func TestExecuteSequentialPhase_InfrastructureErrorReturnsErrToolExec(t *testing.T) {
	// Tool's Execute returns an error (infrastructure failure) →
	// phase returns Return with ErrToolExec wrapping.
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{}, errors.New("disk on fire")
		},
	}
	resp := provider.Response{
		ToolCalls: []provider.ToolCall{{ID: "c1", Name: "foo"}},
	}
	loop, pc := newExecuteTestRig(t, tool, resp)

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return", action.Kind)
	}
	if !errors.Is(action.Err, ErrToolExec) {
		t.Errorf("Err = %v, want chain containing ErrToolExec", action.Err)
	}
	if !contains(action.Err.Error(), "foo") {
		t.Errorf("error %q should name the failing tool", action.Err)
	}
}

func TestExecuteSequentialPhase_UnknownToolFromRegistryReturnsErrToolExec(t *testing.T) {
	// ToolCalls references a tool the registry doesn't have →
	// Dispatch returns an error → phase wraps as ErrToolExec.
	resp := provider.Response{
		ToolCalls: []provider.ToolCall{{ID: "c1", Name: "nonexistent"}},
	}
	loop, pc := newExecuteTestRig(t, nil, resp) // no tools registered

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return", action.Kind)
	}
	if !errors.Is(action.Err, ErrToolExec) {
		t.Errorf("Err = %v, want chain containing ErrToolExec", action.Err)
	}
}

func TestExecuteSequentialPhase_TrackerPassesThroughOnSuccess(t *testing.T) {
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Output: "ok", Success: true}, nil
		},
	}
	resp := provider.Response{ToolCalls: []provider.ToolCall{{ID: "c1", Name: "foo"}}}
	loop, pc := newExecuteTestRig(t, tool, resp)

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Tracker.CallCount != 4 {
		t.Errorf("Tracker.CallCount = %d, want 4 (preserved through phase)",
			action.Tracker.CallCount)
	}
}

func TestHandleCompletionPhase_NoneVerdictDoesNothing(t *testing.T) {
	loop, pc := newExecuteTestRig(t, nil, provider.Response{})
	pc.DoomLoopAction = doomloop.None

	before := len(*pc.History)
	action := loop.handleCompletionPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Errorf("Kind = %s, want continue", action.Kind)
	}
	if len(*pc.History) != before {
		t.Errorf("history len = %d, want %d (no append on None)",
			len(*pc.History), before)
	}
}

func TestHandleCompletionPhase_RedirectAppendsSystemMessage(t *testing.T) {
	loop, pc := newExecuteTestRig(t, nil, provider.Response{})
	pc.DoomLoopAction = doomloop.Redirect
	pc.DoomLoopWarning = "you are repeating yourself"
	pc.DoomLoopRecovery = "try a different approach"

	action := loop.handleCompletionPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	last := (*pc.History)[len(*pc.History)-1]
	if last.Role != "system" {
		t.Errorf("appended role = %q, want system", last.Role)
	}
	if !contains(last.Content[0].Text, "repeating") {
		t.Errorf("warning text missing: %q", last.Content[0].Text)
	}
	if !contains(last.Content[0].Text, "different approach") {
		t.Errorf("recovery text missing: %q", last.Content[0].Text)
	}
}

func TestHandleCompletionPhase_NotifyAppendsSystemMessage(t *testing.T) {
	loop, pc := newExecuteTestRig(t, nil, provider.Response{})
	pc.DoomLoopAction = doomloop.Notify
	pc.DoomLoopWarning = "stop"
	pc.DoomLoopRecovery = "no really stop"

	action := loop.handleCompletionPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Errorf("Kind = %s, want continue", action.Kind)
	}
	last := (*pc.History)[len(*pc.History)-1]
	if last.Role != "system" {
		t.Errorf("appended role = %q, want system", last.Role)
	}
}

func TestHandleCompletionPhase_ForceStopDoesNotAppend(t *testing.T) {
	// ForceStop shouldn't reach handle_completion (process_response
	// returns first), but if it does we don't double-append the
	// warning. The phase just returns Continue with no append.
	loop, pc := newExecuteTestRig(t, nil, provider.Response{})
	pc.DoomLoopAction = doomloop.ForceStop
	pc.DoomLoopWarning = "stop"

	before := len(*pc.History)
	action := loop.handleCompletionPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Errorf("Kind = %s, want continue", action.Kind)
	}
	if len(*pc.History) != before {
		t.Errorf("ForceStop should not append in handle_completion; len = %d, want %d",
			len(*pc.History), before)
	}
}

// makeHookMatcher builds a HookMatcher with the regex compiled.
// Tests that integrate hooks construct matchers by hand rather than
// going through the JSON loader. The exported compiled field has to
// be set via the same internal mechanism LoadFile uses; we use
// regexp.Compile directly and call .Matches to verify Behavior
// elsewhere — this helper only needs to populate the source string
// + command + (optional) timeout, since matcher matching for these
// tests uses the empty-matcher-always-matches path in most cases.
func makeHookMatcher(t *testing.T, pattern, command string) hooks.HookMatcher {
	t.Helper()
	// LoadFile's compileMatchers is internal, so we construct a
	// minimal settings file and parse it.
	var settings hooks.HookSettings
	if pattern == "" {
		// Skip the JSON dance for empty matchers; the public API
		// already says empty patterns always match.
		return hooks.HookMatcher{Command: command}
	}
	tmp := writeTempSettings(t, pattern, command)
	var err error
	settings, err = hooks.LoadFile(tmp)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	matchers := settings.MatchersFor(hooks.HookEventPreToolUse)
	if len(matchers) != 1 {
		t.Fatalf("expected 1 matcher; got %d", len(matchers))
	}
	return matchers[0]
}

// writeTempSettings writes a one-matcher settings file under
// pre_tool_use and returns the path.
func writeTempSettings(t *testing.T, pattern, command string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "settings.json")
	patternJSON, _ := json.Marshal(pattern)
	commandJSON, _ := json.Marshal(command)
	body := `{"hooks":{"pre_tool_use":[{"matcher":` +
		string(patternJSON) + `,"command":` + string(commandJSON) + `}]}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	return path
}

// newHookExecuteTestRig builds a *ReactLoop with the given tool
// registered AND a Hooks manager configured with `matchers` under
// the given event. Returns the loop, the pc primed with one
// tool_call to dispatch, and a pointer to the underlying fakeTool
// so tests can inspect whether it was actually called.
func newHookExecuteTestRig(
	t *testing.T,
	tool *fakeTool,
	event hooks.HookEvent,
	matchers []hooks.HookMatcher,
) (*ReactLoop, *PhaseContext) {
	t.Helper()
	reg := tools.NewRegistry()
	if tool != nil {
		if err := reg.Register(tool); err != nil {
			t.Fatalf("register: %v", err)
		}
	}
	settings := hooks.HookSettings{
		Hooks: map[hooks.HookEvent][]hooks.HookMatcher{event: matchers},
	}
	loop := &ReactLoop{
		Registry: reg,
		Config:   Config{SystemPrompt: "sys"},
		Hooks:    hooks.NewManager(settings, hooks.NewExecutor("")),
	}
	history := []provider.Message{SystemMessage("sys"), UserMessage("hi")}
	pc := NewPhaseContext(
		&history, cost.Tracker{}, budget.New(0), doomloop.New(),
		tools.ToolContext{}, nil, "sys")
	pc.Iter = 1
	pc.LastResponse = provider.Response{
		ToolCalls: []provider.ToolCall{{
			ID:        "c1",
			Name:      tool.name,
			Arguments: json.RawMessage(`{"original":true}`),
		}},
	}
	return loop, pc
}

func TestExecuteSequentialPhase_PreToolUseAllowDispatchesNormally(t *testing.T) {
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Output: "tool result", Success: true}, nil
		},
	}
	matcher := makeHookMatcher(t, "",
		`echo '{"permissionDecision":"allow"}'`)
	loop, pc := newHookExecuteTestRig(t, tool, hooks.HookEventPreToolUse,
		[]hooks.HookMatcher{matcher})

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if tool.calls != 1 {
		t.Errorf("tool.calls = %d, want 1 (allow should dispatch)", tool.calls)
	}
	if got := len(*pc.History); got != 3 {
		t.Fatalf("history len = %d, want 3", got)
	}
	if (*pc.History)[2].Role != "tool" {
		t.Errorf("appended role = %q, want tool", (*pc.History)[2].Role)
	}
}

func TestExecuteSequentialPhase_PreToolUseDenySkipsDispatch(t *testing.T) {
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			t.Errorf("tool.Execute should NOT have been called on deny")
			return tools.ToolResult{}, nil
		},
	}
	matcher := makeHookMatcher(t, "",
		`echo '{"permissionDecision":"deny","reason":"forbidden by policy"}'`)
	loop, pc := newHookExecuteTestRig(t, tool, hooks.HookEventPreToolUse,
		[]hooks.HookMatcher{matcher})

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue (deny is observation, not error)", action.Kind)
	}
	if tool.calls != 0 {
		t.Errorf("tool.calls = %d, want 0 (deny should skip dispatch)", tool.calls)
	}
	if got := len(*pc.History); got != 3 {
		t.Fatalf("history len = %d, want 3 (synthetic blocked result appended)", got)
	}
	last := (*pc.History)[2]
	if last.Role != "tool" {
		t.Errorf("appended role = %q, want tool (synthetic ToolResultMessage)", last.Role)
	}
	if last.ToolCallID != "c1" {
		t.Errorf("ToolCallID = %q, want c1 (pairing preserved)", last.ToolCallID)
	}
	if !strings.Contains(last.Content[0].Text, "forbidden by policy") {
		t.Errorf("blocked output should mention the reason; got %q", last.Content[0].Text)
	}
}

func TestExecuteSequentialPhase_PreToolUseUpdatedInputReplacesArgs(t *testing.T) {
	var receivedArgs json.RawMessage
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, args json.RawMessage) (tools.ToolResult, error) {
			receivedArgs = args
			return tools.ToolResult{Output: "ok", Success: true}, nil
		},
	}
	matcher := makeHookMatcher(t, "",
		`echo '{"updatedInput":{"rewritten":true}}'`)
	loop, pc := newHookExecuteTestRig(t, tool, hooks.HookEventPreToolUse,
		[]hooks.HookMatcher{matcher})

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if !strings.Contains(string(receivedArgs), "rewritten") {
		t.Errorf("tool received args = %q, want rewritten JSON", receivedArgs)
	}
	if strings.Contains(string(receivedArgs), "original") {
		t.Errorf("original args should NOT reach the tool; got %q", receivedArgs)
	}
}

func TestExecuteSequentialPhase_PreToolUseContextAppendedAfterToolResult(t *testing.T) {
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Output: "tool output", Success: true}, nil
		},
	}
	matcher := makeHookMatcher(t, "",
		`echo '{"additionalContext":"audit logged"}'`)
	loop, pc := newHookExecuteTestRig(t, tool, hooks.HookEventPreToolUse,
		[]hooks.HookMatcher{matcher})

	_ = loop.executeSequentialPhase(context.Background(), pc)
	// History: system, user, tool, system(context) — order matters
	// because the tool_call → tool_response pairing must not have
	// anything between.
	if got := len(*pc.History); got != 4 {
		t.Fatalf("history len = %d, want 4", got)
	}
	if (*pc.History)[2].Role != "tool" {
		t.Errorf("history[2].Role = %q, want tool (must come immediately after tool_call)",
			(*pc.History)[2].Role)
	}
	if (*pc.History)[3].Role != "system" {
		t.Errorf("history[3].Role = %q, want system (PreToolUse context)",
			(*pc.History)[3].Role)
	}
	if !strings.Contains((*pc.History)[3].Content[0].Text, "audit logged") {
		t.Errorf("system message should contain hook context")
	}
}

func TestExecuteSequentialPhase_PostToolUseContextAppendedAfterResult(t *testing.T) {
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Output: "ok", Success: true}, nil
		},
	}
	matcher := makeHookMatcher(t, "",
		`echo '{"additionalContext":"post-hook note"}'`)
	loop, pc := newHookExecuteTestRig(t, tool, hooks.HookEventPostToolUse,
		[]hooks.HookMatcher{matcher})

	_ = loop.executeSequentialPhase(context.Background(), pc)
	last := (*pc.History)[len(*pc.History)-1]
	if last.Role != "system" {
		t.Errorf("last role = %q, want system", last.Role)
	}
	if !strings.Contains(last.Content[0].Text, "post-hook note") {
		t.Errorf("PostToolUse context not appended: %q", last.Content[0].Text)
	}
}

func TestExecuteSequentialPhase_BothPreAndPostContextAppearedInOrder(t *testing.T) {
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Output: "ok", Success: true}, nil
		},
	}
	preMatcher := makeHookMatcher(t, "",
		`echo '{"additionalContext":"pre note"}'`)
	postMatcher := makeHookMatcher(t, "",
		`echo '{"additionalContext":"post note"}'`)

	reg := tools.NewRegistry()
	_ = reg.Register(tool)
	settings := hooks.HookSettings{
		Hooks: map[hooks.HookEvent][]hooks.HookMatcher{
			hooks.HookEventPreToolUse:  {preMatcher},
			hooks.HookEventPostToolUse: {postMatcher},
		},
	}
	loop := &ReactLoop{
		Registry: reg,
		Config:   Config{SystemPrompt: "sys"},
		Hooks:    hooks.NewManager(settings, hooks.NewExecutor("")),
	}
	history := []provider.Message{SystemMessage("sys"), UserMessage("hi")}
	pc := NewPhaseContext(
		&history, cost.Tracker{}, budget.New(0), doomloop.New(),
		tools.ToolContext{}, nil, "sys")
	pc.Iter = 1
	pc.LastResponse = provider.Response{
		ToolCalls: []provider.ToolCall{{ID: "c1", Name: "foo"}},
	}

	_ = loop.executeSequentialPhase(context.Background(), pc)
	// Expected: system, user, tool, system(pre note), system(post note)
	if got := len(*pc.History); got != 5 {
		t.Fatalf("history len = %d, want 5", got)
	}
	pre := (*pc.History)[3]
	post := (*pc.History)[4]
	if !strings.Contains(pre.Content[0].Text, "pre note") {
		t.Errorf("expected 'pre note' at index 3; got %q", pre.Content[0].Text)
	}
	if !strings.Contains(post.Content[0].Text, "post note") {
		t.Errorf("expected 'post note' at index 4; got %q", post.Content[0].Text)
	}
}

func TestExecuteSequentialPhase_DenySecondToolFirstStillDispatches(t *testing.T) {
	// Two tool_calls in the same iteration: the matcher denies only
	// when called for the second tool. The first should dispatch
	// normally; the second should be blocked.
	calls := []string{}
	makeTool := func(name string) *fakeTool {
		return &fakeTool{
			name:   name,
			schema: json.RawMessage(`{"type":"object"}`),
			exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
				calls = append(calls, name)
				return tools.ToolResult{Output: name, Success: true}, nil
			},
		}
	}
	t1 := makeTool("alpha")
	t2 := makeTool("beta")

	matcher := hooks.HookMatcher{
		Matcher: "^beta$",
		Command: `echo '{"permissionDecision":"deny","reason":"no beta"}'`,
	}
	// Compile the matcher via the public loader path.
	tmp := writeTempSettings(t, "^beta$",
		`echo '{"permissionDecision":"deny","reason":"no beta"}'`)
	settings, err := hooks.LoadFile(tmp)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	matcher = settings.MatchersFor(hooks.HookEventPreToolUse)[0]

	reg := tools.NewRegistry()
	_ = reg.Register(t1)
	_ = reg.Register(t2)
	loop := &ReactLoop{
		Registry: reg,
		Config:   Config{SystemPrompt: "sys"},
		Hooks: hooks.NewManager(
			hooks.HookSettings{Hooks: map[hooks.HookEvent][]hooks.HookMatcher{
				hooks.HookEventPreToolUse: {matcher},
			}},
			hooks.NewExecutor("")),
	}
	history := []provider.Message{SystemMessage("sys"), UserMessage("hi")}
	pc := NewPhaseContext(&history, cost.Tracker{}, budget.New(0), doomloop.New(),
		tools.ToolContext{}, nil, "sys")
	pc.LastResponse = provider.Response{
		ToolCalls: []provider.ToolCall{
			{ID: "c1", Name: "alpha"},
			{ID: "c2", Name: "beta"},
		},
	}

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if len(calls) != 1 || calls[0] != "alpha" {
		t.Errorf("dispatch sequence = %v, want [alpha] (beta blocked)", calls)
	}
	// History: system, user, tool(alpha-result), tool(beta-blocked-synthetic)
	if got := len(*pc.History); got != 4 {
		t.Fatalf("history len = %d, want 4", got)
	}
	if (*pc.History)[3].ToolCallID != "c2" {
		t.Errorf("synthetic blocked entry should pair with c2; got %q",
			(*pc.History)[3].ToolCallID)
	}
}

func TestExecuteSequentialPhase_CtxCancelDuringPreToolUseReturnsErrInterrupted(t *testing.T) {
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{}, nil
		},
	}
	matcher := makeHookMatcher(t, "", "sleep 10") // would block forever
	loop, pc := newHookExecuteTestRig(t, tool, hooks.HookEventPreToolUse,
		[]hooks.HookMatcher{matcher})

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	action := loop.executeSequentialPhase(ctx, pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return", action.Kind)
	}
	if !errors.Is(action.Err, ErrInterrupted) {
		t.Errorf("err = %v, want chain containing ErrInterrupted", action.Err)
	}
}

func TestExecuteSequentialPhase_PreToolUseTimeoutDoesNotHaltDispatch(t *testing.T) {
	// Per-hook timeout is swallowed by the manager. The phase
	// should treat it as "no opinion" and dispatch normally. Build
	// the matcher directly so TimeoutMs is unambiguous (LoadFile
	// returns a value, and chaining mutations through the slice
	// can mask intent).
	dispatched := false
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			dispatched = true
			return tools.ToolResult{Output: "ok", Success: true}, nil
		},
	}
	matcher := hooks.HookMatcher{
		Command:   "sleep 10",
		TimeoutMs: 200,
	}
	loop, pc := newHookExecuteTestRig(t, tool, hooks.HookEventPreToolUse,
		[]hooks.HookMatcher{matcher})

	start := time.Now()
	action := loop.executeSequentialPhase(context.Background(), pc)
	elapsed := time.Since(start)

	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue (timeout = no opinion); err = %v",
			action.Kind, action.Err)
	}
	if !dispatched {
		t.Errorf("tool should have dispatched after PreToolUse timeout")
	}
	if elapsed > 5*time.Second {
		t.Errorf("phase took %s — timeout apparently not enforced", elapsed)
	}
}

func TestExecuteSequentialPhase_NilHooksRegressionSafe(t *testing.T) {
	// Loop without a Hooks manager should behave exactly as the
	// pre-#34 version. Pinned defensively.
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Output: "ok", Success: true}, nil
		},
	}
	loop, pc := newExecuteTestRig(t, tool, provider.Response{
		ToolCalls: []provider.ToolCall{{ID: "c1", Name: "foo"}},
	})
	// loop.Hooks is nil because newExecuteTestRig doesn't set it.

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Errorf("Kind = %s, want continue", action.Kind)
	}
	if got := len(*pc.History); got != 3 {
		t.Errorf("history len = %d, want 3 (no hooks = pre-#34 behavior)", got)
	}
}

func TestIsHomogeneousSpawnBatch(t *testing.T) {
	cases := []struct {
		name  string
		calls []provider.ToolCall
		want  bool
	}{
		{
			name:  "empty batch",
			calls: nil,
			want:  false,
		},
		{
			name:  "single spawn",
			calls: []provider.ToolCall{{Name: spawnSubagentToolName}},
			want:  false, // single-call optimization
		},
		{
			name: "two spawns",
			calls: []provider.ToolCall{
				{Name: spawnSubagentToolName},
				{Name: spawnSubagentToolName},
			},
			want: true,
		},
		{
			name: "three spawns",
			calls: []provider.ToolCall{
				{Name: spawnSubagentToolName},
				{Name: spawnSubagentToolName},
				{Name: spawnSubagentToolName},
			},
			want: true,
		},
		{
			name: "spawn plus other",
			calls: []provider.ToolCall{
				{Name: spawnSubagentToolName},
				{Name: "bash"},
			},
			want: false,
		},
		{
			name: "all other",
			calls: []provider.ToolCall{
				{Name: "bash"},
				{Name: "read_file"},
			},
			want: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := isHomogeneousSpawnBatch(c.calls); got != c.want {
				t.Errorf("got %v, want %v", got, c.want)
			}
		})
	}
}

// timingTool is a fakeTool that name-matches spawn_subagent (so the
// homogeneous-batch detector picks it up) and tracks when each
// invocation started + the max concurrent in-flight count.
type timingTool struct {
	mu          sync.Mutex
	startTimes  []time.Time
	inFlight    int
	maxInFlight int
	sleepFor    time.Duration
}

func (t *timingTool) Name() string            { return spawnSubagentToolName }
func (t *timingTool) Description() string     { return "fake spawn for tests" }
func (t *timingTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (t *timingTool) Execute(ctx context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
	t.mu.Lock()
	t.startTimes = append(t.startTimes, time.Now())
	t.inFlight++
	if t.inFlight > t.maxInFlight {
		t.maxInFlight = t.inFlight
	}
	t.mu.Unlock()

	select {
	case <-time.After(t.sleepFor):
	case <-ctx.Done():
	}

	t.mu.Lock()
	t.inFlight--
	t.mu.Unlock()
	return tools.ToolResult{Output: "ok", Success: true}, nil
}

// newParallelTestRig builds a *ReactLoop + *PhaseContext primed
// with the given pc.LastResponse.ToolCalls, plus a timingTool
// registered under spawn_subagent. Returns the loop, pc, and the
// timingTool for inspection.
func newParallelTestRig(t *testing.T, calls []provider.ToolCall, sleepFor time.Duration) (*ReactLoop, *PhaseContext, *timingTool) {
	t.Helper()
	tt := &timingTool{sleepFor: sleepFor}
	reg := tools.NewRegistry()
	if err := reg.Register(tt); err != nil {
		t.Fatalf("register: %v", err)
	}
	loop := &ReactLoop{
		Registry: reg,
		Config:   Config{SystemPrompt: "sys"},
	}
	history := []provider.Message{SystemMessage("sys"), UserMessage("hi")}
	pc := NewPhaseContext(&history, cost.Tracker{}, budget.New(0), doomloop.New(),
		tools.ToolContext{}, nil, "sys")
	pc.LastResponse = provider.Response{ToolCalls: calls}
	return loop, pc, tt
}

func TestExecuteSequentialPhase_ParallelSpawnBatchRunsConcurrently(t *testing.T) {
	// Three "spawn" calls each sleep 200ms. If serial, total
	// wall-clock would be ~600ms. If parallel, ~200ms (max of the
	// three). Allow generous margin for test environment slowness.
	calls := []provider.ToolCall{
		{ID: "c1", Name: spawnSubagentToolName, Arguments: json.RawMessage(`{}`)},
		{ID: "c2", Name: spawnSubagentToolName, Arguments: json.RawMessage(`{}`)},
		{ID: "c3", Name: spawnSubagentToolName, Arguments: json.RawMessage(`{}`)},
	}
	loop, pc, tt := newParallelTestRig(t, calls, 200*time.Millisecond)

	start := time.Now()
	action := loop.executeSequentialPhase(context.Background(), pc)
	elapsed := time.Since(start)

	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if elapsed > 500*time.Millisecond {
		t.Errorf("elapsed = %s, want <500ms (sequential would be ~600ms)", elapsed)
	}
	if tt.maxInFlight < 2 {
		t.Errorf("maxInFlight = %d, want >=2 (proves parallelism)", tt.maxInFlight)
	}
	// History should have 3 tool results in declaration order.
	if got := len(*pc.History); got != 5 { // sys + user + 3 tool results
		t.Fatalf("history len = %d, want 5", got)
	}
	for i, want := range []string{"c1", "c2", "c3"} {
		if got := (*pc.History)[2+i].ToolCallID; got != want {
			t.Errorf("history[%d].ToolCallID = %q, want %q", 2+i, got, want)
		}
	}
}

func TestExecuteSequentialPhase_ParallelSpawnBatchPreservesOrder(t *testing.T) {
	// Five spawns with random sleep durations. Whichever finishes
	// first/last, history must reflect the original tool_call
	// order (c1, c2, c3, c4, c5).
	calls := []provider.ToolCall{
		{ID: "c1", Name: spawnSubagentToolName, Arguments: json.RawMessage(`{}`)},
		{ID: "c2", Name: spawnSubagentToolName, Arguments: json.RawMessage(`{}`)},
		{ID: "c3", Name: spawnSubagentToolName, Arguments: json.RawMessage(`{}`)},
		{ID: "c4", Name: spawnSubagentToolName, Arguments: json.RawMessage(`{}`)},
		{ID: "c5", Name: spawnSubagentToolName, Arguments: json.RawMessage(`{}`)},
	}
	loop, pc, _ := newParallelTestRig(t, calls, 50*time.Millisecond)

	_ = loop.executeSequentialPhase(context.Background(), pc)

	want := []string{"c1", "c2", "c3", "c4", "c5"}
	for i, w := range want {
		got := (*pc.History)[2+i].ToolCallID
		if got != w {
			t.Errorf("history[%d].ToolCallID = %q, want %q (declaration order)",
				2+i, got, w)
		}
	}
}

func TestExecuteSequentialPhase_ParallelSpawnBatchCapsConcurrency(t *testing.T) {
	// Six spawns, each sleeping 200ms. Semaphore caps in-flight at
	// MaxParallelSpawn (4). Verify maxInFlight stayed <= cap.
	calls := make([]provider.ToolCall, 6)
	for i := range calls {
		calls[i] = provider.ToolCall{
			ID:        fmt.Sprintf("c%d", i),
			Name:      spawnSubagentToolName,
			Arguments: json.RawMessage(`{}`),
		}
	}
	loop, pc, tt := newParallelTestRig(t, calls, 200*time.Millisecond)

	_ = loop.executeSequentialPhase(context.Background(), pc)

	if tt.maxInFlight > MaxParallelSpawn {
		t.Errorf("maxInFlight = %d, want <= MaxParallelSpawn (%d)",
			tt.maxInFlight, MaxParallelSpawn)
	}
}

func TestExecuteSequentialPhase_MixedBatchFallsBackToSequential(t *testing.T) {
	// Mixed spawn + other tool stays on the sequential path —
	// no parallelism, dispatch order matches tool_call order.
	var dispatched []string
	var mu sync.Mutex
	recordingTool := &fakeTool{
		name:   "recorder",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			mu.Lock()
			dispatched = append(dispatched, "recorder")
			mu.Unlock()
			return tools.ToolResult{Output: "rec", Success: true}, nil
		},
	}
	spawnRecorder := &fakeTool{
		name:   spawnSubagentToolName,
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			mu.Lock()
			dispatched = append(dispatched, spawnSubagentToolName)
			mu.Unlock()
			return tools.ToolResult{Output: "spawn", Success: true}, nil
		},
	}
	reg := tools.NewRegistry()
	_ = reg.Register(recordingTool)
	_ = reg.Register(spawnRecorder)
	loop := &ReactLoop{Registry: reg, Config: Config{SystemPrompt: "sys"}}
	history := []provider.Message{SystemMessage("sys"), UserMessage("hi")}
	pc := NewPhaseContext(&history, cost.Tracker{}, budget.New(0), doomloop.New(),
		tools.ToolContext{}, nil, "sys")
	pc.LastResponse = provider.Response{
		ToolCalls: []provider.ToolCall{
			{ID: "c1", Name: spawnSubagentToolName},
			{ID: "c2", Name: "recorder"},
		},
	}

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	// Sequential dispatch: spawn first, then recorder.
	want := []string{spawnSubagentToolName, "recorder"}
	if !reflect.DeepEqual(dispatched, want) {
		t.Errorf("dispatch order = %v, want %v (sequential fallback)", dispatched, want)
	}
}

func TestExecuteSequentialPhase_SingleSpawnFallsBackToSequential(t *testing.T) {
	// Single spawn — even though it's "all spawn," the predicate
	// returns false so we skip the goroutine overhead.
	calls := []provider.ToolCall{
		{ID: "c1", Name: spawnSubagentToolName, Arguments: json.RawMessage(`{}`)},
	}
	loop, pc, tt := newParallelTestRig(t, calls, 50*time.Millisecond)

	_ = loop.executeSequentialPhase(context.Background(), pc)

	if tt.maxInFlight != 1 {
		t.Errorf("maxInFlight = %d, want 1 (single call, no parallel)", tt.maxInFlight)
	}
}

func TestExecuteSequentialPhase_ParallelBatchCtxCancelReturnsErrInterrupted(t *testing.T) {
	// Pre-cancelled ctx → the per-call ctx checks inside
	// executeOneCall (via the hook fire path) trip first; even
	// without hooks, the dispatcher's post-wg.Wait check catches
	// it. Either way: ErrInterrupted.
	calls := []provider.ToolCall{
		{ID: "c1", Name: spawnSubagentToolName, Arguments: json.RawMessage(`{}`)},
		{ID: "c2", Name: spawnSubagentToolName, Arguments: json.RawMessage(`{}`)},
	}
	loop, pc, _ := newParallelTestRig(t, calls, 50*time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	action := loop.executeSequentialPhase(ctx, pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return", action.Kind)
	}
	if !errors.Is(action.Err, ErrInterrupted) {
		t.Errorf("Err = %v, want chain containing ErrInterrupted", action.Err)
	}
}

func TestExecuteSequentialPhase_ParallelBatchPropagatesInfrastructureError(t *testing.T) {
	// Calls reference a non-existent tool — Registry.Dispatch
	// returns "tool not found" error. The first failing goroutine's
	// action wins; phase returns ErrToolExec.
	reg := tools.NewRegistry() // empty — no spawn_subagent registered
	loop := &ReactLoop{Registry: reg, Config: Config{SystemPrompt: "sys"}}
	history := []provider.Message{SystemMessage("sys"), UserMessage("hi")}
	pc := NewPhaseContext(&history, cost.Tracker{}, budget.New(0), doomloop.New(),
		tools.ToolContext{}, nil, "sys")
	pc.LastResponse = provider.Response{
		ToolCalls: []provider.ToolCall{
			{ID: "c1", Name: spawnSubagentToolName},
			{ID: "c2", Name: spawnSubagentToolName},
		},
	}

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionReturn {
		t.Fatalf("Kind = %s, want return", action.Kind)
	}
	if !errors.Is(action.Err, ErrToolExec) {
		t.Errorf("Err = %v, want chain containing ErrToolExec", action.Err)
	}
}

// newPermissionsTestRig builds a *ReactLoop with the given tool and
// permission policy, plus a pc primed to dispatch one tool_call.
// Mirrors newHookExecuteTestRig's shape so permission tests look
// like the hook tests next to them — same boilerplate, same
// inspection points.
func newPermissionsTestRig(
	t *testing.T,
	tool *fakeTool,
	policy permissions.Policy,
	args json.RawMessage,
) (*ReactLoop, *PhaseContext) {
	t.Helper()
	reg := tools.NewRegistry()
	if err := reg.Register(tool); err != nil {
		t.Fatalf("register: %v", err)
	}
	loop := &ReactLoop{
		Registry:    reg,
		Config:      Config{SystemPrompt: "sys"},
		Permissions: policy,
	}
	history := []provider.Message{SystemMessage("sys"), UserMessage("hi")}
	pc := NewPhaseContext(
		&history, cost.Tracker{}, budget.New(0), doomloop.New(),
		tools.ToolContext{}, nil, "sys")
	pc.Iter = 1
	pc.LastResponse = provider.Response{
		ToolCalls: []provider.ToolCall{{
			ID:        "c1",
			Name:      tool.name,
			Arguments: args,
		}},
	}
	return loop, pc
}

func TestExecuteSequentialPhase_PermissionsAllowDispatchesNormally(t *testing.T) {
	// Policy with an Enabled=true bash entry and no deny patterns
	// must let the dispatch run. This is the "user wrote
	// {bash: {deny_patterns: []}}" case — the entry exists but
	// allows everything.
	settingsPath := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(settingsPath,
		[]byte(`{"permissions":{"foo":{"deny_patterns":["never-matches-zzz"]}}}`),
		0o644); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	policy, err := permissions.LoadFile(settingsPath)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}

	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Output: "ok", Success: true}, nil
		},
	}
	loop, pc := newPermissionsTestRig(t, tool, policy,
		json.RawMessage(`{"command":"ls"}`))

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if tool.calls != 1 {
		t.Errorf("tool.calls = %d, want 1 (allow should dispatch)", tool.calls)
	}
	if got := len(*pc.History); got != 3 {
		t.Fatalf("history len = %d, want 3 (tool result appended)", got)
	}
	last := (*pc.History)[2]
	if !strings.Contains(last.Content[0].Text, "ok") {
		t.Errorf("tool output should be appended; got %q", last.Content[0].Text)
	}
}

func TestExecuteSequentialPhase_PermissionsDenyMatchedPattern(t *testing.T) {
	// Policy denies bash via a deny pattern that matches the args
	// JSON. The tool must NOT execute, the synthetic tool result
	// must mention the matched pattern, and the LoopAction must be
	// Continue so the model gets a chance to pivot.
	//
	// Constructed through LoadFile so the source-string parallel
	// slice (used in the deny reason) is populated by the real
	// path of record.
	settingsPath := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(settingsPath,
		[]byte(`{"permissions":{"foo":{"deny_patterns":["rm -rf"]}}}`),
		0o644); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	policy, err := permissions.LoadFile(settingsPath)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}

	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			t.Errorf("tool.Execute should NOT have been called on permission deny")
			return tools.ToolResult{}, nil
		},
	}
	loop, pc := newPermissionsTestRig(t, tool, policy,
		json.RawMessage(`{"command":"rm -rf /tmp"}`))

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue (deny is observation, not error)", action.Kind)
	}
	if tool.calls != 0 {
		t.Errorf("tool.calls = %d, want 0 (deny should skip dispatch)", tool.calls)
	}
	if got := len(*pc.History); got != 3 {
		t.Fatalf("history len = %d, want 3 (synthetic blocked result appended)", got)
	}
	last := (*pc.History)[2]
	if last.Role != "tool" {
		t.Errorf("appended role = %q, want tool", last.Role)
	}
	if last.ToolCallID != "c1" {
		t.Errorf("ToolCallID = %q, want c1 (pairing preserved)", last.ToolCallID)
	}
	text := last.Content[0].Text
	if !strings.Contains(text, "denied by policy") {
		t.Errorf("output should mention 'denied by policy'; got %q", text)
	}
	if !strings.Contains(text, "rm -rf") {
		t.Errorf("output should quote the matched pattern; got %q", text)
	}
}

func TestExecuteSequentialPhase_PermissionsDenyDisabledTool(t *testing.T) {
	// Policy explicitly disables the tool (Enabled=false). Same
	// short-circuit, different reason — the deny message should
	// say "disabled by policy", not "matches deny pattern".
	settingsPath := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(settingsPath,
		[]byte(`{"permissions":{"foo":{"enabled":false}}}`),
		0o644); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	policy, err := permissions.LoadFile(settingsPath)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}

	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			t.Errorf("disabled tool must not execute")
			return tools.ToolResult{}, nil
		},
	}
	loop, pc := newPermissionsTestRig(t, tool, policy,
		json.RawMessage(`{"any":"args"}`))

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if tool.calls != 0 {
		t.Errorf("tool.calls = %d, want 0", tool.calls)
	}
	text := (*pc.History)[2].Content[0].Text
	if !strings.Contains(text, "disabled by policy") {
		t.Errorf("output should mention 'disabled by policy'; got %q", text)
	}
}

func TestExecuteSequentialPhase_PermissionsZeroPolicyAllowsEverything(t *testing.T) {
	// Regression guarantee for callers (older tests, simple
	// binaries) that don't set Permissions at all. The zero
	// permissions.Policy must let every tool through, preserving
	// v1 behavior end-to-end.
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Output: "ok", Success: true}, nil
		},
	}
	loop, pc := newPermissionsTestRig(t, tool, permissions.Policy{},
		json.RawMessage(`{"command":"rm -rf /"}`))

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if tool.calls != 1 {
		t.Errorf("tool.calls = %d, want 1 (zero Policy must allow)", tool.calls)
	}
}

func TestExecuteSequentialPhase_HookSanitizesBeforePermissionsCheck(t *testing.T) {
	// Critical ordering test: PreToolUse runs BEFORE permissions
	// check, so a hook that rewrites args (via UpdatedInput) lets
	// the user build a sanitize-then-check workflow. Concrete
	// proof of the design choice documented in executeOneCall.
	//
	// Setup: original args mention "forbidden"; hook strips them
	// via UpdatedInput; policy denies args containing "forbidden".
	// Result: dispatch runs because permissions checks the sanitized
	// args, not the original.
	settingsPath := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(settingsPath,
		[]byte(`{"permissions":{"foo":{"deny_patterns":["forbidden"]}}}`),
		0o644); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	policy, err := permissions.LoadFile(settingsPath)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}

	var receivedArgs json.RawMessage
	tool := &fakeTool{
		name:   "foo",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, args json.RawMessage) (tools.ToolResult, error) {
			receivedArgs = args
			return tools.ToolResult{Output: "ok", Success: true}, nil
		},
	}

	// Hook rewrites args to remove the forbidden token.
	matcher := makeHookMatcher(t, "",
		`echo '{"updatedInput":{"clean":true}}'`)
	settings := hooks.HookSettings{
		Hooks: map[hooks.HookEvent][]hooks.HookMatcher{
			hooks.HookEventPreToolUse: {matcher},
		},
	}
	reg := tools.NewRegistry()
	if err := reg.Register(tool); err != nil {
		t.Fatalf("register: %v", err)
	}
	loop := &ReactLoop{
		Registry:    reg,
		Config:      Config{SystemPrompt: "sys"},
		Hooks:       hooks.NewManager(settings, hooks.NewExecutor("")),
		Permissions: policy,
	}
	history := []provider.Message{SystemMessage("sys"), UserMessage("hi")}
	pc := NewPhaseContext(
		&history, cost.Tracker{}, budget.New(0), doomloop.New(),
		tools.ToolContext{}, nil, "sys")
	pc.Iter = 1
	pc.LastResponse = provider.Response{
		ToolCalls: []provider.ToolCall{{
			ID:        "c1",
			Name:      "foo",
			Arguments: json.RawMessage(`{"command":"forbidden thing"}`),
		}},
	}

	action := loop.executeSequentialPhase(context.Background(), pc)
	if action.Kind != LoopActionContinue {
		t.Fatalf("Kind = %s, want continue", action.Kind)
	}
	if tool.calls != 1 {
		t.Errorf("tool.calls = %d, want 1 (hook sanitized away the deny)", tool.calls)
	}
	if !strings.Contains(string(receivedArgs), "clean") {
		t.Errorf("expected hook-rewritten args at tool, got %q", receivedArgs)
	}
	if strings.Contains(string(receivedArgs), "forbidden") {
		t.Errorf("forbidden token leaked through to tool: %q", receivedArgs)
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

// writeTodoState is a test helper: writes a State JSON file under
// a tempdir and returns the path so the loop's TodoStore can load
// it. Keeps the fixture inline with the test so the expected
// rendering shape is easy to eyeball next to the assertion.
func writeTodoState(t *testing.T, state todo.State) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "todos.json")
	store := todo.NewStore(path)
	if err := store.Save(state); err != nil {
		t.Fatalf("save todo state: %v", err)
	}
	return path
}

func TestLLMCallPhase_NilTodoStoreSkipsInjection(t *testing.T) {
	// Backwards compat: a loop with TodoStore unset (the v1
	// default) must produce a Request whose Messages match
	// history exactly. No injection, no extra allocation.
	p := &fakeProvider{responses: []provider.Response{{Content: "ok"}}}
	loop, pc := newLLMTestRig(t, p, nil)
	loop.TodoStore = nil

	_ = loop.llmCallPhase(context.Background(), pc)

	if len(p.requests) != 1 {
		t.Fatalf("requests captured = %d, want 1", len(p.requests))
	}
	if got, want := len(p.requests[0].Messages), len(*pc.History); got != want {
		t.Errorf("Request.Messages len = %d, want %d (nil store should pass history through)",
			got, want)
	}
}

func TestLLMCallPhase_EmptyTodoStateSkipsInjection(t *testing.T) {
	// A configured store with no todos must not produce a
	// "Current plan: (empty)" injection — pure noise.
	path := writeTodoState(t, todo.State{NextID: 1})
	p := &fakeProvider{responses: []provider.Response{{Content: "ok"}}}
	loop, pc := newLLMTestRig(t, p, nil)
	loop.TodoStore = todo.NewStore(path)

	historyLenBefore := len(*pc.History)
	_ = loop.llmCallPhase(context.Background(), pc)

	if got, want := len(p.requests[0].Messages), historyLenBefore; got != want {
		t.Errorf("Request.Messages len = %d, want %d (empty state should skip)",
			got, want)
	}
}

func TestLLMCallPhase_PopulatedTodoStateInjectsSystemMessage(t *testing.T) {
	// Pre-seed three todos at different statuses, then verify
	// the request gains exactly one extra system message at the
	// END of Messages, containing the rendered plan with markers.
	now := time.Now()
	path := writeTodoState(t, todo.State{
		Todos: []todo.Todo{
			{ID: 1, Title: "read the config", Status: todo.StatusCompleted, CreatedAt: now, UpdatedAt: now},
			{ID: 2, Title: "apply the migration", Status: todo.StatusInProgress, CreatedAt: now, UpdatedAt: now},
			{ID: 3, Title: "run the smoke test", Status: todo.StatusPending, CreatedAt: now, UpdatedAt: now},
		},
		NextID: 4,
	})
	p := &fakeProvider{responses: []provider.Response{{Content: "ok"}}}
	loop, pc := newLLMTestRig(t, p, nil)
	loop.TodoStore = todo.NewStore(path)

	historyLenBefore := len(*pc.History)
	_ = loop.llmCallPhase(context.Background(), pc)

	msgs := p.requests[0].Messages
	if got, want := len(msgs), historyLenBefore+1; got != want {
		t.Fatalf("Request.Messages len = %d, want %d (one injection)",
			got, want)
	}

	last := msgs[len(msgs)-1]
	if last.Role != "system" {
		t.Errorf("injected role = %q, want system", last.Role)
	}
	text := joinContentText(last.Content)
	// Header + rendered markers + each title must all show up.
	for _, sub := range []string{
		"Current plan-of-record",
		"[x] 1. read the config",
		"[~] 2. apply the migration",
		"[ ] 3. run the smoke test",
	} {
		if !strings.Contains(text, sub) {
			t.Errorf("injected content missing %q\nfull text:\n%s", sub, text)
		}
	}
}

func TestLLMCallPhase_TodoInjectionDoesNotMutateHistory(t *testing.T) {
	// The helper must return a fresh slice, not mutate pc.History.
	// Otherwise the conversation log accumulates plan snapshots
	// across iterations and prompt caches drift unpredictably.
	now := time.Now()
	path := writeTodoState(t, todo.State{
		Todos: []todo.Todo{
			{ID: 1, Title: "step 1", Status: todo.StatusInProgress, CreatedAt: now, UpdatedAt: now},
		},
		NextID: 2,
	})
	p := &fakeProvider{responses: []provider.Response{{Content: "ok"}}}
	loop, pc := newLLMTestRig(t, p, nil)
	loop.TodoStore = todo.NewStore(path)

	historyLenBefore := len(*pc.History)
	historyRolesBefore := make([]string, len(*pc.History))
	for i, m := range *pc.History {
		historyRolesBefore[i] = m.Role
	}

	_ = loop.llmCallPhase(context.Background(), pc)

	if got := len(*pc.History); got != historyLenBefore {
		t.Errorf("pc.History len = %d, want %d (history must NOT be mutated)",
			got, historyLenBefore)
	}
	for i, m := range *pc.History {
		if m.Role != historyRolesBefore[i] {
			t.Errorf("pc.History[%d].Role = %q, want %q (mutation detected)",
				i, m.Role, historyRolesBefore[i])
		}
	}
}

func TestLLMCallPhase_TodoInjectionAcrossIterationsSeesFreshState(t *testing.T) {
	// Two iterations: between them we update the on-disk state.
	// Each iteration's request must reflect the disk state AT
	// THAT MOMENT, not a cached snapshot from the first turn.
	now := time.Now()
	path := writeTodoState(t, todo.State{
		Todos: []todo.Todo{
			{ID: 1, Title: "step A", Status: todo.StatusInProgress, CreatedAt: now, UpdatedAt: now},
		},
		NextID: 2,
	})
	store := todo.NewStore(path)

	p := &fakeProvider{
		responses: []provider.Response{
			{Content: "ok 1"},
			{Content: "ok 2"},
		},
	}
	loop, pc := newLLMTestRig(t, p, nil)
	loop.TodoStore = store

	// Iteration 1.
	_ = loop.llmCallPhase(context.Background(), pc)

	// Mutate disk state: complete step A, add step B.
	state, err := store.Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	state, err = state.CompleteOne(1, time.Now())
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	state, _ = state.WithTodos([]todo.Todo{
		{Title: "step A"}, // re-pass to keep ID 1's record; WithTodos rewrites the slate
		{Title: "step B"},
	}, time.Now())
	if err := store.Save(state); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Iteration 2 — note we must re-run llmCallPhase on the same
	// loop; pc.Iter doesn't gate the injection.
	pc.Iter = 2
	_ = loop.llmCallPhase(context.Background(), pc)

	if got := len(p.requests); got != 2 {
		t.Fatalf("captured requests = %d, want 2", got)
	}
	first := joinContentText(p.requests[0].Messages[len(p.requests[0].Messages)-1].Content)
	second := joinContentText(p.requests[1].Messages[len(p.requests[1].Messages)-1].Content)
	if first == second {
		t.Errorf("plan injection identical across iterations — must reflect mutated disk state\nfirst: %s\nsecond: %s",
			first, second)
	}
	if !strings.Contains(second, "step B") {
		t.Errorf("iteration 2 plan should mention 'step B' (added between calls); got:\n%s",
			second)
	}
}

// joinContentText pulls the text out of a ContentBlock slice. The
// agents package doesn't expose a public helper for this; the
// fakeProvider tests above access .Text directly, but the plan-
// injection tests want the joined string for substring checks.
func joinContentText(blocks []provider.ContentBlock) string {
	var b strings.Builder
	for _, c := range blocks {
		if c.Kind == provider.ContentText {
			b.WriteString(c.Text)
		}
	}
	return b.String()
}
