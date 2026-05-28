package agents

import (
	"context"
	"encoding/json"
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
