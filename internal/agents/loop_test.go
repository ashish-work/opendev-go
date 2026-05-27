package agents

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/ashishgupta/opendev-go/internal/cost"
	"github.com/ashishgupta/opendev-go/internal/provider"
	"github.com/ashishgupta/opendev-go/internal/tools"
	"github.com/ashishgupta/opendev-go/internal/workflow"
)

// fakeProvider is a scripted Provider for loop tests. Each Call pops
// the next response off `responses` (or returns the next `errors[i]`).
type fakeProvider struct {
	responses []provider.Response
	errors    []error // parallel to responses; nil = use responses[i]
	calls     int
	requests  []provider.Request // captures every request for inspection
}

func (f *fakeProvider) Name() string { return "fake" }

func (f *fakeProvider) Call(_ context.Context, req provider.Request) (provider.Response, error) {
	if f.calls >= len(f.responses) {
		return provider.Response{}, fmt.Errorf("fakeProvider: out of scripted responses (call #%d)", f.calls+1)
	}
	f.requests = append(f.requests, req)
	i := f.calls
	f.calls++
	if i < len(f.errors) && f.errors[i] != nil {
		return provider.Response{}, f.errors[i]
	}
	return f.responses[i], nil
}

// fakeTool is a scripted Tool. Records every Execute call.
type fakeTool struct {
	name   string
	desc   string
	schema json.RawMessage
	exec   func(ctx context.Context, tctx tools.ToolContext, args json.RawMessage) (tools.ToolResult, error)
	calls  int
}

func (f *fakeTool) Name() string            { return f.name }
func (f *fakeTool) Description() string     { return f.desc }
func (f *fakeTool) Schema() json.RawMessage { return f.schema }
func (f *fakeTool) Execute(ctx context.Context, tctx tools.ToolContext, args json.RawMessage) (tools.ToolResult, error) {
	f.calls++
	return f.exec(ctx, tctx, args)
}

// echoTool returns its args back as Output. Convenience.
func echoTool(name string) *fakeTool {
	return &fakeTool{
		name:   name,
		desc:   name + " — echoes args",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, args json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Success: true, Output: "echo: " + string(args)}, nil
		},
	}
}

// newLoop wires a ReactLoop against a scripted provider and a tools registry.
func newLoop(t *testing.T, provider *fakeProvider, toolsList []tools.Tool) *ReactLoop {
	t.Helper()
	reg := tools.NewRegistry()
	for _, tool := range toolsList {
		if err := reg.Register(tool); err != nil {
			t.Fatalf("register: %v", err)
		}
	}
	caller := NewLlmCaller(provider, cost.Pricing{InputPricePerMillion: 1, OutputPricePerMillion: 2})
	return NewReactLoop(caller, reg, Config{
		Workflow: workflow.Config{Execution: workflow.SlotConfig{Model: "test-model"}},
	})
}

func TestSingleTurnCompletion(t *testing.T) {
	p := &fakeProvider{
		responses: []provider.Response{
			{Content: "the answer is 42", FinishReason: "stop",
				Usage: provider.Usage{PromptTokens: 10, CompletionTokens: 5}},
		},
	}
	loop := newLoop(t, p, nil)

	result, tracker, err := loop.Run(context.Background(), "what is the answer?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.Success {
		t.Error("Success = false, want true")
	}
	if result.Content != "the answer is 42" {
		t.Errorf("Content = %q, want %q", result.Content, "the answer is 42")
	}
	// History: system + user + assistant
	if len(result.Messages) != 3 {
		t.Errorf("len(Messages) = %d, want 3", len(result.Messages))
	}
	// Cost tracker updated
	if tracker.CallCount != 1 {
		t.Errorf("CallCount = %d, want 1", tracker.CallCount)
	}
	if tracker.TotalInputTokens != 10 || tracker.TotalOutputTokens != 5 {
		t.Errorf("tokens = %d in, %d out; want 10/5",
			tracker.TotalInputTokens, tracker.TotalOutputTokens)
	}
}

func TestOneToolCallThenCompletion(t *testing.T) {
	tool := echoTool("read_file")
	p := &fakeProvider{
		responses: []provider.Response{
			// Turn 1: model calls the tool.
			{ToolCalls: []provider.ToolCall{
				{ID: "c1", Name: "read_file", Arguments: json.RawMessage(`{"path":"x"}`)},
			}, FinishReason: "tool_calls"},
			// Turn 2: model gives the final answer.
			{Content: "done", FinishReason: "stop"},
		},
	}
	loop := newLoop(t, p, []tools.Tool{tool})

	result, _, err := loop.Run(context.Background(), "read x")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.Success {
		t.Error("Success = false, want true")
	}
	if tool.calls != 1 {
		t.Errorf("tool.calls = %d, want 1", tool.calls)
	}
	// History: system + user + assistant(tool_call) + tool + assistant(final)
	if len(result.Messages) != 5 {
		t.Errorf("len(Messages) = %d, want 5", len(result.Messages))
	}
	// Verify tool result message is in there
	foundToolResult := false
	for _, m := range result.Messages {
		if m.Role == "tool" && m.ToolCallID == "c1" {
			foundToolResult = true
			if !strings.Contains(m.Content[0].Text, "echo:") {
				t.Errorf("tool result text = %q, want substring %q",
					m.Content[0].Text, "echo:")
			}
		}
	}
	if !foundToolResult {
		t.Error("history missing tool result message")
	}
}

func TestMultipleToolCallsInOneTurn(t *testing.T) {
	t1, t2 := echoTool("read_file"), echoTool("bash")
	p := &fakeProvider{
		responses: []provider.Response{
			{ToolCalls: []provider.ToolCall{
				{ID: "a", Name: "read_file", Arguments: json.RawMessage(`{"path":"x"}`)},
				{ID: "b", Name: "bash", Arguments: json.RawMessage(`{"command":"ls"}`)},
			}, FinishReason: "tool_calls"},
			{Content: "ok", FinishReason: "stop"},
		},
	}
	loop := newLoop(t, p, []tools.Tool{t1, t2})

	_, _, err := loop.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if t1.calls != 1 {
		t.Errorf("read_file calls = %d, want 1", t1.calls)
	}
	if t2.calls != 1 {
		t.Errorf("bash calls = %d, want 1", t2.calls)
	}
}

func TestMaxIterations(t *testing.T) {
	// Provider always returns a tool call → loop runs forever (until cap).
	// Vary the args per turn so the doom-loop detector doesn't fire
	// before the iteration cap does — this test is specifically
	// exercising MaxIterations, not doom-loop detection.
	tool := echoTool("noop")
	responses := make([]provider.Response, 100)
	for i := range responses {
		responses[i] = provider.Response{
			ToolCalls: []provider.ToolCall{
				{ID: "x", Name: "noop", Arguments: json.RawMessage(fmt.Sprintf(`{"i":%d}`, i))},
			},
			FinishReason: "tool_calls",
		}
	}
	p := &fakeProvider{responses: responses}

	reg := tools.NewRegistry()
	if err := reg.Register(tool); err != nil {
		t.Fatalf("register: %v", err)
	}
	caller := NewLlmCaller(p, cost.Pricing{})
	cfg := Config{
		Workflow:      workflow.Config{Execution: workflow.SlotConfig{Model: "test"}},
		MaxIterations: 3,
	}
	loop := NewReactLoop(caller, reg, cfg)

	_, _, err := loop.Run(context.Background(), "go")
	if !errors.Is(err, ErrMaxIterations) {
		t.Errorf("err = %v, want wraps ErrMaxIterations", err)
	}
	if p.calls != 3 {
		t.Errorf("provider calls = %d, want 3", p.calls)
	}
}

func TestProviderErrorExitsLoop(t *testing.T) {
	want := errors.New("api blew up")
	p := &fakeProvider{
		responses: []provider.Response{{}},
		errors:    []error{want},
	}
	loop := newLoop(t, p, nil)

	_, _, err := loop.Run(context.Background(), "go")
	if !errors.Is(err, ErrLLM) {
		t.Errorf("err = %v, want wraps ErrLLM", err)
	}
}

func TestCostTrackingAccumulates(t *testing.T) {
	p := &fakeProvider{
		responses: []provider.Response{
			{ToolCalls: []provider.ToolCall{
				{ID: "1", Name: "noop", Arguments: json.RawMessage(`{}`)},
			}, Usage: provider.Usage{PromptTokens: 100, CompletionTokens: 20}},
			{ToolCalls: []provider.ToolCall{
				{ID: "2", Name: "noop", Arguments: json.RawMessage(`{}`)},
			}, Usage: provider.Usage{PromptTokens: 150, CompletionTokens: 30}},
			{Content: "done",
				Usage: provider.Usage{PromptTokens: 200, CompletionTokens: 10}},
		},
	}
	loop := newLoop(t, p, []tools.Tool{echoTool("noop")})

	_, tracker, err := loop.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tracker.CallCount != 3 {
		t.Errorf("CallCount = %d, want 3", tracker.CallCount)
	}
	if tracker.TotalInputTokens != 450 {
		t.Errorf("TotalInputTokens = %d, want 450", tracker.TotalInputTokens)
	}
	if tracker.TotalOutputTokens != 60 {
		t.Errorf("TotalOutputTokens = %d, want 60", tracker.TotalOutputTokens)
	}
}

func TestToolFailureContinuesLoop(t *testing.T) {
	// Tool returns Success: false; loop should NOT exit — should feed
	// the error result back to the model and continue.
	failingTool := &fakeTool{
		name:   "broken",
		desc:   "always fails",
		schema: json.RawMessage(`{"type":"object"}`),
		exec: func(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
			return tools.ToolResult{Success: false, Error: "kaboom"}, nil
		},
	}
	p := &fakeProvider{
		responses: []provider.Response{
			{ToolCalls: []provider.ToolCall{
				{ID: "1", Name: "broken", Arguments: json.RawMessage(`{}`)},
			}},
			{Content: "ack the failure", FinishReason: "stop"},
		},
	}
	loop := newLoop(t, p, []tools.Tool{failingTool})

	result, _, err := loop.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.Success {
		t.Error("Success = false, want true (loop completed)")
	}
	// The error message should appear in the tool result message.
	foundErr := false
	for _, m := range result.Messages {
		if m.Role == "tool" && strings.Contains(m.Content[0].Text, "[ERROR]") {
			foundErr = true
			if !strings.Contains(m.Content[0].Text, "kaboom") {
				t.Errorf("tool result missing error text: %q", m.Content[0].Text)
			}
		}
	}
	if !foundErr {
		t.Error("tool result message missing [ERROR] prefix")
	}
}

func TestUnknownToolBubblesError(t *testing.T) {
	p := &fakeProvider{
		responses: []provider.Response{
			{ToolCalls: []provider.ToolCall{
				{ID: "1", Name: "ghost", Arguments: json.RawMessage(`{}`)},
			}},
		},
	}
	loop := newLoop(t, p, nil) // no tools registered

	_, _, err := loop.Run(context.Background(), "go")
	if !errors.Is(err, ErrToolExec) {
		t.Errorf("err = %v, want wraps ErrToolExec", err)
	}
}

func TestContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancelled

	p := &fakeProvider{
		responses: []provider.Response{{Content: "shouldn't reach"}},
	}
	loop := newLoop(t, p, nil)

	_, _, err := loop.Run(ctx, "go")
	if !errors.Is(err, ErrInterrupted) {
		t.Errorf("err = %v, want wraps ErrInterrupted", err)
	}
	if p.calls != 0 {
		t.Errorf("provider was called %d times despite pre-cancelled ctx", p.calls)
	}
}

func TestNewReactLoopDefaults(t *testing.T) {
	p := &fakeProvider{responses: []provider.Response{{Content: "ok"}}}
	caller := NewLlmCaller(p, cost.Pricing{})
	reg := tools.NewRegistry()

	loop := NewReactLoop(caller, reg, Config{}) // all zeros
	if loop.Config.MaxIterations != DefaultMaxIterations {
		t.Errorf("MaxIterations = %d, want %d",
			loop.Config.MaxIterations, DefaultMaxIterations)
	}
	if loop.Config.SystemPrompt != DefaultSystemPrompt {
		t.Errorf("SystemPrompt should default to DefaultSystemPrompt")
	}
}

func TestBudgetSnapshotInResult(t *testing.T) {
	// Two-turn run: provider reports 100 tokens turn 1, 250 turn 2.
	// Result.Budget should reflect the LAST reported value (250), not
	// the sum — Reported is meant to be the most recent ground-truth
	// anchor, not a cumulative total (that's TotalInputTokens).
	tool := echoTool("noop")
	p := &fakeProvider{
		responses: []provider.Response{
			{ToolCalls: []provider.ToolCall{
				{ID: "1", Name: "noop", Arguments: json.RawMessage(`{}`)},
			}, Usage: provider.Usage{PromptTokens: 100, CompletionTokens: 20}},
			{Content: "done", FinishReason: "stop",
				Usage: provider.Usage{PromptTokens: 250, CompletionTokens: 10}},
		},
	}
	reg := tools.NewRegistry()
	if err := reg.Register(tool); err != nil {
		t.Fatalf("register: %v", err)
	}
	caller := NewLlmCaller(p, cost.Pricing{})
	loop := NewReactLoop(caller, reg, Config{
		Workflow:         workflow.Config{Execution: workflow.SlotConfig{Model: "test"}},
		MaxContextTokens: 10_000,
	})

	result, _, err := loop.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Budget.Reported != 250 {
		t.Errorf("Budget.Reported = %d, want 250 (last turn's value)",
			result.Budget.Reported)
	}
	if result.Budget.Estimated < 250 {
		t.Errorf("Budget.Estimated = %d, want >= 250 (baseline + local delta)",
			result.Budget.Estimated)
	}
	if result.Budget.UsagePct != 0.025 {
		t.Errorf("Budget.UsagePct = %v, want 0.025 (250/10000)",
			result.Budget.UsagePct)
	}
}

func TestBudgetUntouchedWithoutLLMCall(t *testing.T) {
	// Pre-cancelled ctx → loop exits before any LLM call → Budget
	// is the zero snapshot. Guards against panics on the error path.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	p := &fakeProvider{responses: []provider.Response{{Content: "x"}}}
	loop := newLoop(t, p, nil)

	result, _, err := loop.Run(ctx, "go")
	if !errors.Is(err, ErrInterrupted) {
		t.Fatalf("err = %v, want ErrInterrupted", err)
	}
	if result.Budget.Reported != 0 {
		t.Errorf("Budget.Reported = %d, want 0 (no call made)",
			result.Budget.Reported)
	}
}

func TestDoomLoop_HaltsAfterThirdEscalation(t *testing.T) {
	// Provider hammers the same tool with the same args. Detector
	// fires Redirect on call 3, Notify on call 4, ForceStop on call 5.
	identical := provider.Response{
		ToolCalls: []provider.ToolCall{
			{ID: "x", Name: "noop", Arguments: json.RawMessage(`{"k":"v"}`)},
		},
		FinishReason: "tool_calls",
	}
	responses := make([]provider.Response, 20)
	for i := range responses {
		responses[i] = identical
	}
	p := &fakeProvider{responses: responses}
	loop := newLoop(t, p, []tools.Tool{echoTool("noop")})

	_, _, err := loop.Run(context.Background(), "go")
	if !errors.Is(err, ErrDoomLoop) {
		t.Fatalf("err = %v, want wraps ErrDoomLoop", err)
	}
	if p.calls != 5 {
		t.Errorf("provider calls = %d, want 5", p.calls)
	}
}

func TestDoomLoop_RedirectDoesNotHaltDispatch(t *testing.T) {
	// 3 identical calls → Redirect. The loop should INJECT a warning
	// system message and continue dispatching the tools. The 4th
	// response is a clean Content to let the loop finish cleanly.
	identical := provider.Response{
		ToolCalls: []provider.ToolCall{
			{ID: "x", Name: "noop", Arguments: json.RawMessage(`{"k":"v"}`)},
		},
		FinishReason: "tool_calls",
	}
	p := &fakeProvider{
		responses: []provider.Response{
			identical, identical, identical,
			{Content: "ok", FinishReason: "stop"},
		},
	}
	loop := newLoop(t, p, []tools.Tool{echoTool("noop")})

	result, _, err := loop.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.Success {
		t.Error("Success = false, want true (Redirect should not halt)")
	}
	found := false
	for _, m := range result.Messages {
		if m.Role == "system" && len(m.Content) > 0 &&
			strings.Contains(m.Content[0].Text, "stuck") {
			found = true
			break
		}
	}
	if !found {
		t.Error("history missing doom-loop warning system message")
	}
}

func TestSchemasForReturnsSortedToolList(t *testing.T) {
	reg := tools.NewRegistry()
	for _, name := range []string{"zebra", "apple", "mango"} {
		if err := reg.Register(echoTool(name)); err != nil {
			t.Fatalf("register %s: %v", name, err)
		}
	}
	schemas := SchemasFor(reg)
	if len(schemas) != 3 {
		t.Fatalf("len(schemas) = %d, want 3", len(schemas))
	}
	want := []string{"apple", "mango", "zebra"}
	for i, s := range schemas {
		if s.Name != want[i] {
			t.Errorf("schemas[%d].Name = %q, want %q", i, s.Name, want[i])
		}
	}
}
