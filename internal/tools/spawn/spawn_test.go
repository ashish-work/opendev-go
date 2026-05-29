package spawn

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/ashish-work/opendev-go/internal/agents"
	"github.com/ashish-work/opendev-go/internal/agents/subagents"
	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/workflow"
)

// stubProvider implements provider.Provider with scripted responses.
// Returns each entry of responses in order; falls back to a no-tool
// "ok" Content reply when scripts are exhausted (so the child loop
// terminates rather than looping).
type stubProvider struct {
	mu        sync.Mutex
	responses []provider.Response
	idx       int
	requests  []provider.Request
}

func (s *stubProvider) Name() string { return "stub" }

func (s *stubProvider) Call(_ context.Context, req provider.Request) (provider.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, req)
	if s.idx >= len(s.responses) {
		return provider.Response{Content: "subagent ok"}, nil
	}
	resp := s.responses[s.idx]
	s.idx++
	return resp, nil
}

func (s *stubProvider) Stream(_ context.Context, _ provider.Request) (<-chan provider.StreamEvent, error) {
	return nil, errors.New("stubProvider: Stream not used in spawn tests")
}

// newTestTool wires a spawn.Tool with the given stub provider plus
// the given extra tools registered alongside spawn (caller appends
// before spawn registration). Returns the tool, the provider for
// inspection, and the registry.
func newTestTool(t *testing.T, p *stubProvider, extras ...tools.Tool) (*Tool, *stubProvider, *tools.Registry) {
	t.Helper()
	caller := agents.NewLlmCaller(p, cost.Pricing{})
	registry := tools.NewRegistry()
	for _, x := range extras {
		if err := registry.Register(x); err != nil {
			t.Fatalf("register extra tool: %v", err)
		}
	}
	tool := New(Config{
		Caller:     caller,
		Registry:   registry,
		Workflow:   workflow.Config{Execution: workflow.SlotConfig{Model: "stub"}},
		WorkingDir: "/tmp",
		MaxCtx:     128_000,
	})
	if err := registry.Register(tool); err != nil {
		t.Fatalf("register spawn: %v", err)
	}
	return tool, p, registry
}

// invokeTool is a thin Execute wrapper for tests.
func invokeTool(t *testing.T, tool *Tool, ctx context.Context, args any) tools.ToolResult {
	t.Helper()
	raw, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	result, err := tool.Execute(ctx, tools.ToolContext{WorkingDir: "/tmp"}, raw)
	if err != nil {
		t.Fatalf("Execute returned infrastructure error: %v", err)
	}
	return result
}

func TestName(t *testing.T) {
	tool, _, _ := newTestTool(t, &stubProvider{})
	if tool.Name() != ToolName {
		t.Errorf("Name = %q, want %q", tool.Name(), ToolName)
	}
}

func TestDescription_NonEmpty(t *testing.T) {
	tool, _, _ := newTestTool(t, &stubProvider{})
	if tool.Description() == "" {
		t.Errorf("Description should not be empty")
	}
}

func TestSchema_EnumMatchesBuiltins(t *testing.T) {
	tool, _, _ := newTestTool(t, &stubProvider{})
	var parsed map[string]any
	if err := json.Unmarshal(tool.Schema(), &parsed); err != nil {
		t.Fatalf("Schema not valid JSON: %v", err)
	}
	props := parsed["properties"].(map[string]any)
	agentType := props["agent_type"].(map[string]any)
	rawEnum := agentType["enum"].([]any)
	got := make([]string, 0, len(rawEnum))
	for _, v := range rawEnum {
		got = append(got, v.(string))
	}
	want := subagents.BuiltinNames()
	if !reflect.DeepEqual(got, want) {
		t.Errorf("enum = %v, want %v", got, want)
	}
}

func TestSchema_RequiredFields(t *testing.T) {
	tool, _, _ := newTestTool(t, &stubProvider{})
	var parsed map[string]any
	_ = json.Unmarshal(tool.Schema(), &parsed)
	rawReq, _ := parsed["required"].([]any)
	got := make([]string, 0, len(rawReq))
	for _, v := range rawReq {
		got = append(got, v.(string))
	}
	want := []string{"agent_type", "task"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("required = %v, want %v", got, want)
	}
}

func TestExecute_UnknownAgentType(t *testing.T) {
	tool, _, _ := newTestTool(t, &stubProvider{})
	got := invokeTool(t, tool, context.Background(), Args{
		AgentType: "Nonexistent",
		Task:      "anything",
	})
	if got.Success {
		t.Errorf("expected Success=false")
	}
	if !strings.Contains(got.Output, "unknown agent_type") {
		t.Errorf("Output = %q, want to mention 'unknown agent_type'", got.Output)
	}
}

func TestExecute_EmptyTask(t *testing.T) {
	tool, _, _ := newTestTool(t, &stubProvider{})
	got := invokeTool(t, tool, context.Background(), Args{
		AgentType: "Explore",
		Task:      "",
	})
	if got.Success {
		t.Errorf("expected Success=false for empty task")
	}
	if !strings.Contains(got.Output, "task is required") {
		t.Errorf("Output = %q, want to mention 'task is required'", got.Output)
	}
}

func TestExecute_InvalidJSONArgs(t *testing.T) {
	tool, _, _ := newTestTool(t, &stubProvider{})
	result, err := tool.Execute(context.Background(),
		tools.ToolContext{WorkingDir: "/tmp"},
		json.RawMessage("not json"))
	if err != nil {
		t.Errorf("invalid args should not return infrastructure error; got %v", err)
	}
	if result.Success {
		t.Errorf("expected Success=false on invalid args")
	}
	if !strings.Contains(result.Output, "invalid args") {
		t.Errorf("Output = %q, want to mention 'invalid args'", result.Output)
	}
}

func TestExecute_DepthAtLimitRefuses(t *testing.T) {
	tool, _, _ := newTestTool(t, &stubProvider{})
	// Pretend we're already MaxDepth deep — the next call should be
	// refused.
	ctx := ContextWithDepth(context.Background(), DefaultMaxDepth)
	got := invokeTool(t, tool, ctx, Args{
		AgentType: "Explore",
		Task:      "investigate",
	})
	if got.Success {
		t.Errorf("expected Success=false at depth limit")
	}
	if !strings.Contains(got.Output, "depth limit exceeded") {
		t.Errorf("Output = %q, want to mention 'depth limit exceeded'", got.Output)
	}
}

func TestExecute_DepthOneBelowLimitSucceeds(t *testing.T) {
	tool, _, _ := newTestTool(t, &stubProvider{})
	// At MaxDepth-1, the call should still go through.
	ctx := ContextWithDepth(context.Background(), DefaultMaxDepth-1)
	got := invokeTool(t, tool, ctx, Args{
		AgentType: "Explore",
		Task:      "investigate",
	})
	if !got.Success {
		t.Errorf("expected Success=true at depth limit-1; got Output=%q", got.Output)
	}
}

func TestExecute_SuccessfulSpawnReturnsChildContent(t *testing.T) {
	p := &stubProvider{
		responses: []provider.Response{
			{Content: "I checked. The answer is 42.", FinishReason: "stop"},
		},
	}
	tool, _, _ := newTestTool(t, p)
	got := invokeTool(t, tool, context.Background(), Args{
		AgentType: "Explore",
		Task:      "find the answer",
	})
	if !got.Success {
		t.Fatalf("expected Success=true; got Output=%q", got.Output)
	}
	if got.Output != "I checked. The answer is 42." {
		t.Errorf("Output = %q, want exact child content", got.Output)
	}
}

func TestExecute_ChildLoopErrorIsObservationLevel(t *testing.T) {
	// Provider error → child loop returns ErrLLM. Tool should
	// surface as Success=false but NOT propagate as infra error.
	p := &errorProvider{err: errors.New("provider blew up")}
	caller := agents.NewLlmCaller(p, cost.Pricing{})
	registry := tools.NewRegistry()
	tool := New(Config{
		Caller:     caller,
		Registry:   registry,
		Workflow:   workflow.Config{Execution: workflow.SlotConfig{Model: "stub"}},
		WorkingDir: "/tmp",
		MaxCtx:     128_000,
	})
	_ = registry.Register(tool)

	got := invokeTool(t, tool, context.Background(), Args{
		AgentType: "Explore",
		Task:      "go",
	})
	if got.Success {
		t.Errorf("expected Success=false when child loop errored")
	}
	if !strings.Contains(got.Output, "error") {
		t.Errorf("Output = %q, want to mention error", got.Output)
	}
}

func TestExecute_ChildSeesIncrementedDepth(t *testing.T) {
	// Register a capture tool that records the ctx depth when
	// dispatched. The stub provider's first response is a tool_call
	// to capture; second response (auto-generated when scripts
	// exhaust) is a no-tool reply that terminates the child loop.
	var capturedDepth int
	captureCalled := false
	captureTool := &capturingTool{
		fn: func(ctx context.Context) {
			capturedDepth = DepthFromContext(ctx)
			captureCalled = true
		},
	}

	p := &stubProvider{
		responses: []provider.Response{
			{
				ToolCalls: []provider.ToolCall{{
					ID:        "c1",
					Name:      "capture",
					Arguments: json.RawMessage(`{}`),
				}},
			},
			{Content: "done"},
		},
	}
	tool, _, _ := newTestTool(t, p, captureTool)

	got := invokeTool(t, tool, context.Background(), Args{
		AgentType: "Build", // Build has nil Tools so capture passes through
		Task:      "trigger capture",
	})
	if !got.Success {
		t.Fatalf("expected Success; got Output=%q", got.Output)
	}
	if !captureCalled {
		t.Fatal("capture tool was not called")
	}
	if capturedDepth != 1 {
		t.Errorf("child ctx depth = %d, want 1 (top-level → child = depth 1)",
			capturedDepth)
	}
}

func TestExecute_ModelOverrideAppliesToChildRequest(t *testing.T) {
	// When a subagent spec has ModelOverride set, the child loop
	// should issue its request against that model. We can verify by
	// inspecting the recorded request on the stub provider.
	//
	// Built-in specs all have empty ModelOverride for v2, so build
	// a Tool with a synthetic spec via direct buildChildLoop access.
	// We use the public Execute path with a custom spec injected by
	// monkey-patching Builtins for the duration of the test.
	p := &stubProvider{}
	tool, _, _ := newTestTool(t, p)

	originalSpec := subagents.Builtins["Explore"]
	t.Cleanup(func() {
		subagents.Builtins["Explore"] = originalSpec
	})
	override := originalSpec
	override.ModelOverride = "claude-haiku-4-5"
	subagents.Builtins["Explore"] = override

	got := invokeTool(t, tool, context.Background(), Args{
		AgentType: "Explore",
		Task:      "test",
	})
	if !got.Success {
		t.Fatalf("expected Success; got %q", got.Output)
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.requests) == 0 {
		t.Fatalf("no requests captured")
	}
	if got := p.requests[0].Model; got != "claude-haiku-4-5" {
		t.Errorf("child request Model = %q, want override 'claude-haiku-4-5'", got)
	}
}

func TestExecute_TopLevelStartsAtDepthZero(t *testing.T) {
	// A top-level call (no depth in ctx) should treat depth as 0,
	// so the child runs at depth 1.
	if got := DepthFromContext(context.Background()); got != 0 {
		t.Errorf("default depth = %d, want 0", got)
	}
}

func TestExecute_NestedSpawnsCapAtMaxDepth(t *testing.T) {
	// Three levels deep should succeed; the fourth should be
	// refused. Simulate by chained ContextWithDepth calls.
	tool, _, _ := newTestTool(t, &stubProvider{})

	for depth := 0; depth < DefaultMaxDepth; depth++ {
		ctx := ContextWithDepth(context.Background(), depth)
		got := invokeTool(t, tool, ctx, Args{
			AgentType: "Explore",
			Task:      "go",
		})
		if !got.Success {
			t.Errorf("depth=%d should succeed; got Output=%q", depth, got.Output)
		}
	}
	// At MaxDepth, refused.
	ctxAt := ContextWithDepth(context.Background(), DefaultMaxDepth)
	got := invokeTool(t, tool, ctxAt, Args{
		AgentType: "Explore",
		Task:      "go",
	})
	if got.Success {
		t.Errorf("depth=%d should be refused", DefaultMaxDepth)
	}
}

func TestNew_ZeroMaxDepthFallsBackToDefault(t *testing.T) {
	tool := New(Config{
		Caller:   agents.NewLlmCaller(&stubProvider{}, cost.Pricing{}),
		Registry: tools.NewRegistry(),
		Workflow: workflow.Config{Execution: workflow.SlotConfig{Model: "stub"}},
		// MaxDepth omitted → should default to DefaultMaxDepth.
	})
	if tool.cfg.MaxDepth != DefaultMaxDepth {
		t.Errorf("MaxDepth = %d, want default %d",
			tool.cfg.MaxDepth, DefaultMaxDepth)
	}
}

func TestDepthFromContext_ZeroForFreshCtx(t *testing.T) {
	if got := DepthFromContext(context.Background()); got != 0 {
		t.Errorf("fresh ctx depth = %d, want 0", got)
	}
}

func TestContextWithDepth_RoundTrip(t *testing.T) {
	ctx := ContextWithDepth(context.Background(), 7)
	if got := DepthFromContext(ctx); got != 7 {
		t.Errorf("round-trip depth = %d, want 7", got)
	}
}

// --- test helpers ---

// errorProvider always returns the configured error on Call.
type errorProvider struct {
	err error
}

func (e *errorProvider) Name() string { return "error-stub" }
func (e *errorProvider) Call(_ context.Context, _ provider.Request) (provider.Response, error) {
	return provider.Response{}, e.err
}
func (e *errorProvider) Stream(_ context.Context, _ provider.Request) (<-chan provider.StreamEvent, error) {
	return nil, e.err
}

// capturingTool records the dispatch ctx so a test can inspect what
// the child loop's tool dispatch sees. Returns Success=true with a
// fixed Output so the child loop continues.
type capturingTool struct {
	fn func(ctx context.Context)
}

func (c *capturingTool) Name() string { return "capture" }
func (c *capturingTool) Description() string {
	return "test-only tool that records the ctx depth"
}
func (c *capturingTool) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object"}`)
}
func (c *capturingTool) Execute(ctx context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
	c.fn(ctx)
	return tools.ToolResult{Output: "captured", Success: true}, nil
}
