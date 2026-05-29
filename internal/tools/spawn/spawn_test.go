package spawn

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/ashish-work/opendev-go/internal/agents"
	"github.com/ashish-work/opendev-go/internal/agents/subagents"
	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/hooks"
	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/runtime/permissions"
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

func TestBuildChildLoop_ExploreSpecExcludesSpawnFromChildRegistry(t *testing.T) {
	// The Explore spec's Tools whitelist is {read_file, list_files,
	// bash} — spawn_subagent is NOT in that list. After #40's
	// filter integration, the Explore subagent's child loop should
	// see a registry that contains only the whitelisted tools and
	// excludes spawn_subagent. This is the load-bearing security
	// property: an Explore subagent cannot recursively spawn.
	p := &stubProvider{}
	tool, _, registry := newTestTool(t, p)

	// Register the spec's tools onto the same registry so the
	// filter has something to keep.
	for _, name := range []string{"read_file", "list_files", "bash"} {
		registerEcho(t, registry, name)
	}

	spec, _ := subagents.SpecByName("Explore")
	childLoop := tool.buildChildLoop(spec)

	childRegistry := childLoop.Registry
	if _, ok := childRegistry.Get("spawn_subagent"); ok {
		t.Errorf("child registry should NOT include spawn_subagent for Explore spec")
	}
	for _, want := range []string{"read_file", "list_files", "bash"} {
		if _, ok := childRegistry.Get(want); !ok {
			t.Errorf("child registry should include %q for Explore spec", want)
		}
	}
}

func TestBuildChildLoop_BuildSpecKeepsFullRegistry(t *testing.T) {
	// Build's nil Tools means "no restriction" — the child sees
	// the full registry including spawn_subagent (which enables
	// nested spawning chains used by the grandchild test in #38).
	p := &stubProvider{}
	tool, _, registry := newTestTool(t, p)
	for _, name := range []string{"read_file", "bash"} {
		registerEcho(t, registry, name)
	}

	spec, _ := subagents.SpecByName("Build")
	childLoop := tool.buildChildLoop(spec)

	if _, ok := childLoop.Registry.Get("spawn_subagent"); !ok {
		t.Errorf("Build spec child should retain spawn_subagent")
	}
	if _, ok := childLoop.Registry.Get("read_file"); !ok {
		t.Errorf("Build spec child should retain read_file")
	}
}

func TestBuildChildLoop_PlannerSpecForwardReferenceDropsPresentPlan(t *testing.T) {
	// Planner's Tools list includes present_plan, but that tool
	// doesn't exist as a registered Tool in v2. The filter should
	// silently drop the unknown name; the remaining tools survive.
	p := &stubProvider{}
	tool, _, registry := newTestTool(t, p)
	for _, name := range []string{"read_file", "list_files", "bash"} {
		registerEcho(t, registry, name)
	}

	spec, _ := subagents.SpecByName("Planner")
	childLoop := tool.buildChildLoop(spec)

	if _, ok := childLoop.Registry.Get("present_plan"); ok {
		t.Errorf("forward-referenced present_plan should be filtered out")
	}
	for _, want := range []string{"read_file", "list_files", "bash"} {
		if _, ok := childLoop.Registry.Get(want); !ok {
			t.Errorf("Planner spec should include %q", want)
		}
	}
}

// registerEcho is a tiny helper that registers a successful no-op
// tool under the given name. Used by the spec-filter tests above
// to populate the registry with the named tools each spec expects.
func registerEcho(t *testing.T, registry *tools.Registry, name string) {
	t.Helper()
	echo := &noOpTool{name: name}
	if err := registry.Register(echo); err != nil {
		t.Fatalf("register %s: %v", name, err)
	}
}

// noOpTool returns a successful empty ToolResult. Used only for
// the spec-filter tests; the actual tool behavior doesn't matter.
type noOpTool struct{ name string }

func (n *noOpTool) Name() string            { return n.name }
func (n *noOpTool) Description() string     { return "no-op for filter tests" }
func (n *noOpTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (n *noOpTool) Execute(_ context.Context, _ tools.ToolContext, _ json.RawMessage) (tools.ToolResult, error) {
	return tools.ToolResult{Output: "ok", Success: true}, nil
}

// makeHookManagerForSpawn loads a hooks.Manager from a settings.json
// written to a temp dir with the given event → command mapping.
// Returns the manager so spawn-tool tests can wire it onto Config.
func makeHookManagerForSpawn(t *testing.T, eventCommands map[string]string) *hooks.Manager {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "settings.json")
	// Build the JSON manually so we don't depend on a specific
	// marshaler ordering.
	hooksMap := map[string]any{}
	for event, command := range eventCommands {
		hooksMap[event] = []map[string]any{{"command": command}}
	}
	body, _ := json.Marshal(map[string]any{"hooks": hooksMap})
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	settings, err := hooks.LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	return hooks.NewManager(settings, hooks.NewExecutor(""))
}

// newSpawnToolWithHooks wires a spawn.Tool with hooks attached, on
// top of the same stubProvider machinery the other tests use.
func newSpawnToolWithHooks(t *testing.T, p *stubProvider, mgr *hooks.Manager) (*Tool, *tools.Registry) {
	t.Helper()
	caller := agents.NewLlmCaller(p, cost.Pricing{
		InputPricePerMillion:  1.0,
		OutputPricePerMillion: 1.0,
	})
	registry := tools.NewRegistry()
	tool := New(Config{
		Caller:     caller,
		Registry:   registry,
		Workflow:   workflow.Config{Execution: workflow.SlotConfig{Model: "stub"}},
		WorkingDir: "/tmp",
		MaxCtx:     128_000,
		Hooks:      mgr,
	})
	if err := registry.Register(tool); err != nil {
		t.Fatalf("register: %v", err)
	}
	return tool, registry
}

func TestSubagentStart_FiresWithCorrectPayload(t *testing.T) {
	auditPath := filepath.Join(t.TempDir(), "audit.log")
	// Hook command echoes the payload it sees on stdin to the
	// audit file. We can then assert the agent_type and task
	// fields landed.
	cmd := `cat > ` + auditPath
	mgr := makeHookManagerForSpawn(t, map[string]string{
		"subagent_start": cmd,
	})
	p := &stubProvider{
		responses: []provider.Response{{Content: "done", FinishReason: "stop"}},
	}
	tool, _ := newSpawnToolWithHooks(t, p, mgr)

	got := invokeTool(t, tool, context.Background(), Args{
		AgentType: "Explore",
		Task:      "investigate",
	})
	if !got.Success {
		t.Fatalf("expected Success; got Output=%q", got.Output)
	}
	body, err := os.ReadFile(auditPath)
	if err != nil {
		t.Fatalf("read audit: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("audit not valid JSON: %v\n%s", err, body)
	}
	if got := payload["agent_type"]; got != "Explore" {
		t.Errorf("agent_type = %v, want Explore", got)
	}
	if got := payload["task"]; got != "investigate" {
		t.Errorf("task = %v, want 'investigate'", got)
	}
}

func TestSubagentStart_DenyRefusesSpawn(t *testing.T) {
	// Hook returns a deny decision. The spawn should NOT run the
	// child loop; instead it should return an observation-level
	// "blocked by hook" ToolResult.
	mgr := makeHookManagerForSpawn(t, map[string]string{
		"subagent_start": `echo '{"permissionDecision":"deny","reason":"Build subagents disabled here"}'`,
	})
	// The provider would error if called — so any Run call would
	// blow up the test. We use a deliberately empty responses slice
	// (default fall-through still returns "ok"), but the test's
	// real proof is the audit-log check below.
	p := &stubProvider{}
	tool, _ := newSpawnToolWithHooks(t, p, mgr)

	got := invokeTool(t, tool, context.Background(), Args{
		AgentType: "Build",
		Task:      "do something",
	})
	if got.Success {
		t.Errorf("expected Success=false for denied spawn")
	}
	if !strings.Contains(got.Output, "blocked by hook") {
		t.Errorf("Output should mention 'blocked by hook'; got %q", got.Output)
	}
	if !strings.Contains(got.Output, "Build subagents disabled here") {
		t.Errorf("Output should include the hook's reason; got %q", got.Output)
	}
	// The stub provider's Call should NOT have been invoked
	// because the child loop never ran.
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.requests) != 0 {
		t.Errorf("child loop should not have run; provider received %d requests",
			len(p.requests))
	}
}

func TestSubagentStart_DenySuppressesStop(t *testing.T) {
	// Two audit files: one written by Start, one by Stop. After a
	// denied Start, only the Start file should have content (the
	// deny decision itself doesn't get logged — that hook's stdout
	// is consumed for the deny verdict, not for audit).
	auditDir := t.TempDir()
	stopPath := filepath.Join(auditDir, "stop.log")
	mgr := makeHookManagerForSpawn(t, map[string]string{
		"subagent_start": `echo '{"permissionDecision":"deny","reason":"no"}'`,
		"subagent_stop":  `cat > ` + stopPath,
	})
	tool, _ := newSpawnToolWithHooks(t, &stubProvider{}, mgr)

	_ = invokeTool(t, tool, context.Background(), Args{
		AgentType: "Build",
		Task:      "x",
	})

	if data, err := os.ReadFile(stopPath); err == nil && len(data) > 0 {
		t.Errorf("SubagentStop should not have fired after a deny; got %s", data)
	}
}

func TestSubagentStop_FiresAfterSuccessWithCost(t *testing.T) {
	auditPath := filepath.Join(t.TempDir(), "stop.log")
	cmd := `cat > ` + auditPath
	mgr := makeHookManagerForSpawn(t, map[string]string{
		"subagent_stop": cmd,
	})
	p := &stubProvider{
		responses: []provider.Response{{
			Content:      "subagent done",
			FinishReason: "stop",
			Usage:        provider.Usage{PromptTokens: 100, CompletionTokens: 50},
		}},
	}
	tool, _ := newSpawnToolWithHooks(t, p, mgr)

	_ = invokeTool(t, tool, context.Background(), Args{
		AgentType: "Explore",
		Task:      "find x",
	})

	body, err := os.ReadFile(auditPath)
	if err != nil {
		t.Fatalf("read audit: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("audit not valid JSON: %v\n%s", err, body)
	}
	if got := payload["agent_type"]; got != "Explore" {
		t.Errorf("agent_type = %v, want Explore", got)
	}
	if got := payload["result"]; got != "subagent done" {
		t.Errorf("result = %v, want 'subagent done'", got)
	}
	// CostUSD should be > 0 because we configured non-zero
	// pricing and the response carried tokens.
	cost, _ := payload["cost_usd"].(float64)
	if cost <= 0 {
		t.Errorf("cost_usd = %v, want > 0 (pricing was set; tokens flowed)", payload["cost_usd"])
	}
}

func TestSubagentStop_FiresAfterErrorWithPartialResult(t *testing.T) {
	auditPath := filepath.Join(t.TempDir(), "stop.log")
	mgr := makeHookManagerForSpawn(t, map[string]string{
		"subagent_stop": `cat > ` + auditPath,
	})
	// Provider always errors — child loop returns ErrLLM with
	// empty result content. SubagentStop should still fire.
	p := &errorProvider{err: errors.New("provider blew up")}
	caller := agents.NewLlmCaller(p, cost.Pricing{})
	registry := tools.NewRegistry()
	tool := New(Config{
		Caller:     caller,
		Registry:   registry,
		Workflow:   workflow.Config{Execution: workflow.SlotConfig{Model: "stub"}},
		WorkingDir: "/tmp",
		MaxCtx:     128_000,
		Hooks:      mgr,
	})
	_ = registry.Register(tool)

	got := invokeTool(t, tool, context.Background(), Args{
		AgentType: "Explore",
		Task:      "x",
	})
	if got.Success {
		t.Errorf("expected Success=false when child errored")
	}

	body, err := os.ReadFile(auditPath)
	if err != nil {
		t.Fatalf("read audit: %v (SubagentStop should fire even on error)", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("audit not valid JSON: %v\n%s", err, body)
	}
	if got := payload["agent_type"]; got != "Explore" {
		t.Errorf("agent_type = %v, want Explore", got)
	}
}

func TestSubagentStart_NilManagerNoOp(t *testing.T) {
	// Spawn with no Hooks manager → both Start and Stop are no-ops
	// (regression: existing tests pass without hooks wired).
	p := &stubProvider{responses: []provider.Response{{Content: "ok"}}}
	tool, _, _ := newTestTool(t, p)

	got := invokeTool(t, tool, context.Background(), Args{
		AgentType: "Explore",
		Task:      "x",
	})
	if !got.Success {
		t.Errorf("nil-hooks spawn should still succeed; got %q", got.Output)
	}
}

func TestSubagentStart_MatcherTargetsAgentType(t *testing.T) {
	// Two start hooks with different matchers. Only the Explore
	// one should fire when spawning an Explore subagent.
	auditDir := t.TempDir()
	exploreFile := filepath.Join(auditDir, "explore.log")
	buildFile := filepath.Join(auditDir, "build.log")
	// Manually craft the settings to use specific matchers.
	settingsPath := filepath.Join(auditDir, "settings.json")
	body := `{"hooks":{"subagent_start":[
		{"matcher":"^Explore$","command":"cat > ` + exploreFile + `"},
		{"matcher":"^Build$","command":"cat > ` + buildFile + `"}
	]}}`
	if err := os.WriteFile(settingsPath, []byte(body), 0o644); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	settings, err := hooks.LoadFile(settingsPath)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	mgr := hooks.NewManager(settings, hooks.NewExecutor(""))

	p := &stubProvider{responses: []provider.Response{{Content: "ok"}}}
	tool, _ := newSpawnToolWithHooks(t, p, mgr)

	_ = invokeTool(t, tool, context.Background(), Args{
		AgentType: "Explore",
		Task:      "x",
	})

	if _, err := os.Stat(exploreFile); err != nil {
		t.Errorf("Explore-matcher hook should have fired; file missing: %v", err)
	}
	if _, err := os.Stat(buildFile); err == nil {
		t.Errorf("Build-matcher hook should NOT have fired for an Explore spawn")
	}
}

func TestSubagentStart_UpdatedInputIgnoredIn_v2(t *testing.T) {
	// A hook returning updatedInput would conceptually rewrite the
	// child's task. v2 deliberately ignores this; the child sees
	// the original task. We verify by checking the request the
	// stub provider received.
	mgr := makeHookManagerForSpawn(t, map[string]string{
		"subagent_start": `echo '{"updatedInput":{"task":"rewritten task"}}'`,
	})
	p := &stubProvider{
		responses: []provider.Response{{Content: "ok"}},
	}
	tool, _ := newSpawnToolWithHooks(t, p, mgr)

	_ = invokeTool(t, tool, context.Background(), Args{
		AgentType: "Explore",
		Task:      "ORIGINAL task",
	})

	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.requests) == 0 {
		t.Fatal("provider never called")
	}
	// The child's first user message should contain "ORIGINAL task",
	// not the rewritten one.
	req := p.requests[0]
	found := false
	for _, m := range req.Messages {
		for _, b := range m.Content {
			if strings.Contains(b.Text, "ORIGINAL task") {
				found = true
			}
			if strings.Contains(b.Text, "rewritten task") {
				t.Errorf("rewritten task leaked into child's request: %q", b.Text)
			}
		}
	}
	if !found {
		t.Errorf("child request didn't include the original task")
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

func TestBuildChildLoop_InheritsParentPermissions(t *testing.T) {
	// Subagents must run under the same policy the parent runs
	// under — a user's deny pattern applies uniformly across the
	// whole call graph. buildChildLoop is the seam where the
	// policy must flow from spawn.Config to the child *ReactLoop.
	//
	// We construct a Tool with a non-zero Policy, build a child
	// loop, and assert the child carries the same Policy. No LLM
	// call needed; this is a wiring assertion.
	settingsPath := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(settingsPath,
		[]byte(`{"permissions":{"bash":{"deny_patterns":["rm -rf"]}}}`),
		0o644); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	parentPolicy, err := permissions.LoadFile(settingsPath)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if len(parentPolicy.Tools) == 0 {
		t.Fatal("parentPolicy is unexpectedly empty; test setup wrong")
	}

	caller := agents.NewLlmCaller(&stubProvider{}, cost.Pricing{
		InputPricePerMillion:  1.0,
		OutputPricePerMillion: 1.0,
	})
	registry := tools.NewRegistry()
	tool := New(Config{
		Caller:      caller,
		Registry:    registry,
		Workflow:    workflow.Config{Execution: workflow.SlotConfig{Model: "stub"}},
		WorkingDir:  "/tmp",
		MaxCtx:      128_000,
		Permissions: parentPolicy,
	})

	childLoop := tool.buildChildLoop(subagents.SubAgentSpec{
		Name:          "Explore",
		SystemPrompt:  "explore",
		MaxIterations: 5,
	})

	// The child must observe the same policy. We exercise it
	// through the public Check method rather than comparing
	// struct fields by value (regexp pointers won't match by
	// reflect.DeepEqual after a load round-trip, but a Check
	// call proves the deny pattern is live in the child).
	deny := childLoop.Permissions.Check("bash",
		`{"command":"rm -rf /tmp"}`)
	if deny.Allowed {
		t.Fatal("child loop should inherit parent's bash deny pattern, but Check returned Allow")
	}
	if !strings.Contains(deny.Reason, "rm -rf") {
		t.Errorf("expected child's deny reason to quote 'rm -rf', got %q",
			deny.Reason)
	}

	// And a non-matching call still passes through.
	allow := childLoop.Permissions.Check("bash",
		`{"command":"ls -la"}`)
	if !allow.Allowed {
		t.Errorf("non-matching bash call should Allow in child loop, got Deny(%q)",
			allow.Reason)
	}
}

func TestBuildChildLoop_ZeroPermissionsAllowsAllInChild(t *testing.T) {
	// Regression: when the parent has no policy (zero value), the
	// child must also have no policy. Confirms the field truly
	// passes through rather than getting silently defaulted.
	caller := agents.NewLlmCaller(&stubProvider{}, cost.Pricing{
		InputPricePerMillion:  1.0,
		OutputPricePerMillion: 1.0,
	})
	registry := tools.NewRegistry()
	tool := New(Config{
		Caller:     caller,
		Registry:   registry,
		Workflow:   workflow.Config{Execution: workflow.SlotConfig{Model: "stub"}},
		WorkingDir: "/tmp",
		MaxCtx:     128_000,
		// Permissions intentionally omitted — zero value.
	})

	childLoop := tool.buildChildLoop(subagents.SubAgentSpec{
		Name:          "Build",
		SystemPrompt:  "build",
		MaxIterations: 5,
	})

	if got := len(childLoop.Permissions.Tools); got != 0 {
		t.Fatalf("childLoop.Permissions.Tools len = %d, want 0 (zero policy)", got)
	}
	if d := childLoop.Permissions.Check("bash", `{}`); !d.Allowed {
		t.Errorf("zero Policy should Allow any tool, got Deny(%q)", d.Reason)
	}
}
