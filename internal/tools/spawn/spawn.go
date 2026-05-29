// Package spawn implements the spawn_subagent tool — the model-
// facing handle for the Phase 7 subagent system.
//
// When the model calls spawn_subagent with {agent_type, task}, the
// tool looks up the SubAgentSpec from internal/agents/subagents,
// builds a fresh child ReactLoop using the parent's Caller +
// Registry + Hooks but with the spec's role-specific
// SystemPrompt, MaxIterations, and optional ModelOverride, and
// runs it against the supplied task. The child's final Content
// becomes the tool's Output so the parent agent can incorporate
// the result into its own reasoning.
//
// Depth tracking uses a context.Context value so nested spawn
// calls (subagent → spawn → grandchild subagent) can be capped at
// a configurable max depth (default 3) without threading state
// through ToolContext.
package spawn

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/ashish-work/opendev-go/internal/agents"
	"github.com/ashish-work/opendev-go/internal/agents/subagents"
	"github.com/ashish-work/opendev-go/internal/hooks"
	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/workflow"
)

// ToolName is the canonical name the model uses to invoke this
// tool. Stable string so trained-in habits survive refactors.
const ToolName = "spawn_subagent"

// DefaultMaxDepth caps how many levels of nested subagent spawns
// are allowed before the tool refuses. Three is generous — a
// well-designed prompt rarely needs more than two levels. The
// cap prevents pathological recursion (a subagent that spawns
// itself indefinitely) from burning the budget.
const DefaultMaxDepth = 3

// spawnDepthKey is the unexported type used as the context.Context
// value key for tracking spawn depth. Unexported type means only
// this package can read/write the depth, even though it lives on
// the shared ctx.
type spawnDepthKey struct{}

// Config bundles the dependencies the spawn tool needs at
// construction time. Built once in main; passed by value to New
// which caches a *Tool. Caller / Registry / Hooks are shared with
// the parent loop so subagents inherit the same provider client
// and tool surface.
type Config struct {
	// Caller is the shared LLM caller — same provider client the
	// parent loop uses. Subagents charge against the same API key.
	Caller *agents.LlmCaller

	// Registry is the parent's tool registry. Subagents see the
	// same tools by default. spec.Tools is declared but unenforced
	// until #40 lands Registry.Filter; a TODO at buildChildLoop
	// flags the spot.
	Registry *tools.Registry

	// Workflow is the parent's slot configuration. The child loop
	// uses this verbatim, except spec.ModelOverride (when non-empty)
	// replaces Workflow.Execution.Model for the subagent.
	Workflow workflow.Config

	// WorkingDir is the project directory. Subagents inherit it via
	// ToolContext.WorkingDir.
	WorkingDir string

	// MaxCtx is the context-window cap passed to the child loop's
	// budget calibrator.
	MaxCtx int

	// Hooks is the optional lifecycle hook manager. Shared with the
	// parent. SubagentStart / SubagentStop wiring lands in #41.
	Hooks *hooks.Manager

	// MaxDepth overrides DefaultMaxDepth. Zero falls back to the
	// constant. Exposed so binaries (or tests) can tighten the cap
	// without code changes.
	MaxDepth int
}

// Tool implements tools.Tool. Stateless after construction; safe
// to share across goroutines (the dependency pointers it holds
// are themselves safe-for-concurrent-use).
type Tool struct {
	cfg    Config
	schema json.RawMessage
}

// New builds a spawn tool with the given dependencies. Schema is
// computed once here (using the current snapshot of
// subagents.BuiltinNames) and cached on the struct, so the
// tools.ToolSchema the model sees in every request is
// deterministic.
func New(cfg Config) *Tool {
	if cfg.MaxDepth <= 0 {
		cfg.MaxDepth = DefaultMaxDepth
	}
	return &Tool{
		cfg:    cfg,
		schema: buildSchema(),
	}
}

// Name implements tools.Tool. Stable string per ToolName.
func (t *Tool) Name() string { return ToolName }

// Description implements tools.Tool. Short and role-revealing:
// the model uses this to decide WHEN to spawn vs handle a task
// itself.
func (t *Tool) Description() string {
	return "Delegate a focused sub-task to a specialized subagent. " +
		"Use Explore for read-only investigation, Planner to propose " +
		"implementation steps, or Build for full execution with all " +
		"tools. The subagent runs a separate conversation and returns " +
		"its final summary. Prefer spawning when the sub-task has a " +
		"clear scope that would otherwise pollute the main thread."
}

// Schema implements tools.Tool. JSON Schema with an enum
// constrained to the registered built-in spec names so the model
// can't pick a non-existent agent_type.
func (t *Tool) Schema() json.RawMessage { return t.schema }

// Args is the parsed shape of the model's call arguments. JSON
// tags match Schema's property names.
type Args struct {
	AgentType string `json:"agent_type"`
	Task      string `json:"task"`
}

// Execute implements tools.Tool. The dispatch path:
//
//   1. Parse args.
//   2. Validate: known agent_type, non-empty task.
//   3. Check depth from ctx; refuse if at the cap.
//   4. Look up spec via subagents.SpecByName.
//   5. Build a child ReactLoop with the spec's role.
//   6. Run the child with task as the user message.
//   7. Wrap final Content as ToolResult.Output.
//
// Errors are observation-level — Success: false with a descriptive
// Output — so the model can react (try a different agent, simplify
// the task) without the parent loop bailing with ErrToolExec.
//
// Cost from the child loop is logged via slog but NOT aggregated
// into the parent's cost.Tracker yet. The parent's UI undercounts
// subagent spend; SubagentStop hook (#41) carries the cost in its
// payload so an external aggregator can compensate. Known
// limitation; revisit when there's real demand for accurate
// in-UI totals.
func (t *Tool) Execute(ctx context.Context, _ tools.ToolContext, raw json.RawMessage) (tools.ToolResult, error) {
	var args Args
	if err := json.Unmarshal(raw, &args); err != nil {
		return tools.ToolResult{
			Output:  fmt.Sprintf("invalid args: %v", err),
			Success: false,
			Error:   err.Error(),
		}, nil
	}
	if args.Task == "" {
		return tools.ToolResult{
			Output:  "task is required",
			Success: false,
			Error:   "missing task",
		}, nil
	}
	spec, ok := subagents.SpecByName(args.AgentType)
	if !ok {
		return tools.ToolResult{
			Output: fmt.Sprintf(
				"unknown agent_type: %q (valid: %v)",
				args.AgentType, subagents.BuiltinNames(),
			),
			Success: false,
			Error:   "unknown agent_type",
		}, nil
	}

	depth := DepthFromContext(ctx)
	if depth >= t.cfg.MaxDepth {
		return tools.ToolResult{
			Output:  fmt.Sprintf("subagent depth limit exceeded (max=%d)", t.cfg.MaxDepth),
			Success: false,
			Error:   "depth limit",
		}, nil
	}

	// SubagentStart hook fires BEFORE we build the child loop or
	// increment depth — the firing context is "a subagent is about
	// to start under me." A deny refuses the spawn at the
	// observation level (Success: false, no child loop runs, no
	// SubagentStop). UpdatedInput and AdditionalContext are
	// silently ignored — their semantics need a design pass
	// deferred to v3.
	startResult, startErr := t.fireSubagentStart(ctx, spec.Name, args.Task)
	if startErr != nil {
		return tools.ToolResult{
			Output: fmt.Sprintf("subagent %q hook error: %v", spec.Name, startErr),
			Success: false,
			Error:   startErr.Error(),
		}, nil
	}
	if startResult != nil && startResult.IsDeny() {
		return tools.ToolResult{
			Output: fmt.Sprintf("subagent %q blocked by hook: %s",
				spec.Name, startResult.Reason),
			Success: false,
			Error:   "permission denied",
		}, nil
	}

	childLoop := t.buildChildLoop(spec)
	childCtx := ContextWithDepth(ctx, depth+1)

	result, tracker, err := childLoop.Run(childCtx, args.Task)

	// SubagentStop fires regardless of success or error — operators
	// want to audit failed subagents as much as successful ones.
	// Fire-and-forget: hook errors are logged inside fireSubagentStop
	// but never propagate. Permission decisions are ignored — the
	// subagent already ran.
	t.fireSubagentStop(ctx, spec.Name, result.Content, tracker.TotalCostUSD)

	if err != nil {
		// Surface error via slog so operators have stderr-grep
		// access without needing a separate hook config.
		slog.Warn("subagent errored",
			"agent_type", spec.Name,
			"iterations", tracker.CallCount,
			"cost_usd", tracker.TotalCostUSD,
			"depth", depth+1,
			"error", err,
		)
		// Preserve partial Content so the parent agent can see
		// whatever the subagent managed to produce before the
		// failure.
		output := fmt.Sprintf("subagent %q error: %v", spec.Name, err)
		if result.Content != "" {
			output += "\n\nPartial output:\n" + result.Content
		}
		return tools.ToolResult{
			Output:  output,
			Success: false,
			Error:   err.Error(),
		}, nil
	}

	// Log the child's totals so operators have visibility into
	// subagent spend even though the parent's UI doesn't reflect
	// it yet.
	slog.Info("subagent completed",
		"agent_type", spec.Name,
		"iterations", tracker.CallCount,
		"input_tokens", tracker.TotalInputTokens,
		"output_tokens", tracker.TotalOutputTokens,
		"cost_usd", tracker.TotalCostUSD,
		"depth", depth+1,
	)

	return tools.ToolResult{
		Output:  result.Content,
		Success: true,
	}, nil
}

// fireSubagentStart fires the SubagentStart hook with agentType +
// task. Returns the merged FireResult so the caller can check
// IsDeny(). When the Hooks manager is nil, returns (nil, nil) —
// the caller treats that as "no opinion" and proceeds.
//
// agentType is the primary identifier (the matcher target) so a
// hook config can scope to specific specs with patterns like
// "^Build$" or "^(Explore|Planner)$".
//
// UpdatedInput and AdditionalContext on the FireResult are
// silently ignored. Their semantics for a subagent (rewrite the
// task? insert into the child's system prompt?) need a design
// pass deferred to v3. The fields are documented as ignored in
// the package doc comment.
func (t *Tool) fireSubagentStart(ctx context.Context, agentType, task string) (*hooks.FireResult, error) {
	if t.cfg.Hooks == nil {
		return nil, nil
	}
	return t.cfg.Hooks.Fire(ctx, hooks.HookEventSubagentStart, agentType,
		hooks.SubagentStartPayload{
			AgentType: agentType,
			Task:      task,
		})
}

// fireSubagentStop fires the SubagentStop hook with the child's
// result content + cost. Fire-and-forget: errors are logged but
// not returned (the subagent has already completed; there's
// nothing to gate). nil Hooks manager → no-op.
//
// cost_usd is the load-bearing field for this hook — it carries
// the child's tracker.TotalCostUSD so external audit hooks can
// aggregate per-subagent spend that the parent's UI doesn't
// reflect. Addresses the cost-tracking gap flagged in #38.
//
// On error paths the result field is set to whatever partial
// Content the child produced before failing. The error itself is
// NOT in the payload (SubagentStopPayload from #32 has no error
// field and changing it would break hooks already targeting that
// shape). Operators wanting error detail use the slog.Warn emitted
// alongside.
func (t *Tool) fireSubagentStop(ctx context.Context, agentType, result string, costUSD float64) {
	if t.cfg.Hooks == nil {
		return
	}
	_, err := t.cfg.Hooks.Fire(ctx, hooks.HookEventSubagentStop, agentType,
		hooks.SubagentStopPayload{
			AgentType: agentType,
			Result:    result,
			CostUSD:   costUSD,
		})
	if err != nil {
		slog.Warn("SubagentStop hook errored",
			"agent_type", agentType,
			"error", err,
		)
	}
}

// buildChildLoop assembles a ReactLoop configured for the spec's
// role. Caller and Hooks pass through from the parent; the registry
// is scoped to the spec via Registry.Filter so the subagent only
// sees the tools its spec advertises. Config takes the spec's
// SystemPrompt verbatim, the spec's MaxIterations as the cap, and
// overrides the execution model when spec.ModelOverride is
// non-empty.
//
// spec.Tools == nil → full registry (Build's case).
// spec.Tools == [list] → only those tools (Explore / Planner case).
// Unknown tool names in spec.Tools are silently dropped by Filter,
// so a forward-reference like Planner's present_plan is harmless.
func (t *Tool) buildChildLoop(spec subagents.SubAgentSpec) *agents.ReactLoop {
	wf := t.cfg.Workflow
	if spec.ModelOverride != "" {
		wf.Execution.Model = spec.ModelOverride
	}

	scopedRegistry := t.cfg.Registry.Filter(spec.Tools)

	loop := agents.NewReactLoop(t.cfg.Caller, scopedRegistry, agents.Config{
		Workflow:         wf,
		MaxIterations:    spec.MaxIterations,
		SystemPrompt:     spec.SystemPrompt,
		WorkingDir:       t.cfg.WorkingDir,
		MaxContextTokens: t.cfg.MaxCtx,
	})
	loop.Hooks = t.cfg.Hooks
	return loop
}

// DepthFromContext reads the current spawn depth from ctx. Returns
// 0 when no depth has been set (top-level call from the main
// agent). Exported so tests can verify depth propagation; the
// agent loop itself never needs to read it.
func DepthFromContext(ctx context.Context) int {
	if d, ok := ctx.Value(spawnDepthKey{}).(int); ok {
		return d
	}
	return 0
}

// ContextWithDepth returns a child ctx carrying the given spawn
// depth. Exported for the same symmetric-API reason as
// DepthFromContext.
func ContextWithDepth(parent context.Context, depth int) context.Context {
	return context.WithValue(parent, spawnDepthKey{}, depth)
}

// buildSchema constructs the JSON Schema once at tool
// construction. agent_type enum is sourced from
// subagents.BuiltinNames so the schema stays in lockstep with
// the registered specs.
func buildSchema() json.RawMessage {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"agent_type": map[string]any{
				"type":        "string",
				"enum":        subagents.BuiltinNames(),
				"description": "Which subagent role to spawn.",
			},
			"task": map[string]any{
				"type":        "string",
				"description": "The instruction the subagent should carry out. Be specific — the subagent only sees this task and its own system prompt, not the parent's conversation history.",
			},
		},
		"required": []string{"agent_type", "task"},
	}
	b, _ := json.Marshal(schema)
	return b
}

// Compile-time check that *Tool satisfies tools.Tool. If the
// interface evolves, this line fails to build.
var _ tools.Tool = (*Tool)(nil)
