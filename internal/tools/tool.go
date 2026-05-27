// Package tools defines the Tool abstraction and the registry used by the
// agent loop to dispatch tool calls from the model. Concrete tools
// (read_file, bash, edit_file) live in subpackages and implement the Tool
// interface defined here.
//
// The Tool interface is intentionally small (4 methods): the four things
// the loop actually needs to introduce a tool to the model, dispatch
// against it, and handle its result. Anything optional — Category for
// policy decisions, future Streaming for long-running tools — lives on
// secondary interfaces that the registry type-asserts for. This is the
// idiomatic Go pattern: small core interface, optional behavior added
// through additional interfaces (cf. io.Reader vs io.ReadCloser).
package tools

import (
	"context"
	"encoding/json"
	"errors"
)

// Tool is the contract every tool implements. The ReAct loop sees only
// this interface: it asks each registered tool for its Name, Description,
// and Schema (to build the LLM request) and calls Execute when the model
// chooses to invoke the tool.
//
// Kept to 4 methods — slightly over the Go "1-3 methods" guideline, but
// each one is irreducible for the loop to function. Anything optional
// (Categorized, future Streaming, future Cancellable) lives on secondary
// interfaces the registry detects via type assertion.
//
// Tools are stateless implementations: the same Tool value can dispatch
// many concurrent calls. Per-call state lives in args + ToolContext;
// shared state (e.g. an edit_file per-file mutex map) is the tool's
// own concern, kept thread-safe internally.
type Tool interface {
	// Name is the unique tool identifier the model echoes back in
	// ToolCall.Name. Must be stable — changing it breaks in-flight
	// sessions and trained-in tool habits.
	Name() string

	// Description is the human-readable hint shown to the model.
	// Quality matters: poor descriptions degrade tool selection.
	Description() string

	// Schema returns the JSON Schema for the tool's parameters, raw so
	// each tool can ship its own without this package needing per-tool
	// types. Surfaced to the model via ToolSchema.Parameters.
	Schema() json.RawMessage

	// Execute runs the tool with the model's chosen args. Honors ctx
	// for cancellation and timeouts. Returns (ToolResult, error): the
	// error path is reserved for infrastructure failures (registry
	// invariants, ctx cancellation); tool-domain failures (file not
	// found, bash exit code) come back as ToolResult{Success: false,
	// Error: "..."} so they're forwarded to the model as observations.
	Execute(ctx context.Context, tctx ToolContext, args json.RawMessage) (ToolResult, error)
}

// ToolResult is the outcome of a tool execution. Success disambiguates
// "Output set" from "Error set" — there is no need for nullable string
// types because the Success flag tells the consumer which field to read.
type ToolResult struct {
	// Success indicates whether the tool achieved its goal. Maps to
	// "did this produce useful output for the model" — a bash command
	// returning a non-zero exit code is still Success: true if the
	// stderr is the useful observation.
	Success bool

	// Output — the tool's primary observation, shown to the model and
	// (usually) the user. Empty when Success is false.
	Output string

	// Error — failure message shown to the model so it can recover.
	// Empty when Success is true.
	Error string

	// Metadata — tool-specific extras (e.g. file size, exit code, diff
	// stats). Untyped map because each tool owns its own keys; the loop
	// just forwards the map verbatim and the consuming tool (or the
	// REPL status line) reads the keys it knows about.
	Metadata map[string]any

	// DurationMS — wall-clock execution time, populated by the registry.
	// Used by cost tracking and the doom-loop detector's recency
	// signals; tools should leave it zero.
	DurationMS int64

	// LLMSuffix — text appended to the result for the model but hidden
	// from the user. Used to silently nudge recovery (e.g. "Did you
	// mean to use the absolute path?"). Set by the tool or middleware.
	LLMSuffix string
}

// ToolContext carries per-call session state into Execute. Kept minimal
// for v1 — just the working directory for path resolution.
//
// Deferred fields (each will land with its feature, not preemptively):
//   - SessionID (multi-session support)
//   - IsSubagent (subagents with restricted tool scope)
//   - CancelToken (user-initiated interrupt)
//   - DiagnosticProvider (LSP feedback after edits)
//   - SharedState (cross-iteration tool communication)
//   - TimeoutConfig (per-tool timeout overrides)
type ToolContext struct {
	// WorkingDir is the absolute path tools should resolve relative
	// arguments against. The CLI sets this once at startup; tools must
	// not chdir.
	WorkingDir string
}

// Category groups tools for policy and permission decisions (e.g.
// "block all CategoryWrite tools in Plan Mode" — a future feature).
//
// Same tagged-int pattern as provider.ContentKind. CategoryOther is the
// zero value so a tool that forgets to declare a category gets a
// reasonable default.
type Category int

const (
	CategoryOther      Category = iota // default — uncategorized
	CategoryRead                       // file reading, search, listing (Read, Glob, Grep)
	CategoryWrite                      // file editing, writing (Edit, Write)
	CategoryProcess                    // shell/process execution (Bash)
	CategoryWeb                        // network ops (WebFetch, WebSearch)
	CategorySession                    // session management, subagents
	CategoryMemory                     // memory read/write
	CategoryMeta                       // planning, todos, task management
	CategoryMessaging                  // inter-agent messaging
	CategoryAutomation                 // scheduling, cron
	CategorySymbol                     // LSP, AST symbol ops
	CategoryMCP                        // MCP bridge tools
)

// Categorized is an optional interface tools may implement to expose
// their Category. The registry type-asserts for it where needed (policy,
// Plan Mode gating). Tools that don't implement it are treated as
// CategoryOther.
type Categorized interface {
	Category() Category
}

// Tool-domain error sentinels. Wrap with fmt.Errorf("%w: ...") so callers
// can match with errors.Is. Reserved for infrastructure-level failures
// (registry, args validation, permissions, interrupts); tool execution
// failures travel through ToolResult.Error instead.
var (
	ErrToolNotFound     = errors.New("tool not found")
	ErrInvalidParams    = errors.New("invalid parameters")
	ErrPermissionDenied = errors.New("permission denied")
	ErrInterrupted      = errors.New("interrupted by user")
)
