// Package agents holds the ReAct loop, the LLM call orchestrator, prompt
// composition, and the turn-level types they produce. The loop is the
// heart of the agent: it asks the model what to do, dispatches tool
// calls, feeds results back, and iterates until the model completes or
// the loop exits abnormally.
//
// The package is organized as one file per concern:
//
//   - turn.go    — value types (TurnKind, TurnResult, Result) and
//                  sentinel errors the loop produces.
//   - caller.go  — LlmCaller, the thin wrapper that pairs a Provider
//                  with cost tracking so each LLM call updates the
//                  per-Run Tracker.
//   - prompt.go  — system/user/tool message constructors and the
//                  long stable system prompt.
//   - loop.go    — ReactLoop itself: the bounded request → tool
//                  dispatch → continue cycle.
package agents

import (
	"errors"
	"fmt"

	"github.com/ashish-work/opendev-go/internal/budget"
	"github.com/ashish-work/opendev-go/internal/provider"
)

// TurnKind tags the variant of a TurnResult.
//
// Third tagged-struct of the project (after provider.ContentKind and
// tools.Category). The pattern: int Kind plus a union of fields,
// where only the variant's relevant fields are meaningful.
type TurnKind int

const (
	// TurnContinue — the model wants another LLM round without tool
	// calls (rare in practice; usually subsumed by ToolCall or Complete).
	TurnContinue TurnKind = iota

	// TurnToolCall — the model emitted one or more tool calls; the loop
	// dispatches them and feeds results back. Valid field: ToolCalls.
	TurnToolCall

	// TurnComplete — the model produced a final answer; loop exits
	// successfully. Valid fields: Content, Status.
	TurnComplete

	// TurnMaxIterations — the loop hit its iteration cap; exits with
	// ErrMaxIterations to the caller.
	TurnMaxIterations

	// TurnInterrupted — user cancelled (via ctx); exits with
	// ErrInterrupted to the caller.
	TurnInterrupted
)

// TurnResult is the outcome of one iteration of the ReAct loop. The
// loop's main switch reads Kind and acts on the variant-specific fields.
//
// The loop body runs inline and uses native `continue` and early
// `return` for control flow, so no separate "LoopAction" envelope type
// is needed. If a future refactor splits the loop into discrete phase
// functions, a LoopAction wrapper can be added then to bubble
// continue/return from phase to orchestrator.
type TurnResult struct {
	// Kind selects which other fields are meaningful.
	Kind TurnKind

	// ToolCalls — valid when Kind == TurnToolCall. The loop dispatches
	// each through the tool registry in turn order.
	ToolCalls []provider.ToolCall

	// Content — valid when Kind == TurnComplete. The model's final
	// answer text, surfaced as Result.Content to the caller.
	Content string

	// Status — valid when Kind == TurnComplete. Optional completion
	// status from a task_complete tool (e.g. "success", "failed").
	// Empty when unset.
	Status string
}

// Agent-loop error sentinels. Wrap with fmt.Errorf("...: %w", Err...)
// so callers can match with errors.Is. These are separate from the
// tool-layer sentinels in package tools — different abstraction levels,
// disambiguated by package qualification (agents.ErrInterrupted vs
// tools.ErrInterrupted).
//
// APIError (below) is a typed error rather than a sentinel because the
// variant carries fields (status + message). Use errors.As to extract.
var (
	// ErrLLM — wraps any failure originating from the provider call
	// (timeout, transport, parse). Use APIError when the provider
	// returned a structured HTTP error instead.
	ErrLLM = errors.New("LLM call failed")

	// ErrToolExec — wraps a tool dispatch failure that's not just a
	// tool returning ToolResult{Success: false}. Reserved for
	// infrastructure-level tool failures (registry, args validation).
	ErrToolExec = errors.New("tool execution failed")

	// ErrConfig — wraps misconfiguration discovered at runtime
	// (missing API key, malformed model name, etc.).
	ErrConfig = errors.New("configuration error")

	// ErrMaxIterations — the loop exited because it hit its cap.
	// Callers should wrap with the actual limit: fmt.Errorf("%w (limit=%d)", ErrMaxIterations, n).
	ErrMaxIterations = errors.New("max iterations reached")

	// ErrInterrupted — the loop exited because ctx was cancelled
	// (user pressed Ctrl-C, timeout fired, etc.).
	ErrInterrupted = errors.New("interrupted by user")

	// ErrDoomLoop — the doomloop detector escalated to ForceStop
	// because the model called the same tool(s) in a tight cycle.
	// Callers should NOT auto-retry; the user needs to rephrase.
	ErrDoomLoop = errors.New("doom loop detected")
)

// APIError carries a structured HTTP error from the provider — status
// code plus message. Typed struct (not a sentinel) because the variant
// holds data; sentinels can't.
//
// Use errors.As to extract:
//
//	var apiErr *APIError
//	if errors.As(err, &apiErr) {
//	    log.Printf("provider returned %d: %s", apiErr.Status, apiErr.Message)
//	}
type APIError struct {
	Status  int
	Message string
}

// Error implements the error interface.
func (e *APIError) Error() string {
	return fmt.Sprintf("API error %d: %s", e.Status, e.Message)
}

// Result is the final outcome of a complete ReAct loop run, returned
// to the caller (typically the REPL in cmd/opendev). Named Result
// rather than "AgentResult" to avoid the stutter agents.AgentResult.
//
// Optional/future fields like Backgrounded and PartialResult are wired
// in but left empty in v1 — they belong to later features (background
// mode, interrupt-mid-execution recovery).
type Result struct {
	// Content — the model's final reply text. Empty if the loop
	// exited via error (MaxIterations, Interrupted).
	Content string

	// Success — true when the loop completed via TurnComplete with
	// no error. False on MaxIterations, Interrupted, or any error path.
	Success bool

	// Interrupted — true when the loop exited via TurnInterrupted.
	// Carried separately from Success because callers may want to
	// distinguish "user cancelled" from "ran out of iterations".
	Interrupted bool

	// Backgrounded — placeholder for the future background-mode
	// feature (the model yielding control while still working).
	// Always false in v1.
	Backgrounded bool

	// CompletionStatus — optional status from a task_complete tool
	// call (e.g. "success", "failed"). Empty if not set.
	CompletionStatus string

	// Messages — the full conversation history after the run, suitable
	// for persisting to session storage or replaying. Includes system
	// prompt, user input, all assistant turns, and all tool results.
	Messages []provider.Message

	// Budget — the context-window fill picture at end-of-run. Reported
	// is the last value the provider returned; Estimated is the local
	// extrapolation including any messages added after that report.
	// Both are zero before the first LLM call completes.
	Budget budget.Snapshot

	// PartialResult — deferred. Will land with the interrupt-mid-
	// execution recovery feature so partial work can be preserved
	// when the user Ctrl-Cs mid-tool. Until then, callers ignore.
}
