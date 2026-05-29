package hooks

import (
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"time"
)

// Manager dispatches one HookEvent firing across every registered
// matcher and merges the outputs into a single FireResult. The
// settings + executor pair is set at construction; both are
// effectively immutable thereafter, so a Manager is safe to share
// across goroutines.
type Manager struct {
	settings HookSettings
	executor *Executor
}

// NewManager wires a Manager with the given settings (typically
// loaded via hooks.Load) and executor.
func NewManager(settings HookSettings, executor *Executor) *Manager {
	return &Manager{settings: settings, executor: executor}
}

// FireResult is the merged outcome of running every matching hook
// for one event firing. Callers in the agent layer (#34/#35) read
// AdditionalContext to prepend to the operation, UpdatedInput to
// replace the operation's input, PermissionDecision to gate, and
// Reason to surface to the user.
//
// HookResults holds per-hook telemetry for logging. It always
// reflects the order matchers were registered (matching hooks only;
// non-matching matchers are silent).
type FireResult struct {
	// AdditionalContext is the concatenation of every hook's
	// non-empty AdditionalContext, joined with "\n\n" in matcher
	// declaration order.
	AdditionalContext string

	// UpdatedInput is the last non-empty UpdatedInput from any
	// hook in the dispatch order. No transform chaining — each
	// hook sees the ORIGINAL payload on stdin, and the last one
	// to produce an UpdatedInput wins outright.
	UpdatedInput json.RawMessage

	// PermissionDecision is the merged verdict:
	//   - Deny short-circuits the loop and overrides any prior
	//     allow/ask.
	//   - Otherwise the first non-empty opinion wins.
	//   - Empty means no hook gated this event.
	PermissionDecision PermissionDecision

	// Reason is paired with the winning PermissionDecision. When
	// no permission opinion was expressed, it falls back to the
	// first non-empty Reason from any hook (so a hook can attach
	// a notice without gating).
	Reason string

	// HookResults captures per-hook telemetry for logging,
	// debugging, and metrics. Each entry corresponds to one
	// matcher that fired. Non-matching matchers don't appear.
	HookResults []HookOutcome
}

// HookOutcome is the per-hook record kept in FireResult.HookResults.
// Mirrors ExecResult fields but always includes the matcher's
// command (so the manager's caller can identify each entry without
// the matcher in hand) and the per-hook error.
type HookOutcome struct {
	// Command is the matcher's shell command, useful for logs.
	Command string

	// ExitCode is the hook's exit status. -1 when the hook didn't
	// run to completion (Err non-nil).
	ExitCode int

	// Duration is wall-clock time for the hook.
	Duration time.Duration

	// Stderr is the hook's captured stderr.
	Stderr string

	// Decision is the parsed stdout decision (zero on
	// non-completion or unparseable output).
	Decision HookDecision

	// Err is the per-hook infrastructure failure: timeout, spawn
	// failure, payload-marshal failure. Per-hook errors DO NOT
	// halt the dispatch loop; they get logged + the loop
	// continues.
	Err error
}

// IsDeny reports whether the merged FireResult is a deny verdict.
// Convenience for the agent layer's PreToolUse / UserPromptSubmit
// integration so the gate branch reads cleanly.
func (r *FireResult) IsDeny() bool {
	return r.PermissionDecision == PermissionDeny
}

// IsAllow reports whether the merged result is an explicit allow.
// Distinct from "no opinion" — an empty PermissionDecision is NOT
// allow.
func (r *FireResult) IsAllow() bool {
	return r.PermissionDecision == PermissionAllow
}

// IsAsk reports whether the merged result escalates to the user.
// v1 has no interactive prompt; the agent layer treats Ask as
// equivalent to Deny until the prompt arrives.
func (r *FireResult) IsAsk() bool {
	return r.PermissionDecision == PermissionAsk
}

// Fire runs every matching hook for event in declaration order,
// passing primaryIdentifier as the regex test target and payload
// as the JSON stdin. Returns the merged FireResult.
//
// Per-hook failures (timeout, spawn error, malformed output) are
// captured in HookOutcome.Err and the loop continues — a
// telemetry hook timing out shouldn't halt the agent.
//
// Fire's error return is reserved for ctx cancellation: if the
// user cancels mid-event, the currently-running hook gets killed
// and remaining hooks are skipped, with ctx.Err() returned so the
// caller can distinguish "hooks finished" from "user cancelled
// during hooks."
//
// The fast path: zero registered matchers → empty FireResult with
// no executor calls. The agent loop fires many events per turn so
// this matters.
func (mgr *Manager) Fire(
	ctx context.Context,
	event HookEvent,
	primaryIdentifier string,
	payload any,
) (*FireResult, error) {
	result := &FireResult{}

	matchers := mgr.settings.MatchersFor(event)
	if len(matchers) == 0 {
		return result, nil
	}

	var (
		firstReasonAnywhere string
		contextParts        []string
	)

	for _, matcher := range matchers {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if !matcher.Matches(primaryIdentifier) {
			continue
		}

		execResult, runErr := mgr.executor.Run(ctx, event, matcher, payload)

		outcome := HookOutcome{
			Command:  matcher.Command,
			ExitCode: -1,
		}
		if execResult != nil {
			outcome.ExitCode = execResult.ExitCode
			outcome.Duration = execResult.Duration
			outcome.Stderr = execResult.Stderr
			outcome.Decision = execResult.Decision
		}
		outcome.Err = runErr
		result.HookResults = append(result.HookResults, outcome)

		if runErr != nil {
			slog.Warn("hooks: dispatch failed; continuing",
				"event", event.String(),
				"command", matcher.Command,
				"error", runErr)
			continue
		}

		decision := outcome.Decision

		// Track first non-empty reason as the fallback when no
		// permission decision is ultimately made. Lets a non-
		// gating hook attach a notice (e.g., "this tool is being
		// audited") that surfaces even without a verdict.
		if firstReasonAnywhere == "" && decision.Reason != "" {
			firstReasonAnywhere = decision.Reason
		}

		if decision.AdditionalContext != "" {
			contextParts = append(contextParts, decision.AdditionalContext)
		}

		// Last-writer-wins for UpdatedInput. No transform chaining
		// because each hook sees the ORIGINAL payload on stdin;
		// re-marshaling between hooks would be more complex than
		// any current use case warrants.
		if len(decision.UpdatedInput) > 0 {
			result.UpdatedInput = decision.UpdatedInput
		}

		// PermissionDecision merge. Deny is special: it
		// short-circuits AND overrides any prior allow/ask. Other
		// values follow first-wins.
		if decision.PermissionDecision == PermissionDeny {
			result.PermissionDecision = PermissionDeny
			result.Reason = decision.Reason
			break
		}
		if decision.PermissionDecision != "" && result.PermissionDecision == "" {
			result.PermissionDecision = decision.PermissionDecision
			result.Reason = decision.Reason
		}
	}

	result.AdditionalContext = strings.Join(contextParts, "\n\n")

	// Fallback reason when no permission opinion landed: first
	// non-empty reason from anywhere in the dispatch.
	if result.PermissionDecision == "" && result.Reason == "" {
		result.Reason = firstReasonAnywhere
	}

	return result, nil
}
