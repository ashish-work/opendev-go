// Package hooks defines the user-facing lifecycle hook system. A
// hook is a shell command the user configures in settings.json under
// one of the 10 HookEvent names; when the corresponding event fires
// inside the agent, the command runs with the event payload on
// stdin and produces a HookDecision on stdout.
//
// This package is the type-only foundation:
//
//   - event.go defines HookEvent (which lifecycle points exist).
//   - decision.go defines HookDecision (what hooks send back).
//
// Subsequent commits in the Phase 6 arc add the JSON config loader
// (#31), the shell executor (#32), the manager that dispatches
// events to matching hooks (#33), and integration into the agent
// loop and binaries (#34-36).
package hooks

// HookEvent names the lifecycle point at which a hook fires. Iota-
// based int constants follow the same tagged-struct pattern as
// LoopActionKind, StreamEventKind, and ContentKind: cheap to
// compare, exhaustive switches feel natural, easy to add a String
// method for logs and config-file keys.
//
// The 10 events span the full agent lifecycle:
//
//   - SessionStart / SessionEnd bracket a REPL or TUI run.
//   - UserPromptSubmit fires after each user prompt, before the LLM.
//   - PreToolUse / PostToolUse / PostToolUseFailure ring each tool
//     dispatch.
//   - SubagentStart / SubagentStop ring each spawn_subagent
//     dispatch (Phase 7 introduces those).
//   - Stop fires when a turn completes (any path).
//   - PreCompact fires before context compaction (Phase 5).
type HookEvent int

const (
	// HookEventSessionStart fires once when the REPL/TUI launches,
	// before the first user prompt. A hook here can inject
	// project-specific context (CLAUDE.md, git state) or set up
	// session telemetry.
	HookEventSessionStart HookEvent = iota

	// HookEventUserPromptSubmit fires after the user hits enter,
	// before the LLM sees the message. A hook can prepend
	// additional context via HookDecision.AdditionalContext or
	// rewrite the prompt via UpdatedInput.
	HookEventUserPromptSubmit

	// HookEventPreToolUse fires before each tool dispatch. A hook
	// can gate execution via HookDecision.PermissionDecision
	// (allow/deny/ask), rewrite arguments via UpdatedInput, or
	// inject context via AdditionalContext.
	HookEventPreToolUse

	// HookEventPostToolUse fires after a tool dispatch returns
	// successfully. Useful for audit trails and downstream side
	// effects. HookDecision is consumed for AdditionalContext only.
	HookEventPostToolUse

	// HookEventPostToolUseFailure fires after a tool dispatch
	// returns an error. Same hook output handling as PostToolUse;
	// separation lets users wire alerting or retry triggers
	// differently from happy-path observability.
	HookEventPostToolUseFailure

	// HookEventSubagentStart fires before spawn_subagent dispatches
	// a child loop. Useful for subagent depth tracking and
	// observability. (Phase 7 introduces spawn_subagent itself.)
	HookEventSubagentStart

	// HookEventSubagentStop fires when a spawned subagent's loop
	// returns. Carries the subagent's result and cost rollup in
	// the payload.
	HookEventSubagentStop

	// HookEventStop fires when a turn completes via any path
	// (success, error, interruption, max-iter). Useful for
	// per-turn metrics and success/failure telemetry.
	HookEventStop

	// HookEventPreCompact fires before context compaction kicks in
	// (Phase 5). Lets users persist state before a lossy operation
	// or log compaction events.
	HookEventPreCompact

	// HookEventSessionEnd fires once when the REPL/TUI exits.
	// Useful for session cleanup, total-cost reporting, and log
	// shipping.
	HookEventSessionEnd
)

// String returns the stable snake_case identifier for the event.
// These strings double as settings.json keys under the "hooks"
// section, so a hook entry looks like:
//
//	{ "hooks": { "pre_tool_use": [ { "command": "..." } ] } }
//
// Snake_case is the JSON convention used elsewhere in the codebase
// (cache_control, tool_use_id, prompt_tokens_details), so settings
// files read consistently with the rest of the project.
//
// Unknown values return "unknown" — the exhaustive-switch sentinel
// test enforces that every declared constant has an arm here.
func (e HookEvent) String() string {
	switch e {
	case HookEventSessionStart:
		return "session_start"
	case HookEventUserPromptSubmit:
		return "user_prompt_submit"
	case HookEventPreToolUse:
		return "pre_tool_use"
	case HookEventPostToolUse:
		return "post_tool_use"
	case HookEventPostToolUseFailure:
		return "post_tool_use_failure"
	case HookEventSubagentStart:
		return "subagent_start"
	case HookEventSubagentStop:
		return "subagent_stop"
	case HookEventStop:
		return "stop"
	case HookEventPreCompact:
		return "pre_compact"
	case HookEventSessionEnd:
		return "session_end"
	default:
		return "unknown"
	}
}

// Parse converts a snake_case identifier back into a HookEvent. The
// JSON config loader in #31 uses this when reading settings.json:
// for each event-named key under "hooks", it looks up the
// corresponding HookEvent value. Unknown identifiers return (0,
// false) so the loader can skip them with a warning.
//
// Bijection invariant: Parse(e.String()) == (e, true) for every
// declared HookEvent. A round-trip test pins it.
func Parse(s string) (HookEvent, bool) {
	switch s {
	case "session_start":
		return HookEventSessionStart, true
	case "user_prompt_submit":
		return HookEventUserPromptSubmit, true
	case "pre_tool_use":
		return HookEventPreToolUse, true
	case "post_tool_use":
		return HookEventPostToolUse, true
	case "post_tool_use_failure":
		return HookEventPostToolUseFailure, true
	case "subagent_start":
		return HookEventSubagentStart, true
	case "subagent_stop":
		return HookEventSubagentStop, true
	case "stop":
		return HookEventStop, true
	case "pre_compact":
		return HookEventPreCompact, true
	case "session_end":
		return HookEventSessionEnd, true
	default:
		return 0, false
	}
}
