package hooks

import "encoding/json"

// Payload types — one per HookEvent — describing the JSON shape
// that gets piped to a hook's stdin. The executor (executor.go)
// takes `any` for the payload argument so the manager (#33) can
// pick the right type per event; defining the typed structs here
// documents the contract end-to-end so a future hook author can
// `cat /dev/stdin | jq .tool` knowing which fields exist for which
// event.
//
// All structs have JSON tags using snake_case keys to match
// settings.json conventions (cache_control, tool_use_id, etc.).

// SessionStartPayload is the stdin payload for HookEventSessionStart.
// Fires once at REPL/TUI launch.
type SessionStartPayload struct {
	SessionID  string `json:"session_id"`
	WorkingDir string `json:"working_dir"`
}

// UserPromptSubmitPayload is the stdin payload for
// HookEventUserPromptSubmit. Fires after each user prompt, before
// the LLM sees it.
type UserPromptSubmitPayload struct {
	Prompt string `json:"prompt"`
}

// PreToolUsePayload is the stdin payload for HookEventPreToolUse.
// Fires before each tool dispatch. Args is the tool's raw argument
// JSON (passed through verbatim so hooks see the same shape the
// tool will).
type PreToolUsePayload struct {
	Tool string          `json:"tool"`
	Args json.RawMessage `json:"args,omitempty"`
}

// PostToolUsePayload is the stdin payload for HookEventPostToolUse.
// Fires after a successful tool dispatch. Output is the tool's text
// result; Success mirrors the ToolResult.Success flag.
type PostToolUsePayload struct {
	Tool    string `json:"tool"`
	Output  string `json:"output,omitempty"`
	Success bool   `json:"success"`
}

// PostToolUseFailurePayload is the stdin payload for
// HookEventPostToolUseFailure. Fires when a tool dispatch returns
// an infrastructure error (not a Success: false domain failure —
// those go through PostToolUse).
type PostToolUseFailurePayload struct {
	Tool  string `json:"tool"`
	Error string `json:"error"`
}

// SubagentStartPayload is the stdin payload for
// HookEventSubagentStart. Fires before spawn_subagent dispatches a
// child loop. AgentType is the SubAgentSpec name (e.g. "Explore",
// "Planner"); Task is the user-facing description.
type SubagentStartPayload struct {
	AgentType string `json:"agent_type"`
	Task      string `json:"task"`
}

// SubagentStopPayload is the stdin payload for HookEventSubagentStop.
// Fires when a spawned subagent's loop returns. Result is the
// subagent's final Content; CostUSD is what the subagent spent.
type SubagentStopPayload struct {
	AgentType string  `json:"agent_type"`
	Result    string  `json:"result,omitempty"`
	CostUSD   float64 `json:"cost_usd,omitempty"`
}

// StopPayload is the stdin payload for HookEventStop. Fires when a
// turn completes via any path. Error is empty on success.
type StopPayload struct {
	Result string `json:"result,omitempty"`
	Error  string `json:"error,omitempty"`
}

// PreCompactPayload is the stdin payload for HookEventPreCompact.
// Fires before context compaction kicks in. Level is the
// optimization level string ("warning", "mask", "prune",
// "aggressive_mask", "full_compact"); MessageCount is the history
// length at the time of decision.
type PreCompactPayload struct {
	Level        string `json:"level"`
	MessageCount int    `json:"message_count"`
}

// SessionEndPayload is the stdin payload for HookEventSessionEnd.
// Fires once at REPL/TUI exit. CostUSD is the session's cumulative
// cost; CallCount is the cumulative API calls.
type SessionEndPayload struct {
	SessionID string  `json:"session_id"`
	CostUSD   float64 `json:"cost_usd"`
	CallCount int64   `json:"call_count"`
}
