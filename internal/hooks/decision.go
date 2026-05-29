package hooks

import "encoding/json"

// PermissionDecision is the verdict a gating hook (notably
// PreToolUse) returns to allow, block, or escalate a tool call.
// Modeled as a typed string rather than an iota-int so JSON
// serialization round-trips naturally without a custom marshaler —
// the hook process emits "allow" / "deny" / "ask" on stdout and the
// manager (#33) consumes the string directly.
//
// The zero value (empty string) means "no decision" — the hook
// produced no opinion on permissions. The agent layer's default
// policy applies.
type PermissionDecision string

const (
	// PermissionAllow lets the operation proceed. Used by gating
	// hooks (notably PreToolUse) to explicitly approve.
	PermissionAllow PermissionDecision = "allow"

	// PermissionDeny blocks the operation. The manager surfaces
	// HookDecision.Reason to the model as the rejection note so
	// the agent can react.
	PermissionDeny PermissionDecision = "deny"

	// PermissionAsk escalates the decision to the user via an
	// interactive prompt. Not exercised in v1 (no interactive
	// prompt yet); included so the wire vocabulary is complete and
	// future commits can wire it without re-defining values.
	PermissionAsk PermissionDecision = "ask"
)

// HookDecision is the universal response envelope every hook emits
// on stdout. Hook commands print one JSON object matching this
// shape; the manager parses and acts on the populated fields.
//
// Every field is JSON-omitempty so a hook that only wants to
// influence one aspect (e.g. just AdditionalContext) sends a
// minimal envelope. An empty envelope ({}) is valid and means
// "the hook ran but expressed no opinion."
type HookDecision struct {
	// AdditionalContext, when non-empty, is prepended to whatever
	// the operation was about to do. For UserPromptSubmit, it's
	// prepended to the user message; for PreToolUse, it's added as
	// a synthetic observation the LLM sees before deciding what to
	// do next.
	AdditionalContext string `json:"additionalContext,omitempty"`

	// UpdatedInput, when non-empty, replaces the operation's
	// input. For PreToolUse, it overrides the tool's arguments
	// (allowing parameter rewrites). Carried as raw JSON so the
	// shape stays event-dependent — the manager hands it to the
	// downstream consumer without re-parsing.
	UpdatedInput json.RawMessage `json:"updatedInput,omitempty"`

	// PermissionDecision is the allow/deny/ask verdict for gating
	// hooks. Empty for non-gating hooks. The agent layer treats
	// Deny as a hard stop on the current operation.
	PermissionDecision PermissionDecision `json:"permissionDecision,omitempty"`

	// Reason is the human-readable explanation surfaced to the
	// user (and sometimes to the model) when the hook makes a
	// gating decision. Optional but strongly recommended for any
	// Deny — without it the user sees a generic "permission denied"
	// with no recourse.
	Reason string `json:"reason,omitempty"`
}

// IsDeny reports whether the hook denied the operation. Convenience
// for the agent layer's PreToolUse integration so the dispatch
// branch reads cleanly.
func (d HookDecision) IsDeny() bool {
	return d.PermissionDecision == PermissionDeny
}

// IsAllow reports whether the hook explicitly allowed the
// operation. Distinct from "no decision" (empty string) which means
// the default policy applies.
func (d HookDecision) IsAllow() bool {
	return d.PermissionDecision == PermissionAllow
}

// IsAsk reports whether the hook escalated the decision to the user.
// v1 has no interactive prompt yet; the agent layer treats Ask as
// equivalent to Deny until the prompt lands.
func (d HookDecision) IsAsk() bool {
	return d.PermissionDecision == PermissionAsk
}
