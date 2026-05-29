// Package permissions models the runtime policy that gates tool
// dispatch. A Policy holds a per-tool ToolPermission; the loop
// consults Policy.Check before running each tool call (wired in a
// later commit). The data plane lives here; the dispatch-side
// enforcement lives next to executeSequentialPhase.
//
// Loading: a top-level "permissions" key in the same
// .opendev/settings.json file that hooks already reads (see
// config.go). Missing file or missing key = empty Policy = nothing
// is denied, which preserves the v1 default for users who don't
// opt in.
//
// Concurrency: Policy is read-only after Load returns. Check is safe
// for concurrent callers — it touches no mutable state — and the
// compiled regexps inside ToolPermission are themselves
// goroutine-safe (per the regexp package contract). No mutex.
package permissions

import (
	"fmt"
	"regexp"
)

// ToolPermission is the policy for a single tool. The zero value
// has Enabled=false, which when held inside a Policy entry means
// "explicitly disabled" — distinct from "no entry at all" (which
// Check treats as Allow). The loader makes this distinction
// trustworthy by defaulting Enabled to true when the JSON entry
// omits the field.
type ToolPermission struct {
	// Enabled gates whether the tool can be called at all. When
	// false, Check returns Deny regardless of the rest of the
	// policy. Default is true once an entry exists — see the
	// loader's *bool handling.
	Enabled bool

	// AlwaysAllow skips the deny-pattern loop. Useful for "I've
	// already vetted this tool, stop pattern-matching every call."
	// Ignored when Enabled is false.
	AlwaysAllow bool

	// DenyPatterns are matched against the serialized args JSON.
	// First match wins. Patterns apply uniformly across tools —
	// the permissions package intentionally does not know any
	// tool's argument schema. Users who want to target a specific
	// field write a pattern that includes the JSON shape (e.g.
	// `"command":\s*"sudo`).
	DenyPatterns []*regexp.Regexp

	// denyPatternSources holds the original regex strings parallel
	// to DenyPatterns so Decision.Reason can quote what the user
	// wrote rather than the compiled form (which loses anchors
	// like (?m) flags in some Go versions' String()).
	denyPatternSources []string
}

// Policy is the loaded permissions configuration keyed by tool name.
// The zero value is usable — Check on a nil Tools map returns Allow
// for every name, which is the v1-compatible default.
type Policy struct {
	Tools map[string]ToolPermission
}

// Decision is the outcome of Policy.Check. Allowed is the gate;
// Reason is empty on allow and carries a human-readable explanation
// on deny so the model and the user both see WHY the tool call was
// blocked.
type Decision struct {
	Allowed bool
	Reason  string
}

// Allow returns a permissive Decision. Used as a shorthand in
// Check's default-allow paths and exported for callers (mostly
// tests) that want to construct decisions without the struct
// literal.
func Allow() Decision {
	return Decision{Allowed: true}
}

// Deny returns a blocking Decision carrying reason. The reason is
// surfaced to the model as the tool result's Error field (wired in
// a later commit) so the model can understand the gate and adjust.
func Deny(reason string) Decision {
	return Decision{Allowed: false, Reason: reason}
}

// Check evaluates the policy for one tool call. argsJSON is the
// raw arguments string the registry would dispatch with — typically
// the json.RawMessage from the LLM response, converted to string.
//
// The five branches, in order:
//
//  1. No entry for toolName → Allow. Default-allow keeps v1 users
//     who never created a settings.json unaffected.
//  2. Enabled=false → Deny ("tool %q disabled by policy"). The
//     entire tool is off; deny patterns are not consulted.
//  3. AlwaysAllow=true → Allow. Bypass the pattern loop.
//  4. DenyPatterns match → Deny with the matching pattern's source
//     in the reason so the user recognizes the rule that fired.
//     First match wins; later patterns are not evaluated.
//  5. Otherwise → Allow.
//
// Performance: O(number of deny patterns) per call. Patterns are
// pre-compiled at load time. Typical policies have <10 patterns;
// the linear scan is negligible compared to a tool's actual work.
func (p Policy) Check(toolName, argsJSON string) Decision {
	perm, ok := p.Tools[toolName]
	if !ok {
		return Allow()
	}

	if !perm.Enabled {
		return Deny(fmt.Sprintf(
			"tool %q disabled by policy", toolName,
		))
	}

	if perm.AlwaysAllow {
		return Allow()
	}

	for i, re := range perm.DenyPatterns {
		if re.MatchString(argsJSON) {
			source := perm.denyPatternSources[i]
			return Deny(fmt.Sprintf(
				"matches deny pattern %q", source,
			))
		}
	}

	return Allow()
}
