// Package subagents defines the SubAgentSpec type — a description
// of a "subordinate" agent that the main ReactLoop can dispatch a
// child loop into via the spawn_subagent tool (#38). A spec answers
// four questions about the child loop:
//
//   - What's its role? (Name + SystemPrompt)
//   - Which tools can it use? (Tools)
//   - How many iterations is it allowed? (MaxIterations)
//   - Should it use a different LLM than the parent? (ModelOverride)
//
// Three built-in specs cover the common cases — Explore for
// read-only investigation, Planner for "explore then propose a
// plan," Build for full execution. Custom specs aren't supported
// in v2; the three built-ins are the surface. A user who wants
// a different role can fork and edit; a future post-v2 commit
// can add a public RegisterSpec API if the demand is there.
package subagents

import "sort"

// SubAgentSpec describes a subagent's capabilities and constraints.
// Constructed by hand for the three built-ins below and consumed by
// the spawn_subagent tool (#38) at dispatch time.
type SubAgentSpec struct {
	// Name is the human-readable identifier the model passes to
	// spawn_subagent (e.g. "Explore", "Planner", "Build"). Also
	// the key under which the spec is registered in Builtins.
	Name string

	// SystemPrompt is the role description handed to the child
	// loop as its system message. Short and role-specific: what
	// the subagent is, what it can do, what its boss expects back.
	SystemPrompt string

	// Tools is the whitelist of tool names the subagent can call.
	//
	//   - nil   = no restriction (registry passes through unchanged)
	//   - []    = no tools at all (subagent must answer from
	//             context alone — useful for a hypothetical
	//             "summarize this content" spec)
	//   - [...] = exact whitelist; unknown tool names are silently
	//             dropped by Registry.Filter (#40) so a spec can
	//             forward-reference tools that don't exist yet.
	Tools []string

	// MaxIterations caps the child loop's iteration count. Lower
	// than the parent's cap for focused work (Explore: 10),
	// matching for full agents (Build: 25).
	MaxIterations int

	// ModelOverride, when non-empty, replaces the parent's
	// configured execution model for this subagent. Empty means
	// "inherit parent's model" — v2 defaults to inheritance so
	// the curriculum doesn't lock readers into a specific
	// provider's lineup. Users who want cost tiering (e.g. cheap
	// Explore + expensive Build) can edit the built-in or add a
	// future RegisterSpec API.
	ModelOverride string
}

// AllowsTool reports whether the spec permits the named tool.
// nil Tools always returns true (no restriction); a non-nil
// (including empty) Tools enforces strict whitelist membership.
// Used by Registry.Filter (#40) when constructing the subagent's
// scoped tool set.
func (s SubAgentSpec) AllowsTool(name string) bool {
	if s.Tools == nil {
		return true
	}
	for _, t := range s.Tools {
		if t == name {
			return true
		}
	}
	return false
}

// ExploreSpec is the read-only investigation subagent. Its job is
// to answer questions about the codebase by reading and listing
// files; bash is included for content search via `grep -rn` style
// commands. Phase 8 (#42-43) is expected to restrict bash to
// read-only commands when invoked under this spec via the
// permissions policy.
//
// Iteration cap is low — focused exploration should converge
// quickly; if it doesn't, the question is probably too vague.
var ExploreSpec = SubAgentSpec{
	Name: "Explore",
	SystemPrompt: "You are an Explore subagent. " +
		"Your job is to investigate the codebase to answer the user's question. " +
		"You have read-only tools: read_file, list_files, and bash for content search. " +
		"Be thorough but focused — return a concise summary of your findings.",
	Tools:         []string{"read_file", "list_files", "bash"},
	MaxIterations: 10,
}

// PlannerSpec adds a hypothetical present_plan tool on top of
// Explore's read-only set. present_plan is a forward reference —
// it doesn't exist as a tool in v2. The registry filter (#40)
// silently skips tool names the registry doesn't know about, so
// the unknown entry is harmless until the tool lands (a v3
// candidate). If you want a Planner subagent today, it'll only
// have Explore's tools; that still covers most planning tasks
// (read, list, propose — the proposal goes back via the
// subagent's final Content).
var PlannerSpec = SubAgentSpec{
	Name: "Planner",
	SystemPrompt: "You are a Planner subagent. " +
		"Investigate the codebase, then return a clear implementation plan: " +
		"concrete steps, risks, and dependencies. Use your read-only tools to " +
		"understand the problem space before proposing the plan.",
	Tools:         []string{"read_file", "list_files", "bash", "present_plan"},
	MaxIterations: 15,
}

// BuildSpec is the full-access subagent — used when the main
// agent delegates a complete sub-task ("refactor this package",
// "add this feature"). nil Tools means the subagent inherits the
// parent's complete registry. Iteration cap matches the main
// loop's default since Build subagents are essentially full
// agents working on a focused goal.
var BuildSpec = SubAgentSpec{
	Name: "Build",
	SystemPrompt: "You are a Build subagent. " +
		"Execute the requested task using the available tools. " +
		"Report what you accomplished when finished, including any caveats " +
		"the parent agent should know about.",
	Tools:         nil,
	MaxIterations: 25,
}

// Builtins is the registry of built-in subagent specs, keyed by
// Name. spawn_subagent (#38) looks up specs here when dispatching.
// New built-ins (or future user-registered specs) would land in
// this map.
var Builtins = map[string]SubAgentSpec{
	ExploreSpec.Name: ExploreSpec,
	PlannerSpec.Name: PlannerSpec,
	BuildSpec.Name:   BuildSpec,
}

// SpecByName returns the built-in spec for the given name plus an
// ok bool. Unknown names return (zero, false) so the caller can
// distinguish "not found" from "found with empty fields." Matches
// the hooks.Parse pattern.
func SpecByName(name string) (SubAgentSpec, bool) {
	spec, ok := Builtins[name]
	return spec, ok
}

// BuiltinNames returns the registered spec names in deterministic
// sorted order. Used by spawn_subagent (#38) to advertise the
// allowed agent_type enum values in its JSON schema — the order
// has to be stable across binary runs so the schema doesn't churn.
func BuiltinNames() []string {
	names := make([]string, 0, len(Builtins))
	for name := range Builtins {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
