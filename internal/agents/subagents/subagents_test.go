package subagents

import (
	"reflect"
	"testing"
)

func TestAllowsTool_NilToolsAlwaysReturnsTrue(t *testing.T) {
	// nil Tools means "no restriction" — Registry.Filter passes
	// the parent's registry through unchanged.
	s := SubAgentSpec{Name: "x", Tools: nil}
	for _, name := range []string{"bash", "read_file", "unknown", ""} {
		if !s.AllowsTool(name) {
			t.Errorf("nil Tools should allow %q", name)
		}
	}
}

func TestAllowsTool_EmptySliceAllowsNothing(t *testing.T) {
	// Empty slice (distinct from nil) means "no tools at all."
	s := SubAgentSpec{Name: "x", Tools: []string{}}
	for _, name := range []string{"bash", "read_file", "anything"} {
		if s.AllowsTool(name) {
			t.Errorf("empty Tools should not allow %q", name)
		}
	}
}

func TestAllowsTool_WhitelistMembership(t *testing.T) {
	s := SubAgentSpec{
		Name:  "x",
		Tools: []string{"read_file", "list_files"},
	}
	if !s.AllowsTool("read_file") {
		t.Errorf("whitelisted read_file should be allowed")
	}
	if !s.AllowsTool("list_files") {
		t.Errorf("whitelisted list_files should be allowed")
	}
	if s.AllowsTool("bash") {
		t.Errorf("non-whitelisted bash should be denied")
	}
	if s.AllowsTool("") {
		t.Errorf("empty string should not match any whitelist entry")
	}
}

func TestSpecByName_KnownSpecs(t *testing.T) {
	cases := []string{"Explore", "Planner", "Build"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			spec, ok := SpecByName(name)
			if !ok {
				t.Fatalf("SpecByName(%q) returned ok=false", name)
			}
			if spec.Name != name {
				t.Errorf("spec.Name = %q, want %q", spec.Name, name)
			}
			if spec.SystemPrompt == "" {
				t.Errorf("spec %q has empty SystemPrompt", name)
			}
			if spec.MaxIterations <= 0 {
				t.Errorf("spec %q has non-positive MaxIterations: %d",
					name, spec.MaxIterations)
			}
		})
	}
}

func TestSpecByName_UnknownReturnsFalse(t *testing.T) {
	cases := []string{"", "explore", "EXPLORE", "Critique", "Unknown"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			spec, ok := SpecByName(name)
			if ok {
				t.Errorf("SpecByName(%q) returned ok=true; want false", name)
			}
			// Zero value of SubAgentSpec when not found.
			if spec.Name != "" {
				t.Errorf("not-found spec should be zero; got Name=%q", spec.Name)
			}
		})
	}
}

func TestExploreSpec_ReadOnlyToolSet(t *testing.T) {
	s := ExploreSpec
	for _, want := range []string{"read_file", "list_files", "bash"} {
		if !s.AllowsTool(want) {
			t.Errorf("Explore should allow %q", want)
		}
	}
	// Should NOT allow tools that mutate state.
	for _, deny := range []string{"edit_file", "write_file"} {
		if s.AllowsTool(deny) {
			t.Errorf("Explore should not allow mutating tool %q", deny)
		}
	}
}

func TestPlannerSpec_IncludesPresentPlan(t *testing.T) {
	// Forward reference: present_plan doesn't exist as a tool in
	// v2, but the spec declares it so Phase 7's Registry.Filter
	// (#40) and the spawn_subagent tool advertise it consistently
	// when the tool eventually lands.
	if !PlannerSpec.AllowsTool("present_plan") {
		t.Errorf("Planner spec should advertise present_plan in Tools")
	}
	// Also includes Explore's tools.
	for _, want := range []string{"read_file", "list_files", "bash"} {
		if !PlannerSpec.AllowsTool(want) {
			t.Errorf("Planner should include Explore's tool %q", want)
		}
	}
}

func TestBuildSpec_NilToolsMeansFullAccess(t *testing.T) {
	// Build is the unrestricted subagent — nil Tools.
	if BuildSpec.Tools != nil {
		t.Errorf("Build.Tools should be nil; got %v", BuildSpec.Tools)
	}
	// AllowsTool should return true for anything.
	for _, name := range []string{"bash", "edit_file", "write_file", "anything"} {
		if !BuildSpec.AllowsTool(name) {
			t.Errorf("Build should allow %q (nil Tools = no restriction)", name)
		}
	}
}

func TestBuiltins_KeysMatchSpecNames(t *testing.T) {
	// Defensive: catches a typo where a spec gets registered under
	// the wrong key. spawn_subagent (#38) looks up by the key the
	// model emits, which has to equal the spec's Name field.
	for key, spec := range Builtins {
		if spec.Name != key {
			t.Errorf("Builtins[%q].Name = %q; want equal", key, spec.Name)
		}
	}
}

func TestBuiltinNames_SortedDeterministic(t *testing.T) {
	got := BuiltinNames()
	want := []string{"Build", "Explore", "Planner"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("BuiltinNames() = %v, want %v", got, want)
	}
}

func TestBuiltinNames_StableAcrossCalls(t *testing.T) {
	// Two consecutive calls should return identical slices — Go's
	// map iteration is randomized, but sort.Strings normalizes.
	for i := 0; i < 10; i++ {
		if !reflect.DeepEqual(BuiltinNames(), BuiltinNames()) {
			t.Fatalf("BuiltinNames() not stable across calls")
		}
	}
}

func TestEachBuiltin_HasNonEmptySystemPrompt(t *testing.T) {
	// Documentation invariant — every built-in spec's
	// SystemPrompt has to actually describe the subagent's role.
	for _, spec := range Builtins {
		if spec.SystemPrompt == "" {
			t.Errorf("spec %q has empty SystemPrompt", spec.Name)
		}
	}
}

func TestEachBuiltin_HasPositiveMaxIterations(t *testing.T) {
	for _, spec := range Builtins {
		if spec.MaxIterations <= 0 {
			t.Errorf("spec %q has non-positive MaxIterations: %d",
				spec.Name, spec.MaxIterations)
		}
	}
}

func TestEachBuiltin_ModelOverrideEmptyByDefault(t *testing.T) {
	// v2 ships with subagents inheriting the parent's model so the
	// curriculum doesn't lock readers into a specific provider's
	// model lineup. If a future commit changes this default, this
	// test surfaces the deliberate choice.
	for _, spec := range Builtins {
		if spec.ModelOverride != "" {
			t.Errorf("spec %q has ModelOverride %q; want empty (inherit)",
				spec.Name, spec.ModelOverride)
		}
	}
}

func TestBuiltins_ContainsThreeSpecs(t *testing.T) {
	// The plan specifies exactly three built-ins for v2. If a
	// fourth gets added (e.g. Critique), this test fails so the
	// committer remembers to update the plan + docs.
	if len(Builtins) != 3 {
		t.Errorf("Builtins has %d entries; want 3 (Explore, Planner, Build)",
			len(Builtins))
	}
}
