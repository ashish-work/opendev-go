package permissions

import (
	"regexp"
	"strings"
	"testing"
)

// mustCompile is a one-line regex compile for table tests. Test-only
// helper; production code uses the loader's error-returning compile.
func mustCompile(t *testing.T, src string) *regexp.Regexp {
	t.Helper()
	re, err := regexp.Compile(src)
	if err != nil {
		t.Fatalf("mustCompile %q: %v", src, err)
	}
	return re
}

// policyWith is a builder for test policies — keeps the per-case
// setup tight inside the table.
func policyWith(tool string, perm ToolPermission) Policy {
	return Policy{Tools: map[string]ToolPermission{tool: perm}}
}

func TestPolicyCheck_DefaultAllowForUnknownTool(t *testing.T) {
	t.Parallel()

	p := Policy{} // zero value
	d := p.Check("bash", `{"command":"ls"}`)
	if !d.Allowed {
		t.Fatalf("expected Allow for unknown tool, got Deny(%q)", d.Reason)
	}

	// Even with a populated Tools map, a tool name not in the map
	// must Allow — the zero-Tools case and the "tool not configured"
	// case must converge on the same default-allow behavior.
	p2 := policyWith("edit_file", ToolPermission{Enabled: false})
	d2 := p2.Check("bash", `{"command":"ls"}`)
	if !d2.Allowed {
		t.Fatalf("expected Allow for tool not in policy, got Deny(%q)", d2.Reason)
	}
}

func TestPolicyCheck_DisabledToolDenies(t *testing.T) {
	t.Parallel()

	p := policyWith("bash", ToolPermission{Enabled: false})
	d := p.Check("bash", `{"command":"ls"}`)
	if d.Allowed {
		t.Fatal("expected Deny when Enabled=false, got Allow")
	}
	if !strings.Contains(d.Reason, "disabled by policy") {
		t.Fatalf("expected Reason to mention disabled policy, got %q", d.Reason)
	}
	if !strings.Contains(d.Reason, `"bash"`) {
		t.Fatalf("expected Reason to quote tool name, got %q", d.Reason)
	}
}

func TestPolicyCheck_DisabledShortCircuitsBeforeDenyPatterns(t *testing.T) {
	t.Parallel()

	// Enabled=false beats a pattern check: even if patterns would
	// have matched (or not), the tool is fully off. The reason must
	// be the disabled-policy message, not a pattern-match message.
	p := policyWith("bash", ToolPermission{
		Enabled:            false,
		DenyPatterns:       []*regexp.Regexp{mustCompile(t, "rm -rf")},
		denyPatternSources: []string{"rm -rf"},
	})
	d := p.Check("bash", `{"command":"rm -rf /"}`)
	if d.Allowed {
		t.Fatal("expected Deny, got Allow")
	}
	if !strings.Contains(d.Reason, "disabled by policy") {
		t.Fatalf("expected disabled-policy reason, got %q", d.Reason)
	}
}

func TestPolicyCheck_AlwaysAllowBypassesPatterns(t *testing.T) {
	t.Parallel()

	p := policyWith("bash", ToolPermission{
		Enabled:            true,
		AlwaysAllow:        true,
		DenyPatterns:       []*regexp.Regexp{mustCompile(t, "rm -rf")},
		denyPatternSources: []string{"rm -rf"},
	})

	// A request that the deny pattern WOULD match — proving
	// AlwaysAllow short-circuits before pattern evaluation.
	d := p.Check("bash", `{"command":"rm -rf /"}`)
	if !d.Allowed {
		t.Fatalf("expected Allow when AlwaysAllow=true, got Deny(%q)", d.Reason)
	}
}

func TestPolicyCheck_DenyPatternMatchReportsSource(t *testing.T) {
	t.Parallel()

	// The Reason must quote the original regex source as the user
	// wrote it — not the compiled form. Compiled.String() can
	// surprise users with normalized escapes; using the stored
	// source keeps the error recognizable.
	source := "rm -rf /"
	p := policyWith("bash", ToolPermission{
		Enabled:            true,
		DenyPatterns:       []*regexp.Regexp{mustCompile(t, source)},
		denyPatternSources: []string{source},
	})

	d := p.Check("bash", `{"command":"rm -rf /tmp"}`)
	if d.Allowed {
		t.Fatal("expected Deny when a pattern matches, got Allow")
	}
	if !strings.Contains(d.Reason, "matches deny pattern") {
		t.Fatalf("expected matches-deny-pattern reason, got %q", d.Reason)
	}
	if !strings.Contains(d.Reason, source) {
		t.Fatalf("expected Reason to include source pattern %q, got %q",
			source, d.Reason)
	}
}

func TestPolicyCheck_NoMatchAllows(t *testing.T) {
	t.Parallel()

	p := policyWith("bash", ToolPermission{
		Enabled:            true,
		DenyPatterns:       []*regexp.Regexp{mustCompile(t, "rm -rf")},
		denyPatternSources: []string{"rm -rf"},
	})

	d := p.Check("bash", `{"command":"ls -la"}`)
	if !d.Allowed {
		t.Fatalf("expected Allow when no pattern matches, got Deny(%q)", d.Reason)
	}
}

func TestPolicyCheck_FirstMatchWins(t *testing.T) {
	t.Parallel()

	// Two patterns, both would match. Check must report the FIRST
	// one — predictable user model, no surprise about "which rule
	// fired."
	p := policyWith("bash", ToolPermission{
		Enabled: true,
		DenyPatterns: []*regexp.Regexp{
			mustCompile(t, "first"),
			mustCompile(t, "second"),
		},
		denyPatternSources: []string{"first", "second"},
	})

	d := p.Check("bash", "first second both present")
	if d.Allowed {
		t.Fatal("expected Deny, got Allow")
	}
	if !strings.Contains(d.Reason, `"first"`) {
		t.Fatalf("expected first pattern in reason, got %q", d.Reason)
	}
	if strings.Contains(d.Reason, `"second"`) {
		t.Fatalf("expected second pattern NOT in reason, got %q", d.Reason)
	}
}

func TestPolicyCheck_EmptyDenyPatternsAllowsAnything(t *testing.T) {
	t.Parallel()

	// An entry with Enabled=true but no patterns is the "this tool
	// is on" no-op case — useful as a project override against a
	// user-level deny, for instance.
	p := policyWith("bash", ToolPermission{Enabled: true})

	d := p.Check("bash", `{"command":"anything goes here"}`)
	if !d.Allowed {
		t.Fatalf("expected Allow with no patterns, got Deny(%q)", d.Reason)
	}
}

func TestAllowDenyConstructors(t *testing.T) {
	t.Parallel()

	a := Allow()
	if !a.Allowed || a.Reason != "" {
		t.Fatalf("Allow() = %+v, want {Allowed:true, Reason:\"\"}", a)
	}

	d := Deny("nope")
	if d.Allowed || d.Reason != "nope" {
		t.Fatalf("Deny(%q) = %+v, want {Allowed:false, Reason:%q}",
			"nope", d, "nope")
	}
}
