package permissions

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeSettings is a helper that drops a settings.json under a
// fresh tempdir and returns its path. Centralized here so every
// loader test can share the same on-disk shape.
func writeSettings(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "settings.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	return path
}

func TestLoadFile_MissingFileReturnsEmptyPolicy(t *testing.T) {
	t.Parallel()

	p, err := LoadFile(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err != nil {
		t.Fatalf("expected nil error for missing file, got %v", err)
	}
	if len(p.Tools) != 0 {
		t.Fatalf("expected empty Policy, got %d entries", len(p.Tools))
	}
}

func TestLoadFile_EmptyPermissionsBlockReturnsEmptyPolicy(t *testing.T) {
	t.Parallel()

	// File exists, parses fine, has no "permissions" key at all
	// (just a hooks section, say). Must produce a no-op policy.
	path := writeSettings(t, `{"hooks": {}}`)
	p, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if len(p.Tools) != 0 {
		t.Fatalf("expected empty policy, got %d tools", len(p.Tools))
	}
}

func TestLoadFile_ValidEntryCompilesPatterns(t *testing.T) {
	t.Parallel()

	// Patterns are matched against the serialized args JSON, not
	// against any extracted field — so anchors like ^ would bind
	// to the JSON's leading "{", not to the start of the bash
	// command. Tests use unanchored patterns to mirror what
	// users will actually write in practice.
	path := writeSettings(t, `{
		"permissions": {
			"bash": {
				"deny_patterns": ["rm -rf", "sudo "]
			}
		}
	}`)

	p, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}

	perm, ok := p.Tools["bash"]
	if !ok {
		t.Fatal("expected bash entry in loaded policy")
	}
	if !perm.Enabled {
		t.Fatal("expected Enabled to default to true when field is omitted")
	}
	if perm.AlwaysAllow {
		t.Fatal("expected AlwaysAllow default false")
	}
	if got, want := len(perm.DenyPatterns), 2; got != want {
		t.Fatalf("DenyPatterns len = %d, want %d", got, want)
	}
	if got, want := len(perm.denyPatternSources), 2; got != want {
		t.Fatalf("denyPatternSources len = %d, want %d", got, want)
	}

	// Make sure the compiled regexes actually work.
	d := p.Check("bash", `{"command":"rm -rf /"}`)
	if d.Allowed {
		t.Fatalf("expected loaded pattern to deny 'rm -rf /', got Allow")
	}
}

func TestLoadFile_ExplicitEnabledFalseDisables(t *testing.T) {
	t.Parallel()

	path := writeSettings(t, `{
		"permissions": {
			"edit_file": {"enabled": false}
		}
	}`)

	p, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}

	perm := p.Tools["edit_file"]
	if perm.Enabled {
		t.Fatal("expected Enabled=false, got true")
	}

	d := p.Check("edit_file", `{}`)
	if d.Allowed {
		t.Fatal("expected Deny for disabled tool, got Allow")
	}
}

func TestLoadFile_EnabledOmittedDefaultsTrue(t *testing.T) {
	t.Parallel()

	// The defaulting that exists to prevent the footgun: a user
	// who specifies only deny_patterns must get an enabled tool
	// with denies layered on top, not a fully-disabled tool.
	path := writeSettings(t, `{
		"permissions": {
			"bash": {"deny_patterns": ["danger"]}
		}
	}`)

	p, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if !p.Tools["bash"].Enabled {
		t.Fatal("expected Enabled to default true when omitted, got false")
	}
}

func TestLoadFile_AlwaysAllowParsed(t *testing.T) {
	t.Parallel()

	path := writeSettings(t, `{
		"permissions": {
			"bash": {"always_allow": true}
		}
	}`)

	p, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if !p.Tools["bash"].AlwaysAllow {
		t.Fatal("expected AlwaysAllow=true, got false")
	}
}

func TestLoadFile_InvalidRegexErrorsWithPathAndIndex(t *testing.T) {
	t.Parallel()

	path := writeSettings(t, `{
		"permissions": {
			"bash": {"deny_patterns": ["valid", "[unclosed"]}
		}
	}`)

	_, err := LoadFile(path)
	if err == nil {
		t.Fatal("expected error for invalid regex, got nil")
	}
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("expected ErrInvalidConfig, got %v", err)
	}

	msg := err.Error()
	wantSubs := []string{
		"bash",         // tool name
		"deny_patterns", // field name
		"[1]",          // index of the bad pattern
		"[unclosed",    // the offending pattern literal
		path,           // path so the user can find the file
	}
	for _, sub := range wantSubs {
		if !strings.Contains(msg, sub) {
			t.Errorf("error message missing %q: %s", sub, msg)
		}
	}
}

func TestLoadFile_MalformedJSONErrors(t *testing.T) {
	t.Parallel()

	path := writeSettings(t, `{this is not json`)
	_, err := LoadFile(path)
	if err == nil {
		t.Fatal("expected error for malformed JSON")
	}
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("expected ErrInvalidConfig, got %v", err)
	}
}

func TestMerge_ProjectOverridesUserPerTool(t *testing.T) {
	t.Parallel()

	user := Policy{Tools: map[string]ToolPermission{
		"bash": {Enabled: true, AlwaysAllow: true},
	}}
	project := Policy{Tools: map[string]ToolPermission{
		"bash": {Enabled: false},
	}}

	merged := Merge(user, project)
	got := merged.Tools["bash"]
	if got.Enabled {
		t.Fatal("expected project's Enabled=false to win, got Enabled=true")
	}
	if got.AlwaysAllow {
		t.Fatal("expected whole-entry replacement: user's AlwaysAllow must NOT carry over")
	}
}

func TestMerge_UserOnlyToolsPassThrough(t *testing.T) {
	t.Parallel()

	user := Policy{Tools: map[string]ToolPermission{
		"bash":      {Enabled: true},
		"edit_file": {Enabled: false},
	}}
	project := Policy{Tools: map[string]ToolPermission{}}

	merged := Merge(user, project)
	if !merged.Tools["bash"].Enabled {
		t.Fatal("expected bash entry to carry over from user")
	}
	if merged.Tools["edit_file"].Enabled {
		t.Fatal("expected edit_file entry to carry over from user")
	}
}

func TestMerge_ProjectOnlyToolsAdded(t *testing.T) {
	t.Parallel()

	user := Policy{Tools: map[string]ToolPermission{}}
	project := Policy{Tools: map[string]ToolPermission{
		"bash": {Enabled: true, AlwaysAllow: true},
	}}

	merged := Merge(user, project)
	got, ok := merged.Tools["bash"]
	if !ok {
		t.Fatal("expected project-only tool to appear in merged policy")
	}
	if !got.AlwaysAllow {
		t.Fatal("expected project's AlwaysAllow to be preserved")
	}
}

func TestMerge_NeitherMutatesInputs(t *testing.T) {
	t.Parallel()

	user := Policy{Tools: map[string]ToolPermission{
		"bash": {Enabled: true},
	}}
	project := Policy{Tools: map[string]ToolPermission{
		"bash":      {Enabled: false},
		"edit_file": {Enabled: false},
	}}

	_ = Merge(user, project)

	if len(user.Tools) != 1 {
		t.Fatalf("Merge mutated user policy: now %d entries", len(user.Tools))
	}
	if len(project.Tools) != 2 {
		t.Fatalf("Merge mutated project policy: now %d entries", len(project.Tools))
	}
	if !user.Tools["bash"].Enabled {
		t.Fatal("Merge mutated user.Tools[bash]")
	}
}

func TestLoad_UserAndProjectMerged(t *testing.T) {
	// NOT t.Parallel: this test calls t.Setenv to install a fake
	// $HOME, which Go's testing package forbids from running in
	// parallel with sibling tests.

	// Stand up a fake $HOME so userSettingsPath resolves to our
	// tempdir instead of the developer's real ~/.opendev. Go's
	// os.UserHomeDir respects $HOME on Unix.
	fakeHome := t.TempDir()
	t.Setenv("HOME", fakeHome)

	// User settings: enable bash with a single deny.
	userDir := filepath.Join(fakeHome, ".opendev")
	if err := os.MkdirAll(userDir, 0o755); err != nil {
		t.Fatalf("mkdir user: %v", err)
	}
	userSettings := filepath.Join(userDir, "settings.json")
	if err := os.WriteFile(userSettings, []byte(`{
		"permissions": {
			"bash": {"deny_patterns": ["user-deny"]}
		}
	}`), 0o600); err != nil {
		t.Fatalf("write user settings: %v", err)
	}

	// Project settings: replace bash's policy and add edit_file.
	projectRoot := t.TempDir()
	projectDir := filepath.Join(projectRoot, ".opendev")
	if err := os.MkdirAll(projectDir, 0o755); err != nil {
		t.Fatalf("mkdir project: %v", err)
	}
	projectSettings := filepath.Join(projectDir, "settings.json")
	if err := os.WriteFile(projectSettings, []byte(`{
		"permissions": {
			"bash":      {"deny_patterns": ["project-deny"]},
			"edit_file": {"enabled": false}
		}
	}`), 0o600); err != nil {
		t.Fatalf("write project settings: %v", err)
	}

	p, err := Load(projectRoot)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Bash: project replaces user; the user's "user-deny" pattern
	// should NOT fire — only the project's "project-deny" should.
	if d := p.Check("bash", "user-deny matches here"); !d.Allowed {
		t.Fatalf("expected user-only pattern not to apply after project override, got Deny(%q)",
			d.Reason)
	}
	if d := p.Check("bash", "project-deny matches here"); d.Allowed {
		t.Fatal("expected project deny pattern to fire, got Allow")
	}

	// edit_file: project-only entry must have come through.
	if d := p.Check("edit_file", `{}`); d.Allowed {
		t.Fatal("expected edit_file disabled via project policy, got Allow")
	}
}

func TestLoad_NeitherFileExistsReturnsEmpty(t *testing.T) {
	// NOT t.Parallel: t.Setenv blocks parallel siblings.

	fakeHome := t.TempDir()
	t.Setenv("HOME", fakeHome)
	workingDir := t.TempDir()

	p, err := Load(workingDir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(p.Tools) != 0 {
		t.Fatalf("expected empty policy, got %d tools", len(p.Tools))
	}

	// And a Check on this empty policy still defaults to Allow,
	// preserving v1 behavior end-to-end.
	if d := p.Check("bash", `{"command":"anything"}`); !d.Allowed {
		t.Fatalf("expected Allow from empty Load, got Deny(%q)", d.Reason)
	}
}
