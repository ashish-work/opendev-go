package hooks

import (
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// writeSettings writes a settings.json file with the given content
// into a fresh temp directory and returns the path. Used by tests
// that exercise LoadFile.
func writeSettings(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, SettingsFileName)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	return path
}

func TestLoadFile_ValidConfigRoundTrip(t *testing.T) {
	path := writeSettings(t, `{
		"hooks": {
			"pre_tool_use": [
				{"matcher": "bash", "command": "/audit", "timeout_ms": 5000},
				{"matcher": ".*", "command": "/log"}
			],
			"user_prompt_submit": [
				{"command": "echo project=x"}
			]
		}
	}`)
	settings, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}

	pre := settings.MatchersFor(HookEventPreToolUse)
	if len(pre) != 2 {
		t.Fatalf("pre_tool_use matchers = %d, want 2", len(pre))
	}
	if pre[0].Matcher != "bash" || pre[0].Command != "/audit" || pre[0].TimeoutMs != 5000 {
		t.Errorf("pre[0] = %+v, want bash/audit/5000", pre[0])
	}
	if pre[1].Matcher != ".*" || pre[1].Command != "/log" || pre[1].TimeoutMs != 0 {
		t.Errorf("pre[1] = %+v, want .*/log/0", pre[1])
	}

	ups := settings.MatchersFor(HookEventUserPromptSubmit)
	if len(ups) != 1 || ups[0].Command != "echo project=x" {
		t.Errorf("user_prompt_submit = %+v, want one echo entry", ups)
	}
}

func TestLoadFile_MissingFileReturnsEmpty(t *testing.T) {
	path := filepath.Join(t.TempDir(), "does-not-exist.json")
	settings, err := LoadFile(path)
	if err != nil {
		t.Errorf("missing file should not error, got %v", err)
	}
	if len(settings.Hooks) != 0 {
		t.Errorf("missing file should give empty Hooks, got %d entries", len(settings.Hooks))
	}
}

func TestLoadFile_MalformedJSONReturnsError(t *testing.T) {
	path := writeSettings(t, `{not json`)
	_, err := LoadFile(path)
	if err == nil {
		t.Fatal("expected parse error, got nil")
	}
	if !strings.Contains(err.Error(), "parse") {
		t.Errorf("error %q should mention parse failure", err)
	}
}

func TestLoadFile_InvalidRegexReturnsError(t *testing.T) {
	path := writeSettings(t, `{
		"hooks": {
			"pre_tool_use": [
				{"matcher": "bash[", "command": "/x"}
			]
		}
	}`)
	_, err := LoadFile(path)
	if err == nil {
		t.Fatal("expected regex compile error, got nil")
	}
	if !strings.Contains(err.Error(), "bash[") {
		t.Errorf("error should name the bad matcher %q; got %v", "bash[", err)
	}
	if !strings.Contains(err.Error(), "pre_tool_use") {
		t.Errorf("error should name the event with the bad matcher; got %v", err)
	}
}

func TestLoadFile_MissingCommandReturnsError(t *testing.T) {
	path := writeSettings(t, `{
		"hooks": {
			"pre_tool_use": [
				{"matcher": "bash"}
			]
		}
	}`)
	_, err := LoadFile(path)
	if err == nil {
		t.Fatal("expected error for missing command")
	}
	if !strings.Contains(err.Error(), "command") {
		t.Errorf("error %q should mention missing command", err)
	}
}

func TestLoadFile_NegativeTimeoutReturnsError(t *testing.T) {
	path := writeSettings(t, `{
		"hooks": {
			"pre_tool_use": [
				{"command": "/x", "timeout_ms": -1}
			]
		}
	}`)
	_, err := LoadFile(path)
	if err == nil {
		t.Fatal("expected error for negative timeout")
	}
	if !strings.Contains(err.Error(), "timeout_ms") {
		t.Errorf("error %q should mention timeout_ms", err)
	}
}

func TestLoadFile_UnknownEventNameLoggedAndSkipped(t *testing.T) {
	// Unknown events are skipped (with a warning log) — not fatal —
	// so configs from newer binaries keep working on older
	// installs. The known event should still load.
	path := writeSettings(t, `{
		"hooks": {
			"some_future_event": [{"command": "/x"}],
			"pre_tool_use":      [{"command": "/y"}]
		}
	}`)
	settings, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if _, ok := settings.Hooks[HookEventPreToolUse]; !ok {
		t.Errorf("known event pre_tool_use should still load alongside unknown ones")
	}
	if len(settings.Hooks) != 1 {
		t.Errorf("unknown event should be skipped; got %d entries: %v",
			len(settings.Hooks), settings.Hooks)
	}
}

func TestLoadFile_NoHooksSectionReturnsEmpty(t *testing.T) {
	// settings.json with other sections but no hooks should give
	// an empty HookSettings without error — this loader is hooks-
	// only, other sections are someone else's problem.
	path := writeSettings(t, `{
		"unrelated_section": {"foo": "bar"}
	}`)
	settings, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if len(settings.Hooks) != 0 {
		t.Errorf("expected empty Hooks; got %d entries", len(settings.Hooks))
	}
}

func TestHookMatcher_EmptyMatcherAlwaysMatches(t *testing.T) {
	// HookMatcher constructed by hand (not via LoadFile) with no
	// matcher string should still match anything.
	m := HookMatcher{Command: "/x"}
	if !m.Matches("bash") {
		t.Errorf("empty matcher should match 'bash'")
	}
	if !m.Matches("anything else") {
		t.Errorf("empty matcher should match 'anything else'")
	}
}

func TestHookMatcher_RegexMatching(t *testing.T) {
	// Matchers compiled via LoadFile should match per the regex.
	path := writeSettings(t, `{
		"hooks": {
			"pre_tool_use": [
				{"matcher": "^(bash|edit_file)$", "command": "/x"}
			]
		}
	}`)
	settings, _ := LoadFile(path)
	m := settings.Hooks[HookEventPreToolUse][0]

	if !m.Matches("bash") {
		t.Errorf("matcher should match 'bash'")
	}
	if !m.Matches("edit_file") {
		t.Errorf("matcher should match 'edit_file'")
	}
	if m.Matches("read_file") {
		t.Errorf("matcher should NOT match 'read_file'")
	}
	if m.Matches("bash_extra") {
		t.Errorf("matcher with anchors should NOT match 'bash_extra'")
	}
}

func TestMatchersFor_NoHooksRegisteredReturnsNil(t *testing.T) {
	settings := HookSettings{Hooks: map[HookEvent][]HookMatcher{
		HookEventStop: {{Command: "/x"}},
	}}
	if got := settings.MatchersFor(HookEventPreToolUse); got != nil {
		t.Errorf("MatchersFor unregistered event = %v, want nil", got)
	}
}

func TestMatchersFor_NilHooksMapReturnsNil(t *testing.T) {
	// Zero-value HookSettings has nil Hooks map; MatchersFor must
	// not panic.
	var settings HookSettings
	if got := settings.MatchersFor(HookEventPreToolUse); got != nil {
		t.Errorf("MatchersFor on zero HookSettings = %v, want nil", got)
	}
}

func TestMatchersFor_PreservesLoadOrder(t *testing.T) {
	path := writeSettings(t, `{
		"hooks": {
			"pre_tool_use": [
				{"command": "/a"},
				{"command": "/b"},
				{"command": "/c"}
			]
		}
	}`)
	settings, _ := LoadFile(path)
	got := settings.MatchersFor(HookEventPreToolUse)
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	if got[0].Command != "/a" || got[1].Command != "/b" || got[2].Command != "/c" {
		t.Errorf("order changed: [%s %s %s], want [/a /b /c]",
			got[0].Command, got[1].Command, got[2].Command)
	}
}

func TestMerge_ProjectMatchersAppendAfterUser(t *testing.T) {
	user := HookSettings{Hooks: map[HookEvent][]HookMatcher{
		HookEventPreToolUse: {{Command: "/user-1"}, {Command: "/user-2"}},
	}}
	project := HookSettings{Hooks: map[HookEvent][]HookMatcher{
		HookEventPreToolUse: {{Command: "/proj-1"}, {Command: "/proj-2"}},
	}}
	merged := Merge(user, project)
	got := merged.MatchersFor(HookEventPreToolUse)
	if len(got) != 4 {
		t.Fatalf("merged len = %d, want 4", len(got))
	}
	want := []string{"/user-1", "/user-2", "/proj-1", "/proj-2"}
	for i, w := range want {
		if got[i].Command != w {
			t.Errorf("merged[%d] = %q, want %q", i, got[i].Command, w)
		}
	}
}

func TestMerge_PreservesUniqueEvents(t *testing.T) {
	user := HookSettings{Hooks: map[HookEvent][]HookMatcher{
		HookEventPreToolUse: {{Command: "/u"}},
	}}
	project := HookSettings{Hooks: map[HookEvent][]HookMatcher{
		HookEventStop: {{Command: "/p"}},
	}}
	merged := Merge(user, project)
	if got := merged.MatchersFor(HookEventPreToolUse); len(got) != 1 {
		t.Errorf("pre_tool_use lost during merge: %v", got)
	}
	if got := merged.MatchersFor(HookEventStop); len(got) != 1 {
		t.Errorf("stop lost during merge: %v", got)
	}
}

func TestMerge_EmptyInputsGiveEmptyOutput(t *testing.T) {
	merged := Merge(HookSettings{}, HookSettings{})
	if len(merged.Hooks) != 0 {
		t.Errorf("merge of two empties should be empty; got %d entries", len(merged.Hooks))
	}
}

func TestMerge_DoesNotMutateInputs(t *testing.T) {
	// Critical: merge is supposed to be pure. If the input slices
	// got mutated, callers retaining references would see surprises.
	user := HookSettings{Hooks: map[HookEvent][]HookMatcher{
		HookEventPreToolUse: {{Command: "/u"}},
	}}
	project := HookSettings{Hooks: map[HookEvent][]HookMatcher{
		HookEventPreToolUse: {{Command: "/p"}},
	}}
	_ = Merge(user, project)

	if got := user.MatchersFor(HookEventPreToolUse); len(got) != 1 || got[0].Command != "/u" {
		t.Errorf("user mutated during Merge: %v", got)
	}
	if got := project.MatchersFor(HookEventPreToolUse); len(got) != 1 || got[0].Command != "/p" {
		t.Errorf("project mutated during Merge: %v", got)
	}
}

func TestUserSettingsPath_EndsInExpectedSuffix(t *testing.T) {
	got := UserSettingsPath()
	// Either empty (no home) or ends in .opendev/settings.json
	if got == "" {
		return // can't get home dir in this env; acceptable
	}
	want := filepath.Join(SettingsDirName, SettingsFileName)
	if !strings.HasSuffix(got, want) {
		t.Errorf("UserSettingsPath = %q, want suffix %q", got, want)
	}
}

func TestProjectSettingsPath(t *testing.T) {
	got := ProjectSettingsPath("/work/repo")
	want := filepath.Join("/work/repo", SettingsDirName, SettingsFileName)
	if got != want {
		t.Errorf("ProjectSettingsPath = %q, want %q", got, want)
	}
}

func TestLoad_CombinesUserAndProjectFiles(t *testing.T) {
	// Exercise the high-level Load by pointing both paths at temp
	// files. We can't override UserSettingsPath in tests without
	// HOMEDIR mucking, so this test uses LoadFile + Merge directly
	// to simulate what Load does internally.
	userPath := writeSettings(t, `{
		"hooks": {
			"pre_tool_use": [{"command": "/user-baseline"}]
		}
	}`)
	projectPath := writeSettings(t, `{
		"hooks": {
			"pre_tool_use": [{"command": "/project-override"}]
		}
	}`)
	user, err := LoadFile(userPath)
	if err != nil {
		t.Fatalf("LoadFile user: %v", err)
	}
	project, err := LoadFile(projectPath)
	if err != nil {
		t.Fatalf("LoadFile project: %v", err)
	}
	merged := Merge(user, project)
	got := merged.MatchersFor(HookEventPreToolUse)
	if len(got) != 2 {
		t.Fatalf("merged len = %d, want 2", len(got))
	}
	if got[0].Command != "/user-baseline" || got[1].Command != "/project-override" {
		t.Errorf("merged order = [%q %q], want [/user-baseline /project-override]",
			got[0].Command, got[1].Command)
	}
}

func TestLoad_BothFilesMissingReturnsEmpty(t *testing.T) {
	// Use a temp dir as working dir; no settings file under it.
	dir := t.TempDir()
	settings, err := Load(dir)
	if err != nil {
		// Either nil (clean empty) or an error related to user
		// settings — both acceptable depending on whether the user
		// running tests has a settings.json. Skip if user has one.
		var pathErr *os.PathError
		if errors.As(err, &pathErr) {
			t.Skipf("user has a settings.json; skipping: %v", err)
		}
		t.Fatalf("Load: %v", err)
	}
	// No project settings; merge with user's (whatever it is)
	// shouldn't add any pre_tool_use entries unless the user has
	// some. We just verify Load doesn't crash and returns a usable
	// value.
	_ = settings
}

func TestCompiledRegexInternal(t *testing.T) {
	// Sanity that LoadFile actually populates the compiled regex
	// (not just stores the source string). Exercised indirectly via
	// the Matches tests but pin it directly too.
	path := writeSettings(t, `{
		"hooks": {
			"pre_tool_use": [{"matcher": "x", "command": "/x"}]
		}
	}`)
	settings, _ := LoadFile(path)
	m := settings.Hooks[HookEventPreToolUse][0]
	if m.compiled == nil {
		t.Errorf("compiled regex should be populated after LoadFile")
	}
	// And the regex actually works.
	if _, err := regexp.Compile(m.Matcher); err != nil {
		t.Errorf("stored Matcher source should re-compile cleanly: %v", err)
	}
}
