package session

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/ashish-work/opendev-go/internal/hooks"
)

// idShape matches "sess_<8hex>_<unix>".
var idShape = regexp.MustCompile(`^sess_[0-9a-f]{8}_\d+$`)

func TestNew_PopulatesFields(t *testing.T) {
	before := time.Now()
	s := New("/work/repo")
	after := time.Now()

	if s.WorkingDir != "/work/repo" {
		t.Errorf("WorkingDir = %q, want /work/repo", s.WorkingDir)
	}
	if !idShape.MatchString(s.ID) {
		t.Errorf("ID = %q, want sess_<hex8>_<unix>", s.ID)
	}
	if s.StartedAt.Before(before) || s.StartedAt.After(after) {
		t.Errorf("StartedAt = %v, want between %v and %v", s.StartedAt, before, after)
	}
}

func TestNew_UniqueIDs(t *testing.T) {
	// Generate enough that any deterministic ID collisions surface.
	seen := map[string]bool{}
	for i := 0; i < 1000; i++ {
		id := New("/x").ID
		if seen[id] {
			t.Fatalf("duplicate session ID at iter %d: %q", i, id)
		}
		seen[id] = true
	}
}

func TestGenerateID_ShapeStable(t *testing.T) {
	id := generateID(time.Unix(1700000000, 0))
	if !idShape.MatchString(id) {
		t.Errorf("generateID = %q, want sess_<hex8>_<unix>", id)
	}
	if !strings.HasSuffix(id, "_1700000000") {
		t.Errorf("generateID = %q, want suffix _1700000000", id)
	}
}

// writeSettings writes a one-event settings file and returns a
// configured Manager attached to the temp executor.
func newTestManager(t *testing.T, event string, command string) *hooks.Manager {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "settings.json")
	body := `{"hooks":{"` + event + `":[{"command":` + jsonEsc(command) + `}]}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	settings, err := hooks.LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	return hooks.NewManager(settings, hooks.NewExecutor(""))
}

func jsonEsc(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestFireStart_NilManagerIsNoOp(t *testing.T) {
	s := New("/x")
	got, err := s.FireStart(context.Background(), nil)
	if err != nil {
		t.Errorf("nil manager should not error; got %v", err)
	}
	if got.Denied || got.AdditionalContext != "" || got.Reason != "" {
		t.Errorf("nil manager should give zero StartResult; got %+v", got)
	}
}

func TestFireStart_AdditionalContextReturned(t *testing.T) {
	mgr := newTestManager(t, "session_start",
		`echo '{"additionalContext":"Be concise."}'`)
	s := New("/x")

	got, err := s.FireStart(context.Background(), mgr)
	if err != nil {
		t.Fatalf("FireStart: %v", err)
	}
	if got.AdditionalContext != "Be concise." {
		t.Errorf("AdditionalContext = %q, want 'Be concise.'", got.AdditionalContext)
	}
	if got.Denied {
		t.Errorf("Denied should be false on allow-only hook")
	}
}

func TestFireStart_DenySurfacedAsDenied(t *testing.T) {
	mgr := newTestManager(t, "session_start",
		`echo '{"permissionDecision":"deny","reason":"session blocked"}'`)
	s := New("/x")

	got, err := s.FireStart(context.Background(), mgr)
	if err != nil {
		t.Fatalf("FireStart: %v", err)
	}
	if !got.Denied {
		t.Errorf("Denied = false, want true")
	}
	if got.Reason != "session blocked" {
		t.Errorf("Reason = %q, want 'session blocked'", got.Reason)
	}
}

func TestFirePromptSubmit_NilManagerReturnsOriginal(t *testing.T) {
	s := New("/x")
	got, err := s.FirePromptSubmit(context.Background(), nil, "hi")
	if err != nil {
		t.Errorf("nil manager should not error; got %v", err)
	}
	if got.Prompt != "hi" {
		t.Errorf("Prompt = %q, want unchanged input 'hi'", got.Prompt)
	}
	if got.Denied {
		t.Errorf("Denied should be false")
	}
}

func TestFirePromptSubmit_AllowReturnsOriginal(t *testing.T) {
	mgr := newTestManager(t, "user_prompt_submit", `echo '{}'`)
	s := New("/x")
	got, _ := s.FirePromptSubmit(context.Background(), mgr, "hello world")
	if got.Prompt != "hello world" {
		t.Errorf("Prompt = %q, want 'hello world'", got.Prompt)
	}
	if got.Denied {
		t.Errorf("Denied = true, want false (no opinion)")
	}
}

func TestFirePromptSubmit_AdditionalContextPrepended(t *testing.T) {
	mgr := newTestManager(t, "user_prompt_submit",
		`echo '{"additionalContext":"Project: opendev-go"}'`)
	s := New("/x")
	got, _ := s.FirePromptSubmit(context.Background(), mgr, "what is this")
	want := "Project: opendev-go\n\nwhat is this"
	if got.Prompt != want {
		t.Errorf("Prompt = %q, want %q", got.Prompt, want)
	}
}

func TestFirePromptSubmit_UpdatedInputReplacesPrompt(t *testing.T) {
	mgr := newTestManager(t, "user_prompt_submit",
		`echo '{"updatedInput":{"prompt":"REWRITTEN"}}'`)
	s := New("/x")
	got, _ := s.FirePromptSubmit(context.Background(), mgr, "original")
	if got.Prompt != "REWRITTEN" {
		t.Errorf("Prompt = %q, want 'REWRITTEN'", got.Prompt)
	}
}

func TestFirePromptSubmit_ContextPrependedToUpdatedInput(t *testing.T) {
	mgr := newTestManager(t, "user_prompt_submit",
		`echo '{"additionalContext":"AUDIT","updatedInput":{"prompt":"NEW"}}'`)
	s := New("/x")
	got, _ := s.FirePromptSubmit(context.Background(), mgr, "original")
	want := "AUDIT\n\nNEW"
	if got.Prompt != want {
		t.Errorf("Prompt = %q, want %q", got.Prompt, want)
	}
}

func TestFirePromptSubmit_InvalidUpdatedInputFallsBackToOriginal(t *testing.T) {
	// Invalid Prompt field shape: a number where a string is
	// expected. Hook output is otherwise valid HookDecision JSON,
	// so manager parses fine; only the UpdatedInput sub-parse
	// fails.
	mgr := newTestManager(t, "user_prompt_submit",
		`echo '{"updatedInput":{"prompt":42}}'`)
	s := New("/x")
	got, err := s.FirePromptSubmit(context.Background(), mgr, "original")
	if err != nil {
		t.Errorf("invalid UpdatedInput should not error; got %v", err)
	}
	if got.Prompt != "original" {
		t.Errorf("Prompt = %q, want fallback to original", got.Prompt)
	}
}

func TestFirePromptSubmit_EmptyUpdatedInputPromptKeepsOriginal(t *testing.T) {
	// updatedInput with empty prompt string should NOT replace —
	// otherwise a typo'd hook silently blanks the user's input.
	mgr := newTestManager(t, "user_prompt_submit",
		`echo '{"updatedInput":{"prompt":""}}'`)
	s := New("/x")
	got, _ := s.FirePromptSubmit(context.Background(), mgr, "original")
	if got.Prompt != "original" {
		t.Errorf("empty rewrite should not blank input; got %q", got.Prompt)
	}
}

func TestFirePromptSubmit_DenySurfacedAsDenied(t *testing.T) {
	mgr := newTestManager(t, "user_prompt_submit",
		`echo '{"permissionDecision":"deny","reason":"contains forbidden term"}'`)
	s := New("/x")
	got, _ := s.FirePromptSubmit(context.Background(), mgr, "secret prompt")
	if !got.Denied {
		t.Errorf("Denied = false, want true")
	}
	if got.Reason != "contains forbidden term" {
		t.Errorf("Reason = %q", got.Reason)
	}
	// Prompt should be the original (binary uses it for display
	// even on deny so the user sees what they tried to submit).
	if got.Prompt != "secret prompt" {
		t.Errorf("Prompt on deny = %q, want unchanged 'secret prompt'", got.Prompt)
	}
}

func TestFireStop_NilManagerNoOp(t *testing.T) {
	s := New("/x")
	// Should not panic.
	s.FireStop(context.Background(), nil, "result", "")
}

func TestFireStop_FiresWithoutError(t *testing.T) {
	mgr := newTestManager(t, "stop", `echo '{}'`)
	s := New("/x")
	// Fire-and-forget; just verify no panic.
	s.FireStop(context.Background(), mgr, "all done", "")
}

func TestFireEnd_NilManagerNoOp(t *testing.T) {
	s := New("/x")
	s.FireEnd(context.Background(), nil, 0.01, 5)
}

func TestFireEnd_FiresWithoutError(t *testing.T) {
	mgr := newTestManager(t, "session_end", `echo '{}'`)
	s := New("/x")
	s.FireEnd(context.Background(), mgr, 0.42, 17)
}
