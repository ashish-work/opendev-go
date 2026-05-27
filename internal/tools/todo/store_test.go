package todo

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDefaultPath_UnderHome(t *testing.T) {
	got, err := DefaultPath()
	if err != nil {
		t.Fatalf("DefaultPath err: %v", err)
	}
	home, _ := os.UserHomeDir()
	want := filepath.Join(home, ".opendev", "todos.json")
	if got != want {
		t.Errorf("DefaultPath() = %q, want %q", got, want)
	}
}

func TestStore_LoadMissingReturnsEmpty(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(filepath.Join(dir, "todos.json"))
	st, err := store.Load()
	if err != nil {
		t.Fatalf("Load missing file should not error: %v", err)
	}
	if len(st.Todos) != 0 {
		t.Errorf("missing file should yield empty Todos, got %d", len(st.Todos))
	}
	if st.NextID != 1 {
		t.Errorf("missing file should yield NextID=1, got %d", st.NextID)
	}
}

func TestStore_LoadCorruptErrors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "todos.json")
	if err := os.WriteFile(path, []byte("not json"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	store := NewStore(path)
	if _, err := store.Load(); err == nil {
		t.Errorf("Load on corrupt file should error")
	}
}

func TestStore_SaveThenLoadRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "deep", "nested", "todos.json")
	store := NewStore(path)

	now := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	in := State{
		Todos: []Todo{
			{ID: 1, Title: "a", Status: StatusPending, CreatedAt: now, UpdatedAt: now},
			{ID: 2, Title: "b", Status: StatusCompleted, CreatedAt: now, UpdatedAt: now},
		},
		NextID:    3,
		UpdatedAt: now,
	}
	if err := store.Save(in); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, err := os.Stat(filepath.Dir(path)); err != nil {
		t.Errorf("Save should create parent dirs: %v", err)
	}
	got, err := store.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(got.Todos) != 2 || got.NextID != 3 {
		t.Errorf("roundtrip lost data: %+v", got)
	}
	if got.Todos[0].Title != "a" || got.Todos[1].Status != StatusCompleted {
		t.Errorf("fields not preserved: %+v", got.Todos)
	}
}

func TestStore_SaveProducesValidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "todos.json")
	store := NewStore(path)

	now := time.Now()
	st := State{Todos: []Todo{{ID: 1, Title: "x", Status: StatusPending, CreatedAt: now}}, NextID: 2}
	if err := store.Save(st); err != nil {
		t.Fatalf("Save: %v", err)
	}
	raw, _ := os.ReadFile(path)
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Errorf("saved file is not valid JSON: %v", err)
	}
}

func TestState_WithTodos_AssignsIDsFromNextID(t *testing.T) {
	now := time.Now()
	start := State{NextID: 7}
	in := []Todo{{Title: "first"}, {Title: "second"}}
	got, truncated := start.WithTodos(in, now)

	if truncated != 0 {
		t.Errorf("truncated = %d, want 0", truncated)
	}
	if len(got.Todos) != 2 {
		t.Fatalf("Todos count = %d, want 2", len(got.Todos))
	}
	if got.Todos[0].ID != 7 || got.Todos[1].ID != 8 {
		t.Errorf("IDs = [%d, %d], want [7, 8]", got.Todos[0].ID, got.Todos[1].ID)
	}
	if got.NextID != 9 {
		t.Errorf("NextID = %d, want 9", got.NextID)
	}
	for _, todo := range got.Todos {
		if todo.Status != StatusPending {
			t.Errorf("default status %q, want pending", todo.Status)
		}
		if !todo.CreatedAt.Equal(now) || !todo.UpdatedAt.Equal(now) {
			t.Errorf("timestamps not set on %+v", todo)
		}
	}
}

func TestState_WithTodos_PreservesProvidedStatus(t *testing.T) {
	now := time.Now()
	start := State{NextID: 1}
	in := []Todo{
		{Title: "in-progress one", Status: StatusInProgress},
		{Title: "done one", Status: StatusCompleted},
	}
	got, _ := start.WithTodos(in, now)
	if got.Todos[0].Status != StatusInProgress {
		t.Errorf("provided in_progress status lost: %+v", got.Todos[0])
	}
	if got.Todos[1].Status != StatusCompleted {
		t.Errorf("provided completed status lost: %+v", got.Todos[1])
	}
}

func TestState_WithTodos_Truncates(t *testing.T) {
	now := time.Now()
	in := make([]Todo, MaxTodos+5)
	for i := range in {
		in[i].Title = "x"
	}
	got, truncated := State{NextID: 1}.WithTodos(in, now)
	if truncated != 5 {
		t.Errorf("truncated = %d, want 5", truncated)
	}
	if len(got.Todos) != MaxTodos {
		t.Errorf("kept = %d, want %d", len(got.Todos), MaxTodos)
	}
}

func TestState_WithTodos_AdvancesNextIDAcrossWrites(t *testing.T) {
	now := time.Now()
	first, _ := State{NextID: 1}.WithTodos([]Todo{{Title: "a"}, {Title: "b"}}, now)
	if first.NextID != 3 {
		t.Errorf("first NextID = %d, want 3", first.NextID)
	}
	second, _ := first.WithTodos([]Todo{{Title: "c"}}, now)
	if second.Todos[0].ID != 3 {
		t.Errorf("second batch ID = %d, want 3 (NextID continues across writes)", second.Todos[0].ID)
	}
	if second.NextID != 4 {
		t.Errorf("second NextID = %d, want 4", second.NextID)
	}
}

func TestState_UpdateOne_Status(t *testing.T) {
	now := time.Now()
	st, _ := State{NextID: 1}.WithTodos([]Todo{{Title: "a"}, {Title: "b"}}, now)
	updated, err := st.UpdateOne(2, "", StatusInProgress, now)
	if err != nil {
		t.Fatalf("UpdateOne: %v", err)
	}
	if updated.Todos[1].Status != StatusInProgress {
		t.Errorf("status not updated: %+v", updated.Todos[1])
	}
	// Other todo untouched.
	if updated.Todos[0].Status != StatusPending {
		t.Errorf("other todo changed: %+v", updated.Todos[0])
	}
}

func TestState_UpdateOne_TitleKeepsStatus(t *testing.T) {
	now := time.Now()
	st, _ := State{NextID: 1}.WithTodos([]Todo{{Title: "a", Status: StatusInProgress}}, now)
	updated, err := st.UpdateOne(1, "renamed", "", now)
	if err != nil {
		t.Fatalf("UpdateOne: %v", err)
	}
	if updated.Todos[0].Title != "renamed" {
		t.Errorf("title not updated: %+v", updated.Todos[0])
	}
	if updated.Todos[0].Status != StatusInProgress {
		t.Errorf("status changed when title-only update: %+v", updated.Todos[0])
	}
}

func TestState_UpdateOne_NotFoundErrors(t *testing.T) {
	now := time.Now()
	st, _ := State{NextID: 1}.WithTodos([]Todo{{Title: "a"}}, now)
	_, err := st.UpdateOne(99, "", StatusCompleted, now)
	if err == nil {
		t.Errorf("expected error for missing id")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error = %v, want it to say not found", err)
	}
}

func TestState_UpdateOne_InvalidStatusErrors(t *testing.T) {
	now := time.Now()
	st, _ := State{NextID: 1}.WithTodos([]Todo{{Title: "a"}}, now)
	_, err := st.UpdateOne(1, "", "blocked", now)
	if err == nil {
		t.Errorf("expected error for invalid status")
	}
}

func TestState_CompleteOne(t *testing.T) {
	now := time.Now()
	st, _ := State{NextID: 1}.WithTodos([]Todo{{Title: "a"}}, now)
	got, err := st.CompleteOne(1, now)
	if err != nil {
		t.Fatalf("CompleteOne: %v", err)
	}
	if got.Todos[0].Status != StatusCompleted {
		t.Errorf("status = %q, want completed", got.Todos[0].Status)
	}
}

func TestState_Cleared_PreservesNextID(t *testing.T) {
	now := time.Now()
	st, _ := State{NextID: 1}.WithTodos([]Todo{{Title: "a"}, {Title: "b"}}, now)
	cleared := st.Cleared(now)
	if len(cleared.Todos) != 0 {
		t.Errorf("Cleared should empty Todos, got %d", len(cleared.Todos))
	}
	if cleared.NextID != st.NextID {
		t.Errorf("Cleared changed NextID: %d → %d", st.NextID, cleared.NextID)
	}
}

func TestState_Counts(t *testing.T) {
	st := State{Todos: []Todo{
		{Status: StatusPending},
		{Status: StatusPending},
		{Status: StatusInProgress},
		{Status: StatusCompleted},
		{Status: StatusCompleted},
		{Status: StatusCompleted},
	}}
	p, ip, c := st.Counts()
	if p != 2 || ip != 1 || c != 3 {
		t.Errorf("Counts() = (%d, %d, %d), want (2, 1, 3)", p, ip, c)
	}
}

func TestState_Render_Empty(t *testing.T) {
	out := State{}.Render()
	if !strings.Contains(out, "(empty)") {
		t.Errorf("empty render should mention empty: %q", out)
	}
}

func TestState_Render_FullList(t *testing.T) {
	now := time.Now()
	st, _ := State{NextID: 1}.WithTodos([]Todo{
		{Title: "first"},
		{Title: "second", Status: StatusInProgress},
		{Title: "third", Status: StatusCompleted},
	}, now)
	out := st.Render()
	for _, want := range []string{
		"3 total",
		"1 completed",
		"1 in_progress",
		"1 pending",
		"[ ] 1. first",
		"[~] 2. second",
		"[x] 3. third",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\n%s", want, out)
		}
	}
}

// TestState_ImmutableUpdate ensures every State method returns a new
// State rather than mutating the receiver — the immutability invariant
// the package relies on.
func TestState_ImmutableUpdate(t *testing.T) {
	now := time.Now()
	original, _ := State{NextID: 1}.WithTodos([]Todo{{Title: "a"}}, now)
	_, err := original.UpdateOne(1, "renamed", StatusInProgress, now)
	if err != nil {
		t.Fatalf("UpdateOne: %v", err)
	}
	if original.Todos[0].Title != "a" {
		t.Errorf("UpdateOne mutated receiver: original.Todos[0].Title = %q", original.Todos[0].Title)
	}
	if original.Todos[0].Status != StatusPending {
		t.Errorf("UpdateOne mutated receiver status: %q", original.Todos[0].Status)
	}
}
