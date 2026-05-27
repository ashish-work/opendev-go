package todo

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ashish-work/opendev-go/internal/tools"
)

// fixedNow returns a fixed clock for deterministic timestamps in tests.
func fixedNow(s string) func() time.Time {
	tm, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return func() time.Time { return tm }
}

func newTool(t *testing.T) (*Tool, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "todos.json")
	return &Tool{Path: path, Now: fixedNow("2026-01-01T00:00:00Z")}, path
}

func TestTool_Surface(t *testing.T) {
	tool := New()
	if tool.Name() != "todo" {
		t.Errorf("Name() = %q, want todo", tool.Name())
	}
	if tool.Category() != tools.CategoryMeta {
		t.Errorf("Category() = %v, want CategoryMeta", tool.Category())
	}
	var schema map[string]any
	if err := json.Unmarshal(tool.Schema(), &schema); err != nil {
		t.Fatalf("Schema() returned invalid JSON: %v", err)
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties object")
	}
	for _, key := range []string{"action", "todos", "id", "title", "status"} {
		if _, has := props[key]; !has {
			t.Errorf("schema missing property %q", key)
		}
	}
	required, _ := schema["required"].([]any)
	if len(required) != 1 || required[0] != "action" {
		t.Errorf("required = %v, want [action]", required)
	}
}

func TestTool_Execute_Actions(t *testing.T) {
	ctx := context.Background()

	t.Run("list on missing file is empty", func(t *testing.T) {
		tool, _ := newTool(t)
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "list",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false: %s", res.Error)
		}
		if !strings.Contains(res.Output, "(empty)") {
			t.Errorf("expected empty marker, got: %q", res.Output)
		}
		if res.Metadata["total"] != 0 {
			t.Errorf("total = %v, want 0", res.Metadata["total"])
		}
	})

	t.Run("write then list returns the list", func(t *testing.T) {
		tool, path := newTool(t)
		_, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write",
			"todos":  []any{"first", "second", "third"},
		}))
		if err != nil {
			t.Fatalf("write err: %v", err)
		}
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "list",
		}))
		if err != nil {
			t.Fatalf("list err: %v", err)
		}
		for _, want := range []string{"[ ] 1. first", "[ ] 2. second", "[ ] 3. third"} {
			if !strings.Contains(res.Output, want) {
				t.Errorf("output missing %q\n%s", want, res.Output)
			}
		}
		if res.Metadata["total"] != 3 {
			t.Errorf("total = %v, want 3", res.Metadata["total"])
		}
		// File was actually written.
		if _, err := os.Stat(path); err != nil {
			t.Errorf("write should have created %s: %v", path, err)
		}
	})

	t.Run("write accepts object items with status", func(t *testing.T) {
		tool, _ := newTool(t)
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write",
			"todos": []any{
				map[string]any{"title": "in-progress one", "status": "in_progress"},
				map[string]any{"title": "done one", "status": "completed"},
				"a plain string one",
			},
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false: %s", res.Error)
		}
		if !strings.Contains(res.Output, "[~] 1. in-progress one") {
			t.Errorf("in_progress marker missing\n%s", res.Output)
		}
		if !strings.Contains(res.Output, "[x] 2. done one") {
			t.Errorf("completed marker missing\n%s", res.Output)
		}
		if !strings.Contains(res.Output, "[ ] 3. a plain string one") {
			t.Errorf("default pending marker missing\n%s", res.Output)
		}
	})

	t.Run("write rejects invalid status", func(t *testing.T) {
		tool, _ := newTool(t)
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write",
			"todos":  []any{map[string]any{"title": "x", "status": "blocked"}},
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if res.Success {
			t.Fatalf("expected Success=false")
		}
		if !strings.Contains(res.Error, "invalid status") {
			t.Errorf("Error = %q, want invalid status", res.Error)
		}
	})

	t.Run("write rejects empty title in object", func(t *testing.T) {
		tool, _ := newTool(t)
		res, _ := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write",
			"todos":  []any{map[string]any{"title": "   "}},
		}))
		if res.Success || !strings.Contains(res.Error, "title is required") {
			t.Errorf("expected title-required error, got success=%v err=%q", res.Success, res.Error)
		}
	})

	t.Run("write truncates beyond MaxTodos", func(t *testing.T) {
		tool, _ := newTool(t)
		items := make([]any, MaxTodos+3)
		for i := range items {
			items[i] = "x"
		}
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write",
			"todos":  items,
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false: %s", res.Error)
		}
		if !strings.Contains(res.Output, "3 extra dropped") {
			t.Errorf("expected truncation note: %q", res.Output)
		}
		if res.Metadata["truncated"] != 3 {
			t.Errorf("truncated meta = %v, want 3", res.Metadata["truncated"])
		}
	})

	t.Run("update changes status", func(t *testing.T) {
		tool, _ := newTool(t)
		_, _ = tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write",
			"todos":  []any{"a", "b"},
		}))
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "update",
			"id":     2,
			"status": "in_progress",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false: %s", res.Error)
		}
		if !strings.Contains(res.Output, "[~] 2. b") {
			t.Errorf("update didn't reflect in render: %q", res.Output)
		}
	})

	t.Run("update with no fields fails", func(t *testing.T) {
		tool, _ := newTool(t)
		_, _ = tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write", "todos": []any{"a"},
		}))
		res, _ := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "update",
			"id":     1,
		}))
		if res.Success {
			t.Fatalf("expected Success=false for no-op update")
		}
		if !strings.Contains(res.Error, "at least one of title or status") {
			t.Errorf("Error = %q", res.Error)
		}
	})

	t.Run("update missing id fails", func(t *testing.T) {
		tool, _ := newTool(t)
		res, _ := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "update",
			"status": "in_progress",
		}))
		if res.Success || !strings.Contains(res.Error, "id is required") {
			t.Errorf("expected id-required error, got success=%v err=%q", res.Success, res.Error)
		}
	})

	t.Run("update unknown id fails", func(t *testing.T) {
		tool, _ := newTool(t)
		_, _ = tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write", "todos": []any{"a"},
		}))
		res, _ := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "update",
			"id":     99,
			"status": "completed",
		}))
		if res.Success || !strings.Contains(res.Error, "not found") {
			t.Errorf("expected not-found error, got success=%v err=%q", res.Success, res.Error)
		}
	})

	t.Run("complete marks completed", func(t *testing.T) {
		tool, _ := newTool(t)
		_, _ = tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write", "todos": []any{"a", "b"},
		}))
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "complete",
			"id":     1,
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false: %s", res.Error)
		}
		if !strings.Contains(res.Output, "[x] 1. a") {
			t.Errorf("complete didn't reflect in render: %q", res.Output)
		}
	})

	t.Run("complete final todo announces all done", func(t *testing.T) {
		tool, _ := newTool(t)
		_, _ = tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write", "todos": []any{"only"},
		}))
		res, _ := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "complete", "id": 1,
		}))
		if !strings.Contains(res.Output, "All todos are done") {
			t.Errorf("expected all-done announcement, got: %s", res.Output)
		}
	})

	t.Run("complete missing id fails", func(t *testing.T) {
		tool, _ := newTool(t)
		res, _ := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "complete",
		}))
		if res.Success || !strings.Contains(res.Error, "id is required") {
			t.Errorf("expected id-required error")
		}
	})

	t.Run("clear empties the list but preserves NextID", func(t *testing.T) {
		tool, _ := newTool(t)
		_, _ = tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write", "todos": []any{"a", "b"},
		}))
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "clear",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false: %s", res.Error)
		}
		if res.Metadata["total"] != 0 {
			t.Errorf("total = %v, want 0", res.Metadata["total"])
		}
		if res.Metadata["next_id"] != 3 {
			t.Errorf("next_id = %v, want 3 (preserved across clear)", res.Metadata["next_id"])
		}
		// After clear, write again — new items continue numbering.
		res2, _ := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "write", "todos": []any{"new"},
		}))
		if !strings.Contains(res2.Output, "[ ] 3. new") {
			t.Errorf("post-clear write should start at id=3, got: %s", res2.Output)
		}
	})

	t.Run("missing action fails", func(t *testing.T) {
		tool, _ := newTool(t)
		res, _ := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{}))
		if res.Success || !strings.Contains(res.Error, "action is required") {
			t.Errorf("expected action-required error, got success=%v err=%q", res.Success, res.Error)
		}
	})

	t.Run("unknown action fails", func(t *testing.T) {
		tool, _ := newTool(t)
		res, _ := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "delete",
		}))
		if res.Success || !strings.Contains(res.Error, "unknown action") {
			t.Errorf("expected unknown-action error, got success=%v err=%q", res.Success, res.Error)
		}
	})

	t.Run("invalid JSON arguments fail cleanly", func(t *testing.T) {
		tool, _ := newTool(t)
		res, err := tool.Execute(ctx, tools.ToolContext{}, json.RawMessage(`{not json`))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if res.Success || !strings.Contains(res.Error, "invalid arguments") {
			t.Errorf("Error = %q, want invalid arguments", res.Error)
		}
	})

	t.Run("cancelled context returns error", func(t *testing.T) {
		tool, _ := newTool(t)
		cctx, cancel := context.WithCancel(ctx)
		cancel()
		_, err := tool.Execute(cctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"action": "list",
		}))
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	})
}

// TestTool_PersistsAcrossInstances verifies the file-backed design: a
// second Tool instance pointing at the same path sees the first one's
// writes. This is the property that makes "survives REPL restart"
// real.
func TestTool_PersistsAcrossInstances(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "todos.json")
	now := fixedNow("2026-01-01T00:00:00Z")

	first := &Tool{Path: path, Now: now}
	if _, err := first.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
		"action": "write",
		"todos":  []any{"survives"},
	})); err != nil {
		t.Fatalf("first write: %v", err)
	}

	second := &Tool{Path: path, Now: now}
	res, err := second.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
		"action": "list",
	}))
	if err != nil {
		t.Fatalf("second list: %v", err)
	}
	if !strings.Contains(res.Output, "survives") {
		t.Errorf("second Tool didn't see first's data:\n%s", res.Output)
	}
}

// TestTool_SharedLockSerializesConcurrentWrites verifies that two
// goroutines hammering the same todo file via filelock.For don't
// produce a corrupt JSON state. We fire N parallel writes; afterwards
// the file must parse cleanly and contain one of the candidate lists.
func TestTool_SharedLockSerializesConcurrentWrites(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "todos.json")
	now := fixedNow("2026-01-01T00:00:00Z")
	tool := &Tool{Path: path, Now: now}

	const n = 16
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		i := i
		go func() {
			defer wg.Done()
			_, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
				"action": "write",
				"todos":  []any{"writer-" + string(rune('A'+i))},
			}))
			if err != nil {
				t.Errorf("writer %d err: %v", i, err)
			}
		}()
	}
	wg.Wait()

	store := NewStore(path)
	st, err := store.Load()
	if err != nil {
		t.Fatalf("final Load failed — file likely corrupted: %v", err)
	}
	if len(st.Todos) != 1 {
		t.Errorf("final state should hold one todo (last writer wins), got %d", len(st.Todos))
	}
}

// mkArgs marshals a map to JSON for the Execute call.
func mkArgs(t *testing.T, m map[string]any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return json.RawMessage(b)
}
