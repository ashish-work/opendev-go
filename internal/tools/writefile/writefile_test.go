package writefile

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/tools/filelock"
)

func TestTool_Surface(t *testing.T) {
	tool := New()
	if got, want := tool.Name(), "write_file"; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
	if got, want := tool.Category(), tools.CategoryWrite; got != want {
		t.Errorf("Category() = %v, want %v", got, want)
	}
	var schema map[string]any
	if err := json.Unmarshal(tool.Schema(), &schema); err != nil {
		t.Fatalf("Schema() returned invalid JSON: %v", err)
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties object")
	}
	for _, key := range []string{"file_path", "content", "create_dirs"} {
		if _, has := props[key]; !has {
			t.Errorf("schema missing property %q", key)
		}
	}
	required, _ := schema["required"].([]any)
	gotReq := map[string]bool{}
	for _, r := range required {
		gotReq[r.(string)] = true
	}
	for _, want := range []string{"file_path", "content"} {
		if !gotReq[want] {
			t.Errorf("required does not contain %q", want)
		}
	}
}

func TestTool_Execute(t *testing.T) {
	tool := New()
	ctx := context.Background()

	t.Run("basic write creates file", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "out.txt")
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"file_path": path,
			"content":   "hello\nworld\n",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false; Error=%q", res.Error)
		}
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		if string(got) != "hello\nworld\n" {
			t.Errorf("content = %q, want %q", got, "hello\nworld\n")
		}
		if res.Metadata["created"] != true {
			t.Errorf("metadata.created = %v, want true", res.Metadata["created"])
		}
		if res.Metadata["bytes"] != len("hello\nworld\n") {
			t.Errorf("metadata.bytes = %v, want %d", res.Metadata["bytes"], len("hello\nworld\n"))
		}
		if res.Metadata["lines"] != 2 {
			t.Errorf("metadata.lines = %v, want 2", res.Metadata["lines"])
		}
	})

	t.Run("overwrite existing preserves mode", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "existing.txt")
		if err := os.WriteFile(path, []byte("old"), 0o600); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"file_path": path,
			"content":   "new",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false; Error=%q", res.Error)
		}
		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("Stat: %v", err)
		}
		if info.Mode().Perm() != 0o600 {
			t.Errorf("mode = %v, want 0600", info.Mode().Perm())
		}
		got, _ := os.ReadFile(path)
		if string(got) != "new" {
			t.Errorf("content = %q, want %q", got, "new")
		}
		if res.Metadata["created"] != false {
			t.Errorf("metadata.created = %v, want false", res.Metadata["created"])
		}
	})

	t.Run("create_dirs=true makes nested parents", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "deep", "nested", "out.txt")
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"file_path":   path,
			"content":     "hi",
			"create_dirs": true,
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false; Error=%q", res.Error)
		}
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		if string(got) != "hi" {
			t.Errorf("content = %q, want %q", got, "hi")
		}
	})

	t.Run("create_dirs=false fails for missing parent", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "missing", "out.txt")
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"file_path":   path,
			"content":     "hi",
			"create_dirs": false,
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if res.Success {
			t.Fatalf("expected Success=false")
		}
		if !strings.Contains(res.Error, "parent directory does not exist") {
			t.Errorf("Error = %q, want it to mention missing parent", res.Error)
		}
		if _, statErr := os.Stat(path); !errors.Is(statErr, os.ErrNotExist) {
			t.Errorf("file should not exist when parent is missing")
		}
	})

	t.Run("empty content creates empty file", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "empty.txt")
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"file_path": path,
			"content":   "",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false; Error=%q", res.Error)
		}
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		if len(got) != 0 {
			t.Errorf("file size = %d, want 0", len(got))
		}
		if res.Metadata["lines"] != 0 {
			t.Errorf("lines = %v, want 0", res.Metadata["lines"])
		}
	})

	t.Run("missing file_path errors", func(t *testing.T) {
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"content": "hi",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if res.Success {
			t.Fatalf("expected Success=false")
		}
		if !strings.Contains(res.Error, "file_path") {
			t.Errorf("Error = %q, want it to mention file_path", res.Error)
		}
	})

	t.Run("target is directory fails", func(t *testing.T) {
		dir := t.TempDir()
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"file_path": dir,
			"content":   "hi",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if res.Success {
			t.Fatalf("expected Success=false")
		}
		if !strings.Contains(res.Error, "directory") {
			t.Errorf("Error = %q, want it to mention directory", res.Error)
		}
	})

	t.Run("relative path resolves against WorkingDir", func(t *testing.T) {
		dir := t.TempDir()
		res, err := tool.Execute(ctx, tools.ToolContext{WorkingDir: dir}, mkArgs(t, map[string]any{
			"file_path": "rel.txt",
			"content":   "x",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false; Error=%q", res.Error)
		}
		got, err := os.ReadFile(filepath.Join(dir, "rel.txt"))
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		if string(got) != "x" {
			t.Errorf("content = %q, want x", got)
		}
	})

	t.Run("cancelled ctx returns error and does not write", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "cancelled.txt")
		cctx, cancel := context.WithCancel(ctx)
		cancel()
		_, err := tool.Execute(cctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"file_path": path,
			"content":   "hi",
		}))
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
		if _, statErr := os.Stat(path); !errors.Is(statErr, os.ErrNotExist) {
			t.Errorf("file should not exist on cancelled write")
		}
	})

	t.Run("content too large fails", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "huge.txt")
		old := maxContentSize
		maxContentSize = 8
		defer func() { maxContentSize = old }()
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"file_path": path,
			"content":   "way more than eight bytes",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if res.Success {
			t.Fatalf("expected Success=false")
		}
		if !strings.Contains(res.Error, "content too large") {
			t.Errorf("Error = %q, want content too large", res.Error)
		}
	})

	t.Run("invalid JSON arguments fail cleanly", func(t *testing.T) {
		res, err := tool.Execute(ctx, tools.ToolContext{}, json.RawMessage(`{not json`))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if res.Success {
			t.Fatalf("expected Success=false")
		}
		if !strings.Contains(res.Error, "invalid arguments") {
			t.Errorf("Error = %q, want invalid arguments", res.Error)
		}
	})
}

// TestSharedLockSerializesConcurrentWrites checks that the per-file
// mutex extracted into the filelock package actually serializes writes
// to the same path. Without the lock, concurrent atomic-writes would
// still produce a valid file (each one renames-over-the-target
// atomically), but the test also verifies filelock.For returns the
// same mutex pointer per path — the contract the writefile and
// editfile packages rely on for correctness once they're touching
// shared files together.
func TestSharedLockSerializesConcurrentWrites(t *testing.T) {
	tool := New()
	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "shared.txt")

	if filelock.For(path) != filelock.For(path) {
		t.Fatalf("filelock.For returned different mutex pointers for the same path — singleton contract broken")
	}

	const n = 32
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		i := i
		go func() {
			defer wg.Done()
			content := strings.Repeat(string(rune('A'+i)), 4096)
			_, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
				"file_path": path,
				"content":   content,
			}))
			if err != nil {
				t.Errorf("write %d err: %v", i, err)
			}
		}()
	}
	wg.Wait()

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if len(got) != 4096 {
		t.Fatalf("final size = %d, want exactly 4096 (one writer's full content, not interleaved)", len(got))
	}
	first := got[0]
	for i, b := range got {
		if b != first {
			t.Fatalf("byte %d = %q, first byte = %q — file content is mixed across writers", i, b, first)
		}
	}
}

// mkArgs marshals a map to JSON. Calls t.Fatal on error so test bodies
// can stay focused on the assertion rather than argument plumbing.
func mkArgs(t *testing.T, m map[string]any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return json.RawMessage(b)
}
