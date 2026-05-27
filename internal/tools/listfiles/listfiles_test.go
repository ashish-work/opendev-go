package listfiles

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ashish-work/opendev-go/internal/tools"
)

func TestTool_Surface(t *testing.T) {
	tool := New()
	if tool.Name() != "list_files" {
		t.Errorf("Name() = %q, want list_files", tool.Name())
	}
	if tool.Category() != tools.CategoryRead {
		t.Errorf("Category() = %v, want CategoryRead", tool.Category())
	}
	var schema map[string]any
	if err := json.Unmarshal(tool.Schema(), &schema); err != nil {
		t.Fatalf("Schema() returned invalid JSON: %v", err)
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties object")
	}
	for _, key := range []string{"pattern", "path", "max_depth", "ignore", "respect_gitignore", "sort"} {
		if _, has := props[key]; !has {
			t.Errorf("schema missing property %q", key)
		}
	}
}

func TestTool_Execute(t *testing.T) {
	tool := New()
	ctx := context.Background()

	// Helper: build a small fixture tree under a fresh t.TempDir.
	mkTree := func(t *testing.T, files map[string]string) string {
		t.Helper()
		dir := t.TempDir()
		for rel, content := range files {
			full := filepath.Join(dir, rel)
			if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
				t.Fatalf("mkdir: %v", err)
			}
			if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
				t.Fatalf("write %s: %v", rel, err)
			}
		}
		return dir
	}

	t.Run("default pattern lists everything not excluded", func(t *testing.T) {
		dir := mkTree(t, map[string]string{
			"a.go":          "package a",
			"cmd/main.go":   "package main",
			"README.md":     "# hi",
			"node_modules/x.js": "x", // should be excluded
			".git/HEAD":     "ref",   // should be excluded
		})
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path": dir,
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false: %s", res.Error)
		}
		want := []string{"README.md", "a.go", "cmd/main.go"}
		if got := strings.TrimSpace(res.Output); got != strings.Join(want, "\n") {
			t.Errorf("output mismatch\ngot:\n%s\nwant:\n%s", got, strings.Join(want, "\n"))
		}
		if res.Metadata["total_files"] != 3 {
			t.Errorf("total_files = %v, want 3", res.Metadata["total_files"])
		}
	})

	t.Run("glob filters by extension recursively", func(t *testing.T) {
		dir := mkTree(t, map[string]string{
			"a.go":              "",
			"cmd/main.go":       "",
			"cmd/sub/deep.go":   "",
			"README.md":         "",
		})
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path":    dir,
			"pattern": "**/*.go",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false: %s", res.Error)
		}
		want := []string{"a.go", "cmd/main.go", "cmd/sub/deep.go"}
		if got := strings.TrimSpace(res.Output); got != strings.Join(want, "\n") {
			t.Errorf("output mismatch\ngot:\n%s\nwant:\n%s", got, strings.Join(want, "\n"))
		}
	})

	t.Run("top-level glob does NOT recurse", func(t *testing.T) {
		dir := mkTree(t, map[string]string{
			"a.go":            "",
			"sub/b.go":        "",
		})
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path":    dir,
			"pattern": "*.go",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if got := strings.TrimSpace(res.Output); got != "a.go" {
			t.Errorf("output = %q, want %q", got, "a.go")
		}
	})

	t.Run("max_depth caps recursion", func(t *testing.T) {
		dir := mkTree(t, map[string]string{
			"a.go":              "",
			"sub/b.go":          "",
			"sub/deep/c.go":     "",
		})
		zero := 0
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path":      dir,
			"pattern":   "**/*.go",
			"max_depth": zero,
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if got := strings.TrimSpace(res.Output); got != "a.go" {
			t.Errorf("max_depth=0 should yield only top-level. output = %q", got)
		}
	})

	t.Run("user ignore patterns skip files and dirs", func(t *testing.T) {
		dir := mkTree(t, map[string]string{
			"a.log":             "",
			"main.go":           "",
			"tmp/throwaway.txt": "",
			"keep/important.txt": "",
		})
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path":   dir,
			"ignore": []any{"*.log", "tmp/"},
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		out := strings.TrimSpace(res.Output)
		if strings.Contains(out, "a.log") {
			t.Errorf("a.log should be ignored; got:\n%s", out)
		}
		if strings.Contains(out, "tmp/throwaway.txt") {
			t.Errorf("tmp/ should be ignored; got:\n%s", out)
		}
		if !strings.Contains(out, "main.go") || !strings.Contains(out, "keep/important.txt") {
			t.Errorf("expected main.go and keep/important.txt in output:\n%s", out)
		}
	})

	t.Run("respects .gitignore by default", func(t *testing.T) {
		dir := mkTree(t, map[string]string{
			".gitignore":     "*.log\ntemp/\n",
			"main.go":        "",
			"app.log":        "",
			"temp/cache.txt": "",
			"keep.txt":       "",
		})
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path": dir,
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		out := strings.TrimSpace(res.Output)
		if strings.Contains(out, "app.log") || strings.Contains(out, "temp/cache.txt") {
			t.Errorf("gitignored files leaked into output:\n%s", out)
		}
		if !strings.Contains(out, "main.go") || !strings.Contains(out, "keep.txt") {
			t.Errorf("expected main.go and keep.txt; got:\n%s", out)
		}
		// .gitignore itself isn't gitignored, so it should appear.
		if !strings.Contains(out, ".gitignore") {
			t.Errorf("expected .gitignore in output (it's not in itself); got:\n%s", out)
		}
	})

	t.Run("respect_gitignore=false includes ignored files", func(t *testing.T) {
		dir := mkTree(t, map[string]string{
			".gitignore": "*.log\n",
			"app.log":    "",
			"main.go":    "",
		})
		f := false
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path":              dir,
			"respect_gitignore": f,
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		out := strings.TrimSpace(res.Output)
		if !strings.Contains(out, "app.log") {
			t.Errorf("app.log should appear when gitignore is disabled; got:\n%s", out)
		}
	})

	t.Run("default file glob excludes skip *.min.js etc", func(t *testing.T) {
		dir := mkTree(t, map[string]string{
			"main.go":        "",
			"vendor.min.js":  "",
			"deps.bundle.js": "",
			"app.pyc":        "",
		})
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path":    dir,
			"pattern": "**",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		out := strings.TrimSpace(res.Output)
		for _, bad := range []string{"vendor.min.js", "deps.bundle.js", "app.pyc"} {
			if strings.Contains(out, bad) {
				t.Errorf("%s should be excluded by default; got:\n%s", bad, out)
			}
		}
		if !strings.Contains(out, "main.go") {
			t.Errorf("main.go should be present; got:\n%s", out)
		}
	})

	t.Run("mtime sort orders newest first", func(t *testing.T) {
		dir := mkTree(t, map[string]string{
			"old.txt":    "",
			"middle.txt": "",
			"new.txt":    "",
		})
		// Set mtimes deliberately. Use distinct seconds so granular FS still
		// orders them.
		now := time.Now()
		_ = os.Chtimes(filepath.Join(dir, "old.txt"), now.Add(-2*time.Hour), now.Add(-2*time.Hour))
		_ = os.Chtimes(filepath.Join(dir, "middle.txt"), now.Add(-1*time.Hour), now.Add(-1*time.Hour))
		_ = os.Chtimes(filepath.Join(dir, "new.txt"), now, now)

		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path": dir,
			"sort": "mtime",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		want := "new.txt\nmiddle.txt\nold.txt"
		if got := strings.TrimSpace(res.Output); got != want {
			t.Errorf("mtime order wrong\ngot:\n%s\nwant:\n%s", got, want)
		}
	})

	t.Run("sort enum rejects unknown value", func(t *testing.T) {
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path": t.TempDir(),
			"sort": "weird",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if res.Success {
			t.Fatalf("expected Success=false for unknown sort")
		}
		if !strings.Contains(res.Error, "sort") {
			t.Errorf("Error = %q, want it to mention sort", res.Error)
		}
	})

	t.Run("missing directory fails cleanly", func(t *testing.T) {
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path": "/this/definitely/does/not/exist/anywhere/123",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if res.Success {
			t.Fatalf("expected Success=false")
		}
		if !strings.Contains(res.Error, "directory not found") {
			t.Errorf("Error = %q, want directory not found", res.Error)
		}
	})

	t.Run("target is file not directory", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "regular.txt")
		_ = os.WriteFile(path, []byte("hi"), 0o644)
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path": path,
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if res.Success {
			t.Fatalf("expected Success=false for non-directory path")
		}
		if !strings.Contains(res.Error, "not a directory") {
			t.Errorf("Error = %q, want it to say not a directory", res.Error)
		}
	})

	t.Run("empty results produce helpful message", func(t *testing.T) {
		dir := t.TempDir()
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path":    dir,
			"pattern": "*.nonexistent",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false: %s", res.Error)
		}
		if !strings.Contains(res.Output, "No files matching") {
			t.Errorf("empty result should include helpful message, got %q", res.Output)
		}
		if res.Metadata["total_files"] != 0 {
			t.Errorf("total_files = %v, want 0", res.Metadata["total_files"])
		}
	})

	t.Run("trailing ** triggers hint", func(t *testing.T) {
		dir := t.TempDir()
		res, _ := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path":    dir,
			"pattern": "subdir/**",
		}))
		if !strings.Contains(res.Output, "Hint") {
			t.Errorf("expected hint about ** semantics, got:\n%s", res.Output)
		}
	})

	t.Run("results truncate at MaxResults", func(t *testing.T) {
		old := MaxResults
		MaxResults = 5
		defer func() { MaxResults = old }()

		dir := t.TempDir()
		for i := 0; i < 12; i++ {
			_ = os.WriteFile(filepath.Join(dir, "f"+string(rune('a'+i))+".txt"), []byte("x"), 0o644)
		}
		res, err := tool.Execute(ctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path": dir,
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false: %s", res.Error)
		}
		if res.Metadata["truncated"] != true {
			t.Errorf("truncated metadata = %v, want true", res.Metadata["truncated"])
		}
		if !strings.Contains(res.Output, "truncated at 5") {
			t.Errorf("expected truncation footer, got:\n%s", res.Output)
		}
	})

	t.Run("cancelled context returns ctx error", func(t *testing.T) {
		dir := t.TempDir()
		// Create a deeper tree so the walk has work to do.
		for i := 0; i < 8; i++ {
			sub := filepath.Join(dir, "a", "b", "c", "d", "e", "f")
			_ = os.MkdirAll(sub, 0o755)
			_ = os.WriteFile(filepath.Join(sub, "x.go"), []byte("x"), 0o644)
		}
		cctx, cancel := context.WithCancel(ctx)
		cancel()
		_, err := tool.Execute(cctx, tools.ToolContext{}, mkArgs(t, map[string]any{
			"path": dir,
		}))
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	})

	t.Run("relative path resolves against WorkingDir", func(t *testing.T) {
		dir := t.TempDir()
		_ = os.MkdirAll(filepath.Join(dir, "sub"), 0o755)
		_ = os.WriteFile(filepath.Join(dir, "sub", "x.txt"), []byte("x"), 0o644)

		res, err := tool.Execute(ctx, tools.ToolContext{WorkingDir: dir}, mkArgs(t, map[string]any{
			"path":    "sub",
			"pattern": "**",
		}))
		if err != nil {
			t.Fatalf("Execute err: %v", err)
		}
		if !res.Success {
			t.Fatalf("Success=false: %s", res.Error)
		}
		if strings.TrimSpace(res.Output) != "x.txt" {
			t.Errorf("output = %q, want x.txt", res.Output)
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

// mkArgs marshals a map to JSON; t.Fatal on error so test bodies stay
// focused on assertions.
func mkArgs(t *testing.T, m map[string]any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return json.RawMessage(b)
}
