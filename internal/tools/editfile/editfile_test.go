package editfile

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/ashishgupta/opendev-go/internal/tools"
)

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(b)
}

func exec(t *testing.T, workingDir string, a args) tools.ToolResult {
	t.Helper()
	raw, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	result, err := New().Execute(context.Background(), tools.ToolContext{WorkingDir: workingDir}, raw)
	if err != nil {
		t.Fatalf("Execute returned Go error (should be Success: false): %v", err)
	}
	return result
}

// TestUniqueMatchReplaced exercises the single-occurrence happy path.
// Without replace_all, exactly one match must exist — that's the
// disambiguation contract (multi-match without replace_all errors out;
// see TestAmbiguousMatchWithoutReplaceAll).
func TestUniqueMatchReplaced(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	writeFile(t, path, "hello world\ngoodbye world\n")

	got := exec(t, dir, args{
		FilePath:  "f.txt",
		OldString: "hello world", // unique substring
		NewString: "HI world",
	})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	want := "HI world\ngoodbye world\n"
	if actual := readFile(t, path); actual != want {
		t.Errorf("file content = %q, want %q", actual, want)
	}
	if got.Metadata["occurrences"] != 1 {
		t.Errorf("occurrences = %v, want 1", got.Metadata["occurrences"])
	}
}

func TestReplaceAllOccurrences(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	writeFile(t, path, "foo bar foo baz foo\n")

	got := exec(t, dir, args{
		FilePath:   "f.txt",
		OldString:  "foo",
		NewString:  "X",
		ReplaceAll: true,
	})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	want := "X bar X baz X\n"
	if actual := readFile(t, path); actual != want {
		t.Errorf("file content = %q, want %q", actual, want)
	}
	if got.Metadata["occurrences"] != 3 {
		t.Errorf("occurrences = %v, want 3", got.Metadata["occurrences"])
	}
}

func TestOldStringNotFound(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	writeFile(t, path, "hello\n")

	got := exec(t, dir, args{
		FilePath: "f.txt", OldString: "missing", NewString: "x",
	})
	if got.Success {
		t.Error("Success = true, want false")
	}
	if !strings.Contains(got.Error, "not found") {
		t.Errorf("Error = %q, want substring %q", got.Error, "not found")
	}
	if readFile(t, path) != "hello\n" {
		t.Error("file mutated despite failure")
	}
}

func TestAmbiguousMatchWithoutReplaceAll(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	writeFile(t, path, "x x x\n")

	got := exec(t, dir, args{
		FilePath: "f.txt", OldString: "x", NewString: "y",
	})
	if got.Success {
		t.Error("Success = true, want false (ambiguous)")
	}
	if !strings.Contains(got.Error, "3 occurrences") {
		t.Errorf("Error = %q, want substring %q", got.Error, "3 occurrences")
	}
	if readFile(t, path) != "x x x\n" {
		t.Error("file mutated despite ambiguity error")
	}
}

func TestEmptyOldStringRejected(t *testing.T) {
	got := exec(t, "", args{FilePath: "anything", OldString: "", NewString: "x"})
	if got.Success {
		t.Error("Success = true, want false")
	}
	if !strings.Contains(got.Error, "old_string is required") {
		t.Errorf("Error = %q, want substring %q", got.Error, "old_string is required")
	}
}

func TestNoOpEditRejected(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	writeFile(t, path, "hello\n")

	got := exec(t, dir, args{
		FilePath: "f.txt", OldString: "hello", NewString: "hello",
	})
	if got.Success {
		t.Error("Success = true, want false (no-op)")
	}
	if !strings.Contains(got.Error, "no-op") {
		t.Errorf("Error = %q, want substring %q", got.Error, "no-op")
	}
}

func TestFileNotFound(t *testing.T) {
	dir := t.TempDir()
	got := exec(t, dir, args{
		FilePath: "nope.txt", OldString: "x", NewString: "y",
	})
	if got.Success {
		t.Error("Success = true, want false")
	}
	if !strings.Contains(got.Error, "file not found") {
		t.Errorf("Error = %q, want substring %q", got.Error, "file not found")
	}
}

func TestFileTooLarge(t *testing.T) {
	prev := maxFileSize
	maxFileSize = 8
	t.Cleanup(func() { maxFileSize = prev })

	dir := t.TempDir()
	path := filepath.Join(dir, "fat.txt")
	writeFile(t, path, "this is way more than 8 bytes\n")

	got := exec(t, dir, args{FilePath: "fat.txt", OldString: "this", NewString: "that"})
	if got.Success {
		t.Error("Success = true, want false")
	}
	if !strings.Contains(got.Error, "too large") {
		t.Errorf("Error = %q, want substring %q", got.Error, "too large")
	}
}

func TestPathIsDirectory(t *testing.T) {
	dir := t.TempDir()
	got := exec(t, dir, args{FilePath: ".", OldString: "x", NewString: "y"})
	if got.Success {
		t.Error("Success = true, want false")
	}
	if !strings.Contains(got.Error, "is a directory") {
		t.Errorf("Error = %q, want substring %q", got.Error, "is a directory")
	}
}

func TestRelativePathResolvesAgainstWorkingDir(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "rel.txt"), "before\n")

	got := exec(t, dir, args{
		FilePath: "rel.txt", OldString: "before", NewString: "after",
	})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	if actual := readFile(t, filepath.Join(dir, "rel.txt")); actual != "after\n" {
		t.Errorf("file content = %q, want %q", actual, "after\n")
	}
}

func TestPermissionsPreservedOnAtomicWrite(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "perm.txt")
	if err := os.WriteFile(path, []byte("hi\n"), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	got := exec(t, dir, args{
		FilePath: "perm.txt", OldString: "hi", NewString: "hey",
	})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Errorf("permissions = %o, want 0o600 (atomic rename should preserve mode)", info.Mode().Perm())
	}
}

// TestConcurrentEditsToSameFile fires multiple edits at the same file
// in parallel. The per-file mutex must serialize them so no edit is
// lost AND the file ends up in a consistent state (each "FROM" exactly
// replaced once across all goroutines).
func TestConcurrentEditsToSameFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "shared.txt")
	writeFile(t, path, "AAA\nBBB\nCCC\nDDD\nEEE\n")

	// Each goroutine replaces one unique line.
	edits := []struct{ from, to string }{
		{"AAA", "a1"},
		{"BBB", "b1"},
		{"CCC", "c1"},
		{"DDD", "d1"},
		{"EEE", "e1"},
	}

	var wg sync.WaitGroup
	wg.Add(len(edits))
	for _, e := range edits {
		go func(from, to string) {
			defer wg.Done()
			raw, _ := json.Marshal(args{
				FilePath: "shared.txt", OldString: from, NewString: to,
			})
			result, err := New().Execute(context.Background(),
				tools.ToolContext{WorkingDir: dir}, raw)
			if err != nil {
				t.Errorf("Execute %s: %v", from, err)
				return
			}
			if !result.Success {
				t.Errorf("Success = false for %s: %s", from, result.Error)
			}
		}(e.from, e.to)
	}
	wg.Wait()

	// All five edits must have landed. None can have been lost to a race.
	final := readFile(t, path)
	for _, e := range edits {
		if !strings.Contains(final, e.to) {
			t.Errorf("final file missing %q (lost edit?); content: %q", e.to, final)
		}
		if strings.Contains(final, e.from) {
			t.Errorf("final file still contains %q (lost edit?); content: %q", e.from, final)
		}
	}
}

// TestConcurrentEditsToDifferentFiles confirms locks are per-file:
// edits to distinct paths must not block each other. We can't easily
// verify "no blocking" without injecting timing, but we can at least
// confirm they all succeed concurrently without corruption.
func TestConcurrentEditsToDifferentFiles(t *testing.T) {
	dir := t.TempDir()
	const n = 10

	for i := 0; i < n; i++ {
		writeFile(t, filepath.Join(dir, fmtName(i)), "from\n")
	}

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			raw, _ := json.Marshal(args{
				FilePath: fmtName(i), OldString: "from", NewString: "to",
			})
			result, err := New().Execute(context.Background(),
				tools.ToolContext{WorkingDir: dir}, raw)
			if err != nil {
				t.Errorf("Execute %d: %v", i, err)
				return
			}
			if !result.Success {
				t.Errorf("Success = false for file %d: %s", i, result.Error)
			}
		}(i)
	}
	wg.Wait()

	for i := 0; i < n; i++ {
		if got := readFile(t, filepath.Join(dir, fmtName(i))); got != "to\n" {
			t.Errorf("file %d = %q, want %q", i, got, "to\n")
		}
	}
}

// fmtName produces deterministic filenames f00.txt, f01.txt, ...
func fmtName(i int) string {
	return "f" + string(rune('0'+i/10)) + string(rune('0'+i%10)) + ".txt"
}

func TestOutputContainsUnifiedDiffMarkers(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "f.txt"), "before\n")
	got := exec(t, dir, args{FilePath: "f.txt", OldString: "before", NewString: "after"})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	// sergi/go-diff unified output: every changed line prefixed with
	// "-" (removed) or "+" (added).
	if !strings.Contains(got.Output, "-before") {
		t.Errorf("Output missing '-before' marker: %q", got.Output)
	}
	if !strings.Contains(got.Output, "+after") {
		t.Errorf("Output missing '+after' marker: %q", got.Output)
	}
}

func TestSimplePassReportedInMetadata(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	writeFile(t, path, "hello\n")

	got := exec(t, dir, args{FilePath: "f.txt", OldString: "hello", NewString: "world"})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	if got.Metadata["matcher_pass"] != "simple" {
		t.Errorf("matcher_pass = %v, want %q", got.Metadata["matcher_pass"], "simple")
	}
}

func TestInvalidJSONArguments(t *testing.T) {
	bad := json.RawMessage(`{not valid`)
	got, err := New().Execute(context.Background(), tools.ToolContext{}, bad)
	if err != nil {
		t.Fatalf("Execute returned Go error: %v", err)
	}
	if got.Success {
		t.Error("Success = true, want false")
	}
	if !strings.Contains(got.Error, "invalid arguments") {
		t.Errorf("Error = %q, want substring %q", got.Error, "invalid arguments")
	}
}

func TestSchemaIsValidJSON(t *testing.T) {
	var parsed map[string]any
	if err := json.Unmarshal(New().Schema(), &parsed); err != nil {
		t.Fatalf("Schema is not valid JSON: %v", err)
	}
	if parsed["type"] != "object" {
		t.Errorf(`Schema["type"] = %v, want "object"`, parsed["type"])
	}
	req, _ := parsed["required"].([]any)
	wantRequired := map[string]bool{"file_path": false, "old_string": false, "new_string": false}
	for _, name := range req {
		if s, ok := name.(string); ok {
			if _, exists := wantRequired[s]; exists {
				wantRequired[s] = true
			}
		}
	}
	for name, found := range wantRequired {
		if !found {
			t.Errorf(`Schema "required" missing %q: %v`, name, req)
		}
	}
}

func TestNameAndDescription(t *testing.T) {
	tool := New()
	if tool.Name() != "edit_file" {
		t.Errorf("Name() = %q, want %q", tool.Name(), "edit_file")
	}
	if len(tool.Description()) < 30 {
		t.Errorf("Description() too short")
	}
}
