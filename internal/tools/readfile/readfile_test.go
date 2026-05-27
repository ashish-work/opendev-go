package readfile

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ashishgupta/opendev-go/internal/tools"
)

// writeFile is a tiny helper for staging fixture files in t.TempDir.
func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// exec is a shorthand for calling Tool.Execute with a JSON-marshaled
// args struct and a ToolContext with WorkingDir set.
func exec(t *testing.T, workingDir string, a args) tools.ToolResult {
	t.Helper()
	raw, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	result, err := New().Execute(context.Background(), tools.ToolContext{WorkingDir: workingDir}, raw)
	if err != nil {
		t.Fatalf("Execute returned error (should be tool-domain failure instead): %v", err)
	}
	return result
}

func TestReadSmallFile(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "hello.txt"), "line one\nline two\nline three\n")

	got := exec(t, dir, args{Path: "hello.txt"})

	if !got.Success {
		t.Fatalf("Success = false, Error = %q", got.Error)
	}
	want := "     1\tline one\n     2\tline two\n     3\tline three\n"
	if got.Output != want {
		t.Errorf("Output mismatch\n got:  %q\n want: %q", got.Output, want)
	}
	if got.Metadata["total_lines"].(int) != 3 {
		t.Errorf("total_lines = %v, want 3", got.Metadata["total_lines"])
	}
}

func TestReadWithOffset(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "f.txt"), "a\nb\nc\nd\ne\n")

	got := exec(t, dir, args{Path: "f.txt", Offset: 3})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	want := "     3\tc\n     4\td\n     5\te\n"
	if got.Output != want {
		t.Errorf("Output mismatch\n got:  %q\n want: %q", got.Output, want)
	}
}

func TestReadWithLimit(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "f.txt"), "a\nb\nc\nd\ne\n")

	got := exec(t, dir, args{Path: "f.txt", Limit: 2})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	want := "     1\ta\n     2\tb\n"
	if got.Output != want {
		t.Errorf("Output mismatch\n got:  %q\n want: %q", got.Output, want)
	}
}

func TestReadWithOffsetAndLimit(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "f.txt"), "a\nb\nc\nd\ne\n")

	got := exec(t, dir, args{Path: "f.txt", Offset: 2, Limit: 2})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	want := "     2\tb\n     3\tc\n"
	if got.Output != want {
		t.Errorf("Output mismatch\n got:  %q\n want: %q", got.Output, want)
	}
}

func TestEmptyPathReturnsFailResult(t *testing.T) {
	got := exec(t, "", args{Path: ""})
	if got.Success {
		t.Error("Success = true, want false")
	}
	if !strings.Contains(got.Error, "path is required") {
		t.Errorf("Error = %q, want substring %q", got.Error, "path is required")
	}
}

func TestFileNotFound(t *testing.T) {
	dir := t.TempDir()
	got := exec(t, dir, args{Path: "nope.txt"})
	if got.Success {
		t.Error("Success = true, want false")
	}
	if !strings.Contains(got.Error, "file not found") {
		t.Errorf("Error = %q, want substring %q", got.Error, "file not found")
	}
}

func TestPathIsDirectory(t *testing.T) {
	dir := t.TempDir()
	got := exec(t, dir, args{Path: "."})
	if got.Success {
		t.Error("Success = true, want false")
	}
	if !strings.Contains(got.Error, "is a directory") {
		t.Errorf("Error = %q, want substring %q", got.Error, "is a directory")
	}
}

func TestRelativePathResolvesAgainstWorkingDir(t *testing.T) {
	// Stage a file in a temp dir; pass a different cwd as WorkingDir.
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "rel.txt"), "content\n")

	got := exec(t, dir, args{Path: "rel.txt"}) // relative path
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	if !strings.Contains(got.Output, "content") {
		t.Errorf("Output missing expected content; got: %q", got.Output)
	}
}

func TestAbsolutePathIgnoresWorkingDir(t *testing.T) {
	dir := t.TempDir()
	abs := filepath.Join(dir, "abs.txt")
	writeFile(t, abs, "absolute\n")

	// Pass WorkingDir as /tmp/nonexistent; the absolute path should still resolve.
	got := exec(t, "/tmp/nonexistent-dir-xyz", args{Path: abs})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	if !strings.Contains(got.Output, "absolute") {
		t.Errorf("Output missing expected content; got: %q", got.Output)
	}
}

func TestLongLineTruncated(t *testing.T) {
	dir := t.TempDir()
	longLine := strings.Repeat("x", maxLineLength+500)
	writeFile(t, filepath.Join(dir, "long.txt"), longLine+"\n")

	got := exec(t, dir, args{Path: "long.txt"})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	if !strings.Contains(got.Output, "...[line truncated]") {
		t.Error("Output missing line-truncation marker")
	}
}

func TestOutputTruncatedAtTotalCap(t *testing.T) {
	// Override the cap so we don't have to write a 50KB file.
	prev := maxOutputBytes
	maxOutputBytes = 200
	t.Cleanup(func() { maxOutputBytes = prev })

	dir := t.TempDir()
	// 50 lines of 20 chars each ≈ ~1500 bytes after line-number prefixes,
	// well over our 200-byte cap.
	var sb strings.Builder
	for i := 0; i < 50; i++ {
		sb.WriteString(strings.Repeat("y", 20))
		sb.WriteString("\n")
	}
	writeFile(t, filepath.Join(dir, "big.txt"), sb.String())

	got := exec(t, dir, args{Path: "big.txt"})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	if !strings.Contains(got.Output, "...[output truncated]") {
		t.Error("Output missing total-output-truncation marker")
	}
	if got.Metadata["truncated"] != true {
		t.Errorf("Metadata[truncated] = %v, want true", got.Metadata["truncated"])
	}
}

func TestFileTooLarge(t *testing.T) {
	// Override the cap so we don't have to write a 10MB file.
	prev := maxFileSize
	maxFileSize = 16
	t.Cleanup(func() { maxFileSize = prev })

	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "fat.txt"), strings.Repeat("z", 32))

	got := exec(t, dir, args{Path: "fat.txt"})
	if got.Success {
		t.Error("Success = true, want false (size cap exceeded)")
	}
	if !strings.Contains(got.Error, "too large") {
		t.Errorf("Error = %q, want substring %q", got.Error, "too large")
	}
}

func TestOffsetPastEOF(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "f.txt"), "a\nb\nc\n")

	got := exec(t, dir, args{Path: "f.txt", Offset: 100})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	if got.Output != "" {
		t.Errorf("Output = %q, want empty", got.Output)
	}
}

func TestNegativeOffsetTreatedAsStart(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "f.txt"), "a\nb\n")

	got := exec(t, dir, args{Path: "f.txt", Offset: -5})
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	want := "     1\ta\n     2\tb\n"
	if got.Output != want {
		t.Errorf("Output mismatch\n got:  %q\n want: %q", got.Output, want)
	}
}

func TestInvalidJSONArguments(t *testing.T) {
	bad := json.RawMessage(`{not valid`)
	got, err := New().Execute(context.Background(), tools.ToolContext{}, bad)
	if err != nil {
		t.Fatalf("Execute returned Go error (should be Success: false): %v", err)
	}
	if got.Success {
		t.Error("Success = true, want false")
	}
	if !strings.Contains(got.Error, "invalid arguments") {
		t.Errorf("Error = %q, want substring %q", got.Error, "invalid arguments")
	}
}

func TestSchemaIsValidJSON(t *testing.T) {
	schema := New().Schema()
	var parsed map[string]any
	if err := json.Unmarshal(schema, &parsed); err != nil {
		t.Fatalf("Schema is not valid JSON: %v", err)
	}
	if parsed["type"] != "object" {
		t.Errorf(`Schema["type"] = %v, want "object"`, parsed["type"])
	}
	req, ok := parsed["required"].([]any)
	if !ok {
		t.Fatalf(`Schema["required"] is not a slice: %T`, parsed["required"])
	}
	found := false
	for _, name := range req {
		if name == "path" {
			found = true
		}
	}
	if !found {
		t.Errorf(`Schema "required" missing "path": %v`, req)
	}
}

func TestNameAndDescription(t *testing.T) {
	tool := New()
	if tool.Name() != "read_file" {
		t.Errorf("Name() = %q, want %q", tool.Name(), "read_file")
	}
	if len(tool.Description()) < 20 {
		t.Errorf("Description() too short: %q", tool.Description())
	}
}
