package truncation

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// withTempOutputDir redirects OutputDir() to a t.TempDir() for the
// duration of a test. Returns the path so the test can assert against
// files written there.
func withTempOutputDir(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	return filepath.Join(tmp, ".opendev", "tool-output")
}

func TestTruncate_UnderLimits_PassesThrough(t *testing.T) {
	_ = withTempOutputDir(t)
	short := "hello\nworld\n"
	r := Truncate(short, 0, 0, Head)
	if r.Truncated {
		t.Errorf("Truncated = true, want false for %d bytes", len(short))
	}
	if r.Content != short {
		t.Errorf("Content = %q, want %q", r.Content, short)
	}
	if r.OutputPath != "" {
		t.Errorf("OutputPath = %q, want empty (no overflow needed)", r.OutputPath)
	}
}

func TestTruncate_ExactlyAtLimits_PassesThrough(t *testing.T) {
	_ = withTempOutputDir(t)
	// 5 lines, well under both line and byte caps.
	r := Truncate("a\nb\nc\nd\ne", 5, 100, Head)
	if r.Truncated {
		t.Errorf("Truncated = true at exact line cap, want false")
	}
}

func TestTruncate_OverLineLimit_SpillsAndPreviews(t *testing.T) {
	dir := withTempOutputDir(t)
	// 100 lines, cap at 10.
	lines := make([]string, 100)
	for i := range lines {
		lines[i] = "line " + strings.Repeat("x", 5)
	}
	text := strings.Join(lines, "\n")
	r := Truncate(text, 10, 0, Head)

	if !r.Truncated {
		t.Fatal("Truncated = false, want true")
	}
	if !strings.Contains(r.Content, "90 lines truncated") {
		t.Errorf("Content missing '90 lines truncated': %q", firstN(r.Content, 200))
	}
	if !strings.Contains(r.Content, "Full output saved to:") {
		t.Errorf("Content missing 'Full output saved to:' hint")
	}
	if r.OutputPath == "" {
		t.Fatal("OutputPath empty, want path to overflow file")
	}
	if !strings.HasPrefix(r.OutputPath, dir) {
		t.Errorf("OutputPath = %q, want prefix %q", r.OutputPath, dir)
	}

	// The file should hold the FULL original text.
	got, err := os.ReadFile(r.OutputPath)
	if err != nil {
		t.Fatalf("read overflow file: %v", err)
	}
	if string(got) != text {
		t.Errorf("overflow file content differs: got %d bytes, want %d",
			len(got), len(text))
	}
}

func TestTruncate_OverByteLimit_ReportsBytes(t *testing.T) {
	_ = withTempOutputDir(t)
	// Single long line over the byte cap.
	text := strings.Repeat("x", 1000)
	r := Truncate(text, 0, 100, Head)
	if !r.Truncated {
		t.Fatal("Truncated = false, want true")
	}
	if !strings.Contains(r.Content, "bytes truncated") {
		t.Errorf("Content should report bytes truncation: %q", firstN(r.Content, 200))
	}
}

func TestTruncate_TailDirection(t *testing.T) {
	_ = withTempOutputDir(t)
	// 50 lines numbered 1..50; keep the last 5 with Tail.
	var b strings.Builder
	for i := 1; i <= 50; i++ {
		b.WriteString("line")
		b.WriteString(strings.Repeat("X", 1))
		b.WriteString("\n")
	}
	// Actually let me build distinguishable lines:
	b.Reset()
	for i := 1; i <= 50; i++ {
		if i > 1 {
			b.WriteByte('\n')
		}
		b.WriteString("L")
		b.WriteByte(byte('0' + i%10))
	}
	text := b.String()

	r := Truncate(text, 5, 0, Tail)
	if !r.Truncated {
		t.Fatal("Truncated = false")
	}
	// Tail mode: marker should appear BEFORE the preview.
	markerIdx := strings.Index(r.Content, "truncated...")
	previewIdx := strings.LastIndex(r.Content, "L")
	if markerIdx == -1 || previewIdx == -1 || markerIdx >= previewIdx {
		t.Errorf("Tail layout wrong; expected marker before preview\nContent: %q",
			firstN(r.Content, 300))
	}
	// Preview should end with line 50 (last "L0" in cyclic 0..9 sequence).
	if !strings.HasSuffix(strings.TrimSpace(r.Content), "L0") {
		t.Errorf("Tail preview should end with last line; got tail = %q",
			r.Content[max(0, len(r.Content)-30):])
	}
}

func TestTruncate_HitMaxOverflowBytes_FileItselfTruncated(t *testing.T) {
	_ = withTempOutputDir(t)
	// Build text well over MaxOverflowBytes (1 MB).
	text := strings.Repeat("x", 1_500_000)
	r := Truncate(text, 0, 0, Head)
	if !r.Truncated {
		t.Fatal("Truncated = false")
	}
	if r.OutputPath == "" {
		t.Fatal("OutputPath empty")
	}
	body, err := os.ReadFile(r.OutputPath)
	if err != nil {
		t.Fatalf("read overflow file: %v", err)
	}
	if len(body) > MaxOverflowBytes+200 { // +200 for the omitted-bytes marker
		t.Errorf("overflow file too large: %d bytes (cap=%d)",
			len(body), MaxOverflowBytes)
	}
	if !strings.Contains(string(body), "bytes omitted from overflow file") {
		t.Errorf("overflow file missing the omitted-bytes marker")
	}
}

func TestSplitLines_TrailingNewlineSemantics(t *testing.T) {
	cases := []struct {
		input string
		want  []string
	}{
		{"", nil},
		{"a", []string{"a"}},
		{"a\n", []string{"a"}},
		{"a\nb", []string{"a", "b"}},
		{"a\nb\n", []string{"a", "b"}},
		{"\n", []string{""}},
	}
	for _, c := range cases {
		got := splitLines(c.input)
		if !equalSlices(got, c.want) {
			t.Errorf("splitLines(%q) = %v, want %v", c.input, got, c.want)
		}
	}
}

func TestNewOverflowFilename_FormatAndUnique(t *testing.T) {
	a := newOverflowFilename()
	b := newOverflowFilename()
	if !strings.HasPrefix(a, "tool_") || !strings.HasPrefix(b, "tool_") {
		t.Errorf("names lack tool_ prefix: %q, %q", a, b)
	}
	if a == b {
		t.Errorf("collision: %q == %q", a, b)
	}
}

func TestCleanupOldFiles_RemovesAging(t *testing.T) {
	dir := withTempOutputDir(t)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// File 1: 10 days old → should be removed.
	old := filepath.Join(dir, "tool_old_aaaaaaaa")
	if err := os.WriteFile(old, []byte("x"), 0o644); err != nil {
		t.Fatalf("write old: %v", err)
	}
	tenDaysAgo := time.Now().Add(-10 * 24 * time.Hour)
	if err := os.Chtimes(old, tenDaysAgo, tenDaysAgo); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
	// File 2: just created → should survive.
	fresh := filepath.Join(dir, "tool_fresh_bbbbbbbb")
	if err := os.WriteFile(fresh, []byte("x"), 0o644); err != nil {
		t.Fatalf("write fresh: %v", err)
	}
	// File 3: unrelated name → never touched.
	unrelated := filepath.Join(dir, "not-mine.txt")
	if err := os.WriteFile(unrelated, []byte("x"), 0o644); err != nil {
		t.Fatalf("write unrelated: %v", err)
	}
	if err := os.Chtimes(unrelated, tenDaysAgo, tenDaysAgo); err != nil {
		t.Fatalf("chtimes unrelated: %v", err)
	}

	CleanupOldFiles()

	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Errorf("old file should have been removed, got err = %v", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Errorf("fresh file should survive: %v", err)
	}
	if _, err := os.Stat(unrelated); err != nil {
		t.Errorf("unrelated file should survive: %v", err)
	}
}

func TestCleanupOldFiles_NoDir_Silent(t *testing.T) {
	// HOME points at empty tmp dir; OutputDir doesn't exist.
	_ = withTempOutputDir(t)
	// Should not panic / error / write anywhere.
	CleanupOldFiles()
}

// firstN returns the first n chars of s for friendlier error messages.
func firstN(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func equalSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
