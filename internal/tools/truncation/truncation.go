// Package truncation handles oversized tool output by spilling the full
// content to disk and returning a preview + file path to the model.
//
// When a tool produces more than MaxLines (2000) or MaxBytes (50 KB),
// the full output is saved to a uniquely-named file under
// $HOME/.opendev/tool-output/, and the model receives a preview plus a
// hint pointing at that file. The model can recover any detail it cares
// about via read_file on the returned path — nothing is lost, but the
// conversation history stays bounded.
//
// This is structurally different from the rule-based summarizer in
// internal/agents/summarize, which lossily replaces long output with a
// one-liner. Spillover preserves the full content; summarize discards
// it. Both ship: spillover runs at tool-execution time (eager), the
// summary library is held back for a future history compactor (lazy).
package truncation

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Configuration constants.
const (
	// MaxLines is the line cap before truncation kicks in.
	MaxLines = 2000

	// MaxBytes is the byte cap before truncation kicks in (50 KB).
	MaxBytes = 50 * 1024

	// MaxOverflowBytes caps the overflow file's size at 1 MB so a
	// single tool call cannot write unbounded data to disk. When the
	// raw output exceeds this, the saved file holds head + tail
	// snippets with an "[N bytes omitted]" marker in between.
	MaxOverflowBytes = 1024 * 1024

	// RetentionDays controls how long overflow files survive before
	// CleanupOldFiles sweeps them.
	RetentionDays = 7
)

// Direction selects which end of the text to preserve when truncating.
type Direction int

const (
	// Head keeps the FIRST N lines (most common — bash, file dumps).
	Head Direction = iota

	// Tail keeps the LAST N lines (useful for log tails).
	Tail
)

// Result reports the outcome of a Truncate call.
type Result struct {
	// Content is what the model sees: the raw text if no truncation,
	// or a preview + hint if truncated.
	Content string

	// Truncated is true when the input exceeded the limits.
	Truncated bool

	// OutputPath is the absolute path to the overflow file containing
	// the full text. Empty when there was no truncation, OR when
	// writing the overflow file failed (we still return the preview;
	// the model just won't have a recovery path).
	OutputPath string
}

// OutputDir returns the directory where overflow files are stored.
// Default: $HOME/.opendev/tool-output/. Falls back to a sub-directory
// of os.TempDir() when $HOME is unavailable (rare).
func OutputDir() string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return filepath.Join(os.TempDir(), "opendev-tool-output")
	}
	return filepath.Join(home, ".opendev", "tool-output")
}

// Truncate inspects text against maxLines + maxBytes limits. If it
// fits within both, returns the text verbatim. Otherwise spills the
// full text to disk and returns a preview + hint.
//
// Pass 0 (or negative) for maxLines/maxBytes to use package defaults.
//
// Examples:
//
//	r := Truncate("hello\n", 0, 0, Head)
//	// r.Truncated == false, r.Content == "hello\n", r.OutputPath == ""
//
//	r := Truncate(big, 0, 0, Head)
//	// r.Truncated == true,
//	// r.Content   == "<preview>\n\n...N lines truncated...\n\n[Full output saved to: /path]"
//	// r.OutputPath == "/Users/x/.opendev/tool-output/tool_1735.._a1b2c3"
func Truncate(text string, maxLines, maxBytes int, direction Direction) Result {
	if maxLines <= 0 {
		maxLines = MaxLines
	}
	if maxBytes <= 0 {
		maxBytes = MaxBytes
	}

	lines := splitLines(text)
	totalBytes := len(text)

	// Fast path: under both limits, no work needed.
	if len(lines) <= maxLines && totalBytes <= maxBytes {
		return Result{Content: text}
	}

	kept, bytesKept, hitBytes := collectLines(lines, maxLines, maxBytes, direction)

	removed := len(lines) - len(kept)
	unit := "lines"
	if hitBytes {
		removed = totalBytes - bytesKept
		unit = "bytes"
	}

	preview := strings.Join(kept, "\n")
	outputPath := spillToDisk(text)

	hint := buildHint(outputPath)
	content := assemble(preview, removed, unit, hint, direction)

	return Result{
		Content:    content,
		Truncated:  true,
		OutputPath: outputPath,
	}
}

// CleanupOldFiles deletes files in OutputDir() that are older than
// RetentionDays. Idempotent; silent when the directory doesn't exist.
// Call from main.go startup to keep disk usage bounded.
func CleanupOldFiles() {
	dir := OutputDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return // dir doesn't exist yet — nothing to clean
	}
	cutoff := time.Now().Add(-time.Duration(RetentionDays) * 24 * time.Hour)
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "tool_") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if info.ModTime().Before(cutoff) {
			_ = os.Remove(filepath.Join(dir, e.Name()))
		}
	}
}

// splitLines splits on \n and strips a single trailing \n so "a\n"
// yields ["a"] not ["a", ""]. This matches typical "line count"
// semantics where a trailing newline does not add an empty line.
func splitLines(text string) []string {
	if text == "" {
		return nil
	}
	s := strings.TrimSuffix(text, "\n")
	return strings.Split(s, "\n")
}

// collectLines walks lines per direction, accumulating until either
// the line cap or byte cap is hit. Returns the kept lines (in original
// order), the bytes they account for (including \n separators), and
// whether the byte cap fired.
func collectLines(lines []string, maxLines, maxBytes int, direction Direction) (kept []string, bytes int, hitBytes bool) {
	switch direction {
	case Head:
		for i, line := range lines {
			if i >= maxLines {
				break
			}
			extra := 0
			if i > 0 {
				extra = 1 // \n separator before this line
			}
			lineBytes := len(line) + extra
			if bytes+lineBytes > maxBytes {
				hitBytes = true
				break
			}
			kept = append(kept, line)
			bytes += lineBytes
		}
	case Tail:
		// Walk from the end; keep until cap hit; then reverse.
		var reversed []string
		for idx := 0; idx < len(lines); idx++ {
			i := len(lines) - 1 - idx
			if idx >= maxLines {
				break
			}
			extra := 0
			if idx > 0 {
				extra = 1
			}
			lineBytes := len(lines[i]) + extra
			if bytes+lineBytes > maxBytes {
				hitBytes = true
				break
			}
			reversed = append(reversed, lines[i])
			bytes += lineBytes
		}
		// Reverse back into chronological order.
		kept = make([]string, len(reversed))
		for j, line := range reversed {
			kept[len(reversed)-1-j] = line
		}
	}
	return kept, bytes, hitBytes
}

// buildHint formats the "look here for the full output" message that
// gets appended to (or prepended around) the preview. Falls back to a
// shorter message when we failed to write the overflow file.
func buildHint(outputPath string) string {
	if outputPath == "" {
		return "The tool call succeeded but the output was truncated."
	}
	return fmt.Sprintf(
		"The tool call succeeded but the output was truncated. "+
			"Full output saved to: %s\n"+
			"Use read_file with offset/limit to view specific sections.",
		outputPath,
	)
}

// assemble joins preview + truncation marker + hint into the final
// Content string. Marker placement depends on direction (after preview
// for Head, before preview for Tail).
func assemble(preview string, removed int, unit, hint string, direction Direction) string {
	switch direction {
	case Tail:
		return fmt.Sprintf("...%d %s truncated...\n\n%s\n\n%s", removed, unit, hint, preview)
	default: // Head
		return fmt.Sprintf("%s\n\n...%d %s truncated...\n\n%s", preview, removed, unit, hint)
	}
}

// spillToDisk creates OutputDir() (if needed) and writes a uniquely-
// named file with the full text. Returns "" on any failure — caller
// proceeds without an output path.
func spillToDisk(text string) string {
	dir := OutputDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return ""
	}
	path := filepath.Join(dir, newOverflowFilename())

	body := text
	if len(body) > MaxOverflowBytes {
		// Even the overflow file has a cap; head 75% + tail 25% + marker.
		head := MaxOverflowBytes * 3 / 4
		tail := MaxOverflowBytes - head
		omitted := len(body) - head - tail
		body = fmt.Sprintf(
			"%s\n\n[... %d bytes omitted from overflow file ...]\n\n%s",
			body[:head],
			omitted,
			body[len(body)-tail:],
		)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		return ""
	}
	return path
}

// newOverflowFilename returns "tool_<unix-ms>_<8-hex>" — millisecond
// timestamp + random suffix gives effectively zero collision risk
// while keeping names sortable in directory listings.
func newOverflowFilename() string {
	ms := time.Now().UnixMilli()
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		// Fall back to a non-random suffix; collisions still unlikely
		// at millisecond granularity.
		return fmt.Sprintf("tool_%d_fallback", ms)
	}
	return fmt.Sprintf("tool_%d_%s", ms, hex.EncodeToString(b[:]))
}
