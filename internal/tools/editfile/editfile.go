// Package editfile implements the edit_file tool — the model's way to
// modify file contents via find/replace.
//
// The exact-match path is the foundation; the fuzzy-matcher chain in
// the match subpackage handles whitespace and indentation drift in
// the model's old_string input. This commit wires up the dispatch
// (currently just the Simple pass); later commits add the rest of
// the chain one by one.
//
// Concurrency: a per-file mutex prevents two simultaneous edits to
// the same path from racing. Different files are independently
// lockable — the read/modify/write cycle is fast, but the guarantee
// matters once parallel tool dispatch lands in a later release.
package editfile

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/sergi/go-diff/diffmatchpatch"

	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/tools/editfile/match"
)

// ToolName is the canonical name the model uses to invoke this tool.
const ToolName = "edit_file"

// Tunable limits as package vars so tests can override.
var (
	// maxFileSize matches readfile's cap (10 MB). Same reasoning:
	// larger files would overflow the context window anyway.
	maxFileSize int64 = 10 * 1024 * 1024
)

// fileLocks maps absolute paths to per-file mutexes. sync.Map is the
// right shape: many keys (one per touched file), each touched briefly
// during read/modify/write. LoadOrStore atomically creates the mutex
// the first time a path is edited.
//
// Locks are never deleted — they're cheap (24 bytes) and the set of
// files touched in a session is bounded. Garbage-collecting the map
// would require ref-counting; not worth the complexity for v1.
var fileLocks sync.Map

// lockFor returns the (singleton) mutex for the given absolute path.
// Always paired with Lock/defer Unlock at the call site.
func lockFor(path string) *sync.Mutex {
	actual, _ := fileLocks.LoadOrStore(path, &sync.Mutex{})
	return actual.(*sync.Mutex)
}

// Tool implements tools.Tool for the edit_file operation. Stateless —
// per-file synchronization lives in the package-level fileLocks map.
type Tool struct{}

// New returns a ready-to-register Tool.
func New() *Tool { return &Tool{} }

// Compile-time guard.
var _ tools.Tool = (*Tool)(nil)

// Name implements tools.Tool. Stable — used in tool_calls history.
func (t *Tool) Name() string { return ToolName }

// Description tells the model the find/replace semantics + the
// ambiguity contract (multiple matches without replace_all errors out).
func (t *Tool) Description() string {
	return "Edit a file by finding and replacing exact text. " +
		"old_string must match the file content verbatim (including whitespace). " +
		"If old_string appears more than once and replace_all is false, " +
		"the call fails — include more surrounding context in old_string " +
		"to disambiguate, or set replace_all=true. " +
		"Returns a diff showing what changed."
}

// Schema is the JSON Schema for the model's tool-call arguments.
func (t *Tool) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"file_path": {
				"type": "string",
				"description": "Path to the file. Absolute, or relative to the working directory."
			},
			"old_string": {
				"type": "string",
				"description": "Exact text to find (whitespace must match)."
			},
			"new_string": {
				"type": "string",
				"description": "Replacement text. Empty string deletes old_string."
			},
			"replace_all": {
				"type": "boolean",
				"description": "Replace every occurrence. Default false (first match only; errors on ambiguity)."
			}
		},
		"required": ["file_path", "old_string", "new_string"]
	}`)
}

type args struct {
	FilePath   string `json:"file_path"`
	OldString  string `json:"old_string"`
	NewString  string `json:"new_string"`
	ReplaceAll bool   `json:"replace_all"`
}

// Execute performs the find/replace edit. Tool-domain failures
// (file not found, ambiguous match, no-op edit) return
// ToolResult{Success: false}, not Go errors. Infrastructure failures
// (ctx cancellation, write errors) surface as Go errors.
func (t *Tool) Execute(ctx context.Context, tctx tools.ToolContext, raw json.RawMessage) (tools.ToolResult, error) {
	var a args
	if err := json.Unmarshal(raw, &a); err != nil {
		return failf("invalid arguments: %v", err), nil
	}

	if a.FilePath == "" {
		return failf("file_path is required"), nil
	}
	if a.OldString == "" {
		return failf("old_string is required and must be non-empty (an empty pattern would match everywhere)"), nil
	}
	if a.OldString == a.NewString {
		return failf("no-op edit: old_string and new_string are identical"), nil
	}

	abs := resolvePath(a.FilePath, tctx.WorkingDir)

	info, err := os.Stat(abs)
	if err != nil {
		if os.IsNotExist(err) {
			return failf("file not found: %s", a.FilePath), nil
		}
		return failf("stat: %v", err), nil
	}
	if info.IsDir() {
		return failf("%q is a directory, not a file", a.FilePath), nil
	}
	if info.Size() > maxFileSize {
		return failf("file too large: %d bytes (max %d)", info.Size(), maxFileSize), nil
	}

	// Ctx check before grabbing the lock — cancellation should win
	// over queuing on a busy file.
	if err := ctx.Err(); err != nil {
		return tools.ToolResult{}, err
	}

	// Per-file serialization. Different files don't contend.
	mu := lockFor(abs)
	mu.Lock()
	defer mu.Unlock()

	original, err := os.ReadFile(abs)
	if err != nil {
		return failf("read: %v", err), nil
	}
	content := string(original)

	// Run the matcher chain. If no pass finds the model's old_string
	// (with any tolerated drift), report not-found.
	matchResult, ok := match.Find(content, a.OldString)
	if !ok {
		return failf("old_string not found in %s", a.FilePath), nil
	}

	// `actual` is the substring of the file we'll actually replace —
	// may differ from a.OldString if a fuzzy pass corrected whitespace.
	actual := matchResult.Actual

	// Ambiguity check runs against the corrected `actual` so the count
	// reflects what we'd really replace.
	count := match.CountOccurrences(content, actual)
	if count == 0 {
		// Defensive: matcher said yes, count says no. Shouldn't happen
		// because passes return substrings of `original`. Surface
		// loudly if it ever does.
		return failf("internal: matcher returned a span that does not appear in file"), nil
	}
	if count > 1 && !a.ReplaceAll {
		return failf(
			"%d occurrences of old_string found in %s (matched via pass %q); "+
				"include more surrounding context in old_string to make it unique, "+
				"or set replace_all=true",
			count, a.FilePath, matchResult.PassName,
		), nil
	}

	var newContent string
	var replaced int
	if a.ReplaceAll {
		newContent = strings.ReplaceAll(content, actual, a.NewString)
		replaced = count
	} else {
		newContent = strings.Replace(content, actual, a.NewString, 1)
		replaced = 1
	}

	if err := atomicWrite(abs, []byte(newContent), info.Mode()); err != nil {
		// Write failures are infrastructure errors — partial state
		// could leave the user in a bad place, so surface as a Go
		// error rather than a tool-domain result.
		return tools.ToolResult{}, fmt.Errorf("edit_file: write %s: %w", abs, err)
	}

	meta := map[string]any{
		"file_path":    a.FilePath,
		"occurrences":  replaced,
		"bytes_before": len(original),
		"bytes_after":  len(newContent),
		"replace_all":  a.ReplaceAll,
		"matcher_pass": matchResult.PassName,
	}

	return tools.ToolResult{
		Success:  true,
		Output:   renderDiff(a.FilePath, actual, a.NewString, replaced, matchResult.PassName),
		Metadata: meta,
	}, nil
}

// renderDiff produces a line-oriented unified-style diff of the edit
// using sergi/go-diff. We diff at LINE granularity (not character) so
// the output reads cleanly: every line in the changed region is tagged
// with "-", "+", or " " — the format the model has seen in every git
// diff it was trained on.
//
// The header line names the file, occurrence count, and which matcher
// pass fired (helpful when a fuzzy pass corrected the model's input —
// the model can spot drift in its own behavior).
func renderDiff(path, oldText, newText string, n int, passName string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "edited %s (%d occurrence", path, n)
	if n != 1 {
		b.WriteString("s")
	}
	if passName != "" && passName != "simple" {
		fmt.Fprintf(&b, ", matched via %s)\n", passName)
	} else {
		b.WriteString(")\n")
	}
	b.WriteString(unifiedLineDiff(oldText, newText))
	return b.String()
}

// unifiedLineDiff returns a "-"/"+"/" "-prefixed line-by-line diff
// of (oldText, newText). Uses sergi/go-diff's lines-to-chars trick to
// drop char-level noise from the output.
func unifiedLineDiff(oldText, newText string) string {
	dmp := diffmatchpatch.New()
	encOld, encNew, lineArr := dmp.DiffLinesToChars(oldText, newText)
	diffs := dmp.DiffMain(encOld, encNew, false)
	diffs = dmp.DiffCharsToLines(diffs, lineArr)

	var b strings.Builder
	for _, d := range diffs {
		// Trim only the very last "\n" so we can emit one prefix per
		// line; the underlying lines keep any in-string newlines.
		text := strings.TrimRight(d.Text, "\n")
		if text == "" {
			continue
		}
		var prefix string
		switch d.Type {
		case diffmatchpatch.DiffInsert:
			prefix = "+"
		case diffmatchpatch.DiffDelete:
			prefix = "-"
		case diffmatchpatch.DiffEqual:
			prefix = " "
		}
		for _, line := range strings.Split(text, "\n") {
			b.WriteString(prefix)
			b.WriteString(line)
			b.WriteString("\n")
		}
	}
	return b.String()
}

// resolvePath joins a relative path against the working directory.
// Absolute paths pass through unchanged.
func resolvePath(p, workingDir string) string {
	if filepath.IsAbs(p) || workingDir == "" {
		return p
	}
	return filepath.Join(workingDir, p)
}

// atomicWrite writes data to a temp file in the same directory, then
// renames it over the target. rename(2) is atomic on POSIX within a
// single filesystem, so the file is either fully updated or fully
// preserved — never half-written. mode preserves the original file's
// permission bits.
func atomicWrite(path string, data []byte, mode os.FileMode) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".edit-*")
	if err != nil {
		return fmt.Errorf("create temp: %w", err)
	}
	tmpName := tmp.Name()

	// On any error after the temp is created, remove it.
	cleanup := func() { _ = os.Remove(tmpName) }

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		cleanup()
		return fmt.Errorf("write temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		cleanup()
		return fmt.Errorf("close temp: %w", err)
	}
	if err := os.Chmod(tmpName, mode); err != nil {
		cleanup()
		return fmt.Errorf("chmod temp: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		cleanup()
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}

// failf builds a Success:false ToolResult with a formatted message.
func failf(format string, args ...any) tools.ToolResult {
	return tools.ToolResult{
		Success: false,
		Error:   fmt.Sprintf(format, args...),
	}
}
