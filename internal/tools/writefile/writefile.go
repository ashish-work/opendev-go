// Package writefile implements the write_file tool — the model's way
// to create a new file or overwrite an existing one with the supplied
// content.
//
// Distinct from edit_file: edit_file is find/replace on an existing
// file and requires a non-empty old_string. write_file is a whole-file
// write — it creates the file (or replaces all of its content). The
// two are complementary, not redundant: use edit_file to change parts
// of a file, write_file to replace or create it.
//
// Concurrency: shares the per-file mutex map with editfile via the
// filelock package, so two concurrent edits/writes to the same path
// serialize and never interleave bytes.
package writefile

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/tools/filelock"
	"github.com/ashish-work/opendev-go/internal/tools/pathutil"
)

// ToolName is the canonical name the model uses to invoke this tool.
const ToolName = "write_file"

// Tunable limit as a package var so tests can override.
var (
	// maxContentSize caps the body to refuse pathological writes.
	// 10 MiB matches the readfile/editfile cap; anything larger would
	// dominate the context window on the next round-trip anyway.
	maxContentSize int = 10 * 1024 * 1024
)

// Tool implements tools.Tool for the write_file operation. Stateless —
// per-file synchronization is delegated to the filelock package.
type Tool struct{}

// New returns a ready-to-register Tool.
func New() *Tool { return &Tool{} }

// Compile-time guards.
var (
	_ tools.Tool        = (*Tool)(nil)
	_ tools.Categorized = (*Tool)(nil)
)

// Name implements tools.Tool. Stable — used in tool_calls history.
func (t *Tool) Name() string { return ToolName }

// Description tells the model the create-or-overwrite semantics and
// when to prefer this over edit_file.
func (t *Tool) Description() string {
	return "Create a new file or overwrite an existing one with the given content. " +
		"Optionally creates parent directories. Atomic: on success, the file is " +
		"fully written; on failure, it is left unchanged. " +
		"Use this for whole-file writes; use edit_file for find/replace within an existing file."
}

// Category implements tools.Categorized.
func (t *Tool) Category() tools.Category { return tools.CategoryWrite }

// Schema is the JSON Schema for the model's tool-call arguments.
func (t *Tool) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"file_path": {
				"type": "string",
				"description": "Path to the file. Absolute, or relative to the working directory."
			},
			"content": {
				"type": "string",
				"description": "Full file content to write. Empty string creates an empty file."
			},
			"create_dirs": {
				"type": "boolean",
				"description": "If true (default), create missing parent directories. If false, fail when the parent directory does not exist."
			}
		},
		"required": ["file_path", "content"]
	}`)
}

type args struct {
	FilePath   string `json:"file_path"`
	Content    string `json:"content"`
	CreateDirs *bool  `json:"create_dirs,omitempty"`
}

// Execute writes the file. Tool-domain failures (parent missing when
// create_dirs=false, target is a directory, etc.) return
// ToolResult{Success:false}. Infrastructure failures (ctx cancellation,
// disk write errors) surface as Go errors.
func (t *Tool) Execute(ctx context.Context, tctx tools.ToolContext, raw json.RawMessage) (tools.ToolResult, error) {
	var a args
	if err := json.Unmarshal(raw, &a); err != nil {
		return failf("invalid arguments: %v", err), nil
	}

	if a.FilePath == "" {
		return failf("file_path is required"), nil
	}
	if len(a.Content) > maxContentSize {
		return failf("content too large: %d bytes (max %d)", len(a.Content), maxContentSize), nil
	}

	// Refuse paths that look like credential/secret files. Detection
	// is filename-based and case-insensitive — id_rsa is treated as
	// a private key whether it sits in ~/.ssh/ or /tmp/. Runs BEFORE
	// any disk work so the file isn't created when the refusal
	// triggers on a path that doesn't yet exist.
	if reason := pathutil.SensitiveReason(a.FilePath); reason != "" {
		return failf(
			"refusing to write to %s: %s — this file likely contains secrets. "+
				"If you really need to modify it, edit it manually.",
			a.FilePath, reason,
		), nil
	}

	createDirs := true
	if a.CreateDirs != nil {
		createDirs = *a.CreateDirs
	}

	abs := resolvePath(a.FilePath, tctx.WorkingDir)
	parent := filepath.Dir(abs)

	// Ctx check before any disk work — cancellation should win over IO.
	if err := ctx.Err(); err != nil {
		return tools.ToolResult{}, err
	}

	if _, err := os.Stat(parent); err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return failf("stat parent: %v", err), nil
		}
		if !createDirs {
			return failf("parent directory does not exist: %s (set create_dirs=true to auto-create)", parent), nil
		}
		if err := os.MkdirAll(parent, 0o755); err != nil {
			return tools.ToolResult{}, fmt.Errorf("write_file: mkdir %s: %w", parent, err)
		}
	}

	// Per-file serialization via the shared lock — coordinates with
	// editfile so two writers to the same path don't race.
	mu := filelock.For(abs)
	mu.Lock()
	defer mu.Unlock()

	// Default mode for new files; preserve mode for existing files.
	mode := os.FileMode(0o644)
	created := true
	if info, err := os.Stat(abs); err == nil {
		if info.IsDir() {
			return failf("%q is a directory, not a file", a.FilePath), nil
		}
		mode = info.Mode().Perm()
		created = false
	} else if !errors.Is(err, os.ErrNotExist) {
		return failf("stat: %v", err), nil
	}

	if err := filelock.AtomicWrite(abs, []byte(a.Content), mode); err != nil {
		return tools.ToolResult{}, fmt.Errorf("write_file: %w", err)
	}

	lines := countLines(a.Content)
	verb := "overwrote"
	if created {
		verb = "created"
	}

	meta := map[string]any{
		"file_path": a.FilePath,
		"bytes":     len(a.Content),
		"lines":     lines,
		"created":   created,
	}

	return tools.ToolResult{
		Success:  true,
		Output:   fmt.Sprintf("%s %s (%d bytes, %d lines)", verb, a.FilePath, len(a.Content), lines),
		Metadata: meta,
	}, nil
}

// resolvePath joins a relative path against the working directory.
// Absolute paths pass through unchanged.
func resolvePath(p, workingDir string) string {
	if filepath.IsAbs(p) || workingDir == "" {
		return p
	}
	return filepath.Join(workingDir, p)
}

// countLines counts the lines in s. An empty string is 0 lines; "a" is
// 1 line; "a\n" is 1 line; "a\nb" is 2 lines. Matches how an editor
// would report line count.
func countLines(s string) int {
	if s == "" {
		return 0
	}
	n := strings.Count(s, "\n")
	if !strings.HasSuffix(s, "\n") {
		n++
	}
	return n
}

// failf builds a Success:false ToolResult with a formatted message.
func failf(format string, args ...any) tools.ToolResult {
	return tools.ToolResult{
		Success: false,
		Error:   fmt.Sprintf(format, args...),
	}
}
