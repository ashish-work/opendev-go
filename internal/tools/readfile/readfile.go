// Package readfile implements the read_file tool — the model's primary
// way to inspect file contents during a session. v1 reads a single file
// with optional line offset/limit and returns cat -n style output.
//
// Deferred for later: directory listings, binary-file detection, fuzzy
// file-not-found suggestions. The v1 surface focuses on the happy path
// the ReAct loop needs to close.
package readfile

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ashishgupta/opendev-go/internal/tools"
)

// ToolName is the canonical name the model uses to invoke this tool.
// snake_case for consistency with OpenAI's tool-naming conventions.
const ToolName = "read_file"

// Tunable limits, kept as package-level vars rather than consts so
// tests can override them (without exposing setters in the public API).
var (
	// maxFileSize is the largest file we'll attempt to read. Files
	// past this size error out — large binaries and logs would
	// blow up the context window anyway.
	maxFileSize int64 = 10 * 1024 * 1024 // 10 MB

	// defaultMaxLines is the number of lines returned when the
	// caller omits the limit param. 2000 ≈ a normal source file.
	defaultMaxLines = 2000

	// maxLineLength truncates individual very-long lines (e.g. a
	// minified JS bundle on one line). Marked with a trailing tag.
	maxLineLength = 2000

	// maxOutputBytes caps the total output size to prevent a single
	// read_file call from dominating the model's context window.
	maxOutputBytes = 50 * 1024 // 50 KB
)

// Tool implements tools.Tool for the read_file operation. Stateless —
// safe to share across goroutines and reuse across calls.
type Tool struct{}

// New returns a ready-to-register Tool. Constructor function for
// symmetry with other tool packages and future configuration knobs.
func New() *Tool { return &Tool{} }

// Compile-time assertion that *Tool satisfies tools.Tool. Catches
// interface drift at build time rather than first runtime call.
var _ tools.Tool = (*Tool)(nil)

// Name implements tools.Tool. Must stay stable — the model echoes
// this name back in ToolCall.Name and history references depend on it.
func (t *Tool) Name() string { return ToolName }

// Description is what the LLM sees in the tools array. Quality matters
// here: vague descriptions degrade tool selection. We tell the model
// what the tool does, what its inputs mean, and what its output looks
// like (line-numbered prefix) so it can parse responses correctly.
func (t *Tool) Description() string {
	return "Read the contents of a file from the local filesystem. " +
		"Returns the file contents prefixed with line numbers (cat -n style). " +
		"Use 'offset' (1-based) to start at a specific line, and 'limit' to cap " +
		"the number of lines returned. Defaults to the first 2000 lines if not set."
}

// Schema is the JSON Schema describing this tool's parameters. Surfaced
// to the model via provider.ToolSchema.Parameters. Kept as a static
// raw-string literal (faster, easier to read) rather than built at
// runtime from a map.
func (t *Tool) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"path": {
				"type": "string",
				"description": "Path to the file. Absolute, or relative to the working directory."
			},
			"offset": {
				"type": "integer",
				"description": "1-based line number to start reading from. Omit to start at line 1."
			},
			"limit": {
				"type": "integer",
				"description": "Maximum number of lines to return. Omit for default (2000)."
			}
		},
		"required": ["path"]
	}`)
}

// args is the parsed shape of the JSON arguments the model sends. Field
// tags map snake_case JSON keys to Go CamelCase fields.
type args struct {
	Path   string `json:"path"`
	Offset int    `json:"offset"`
	Limit  int    `json:"limit"`
}

// Execute reads the file, applies offset/limit, and returns formatted
// output. Tool-domain failures (path missing, file not found, is a
// directory, too large) return ToolResult{Success: false} with an
// error message for the model to react to — NOT a Go error. Errors
// from this method are reserved for infrastructure failures (ctx
// cancellation, args unmarshal — though args errors also flow as
// Success: false here so the model sees the error as an observation
// and can adjust, rather than the Go caller treating it as fatal).
func (t *Tool) Execute(ctx context.Context, tctx tools.ToolContext, raw json.RawMessage) (tools.ToolResult, error) {
	var a args
	if err := json.Unmarshal(raw, &a); err != nil {
		return failf("invalid arguments: %v", err), nil
	}
	if a.Path == "" {
		return failf("path is required"), nil
	}

	abs := resolvePath(a.Path, tctx.WorkingDir)

	info, err := os.Stat(abs)
	if err != nil {
		if os.IsNotExist(err) {
			return failf("file not found: %s", a.Path), nil
		}
		return failf("stat: %v", err), nil
	}
	if info.IsDir() {
		return failf("%q is a directory, not a file", a.Path), nil
	}
	if info.Size() > maxFileSize {
		return failf("file too large: %d bytes (max %d)", info.Size(), maxFileSize), nil
	}

	// Honor ctx in case we got cancelled between stat and read.
	if err := ctx.Err(); err != nil {
		return tools.ToolResult{}, err
	}

	data, err := os.ReadFile(abs)
	if err != nil {
		return failf("read: %v", err), nil
	}

	output, meta := format(string(data), a.Offset, a.Limit)

	return tools.ToolResult{
		Success:  true,
		Output:   output,
		Metadata: meta,
	}, nil
}

// resolvePath joins relative paths against the tool context's working
// directory; absolute paths pass through unchanged.
func resolvePath(p, workingDir string) string {
	if filepath.IsAbs(p) || workingDir == "" {
		return p
	}
	return filepath.Join(workingDir, p)
}

// format produces the cat -n style output, honoring offset and limit,
// truncating overly long lines, and capping total output at
// maxOutputBytes. Returns (output, metadata).
//
// Normalization rules:
//   - Negative offset or 0 → start at line 1.
//   - Negative or zero limit → use defaultMaxLines.
//   - Offset past EOF → empty output with metadata showing total lines.
func format(content string, offset, limit int) (string, map[string]any) {
	// Split preserves all lines including the last one if it has no
	// trailing newline. We trim a single trailing empty element that
	// strings.Split would produce for files ending in "\n".
	lines := strings.Split(content, "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	total := len(lines)

	start := offset - 1
	if start < 0 {
		start = 0
	}
	if start > total {
		start = total
	}

	if limit <= 0 {
		limit = defaultMaxLines
	}
	end := start + limit
	if end > total {
		end = total
	}

	var b strings.Builder
	truncated := false
	for i := start; i < end; i++ {
		line := lines[i]
		if len(line) > maxLineLength {
			line = line[:maxLineLength] + "...[line truncated]"
		}
		entry := fmt.Sprintf("%6d\t%s\n", i+1, line)

		// Cap total output. We compare BEFORE writing so the marker
		// is the last thing the model sees, not buried mid-output.
		if b.Len()+len(entry) > maxOutputBytes {
			truncated = true
			break
		}
		b.WriteString(entry)
	}
	if truncated {
		b.WriteString("...[output truncated]\n")
	}

	meta := map[string]any{
		"total_lines": total,
		"start_line":  start + 1, // 1-based for the model
		"end_line":    end,
		"truncated":   truncated,
	}
	if start >= total && total > 0 {
		// Offset past EOF — flag so the model knows.
		meta["start_line"] = total + 1
	}
	return b.String(), meta
}

// failf is a small helper to build a Success: false ToolResult.
func failf(format string, args ...any) tools.ToolResult {
	return tools.ToolResult{
		Success: false,
		Error:   fmt.Sprintf(format, args...),
	}
}
