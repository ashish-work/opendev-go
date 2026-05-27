// Package bash implements the bash / run_command tool — the model's
// way to execute arbitrary shell commands during a session. v1 supports
// foreground execution only: one command runs, blocks for output, and
// returns when it finishes or hits a timeout.
//
// Deferred for later: background processes, danger-pattern detection,
// streaming output, sandboxing. v1 trusts the model not to fork-bomb us
// and assumes a non-hostile local environment.
//
// We invoke `sh -c "command"` rather than splitting the command string
// ourselves, which gives the model pipes, redirects, env vars, and
// command chaining (&&, ||, ;) for free.
package bash

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"time"

	"github.com/ashishgupta/opendev-go/internal/tools"
)

// ToolName is the canonical name the model uses to invoke this tool.
const ToolName = "bash"

// Tunable limits as package vars so tests can override without exposing
// public setters.
var (
	// defaultTimeoutSec is applied when the caller omits timeout_sec.
	// 60s covers most one-off commands without letting hangs linger.
	defaultTimeoutSec = 60

	// maxTimeoutSec is the hard ceiling. The model can request a longer
	// timeout but we clamp here — runaway commands are a worse failure
	// mode than a falsely-short timeout the model can re-run.
	maxTimeoutSec = 600 // 10 minutes

	// maxOutputBytes caps combined stdout+stderr per call. Larger output
	// would dominate the context window; the trailing marker tells the
	// model the result was truncated.
	maxOutputBytes = 50 * 1024 // 50 KB
)

// Tool implements tools.Tool for shell command execution. Stateless,
// safe for concurrent reuse.
type Tool struct{}

// New returns a ready-to-register Tool.
func New() *Tool { return &Tool{} }

// Compile-time assertion that *Tool satisfies tools.Tool.
var _ tools.Tool = (*Tool)(nil)

// Name implements tools.Tool. Stable — used in tool_calls history.
func (t *Tool) Name() string { return ToolName }

// Description tells the model what this tool does and the contract for
// timeout + output truncation so it can plan around limits.
func (t *Tool) Description() string {
	return "Execute a shell command via `sh -c`. Returns combined stdout+stderr. " +
		"Supports pipes, redirects, env vars, and chained commands (&&, ||, ;). " +
		"Defaults to a 60-second timeout; specify `timeout_sec` (up to 600) " +
		"for longer-running commands. Output is truncated at 50 KB."
}

// Schema is the JSON Schema for this tool's parameters, surfaced to the
// model via provider.ToolSchema.Parameters.
func (t *Tool) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"command": {
				"type": "string",
				"description": "Shell command to execute (passed to sh -c)."
			},
			"timeout_sec": {
				"type": "integer",
				"description": "Maximum runtime in seconds (default 60, max 600)."
			}
		},
		"required": ["command"]
	}`)
}

// args is the parsed shape of the model's JSON arguments.
type args struct {
	Command    string `json:"command"`
	TimeoutSec int    `json:"timeout_sec"`
}

// Execute runs the command via `sh -c`, captures combined stdout+stderr,
// and returns a ToolResult. Per the Tool.Execute contract:
//
//   - Tool-domain failures (non-zero exit, timeout, empty command) →
//     ToolResult{Success: false, Output: ..., Error: ...}, nil error.
//   - Infrastructure failures (outer ctx cancellation) → Go error.
//
// The outer ctx (from the loop) wins over our internal timeout — if the
// user Ctrl-Cs, we abort even mid-command.
func (t *Tool) Execute(ctx context.Context, tctx tools.ToolContext, raw json.RawMessage) (tools.ToolResult, error) {
	var a args
	if err := json.Unmarshal(raw, &a); err != nil {
		return failf("invalid arguments: %v", err), nil
	}
	if a.Command == "" {
		return failf("command is required"), nil
	}

	// Normalize timeout: ≤0 → default, > max → clamped.
	timeoutSec := a.TimeoutSec
	if timeoutSec <= 0 {
		timeoutSec = defaultTimeoutSec
	}
	if timeoutSec > maxTimeoutSec {
		timeoutSec = maxTimeoutSec
	}

	runCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSec)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(runCtx, "sh", "-c", a.Command)
	cmd.Dir = tctx.WorkingDir

	rawOutput, runErr := cmd.CombinedOutput()
	output, truncated := truncateOutput(rawOutput)

	meta := map[string]any{
		"command":          a.Command,
		"output_truncated": truncated,
	}
	if code, ok := exitCodeOf(runErr); ok {
		meta["exit_code"] = code
	} else if runErr == nil {
		meta["exit_code"] = 0
	}

	// Outer-ctx cancellation: surface as a Go error so the loop knows
	// to stop, NOT as a tool observation. The user said "stop".
	if errors.Is(ctx.Err(), context.Canceled) {
		return tools.ToolResult{}, ctx.Err()
	}

	// Inner timeout: tool-domain failure. The partial output captured
	// before the kill is still useful to the model.
	if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
		return tools.ToolResult{
			Success:  false,
			Output:   string(output),
			Error:    fmt.Sprintf("command timed out after %ds", timeoutSec),
			Metadata: meta,
		}, nil
	}

	// Non-zero exit: tool-domain failure, output kept for the model.
	if runErr != nil {
		return tools.ToolResult{
			Success:  false,
			Output:   string(output),
			Error:    runErr.Error(),
			Metadata: meta,
		}, nil
	}

	return tools.ToolResult{
		Success:  true,
		Output:   string(output),
		Metadata: meta,
	}, nil
}

// truncateOutput caps b at maxOutputBytes, appending a marker when
// truncation occurred. Returns the (possibly truncated) bytes and the
// truncation flag.
//
// This is the v1 simple cap: it discards everything past the limit.
// A later commit replaces this with a spillover-to-disk approach so
// the model can still get at the full output if needed.
func truncateOutput(b []byte) ([]byte, bool) {
	if len(b) <= maxOutputBytes {
		return b, false
	}
	out := make([]byte, 0, maxOutputBytes+32)
	out = append(out, b[:maxOutputBytes]...)
	out = append(out, []byte("\n...[output truncated]")...)
	return out, true
}

// exitCodeOf pulls the OS exit code out of an exec error. Returns
// (0, false) when err isn't an *exec.ExitError (e.g. command not found,
// timeout, internal failure).
func exitCodeOf(err error) (int, bool) {
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode(), true
	}
	return 0, false
}

// failf builds a Success:false ToolResult with a formatted error.
func failf(format string, args ...any) tools.ToolResult {
	return tools.ToolResult{
		Success: false,
		Error:   fmt.Sprintf(format, args...),
	}
}
