package hooks

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"time"
)

// DefaultHookTimeout is the per-hook wall-clock budget when neither
// the Executor nor the HookMatcher specifies one. 30 seconds is
// generous enough for hooks that hit the network (audit servers,
// telemetry endpoints) and short enough that a hung hook doesn't
// stall the agent forever.
const DefaultHookTimeout = 30 * time.Second

// maxHookOutputBytes caps the stdout and stderr buffers per hook.
// A misbehaving hook can't OOM the agent; excess is silently
// dropped. 64 KB is generous for the JSON-decision use case (a
// decision is typically <1 KB) and for diagnostic stderr.
const maxHookOutputBytes = 64 * 1024

// shellPath is the interpreter used to run hook commands. sh -c lets
// the user's command use quoting, env-var expansion, pipelines, and
// redirection the way every other hook system in the world does.
// Unix-only — same platform constraint as the bash tool.
const shellPath = "/bin/sh"

// EnvWorkingDirKey is the env var name that carries the project
// working dir into each hook. Hooks that want it without parsing
// the JSON payload can read $OPENDEV_WORKING_DIR.
const EnvWorkingDirKey = "OPENDEV_WORKING_DIR"

// EnvHookEventKey is the env var name that carries the firing
// event into each hook. Lets a single command get reused across
// multiple events with `case "$OPENDEV_HOOK_EVENT" in ...`.
const EnvHookEventKey = "OPENDEV_HOOK_EVENT"

// ExecResult is the structured outcome of running one hook. The
// fields separate "the hook ran" (always populated when err is nil)
// from "what it said" (Decision) and "how it ran" (ExitCode,
// Stderr, Duration).
//
// A non-zero ExitCode with err == nil is a normal outcome — the
// hook ran and exited non-zero. The exit-code protocol (#36) maps
// codes to allow/block/log semantics; the executor doesn't
// interpret them.
type ExecResult struct {
	// Decision is the parsed stdout JSON, or zero HookDecision when
	// the hook produced no parseable output (empty stdout, invalid
	// JSON, or just informational text).
	Decision HookDecision

	// ExitCode is the process's exit status. 0 is success; non-zero
	// is the hook saying something the protocol layer interprets.
	// -1 indicates the process didn't run to completion (timeout,
	// signal, spawn failure surfaced as a non-nil error).
	ExitCode int

	// Stderr is the captured stderr output, capped at 64 KB. The
	// manager logs it; not part of the decision flow.
	Stderr string

	// Duration is the wall-clock time the hook took. Useful for
	// telemetry and detecting slow hooks before they hit the
	// timeout.
	Duration time.Duration
}

// Executor runs hook shell commands. Safe for concurrent use across
// goroutines — no shared mutable state.
type Executor struct {
	// WorkingDir is the project root path passed to hooks via
	// $OPENDEV_WORKING_DIR. Empty is fine (the env var is just
	// unset in that case).
	WorkingDir string

	// DefaultTimeout overrides DefaultHookTimeout. Zero falls back
	// to the package constant. Matcher.TimeoutMs (when non-zero)
	// overrides this per call.
	DefaultTimeout time.Duration
}

// NewExecutor returns an Executor with the given working directory
// and the package's default timeout.
func NewExecutor(workingDir string) *Executor {
	return &Executor{WorkingDir: workingDir}
}

// Run executes one hook command. Marshals payload as JSON onto the
// hook's stdin, spawns the command via sh -c, enforces a timeout
// (matcher.TimeoutMs > 0 wins; else Executor.DefaultTimeout; else
// the package constant), and returns the structured outcome.
//
// The error return is reserved for infrastructure failures only —
// payload marshaling failed, the spawn itself failed, the process
// was killed by a signal we didn't send (timeout surfaces as
// context.DeadlineExceeded wrapped here). A hook that exits non-
// zero is a successful run and surfaces via ExecResult.ExitCode.
//
// event is included in env so commands shared across multiple
// events can branch on $OPENDEV_HOOK_EVENT.
func (e *Executor) Run(
	ctx context.Context,
	event HookEvent,
	matcher HookMatcher,
	payload any,
) (*ExecResult, error) {
	stdinJSON, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("hooks: marshal payload: %w", err)
	}

	timeout := e.timeoutFor(matcher)
	cmdCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, shellPath, "-c", matcher.Command)
	cmd.Stdin = bytes.NewReader(stdinJSON)
	cmd.Env = append(os.Environ(),
		EnvHookEventKey+"="+event.String(),
	)
	if e.WorkingDir != "" {
		cmd.Env = append(cmd.Env, EnvWorkingDirKey+"="+e.WorkingDir)
	}

	stdout := &limitedBuffer{limit: maxHookOutputBytes}
	stderr := &limitedBuffer{limit: maxHookOutputBytes}
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	start := time.Now()
	runErr := cmd.Run()
	duration := time.Since(start)

	result := &ExecResult{
		Stderr:   stderr.String(),
		Duration: duration,
		ExitCode: -1,
	}

	if cmd.ProcessState != nil {
		result.ExitCode = cmd.ProcessState.ExitCode()
	}

	// Classify the error:
	//   - Timeout: ctx deadline fired → wrap the deadline error.
	//   - ExitError: the process ran and exited non-zero → NOT an
	//     error from our perspective. ExitCode already captured.
	//   - Other: infrastructure failure (spawn, signal we didn't
	//     send) → surface as the error return.
	if runErr != nil {
		if cmdCtx.Err() == context.DeadlineExceeded {
			return result, fmt.Errorf("hooks: timeout after %s: %w",
				timeout, context.DeadlineExceeded)
		}
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			// Process exited non-zero. That's a normal outcome.
			// Continue to stdout parsing below.
		} else {
			return result, fmt.Errorf("hooks: run %q: %w", matcher.Command, runErr)
		}
	}

	result.Decision = parseDecision(stdout.Bytes(), result.ExitCode, matcher.Command)
	return result, nil
}

// timeoutFor picks the timeout to use for one hook call. Per-matcher
// override wins; then the Executor's DefaultTimeout; then the
// package constant.
func (e *Executor) timeoutFor(matcher HookMatcher) time.Duration {
	if matcher.TimeoutMs > 0 {
		return time.Duration(matcher.TimeoutMs) * time.Millisecond
	}
	if e.DefaultTimeout > 0 {
		return e.DefaultTimeout
	}
	return DefaultHookTimeout
}

// defaultBlockedReason is the reason attached to an exit-2 deny
// when the hook didn't emit a custom one on stdout. Short and
// action-revealing; authors who want better wording emit JSON.
const defaultBlockedReason = "blocked by hook command"

// parseDecision turns a hook's stdout + exit code into a
// HookDecision. The exit code is a secondary control channel:
//
//   - exit 0: parse stdout as HookDecision JSON (existing behavior).
//     Empty stdout or invalid JSON returns the zero HookDecision —
//     hooks that emit log lines instead of decisions don't break
//     the agent.
//
//   - exit 2: block the operation. Override PermissionDecision to
//     deny, regardless of what stdout said. If the JSON parses,
//     keep its AdditionalContext, UpdatedInput, and Reason
//     (so a blocking hook can still attach an explanation and
//     rewrite input). Default Reason to defaultBlockedReason when
//     stdout didn't supply one. Exit code wins over stdout when
//     they disagree — shell scripts more commonly forget to print
//     JSON than forget exit, so the exit code is the authoritative
//     signal.
//
//   - other non-zero: hook itself failed (command-not-found,
//     generic error). Stdout JSON is ignored entirely — its
//     half-baked content shouldn't be interpreted as a decision.
//     Empty HookDecision returned + warning logged so operators see
//     the non-zero in slog.
//
// Timeout and spawn-failure paths return early from Run before this
// function runs, so the exit code here is always >= 0.
func parseDecision(stdoutBytes []byte, exitCode int, command string) HookDecision {
	d := parseStdoutJSON(stdoutBytes, command, exitCode == 0)

	switch exitCode {
	case 0:
		return d
	case 2:
		d.PermissionDecision = PermissionDeny
		if d.Reason == "" {
			d.Reason = defaultBlockedReason
		}
		return d
	default:
		slog.Warn("hooks: command exited non-zero; ignoring output",
			"command", command, "exit_code", exitCode)
		return HookDecision{}
	}
}

// parseStdoutJSON unmarshals stdout into a HookDecision. logWarnings
// controls whether malformed JSON is loud — on the exit-0 path we
// warn (user's hook is misbehaving), on the exit-2 path we don't
// (a JSON-less deny is a feature, not a bug).
func parseStdoutJSON(stdoutBytes []byte, command string, logWarnings bool) HookDecision {
	stdoutBytes = bytes.TrimSpace(stdoutBytes)
	if len(stdoutBytes) == 0 {
		return HookDecision{}
	}
	var d HookDecision
	if err := json.Unmarshal(stdoutBytes, &d); err != nil {
		if logWarnings {
			slog.Warn("hooks: stdout not parseable as HookDecision; ignoring",
				"command", command, "error", err)
		}
		return HookDecision{}
	}
	return d
}

// limitedBuffer is a bytes.Buffer that silently drops writes once
// it hits the limit. Used to cap stdout and stderr so a misbehaving
// hook can't dump gigabytes into the agent's memory.
//
// Implements io.Writer; not safe for concurrent writes (exec.Cmd
// uses a single goroutine per stream).
type limitedBuffer struct {
	buf   bytes.Buffer
	limit int
}

func (lb *limitedBuffer) Write(p []byte) (int, error) {
	remaining := lb.limit - lb.buf.Len()
	if remaining <= 0 {
		// Claim full write so the upstream pipe drains; we just
		// drop the bytes silently.
		return len(p), nil
	}
	if len(p) > remaining {
		_, _ = lb.buf.Write(p[:remaining])
		return len(p), nil
	}
	return lb.buf.Write(p)
}

// Bytes returns the buffered contents.
func (lb *limitedBuffer) Bytes() []byte { return lb.buf.Bytes() }

// String returns the buffered contents as a string.
func (lb *limitedBuffer) String() string { return lb.buf.String() }

// Len returns the current buffer length.
func (lb *limitedBuffer) Len() int { return lb.buf.Len() }
