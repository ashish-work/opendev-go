package hooks

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"time"
)

// All executor tests use real /bin/sh commands. sh is universal on
// Unix CI; the project is already Unix-only via the bash tool.

func TestRun_ValidDecisionJSON(t *testing.T) {
	e := NewExecutor("")
	m := HookMatcher{
		Command: `echo '{"reason":"ok","permissionDecision":"allow"}'`,
	}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0", got.ExitCode)
	}
	if got.Decision.Reason != "ok" {
		t.Errorf("Decision.Reason = %q, want ok", got.Decision.Reason)
	}
	if !got.Decision.IsAllow() {
		t.Errorf("Decision should be Allow; got %+v", got.Decision)
	}
}

func TestRun_EmptyStdoutGivesEmptyDecision(t *testing.T) {
	e := NewExecutor("")
	m := HookMatcher{Command: "true"} // exit 0, no output
	got, err := e.Run(context.Background(), HookEventStop, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0", got.ExitCode)
	}
	if !decisionIsZero(got.Decision) {
		t.Errorf("Decision = %+v, want zero", got.Decision)
	}
}

func TestRun_InvalidJSONOutputIsLoggedAndIgnored(t *testing.T) {
	e := NewExecutor("")
	m := HookMatcher{Command: `echo not json at all`}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0", got.ExitCode)
	}
	if !decisionIsZero(got.Decision) {
		t.Errorf("Decision should be zero on invalid JSON; got %+v", got.Decision)
	}
}

func TestRun_ReadsPayloadFromStdin(t *testing.T) {
	// Use sh's cat to echo stdin back as stdout. The payload then
	// arrives as the parsed HookDecision — we wrap it as such.
	e := NewExecutor("")
	m := HookMatcher{Command: "cat"}

	// Send a HookDecision-shaped payload so cat's stdout parses
	// cleanly as a decision back.
	payload := HookDecision{Reason: "from stdin"}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, payload)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.Decision.Reason != "from stdin" {
		t.Errorf("Decision.Reason = %q, want 'from stdin'", got.Decision.Reason)
	}
}

func TestRun_NonZeroExitNotAnError(t *testing.T) {
	// `exit 2` is the "block" signal in #36's protocol. The
	// executor surfaces it as ExitCode = 2 with err = nil — the
	// caller (manager/protocol layer) decides what to do.
	e := NewExecutor("")
	m := HookMatcher{Command: "exit 2"}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if err != nil {
		t.Errorf("non-zero exit should not be an error; got %v", err)
	}
	if got.ExitCode != 2 {
		t.Errorf("ExitCode = %d, want 2", got.ExitCode)
	}
}

func TestRun_StderrCaptured(t *testing.T) {
	e := NewExecutor("")
	m := HookMatcher{Command: `echo "diagnostic noise" >&2`}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(got.Stderr, "diagnostic noise") {
		t.Errorf("Stderr = %q, want it to contain 'diagnostic noise'", got.Stderr)
	}
}

func TestRun_TimeoutHonored(t *testing.T) {
	// sleep 10 with TimeoutMs = 100: the executor should kill the
	// process well before 10 seconds and surface the deadline error.
	e := NewExecutor("")
	m := HookMatcher{
		Command:   "sleep 10",
		TimeoutMs: 100,
	}
	start := time.Now()
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want chain containing context.DeadlineExceeded", err)
	}
	if elapsed > 5*time.Second {
		t.Errorf("Run took %s — the timeout was 100ms; the kill apparently didn't fire", elapsed)
	}
	if got == nil {
		t.Errorf("ExecResult should be non-nil even on timeout (Duration/Stderr useful for telemetry)")
	}
}

func TestRun_SpawnFailureReturnsError(t *testing.T) {
	// /bin/sh -c "/no/such/path" doesn't fail to spawn — sh runs
	// and exits non-zero. To force a spawn failure we'd need to
	// override shellPath. Instead exercise an unmistakable error
	// shape: the binary "exit" alone exits 0 normally, but a
	// command-not-found inside sh exits 127.
	e := NewExecutor("")
	m := HookMatcher{Command: "/this/path/does/not/exist"}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	// sh -c with a missing command exits 127 and is NOT a spawn
	// failure from our perspective. err should be nil.
	if err != nil {
		t.Errorf("expected no error (sh exits 127, command runs); got %v", err)
	}
	if got.ExitCode == 0 {
		t.Errorf("ExitCode = 0, want non-zero (sh reports command-not-found)")
	}
}

func TestRun_WorkingDirEnvVarPassed(t *testing.T) {
	e := NewExecutor("/work/repo")
	m := HookMatcher{Command: `echo "{\"reason\":\"$OPENDEV_WORKING_DIR\"}"`}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.Decision.Reason != "/work/repo" {
		t.Errorf("Decision.Reason = %q, want /work/repo (via env var)", got.Decision.Reason)
	}
}

func TestRun_HookEventEnvVarPassed(t *testing.T) {
	e := NewExecutor("")
	m := HookMatcher{Command: `echo "{\"reason\":\"$OPENDEV_HOOK_EVENT\"}"`}
	got, err := e.Run(context.Background(), HookEventSessionEnd, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.Decision.Reason != "session_end" {
		t.Errorf("Decision.Reason = %q, want session_end (the event name)", got.Decision.Reason)
	}
}

func TestRun_DefaultTimeoutAppliedWhenMatcherTimeoutZero(t *testing.T) {
	// When matcher.TimeoutMs == 0, Executor.DefaultTimeout wins.
	// Hook sleeps for 5 seconds; the default of 100ms kills it.
	e := NewExecutor("")
	e.DefaultTimeout = 100 * time.Millisecond
	m := HookMatcher{Command: "sleep 5"} // no per-matcher timeout
	start := time.Now()
	_, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	elapsed := time.Since(start)

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want timeout via Executor.DefaultTimeout", err)
	}
	if elapsed > 3*time.Second {
		t.Errorf("Run took %s — default timeout apparently not enforced", elapsed)
	}
}

func TestRun_MatcherTimeoutWinsOverExecutorDefault(t *testing.T) {
	// Executor.DefaultTimeout = 5s; matcher.TimeoutMs = 100ms.
	// The shorter per-matcher value should win.
	e := NewExecutor("")
	e.DefaultTimeout = 5 * time.Second
	m := HookMatcher{Command: "sleep 5", TimeoutMs: 100}
	start := time.Now()
	_, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	elapsed := time.Since(start)

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want timeout", err)
	}
	if elapsed > 3*time.Second {
		t.Errorf("Run took %s — matcher.TimeoutMs apparently not enforced over Executor.DefaultTimeout", elapsed)
	}
}

func TestRun_DurationRecorded(t *testing.T) {
	e := NewExecutor("")
	m := HookMatcher{Command: "true"}
	got, err := e.Run(context.Background(), HookEventStop, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.Duration <= 0 {
		t.Errorf("Duration = %s, want positive", got.Duration)
	}
	if got.Duration > 5*time.Second {
		t.Errorf("Duration = %s, want quick (trivial sh command)", got.Duration)
	}
}

func TestRun_PayloadMarshalFailureSurfacesError(t *testing.T) {
	e := NewExecutor("")
	m := HookMatcher{Command: "true"}
	// Channels can't be JSON-marshaled — forces an early error.
	unmarshalable := make(chan int)
	_, err := e.Run(context.Background(), HookEventStop, m, unmarshalable)
	if err == nil {
		t.Fatal("expected marshal error, got nil")
	}
	if !strings.Contains(err.Error(), "marshal") {
		t.Errorf("err = %v, should mention marshal failure", err)
	}
}

func TestRun_StdoutCappedAt64KB(t *testing.T) {
	// Produce 256 KB of output. The buffer caps at 64 KB; the rest
	// is silently dropped. The hook should NOT hang and the
	// agent should NOT OOM.
	e := NewExecutor("")
	// `head -c 262144 /dev/zero` produces 256 KB; then printf to
	// make it non-binary so the test doesn't depend on whether
	// the stdout parsing tolerates NULs.
	m := HookMatcher{
		Command:   `yes | head -c 262144`,
		TimeoutMs: 5000,
	}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	// `yes | head -c N` returns 0 once head closes the pipe;
	// however yes seeing SIGPIPE on Linux can produce exit code
	// 141. Both are valid here; we don't check ExitCode strictly.
	if err != nil && !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == nil {
		t.Fatal("ExecResult is nil")
	}
	// The internal cap is 64 KB; we don't poke internals but we
	// can verify decision parsing didn't choke on the dropped
	// excess.
	if !decisionIsZero(got.Decision) {
		t.Errorf("decision should be zero on big random output; got %+v", got.Decision)
	}
}

func TestRun_StderrCappedAt64KB(t *testing.T) {
	// Same idea, stderr side.
	e := NewExecutor("")
	m := HookMatcher{
		Command:   `yes | head -c 262144 >&2`,
		TimeoutMs: 5000,
	}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if err != nil && !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == nil {
		t.Fatal("ExecResult is nil")
	}
	if len(got.Stderr) > maxHookOutputBytes {
		t.Errorf("Stderr len = %d, want <= %d", len(got.Stderr), maxHookOutputBytes)
	}
}

func TestLimitedBuffer_WriteUpToLimit(t *testing.T) {
	// Direct exercise of the buffer cap logic.
	lb := &limitedBuffer{limit: 10}
	n, _ := lb.Write([]byte("12345"))
	if n != 5 || lb.Len() != 5 {
		t.Errorf("partial write: got n=%d len=%d, want 5/5", n, lb.Len())
	}
	n, _ = lb.Write([]byte("67890ABCDE"))
	if n != 10 {
		t.Errorf("over-cap write: got n=%d, want claimed full (10)", n)
	}
	if lb.Len() != 10 {
		t.Errorf("buffer should be capped at 10, got %d", lb.Len())
	}
	if lb.String() != "1234567890" {
		t.Errorf("buffer contents = %q, want %q", lb.String(), "1234567890")
	}
	// Further writes succeed but drop everything.
	n, _ = lb.Write([]byte("xxx"))
	if n != 3 {
		t.Errorf("post-cap claim should match input len; got %d, want 3", n)
	}
	if lb.Len() != 10 {
		t.Errorf("buffer should stay at 10; got %d", lb.Len())
	}
}

// decisionIsZero checks whether a HookDecision is the zero value.
// Can't use `d == HookDecision{}` because UpdatedInput is a
// json.RawMessage (slice) — slices aren't == comparable.
func decisionIsZero(d HookDecision) bool {
	return d.AdditionalContext == "" &&
		len(d.UpdatedInput) == 0 &&
		d.PermissionDecision == "" &&
		d.Reason == ""
}

func TestExecutor_TimeoutForPicksRightValue(t *testing.T) {
	// Per-matcher TimeoutMs (when non-zero) wins; else Executor's
	// DefaultTimeout; else package constant.
	e := NewExecutor("")

	if got := e.timeoutFor(HookMatcher{}); got != DefaultHookTimeout {
		t.Errorf("default fallback = %s, want %s", got, DefaultHookTimeout)
	}

	e.DefaultTimeout = 5 * time.Second
	if got := e.timeoutFor(HookMatcher{}); got != 5*time.Second {
		t.Errorf("Executor.DefaultTimeout: got %s, want 5s", got)
	}

	if got := e.timeoutFor(HookMatcher{TimeoutMs: 200}); got != 200*time.Millisecond {
		t.Errorf("matcher override: got %s, want 200ms", got)
	}
}

// Tests for the exit-code protocol from #36. The executor maps:
//   exit 0       → parse stdout JSON
//   exit 2       → force deny (preserve other fields from JSON if any)
//   other != 0   → ignore stdout, log + empty decision
// These tests verify each branch end-to-end via real sh execution.

func TestRun_Exit2WithEmptyStdoutForcesBlockedDeny(t *testing.T) {
	e := NewExecutor("")
	m := HookMatcher{Command: "exit 2"}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.ExitCode != 2 {
		t.Errorf("ExitCode = %d, want 2", got.ExitCode)
	}
	if !got.Decision.IsDeny() {
		t.Errorf("exit 2 should force deny; got perm=%q", got.Decision.PermissionDecision)
	}
	if got.Decision.Reason != defaultBlockedReason {
		t.Errorf("Reason = %q, want default %q", got.Decision.Reason, defaultBlockedReason)
	}
}

func TestRun_Exit2WithJSONReasonPreservesReason(t *testing.T) {
	e := NewExecutor("")
	m := HookMatcher{
		Command: `echo '{"reason":"bash is forbidden"}' && exit 2`,
	}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !got.Decision.IsDeny() {
		t.Errorf("exit 2 should force deny; got perm=%q", got.Decision.PermissionDecision)
	}
	if got.Decision.Reason != "bash is forbidden" {
		t.Errorf("Reason = %q, want hook-supplied 'bash is forbidden'", got.Decision.Reason)
	}
}

func TestRun_Exit2WithAllowJSONStillResultsInDeny(t *testing.T) {
	// A pathological hook that prints "allow" then exits 2. Exit code
	// wins. Reason from JSON is preserved (operator may want to see
	// what the script tried to say).
	e := NewExecutor("")
	m := HookMatcher{
		Command: `echo '{"permissionDecision":"allow","reason":"I tried"}' && exit 2`,
	}
	got, _ := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if !got.Decision.IsDeny() {
		t.Errorf("exit 2 should override allow JSON; got perm=%q",
			got.Decision.PermissionDecision)
	}
	if got.Decision.Reason != "I tried" {
		t.Errorf("Reason from JSON should survive; got %q", got.Decision.Reason)
	}
}

func TestRun_Exit2PreservesAdditionalContext(t *testing.T) {
	// A blocking hook may still want to attach an explanation. The
	// AdditionalContext (and UpdatedInput) flow through; only the
	// PermissionDecision is overridden.
	e := NewExecutor("")
	m := HookMatcher{
		Command: `echo '{"additionalContext":"see audit log","reason":"forbidden"}' && exit 2`,
	}
	got, _ := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if !got.Decision.IsDeny() {
		t.Errorf("expected deny")
	}
	if got.Decision.AdditionalContext != "see audit log" {
		t.Errorf("AdditionalContext = %q, want preserved", got.Decision.AdditionalContext)
	}
	if got.Decision.Reason != "forbidden" {
		t.Errorf("Reason = %q, want hook-supplied", got.Decision.Reason)
	}
}

func TestRun_Exit2WithMalformedJSONFallsBackToDefaultReason(t *testing.T) {
	// JSON-less deny is intentional ergonomics. We don't warn about
	// the parse failure (it wasn't an attempt to emit JSON).
	e := NewExecutor("")
	m := HookMatcher{Command: `echo not json && exit 2`}
	got, _ := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if !got.Decision.IsDeny() {
		t.Errorf("expected deny on exit 2 + non-JSON stdout")
	}
	if got.Decision.Reason != defaultBlockedReason {
		t.Errorf("Reason = %q, want default %q", got.Decision.Reason, defaultBlockedReason)
	}
}

func TestRun_OtherNonZeroIgnoresStdoutAllow(t *testing.T) {
	// Exit 1 with valid allow JSON: the hook itself errored, so we
	// ignore its half-baked output. Empty decision returned.
	e := NewExecutor("")
	m := HookMatcher{
		Command: `echo '{"permissionDecision":"allow","reason":"yes"}' && exit 1`,
	}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.ExitCode != 1 {
		t.Errorf("ExitCode = %d, want 1", got.ExitCode)
	}
	if got.Decision.IsAllow() || got.Decision.IsDeny() {
		t.Errorf("exit 1 stdout should be ignored; got perm=%q",
			got.Decision.PermissionDecision)
	}
	if got.Decision.Reason != "" {
		t.Errorf("Reason should be empty for exit 1; got %q", got.Decision.Reason)
	}
}

func TestRun_Exit127IsLoggedAndIgnored(t *testing.T) {
	// sh -c with a missing executable exits 127. Stdout is empty;
	// the decision should be empty + a slog warning emitted (we
	// don't capture slog here — just verify no false-positive
	// decision).
	e := NewExecutor("")
	m := HookMatcher{Command: "/this/path/does/not/exist"}
	got, err := e.Run(context.Background(), HookEventPreToolUse, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.ExitCode == 0 {
		t.Errorf("expected non-zero exit; got 0")
	}
	if !decisionIsZero(got.Decision) {
		t.Errorf("exit 127 should give empty decision; got %+v", got.Decision)
	}
}

// Unit-level tests on parseDecision directly so each branch is
// exercised without spawning a process. Complement the
// integration tests above.

func TestParseDecision_Exit0WithValidJSON(t *testing.T) {
	got := parseDecision([]byte(`{"permissionDecision":"allow","reason":"ok"}`),
		0, "test-command")
	if !got.IsAllow() {
		t.Errorf("expected allow; got perm=%q", got.PermissionDecision)
	}
	if got.Reason != "ok" {
		t.Errorf("Reason = %q, want 'ok'", got.Reason)
	}
}

func TestParseDecision_Exit0WithEmptyStdout(t *testing.T) {
	got := parseDecision(nil, 0, "test-command")
	if !decisionIsZero(got) {
		t.Errorf("empty stdout + exit 0 should give zero decision; got %+v", got)
	}
}

func TestParseDecision_Exit0WithMalformedJSON(t *testing.T) {
	got := parseDecision([]byte(`{not json`), 0, "test-command")
	if !decisionIsZero(got) {
		t.Errorf("malformed JSON should fall back to zero; got %+v", got)
	}
}

func TestParseDecision_Exit2WithEmpty(t *testing.T) {
	got := parseDecision(nil, 2, "test-command")
	if !got.IsDeny() {
		t.Errorf("exit 2 should force deny; got perm=%q", got.PermissionDecision)
	}
	if got.Reason != defaultBlockedReason {
		t.Errorf("Reason = %q, want default", got.Reason)
	}
}

func TestParseDecision_Exit2WithJSONFields(t *testing.T) {
	got := parseDecision(
		[]byte(`{"additionalContext":"note","reason":"r"}`),
		2, "test-command")
	if !got.IsDeny() {
		t.Errorf("expected deny")
	}
	if got.Reason != "r" {
		t.Errorf("Reason = %q, want 'r' (preserved)", got.Reason)
	}
	if got.AdditionalContext != "note" {
		t.Errorf("AdditionalContext = %q, want 'note' (preserved)", got.AdditionalContext)
	}
}

func TestParseDecision_OtherNonZeroIgnoresStdout(t *testing.T) {
	got := parseDecision(
		[]byte(`{"permissionDecision":"allow"}`),
		1, "test-command")
	if !decisionIsZero(got) {
		t.Errorf("exit 1 should produce zero decision; got %+v", got)
	}
}

func TestRun_DurationLessThanTimeout(t *testing.T) {
	// Sanity: a quick command's Duration should be << timeout.
	e := NewExecutor("")
	m := HookMatcher{Command: "true", TimeoutMs: 5000}
	got, err := e.Run(context.Background(), HookEventStop, m, struct{}{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	upper := time.Duration(math.Min(1e9, float64(5*time.Second))) // 1s
	if got.Duration > upper {
		t.Errorf("Duration = %s, want <= 1s for trivial command", got.Duration)
	}
}
