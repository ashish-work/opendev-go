//go:build !windows

package bash

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/ashishgupta/opendev-go/internal/tools"
)

// run is a shorthand for Execute with a marshaled args struct.
func run(t *testing.T, workingDir string, a args, ctx context.Context) (tools.ToolResult, error) {
	t.Helper()
	raw, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return New().Execute(ctx, tools.ToolContext{WorkingDir: workingDir}, raw)
}

func TestEchoSuccess(t *testing.T) {
	got, err := run(t, "", args{Command: "echo hello"}, context.Background())
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !got.Success {
		t.Fatalf("Success = false, Error = %q", got.Error)
	}
	if !strings.Contains(got.Output, "hello") {
		t.Errorf("Output = %q, want substring %q", got.Output, "hello")
	}
	if got.Metadata["exit_code"] != 0 {
		t.Errorf("exit_code = %v, want 0", got.Metadata["exit_code"])
	}
}

func TestNonZeroExitFails(t *testing.T) {
	got, err := run(t, "", args{Command: "false"}, context.Background())
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got.Success {
		t.Error("Success = true, want false")
	}
	if got.Metadata["exit_code"] != 1 {
		t.Errorf("exit_code = %v, want 1", got.Metadata["exit_code"])
	}
}

func TestEmptyCommand(t *testing.T) {
	got, err := run(t, "", args{Command: ""}, context.Background())
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got.Success {
		t.Error("Success = true, want false")
	}
	if !strings.Contains(got.Error, "command is required") {
		t.Errorf("Error = %q, want substring %q", got.Error, "command is required")
	}
}

func TestCombinedStdoutStderr(t *testing.T) {
	got, err := run(t, "",
		args{Command: "echo out; echo err 1>&2"}, context.Background())
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	if !strings.Contains(got.Output, "out") {
		t.Errorf("Output missing stdout 'out': %q", got.Output)
	}
	if !strings.Contains(got.Output, "err") {
		t.Errorf("Output missing stderr 'err': %q", got.Output)
	}
}

func TestWorkingDirRespected(t *testing.T) {
	dir := t.TempDir()
	got, err := run(t, dir, args{Command: "pwd"}, context.Background())
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	// On macOS, t.TempDir() resolves through /private/var symlinks;
	// pwd may return either form. Match by suffix instead.
	out := strings.TrimSpace(got.Output)
	if !strings.HasSuffix(out, dir) && !strings.HasSuffix(dir, strings.TrimPrefix(out, "/private")) {
		t.Errorf("pwd output = %q, want suffix matching %q", out, dir)
	}
}

func TestTimeoutKillsLongCommand(t *testing.T) {
	start := time.Now()
	got, err := run(t, "",
		args{Command: "sleep 5", TimeoutSec: 1},
		context.Background())
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got.Success {
		t.Error("Success = true, want false (timeout)")
	}
	if !strings.Contains(got.Error, "timed out") {
		t.Errorf("Error = %q, want substring %q", got.Error, "timed out")
	}
	// Should have completed in ~1s, not 5s. Allow generous slack.
	if elapsed > 3*time.Second {
		t.Errorf("elapsed = %v, want ~1s — timeout not enforced", elapsed)
	}
}

func TestOutputTruncated(t *testing.T) {
	prev := maxOutputBytes
	maxOutputBytes = 100
	t.Cleanup(func() { maxOutputBytes = prev })

	// Print 500 chars; truncated to 100 + marker.
	got, err := run(t, "",
		args{Command: "printf 'x%.0s' {1..500}"},
		context.Background())
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
	if !strings.Contains(got.Output, "...[output truncated]") {
		t.Errorf("Output missing truncation marker: %q", got.Output)
	}
	if got.Metadata["output_truncated"] != true {
		t.Errorf("output_truncated = %v, want true", got.Metadata["output_truncated"])
	}
}

func TestOuterContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancelled

	_, err := run(t, "", args{Command: "sleep 5"}, ctx)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestTimeoutSecClampedToMax(t *testing.T) {
	// Override max so the test doesn't actually wait the default 600s
	// even if clamping silently broke.
	prev := maxTimeoutSec
	maxTimeoutSec = 2
	t.Cleanup(func() { maxTimeoutSec = prev })

	// Request 99999s — should clamp to 2s. Command runs fast so
	// effectively all this verifies is the path doesn't blow up.
	got, err := run(t, "",
		args{Command: "echo ok", TimeoutSec: 99999},
		context.Background())
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
}

func TestNegativeTimeoutUsesDefault(t *testing.T) {
	got, err := run(t, "",
		args{Command: "echo ok", TimeoutSec: -5},
		context.Background())
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !got.Success {
		t.Fatalf("Success = false: %s", got.Error)
	}
}

func TestNonexistentCommand(t *testing.T) {
	got, err := run(t, "",
		args{Command: "xyznotacommand_zzz"},
		context.Background())
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got.Success {
		t.Error("Success = true, want false")
	}
	// exit_code from `sh -c` for not-found is typically 127.
	if code, ok := got.Metadata["exit_code"].(int); !ok || code == 0 {
		t.Errorf("exit_code = %v, want non-zero", got.Metadata["exit_code"])
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
	var parsed map[string]any
	if err := json.Unmarshal(New().Schema(), &parsed); err != nil {
		t.Fatalf("Schema is not valid JSON: %v", err)
	}
	if parsed["type"] != "object" {
		t.Errorf(`Schema["type"] = %v, want "object"`, parsed["type"])
	}
	req, _ := parsed["required"].([]any)
	found := false
	for _, name := range req {
		if name == "command" {
			found = true
		}
	}
	if !found {
		t.Errorf(`Schema "required" missing "command": %v`, req)
	}
}

func TestNameAndDescription(t *testing.T) {
	tool := New()
	if tool.Name() != "bash" {
		t.Errorf("Name() = %q, want %q", tool.Name(), "bash")
	}
	if len(tool.Description()) < 20 {
		t.Errorf("Description() too short")
	}
}
