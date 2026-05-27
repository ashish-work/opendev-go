package doomloop

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ashishgupta/opendev-go/internal/provider"
)

// call is a tiny constructor — keeps test cases readable.
func call(name, args string) provider.ToolCall {
	return provider.ToolCall{
		ID:        "test-id",
		Name:      name,
		Arguments: json.RawMessage(args),
	}
}

func TestNewDetectorIsEmpty(t *testing.T) {
	d := New()
	if d.NudgeCount() != 0 {
		t.Errorf("NudgeCount = %d, want 0", d.NudgeCount())
	}
	a, _, _ := d.Check(nil)
	if a != None {
		t.Errorf("empty check = %v, want None", a)
	}
}

func TestSingleStepCycle(t *testing.T) {
	d := New()
	for i := 0; i < Threshold-1; i++ {
		a, _, _ := d.Check([]provider.ToolCall{call("read_file", `{"path":"x"}`)})
		if a != None {
			t.Fatalf("call %d: action = %v, want None", i+1, a)
		}
	}
	a, warning, recovery := d.Check([]provider.ToolCall{call("read_file", `{"path":"x"}`)})
	if a != Redirect {
		t.Errorf("3rd identical call: action = %v, want Redirect", a)
	}
	if !strings.Contains(warning, "read_file") {
		t.Errorf("warning missing tool name: %q", warning)
	}
	if !strings.Contains(warning, "stuck") {
		t.Errorf("warning missing 'stuck': %q", warning)
	}
	if recovery == "" {
		t.Error("recovery text empty on Redirect")
	}
}

func TestTwoStepCycle(t *testing.T) {
	d := New()
	// A,B repeated 3 times — 6 calls total.
	for i := 0; i < Threshold; i++ {
		_, _, _ = d.Check([]provider.ToolCall{call("read_file", `{}`)})
		a, warn, _ := d.Check([]provider.ToolCall{call("bash", `{}`)})
		if i < Threshold-1 {
			if a != None {
				t.Fatalf("step %d/B: action = %v, want None", i, a)
			}
		} else {
			if a != Redirect {
				t.Errorf("final step: action = %v, want Redirect", a)
			}
			if !strings.Contains(warn, "2-step") {
				t.Errorf("warning should mention 2-step cycle: %q", warn)
			}
			if !strings.Contains(warn, "read_file → bash") {
				t.Errorf("warning should name both tools: %q", warn)
			}
		}
	}
}

func TestThreeStepCycle(t *testing.T) {
	d := New()
	// A,B,C repeated 3 times — 9 calls.
	tools := []string{"read_file", "bash", "edit_file"}
	calls := 0
	var lastAction Action
	var lastWarn string
	for i := 0; i < Threshold; i++ {
		for _, name := range tools {
			lastAction, lastWarn, _ = d.Check([]provider.ToolCall{call(name, `{}`)})
			calls++
		}
	}
	if calls != 9 {
		t.Fatalf("did %d calls, want 9", calls)
	}
	if lastAction != Redirect {
		t.Errorf("after 9 cyclic calls: action = %v, want Redirect", lastAction)
	}
	if !strings.Contains(lastWarn, "3-step") {
		t.Errorf("warning should mention 3-step cycle: %q", lastWarn)
	}
}

func TestDifferentArgsAvoidsFalsePositive(t *testing.T) {
	d := New()
	// Same tool name, different args each time — NOT a doom loop.
	for i := 0; i < 5; i++ {
		args := `{"path":"file` + string(rune('0'+i)) + `"}`
		a, _, _ := d.Check([]provider.ToolCall{call("read_file", args)})
		if a != None {
			t.Errorf("call %d (args=%s) flagged as %v, expected None", i, args, a)
		}
	}
}

func TestEscalation_RedirectThenNotifyThenForceStop(t *testing.T) {
	// Pins the per-call escalation cadence:
	//   3rd identical call → Redirect (window now has 3 c's; cycle fires)
	//   4th identical call → Notify   (window has 4; last-3 still a cycle)
	//   5th identical call → ForceStop + window cleared
	d := New()
	c := call("noop", `{}`)

	// Calls 1 + 2: no detection yet.
	if a, _, _ := d.Check([]provider.ToolCall{c}); a != None {
		t.Fatalf("call 1: action = %v, want None", a)
	}
	if a, _, _ := d.Check([]provider.ToolCall{c}); a != None {
		t.Fatalf("call 2: action = %v, want None", a)
	}

	// Call 3 → Redirect.
	if a, _, _ := d.Check([]provider.ToolCall{c}); a != Redirect {
		t.Fatalf("call 3: action = %v, want Redirect", a)
	}
	if d.NudgeCount() != 1 {
		t.Errorf("NudgeCount after Redirect = %d, want 1", d.NudgeCount())
	}

	// Call 4 → Notify (escalation steps with EACH identical call after
	// the threshold is reached).
	if a, _, _ := d.Check([]provider.ToolCall{c}); a != Notify {
		t.Fatalf("call 4: action = %v, want Notify", a)
	}
	if d.NudgeCount() != 2 {
		t.Errorf("NudgeCount after Notify = %d, want 2", d.NudgeCount())
	}

	// Call 5 → ForceStop, window cleared.
	a, _, recovery := d.Check([]provider.ToolCall{c})
	if a != ForceStop {
		t.Fatalf("call 5: action = %v, want ForceStop", a)
	}
	if d.NudgeCount() != 3 {
		t.Errorf("NudgeCount after ForceStop = %d, want 3", d.NudgeCount())
	}
	if recovery != "" {
		t.Errorf("recovery on ForceStop should be empty, got %q", recovery)
	}
	if len(d.recent) != 0 {
		t.Errorf("recent window after ForceStop = %d, want 0 (cleared)", len(d.recent))
	}
}

func TestSlidingWindow_NoFalsePositiveOnHeterogeneousHistory(t *testing.T) {
	d := New()
	// 25 calls, all different names → never a cycle.
	for i := 0; i < 25; i++ {
		name := "tool" + string(rune('a'+(i%20)))
		a, _, _ := d.Check([]provider.ToolCall{call(name, `{}`)})
		if a != None {
			t.Errorf("call %d (%q) flagged %v on heterogeneous history", i, name, a)
		}
	}
	// Window stays capped at MaxRecent.
	if len(d.recent) != MaxRecent {
		t.Errorf("len(recent) = %d, want MaxRecent=%d", len(d.recent), MaxRecent)
	}
}

func TestReset(t *testing.T) {
	d := New()
	c := call("noop", `{}`)
	d.Check([]provider.ToolCall{c, c, c}) // pushes through Redirect
	if d.NudgeCount() == 0 {
		t.Fatalf("setup failed: NudgeCount = 0")
	}
	d.Reset()
	if d.NudgeCount() != 0 {
		t.Errorf("after Reset, NudgeCount = %d, want 0", d.NudgeCount())
	}
	if len(d.recent) != 0 {
		t.Errorf("after Reset, len(recent) = %d, want 0", len(d.recent))
	}
}

func TestMultipleCallsPerCheck(t *testing.T) {
	// One Check with three tool calls in a single batch should fire
	// the 1-step cycle too — fingerprints are appended in order.
	d := New()
	c := call("noop", `{}`)
	a, _, _ := d.Check([]provider.ToolCall{c, c, c})
	if a != Redirect {
		t.Errorf("3 calls in one batch: action = %v, want Redirect", a)
	}
}

func TestActionStringer(t *testing.T) {
	cases := []struct {
		a    Action
		want string
	}{
		{None, "none"},
		{Redirect, "redirect"},
		{Notify, "notify"},
		{ForceStop, "force_stop"},
		{Action(99), "unknown"},
	}
	for _, c := range cases {
		if got := c.a.String(); got != c.want {
			t.Errorf("Action(%d).String() = %q, want %q", c.a, got, c.want)
		}
	}
}

func TestFingerprint_Stable(t *testing.T) {
	// Same inputs → same fingerprint, every time.
	a := fingerprint("read_file", json.RawMessage(`{"path":"x"}`))
	b := fingerprint("read_file", json.RawMessage(`{"path":"x"}`))
	if a != b {
		t.Errorf("fingerprint not deterministic: %q vs %q", a, b)
	}
}

func TestFingerprint_DifferentArgsDifferentFingerprint(t *testing.T) {
	a := fingerprint("read_file", json.RawMessage(`{"path":"x"}`))
	b := fingerprint("read_file", json.RawMessage(`{"path":"y"}`))
	if a == b {
		t.Errorf("different args produced same fingerprint: %q", a)
	}
}
