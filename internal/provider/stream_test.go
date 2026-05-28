package provider

import (
	"errors"
	"testing"
)

func TestStreamEventKind_String(t *testing.T) {
	cases := []struct {
		kind StreamEventKind
		want string
	}{
		{StreamEventTextDelta, "text_delta"},
		{StreamEventReasoningDelta, "reasoning_delta"},
		{StreamEventToolCallStart, "tool_call_start"},
		{StreamEventToolCallDelta, "tool_call_delta"},
		{StreamEventToolCallDone, "tool_call_done"},
		{StreamEventUsage, "usage"},
		{StreamEventDone, "done"},
		{StreamEventError, "error"},
		{StreamEventKind(-1), "unknown"},
		{StreamEventKind(999), "unknown"},
	}
	for _, c := range cases {
		if got := c.kind.String(); got != c.want {
			t.Errorf("StreamEventKind(%d).String() = %q, want %q", c.kind, got, c.want)
		}
	}
}

// Defensive: catches a future edit that accidentally collapses two
// iota constants onto the same value (e.g. by re-ordering and
// dropping a name). All eight kinds must have distinct integer
// values.
func TestStreamEventKind_DistinctValues(t *testing.T) {
	all := []StreamEventKind{
		StreamEventTextDelta,
		StreamEventReasoningDelta,
		StreamEventToolCallStart,
		StreamEventToolCallDelta,
		StreamEventToolCallDone,
		StreamEventUsage,
		StreamEventDone,
		StreamEventError,
	}
	seen := map[StreamEventKind]bool{}
	for _, k := range all {
		if seen[k] {
			t.Fatalf("duplicate StreamEventKind value: %d (%s)", k, k)
		}
		seen[k] = true
	}
}

func TestNewTextDelta(t *testing.T) {
	ev := NewTextDelta("hello")
	if ev.Kind != StreamEventTextDelta {
		t.Errorf("Kind = %s, want text_delta", ev.Kind)
	}
	if ev.Text != "hello" {
		t.Errorf("Text = %q, want %q", ev.Text, "hello")
	}
	// Other fields should be zero.
	if ev.ToolCall != (ToolCallDelta{}) {
		t.Errorf("ToolCall should be zero, got %+v", ev.ToolCall)
	}
	if ev.Response != nil || ev.Err != nil {
		t.Errorf("Response/Err should be nil, got %v / %v", ev.Response, ev.Err)
	}
}

func TestNewReasoningDelta(t *testing.T) {
	ev := NewReasoningDelta("thinking…")
	if ev.Kind != StreamEventReasoningDelta {
		t.Errorf("Kind = %s, want reasoning_delta", ev.Kind)
	}
	if ev.Text != "thinking…" {
		t.Errorf("Text = %q", ev.Text)
	}
}

func TestNewToolCallStart(t *testing.T) {
	ev := NewToolCallStart(2, "call_abc", "read_file")
	if ev.Kind != StreamEventToolCallStart {
		t.Errorf("Kind = %s, want tool_call_start", ev.Kind)
	}
	if ev.ToolCall.Index != 2 {
		t.Errorf("Index = %d, want 2", ev.ToolCall.Index)
	}
	if ev.ToolCall.ID != "call_abc" {
		t.Errorf("ID = %q", ev.ToolCall.ID)
	}
	if ev.ToolCall.Name != "read_file" {
		t.Errorf("Name = %q", ev.ToolCall.Name)
	}
	if ev.ToolCall.Arguments != "" {
		t.Errorf("Arguments should be empty on Start, got %q", ev.ToolCall.Arguments)
	}
}

func TestNewToolCallDelta(t *testing.T) {
	ev := NewToolCallDelta(2, `{"path":`)
	if ev.Kind != StreamEventToolCallDelta {
		t.Errorf("Kind = %s, want tool_call_delta", ev.Kind)
	}
	if ev.ToolCall.Index != 2 {
		t.Errorf("Index = %d, want 2", ev.ToolCall.Index)
	}
	if ev.ToolCall.Arguments != `{"path":` {
		t.Errorf("Arguments = %q", ev.ToolCall.Arguments)
	}
	// ID/Name not populated on Delta — they're only on Start/Done.
	if ev.ToolCall.ID != "" || ev.ToolCall.Name != "" {
		t.Errorf("ID/Name should be empty on Delta, got %q / %q",
			ev.ToolCall.ID, ev.ToolCall.Name)
	}
}

func TestNewToolCallDone(t *testing.T) {
	ev := NewToolCallDone(2, "call_abc", "read_file", `{"path":"x"}`)
	if ev.Kind != StreamEventToolCallDone {
		t.Errorf("Kind = %s, want tool_call_done", ev.Kind)
	}
	want := ToolCallDelta{
		Index:     2,
		ID:        "call_abc",
		Name:      "read_file",
		Arguments: `{"path":"x"}`,
	}
	if ev.ToolCall != want {
		t.Errorf("ToolCall = %+v, want %+v", ev.ToolCall, want)
	}
}

func TestNewUsage(t *testing.T) {
	u := Usage{PromptTokens: 100, CachedTokens: 50, CompletionTokens: 20}
	ev := NewUsage(u)
	if ev.Kind != StreamEventUsage {
		t.Errorf("Kind = %s, want usage", ev.Kind)
	}
	if ev.Usage != u {
		t.Errorf("Usage = %+v, want %+v", ev.Usage, u)
	}
}

func TestNewDone(t *testing.T) {
	resp := &Response{Content: "final answer", FinishReason: "stop"}
	ev := NewDone(resp)
	if ev.Kind != StreamEventDone {
		t.Errorf("Kind = %s, want done", ev.Kind)
	}
	if ev.Response != resp {
		t.Errorf("Response pointer mismatch")
	}
	if ev.Response.Content != "final answer" {
		t.Errorf("Response.Content = %q", ev.Response.Content)
	}
}

func TestNewDone_NilResponseAllowed(t *testing.T) {
	// Edge: an adapter might emit Done with a nil Response in pathological
	// cases (e.g. empty model output). The constructor doesn't reject it;
	// the loop is the layer that decides what to do with empty output.
	ev := NewDone(nil)
	if ev.Kind != StreamEventDone {
		t.Errorf("Kind = %s, want done", ev.Kind)
	}
	if ev.Response != nil {
		t.Errorf("Response = %+v, want nil", ev.Response)
	}
}

func TestNewError(t *testing.T) {
	want := errors.New("connection reset")
	ev := NewError(want)
	if ev.Kind != StreamEventError {
		t.Errorf("Kind = %s, want error", ev.Kind)
	}
	if !errors.Is(ev.Err, want) {
		t.Errorf("Err = %v, want %v", ev.Err, want)
	}
}

// Exhaustive-switch sentinel: if a new StreamEventKind is added
// without a String() arm, this test fails because the new value
// returns "unknown". The compiler can't enforce exhaustiveness for
// iota constants, so we lean on a runtime check here.
//
// The list mirrors the const block declaration order in stream.go;
// a developer adding a new variant has to update both places to
// keep this test passing.
func TestStreamEventKind_AllKindsHaveStringArm(t *testing.T) {
	all := []StreamEventKind{
		StreamEventTextDelta,
		StreamEventReasoningDelta,
		StreamEventToolCallStart,
		StreamEventToolCallDelta,
		StreamEventToolCallDone,
		StreamEventUsage,
		StreamEventDone,
		StreamEventError,
	}
	for _, k := range all {
		if s := k.String(); s == "unknown" {
			t.Errorf("StreamEventKind(%d) returned %q — add a case to String()", k, s)
		}
	}
}
