package hooks

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// All payload types share the same test shape: marshal → unmarshal
// → deep-equal. The point isn't to test encoding/json (it works) —
// it's to pin the JSON tag names so a refactor that renames a field
// silently changes the wire contract and breaks every hook in the
// wild.

func TestSessionStartPayload_RoundTrip(t *testing.T) {
	original := SessionStartPayload{
		SessionID:  "sess_abc",
		WorkingDir: "/work/repo",
	}
	roundTrip(t, original, &SessionStartPayload{})
	assertJSONHasKey(t, original, "session_id")
	assertJSONHasKey(t, original, "working_dir")
}

func TestUserPromptSubmitPayload_RoundTrip(t *testing.T) {
	original := UserPromptSubmitPayload{Prompt: "what time is it"}
	roundTrip(t, original, &UserPromptSubmitPayload{})
	assertJSONHasKey(t, original, "prompt")
}

func TestPreToolUsePayload_RoundTrip(t *testing.T) {
	original := PreToolUsePayload{
		Tool: "bash",
		Args: json.RawMessage(`{"cmd":"ls"}`),
	}
	roundTrip(t, original, &PreToolUsePayload{})
	assertJSONHasKey(t, original, "tool")
	assertJSONHasKey(t, original, "args")
}

func TestPreToolUsePayload_OmitEmptyArgs(t *testing.T) {
	// A no-arg tool call should omit the args field entirely.
	original := PreToolUsePayload{Tool: "ping"}
	b, _ := json.Marshal(original)
	if strings.Contains(string(b), "args") {
		t.Errorf("empty Args should be omitted; got %s", b)
	}
}

func TestPostToolUsePayload_RoundTrip(t *testing.T) {
	original := PostToolUsePayload{
		Tool:    "read_file",
		Output:  "package main",
		Success: true,
	}
	roundTrip(t, original, &PostToolUsePayload{})
	assertJSONHasKey(t, original, "tool")
	assertJSONHasKey(t, original, "output")
	assertJSONHasKey(t, original, "success")
}

func TestPostToolUseFailurePayload_RoundTrip(t *testing.T) {
	original := PostToolUseFailurePayload{
		Tool:  "bash",
		Error: "command not found",
	}
	roundTrip(t, original, &PostToolUseFailurePayload{})
	assertJSONHasKey(t, original, "tool")
	assertJSONHasKey(t, original, "error")
}

func TestSubagentStartPayload_RoundTrip(t *testing.T) {
	original := SubagentStartPayload{
		AgentType: "Explore",
		Task:      "find the longest go file",
	}
	roundTrip(t, original, &SubagentStartPayload{})
	assertJSONHasKey(t, original, "agent_type")
	assertJSONHasKey(t, original, "task")
}

func TestSubagentStopPayload_RoundTrip(t *testing.T) {
	original := SubagentStopPayload{
		AgentType: "Explore",
		Result:    "longest is main.go at 500 lines",
		CostUSD:   0.0042,
	}
	roundTrip(t, original, &SubagentStopPayload{})
	assertJSONHasKey(t, original, "agent_type")
	assertJSONHasKey(t, original, "result")
	assertJSONHasKey(t, original, "cost_usd")
}

func TestStopPayload_RoundTrip(t *testing.T) {
	original := StopPayload{Result: "all done"}
	roundTrip(t, original, &StopPayload{})
	assertJSONHasKey(t, original, "result")
}

func TestStopPayload_OmitEmpty(t *testing.T) {
	b, _ := json.Marshal(StopPayload{})
	if string(b) != "{}" {
		t.Errorf("empty StopPayload should marshal to {}; got %s", b)
	}
}

func TestPreCompactPayload_RoundTrip(t *testing.T) {
	original := PreCompactPayload{
		Level:        "mask",
		MessageCount: 420,
	}
	roundTrip(t, original, &PreCompactPayload{})
	assertJSONHasKey(t, original, "level")
	assertJSONHasKey(t, original, "message_count")
}

func TestSessionEndPayload_RoundTrip(t *testing.T) {
	original := SessionEndPayload{
		SessionID: "sess_abc",
		CostUSD:   0.42,
		CallCount: 17,
	}
	roundTrip(t, original, &SessionEndPayload{})
	assertJSONHasKey(t, original, "session_id")
	assertJSONHasKey(t, original, "cost_usd")
	assertJSONHasKey(t, original, "call_count")
}

// roundTrip marshals original, unmarshals into dest, and asserts
// the deep-equality with what came back. Generic helper for every
// payload type.
func roundTrip(t *testing.T, original any, dest any) {
	t.Helper()
	b, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := json.Unmarshal(b, dest); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	// Compare via reflection so we can pass the dest pointer in
	// once and the helper figures out what to dereference.
	gotVal := reflect.ValueOf(dest).Elem().Interface()
	if !reflect.DeepEqual(original, gotVal) {
		t.Errorf("round-trip mismatch:\n got: %+v\nwant: %+v", gotVal, original)
	}
}

// assertJSONHasKey marshals v and checks that the encoded JSON
// contains the given key. Used to pin JSON tag names.
func assertJSONHasKey(t *testing.T, v any, key string) {
	t.Helper()
	b, _ := json.Marshal(v)
	if !strings.Contains(string(b), `"`+key+`"`) {
		t.Errorf("JSON output missing key %q; got %s", key, b)
	}
}
