package hooks

import "testing"

// allEvents is the canonical list of HookEvent constants. Any new
// event must be appended here so the exhaustive tests pick it up.
var allEvents = []HookEvent{
	HookEventSessionStart,
	HookEventUserPromptSubmit,
	HookEventPreToolUse,
	HookEventPostToolUse,
	HookEventPostToolUseFailure,
	HookEventSubagentStart,
	HookEventSubagentStop,
	HookEventStop,
	HookEventPreCompact,
	HookEventSessionEnd,
}

func TestHookEvent_AllPairwiseDistinct(t *testing.T) {
	// Defensive: catches an iota re-order that accidentally
	// collapses two events onto the same value.
	seen := map[HookEvent]bool{}
	for _, e := range allEvents {
		if seen[e] {
			t.Fatalf("duplicate HookEvent value: %d (%s)", e, e)
		}
		seen[e] = true
	}
	if len(seen) != 10 {
		t.Errorf("expected 10 distinct events, got %d", len(seen))
	}
}

func TestHookEvent_String(t *testing.T) {
	cases := []struct {
		event HookEvent
		want  string
	}{
		{HookEventSessionStart, "session_start"},
		{HookEventUserPromptSubmit, "user_prompt_submit"},
		{HookEventPreToolUse, "pre_tool_use"},
		{HookEventPostToolUse, "post_tool_use"},
		{HookEventPostToolUseFailure, "post_tool_use_failure"},
		{HookEventSubagentStart, "subagent_start"},
		{HookEventSubagentStop, "subagent_stop"},
		{HookEventStop, "stop"},
		{HookEventPreCompact, "pre_compact"},
		{HookEventSessionEnd, "session_end"},
		{HookEvent(-1), "unknown"},
		{HookEvent(999), "unknown"},
	}
	for _, c := range cases {
		if got := c.event.String(); got != c.want {
			t.Errorf("HookEvent(%d).String() = %q, want %q", c.event, got, c.want)
		}
	}
}

func TestParse_KnownEvents(t *testing.T) {
	for _, e := range allEvents {
		got, ok := Parse(e.String())
		if !ok {
			t.Errorf("Parse(%q) returned ok=false; want true", e.String())
			continue
		}
		if got != e {
			t.Errorf("Parse(%q) = %s, want %s", e.String(), got, e)
		}
	}
}

func TestParse_UnknownReturnsFalse(t *testing.T) {
	cases := []string{
		"",
		"unknown",
		"PreToolUse",  // wrong case — Parse is case-sensitive
		"pretooluse",  // missing separator
		"pre-tool-use", // wrong separator
		"some_future_event",
	}
	for _, s := range cases {
		got, ok := Parse(s)
		if ok {
			t.Errorf("Parse(%q) returned (%s, true); want (_, false)", s, got)
		}
		if got != 0 {
			t.Errorf("Parse(%q) returned non-zero event %d on failure", s, got)
		}
	}
}

// Bijection invariant: every declared HookEvent round-trips through
// String → Parse. This is the load-bearing test for the loader (#31),
// which uses Parse on the same strings String() produces.
func TestHookEvent_StringParseRoundTrip(t *testing.T) {
	for _, e := range allEvents {
		s := e.String()
		got, ok := Parse(s)
		if !ok {
			t.Errorf("round-trip: Parse(%q) failed for event %d", s, e)
			continue
		}
		if got != e {
			t.Errorf("round-trip: Parse(String(%d)) = %d, want %d", e, got, e)
		}
	}
}

// Exhaustive-switch sentinel. A new HookEvent constant without a
// corresponding String() arm will return "unknown" and fail this
// test, alerting the future committer to update both places (plus
// Parse, plus allEvents).
func TestHookEvent_AllEventsHaveStringArm(t *testing.T) {
	for _, e := range allEvents {
		if s := e.String(); s == "unknown" {
			t.Errorf("HookEvent(%d) returned %q — add a case to String()", e, s)
		}
	}
}

// Exhaustive-switch sentinel for Parse. Mirrors the String() check —
// every declared constant must be reachable from a snake_case string.
func TestParse_AllEventsReachable(t *testing.T) {
	for _, e := range allEvents {
		_, ok := Parse(e.String())
		if !ok {
			t.Errorf("Parse cannot reach event %d (%s) — add a case", e, e)
		}
	}
}
