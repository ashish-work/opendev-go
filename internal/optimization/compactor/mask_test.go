package compactor

import (
	"reflect"
	"testing"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// --- Builders ---------------------------------------------------

// sysMsg, userMsg, asstMsg build the non-tool roles. Tests
// construct synthetic history slices inline so the assertions stay
// next to the message shape.
func sysMsg(text string) provider.Message {
	return provider.Message{
		Role:    "system",
		Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: text}},
	}
}

func userMsg(text string) provider.Message {
	return provider.Message{
		Role:    "user",
		Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: text}},
	}
}

func asstMsg(text string) provider.Message {
	return provider.Message{
		Role:    "assistant",
		Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: text}},
	}
}

// toolMsg builds a role=tool message with a given ToolCallID and
// payload text. The Name field is populated to match the wire
// shape but is irrelevant to the masking decision.
func toolMsg(id, text string) provider.Message {
	return provider.Message{
		Role:       "tool",
		ToolCallID: id,
		Name:       "fake_tool",
		Content:    []provider.ContentBlock{{Kind: provider.ContentText, Text: text}},
	}
}

// toolText extracts the first text block's content. Tests use it
// to check whether masking landed or was skipped.
func toolText(m provider.Message) string {
	if len(m.Content) == 0 {
		return ""
	}
	return m.Content[0].Text
}

// --- Tests ------------------------------------------------------

func TestMaskObservations_EmptyHistoryReturnsEmpty(t *testing.T) {
	got := MaskObservations(nil, DefaultRecentKept)
	if len(got) != 0 {
		t.Errorf("nil history → len=%d, want 0", len(got))
	}

	got = MaskObservations([]provider.Message{}, DefaultRecentKept)
	if len(got) != 0 {
		t.Errorf("empty history → len=%d, want 0", len(got))
	}
}

func TestMaskObservations_NoToolMessagesUnchanged(t *testing.T) {
	history := []provider.Message{
		sysMsg("system"),
		userMsg("user"),
		asstMsg("assistant"),
	}
	got := MaskObservations(history, DefaultRecentKept)
	if !reflect.DeepEqual(got, history) {
		t.Errorf("history with no tool messages should pass through unchanged\ngot:  %v\nwant: %v", got, history)
	}
}

func TestMaskObservations_FewerToolMessagesThanRecentLeavesAllRaw(t *testing.T) {
	// 3 tool messages, recent=10 → nothing masked.
	history := []provider.Message{
		sysMsg("system"),
		userMsg("user"),
		toolMsg("c1", "result 1"),
		toolMsg("c2", "result 2"),
		toolMsg("c3", "result 3"),
	}
	got := MaskObservations(history, 10)
	if !reflect.DeepEqual(got, history) {
		t.Errorf("history with fewer tool msgs than recent should pass through unchanged")
	}
}

func TestMaskObservations_RecentEqualsCountLeavesAllRaw(t *testing.T) {
	// 3 tool messages, recent=3 → all preserved.
	history := []provider.Message{
		toolMsg("c1", "result 1"),
		toolMsg("c2", "result 2"),
		toolMsg("c3", "result 3"),
	}
	got := MaskObservations(history, 3)
	if !reflect.DeepEqual(got, history) {
		t.Errorf("recent equals count should leave all raw\ngot:  %v\nwant: %v", got, history)
	}
}

func TestMaskObservations_MasksOldestBeyondRecent(t *testing.T) {
	// 5 tool messages, recent=2 → oldest 3 masked, recent 2 kept.
	history := []provider.Message{
		toolMsg("c1", "old result 1"),
		toolMsg("c2", "old result 2"),
		toolMsg("c3", "old result 3"),
		toolMsg("c4", "recent result 4"),
		toolMsg("c5", "recent result 5"),
	}
	got := MaskObservations(history, 2)

	if want := "[ref:c1]"; toolText(got[0]) != want {
		t.Errorf("got[0] text = %q, want %q", toolText(got[0]), want)
	}
	if want := "[ref:c2]"; toolText(got[1]) != want {
		t.Errorf("got[1] text = %q, want %q", toolText(got[1]), want)
	}
	if want := "[ref:c3]"; toolText(got[2]) != want {
		t.Errorf("got[2] text = %q, want %q", toolText(got[2]), want)
	}
	if toolText(got[3]) != "recent result 4" {
		t.Errorf("got[3] should stay raw; got %q", toolText(got[3]))
	}
	if toolText(got[4]) != "recent result 5" {
		t.Errorf("got[4] should stay raw; got %q", toolText(got[4]))
	}
}

func TestMaskObservations_MarkerFormatUsesToolCallID(t *testing.T) {
	// The masked content must include the ToolCallID verbatim
	// inside the marker, with no transformation. Otherwise the
	// model can't map the marker back to the tool_call.
	history := []provider.Message{
		toolMsg("call_abc_123", "old"),
		toolMsg("c2", "new"),
	}
	got := MaskObservations(history, 1)
	if want := "[ref:call_abc_123]"; toolText(got[0]) != want {
		t.Errorf("marker = %q, want %q", toolText(got[0]), want)
	}
}

func TestMaskObservations_PreservesToolCallIDAndName(t *testing.T) {
	// Wire-format pairing depends on ToolCallID surviving the
	// transformation. Name is preserved by convention so the
	// model still sees which tool was called even after the
	// content gets collapsed.
	history := []provider.Message{
		{
			Role:       "tool",
			ToolCallID: "c1",
			Name:       "read_file",
			Content:    []provider.ContentBlock{{Kind: provider.ContentText, Text: "big file content"}},
		},
		toolMsg("c2", "recent"),
	}
	got := MaskObservations(history, 1)

	if got[0].ToolCallID != "c1" {
		t.Errorf("ToolCallID = %q, want c1 (must survive masking)", got[0].ToolCallID)
	}
	if got[0].Name != "read_file" {
		t.Errorf("Name = %q, want read_file", got[0].Name)
	}
	if got[0].Role != "tool" {
		t.Errorf("Role = %q, want tool", got[0].Role)
	}
}

func TestMaskObservations_NonToolRolesUntouchedAroundToolMessages(t *testing.T) {
	// Even when masking is happening, non-tool messages must pass
	// through with byte-identical content. Catches a regression
	// where a refactor accidentally re-wraps non-tool text.
	system := sysMsg("system rules")
	user := userMsg("user prompt")
	asst := asstMsg("assistant reply")
	history := []provider.Message{
		system,
		user,
		toolMsg("c1", "tool 1"),
		asst,
		toolMsg("c2", "tool 2"),
		toolMsg("c3", "tool 3"),
	}
	got := MaskObservations(history, 1)

	if !reflect.DeepEqual(got[0], system) {
		t.Errorf("system message mutated")
	}
	if !reflect.DeepEqual(got[1], user) {
		t.Errorf("user message mutated")
	}
	if !reflect.DeepEqual(got[3], asst) {
		t.Errorf("assistant message mutated")
	}
}

func TestMaskObservations_EmptyToolCallIDLeftRaw(t *testing.T) {
	// Defensive: a tool message without an ID has no useful
	// marker we can build. Mask would produce "[ref:]" which
	// the model can't dereference. Leave raw.
	noID := provider.Message{
		Role:    "tool",
		Name:    "weird_tool",
		Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "orphan output"}},
	}
	history := []provider.Message{
		noID,
		toolMsg("c1", "valid"),
		toolMsg("c2", "valid recent"),
	}
	got := MaskObservations(history, 1)
	// noID stays raw; c1 gets masked; c2 stays raw.
	if toolText(got[0]) != "orphan output" {
		t.Errorf("empty-ID tool message must NOT be masked; got %q", toolText(got[0]))
	}
	if got[0].ToolCallID != "" {
		t.Errorf("empty-ID preservation expected, got %q", got[0].ToolCallID)
	}
	if toolText(got[1]) != "[ref:c1]" {
		t.Errorf("c1 should be masked; got %q", toolText(got[1]))
	}
	if toolText(got[2]) != "valid recent" {
		t.Errorf("c2 should stay raw; got %q", toolText(got[2]))
	}
}

func TestMaskObservations_AlreadyMaskedIsNoOp(t *testing.T) {
	// Idempotence: calling MaskObservations on history that
	// already contains "[ref:..." markers must NOT double-wrap.
	// The safety phase will call this on every iteration; without
	// this property, the loop would corrupt history into
	// "[ref:[ref:c1]]" recursion.
	history := []provider.Message{
		// Pre-masked, simulating a prior pass.
		{
			Role:       "tool",
			ToolCallID: "c1",
			Name:       "fake_tool",
			Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "[ref:c1]"},
			},
		},
		toolMsg("c2", "recent"),
	}
	got := MaskObservations(history, 1)
	if toolText(got[0]) != "[ref:c1]" {
		t.Errorf("already-masked content must be preserved; got %q", toolText(got[0]))
	}
}

func TestMaskObservations_DoubleApplicationIsStable(t *testing.T) {
	// Stronger version of idempotence: feed the output of one
	// call into a second call and verify the result is
	// byte-identical. Future contributors may forget the
	// AlreadyMasked check; this test catches them.
	original := []provider.Message{
		toolMsg("c1", "old result"),
		toolMsg("c2", "older result"),
		toolMsg("c3", "recent"),
	}
	once := MaskObservations(original, 1)
	twice := MaskObservations(once, 1)
	if !reflect.DeepEqual(once, twice) {
		t.Errorf("double application diverged\nonce:  %v\ntwice: %v", once, twice)
	}
}

func TestMaskObservations_InputSliceNotMutated(t *testing.T) {
	// Immutability discipline: callers (the safety phase) assume
	// pc.History is untouched. We capture references to the
	// original messages' Content slices and verify they still
	// hold the original text after MaskObservations returns.
	originalTexts := []string{"raw 1", "raw 2", "raw 3"}
	history := []provider.Message{
		toolMsg("c1", originalTexts[0]),
		toolMsg("c2", originalTexts[1]),
		toolMsg("c3", originalTexts[2]),
	}

	_ = MaskObservations(history, 1)

	for i, want := range originalTexts {
		if got := toolText(history[i]); got != want {
			t.Errorf("history[%d] mutated: got %q, want %q", i, got, want)
		}
	}
}

func TestMaskObservations_RecentZeroMasksAll(t *testing.T) {
	history := []provider.Message{
		toolMsg("c1", "first"),
		toolMsg("c2", "second"),
	}
	got := MaskObservations(history, 0)
	if toolText(got[0]) != "[ref:c1]" {
		t.Errorf("recent=0 should mask c1; got %q", toolText(got[0]))
	}
	if toolText(got[1]) != "[ref:c2]" {
		t.Errorf("recent=0 should mask c2; got %q", toolText(got[1]))
	}
}

func TestMaskObservations_NegativeRecentTreatedAsZero(t *testing.T) {
	// Documented edge case: negative recent collapses to 0
	// (mask everything). Test pins it so a future change can't
	// silently flip to "treat as default" or similar.
	history := []provider.Message{
		toolMsg("c1", "first"),
	}
	got := MaskObservations(history, -5)
	if toolText(got[0]) != "[ref:c1]" {
		t.Errorf("negative recent should mask; got %q", toolText(got[0]))
	}
}

func TestMaskObservations_ReturnsNewSliceNotSharedBacking(t *testing.T) {
	// Even when no masking happens (recent >= count), the
	// returned slice should be a fresh allocation so the caller
	// can't accidentally mutate the input by writing through
	// the output. This is a subtle property that prevents
	// "spooky action at a distance" bugs.
	history := []provider.Message{
		toolMsg("c1", "kept"),
	}
	got := MaskObservations(history, 10)

	// Modify the output's content and verify the input did not
	// change. (We can't compare slice headers directly because
	// our implementation always allocates a new slice.)
	got[0].Content[0].Text = "mutated by test"
	if toolText(history[0]) != "kept" {
		t.Errorf("input mutated through output: history[0] = %q",
			toolText(history[0]))
	}
}

func TestMarkerFor_Format(t *testing.T) {
	// Direct unit test of the helper so any future change to
	// the marker format triggers a clear failure.
	cases := []struct {
		id, want string
	}{
		{"c1", "[ref:c1]"},
		{"call_abc", "[ref:call_abc]"},
		{"", "[ref:]"}, // markerFor doesn't filter; MaskObservations gates that
	}
	for _, tc := range cases {
		if got := markerFor(tc.id); got != tc.want {
			t.Errorf("markerFor(%q) = %q, want %q", tc.id, got, tc.want)
		}
	}
}
