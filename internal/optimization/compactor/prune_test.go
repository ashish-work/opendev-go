package compactor

import (
	"reflect"
	"strings"
	"testing"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// --- Helpers ----------------------------------------------------
//
// sysMsg / userMsg / asstMsg / toolMsg / toolText are shared with
// mask_test.go (same package). Prune tests reuse them and add only
// what they need.

// namedToolMsg builds a role=tool message with an explicit tool
// Name so protected-tool tests can target read_file/edit_file/etc.
func namedToolMsg(id, name, text string) provider.Message {
	return provider.Message{
		Role:       "tool",
		ToolCallID: id,
		Name:       name,
		Content:    []provider.ContentBlock{{Kind: provider.ContentText, Text: text}},
	}
}

// longText returns a string of n 'x' runes — used to push an
// output over or under the size threshold deterministically.
func longText(n int) string { return strings.Repeat("x", n) }

// --- Tests ------------------------------------------------------

func TestPrune_EmptyHistoryReturnsEmpty(t *testing.T) {
	if got := Prune(nil, DefaultPruneMaxLen); len(got) != 0 {
		t.Errorf("nil history → len=%d, want 0", len(got))
	}
	if got := Prune([]provider.Message{}, DefaultPruneMaxLen); len(got) != 0 {
		t.Errorf("empty history → len=%d, want 0", len(got))
	}
}

func TestPrune_NonToolRolesUntouched(t *testing.T) {
	history := []provider.Message{
		sysMsg("system"),
		userMsg("user"),
		asstMsg("assistant"),
	}
	got := Prune(history, DefaultPruneMaxLen)
	if !reflect.DeepEqual(got, history) {
		t.Errorf("non-tool roles must pass through unchanged\ngot:  %v\nwant: %v", got, history)
	}
}

func TestPrune_ShortOutputPruned(t *testing.T) {
	// A short, non-protected tool output is collapsed to "[pruned]".
	history := []provider.Message{
		toolMsg("c1", "exit 0, nothing else to report"), // ~30 chars
	}
	got := Prune(history, DefaultPruneMaxLen)
	if toolText(got[0]) != markerPruned {
		t.Errorf("short output should be pruned; got %q", toolText(got[0]))
	}
}

func TestPrune_LongOutputLeftRaw(t *testing.T) {
	// An output at or above maxLen carries enough signal to keep.
	big := longText(DefaultPruneMaxLen) // exactly maxLen → not < maxLen
	history := []provider.Message{
		toolMsg("c1", big),
	}
	got := Prune(history, DefaultPruneMaxLen)
	if toolText(got[0]) != big {
		t.Errorf("output at maxLen must stay raw; got len %d", len(toolText(got[0])))
	}
}

func TestPrune_BoundaryJustBelowMaxLenPruned(t *testing.T) {
	// maxLen-1 is short → pruned. Pins the strict "< maxLen" upper
	// bound so a future >= slip is caught.
	history := []provider.Message{
		toolMsg("c1", longText(DefaultPruneMaxLen-1)),
	}
	got := Prune(history, DefaultPruneMaxLen)
	if toolText(got[0]) != markerPruned {
		t.Errorf("maxLen-1 should prune; got %q", toolText(got[0]))
	}
}

func TestPrune_ProtectedToolsNeverPruned(t *testing.T) {
	// read_file / edit_file / write_file outputs survive even when
	// short, because the model references them by ID downstream.
	for _, name := range ProtectedTools {
		history := []provider.Message{
			namedToolMsg("c1", name, "small but precious"),
		}
		got := Prune(history, DefaultPruneMaxLen)
		if toolText(got[0]) != "small but precious" {
			t.Errorf("protected tool %q must not be pruned; got %q", name, toolText(got[0]))
		}
	}
}

func TestPrune_ProtectedToolsConfigurable(t *testing.T) {
	// Operators can reconfigure the protected set wholesale. Save and
	// restore so other tests see the default.
	original := ProtectedTools
	defer func() { ProtectedTools = original }()

	ProtectedTools = []string{"bash"} // protect bash, unprotect read_file
	history := []provider.Message{
		namedToolMsg("c1", "bash", "kept now"),
		namedToolMsg("c2", "read_file", "pruned now"),
	}
	got := Prune(history, DefaultPruneMaxLen)
	if toolText(got[0]) != "kept now" {
		t.Errorf("newly protected bash should survive; got %q", toolText(got[0]))
	}
	if toolText(got[1]) != markerPruned {
		t.Errorf("unprotected read_file should prune; got %q", toolText(got[1]))
	}
}

func TestPrune_TinyOutputLeftRaw(t *testing.T) {
	// An output at or below the marker's own length gains nothing
	// from pruning — replacing it would only grow the message — so
	// it is left raw.
	tiny := "ok" // 2 chars, well under len("[pruned]")=8
	history := []provider.Message{
		toolMsg("c1", tiny),
	}
	got := Prune(history, DefaultPruneMaxLen)
	if toolText(got[0]) != tiny {
		t.Errorf("tiny output must stay raw; got %q", toolText(got[0]))
	}
}

func TestPrune_OutputEqualToMarkerLenLeftRaw(t *testing.T) {
	// Exactly len("[pruned]") chars: the open lower bound (n >
	// len(marker)) means this stays raw — no token saving available.
	exact := longText(len(markerPruned)) // 8 chars
	history := []provider.Message{
		toolMsg("c1", exact),
	}
	got := Prune(history, DefaultPruneMaxLen)
	if toolText(got[0]) != exact {
		t.Errorf("output equal to marker length must stay raw; got %q", toolText(got[0]))
	}
}

func TestPrune_AlreadyMaskedNotPruned(t *testing.T) {
	// A "[ref:<id>]" marker is short and from a non-protected tool,
	// but pruning it would strip the dereference handle. Masked and
	// pruned messages must stay disjoint.
	history := []provider.Message{
		{
			Role:       "tool",
			ToolCallID: "c1",
			Name:       "bash",
			Content:    []provider.ContentBlock{{Kind: provider.ContentText, Text: "[ref:c1]"}},
		},
	}
	got := Prune(history, DefaultPruneMaxLen)
	if toolText(got[0]) != "[ref:c1]" {
		t.Errorf("masking marker must survive prune; got %q", toolText(got[0]))
	}
}

func TestPrune_AlreadyPrunedIsNoOp(t *testing.T) {
	// Idempotence: re-pruning a "[pruned]" message is a no-op.
	history := []provider.Message{
		{
			Role:       "tool",
			ToolCallID: "c1",
			Name:       "bash",
			Content:    []provider.ContentBlock{{Kind: provider.ContentText, Text: markerPruned}},
		},
	}
	got := Prune(history, DefaultPruneMaxLen)
	if toolText(got[0]) != markerPruned {
		t.Errorf("already-pruned message must be preserved; got %q", toolText(got[0]))
	}
}

func TestPrune_DoubleApplicationIsStable(t *testing.T) {
	// f(f(x)) == f(x), byte for byte — guards against a future
	// contributor dropping the isAlreadyPruned check.
	original := []provider.Message{
		toolMsg("c1", "short bash output here"),
		namedToolMsg("c2", "read_file", "protected content"),
		toolMsg("c3", longText(DefaultPruneMaxLen+50)),
	}
	once := Prune(original, DefaultPruneMaxLen)
	twice := Prune(once, DefaultPruneMaxLen)
	if !reflect.DeepEqual(once, twice) {
		t.Errorf("double application diverged\nonce:  %v\ntwice: %v", once, twice)
	}
}

func TestPrune_PreservesToolCallIDAndName(t *testing.T) {
	// Wire-format pairing depends on ToolCallID surviving; Name is
	// preserved so the model still sees which tool produced the gap.
	history := []provider.Message{
		namedToolMsg("c1", "grep", "3 matches in 2 files"),
	}
	got := Prune(history, DefaultPruneMaxLen)
	if got[0].ToolCallID != "c1" {
		t.Errorf("ToolCallID = %q, want c1", got[0].ToolCallID)
	}
	if got[0].Name != "grep" {
		t.Errorf("Name = %q, want grep", got[0].Name)
	}
	if got[0].Role != "tool" {
		t.Errorf("Role = %q, want tool", got[0].Role)
	}
	if toolText(got[0]) != markerPruned {
		t.Errorf("content should be pruned; got %q", toolText(got[0]))
	}
}

func TestPrune_InputSliceNotMutated(t *testing.T) {
	originalTexts := []string{"short one here", "another short one"}
	history := []provider.Message{
		toolMsg("c1", originalTexts[0]),
		toolMsg("c2", originalTexts[1]),
	}
	_ = Prune(history, DefaultPruneMaxLen)
	for i, want := range originalTexts {
		if got := toolText(history[i]); got != want {
			t.Errorf("history[%d] mutated: got %q, want %q", i, got, want)
		}
	}
}

func TestPrune_ReturnsNewSliceNotSharedBacking(t *testing.T) {
	// Even on a pure pass-through (long output, nothing pruned) the
	// returned slice must be independent so callers can't mutate the
	// input by writing through the output.
	history := []provider.Message{
		toolMsg("c1", longText(DefaultPruneMaxLen+10)),
	}
	got := Prune(history, DefaultPruneMaxLen)
	got[0].Content[0].Text = "mutated by test"
	if strings.HasPrefix(toolText(history[0]), "mutated") {
		t.Errorf("input mutated through output")
	}
}

func TestPrune_MaxLenZeroPrunesNothing(t *testing.T) {
	// maxLen <= 0 is the disabled default: nothing satisfies the
	// upper bound, so everything passes through.
	history := []provider.Message{
		toolMsg("c1", "would-be-pruned output"),
	}
	got := Prune(history, 0)
	if toolText(got[0]) != "would-be-pruned output" {
		t.Errorf("maxLen=0 should prune nothing; got %q", toolText(got[0]))
	}
}

func TestPrune_MixedHistorySelectivePruning(t *testing.T) {
	// End-to-end shape: a realistic mix where exactly the short,
	// non-protected, non-marker tool outputs collapse and everything
	// else is preserved verbatim.
	system := sysMsg("rules")
	user := userMsg("do the thing")
	asst := asstMsg("calling tools")
	bigBash := longText(DefaultPruneMaxLen + 100)

	history := []provider.Message{
		system,
		user,
		asst,
		toolMsg("c1", "short bash status output"),          // pruned
		namedToolMsg("c2", "read_file", "short file body"), // protected → kept
		toolMsg("c3", bigBash),                             // long → kept
		{ // already masked → kept
			Role: "tool", ToolCallID: "c4", Name: "list_files",
			Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "[ref:c4]"}},
		},
	}
	got := Prune(history, DefaultPruneMaxLen)

	if !reflect.DeepEqual(got[0], system) || !reflect.DeepEqual(got[1], user) || !reflect.DeepEqual(got[2], asst) {
		t.Errorf("non-tool messages mutated")
	}
	if toolText(got[3]) != markerPruned {
		t.Errorf("c1 should be pruned; got %q", toolText(got[3]))
	}
	if toolText(got[4]) != "short file body" {
		t.Errorf("c2 (read_file) should be kept; got %q", toolText(got[4]))
	}
	if toolText(got[5]) != bigBash {
		t.Errorf("c3 (long) should be kept; got len %d", len(toolText(got[5])))
	}
	if toolText(got[6]) != "[ref:c4]" {
		t.Errorf("c4 (masked) should be kept; got %q", toolText(got[6]))
	}
}

func TestPrune_MultiBlockTextLengthSummed(t *testing.T) {
	// contentTextLen sums across blocks: two blocks of 120 chars
	// total 240 (>= maxLen) → not short → kept. Guards against a
	// "first block only" length bug.
	history := []provider.Message{
		{
			Role: "tool", ToolCallID: "c1", Name: "bash",
			Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: longText(120)},
				{Kind: provider.ContentText, Text: longText(120)},
			},
		},
	}
	got := Prune(history, DefaultPruneMaxLen)
	if toolText(got[0]) == markerPruned {
		t.Errorf("multi-block output summing to >= maxLen must not be pruned")
	}
}
