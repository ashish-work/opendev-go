package tui

import (
	"fmt"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ashish-work/opendev-go/internal/agents"
	"github.com/ashish-work/opendev-go/internal/budget"
	"github.com/ashish-work/opendev-go/internal/cost"
)

// Bubble Tea's tea.Program blocks on real terminal I/O, which is
// hostile to unit tests. We exercise the model's Init/Update/View
// contract directly — pure functions, easy to test — and verify the
// interactive surface manually.
//
// Tests that need a real *agents.ReactLoop construct one with a stub
// Provider; see newTestModel below.

func TestInitialModel_TextareaFocused(t *testing.T) {
	m := initialModel(nil, "")
	if !m.textarea.Focused() {
		t.Errorf("textarea should be focused on startup")
	}
	if m.textarea.Value() != "" {
		t.Errorf("textarea should start empty, got %q", m.textarea.Value())
	}
	if len(m.history) != 0 {
		t.Errorf("history should start empty, got %d entries", len(m.history))
	}
	if m.quitting {
		t.Errorf("model should not start in quitting state")
	}
	if m.thinking {
		t.Errorf("model should not start in thinking state")
	}
}

func TestInit_ReturnsBlinkCmd(t *testing.T) {
	m := initialModel(nil, "")
	if cmd := m.Init(); cmd == nil {
		t.Errorf("Init() should return textarea.Blink to start cursor blinking, got nil")
	}
}

func TestUpdate_WindowSizeMsgResizesWidgets(t *testing.T) {
	m := initialModel(nil, "")
	next, _ := m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	got := next.(model)
	if got.width != 120 || got.height != 40 {
		t.Errorf("dimensions not stored: width=%d height=%d", got.width, got.height)
	}
	if got.viewport.Width != 120 {
		t.Errorf("viewport.Width = %d, want 120", got.viewport.Width)
	}
	wantViewportHeight := 40 - statusBarHeight - inputHeight - dividerHeight
	if got.viewport.Height != wantViewportHeight {
		t.Errorf("viewport.Height = %d, want %d", got.viewport.Height, wantViewportHeight)
	}
}

func TestUpdate_TinyTerminalClampsViewport(t *testing.T) {
	// If the terminal is so small that input + divider > height, the
	// viewport height should clamp to 0 instead of going negative.
	m := initialModel(nil, "")
	next, _ := m.Update(tea.WindowSizeMsg{Width: 40, Height: 3})
	got := next.(model)
	if got.viewport.Height != 0 {
		t.Errorf("viewport.Height = %d, want 0 (clamped)", got.viewport.Height)
	}
}

func TestUpdate_CtrlCQuitsWhenIdle(t *testing.T) {
	m := initialModel(nil, "")
	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	if !next.(model).quitting {
		t.Errorf("Ctrl-C should set quitting=true when idle")
	}
	if cmd == nil {
		t.Fatal("Ctrl-C while idle should return tea.Quit command")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Errorf("Ctrl-C cmd produced %T, want tea.QuitMsg", cmd())
	}
}

func TestUpdate_CtrlCCancelsTurnWhenThinking(t *testing.T) {
	// Build a model that's in the middle of a turn: thinking=true,
	// turnCancel set. Ctrl-C should cancel the turn (call the cancel
	// func) and NOT quit the program.
	cancelled := false
	m := initialModel(nil, "")
	m.thinking = true
	m.turnCancel = func() { cancelled = true }

	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	got := next.(model)
	if !cancelled {
		t.Errorf("Ctrl-C while thinking should invoke turnCancel")
	}
	if got.quitting {
		t.Errorf("Ctrl-C while thinking should NOT set quitting")
	}
	if cmd != nil {
		// We expect nil — the cancellation just signals the in-flight
		// goroutine. The follow-up turnCompleteMsg arrives via the
		// already-dispatched runTurnCmd, not via a new cmd here.
		t.Errorf("Ctrl-C while thinking should return nil cmd, got %v", cmd)
	}
}

func TestUpdate_CtrlDSubmitsAndClears(t *testing.T) {
	loop := newTestLoop(t, stubProvider{})
	m := initialModel(loop, "")
	m, _ = applyWindowSize(m, 100, 30)
	m = typeInto(m, "hello world")

	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	got := next.(model)

	// Submit should optimistically append the user message, enter
	// thinking state, clear the textarea, and return a runTurnCmd
	// for Bubble Tea to execute.
	if !got.thinking {
		t.Errorf("Ctrl-D should put model into thinking state")
	}
	if got.turnCancel == nil {
		t.Errorf("Ctrl-D should store a cancel func for the in-flight turn")
	}
	if len(got.history) != 1 {
		t.Fatalf("optimistic history len = %d, want 1 (just the user message)", len(got.history))
	}
	if got.history[0].role != roleUser || got.history[0].content != "hello world" {
		t.Errorf("optimistic history[0] = %+v, want {user, 'hello world'}", got.history[0])
	}
	if got.textarea.Value() != "" {
		t.Errorf("textarea should be empty after submit, got %q", got.textarea.Value())
	}
	if cmd == nil {
		t.Errorf("Ctrl-D should return a runTurnCmd, got nil")
	}
}

func TestUpdate_CtrlDIgnoredWhileThinking(t *testing.T) {
	m := initialModel(nil, "")
	m.thinking = true
	m = typeInto(m, "second submit while first is running")
	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	got := next.(model)
	// History shouldn't have grown; cmd is nil; thinking stays true.
	if len(got.history) != 0 {
		t.Errorf("history should not grow during in-flight turn, got %d", len(got.history))
	}
	if cmd != nil {
		t.Errorf("Ctrl-D while thinking should be a no-op, got cmd %v", cmd)
	}
	if !got.thinking {
		t.Errorf("Ctrl-D while thinking should leave thinking=true")
	}
}

func TestUpdate_CtrlDEmptyIsNoOp(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	// No input typed. Submit anyway.
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	got := next.(model)
	if len(got.history) != 0 {
		t.Errorf("empty Ctrl-D should not append to history, got %d entries", len(got.history))
	}
	if got.thinking {
		t.Errorf("empty Ctrl-D should not enter thinking state")
	}
}

func TestUpdate_CtrlDWhitespaceIsNoOp(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m = typeInto(m, "   \n\t  ")
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	got := next.(model)
	if len(got.history) != 0 {
		t.Errorf("whitespace-only submit should be a no-op, got %d history entries", len(got.history))
	}
	if got.thinking {
		t.Errorf("whitespace-only Ctrl-D should not enter thinking state")
	}
}

func TestView_EmptyShowsHelpHint(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	out := m.View()
	if !strings.Contains(out, "no messages yet") {
		t.Errorf("empty viewport should show help hint, got:\n%s", out)
	}
}

func TestView_PreWindowSizeFallback(t *testing.T) {
	m := initialModel(nil, "")
	// No WindowSizeMsg yet — width is 0.
	out := m.View()
	if !strings.Contains(out, "starting") {
		t.Errorf("pre-resize View should show starting hint, got %q", out)
	}
}

func TestView_EmptyWhenQuitting(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m.quitting = true
	if out := m.View(); out != "" {
		t.Errorf("View() should be empty when quitting, got %q", out)
	}
}

func TestUpdate_CtrlTTogglesToolsExpanded(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	if m.toolsExpanded {
		t.Fatalf("toolsExpanded should default to false")
	}
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlT})
	got := next.(model)
	if !got.toolsExpanded {
		t.Errorf("Ctrl-T should flip toolsExpanded to true")
	}
	// Second press flips back.
	next, _ = got.Update(tea.KeyMsg{Type: tea.KeyCtrlT})
	got = next.(model)
	if got.toolsExpanded {
		t.Errorf("second Ctrl-T should flip toolsExpanded back to false")
	}
}

func TestUpdate_PageKeysScrollViewport(t *testing.T) {
	// Set up a model with enough content that the viewport scrolls,
	// then verify PgUp moves the YOffset up. Specifics:
	//   - viewport height = 5 lines
	//   - viewport content = 20 lines
	//   - GotoBottom puts YOffset at max (>0)
	//   - PgUp should reduce YOffset
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 80, 10)
	// height=10 → viewport.Height = 10 - inputHeight(5) - divider(1) = 4
	// fill with 30 distinct lines so scroll position is meaningful
	lines := make([]string, 30)
	for i := range lines {
		lines[i] = fmt.Sprintf("line %02d", i)
	}
	m.viewport.SetContent(strings.Join(lines, "\n"))
	m.viewport.GotoBottom()
	startOffset := m.viewport.YOffset
	if startOffset == 0 {
		t.Fatalf("setup: GotoBottom should produce non-zero YOffset with overflow content")
	}

	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyPgUp})
	got := next.(model)
	if got.viewport.YOffset >= startOffset {
		t.Errorf("PgUp should decrease YOffset; before=%d after=%d", startOffset, got.viewport.YOffset)
	}

	// Now PgDown should bring it back (within the same range).
	next2, _ := got.Update(tea.KeyMsg{Type: tea.KeyPgDown})
	got2 := next2.(model)
	if got2.viewport.YOffset <= got.viewport.YOffset {
		t.Errorf("PgDown should increase YOffset; before=%d after=%d",
			got.viewport.YOffset, got2.viewport.YOffset)
	}
}

func TestUpdate_PageKeysDoNotTouchTextarea(t *testing.T) {
	// PgUp/PgDn should be intercepted before reaching the textarea.
	// Verify by typing content first, then sending PgUp — content
	// should be unchanged (no textarea-side effect).
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 80, 30)
	m = typeInto(m, "do not touch me")
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyPgUp})
	got := next.(model)
	if got.textarea.Value() != "do not touch me" {
		t.Errorf("PgUp should not modify textarea content, got %q", got.textarea.Value())
	}
}

func TestUpdate_CtrlHomeAndCtrlEndJumpToTopAndBottom(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 80, 10)
	lines := make([]string, 30)
	for i := range lines {
		lines[i] = fmt.Sprintf("line %02d", i)
	}
	m.viewport.SetContent(strings.Join(lines, "\n"))
	m.viewport.GotoBottom()

	// Ctrl-Home → top.
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlHome})
	if next.(model).viewport.YOffset != 0 {
		t.Errorf("Ctrl-Home should put YOffset = 0, got %d", next.(model).viewport.YOffset)
	}

	// Ctrl-End → bottom.
	next2, _ := next.(model).Update(tea.KeyMsg{Type: tea.KeyCtrlEnd})
	if next2.(model).viewport.YOffset == 0 {
		t.Errorf("Ctrl-End should put YOffset > 0 with overflow content")
	}
}

func TestUpdate_CtrlTRepaintsViewport(t *testing.T) {
	// With a tool message in history, toggling toolsExpanded should
	// change the rendered viewport content (different hint visibility,
	// different body length). Simplest check: rendered output differs
	// before vs after Ctrl-T.
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 80, 30)
	m.history = []viewMessage{
		{role: roleTool, toolName: "bash", content: strings.Join(
			[]string{"a", "b", "c", "d", "e"}, "\n")},
	}
	m.viewport.SetContent(m.renderHistory())
	collapsedOut := m.View()

	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlT})
	expandedOut := next.(model).View()

	if collapsedOut == expandedOut {
		t.Errorf("Ctrl-T should change rendered output when a tool message is present")
	}
}

func TestView_ThinkingIndicatorAppearsDuringTurn(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m.thinking = true
	// Refresh the viewport's cached content. In production code, the
	// Ctrl-D handler does this implicitly via SetContent(...) right
	// after flipping thinking=true; here we mimic that step because
	// we set the bool directly.
	m.viewport.SetContent(m.renderHistory())
	out := m.View()
	if !strings.Contains(out, "thinking") {
		t.Errorf("View() should show 'thinking' indicator while in-flight, got:\n%s", out)
	}
}

// ---- helpers ----

func TestView_StatusBarShowsModelName(t *testing.T) {
	m := initialModel(nil, "claude-opus-9")
	m, _ = applyWindowSize(m, 120, 30)
	out := m.View()
	if !strings.Contains(out, "claude-opus-9") {
		t.Errorf("View() should include model name in the status bar, got:\n%s", out)
	}
}

func TestView_StatusBarShowsCumulativeMetrics(t *testing.T) {
	m := initialModel(nil, "x")
	m, _ = applyWindowSize(m, 120, 30)
	// Pretend two turns happened with these accumulated totals.
	m.tracker.CallCount = 5
	m.tracker.TotalInputTokens = 1234
	m.tracker.TotalOutputTokens = 567
	m.tracker.TotalCostUSD = 0.0042 // sub-cent → 4-decimal FormatCost
	m.lastBudget.UsagePct = 0.024   // 2.4%

	out := m.View()
	for _, want := range []string{
		"iter 5",
		"in 1234",
		"out 567",
		"2.4%",
		"$0.0042",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("status bar should contain %q; got:\n%s", want, out)
		}
	}
}

func TestUpdate_TurnCompleteAccumulatesTracker(t *testing.T) {
	// Two consecutive turns: the per-turn trackers (the loop's
	// returned values) sum into the model's session tracker.
	m := initialModel(nil, "x")
	m, _ = applyWindowSize(m, 100, 30)

	// First turn lands.
	next, _ := m.Update(turnCompleteMsg{
		tracker: trackerWith(1, 100, 50, 0.001),
	})
	m = next.(model)
	if m.tracker.CallCount != 1 || m.tracker.TotalInputTokens != 100 {
		t.Fatalf("after first turn, tracker = %+v", m.tracker)
	}

	// Second turn lands and adds.
	next2, _ := m.Update(turnCompleteMsg{
		tracker: trackerWith(2, 200, 80, 0.003),
	})
	m = next2.(model)
	if m.tracker.CallCount != 3 {
		t.Errorf("CallCount = %d, want 3 (1+2)", m.tracker.CallCount)
	}
	if m.tracker.TotalInputTokens != 300 {
		t.Errorf("TotalInputTokens = %d, want 300 (100+200)", m.tracker.TotalInputTokens)
	}
	if m.tracker.TotalOutputTokens != 130 {
		t.Errorf("TotalOutputTokens = %d, want 130 (50+80)", m.tracker.TotalOutputTokens)
	}
	got := m.tracker.TotalCostUSD
	want := 0.004
	// float arithmetic — allow tiny epsilon.
	if got < want-1e-9 || got > want+1e-9 {
		t.Errorf("TotalCostUSD = %v, want %v", got, want)
	}
}

func TestUpdate_TurnCompleteCapturesLatestBudget(t *testing.T) {
	m := initialModel(nil, "x")
	m, _ = applyWindowSize(m, 100, 30)
	m.lastBudget.UsagePct = 0.10 // pretend an earlier snapshot

	// New turn lands with a different Budget; the snapshot should
	// REPLACE (not accumulate) so the status bar reflects the
	// latest context-window load.
	next, _ := m.Update(turnCompleteMsg{
		// minimal Result that just carries Budget data
		result: turnResultWithBudget(0.42, 5000, 6000),
	})
	got := next.(model).lastBudget
	if got.UsagePct != 0.42 {
		t.Errorf("UsagePct = %v, want 0.42 (latest, not accumulated)", got.UsagePct)
	}
	if got.Reported != 5000 || got.Estimated != 6000 {
		t.Errorf("Reported/Estimated = %d/%d, want 5000/6000", got.Reported, got.Estimated)
	}
}

// applyWindowSize sends a WindowSizeMsg through Update so subsequent
// assertions can rely on the widgets being properly dimensioned.
func applyWindowSize(m model, w, h int) (model, tea.Cmd) {
	next, cmd := m.Update(tea.WindowSizeMsg{Width: w, Height: h})
	return next.(model), cmd
}

// typeInto simulates the user typing a string into the textarea by
// pushing each rune through the textarea's Update directly. The
// model-level Update path can't easily synthesize a multi-rune
// KeyMsg, so we route around it for test setup — production keystroke
// behavior is unchanged.
func typeInto(m model, s string) model {
	m.textarea.SetValue(s)
	return m
}

// trackerWith builds a cost.Tracker carrying the given totals for
// turnCompleteMsg test setup. The model adds these into its session
// totals when the message arrives.
func trackerWith(callCount, in, out int64, costUSD float64) cost.Tracker {
	return cost.Tracker{
		CallCount:         callCount,
		TotalInputTokens:  in,
		TotalOutputTokens: out,
		TotalCostUSD:      costUSD,
	}
}

// turnResultWithBudget builds an agents.Result whose only populated
// field is Budget — used to test the status bar's "latest snapshot
// wins" semantics without setting up full message history.
func turnResultWithBudget(pct float64, reported, estimated int) agents.Result {
	return agents.Result{
		Budget: budget.Snapshot{
			Reported:  reported,
			Estimated: estimated,
			UsagePct:  pct,
		},
	}
}
