package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

// Bubble Tea's tea.Program blocks on real terminal I/O, which is
// hostile to unit tests. We exercise the model's Init/Update/View
// contract directly — pure functions, easy to test — and verify the
// interactive surface manually.
//
// Tests that need a real *agents.ReactLoop construct one with a stub
// Provider; see newTestModel below.

func TestInitialModel_TextareaFocused(t *testing.T) {
	m := initialModel(nil)
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
	m := initialModel(nil)
	if cmd := m.Init(); cmd == nil {
		t.Errorf("Init() should return textarea.Blink to start cursor blinking, got nil")
	}
}

func TestUpdate_WindowSizeMsgResizesWidgets(t *testing.T) {
	m := initialModel(nil)
	next, _ := m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	got := next.(model)
	if got.width != 120 || got.height != 40 {
		t.Errorf("dimensions not stored: width=%d height=%d", got.width, got.height)
	}
	if got.viewport.Width != 120 {
		t.Errorf("viewport.Width = %d, want 120", got.viewport.Width)
	}
	wantViewportHeight := 40 - inputHeight - dividerHeight
	if got.viewport.Height != wantViewportHeight {
		t.Errorf("viewport.Height = %d, want %d", got.viewport.Height, wantViewportHeight)
	}
}

func TestUpdate_TinyTerminalClampsViewport(t *testing.T) {
	// If the terminal is so small that input + divider > height, the
	// viewport height should clamp to 0 instead of going negative.
	m := initialModel(nil)
	next, _ := m.Update(tea.WindowSizeMsg{Width: 40, Height: 3})
	got := next.(model)
	if got.viewport.Height != 0 {
		t.Errorf("viewport.Height = %d, want 0 (clamped)", got.viewport.Height)
	}
}

func TestUpdate_CtrlCQuitsWhenIdle(t *testing.T) {
	m := initialModel(nil)
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
	m := initialModel(nil)
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
	m := initialModel(loop)
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
	m := initialModel(nil)
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
	m := initialModel(nil)
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
	m := initialModel(nil)
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
	m := initialModel(nil)
	m, _ = applyWindowSize(m, 100, 30)
	out := m.View()
	if !strings.Contains(out, "no messages yet") {
		t.Errorf("empty viewport should show help hint, got:\n%s", out)
	}
}

func TestView_PreWindowSizeFallback(t *testing.T) {
	m := initialModel(nil)
	// No WindowSizeMsg yet — width is 0.
	out := m.View()
	if !strings.Contains(out, "starting") {
		t.Errorf("pre-resize View should show starting hint, got %q", out)
	}
}

func TestView_EmptyWhenQuitting(t *testing.T) {
	m := initialModel(nil)
	m, _ = applyWindowSize(m, 100, 30)
	m.quitting = true
	if out := m.View(); out != "" {
		t.Errorf("View() should be empty when quitting, got %q", out)
	}
}

func TestView_ThinkingIndicatorAppearsDuringTurn(t *testing.T) {
	m := initialModel(nil)
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
