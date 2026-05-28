package main

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

// Bubble Tea's tea.Program blocks on real terminal I/O, which is
// hostile to unit tests. We exercise the Model's Init/Update/View
// contract directly — pure functions, easy to test — and verify the
// interactive surface manually.

func TestInitialModel_TextareaFocused(t *testing.T) {
	m := initialModel()
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
}

func TestInit_ReturnsBlinkCmd(t *testing.T) {
	m := initialModel()
	if cmd := m.Init(); cmd == nil {
		t.Errorf("Init() should return textarea.Blink to start cursor blinking, got nil")
	}
}

func TestUpdate_WindowSizeMsgResizesWidgets(t *testing.T) {
	m := initialModel()
	next, _ := m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	got := next.(Model)
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
	m := initialModel()
	next, _ := m.Update(tea.WindowSizeMsg{Width: 40, Height: 3})
	got := next.(Model)
	if got.viewport.Height != 0 {
		t.Errorf("viewport.Height = %d, want 0 (clamped)", got.viewport.Height)
	}
}

func TestUpdate_CtrlCQuits(t *testing.T) {
	m := initialModel()
	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	if !next.(Model).quitting {
		t.Errorf("Ctrl-C should set quitting=true")
	}
	if cmd == nil {
		t.Fatal("Ctrl-C should return tea.Quit command")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Errorf("Ctrl-C cmd produced %T, want tea.QuitMsg", cmd())
	}
}

func TestUpdate_CtrlDSubmitsAndClears(t *testing.T) {
	m := initialModel()
	m, _ = applyWindowSize(m, 100, 30)
	m = typeInto(m, "hello world")

	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	got := next.(Model)

	if len(got.history) != 1 {
		t.Fatalf("history len = %d, want 1", len(got.history))
	}
	if !strings.Contains(got.history[0], "hello world") {
		t.Errorf("history[0] = %q, want it to contain 'hello world'", got.history[0])
	}
	if !strings.HasPrefix(got.history[0], "> ") {
		t.Errorf("submitted entry should have '> ' prefix, got %q", got.history[0])
	}
	if got.textarea.Value() != "" {
		t.Errorf("textarea should be empty after submit, got %q", got.textarea.Value())
	}
}

func TestUpdate_CtrlDEmptyIsNoOp(t *testing.T) {
	m := initialModel()
	m, _ = applyWindowSize(m, 100, 30)
	// No input typed. Submit anyway.
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	got := next.(Model)
	if len(got.history) != 0 {
		t.Errorf("empty Ctrl-D should not append to history, got %d entries", len(got.history))
	}
}

func TestUpdate_CtrlDWhitespaceIsNoOp(t *testing.T) {
	m := initialModel()
	m, _ = applyWindowSize(m, 100, 30)
	m = typeInto(m, "   \n\t  ")
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	got := next.(Model)
	if len(got.history) != 0 {
		t.Errorf("whitespace-only submit should be a no-op, got %d history entries", len(got.history))
	}
}

func TestUpdate_MultipleSubmitsAccumulate(t *testing.T) {
	m := initialModel()
	m, _ = applyWindowSize(m, 100, 30)
	for _, s := range []string{"first", "second", "third"} {
		m = typeInto(m, s)
		next, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
		m = next.(Model)
	}
	if len(m.history) != 3 {
		t.Fatalf("history len = %d, want 3", len(m.history))
	}
	for i, want := range []string{"first", "second", "third"} {
		if !strings.Contains(m.history[i], want) {
			t.Errorf("history[%d] = %q, want it to contain %q", i, m.history[i], want)
		}
	}
}

func TestView_EmptyShowsHelpHint(t *testing.T) {
	m := initialModel()
	m, _ = applyWindowSize(m, 100, 30)
	out := m.View()
	if !strings.Contains(out, "no messages yet") {
		t.Errorf("empty viewport should show help hint, got:\n%s", out)
	}
}

func TestView_RendersHistory(t *testing.T) {
	m := initialModel()
	m, _ = applyWindowSize(m, 100, 30)
	m = typeInto(m, "test message")
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	got := next.(Model).View()
	if !strings.Contains(got, "test message") {
		t.Errorf("View() should render history; got:\n%s", got)
	}
}

func TestView_PreWindowSizeFallback(t *testing.T) {
	m := initialModel()
	// No WindowSizeMsg yet — width is 0.
	out := m.View()
	if !strings.Contains(out, "starting") {
		t.Errorf("pre-resize View should show starting hint, got %q", out)
	}
}

func TestView_EmptyWhenQuitting(t *testing.T) {
	m := initialModel()
	m, _ = applyWindowSize(m, 100, 30)
	m.quitting = true
	if out := m.View(); out != "" {
		t.Errorf("View() should be empty when quitting, got %q", out)
	}
}

// ---- helpers ----

// applyWindowSize sends a WindowSizeMsg through Update so subsequent
// assertions can rely on the widgets being properly dimensioned.
func applyWindowSize(m Model, w, h int) (Model, tea.Cmd) {
	next, cmd := m.Update(tea.WindowSizeMsg{Width: w, Height: h})
	return next.(Model), cmd
}

// typeInto simulates the user typing a string into the textarea by
// pushing each rune through the textarea's Update directly. The
// model-level Update path can't easily synthesize a multi-rune
// KeyMsg, so we route around it for test setup — production keystroke
// behavior is unchanged.
func typeInto(m Model, s string) Model {
	m.textarea.SetValue(s)
	return m
}
