package main

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

// Bubble Tea's tea.Program blocks on real terminal I/O, which is
// hostile to unit tests. We sidestep that by exercising the Model's
// Init/Update/View contract directly — pure functions, easy to test —
// and verify the interactive surface (rendering, keybinds) manually.
//
// This buys most of the safety net unit tests would normally provide:
// quit-key handling is verified, the initial state is verified, the
// rendered output is verified. The thing it doesn't catch is "does
// the binary actually launch in a real terminal" — that's the manual
// test recipe in the commit body.

func TestInitialModel_ZeroValue(t *testing.T) {
	m := initialModel()
	if m.quitting {
		t.Errorf("initial model should not be quitting")
	}
}

func TestInit_NoCommand(t *testing.T) {
	m := initialModel()
	if cmd := m.Init(); cmd != nil {
		t.Errorf("Init() should return nil (no startup work yet), got %v", cmd)
	}
}

// quitKeys covers the three keys we want to terminate on. Future
// commits will narrow this set (q becomes a literal character once
// the textarea has focus), but for the scaffold all three are global
// quits.
var quitKeys = []string{"ctrl+c", "q", "esc"}

func TestUpdate_QuitKeysSetQuittingAndEmitTeaQuit(t *testing.T) {
	for _, key := range quitKeys {
		t.Run(key, func(t *testing.T) {
			m := initialModel()
			msg := keyMsg(key)
			next, cmd := m.Update(msg)
			if !next.(Model).quitting {
				t.Errorf("%s should set quitting=true", key)
			}
			if cmd == nil {
				t.Errorf("%s should return a non-nil command (tea.Quit)", key)
			}
			// We can't easily compare cmd to tea.Quit (it's a function
			// value), but we can run it and confirm it produces the
			// QuitMsg the runtime expects.
			out := cmd()
			if _, ok := out.(tea.QuitMsg); !ok {
				t.Errorf("%s cmd produced %T, want tea.QuitMsg", key, out)
			}
		})
	}
}

func TestUpdate_OtherKeysNoOp(t *testing.T) {
	m := initialModel()
	for _, key := range []string{"a", "1", "enter", "tab", "f1"} {
		next, cmd := m.Update(keyMsg(key))
		if next.(Model).quitting {
			t.Errorf("%q should NOT trigger quit", key)
		}
		if cmd != nil {
			t.Errorf("%q should return nil cmd, got %v", key, cmd)
		}
	}
}

func TestView_ShowsPlaceholderWhenNotQuitting(t *testing.T) {
	m := initialModel()
	out := m.View()
	if out == "" {
		t.Fatal("View() returned empty string while not quitting")
	}
	for _, want := range []string{"opendev-tui", "scaffold", "Quit"} {
		if !strings.Contains(out, want) {
			t.Errorf("View() missing %q in output:\n%s", want, out)
		}
	}
}

func TestView_EmptyWhenQuitting(t *testing.T) {
	m := initialModel()
	m.quitting = true
	if out := m.View(); out != "" {
		t.Errorf("View() should return empty string when quitting, got %q", out)
	}
}

// keyMsg builds a tea.KeyMsg from a key string the same way Bubble
// Tea's input loop produces them. Wrapper so the table-driven tests
// above stay readable.
func keyMsg(s string) tea.KeyMsg {
	switch s {
	case "ctrl+c":
		return tea.KeyMsg{Type: tea.KeyCtrlC}
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEsc}
	case "enter":
		return tea.KeyMsg{Type: tea.KeyEnter}
	case "tab":
		return tea.KeyMsg{Type: tea.KeyTab}
	case "f1":
		return tea.KeyMsg{Type: tea.KeyF1}
	default:
		return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
	}
}
