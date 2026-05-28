// Command opendev-tui is the Bubble Tea-based interactive front-end
// for the agent. It runs alongside cmd/opendev (the REPL) — both are
// real binaries against the same internal/agents core, so a reader of
// the repo can study either presentation layer in isolation.
//
// This commit is the SCAFFOLD only. Everything visible right now is a
// placeholder; the next several commits flesh out the Model with a
// textarea + viewport (#40), conversation history (#41), the agent
// loop (#42), tool-call cards (#43), and the status line (#44).
//
// Design choice — Elm-style Model/Update/View: every state change
// flows through a Msg into Update, which returns a (new model, cmd).
// Pure functions, no mutation, easy to test. Bubble Tea is the
// canonical Go TUI framework that bakes this pattern in.
package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Model is the entire TUI state. Today it has one field (quitting);
// over the next commits it grows to hold a textarea, viewport,
// message history, status info, and references to the agent loop.
//
// The zero value of Model is a valid initial state — initialModel()
// exists for symmetry with the convention the framework uses, not
// because anything needs initializing yet.
type Model struct {
	// quitting is set in Update when the user presses a quit key, and
	// read in View to skip the placeholder so the alt screen clears
	// cleanly on exit.
	quitting bool
}

// initialModel constructs the starting state. The conventional name
// in Bubble Tea apps is `initialModel`; tests can also call it to
// build a Model under test.
func initialModel() Model {
	return Model{}
}

// Init returns the first command to run when the program starts.
// Nothing yet — future commits return a textarea.Blink command here
// to start the input cursor blinking.
func (m Model) Init() tea.Cmd {
	return nil
}

// Update is the heart of the Elm architecture: take the current model
// + a message, return a new model + any side-effect command. Pure;
// never mutates the receiver.
//
// Bubble Tea fans messages from terminal input, timers, and any
// tea.Cmd callbacks through this single funnel — there's exactly one
// place where state transitions happen, which makes the program easy
// to reason about and easy to test.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		// Quit keys. Ctrl-C is the universal "stop" signal; q and Esc
		// are conveniences. Once the textarea lands (#40), q inside
		// the input field will be a literal character — we'll move
		// quit-on-q to a help-overlay focus mode and keep only Ctrl-C
		// as a global quit.
		switch msg.String() {
		case "ctrl+c", "q", "esc":
			m := m
			m.quitting = true
			return m, tea.Quit
		}
	}
	return m, nil
}

// View renders the current model to a string. Bubble Tea diffs the
// previous frame and only repaints what changed, so View can re-build
// from scratch on every call without performance worry.
//
// When quitting, returning an empty string lets the alt screen clean
// up without leaving a stale placeholder behind.
func (m Model) View() string {
	if m.quitting {
		return ""
	}
	return placeholderStyle.Render(
		"opendev-tui scaffold\n\n" +
			"This is the foundation commit for the TUI. The interactive interface " +
			"comes online over the next few commits — first a textarea + viewport, " +
			"then conversation history, then the agent loop wired in.\n\n" +
			"For now this binary just verifies the Bubble Tea framework is wired " +
			"and responsive.\n\n" +
			"Quit: q / Esc / Ctrl-C",
	)
}

// placeholderStyle is the only styling this commit needs. lipgloss
// renders ANSI escape sequences for color + padding. We're using a
// modest blue (256-color index 39) so it works in 256-color terminals
// AND in dark-on-light themes.
var placeholderStyle = lipgloss.NewStyle().
	Padding(1, 2).
	Foreground(lipgloss.Color("39"))

// main wires the Model into a tea.Program and runs it. Two options
// worth knowing:
//
//   - WithAltScreen swaps to the terminal's alternate screen buffer
//     (like vim or less). Restores the user's previous shell content
//     on exit instead of leaving the TUI's last frame in scrollback.
//   - WithMouseCellMotion enables mouse events. Not needed in this
//     commit but will be wired when we add scroll-by-mouse for the
//     viewport (#40+).
func main() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "opendev-tui:", err)
		os.Exit(1)
	}
}
