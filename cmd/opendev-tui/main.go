// Command opendev-tui is the Bubble Tea-based interactive front-end
// for the agent. It runs alongside cmd/opendev (the REPL) — both are
// real binaries against the same internal/agents core, so a reader of
// the repo can study either presentation layer in isolation.
//
// This commit (#40 in v2 Phase 1.5) wires up the two real widgets:
//   - bubbles/textarea: multi-line input at the bottom of the screen.
//   - bubbles/viewport: scrollable output area filling the rest.
//
// The agent loop is NOT yet wired (that lands in #42); submission
// here echoes the input into the viewport so the visual round-trip
// is observable. The data shape (a slice of history lines) is the
// minimum scaffolding #41 will replace with a typed Message struct.
//
// Design choice — Elm-style Model/Update/View: every state change
// flows through a Msg into Update, which returns a (new model, cmd).
// Pure functions, no mutation, easy to test. Bubble Tea is the
// canonical Go TUI framework that bakes this pattern in.
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Layout constants. Fixed values for now; later commits can make
// these dynamic if it feels wrong in practice.
const (
	// inputHeight is how many lines the textarea takes at the bottom.
	// Five fits a typical multi-line prompt without dominating the
	// screen. The textarea scrolls internally past five lines.
	inputHeight = 5

	// dividerHeight is the single line of "─" between viewport and
	// textarea. Visual separator only; no semantics.
	dividerHeight = 1
)

// Model is the entire TUI state. Compared to #39 (which had just one
// bool field) this commit grows the model to hold both real widgets,
// the terminal dimensions, and a history slice. The history is plain
// strings for now; #41 replaces it with a typed Message struct.
type Model struct {
	// textarea is the bottom input panel. Multi-line, with a placeholder,
	// Enter for newline, Ctrl-D for submit.
	textarea textarea.Model

	// viewport is the scrollable output area. Reads its content from
	// history (re-built on every submit).
	viewport viewport.Model

	// width / height track the terminal size, updated by every
	// tea.WindowSizeMsg. Both widgets are resized from these.
	width  int
	height int

	// history is the list of submitted lines, rendered into the
	// viewport. One entry per submit. Plain strings only — typed
	// roles arrive in #41.
	history []string

	// quitting flips true when a global-quit key arrives; View
	// returns "" to let the alt screen clean up.
	quitting bool
}

// initialModel constructs the starting state with both widgets ready.
// Dimensions are zero until the first tea.WindowSizeMsg arrives — that
// happens immediately on program start, so the user never sees a
// zero-sized panel.
func initialModel() Model {
	ta := textarea.New()
	ta.Placeholder = "Type a message — Ctrl-D submits, Ctrl-C quits"
	ta.CharLimit = 0 // unlimited; we cap effective size via the agent's token budget
	ta.SetHeight(inputHeight)
	ta.Focus()

	vp := viewport.New(0, 0) // sized properly on first WindowSizeMsg

	return Model{
		textarea: ta,
		viewport: vp,
	}
}

// Init returns the first command to run when the program starts.
// textarea.Blink ticks every ~0.5s to flip the cursor visibility,
// producing the blinking-cursor effect. Without this command, the
// cursor stays solid.
func (m Model) Init() tea.Cmd {
	return textarea.Blink
}

// Update is the heart of the Elm architecture: take the current model
// + a message, return a new model + any side-effect command. Pure;
// never mutates the receiver.
//
// Three classes of message we handle:
//   - tea.WindowSizeMsg: resize both widgets to fit the terminal.
//   - tea.KeyMsg: intercept Ctrl-C (quit) and Ctrl-D (submit) before
//     they reach the textarea; everything else forwards.
//   - everything else: forward to both widgets and batch their commands.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		// Re-size widgets. textarea uses SetWidth (pointer-receiver
		// method that recomputes wrap), viewport sizes via fields.
		m.textarea.SetWidth(m.width)
		m.viewport.Width = m.width
		m.viewport.Height = max(0, m.height-inputHeight-dividerHeight)

		// Re-flow viewport content because width changed.
		m.viewport.SetContent(m.renderHistory())
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			m.quitting = true
			return m, tea.Quit

		case "ctrl+d":
			// Submit: pull the textarea's content, push it into history,
			// clear the textarea, refresh and bottom-scroll the viewport.
			input := strings.TrimSpace(m.textarea.Value())
			if input == "" {
				// Nothing to submit. Empty Ctrl-D is a no-op rather
				// than an error — keeps the UX forgiving.
				return m, nil
			}
			m.history = append(m.history, "> "+input)
			m.textarea.Reset()
			m.viewport.SetContent(m.renderHistory())
			m.viewport.GotoBottom()
			return m, nil
		}
	}

	// Default path: forward to both widgets, batch their commands. Both
	// need to see most messages (KeyMsg routes into textarea for typing
	// and into viewport for scroll-bindings; WindowSizeMsg already
	// returned above so we're safe to forward here).
	var taCmd, vpCmd tea.Cmd
	m.textarea, taCmd = m.textarea.Update(msg)
	m.viewport, vpCmd = m.viewport.Update(msg)
	return m, tea.Batch(taCmd, vpCmd)
}

// renderHistory joins history entries with blank lines between them.
// This is the placeholder format — #41 will introduce role-aware
// rendering with distinct styles per message type.
func (m Model) renderHistory() string {
	if len(m.history) == 0 {
		return helpStyle.Render(
			"(no messages yet — type below and press Ctrl-D to submit)",
		)
	}
	return strings.Join(m.history, "\n\n")
}

// View renders the current model to a string. lipgloss.JoinVertical
// stacks the viewport, divider, and textarea with left-alignment.
// Bubble Tea diffs the previous frame and only repaints what changed.
func (m Model) View() string {
	if m.quitting {
		return ""
	}
	if m.width == 0 {
		// Pre-WindowSizeMsg: the very first frame before the runtime
		// reports the terminal size. Show a minimal hint so the
		// screen isn't blank.
		return "opendev-tui starting..."
	}
	divider := dividerStyle.Render(strings.Repeat("─", m.width))
	return lipgloss.JoinVertical(
		lipgloss.Left,
		m.viewport.View(),
		divider,
		m.textarea.View(),
	)
}

// dividerStyle paints the single ─ line between viewport and textarea
// in a low-contrast color so it reads as structure, not content.
var dividerStyle = lipgloss.NewStyle().
	Foreground(lipgloss.Color("240"))

// helpStyle dims the empty-viewport hint so it reads as meta, not as
// a real message.
var helpStyle = lipgloss.NewStyle().
	Foreground(lipgloss.Color("241")).
	Italic(true).
	Padding(1, 2)

// max is a tiny helper for the viewport sizing math. Go's stdlib has
// generic max in Go 1.21+, but using a local helper keeps the
// dependency on language-version features explicit and obvious.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// main wires the Model into a tea.Program and runs it. WithAltScreen
// swaps to the terminal's alternate screen buffer (like vim or less)
// so the TUI doesn't pollute the user's scrollback on exit.
func main() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "opendev-tui:", err)
		os.Exit(1)
	}
}
