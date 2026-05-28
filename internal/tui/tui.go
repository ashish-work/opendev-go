// Package tui implements the Bubble Tea interactive front-end for
// the agent. The binary at cmd/opendev-tui is a thin entry point
// that calls Run; everything substantive lives here.
//
// Why this lives in internal/ rather than under cmd/: Go convention
// puts only entry-point glue under cmd/ and the actual library code
// under internal/. The TUI is going to grow substantially (status
// line, tool cards, agent integration, streaming) — keeping the
// growth under internal/tui keeps cmd/opendev-tui/main.go tiny and
// reusable. Future binaries (a web server using the same TUI types
// over a WebSocket, for example) could import internal/tui without
// pulling in main.
//
// Internals are intentionally unexported (lowercase model, viewMessage,
// styles). Only Run is exported. If a later commit needs more surface
// — for example, pre-configuring the model with an agent loop and
// initial messages — we add it then.
//
// Design choice — Elm-style Model/Update/View: every state change
// flows through a Msg into Update, which returns a (new model, cmd).
// Pure functions, no mutation, easy to test. Bubble Tea is the
// canonical Go TUI framework that bakes this pattern in.
package tui

import (
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

// model is the entire TUI state. Compared to the scaffold commit
// (one bool field), it now holds both real widgets, the terminal
// dimensions, and a history slice. The history is plain strings for
// now; a later commit replaces it with a typed message type.
type model struct {
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

	// history is the list of messages rendered into the viewport,
	// one viewMessage per role-tagged turn. Until the agent is wired
	// in the next commit, each submit appends a real user message
	// plus two placeholder messages (tool and assistant) so all
	// three role styles are visible.
	history []viewMessage

	// quitting flips true when a global-quit key arrives; View
	// returns "" to let the alt screen clean up.
	quitting bool
}

// initialModel constructs the starting state with both widgets ready.
// Dimensions are zero until the first tea.WindowSizeMsg arrives — that
// happens immediately on program start, so the user never sees a
// zero-sized panel.
func initialModel() model {
	ta := textarea.New()
	ta.Placeholder = "Type a message — Ctrl-D submits, Ctrl-C quits"
	ta.CharLimit = 0 // unlimited; we cap effective size via the agent's token budget
	ta.SetHeight(inputHeight)
	ta.Focus()

	vp := viewport.New(0, 0) // sized properly on first WindowSizeMsg

	return model{
		textarea: ta,
		viewport: vp,
	}
}

// Init returns the first command to run when the program starts.
// textarea.Blink ticks every ~0.5s to flip the cursor visibility,
// producing the blinking-cursor effect. Without this command, the
// cursor stays solid.
func (m model) Init() tea.Cmd {
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
func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
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
			// Submit: pull the textarea's content, push the user message
			// into history along with placeholder tool + assistant
			// messages so the three role styles are visible from this
			// commit. The placeholders are stand-ins for what the real
			// agent loop produces; the next commit replaces them with
			// genuine turn output. Clear the textarea, refresh and
			// bottom-scroll the viewport.
			input := strings.TrimSpace(m.textarea.Value())
			if input == "" {
				// Nothing to submit. Empty Ctrl-D is a no-op rather
				// than an error — keeps the UX forgiving.
				return m, nil
			}
			m.history = append(m.history,
				viewMessage{role: roleUser, content: input},
				viewMessage{
					role:     roleTool,
					toolName: "echo",
					content:  "(placeholder tool output — the real agent wires up in the next commit)",
				},
				viewMessage{
					role:    roleAssistant,
					content: "(placeholder assistant reply — the real agent wires up in the next commit)",
				},
			)
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

// renderHistory walks history and produces the viewport content. Each
// entry renders independently via viewMessage.render, then they're
// joined with a blank line between blocks for visual breathing room.
// Empty history shows a help hint dimmed to read as meta.
func (m model) renderHistory() string {
	if len(m.history) == 0 {
		return helpStyle.Render(
			"(no messages yet — type below and press Ctrl-D to submit)",
		)
	}
	rendered := make([]string, len(m.history))
	for i, msg := range m.history {
		rendered[i] = msg.render(m.viewport.Width)
	}
	return strings.Join(rendered, "\n\n")
}

// View renders the current model to a string. lipgloss.JoinVertical
// stacks the viewport, divider, and textarea with left-alignment.
// Bubble Tea diffs the previous frame and only repaints what changed.
func (m model) View() string {
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

// Run starts the TUI. Blocks until the user exits (Ctrl-C, q, Esc).
// Returns nil on a clean exit; non-nil error if the program failed to
// initialize or the runtime crashed.
//
// This is the only exported function in the package. Future commits
// will extend the signature (e.g., to accept an *agents.ReactLoop)
// rather than exposing model construction directly.
func Run() error {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	_, err := p.Run()
	return err
}
