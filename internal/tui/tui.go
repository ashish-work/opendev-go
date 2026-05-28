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
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ashish-work/opendev-go/internal/agents"
	"github.com/ashish-work/opendev-go/internal/provider"
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

// model is the entire TUI state. Holds both widgets, the terminal
// dimensions, the message history, the agent loop (wired by Run),
// and the in-flight turn's cancel function while one is running.
type model struct {
	// textarea is the bottom input panel. Multi-line, with a placeholder,
	// Enter for newline, Ctrl-D for submit.
	textarea textarea.Model

	// viewport is the scrollable output area. Reads its content from
	// history (re-built on every submit and turn completion).
	viewport viewport.Model

	// width / height track the terminal size, updated by every
	// tea.WindowSizeMsg. Both widgets are resized from these.
	width  int
	height int

	// history is the rendered transcript. Each Ctrl-D submit appends
	// the user message optimistically; when the turn completes we
	// REPLACE history with the loop's full message list (translated)
	// so the user message stays consistent with the agent's record.
	history []viewMessage

	// loop is the agent loop the TUI invokes on every submit. Wired
	// at construction time; never changes after Run.
	loop *agents.ReactLoop

	// thinking is true between the moment we dispatch a turn and the
	// moment turnCompleteMsg arrives. While thinking, the viewport
	// shows a "⋯ thinking" indicator and Ctrl-C cancels the turn
	// instead of quitting.
	thinking bool

	// turnCancel cancels the context the in-flight turn is running
	// against. Set when we dispatch a turn, called on Ctrl-C during
	// thinking, cleared when the turn completes.
	turnCancel context.CancelFunc

	// quitting flips true when a global-quit key arrives; View
	// returns "" to let the alt screen clean up.
	quitting bool
}

// initialModel constructs the starting state with both widgets ready
// and the agent loop attached. Dimensions are zero until the first
// tea.WindowSizeMsg arrives — that happens immediately on program
// start, so the user never sees a zero-sized panel.
func initialModel(loop *agents.ReactLoop) model {
	ta := textarea.New()
	ta.Placeholder = "Type a message — Ctrl-D submits, Ctrl-C cancels/quits"
	ta.CharLimit = 0 // unlimited; we cap effective size via the agent's token budget
	ta.SetHeight(inputHeight)
	ta.Focus()

	vp := viewport.New(0, 0) // sized properly on first WindowSizeMsg

	return model{
		textarea: ta,
		viewport: vp,
		loop:     loop,
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
			// Two-mode Ctrl-C: cancel an in-flight turn if there is one,
			// otherwise quit the program. The first press during a turn
			// unwinds the loop cleanly via context cancellation; only
			// when no turn is running does Ctrl-C terminate the binary.
			if m.thinking {
				if m.turnCancel != nil {
					m.turnCancel()
				}
				// Don't quit; the runTurnCmd goroutine will still emit
				// a turnCompleteMsg with err = context.Canceled, which
				// we handle below to update history + state.
				return m, nil
			}
			m.quitting = true
			return m, tea.Quit

		case "ctrl+d":
			// Reject a submit while a turn is already running. The user
			// can press Ctrl-C to cancel the current turn first.
			if m.thinking {
				return m, nil
			}
			input := strings.TrimSpace(m.textarea.Value())
			if input == "" {
				// Empty Ctrl-D is a no-op rather than an error — keeps
				// the UX forgiving.
				return m, nil
			}
			// Optimistic append: show the user's message in the
			// viewport immediately so they see something happen even
			// before the loop returns. When the turn completes we
			// replace history with the loop's full record (which
			// includes this same user message), so there's no drift.
			m.history = append(m.history, viewMessage{role: roleUser, content: input})
			m.textarea.Reset()

			// Build the cancellable context, store the cancel func so
			// Ctrl-C can call it, and dispatch the turn.
			ctx, cancel := context.WithCancel(context.Background())
			m.turnCancel = cancel
			m.thinking = true

			m.viewport.SetContent(m.renderHistory())
			m.viewport.GotoBottom()

			return m, runTurnCmd(ctx, m.loop, input)
		}

	case turnCompleteMsg:
		// Turn finished — success, error, or cancellation. Drop the
		// thinking state and the (now-fired) cancel func before
		// inspecting the result.
		m.thinking = false
		m.turnCancel = nil

		switch {
		case msg.err == nil:
			// Success path: replace history with the agent's full
			// message list, translated to viewMessages. Replacement
			// rather than append because Result.Messages includes
			// the user message we already optimistically appended;
			// re-rendering from the authoritative loop record keeps
			// the transcript clean.
			m.history = translateMessages(msg.result.Messages)

		case errors.Is(msg.err, agents.ErrInterrupted) || errors.Is(msg.err, context.Canceled):
			// User cancelled. The loop wraps ctx.Canceled with
			// agents.ErrInterrupted (its sentinel for "user pressed
			// Ctrl-C") — that's our primary signal. We also check
			// context.Canceled defensively in case a future code
			// path returns it un-wrapped. Either way, append a
			// brief notice so the user knows it landed; preserve
			// whatever messages the loop did produce before
			// cancellation (some tool calls may have completed).
			m.history = appendOrReplaceHistory(m.history, msg.result.Messages)
			m.history = append(m.history, viewMessage{
				role:    roleAssistant,
				content: "(turn cancelled)",
			})

		default:
			// Any other failure (API error, max-iter, doomloop, tool-
			// exec). Show the error so the user can react. Preserve
			// partial loop messages too.
			m.history = appendOrReplaceHistory(m.history, msg.result.Messages)
			m.history = append(m.history, viewMessage{
				role:    roleAssistant,
				content: fmt.Sprintf("(error: %v)", msg.err),
			})
		}

		m.viewport.SetContent(m.renderHistory())
		m.viewport.GotoBottom()
		return m, nil
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
// While a turn is in flight a "⋯ thinking" indicator follows the last
// entry; it disappears when the turn completes and the view repaints.
// Empty history shows a help hint dimmed to read as meta.
func (m model) renderHistory() string {
	if len(m.history) == 0 && !m.thinking {
		return helpStyle.Render(
			"(no messages yet — type below and press Ctrl-D to submit)",
		)
	}
	rendered := make([]string, 0, len(m.history)+1)
	for _, msg := range m.history {
		rendered = append(rendered, msg.render(m.viewport.Width))
	}
	if m.thinking {
		rendered = append(rendered, thinkingStyle.Render("⋯ thinking — Ctrl-C to cancel"))
	}
	return strings.Join(rendered, "\n\n")
}

// appendOrReplaceHistory picks the better of two histories when a turn
// fails partway through. If the loop produced at least the user
// message (so its slice is at least as informative as the optimistic
// one), use the translated loop messages. Otherwise keep the
// optimistic history that has the user's most recent submit. This
// avoids "user message disappears after Ctrl-C" while still surfacing
// any tool messages the loop captured before erroring.
func appendOrReplaceHistory(existing []viewMessage, loopMsgs []provider.Message) []viewMessage {
	translated := translateMessages(loopMsgs)
	if len(translated) >= len(existing) {
		return translated
	}
	return existing
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

// thinkingStyle paints the "⋯ thinking" indicator in a muted color so
// it reads as state, not content. Lands at the tail of viewport
// output while a turn is in flight.
var thinkingStyle = lipgloss.NewStyle().
	Foreground(lipgloss.Color("245")).
	Italic(true)

// max is a tiny helper for the viewport sizing math. Go's stdlib has
// generic max in Go 1.21+, but using a local helper keeps the
// dependency on language-version features explicit and obvious.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Run starts the TUI against the given agent loop. Blocks until the
// user exits (Ctrl-C while idle). Returns nil on a clean exit; non-
// nil error if the program failed to initialize or the runtime
// crashed. The loop must be fully constructed — Provider, registry,
// workflow config — before Run is called; the TUI owns no
// dependency wiring of its own.
func Run(loop *agents.ReactLoop) error {
	p := tea.NewProgram(initialModel(loop), tea.WithAltScreen())
	_, err := p.Run()
	return err
}
