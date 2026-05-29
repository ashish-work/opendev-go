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
	"github.com/ashish-work/opendev-go/internal/budget"
	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/hooks"
	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/session"
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

	// toolsExpanded selects how every tool message in history is
	// rendered. False (default) collapses each tool card to its
	// first few lines plus a "(… N more)" hint; true expands them
	// all to full content. Toggled by Ctrl-T.
	//
	// Single global flag rather than per-card state: simpler to
	// implement and tests, demonstrates the collapse pattern
	// cleanly. Per-card focus + Tab cycling is a clean follow-up
	// that builds on this without rewriting it.
	toolsExpanded bool

	// quitting flips true when a global-quit key arrives; View
	// returns "" to let the alt screen clean up.
	quitting bool

	// modelName is the model identifier the binary launched against
	// (e.g. "gpt-4o-mini"). Stable for the session; surfaced in the
	// status bar.
	modelName string

	// tracker accumulates cost/token totals across every turn this
	// session. The agent loop returns a FRESH tracker per turn; we
	// fold each turn's tracker into m.tracker so the status bar
	// shows cumulative session totals instead of per-turn values.
	tracker cost.Tracker

	// lastBudget is the most recent context-window snapshot,
	// captured from each turn's Result.Budget. Drives the "ctx N.N%"
	// display on the status bar.
	lastBudget budget.Snapshot

	// streamCh is the per-turn StreamEvent channel the agent loop
	// writes to. Lifecycle: created at Ctrl-D submit, closed at
	// turnCompleteMsg. nil while idle. Owned by the model — only
	// Update reads it (via the nextStreamEventCmd Cmd), only the
	// agent loop writes to it.
	streamCh chan provider.StreamEvent

	// pendingAssistantIdx is the index into history of the
	// in-progress assistant viewMessage currently absorbing
	// streamed TextDelta events. -1 means no message is in progress
	// (next TextDelta will append a new one). Reset on ToolCallStart
	// and on turnCompleteMsg so subsequent text starts a fresh
	// assistant entry rather than concatenating into the previous
	// iteration's reply.
	pendingAssistantIdx int

	// hookManager is the optional lifecycle hook manager. nil is
	// valid and means "no hooks fire" — the Ctrl-D handler skips
	// the prompt-submit step and runs the turn directly. Set via
	// Run.
	hookManager *hooks.Manager

	// session carries the per-run identity passed to lifecycle hook
	// payloads (UserPromptSubmit, Stop). nil when hookManager is
	// nil. Set via Run.
	session *session.Session

	// pendingTurnCtx is the context the in-flight turn will run
	// against, captured when Ctrl-D fires and held until
	// promptSubmitResultMsg dispatches the turn. nil while idle.
	// Bubble Tea messages can't carry context.Context cleanly, so
	// we stash it on the model rather than threading through Msg.
	pendingTurnCtx context.Context

	// pendingTurnInput is the user's prompt held between Ctrl-D
	// and promptSubmitResultMsg. Empty while idle.
	pendingTurnInput string
}

// initialModel constructs the starting state with both widgets ready,
// the agent loop attached, and the model name stashed for the status
// bar. Dimensions are zero until the first tea.WindowSizeMsg arrives
// — that happens immediately on program start, so the user never
// sees a zero-sized panel.
//
// hookManager and sess are optional. When both are non-nil the
// Ctrl-D handler fires UserPromptSubmit before running each turn
// (deny short-circuits with an in-history notice); when either is
// nil the handler runs the turn directly.
func initialModel(loop *agents.ReactLoop, modelName string, hookManager *hooks.Manager, sess *session.Session) model {
	ta := textarea.New()
	ta.Placeholder = "Type — Ctrl-D submit · Ctrl-T toggle tool details · PgUp/PgDn scroll · Ctrl-C cancel/quit"
	ta.CharLimit = 0 // unlimited; we cap effective size via the agent's token budget
	ta.SetHeight(inputHeight)
	ta.Focus()

	vp := viewport.New(0, 0) // sized properly on first WindowSizeMsg

	return model{
		textarea:            ta,
		viewport:            vp,
		loop:                loop,
		modelName:           modelName,
		pendingAssistantIdx: -1,
		hookManager:         hookManager,
		session:             sess,
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
		// Viewport height carves out the status bar (1 row),
		// divider (1 row), and textarea (5 rows) from the total
		// terminal height.
		m.textarea.SetWidth(m.width)
		m.viewport.Width = m.width
		m.viewport.Height = max(0, m.height-statusBarHeight-inputHeight-dividerHeight)

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

		case "ctrl+t":
			// Toggle the global "expand all tool cards" flag and
			// repaint. The toggle is allowed at any time — during a
			// turn it just affects the visual state of completed
			// tool messages; while idle it's a pure UX preference.
			m.toolsExpanded = !m.toolsExpanded
			m.viewport.SetContent(m.renderHistory())
			return m, nil

		case "pgup", "pgdown":
			// Scroll the viewport ONLY, not the textarea. Without
			// this intercept the keys would forward to both widgets
			// (the default below) and the textarea would jump its
			// cursor by page in lockstep — split attention. Lets
			// the user scroll up through history (e.g. to see a
			// tool card that scrolled off the top of the screen)
			// without losing their place in the input box.
			var vpCmd tea.Cmd
			m.viewport, vpCmd = m.viewport.Update(msg)
			return m, vpCmd

		case "ctrl+home":
			// Jump to the top of the viewport (start of conversation).
			// Useful when the model dumped a huge response and you
			// want to read it from the beginning. Ctrl-Home not
			// plain Home because Home is textarea's start-of-line.
			m.viewport.GotoTop()
			return m, nil

		case "ctrl+end":
			// Jump to the bottom of the viewport (latest content).
			// Mirror of Ctrl-Home for completeness.
			m.viewport.GotoBottom()
			return m, nil

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

			// Build the cancellable context so Ctrl-C can interrupt
			// either the prompt-submit hook or the turn itself.
			ctx, cancel := context.WithCancel(context.Background())
			m.turnCancel = cancel
			m.thinking = true
			m.viewport.SetContent(m.renderHistory())
			m.viewport.GotoBottom()

			// Fast path: no hook manager → start the turn directly,
			// same Cmd shape as before #35.
			if m.hookManager == nil || m.session == nil {
				m.streamCh = make(chan provider.StreamEvent, streamSinkBufferSize)
				m.pendingAssistantIdx = -1
				return m, tea.Batch(
					runTurnCmd(ctx, m.loop, input, m.streamCh),
					nextStreamEventCmd(m.streamCh),
				)
			}

			// Slow path: stash ctx + input on the model, fire the
			// prompt-submit hook async, await promptSubmitResultMsg.
			m.pendingTurnCtx = ctx
			m.pendingTurnInput = input
			return m, firePromptSubmitCmd(ctx, m.session, m.hookManager, input)
		}

	case promptSubmitResultMsg:
		if msg.denied {
			// Show the deny reason inline; user keeps interacting.
			m.history = append(m.history, viewMessage{
				role:    roleAssistant,
				content: fmt.Sprintf("(denied by hook: %s)", msg.reason),
			})
			m.thinking = false
			if m.turnCancel != nil {
				m.turnCancel()
				m.turnCancel = nil
			}
			m.pendingTurnCtx = nil
			m.pendingTurnInput = ""
			m.viewport.SetContent(m.renderHistory())
			m.viewport.GotoBottom()
			return m, nil
		}
		// Approved: start the turn with the (possibly rewritten) prompt.
		m.streamCh = make(chan provider.StreamEvent, streamSinkBufferSize)
		m.pendingAssistantIdx = -1
		ctx := m.pendingTurnCtx
		m.pendingTurnCtx = nil
		m.pendingTurnInput = ""
		return m, tea.Batch(
			runTurnCmd(ctx, m.loop, msg.prompt, m.streamCh),
			nextStreamEventCmd(m.streamCh),
		)

	case stopHookDoneMsg:
		// Stop hook is fire-and-forget; no UI work needed.
		return m, nil

	case streamEventMsg:
		// One streamed event lands. Update the live transcript and
		// re-arm the read Cmd so the next event flows in.
		m = applyStreamEvent(m, msg.event)
		m.viewport.SetContent(m.renderHistory())
		m.viewport.GotoBottom()
		// Re-issue the read Cmd if the channel is still alive. A nil
		// channel here means turnCompleteMsg already cleaned up, which
		// shouldn't normally happen (turnCompleteMsg closes the
		// channel, which produces streamSinkClosedMsg, not another
		// streamEventMsg), but we're defensive.
		if m.streamCh == nil {
			return m, nil
		}
		return m, nextStreamEventCmd(m.streamCh)

	case streamSinkClosedMsg:
		// Channel closed. Stop chaining reads. turnCompleteMsg arrives
		// (or already arrived) on a separate path; no UI work needed
		// here. Leaving m.streamCh as-is until turnCompleteMsg
		// finalizes is fine — Update never reads from it directly.
		return m, nil

	case turnCompleteMsg:
		// Turn finished — success, error, or cancellation. Drop the
		// thinking state and the (now-fired) cancel func before
		// inspecting the result. Close the stream channel so the
		// pending nextStreamEventCmd unblocks with
		// streamSinkClosedMsg.
		m.thinking = false
		m.turnCancel = nil
		if m.streamCh != nil {
			close(m.streamCh)
			m.streamCh = nil
		}
		m.pendingAssistantIdx = -1

		// Fold the per-turn tracker into the session tracker so the
		// status bar shows CUMULATIVE totals (each loop.Run returns
		// a fresh tracker holding only that turn's data). BudgetUSD
		// is per-session config, not a sum, so we preserve the
		// existing value rather than adding.
		m.tracker.TotalInputTokens += msg.tracker.TotalInputTokens
		m.tracker.TotalOutputTokens += msg.tracker.TotalOutputTokens
		m.tracker.TotalCacheReadTokens += msg.tracker.TotalCacheReadTokens
		m.tracker.TotalCostUSD += msg.tracker.TotalCostUSD
		m.tracker.CallCount += msg.tracker.CallCount

		// The latest Budget snapshot is the freshest read on
		// context-window load; status bar wants that, not an
		// accumulated value.
		m.lastBudget = msg.result.Budget

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

		// Fire Stop hook (fire-and-forget; UI returns immediately).
		// Skip the Cmd when hooks aren't wired so we don't spawn
		// a goroutine to do nothing.
		if m.hookManager != nil && m.session != nil {
			var replyText, errStr string
			if msg.err != nil {
				errStr = msg.err.Error()
			} else {
				replyText = msg.result.Content
			}
			return m, fireStopCmd(context.Background(), m.session, m.hookManager, replyText, errStr)
		}
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
		rendered = append(rendered, msg.render(m.viewport.Width, m.toolsExpanded))
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
// stacks the status bar, viewport, divider, and textarea with left
// alignment. Bubble Tea diffs the previous frame and only repaints
// what changed.
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
	status := renderStatus(m.width, statusState{
		modelName: m.modelName,
		tracker:   m.tracker,
		budget:    m.lastBudget,
	})
	divider := dividerStyle.Render(strings.Repeat("─", m.width))
	return lipgloss.JoinVertical(
		lipgloss.Left,
		status,
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

// applyStreamEvent folds one StreamEvent into the model's transcript.
// Returns the updated model. Pure transformation; caller refreshes
// the viewport.
//
// Event handling:
//   - TextDelta: extend the pending assistant message, or create one
//     when none is in progress.
//   - ToolCallStart: reset the pending assistant (so the next
//     TextDelta starts fresh) and append a placeholder tool entry.
//   - ToolCallDone: replace the placeholder body with the assembled
//     JSON arguments (if any) so the user sees what was invoked.
//   - Error: append an "(stream error: ...)" assistant message; reset
//     the pending state so the next iteration starts cleanly.
//   - Everything else (Usage, Done, ReasoningDelta, ToolCallDelta):
//     ignored at the live-transcript level. Usage flows in via
//     turnCompleteMsg; Done is just a turn-iteration boundary;
//     ReasoningDelta has no dedicated display slot yet; ToolCallDelta
//     fragments would render as ugly partial JSON.
func applyStreamEvent(m model, ev provider.StreamEvent) model {
	switch ev.Kind {
	case provider.StreamEventTextDelta:
		if m.pendingAssistantIdx == -1 {
			m.history = append(m.history, viewMessage{
				role:    roleAssistant,
				content: ev.Text,
			})
			m.pendingAssistantIdx = len(m.history) - 1
		} else {
			m.history[m.pendingAssistantIdx].content += ev.Text
		}

	case provider.StreamEventToolCallStart:
		// Reset the pending assistant slot so any further text deltas
		// start a NEW assistant message rather than appending into the
		// previous iteration's reply.
		m.pendingAssistantIdx = -1
		m.history = append(m.history, viewMessage{
			role:     roleTool,
			toolName: ev.ToolCall.Name,
			content:  "(running...)",
		})

	case provider.StreamEventToolCallDone:
		// Find the most recent tool message with this name and swap
		// its content from "(running...)" to the assembled args. This
		// is a heuristic but works in practice because a single tool
		// is typically invoked once per iteration; if the model calls
		// the same tool twice, the second invocation just gets a
		// fresh "(running...)" card from the next ToolCallStart.
		for i := len(m.history) - 1; i >= 0; i-- {
			if m.history[i].role == roleTool && m.history[i].toolName == ev.ToolCall.Name &&
				m.history[i].content == "(running...)" {
				if ev.ToolCall.Arguments != "" {
					m.history[i].content = ev.ToolCall.Arguments
				} else {
					m.history[i].content = "(no args)"
				}
				break
			}
		}

	case provider.StreamEventError:
		m.pendingAssistantIdx = -1
		errText := "(stream error)"
		if ev.Err != nil {
			errText = "(stream error: " + ev.Err.Error() + ")"
		}
		m.history = append(m.history, viewMessage{
			role:    roleAssistant,
			content: errText,
		})

	default:
		// Done / Usage / ReasoningDelta / ToolCallDelta: nothing to do
		// at the live-transcript level for this commit. Adding hooks
		// here (reasoning panels, mid-stream usage refresh, etc.) is a
		// follow-up.
	}
	return m
}

// Run starts the TUI against the given agent loop. Blocks until the
// user exits (Ctrl-C while idle). Returns nil on a clean exit; non-
// nil error if the program failed to initialize or the runtime
// crashed. The loop must be fully constructed — Provider, registry,
// workflow config — before Run is called; the TUI owns no
// dependency wiring of its own.
//
// modelName is shown in the status bar. It's passed in (rather than
// pulled from loop.Config.Workflow) so the binary stays the single
// source of truth for its own configuration.
//
// hookManager and sess are optional. Both nil means hooks don't
// fire; when both are non-nil the TUI fires UserPromptSubmit before
// each turn and Stop after each turn completes. SessionStart and
// SessionEnd are fired by the binary itself (before/after Run)
// since they bracket the whole TUI runtime.
func Run(loop *agents.ReactLoop, modelName string, sess *session.Session, hookManager *hooks.Manager) error {
	p := tea.NewProgram(initialModel(loop, modelName, hookManager, sess), tea.WithAltScreen())
	_, err := p.Run()
	return err
}
