package tui

import (
	"context"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ashish-work/opendev-go/internal/agents"
	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/provider"
)

// streamSinkBufferSize is the capacity of the per-turn StreamEvent
// channel that flows from the agent loop into the TUI's Update. Bigger
// than the provider's 8-element buffer (a brief Update delay shouldn't
// back-pressure the network goroutine) but small enough that a truly
// stalled consumer surfaces as visible lag rather than unbounded
// memory growth.
const streamSinkBufferSize = 32

// turnCompleteMsg is the tea.Msg fired when an agent turn finishes —
// success, error, or cancellation. The Update handler reads it and
// transitions out of thinking state.
//
// Tagged via a typed message struct rather than a sentinel value so a
// future commit can add fields (e.g. partial streaming chunks, mid-
// turn cost updates) without breaking the dispatch shape.
type turnCompleteMsg struct {
	// result is the agent loop's full outcome. Even on error the loop
	// returns a Result with the messages it accumulated up to the
	// failure point — those are still worth rendering.
	result agents.Result

	// tracker is the cost/token tally after this turn. Folded into the
	// session totals shown on the status bar.
	tracker cost.Tracker

	// err is non-nil on any non-success exit:
	//   - context.Canceled (user pressed Ctrl-C)
	//   - context.DeadlineExceeded
	//   - agents.ErrLLM, agents.ErrToolExec, agents.ErrMaxIterations,
	//     agents.ErrInterrupted, agents.ErrDoomLoop
	//   - APIError wrapping HTTP failures
	err error
}

// streamEventMsg carries one provider.StreamEvent into Update. The
// Update handler interprets the event to update the live transcript
// (text deltas accumulate into a pending assistant message; tool
// call starts append a placeholder).
//
// Wrapping the event in a tea.Msg type lets Update dispatch off the
// struct's identity in the normal type-switch; raw provider.StreamEvent
// could clash with future provider message types from elsewhere.
type streamEventMsg struct {
	event provider.StreamEvent
}

// streamSinkClosedMsg signals that the per-turn StreamEvent channel
// has closed — either because the turn completed (Update closed it on
// turnCompleteMsg) or because the producer stopped writing. Update
// uses it to stop chaining nextStreamEventCmd; without this signal the
// chain would block forever on a closed channel after the turn ends.
type streamSinkClosedMsg struct{}

// runTurnCmd builds the tea.Cmd that actually invokes the agent loop.
// Bubble Tea's runtime executes Cmds in a goroutine and feeds the
// returned Msg back into Update; this is the framework's only way to
// safely do off-UI-thread work. The ctx threaded in here is the
// cancellable context the model retains a CancelFunc for, so a Ctrl-C
// while thinking calls cancel() and the goroutine unwinds via
// ctx.Err().
//
// sink is the per-turn channel the loop will stream events to. nil
// runs the non-streaming path (Run rather than RunWithStream); the
// TUI always passes a real channel.
func runTurnCmd(
	ctx context.Context,
	loop *agents.ReactLoop,
	input string,
	sink chan<- provider.StreamEvent,
) tea.Cmd {
	return func() tea.Msg {
		result, tracker, err := loop.RunWithStream(ctx, input, sink)
		return turnCompleteMsg{
			result:  result,
			tracker: tracker,
			err:     err,
		}
	}
}

// nextStreamEventCmd reads one event from sink and produces a
// streamEventMsg (or streamSinkClosedMsg on close). Update re-issues
// this Cmd after handling each event to drive the read loop. This is
// Bubble Tea's standard channel-to-message bridge.
//
// Reading one event per Cmd invocation rather than draining in a
// goroutine lets Update see events serialized through the same
// message queue as everything else (key presses, window resizes), so
// rendering stays consistent.
func nextStreamEventCmd(sink <-chan provider.StreamEvent) tea.Cmd {
	return func() tea.Msg {
		ev, ok := <-sink
		if !ok {
			return streamSinkClosedMsg{}
		}
		return streamEventMsg{event: ev}
	}
}
