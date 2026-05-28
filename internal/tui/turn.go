package tui

import (
	"context"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ashish-work/opendev-go/internal/agents"
	"github.com/ashish-work/opendev-go/internal/cost"
)

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

	// tracker is the cost/token tally after this turn. Drawn into the
	// status line in a later commit.
	tracker cost.Tracker

	// err is non-nil on any non-success exit:
	//   - context.Canceled (user pressed Ctrl-C)
	//   - context.DeadlineExceeded
	//   - agents.ErrLLM, agents.ErrToolExec, agents.ErrMaxIterations,
	//     agents.ErrInterrupted, agents.ErrDoomLoop
	//   - APIError wrapping HTTP failures
	err error
}

// runTurnCmd builds the tea.Cmd that actually invokes the agent loop.
// Bubble Tea's runtime executes Cmds in a goroutine and feeds the
// returned Msg back into Update; this is the framework's only way to
// safely do off-UI-thread work. The ctx threaded in here is the cancel-
// able context the model retains a CancelFunc for, so a Ctrl-C while
// thinking calls cancel() and the goroutine unwinds via ctx.Err().
//
// The ReactLoop.Run signature is (ctx, userTask) → (Result, Tracker,
// error). We surface all three on the Msg even when err is non-nil,
// since the loop still produces useful state on partial failures.
func runTurnCmd(ctx context.Context, loop *agents.ReactLoop, input string) tea.Cmd {
	return func() tea.Msg {
		result, tracker, err := loop.Run(ctx, input)
		return turnCompleteMsg{
			result:  result,
			tracker: tracker,
			err:     err,
		}
	}
}
