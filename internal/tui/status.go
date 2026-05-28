package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/ashish-work/opendev-go/internal/budget"
	"github.com/ashish-work/opendev-go/internal/cost"
)

// statusBarHeight is the number of rows the status bar occupies. One
// line is enough — the bar is information-dense but compact. If we
// ever want a multi-line bar (e.g. with git branch + MCP status on a
// second row), bumping this value flows through to the viewport
// height calculation automatically.
const statusBarHeight = 1

// statusState bundles everything the bar needs to render. Passed as
// a value (small struct) to renderStatus so the rendering function
// stays a pure transform on its inputs — no need for it to touch the
// full model type.
type statusState struct {
	// modelName is the model the binary launched against. Stable for
	// the whole session.
	modelName string

	// tracker carries the cumulative session totals (iterations, in/
	// out tokens, cost). Different from the per-turn Tracker the
	// loop returns — see Update's turnCompleteMsg handler for the
	// accumulation logic.
	tracker cost.Tracker

	// budget is the most recent context-window snapshot. Reset to
	// zero between Reported/Estimated/UsagePct before the first turn
	// completes; that's fine — the bar shows 0.0% in that state.
	budget budget.Snapshot
}

// renderStatus produces the single-line top status bar. Width is the
// terminal's full width; the bar fills it edge-to-edge with a styled
// background so the metrics are visually anchored. Truncates with an
// ellipsis when the content is wider than the terminal.
//
// Field order chosen for stability: identity (model) first, work
// done (iter / tokens) next, current load (ctx%) third, cost last.
// Cost lives at the tail because it's the eye-catching number the
// user glances at most — putting it at the end gives a consistent
// rightward reading destination, even though we left-align overall.
func renderStatus(width int, s statusState) string {
	if width < 1 {
		// Defensive: avoid passing zero/negative to lipgloss.Width
		// (which would result in a 0-col render that confuses the
		// outer JoinVertical).
		return ""
	}

	parts := []string{}
	if s.modelName != "" {
		parts = append(parts, s.modelName)
	}
	parts = append(parts,
		fmt.Sprintf("iter %d", s.tracker.CallCount),
		fmt.Sprintf("in %d", s.tracker.TotalInputTokens),
		fmt.Sprintf("out %d", s.tracker.TotalOutputTokens),
		fmt.Sprintf("ctx %.1f%%", s.budget.UsagePct*100),
		s.tracker.FormatCost(),
	)
	line := " " + strings.Join(parts, " · ") // leading space for padding inside the bar

	// Single-line truncation: lipgloss's MaxWidth WRAPS rather than
	// truncates, which would break the layout (status bar must stay
	// exactly one row). Do it ourselves: count runes, cap at width
	// (leaving room for the ellipsis when truncating). Plain text at
	// this point — no ANSI to navigate — so rune count is the visible
	// width.
	if runes := []rune(line); len(runes) > width {
		switch {
		case width <= 1:
			line = "…"
		default:
			line = string(runes[:width-1]) + "…"
		}
	}

	return statusBarStyle.Width(width).Render(line)
}

// statusBarStyle paints the bar's background and default text color.
// Dark gray background (236) + light foreground (252) is the
// conventional Vim/IDE status-bar look: distinct from the terminal
// background but not aggressive. One style for the whole bar —
// dimmer separator dots were tried but coloring them differently
// required ANSI-aware truncation (lipgloss inflates byte length),
// which we sidestep by keeping the line plain-text until the very
// last Render call.
var statusBarStyle = lipgloss.NewStyle().
	Foreground(lipgloss.Color("252")).
	Background(lipgloss.Color("236"))
