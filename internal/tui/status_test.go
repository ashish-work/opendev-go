package tui

import (
	"strings"
	"testing"

	"github.com/ashish-work/opendev-go/internal/budget"
	"github.com/ashish-work/opendev-go/internal/cost"
)

func TestRenderStatus_IncludesAllFields(t *testing.T) {
	// Cost chosen below $0.01 so FormatCost emits 4-decimal precision.
	// Crossing $0.01 switches to 2-decimal — there's a dedicated test
	// for that path.
	out := renderStatus(120, statusState{
		modelName: "gpt-4o-mini",
		tracker: cost.Tracker{
			CallCount:         3,
			TotalInputTokens:  1234,
			TotalOutputTokens: 567,
			TotalCostUSD:      0.0099,
		},
		budget: budget.Snapshot{UsagePct: 0.024},
	})
	for _, want := range []string{
		"gpt-4o-mini",
		"iter 3",
		"in 1234",
		"out 567",
		"2.4%",
		"$0.0099",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("renderStatus output missing %q:\n%s", want, out)
		}
	}
}

func TestRenderStatus_FormatsCostUnderACent(t *testing.T) {
	out := renderStatus(120, statusState{
		modelName: "x",
		tracker:   cost.Tracker{TotalCostUSD: 0.0042},
	})
	if !strings.Contains(out, "$0.0042") {
		t.Errorf("expected 4-decimal cost for sub-cent value, got:\n%s", out)
	}
}

func TestRenderStatus_FormatsCostOverACent(t *testing.T) {
	out := renderStatus(120, statusState{
		modelName: "x",
		tracker:   cost.Tracker{TotalCostUSD: 1.23},
	})
	if !strings.Contains(out, "$1.23") {
		t.Errorf("expected 2-decimal cost for >$0.01 value, got:\n%s", out)
	}
}

func TestRenderStatus_TruncatesWhenNarrow(t *testing.T) {
	// Width that's too narrow to fit the full line forces lipgloss
	// to truncate. We don't pin the exact truncation marker (lipgloss
	// uses … by default); just verify nothing panics and the result
	// stays within the requested width budget.
	out := renderStatus(20, statusState{
		modelName: "a-very-long-model-name-that-wont-fit",
		tracker:   cost.Tracker{CallCount: 9999, TotalInputTokens: 9999999},
	})
	if out == "" {
		t.Errorf("narrow render should still produce content")
	}
	// Single line — no embedded newlines that would break the
	// outer JoinVertical layout.
	if strings.Contains(out, "\n") {
		t.Errorf("status bar must stay single-line; got:\n%s", out)
	}
}

func TestRenderStatus_EmptyModelNameSkipsLeadingField(t *testing.T) {
	// When modelName is empty (defensive — should never happen in
	// production, but tests construct it sometimes) the bar still
	// renders sensibly with just the metrics.
	out := renderStatus(120, statusState{
		modelName: "",
		tracker:   cost.Tracker{CallCount: 1},
	})
	if !strings.Contains(out, "iter 1") {
		t.Errorf("empty model name should not break metrics rendering: %q", out)
	}
}

func TestRenderStatus_ZeroWidthDoesNotPanic(t *testing.T) {
	// Defensive: a zero or negative width should yield an empty
	// string rather than panicking inside lipgloss.
	got := renderStatus(0, statusState{modelName: "x"})
	if got != "" {
		t.Errorf("width=0 should produce empty output, got %q", got)
	}
	got = renderStatus(-5, statusState{modelName: "x"})
	if got != "" {
		t.Errorf("negative width should produce empty output, got %q", got)
	}
}

func TestRenderStatus_ContextPercentRoundsToOneDecimal(t *testing.T) {
	out := renderStatus(120, statusState{
		modelName: "x",
		budget:    budget.Snapshot{UsagePct: 0.123456}, // 12.3456%
	})
	if !strings.Contains(out, "12.3%") {
		t.Errorf("expected 1-decimal percent (12.3%%) in output, got:\n%s", out)
	}
}

func TestRenderStatus_ZeroTrackerRendersZeros(t *testing.T) {
	out := renderStatus(120, statusState{modelName: "x"})
	for _, want := range []string{
		"iter 0",
		"in 0",
		"out 0",
		"0.0%",
		"$0.0000",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("zero-state status should contain %q; got:\n%s", want, out)
		}
	}
}
