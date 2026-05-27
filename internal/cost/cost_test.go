package cost

import (
	"math"
	"testing"
)

const floatTolerance = 1e-9

func floatEq(a, b float64) bool {
	return math.Abs(a-b) < floatTolerance
}

// TestTrackerRecordUsage covers the main cost-computation branches:
// simple input+output, cache-read discount, 200K-threshold tiered
// pricing, and zero-pricing (tokens advance but cost stays 0).
func TestTrackerRecordUsage(t *testing.T) {
	cases := []struct {
		name          string
		usage         TokenUsage
		pricing       Pricing
		wantCost      float64
		wantInputTok  int64
		wantOutputTok int64
	}{
		{
			// Kept under 200K threshold so the simple branch (not tiered)
			// is exercised — the 200K-threshold case below covers tiered.
			name:    "simple input+output (under 200K)",
			usage:   TokenUsage{PromptTokens: 100_000, CompletionTokens: 500_000},
			pricing: Pricing{InputPricePerMillion: 3.0, OutputPricePerMillion: 15.0},
			// 100K * $3/M  = $0.30
			// 500K * $15/M = $7.50
			// total        = $7.80
			wantCost:      7.80,
			wantInputTok:  100_000,
			wantOutputTok: 500_000,
		},
		{
			name:    "cache read at 10% discount",
			usage:   TokenUsage{PromptTokens: 100_000, CacheReadTokens: 100_000},
			pricing: Pricing{InputPricePerMillion: 3.0},
			// input: 100K * $3/M = $0.30
			// cache: 100K * $3/M * 0.1 = $0.03
			// total: $0.33
			wantCost:     0.33,
			wantInputTok: 100_000,
		},
		{
			name:    "over 200K threshold tiered pricing",
			usage:   TokenUsage{PromptTokens: 300_000},
			pricing: Pricing{InputPricePerMillion: 3.0},
			// first 200K * $3/M       = $0.60
			// next  100K * $3/M * 1.5 = $0.45
			// total                   = $1.05
			wantCost:     1.05,
			wantInputTok: 300_000,
		},
		{
			name:    "zero pricing keeps cost zero",
			usage:   TokenUsage{PromptTokens: 1_000_000, CompletionTokens: 500_000},
			pricing: Pricing{}, // both prices zero
			// Tokens still advance; cost stays 0.
			wantCost:      0,
			wantInputTok:  1_000_000,
			wantOutputTok: 500_000,
		},
		{
			name:    "output only",
			usage:   TokenUsage{CompletionTokens: 2_000_000},
			pricing: Pricing{InputPricePerMillion: 3.0, OutputPricePerMillion: 15.0},
			// 2M * $15 = $30
			wantCost:      30.0,
			wantOutputTok: 2_000_000,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tracker := Tracker{}
			got, cost := tracker.RecordUsage(tc.usage, tc.pricing)

			if !floatEq(cost, tc.wantCost) {
				t.Errorf("incremental cost = %v, want %v", cost, tc.wantCost)
			}
			if !floatEq(got.TotalCostUSD, tc.wantCost) {
				t.Errorf("total cost = %v, want %v", got.TotalCostUSD, tc.wantCost)
			}
			if got.TotalInputTokens != tc.wantInputTok {
				t.Errorf("TotalInputTokens = %d, want %d", got.TotalInputTokens, tc.wantInputTok)
			}
			if got.TotalOutputTokens != tc.wantOutputTok {
				t.Errorf("TotalOutputTokens = %d, want %d", got.TotalOutputTokens, tc.wantOutputTok)
			}
			if got.CallCount != 1 {
				t.Errorf("CallCount = %d, want 1", got.CallCount)
			}
		})
	}
}

// TestTrackerAccumulates verifies that repeated RecordUsage calls
// compound correctly across calls (the common loop path).
func TestTrackerAccumulates(t *testing.T) {
	tracker := Tracker{}
	pricing := Pricing{InputPricePerMillion: 3.0, OutputPricePerMillion: 15.0}

	// Three calls, each 100K input + 50K output.
	// Per call: 100K * $3/M + 50K * $15/M = $0.30 + $0.75 = $1.05
	for i := 0; i < 3; i++ {
		tracker, _ = tracker.RecordUsage(
			TokenUsage{PromptTokens: 100_000, CompletionTokens: 50_000},
			pricing,
		)
	}

	if tracker.CallCount != 3 {
		t.Errorf("CallCount = %d, want 3", tracker.CallCount)
	}
	if tracker.TotalInputTokens != 300_000 {
		t.Errorf("TotalInputTokens = %d, want 300_000", tracker.TotalInputTokens)
	}
	if tracker.TotalOutputTokens != 150_000 {
		t.Errorf("TotalOutputTokens = %d, want 150_000", tracker.TotalOutputTokens)
	}
	wantTotal := 3 * 1.05
	if !floatEq(tracker.TotalCostUSD, wantTotal) {
		t.Errorf("TotalCostUSD = %v, want %v", tracker.TotalCostUSD, wantTotal)
	}
}

// TestTrackerImmutable guards the global "no mutation" rule: calling
// RecordUsage on a Tracker value must NOT change the original.
func TestTrackerImmutable(t *testing.T) {
	original := Tracker{}
	_, _ = original.RecordUsage(
		TokenUsage{PromptTokens: 1_000_000},
		Pricing{InputPricePerMillion: 3.0},
	)
	if original.CallCount != 0 {
		t.Errorf("original mutated: CallCount = %d, want 0", original.CallCount)
	}
	if original.TotalInputTokens != 0 {
		t.Errorf("original mutated: TotalInputTokens = %d, want 0", original.TotalInputTokens)
	}
	if original.TotalCostUSD != 0 {
		t.Errorf("original mutated: TotalCostUSD = %v, want 0", original.TotalCostUSD)
	}
}

// TestTrackerBudget covers IsOverBudget + RemainingBudget across the
// four meaningful states: no budget, under, exactly at, over.
func TestTrackerBudget(t *testing.T) {
	cases := []struct {
		name          string
		tracker       Tracker
		wantOver      bool
		wantRemaining float64
	}{
		{
			name:          "no budget set",
			tracker:       Tracker{TotalCostUSD: 100.0},
			wantOver:      false,
			wantRemaining: 0,
		},
		{
			name:          "under budget",
			tracker:       Tracker{TotalCostUSD: 2.50, BudgetUSD: 5.0},
			wantOver:      false,
			wantRemaining: 2.50,
		},
		{
			name:          "exactly at budget",
			tracker:       Tracker{TotalCostUSD: 5.0, BudgetUSD: 5.0},
			wantOver:      true,
			wantRemaining: 0,
		},
		{
			name:          "over budget clamps remaining to 0",
			tracker:       Tracker{TotalCostUSD: 7.50, BudgetUSD: 5.0},
			wantOver:      true,
			wantRemaining: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.tracker.IsOverBudget(); got != tc.wantOver {
				t.Errorf("IsOverBudget() = %v, want %v", got, tc.wantOver)
			}
			if got := tc.tracker.RemainingBudget(); !floatEq(got, tc.wantRemaining) {
				t.Errorf("RemainingBudget() = %v, want %v", got, tc.wantRemaining)
			}
		})
	}
}

// TestFormatCost covers the sub-penny / dollars-and-cents formatting
// branches: < $0.01 shows 4 decimals, ≥ $0.01 shows 2 decimals.
func TestFormatCost(t *testing.T) {
	cases := []struct {
		name string
		cost float64
		want string
	}{
		{"zero shows 4 decimals", 0.0, "$0.0000"},
		{"sub-penny shows 4 decimals", 0.0042, "$0.0042"},
		{"just below penny", 0.0099, "$0.0099"},
		{"exactly one cent shows 2 decimals", 0.01, "$0.01"},
		{"dollars and cents", 1.23, "$1.23"},
		{"large amount", 123.45, "$123.45"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tracker := Tracker{TotalCostUSD: tc.cost}
			if got := tracker.FormatCost(); got != tc.want {
				t.Errorf("FormatCost() = %q, want %q", got, tc.want)
			}
		})
	}
}
