package optimization

import (
	"testing"

	"github.com/ashish-work/opendev-go/internal/budget"
)

// snapAt builds a budget.Snapshot with the given UsagePct. The
// other fields don't matter for Check — we only read UsagePct —
// but populating Reported keeps test data realistic.
func snapAt(pct float64) budget.Snapshot {
	return budget.Snapshot{
		Reported:  int(pct * 128_000),
		Estimated: int(pct * 128_000),
		UsagePct:  pct,
	}
}

func TestCheck_HealthyBelowWarning(t *testing.T) {
	// Healthy band: 0.0 up to but not including the warning
	// threshold. We probe a few points to catch a stale
	// implementation that hard-coded a single check value.
	for _, pct := range []float64{0.0, 0.10, 0.50, 0.69, 0.6999} {
		if got := Check(snapAt(pct)); got != LevelHealthy {
			t.Errorf("Check(usage=%.4f) = %v, want LevelHealthy", pct, got)
		}
	}
}

func TestCheck_WarningAtAndAboveThreshold(t *testing.T) {
	// 0.70 must classify as Warning (>= semantics on the
	// boundary). Stays in the band until MaskObservations'
	// threshold.
	for _, pct := range []float64{0.70, 0.71, 0.75, 0.79, 0.7999} {
		if got := Check(snapAt(pct)); got != LevelWarning {
			t.Errorf("Check(usage=%.4f) = %v, want LevelWarning", pct, got)
		}
	}
}

func TestCheck_MaskObservationsAtAndAboveThreshold(t *testing.T) {
	for _, pct := range []float64{0.80, 0.82, 0.84, 0.8499} {
		if got := Check(snapAt(pct)); got != LevelMaskObservations {
			t.Errorf("Check(usage=%.4f) = %v, want LevelMaskObservations", pct, got)
		}
	}
}

func TestCheck_PruneAtAndAboveThreshold(t *testing.T) {
	for _, pct := range []float64{0.85, 0.87, 0.89, 0.8999} {
		if got := Check(snapAt(pct)); got != LevelPrune {
			t.Errorf("Check(usage=%.4f) = %v, want LevelPrune", pct, got)
		}
	}
}

func TestCheck_AggressiveMaskAtAndAboveThreshold(t *testing.T) {
	for _, pct := range []float64{0.90, 0.95, 0.98, 0.9899} {
		if got := Check(snapAt(pct)); got != LevelAggressiveMask {
			t.Errorf("Check(usage=%.4f) = %v, want LevelAggressiveMask", pct, got)
		}
	}
}

func TestCheck_FullCompactAtAndAboveThreshold(t *testing.T) {
	// Including over-100% — that happens when prompt_tokens
	// briefly exceeds the configured max (e.g. provider returned
	// a value that the calibrator hasn't reconciled). Don't
	// crash; classify as FullCompact and let the compactor
	// recover.
	for _, pct := range []float64{0.99, 1.00, 1.20, 1.50} {
		if got := Check(snapAt(pct)); got != LevelFullCompact {
			t.Errorf("Check(usage=%.4f) = %v, want LevelFullCompact", pct, got)
		}
	}
}

func TestCheck_ZeroSnapshotIsHealthy(t *testing.T) {
	// The first iteration has no API call yet; the calibrator's
	// zero value yields UsagePct == 0.0. Must classify as
	// Healthy so we don't spuriously trigger optimization before
	// there's any data.
	if got := Check(budget.Snapshot{}); got != LevelHealthy {
		t.Errorf("Check(zero Snapshot) = %v, want LevelHealthy", got)
	}
}

func TestCheck_BoundariesAreInclusiveLow(t *testing.T) {
	// Boundary discipline — each threshold belongs to the band
	// it OPENS, not the band it CLOSES. This is the >= semantics
	// in Check's switch arms; pin it explicitly because the
	// alternative (>) is a one-character typo and would shift
	// every threshold up by one ULP.
	cases := []struct {
		pct  float64
		want Level
	}{
		{ThresholdWarning, LevelWarning},                   // 0.70
		{ThresholdMaskObservations, LevelMaskObservations}, // 0.80
		{ThresholdPrune, LevelPrune},                       // 0.85
		{ThresholdAggressiveMask, LevelAggressiveMask},     // 0.90
		{ThresholdFullCompact, LevelFullCompact},           // 0.99
	}
	for _, tc := range cases {
		if got := Check(snapAt(tc.pct)); got != tc.want {
			t.Errorf("Check(usage=%.2f) = %v, want %v (boundary inclusion)",
				tc.pct, got, tc.want)
		}
	}
}

func TestLevelString_AllNamedValues(t *testing.T) {
	// Table-driven; if a Level is added without a String case,
	// the test below catches it via the "unknown" fallback test
	// being unreachable.
	cases := []struct {
		level Level
		want  string
	}{
		{LevelHealthy, "healthy"},
		{LevelWarning, "warning"},
		{LevelMaskObservations, "mask_observations"},
		{LevelPrune, "prune"},
		{LevelAggressiveMask, "aggressive_mask"},
		{LevelFullCompact, "full_compact"},
	}
	for _, tc := range cases {
		if got := tc.level.String(); got != tc.want {
			t.Errorf("Level(%d).String() = %q, want %q", tc.level, got, tc.want)
		}
	}
}

func TestLevelString_UnknownFallback(t *testing.T) {
	// Future Level values added without updating String must
	// degrade gracefully — no panic, no empty string, no
	// "Level(7)" Go default. The fallback exists so a stale
	// binary that receives a future Level over the wire (e.g.
	// from a persisted session) logs cleanly rather than
	// crashing.
	if got := Level(99).String(); got != "unknown" {
		t.Errorf("Level(99).String() = %q, want %q", got, "unknown")
	}
}
