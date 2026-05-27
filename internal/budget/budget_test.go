package budget

import (
	"strings"
	"testing"

	"github.com/ashish-work/opendev-go/internal/provider"
)

func TestCountTokens(t *testing.T) {
	tests := []struct {
		name  string
		input string
		// Expected is an exact value from running the same algorithm —
		// we hand-calculate, not call a real tokenizer. The point is
		// regression: if someone changes the algorithm, these numbers
		// flag it.
		want int
	}{
		{"empty", "", 0},
		{"whitespace only", "   \n\t  ", 0},
		{"single short word", "hello", 1},                  // (1*3+2)/4 = 1
		{"two short words", "hello world", 2},              // (2*3+2)/4 = 2
		{"short with punctuation", "hi,", 2},               // 1 + ceil(1/2)=1 → 2 wc → (6+2)/4 = 2
		{"long identifier", "supercalifragilistic", 4},     // len=20 → ceil(20/4)=5 wc → (15+2)/4=4
		// "fn"=1, "foo()"=2, "{"=2, "return"=1, "42;"=2, "}"=2 → 10
		// (10*3+2)/4 = 8.
		{"mixed code with punctuation", "fn foo() { return 42; }", 8},
		{"newlines treated as whitespace", "a\nb\nc", 2},   // 3 wc → (9+2)/4 = 2
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CountTokens(tt.input); got != tt.want {
				t.Errorf("CountTokens(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}

func TestCountTokens_LongString(t *testing.T) {
	// Sanity check on a paragraph of realistic prose: should fall in
	// the expected ballpark (~75% of word count, more or less).
	text := strings.Repeat("the quick brown fox jumps over the lazy dog ", 100)
	got := CountTokens(text)
	// 900 words × 0.75 = 675 (give or take rounding)
	if got < 600 || got > 750 {
		t.Errorf("CountTokens(900-word paragraph) = %d, want ~675", got)
	}
}

func TestCalibrator_Update_StoresLastValue(t *testing.T) {
	c := New(128_000)
	if c.Reported() != 0 {
		t.Errorf("fresh Calibrator.Reported() = %d, want 0", c.Reported())
	}

	c = c.Update(1000, 3)
	if c.Reported() != 1000 {
		t.Errorf("after Update(1000,3): Reported = %d, want 1000", c.Reported())
	}

	c = c.Update(2500, 7)
	if c.Reported() != 2500 {
		t.Errorf("after Update(2500,7): Reported = %d, want 2500", c.Reported())
	}
}

func TestCalibrator_Update_IgnoresZero(t *testing.T) {
	c := New(128_000).Update(1000, 3)
	c2 := c.Update(0, 5)
	if c2.Reported() != 1000 {
		t.Errorf("after Update(0,5): Reported = %d, want 1000 unchanged", c2.Reported())
	}
}

func TestCalibrator_Estimate_NoBaseline(t *testing.T) {
	c := New(128_000)
	messages := []provider.Message{
		msg("user", "what is the capital of France"),
	}
	got := c.Estimate(messages, "system prompt here")
	if got == 0 {
		t.Error("Estimate with messages but no baseline returned 0; want > 0")
	}
}

func TestCalibrator_Estimate_WithBaseline(t *testing.T) {
	c := New(128_000).Update(500, 3) // baseline: 500 tokens covers 3 messages

	// Same 3 messages → estimate matches baseline exactly.
	threeMsgs := []provider.Message{msg("system", "x"), msg("user", "y"), msg("assistant", "z")}
	if got := c.Estimate(threeMsgs, ""); got != 500 {
		t.Errorf("Estimate(3 msgs == calibration point) = %d, want 500", got)
	}

	// Add a 4th message — estimate = 500 + local count of msg #4.
	fourMsgs := append(threeMsgs, msg("user", "hello world today"))
	got := c.Estimate(fourMsgs, "")
	if got <= 500 {
		t.Errorf("Estimate(4 msgs, baseline=500@3) = %d, want > 500", got)
	}
	// Sanity: shouldn't be wildly larger; the added message is short.
	if got > 510 {
		t.Errorf("Estimate(4 msgs) = %d, want closer to 500 (small delta)", got)
	}
}

func TestCalibrator_Estimate_HistoryShrunk(t *testing.T) {
	// If a future caller shrinks history (e.g. compaction), Estimate
	// should fall back to the baseline rather than producing nonsense.
	c := New(128_000).Update(1000, 5)
	twoMsgs := []provider.Message{msg("user", "a"), msg("user", "b")}
	if got := c.Estimate(twoMsgs, ""); got != 1000 {
		t.Errorf("Estimate(2 msgs, baseline=1000@5) = %d, want 1000", got)
	}
}

func TestCalibrator_InvalidateCalibration(t *testing.T) {
	c := New(128_000).Update(1000, 3)
	c = c.InvalidateCalibration()
	if c.Reported() != 0 {
		t.Errorf("Reported after invalidate = %d, want 0", c.Reported())
	}
	// And Estimate now does a fully-local count.
	messages := []provider.Message{msg("user", "hello world")}
	got := c.Estimate(messages, "")
	if got == 0 {
		t.Error("Estimate after invalidate should return local count, got 0")
	}
}

func TestCalibrator_UsagePct(t *testing.T) {
	// No baseline → 0.
	c := New(1000)
	if got := c.UsagePct(); got != 0.0 {
		t.Errorf("UsagePct no baseline = %v, want 0.0", got)
	}

	// With baseline.
	c = c.Update(250, 1)
	if got := c.UsagePct(); got != 0.25 {
		t.Errorf("UsagePct(250/1000) = %v, want 0.25", got)
	}

	// Zero max → 0 (avoid divide-by-zero).
	c2 := New(0).Update(100, 1)
	if got := c2.UsagePct(); got != 0.0 {
		t.Errorf("UsagePct with MaxContextTokens=0 = %v, want 0.0", got)
	}
}

func TestCalibrator_Snapshot(t *testing.T) {
	c := New(10_000).Update(500, 2)
	messages := []provider.Message{
		msg("system", "be nice"),
		msg("user", "hi"),
		msg("assistant", "hello world hello world hello world"),
	}
	snap := c.Snapshot(messages, "")
	if snap.Reported != 500 {
		t.Errorf("Snapshot.Reported = %d, want 500", snap.Reported)
	}
	if snap.Estimated <= 500 {
		t.Errorf("Snapshot.Estimated = %d, want > 500 (delta from msg #3)", snap.Estimated)
	}
	if snap.UsagePct != 0.05 {
		t.Errorf("Snapshot.UsagePct = %v, want 0.05", snap.UsagePct)
	}
}

func TestCalibrator_Immutability(t *testing.T) {
	// Update returns a new value; original must be untouched.
	original := New(1000)
	updated := original.Update(500, 1)
	if original.Reported() != 0 {
		t.Errorf("original mutated by Update: Reported = %d, want 0", original.Reported())
	}
	if updated.Reported() != 500 {
		t.Errorf("updated.Reported() = %d, want 500", updated.Reported())
	}
}

// msg is a tiny constructor — keeps test cases readable.
func msg(role, text string) provider.Message {
	return provider.Message{
		Role: role,
		Content: []provider.ContentBlock{
			{Kind: provider.ContentText, Text: text},
		},
	}
}
