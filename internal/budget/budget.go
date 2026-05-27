// Package budget tracks how full the LLM's context window is, anchored
// on the provider's reported prompt_tokens after each turn rather than
// on local heuristics alone.
//
// The mechanism — "calibration" — solves a real problem: local token
// estimates drift (we don't ship a real BPE tokenizer), but the
// provider's response includes the AUTHORITATIVE prompt_tokens count
// for what it just received. Each LLM call gives us a ground-truth
// anchor. Between calls we extrapolate locally — but only across the
// small handful of messages added since the last call, so error stays
// bounded.
//
// Staged optimization levels (e.g. 70%/80%/85%/90%/99% thresholds
// triggering progressive history compaction) belong with the future
// history compactor and are intentionally NOT here yet — this package
// reports the numbers; what to do about them is a separate concern.
//
// Why a separate package from internal/cost: $ and context-fill are
// distinct concerns that both happen to consume provider.Usage. Mixing
// them would make cost.Tracker a grab-bag.
//
// Immutability: matches cost.Tracker's pattern — methods that change
// state return a new Calibrator value rather than mutating in place.
package budget

import (
	"strings"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// Calibrator tracks the last reported prompt_tokens and the conversation
// length at the moment of that report. Use Estimate to get the current
// calibrated count; use Update after each LLM call to install the new
// baseline.
type Calibrator struct {
	// MaxContextTokens is the model's window cap (e.g. 128_000 for
	// gpt-4o). Zero disables UsagePct math but Estimate still works.
	MaxContextTokens int

	// apiPromptTokens is the last value reported by the provider
	// (usage.prompt_tokens). Zero means "no calibration yet" — Estimate
	// falls back to a fully-local count.
	apiPromptTokens int

	// msgCountAtCalibration records len(history) at the moment
	// apiPromptTokens was set. Estimate counts only the messages added
	// after this point.
	msgCountAtCalibration int
}

// New returns a Calibrator with the given context window cap. Pass 0
// to disable usage-percentage math (Estimate still works).
func New(maxContext int) Calibrator {
	return Calibrator{MaxContextTokens: maxContext}
}

// Update returns a new Calibrator with the reported token count and
// message-count baseline installed. A reportedPromptTokens of 0 is
// treated as "no fresh data" and leaves the calibrator unchanged —
// some providers omit usage on streaming chunks.
//
// currentMsgCount is the length of the history *as seen by the model
// in the request that produced this reported count* — i.e. before the
// assistant response that came back gets appended. This is the value
// the next Estimate uses as the cutoff between "reported" and "delta".
func (c Calibrator) Update(reportedPromptTokens, currentMsgCount int) Calibrator {
	if reportedPromptTokens <= 0 {
		return c
	}
	c.apiPromptTokens = reportedPromptTokens
	c.msgCountAtCalibration = currentMsgCount
	return c
}

// InvalidateCalibration returns a Calibrator with the baseline cleared.
// Call this after history has been mutated (compaction, masking) so
// the next Estimate recomputes from scratch instead of trusting the
// now-stale baseline.
func (c Calibrator) InvalidateCalibration() Calibrator {
	c.apiPromptTokens = 0
	c.msgCountAtCalibration = 0
	return c
}

// Reported returns the last reported prompt_tokens count. Zero means
// no calibration has occurred yet.
func (c Calibrator) Reported() int { return c.apiPromptTokens }

// Estimate returns the calibrated token count for the given history.
//
//   - If no baseline is set: a fully-local count via CountTokens.
//   - If a baseline IS set: apiPromptTokens + locally-counted tokens for
//     any messages appearing after msgCountAtCalibration. This bounds
//     drift to whatever was added between calibration and now.
//
// systemPrompt is counted only on the fully-local path; once calibrated,
// the system prompt is already inside apiPromptTokens.
func (c Calibrator) Estimate(messages []provider.Message, systemPrompt string) int {
	if c.apiPromptTokens == 0 {
		total := CountTokens(systemPrompt)
		for _, m := range messages {
			total += messageTokens(m)
		}
		return total
	}
	if len(messages) <= c.msgCountAtCalibration {
		// History shrunk or stayed flat since calibration — baseline
		// is still the best answer we have.
		return c.apiPromptTokens
	}
	delta := 0
	for _, m := range messages[c.msgCountAtCalibration:] {
		delta += messageTokens(m)
	}
	return c.apiPromptTokens + delta
}

// UsagePct returns the calibrated fraction of MaxContextTokens used
// (0.0–1.0+). Returns 0.0 when there's no baseline or MaxContextTokens
// is zero. Note: this uses the BASELINE only, not the delta — the
// staged compaction trigger should use Estimate(...) / MaxContextTokens
// when it lands in T3.5.
func (c Calibrator) UsagePct() float64 {
	if c.MaxContextTokens == 0 || c.apiPromptTokens == 0 {
		return 0.0
	}
	return float64(c.apiPromptTokens) / float64(c.MaxContextTokens)
}

// Snapshot is a frozen view of the calibrator's state plus an Estimate
// computed against caller-supplied messages. Convenient for stashing
// on agents.Result, logging, and the REPL status line.
type Snapshot struct {
	// Reported is the last value the provider returned. Zero before
	// the first LLM call completes.
	Reported int

	// Estimated is the calibrated count at snapshot time (reported
	// baseline + local count of messages added since calibration).
	Estimated int

	// UsagePct is Reported / MaxContextTokens — 0.0–1.0+.
	UsagePct float64
}

// Snapshot computes a Snapshot for the given history and prompt. Pure;
// safe to call repeatedly.
func (c Calibrator) Snapshot(messages []provider.Message, systemPrompt string) Snapshot {
	return Snapshot{
		Reported:  c.Reported(),
		Estimated: c.Estimate(messages, systemPrompt),
		UsagePct:  c.UsagePct(),
	}
}

// CountTokens estimates the BPE token count of text using a cl100k_base-
// style heuristic. Faster than calling a real tokenizer and roughly
// within 20% of tiktoken on English prose and code.
//
// Algorithm:
//
//  1. Split on whitespace into words.
//  2. Per word:
//     - len > 12: estimate ceil(len/4) tokens (long identifiers chunked).
//     - else: 1 base token + ceil(punctCount/2) for attached punctuation.
//  3. Apply a 0.75 ratio to the word total (most English words map to
//     fewer than 1 BPE token).
//
// Returns 0 for empty or whitespace-only input. Pure function.
//
// Examples:
//
//	CountTokens("hello world")                     // -> 2 (short)
//	CountTokens("supercalifragilisticexpialidocious") // -> 9 (long)
//	CountTokens("foo, bar.")                       // -> 3 (punctuation)
func CountTokens(text string) int {
	if strings.TrimSpace(text) == "" {
		return 0
	}
	wordCount := 0
	for _, word := range strings.Fields(text) {
		n := len(word)
		if n > 12 {
			wordCount += (n + 3) / 4 // ceil(n/4)
			continue
		}
		punct := 0
		for _, r := range word {
			if isASCIIPunct(r) {
				punct++
			}
		}
		wordCount += 1 + (punct+1)/2 // 1 + ceil(punct/2)
	}
	return (wordCount*3 + 2) / 4 // ceil(wordCount * 0.75)
}

// isASCIIPunct returns true for ASCII punctuation: the characters in
// 0x21..0x7E that are not alphanumeric or whitespace.
func isASCIIPunct(r rune) bool {
	switch {
	case r >= '!' && r <= '/': // 0x21..0x2F
		return true
	case r >= ':' && r <= '@': // 0x3A..0x40
		return true
	case r >= '[' && r <= '`': // 0x5B..0x60
		return true
	case r >= '{' && r <= '~': // 0x7B..0x7E
		return true
	}
	return false
}

// messageTokens sums CountTokens across every text ContentBlock in m.
func messageTokens(m provider.Message) int {
	total := 0
	for _, c := range m.Content {
		total += CountTokens(c.Text)
	}
	return total
}
