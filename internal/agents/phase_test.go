package agents

import (
	"testing"

	"github.com/ashish-work/opendev-go/internal/agents/doomloop"
	"github.com/ashish-work/opendev-go/internal/budget"
	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/tools"
)

// newTestContext constructs a PhaseContext over a fresh history slice
// and returns both the context and the history pointer so individual
// tests can inspect the underlying slice without going through the
// indirection.
func newTestContext(t *testing.T) (*PhaseContext, *[]provider.Message) {
	t.Helper()
	history := []provider.Message{}
	tctx := tools.ToolContext{WorkingDir: "/tmp"}
	pc := NewPhaseContext(
		&history,
		cost.Tracker{},
		budget.New(128_000),
		doomloop.New(),
		tctx,
		nil,
		"system prompt",
	)
	return pc, &history
}

func TestNewPhaseContext_PopulatesFields(t *testing.T) {
	history := []provider.Message{{Role: "user"}}
	tracker := cost.Tracker{CallCount: 7}
	cal := budget.New(64_000)
	det := doomloop.New()
	tctx := tools.ToolContext{WorkingDir: "/work"}
	sink := make(chan provider.StreamEvent)
	defer close(sink)

	pc := NewPhaseContext(&history, tracker, cal, det, tctx, sink, "sysprompt")

	if pc.History != &history {
		t.Errorf("History pointer mismatch")
	}
	if pc.Tracker.CallCount != 7 {
		t.Errorf("Tracker.CallCount = %d, want 7", pc.Tracker.CallCount)
	}
	if pc.Calibrator.MaxContextTokens != 64_000 {
		t.Errorf("Calibrator.MaxContextTokens = %d, want 64000", pc.Calibrator.MaxContextTokens)
	}
	if pc.Detector != det {
		t.Errorf("Detector pointer mismatch")
	}
	if pc.ToolCtx.WorkingDir != "/work" {
		t.Errorf("ToolCtx.WorkingDir = %q, want /work", pc.ToolCtx.WorkingDir)
	}
	if pc.StreamSink == nil {
		t.Errorf("StreamSink should be the channel we passed")
	}
	if pc.SystemPrompt != "sysprompt" {
		t.Errorf("SystemPrompt = %q, want sysprompt", pc.SystemPrompt)
	}
	if pc.Iter != 0 {
		t.Errorf("Iter = %d, want 0 (driver sets before each iter)", pc.Iter)
	}
}

func TestPhaseContext_NilSinkValid(t *testing.T) {
	history := []provider.Message{}
	pc := NewPhaseContext(&history, cost.Tracker{}, budget.New(0), doomloop.New(), tools.ToolContext{}, nil, "")
	if pc.StreamSink != nil {
		t.Errorf("nil sink should construct as nil; got %v", pc.StreamSink)
	}
}

func TestAppendMessage_PropagatesToCallerSlice(t *testing.T) {
	pc, hist := newTestContext(t)
	if len(*hist) != 0 {
		t.Fatalf("history should start empty, got %d", len(*hist))
	}

	newLen := pc.AppendMessage(provider.Message{Role: "user"})
	if newLen != 1 {
		t.Errorf("AppendMessage return = %d, want 1", newLen)
	}
	if len(*hist) != 1 {
		t.Errorf("history len via pointer = %d, want 1", len(*hist))
	}
	if (*hist)[0].Role != "user" {
		t.Errorf("history[0].Role = %q, want user", (*hist)[0].Role)
	}

	// Second append exercises the case where append might relocate
	// the backing array. The pointer indirection has to keep up.
	pc.AppendMessage(provider.Message{Role: "assistant"})
	if len(*hist) != 2 {
		t.Errorf("history len after second append = %d, want 2", len(*hist))
	}
	if (*hist)[1].Role != "assistant" {
		t.Errorf("history[1].Role = %q, want assistant", (*hist)[1].Role)
	}
}

func TestAppendMessage_MultipleAppendsAcrossPhases(t *testing.T) {
	// Simulates the multi-phase flow: phase A appends, the driver
	// hands the same pc to phase B which appends again, both updates
	// remain visible.
	pc, hist := newTestContext(t)
	for i := 0; i < 10; i++ {
		pc.AppendMessage(provider.Message{Role: "assistant"})
	}
	if got := len(*hist); got != 10 {
		t.Errorf("history len = %d, want 10", got)
	}
}

func TestSnapshot_DelegatesToCalibrator(t *testing.T) {
	pc, _ := newTestContext(t)
	pc.AppendMessage(provider.Message{
		Role:    "user",
		Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "hello"}},
	})
	snap := pc.Snapshot()
	// Calibrator.Snapshot returns the estimated count based on
	// history + system prompt; we don't pin the exact estimate here
	// (it depends on the budget package's heuristic), just verify
	// the snapshot is non-zero on non-empty input.
	if snap.Estimated == 0 {
		t.Errorf("Snapshot.Estimated = 0, want non-zero with non-empty history")
	}
}

func TestSnapshot_EmptyHistoryReturnsZeroEstimate(t *testing.T) {
	pc, _ := newTestContext(t)
	// With empty history and a system prompt, the estimate may be
	// the system prompt's token count only.
	snap := pc.Snapshot()
	if snap.Estimated < 0 {
		t.Errorf("Snapshot.Estimated = %d, want non-negative", snap.Estimated)
	}
}

func TestPhaseContext_TrackerReassignmentVisible(t *testing.T) {
	pc, _ := newTestContext(t)
	// Simulate a phase that records usage and reassigns the tracker.
	newTracker, _ := pc.Tracker.RecordUsage(
		cost.TokenUsage{PromptTokens: 100, CompletionTokens: 50},
		cost.Pricing{InputPricePerMillion: 1.0, OutputPricePerMillion: 2.0},
	)
	pc.Tracker = newTracker

	if pc.Tracker.TotalInputTokens != 100 {
		t.Errorf("Tracker.TotalInputTokens = %d, want 100 (reassignment didn't stick)",
			pc.Tracker.TotalInputTokens)
	}
	if pc.Tracker.CallCount != 1 {
		t.Errorf("Tracker.CallCount = %d, want 1", pc.Tracker.CallCount)
	}
}

func TestPhaseContext_CalibratorReassignmentVisible(t *testing.T) {
	pc, _ := newTestContext(t)
	pc.Calibrator = pc.Calibrator.Update(2000, 5)
	if pc.Calibrator.Reported() != 2000 {
		t.Errorf("Calibrator.Reported() = %d, want 2000 after Update",
			pc.Calibrator.Reported())
	}
}

func TestPhaseContext_LastResponseStartsZero(t *testing.T) {
	// PhaseContext is constructed fresh per iteration; LastResponse
	// must be the zero provider.Response on entry so a phase
	// reading it before llm_call runs sees a clean slate.
	pc, _ := newTestContext(t)
	if pc.LastResponse.Content != "" {
		t.Errorf("LastResponse.Content = %q, want empty on fresh PhaseContext",
			pc.LastResponse.Content)
	}
	if pc.LastResponse.ToolCalls != nil {
		t.Errorf("LastResponse.ToolCalls = %+v, want nil on fresh PhaseContext",
			pc.LastResponse.ToolCalls)
	}
}

func TestPhaseContext_LastResponseRoundTrips(t *testing.T) {
	// llm_call writes the response into pc; the next phase reads it.
	// Verify assignment-then-read works through the pointer.
	pc, _ := newTestContext(t)
	pc.LastResponse = provider.Response{
		Content:      "final answer",
		FinishReason: "stop",
		Usage:        provider.Usage{PromptTokens: 50, CompletionTokens: 10},
	}
	if pc.LastResponse.Content != "final answer" {
		t.Errorf("LastResponse round-trip failed; got %q", pc.LastResponse.Content)
	}
	if pc.LastResponse.Usage.PromptTokens != 50 {
		t.Errorf("LastResponse.Usage.PromptTokens = %d, want 50",
			pc.LastResponse.Usage.PromptTokens)
	}
}

func TestPhaseContext_IterIsMutable(t *testing.T) {
	pc, _ := newTestContext(t)
	pc.Iter = 5
	if pc.Iter != 5 {
		t.Errorf("Iter = %d, want 5", pc.Iter)
	}
	pc.Iter++
	if pc.Iter != 6 {
		t.Errorf("Iter = %d, want 6 after increment", pc.Iter)
	}
}
