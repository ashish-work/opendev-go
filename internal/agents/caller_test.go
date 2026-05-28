package agents

import (
	"context"
	"errors"
	"testing"

	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/provider"
)

// scriptedStreamProvider is a minimal provider.Provider that emits a
// canned sequence of StreamEvents and (optionally) a non-streaming
// Call response. Used to exercise LlmCaller.Stream without going
// through a real adapter.
type scriptedStreamProvider struct {
	events    []provider.StreamEvent
	setupErr  error
	closeWith error // if non-zero, replace the last event with Error before close
}

func (p *scriptedStreamProvider) Name() string { return "scripted-stream" }

func (p *scriptedStreamProvider) Call(_ context.Context, _ provider.Request) (provider.Response, error) {
	// Stream tests don't exercise Call.
	return provider.Response{}, errors.New("Call not used in this test")
}

func (p *scriptedStreamProvider) Stream(ctx context.Context, _ provider.Request) (<-chan provider.StreamEvent, error) {
	if p.setupErr != nil {
		return nil, p.setupErr
	}
	ch := make(chan provider.StreamEvent, len(p.events)+1)
	go func() {
		defer close(ch)
		for _, ev := range p.events {
			select {
			case ch <- ev:
			case <-ctx.Done():
				return
			}
		}
		if p.closeWith != nil {
			select {
			case ch <- provider.NewError(p.closeWith):
			case <-ctx.Done():
			}
		}
	}()
	return ch, nil
}

func TestCallerStream_AssemblesResponseFromDone(t *testing.T) {
	final := &provider.Response{
		Content:      "hello world",
		FinishReason: "stop",
		Usage:        provider.Usage{PromptTokens: 10, CompletionTokens: 5},
	}
	p := &scriptedStreamProvider{
		events: []provider.StreamEvent{
			provider.NewTextDelta("hello "),
			provider.NewTextDelta("world"),
			provider.NewDone(final),
		},
	}
	c := NewLlmCaller(p, cost.Pricing{})

	resp, _, err := c.Stream(context.Background(), provider.Request{}, cost.Tracker{}, nil)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if resp.Content != "hello world" {
		t.Errorf("Content = %q, want %q", resp.Content, "hello world")
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q", resp.FinishReason)
	}
}

func TestCallerStream_ForwardsEventsToSink(t *testing.T) {
	p := &scriptedStreamProvider{
		events: []provider.StreamEvent{
			provider.NewTextDelta("a"),
			provider.NewTextDelta("b"),
			provider.NewTextDelta("c"),
			provider.NewDone(&provider.Response{Content: "abc"}),
		},
	}
	c := NewLlmCaller(p, cost.Pricing{})

	// Buffered enough to hold all events; reader collects after Stream returns.
	sink := make(chan provider.StreamEvent, 16)
	_, _, err := c.Stream(context.Background(), provider.Request{}, cost.Tracker{}, sink)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	close(sink)

	var got []provider.StreamEventKind
	for ev := range sink {
		got = append(got, ev.Kind)
	}
	want := []provider.StreamEventKind{
		provider.StreamEventTextDelta,
		provider.StreamEventTextDelta,
		provider.StreamEventTextDelta,
		provider.StreamEventDone,
	}
	if len(got) != len(want) {
		t.Fatalf("sink event count = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("sink[%d] = %s, want %s", i, got[i], want[i])
		}
	}
}

func TestCallerStream_NilSinkStillAssembles(t *testing.T) {
	p := &scriptedStreamProvider{
		events: []provider.StreamEvent{
			provider.NewTextDelta("x"),
			provider.NewDone(&provider.Response{Content: "x"}),
		},
	}
	c := NewLlmCaller(p, cost.Pricing{})

	resp, _, err := c.Stream(context.Background(), provider.Request{}, cost.Tracker{}, nil)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if resp.Content != "x" {
		t.Errorf("Content = %q, want %q", resp.Content, "x")
	}
}

func TestCallerStream_RecordsUsageOnTracker(t *testing.T) {
	final := &provider.Response{
		Content: "ok",
		Usage: provider.Usage{
			PromptTokens:     1000,
			CompletionTokens: 200,
			CachedTokens:     500,
		},
	}
	p := &scriptedStreamProvider{
		events: []provider.StreamEvent{provider.NewDone(final)},
	}
	c := NewLlmCaller(p, cost.Pricing{
		InputPricePerMillion:  3.00,
		OutputPricePerMillion: 15.00,
	})

	_, tracker, err := c.Stream(context.Background(), provider.Request{}, cost.Tracker{}, nil)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if tracker.TotalInputTokens != 1000 {
		t.Errorf("TotalInputTokens = %d, want 1000", tracker.TotalInputTokens)
	}
	if tracker.TotalOutputTokens != 200 {
		t.Errorf("TotalOutputTokens = %d, want 200", tracker.TotalOutputTokens)
	}
	if tracker.TotalCacheReadTokens != 500 {
		t.Errorf("TotalCacheReadTokens = %d, want 500", tracker.TotalCacheReadTokens)
	}
	if tracker.CallCount != 1 {
		t.Errorf("CallCount = %d, want 1", tracker.CallCount)
	}
	// Cost = inputCost + cacheCost + outputCost, where the cache adder
	// applies on top of the prompt cost at the discount rate (matches
	// the v1 cost-tracker behavior; we don't second-guess that here).
	if got := tracker.TotalCostUSD; got <= 0 {
		t.Errorf("TotalCostUSD = %v, want positive", got)
	}
}

func TestCallerStream_PropagatesStreamError(t *testing.T) {
	p := &scriptedStreamProvider{
		events: []provider.StreamEvent{
			provider.NewTextDelta("hi"),
		},
		closeWith: errors.New("provider exploded"),
	}
	c := NewLlmCaller(p, cost.Pricing{})

	_, _, err := c.Stream(context.Background(), provider.Request{}, cost.Tracker{}, nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if err.Error() != "provider exploded" {
		t.Errorf("err = %v, want 'provider exploded'", err)
	}
}

func TestCallerStream_PropagatesSetupError(t *testing.T) {
	p := &scriptedStreamProvider{setupErr: errors.New("auth missing")}
	c := NewLlmCaller(p, cost.Pricing{})
	_, _, err := c.Stream(context.Background(), provider.Request{}, cost.Tracker{}, nil)
	if err == nil || err.Error() != "auth missing" {
		t.Errorf("err = %v, want setup error", err)
	}
}

func TestCallerStream_NoTerminalEventReturnsIncompleteError(t *testing.T) {
	p := &scriptedStreamProvider{
		events: []provider.StreamEvent{
			provider.NewTextDelta("oops"),
			// no Done, no Error — channel closes after this
		},
	}
	c := NewLlmCaller(p, cost.Pricing{})
	_, _, err := c.Stream(context.Background(), provider.Request{}, cost.Tracker{}, nil)
	if !errors.Is(err, ErrStreamIncomplete) {
		t.Errorf("err = %v, want ErrStreamIncomplete", err)
	}
}

func TestCallerStream_CtxCancelUnblocksStalledSink(t *testing.T) {
	// Many events, but the sink is unbuffered and we never read from it.
	// Without the ctx-aware send, the goroutine would deadlock here.
	events := make([]provider.StreamEvent, 100)
	for i := range events {
		events[i] = provider.NewTextDelta("x")
	}
	events = append(events, provider.NewDone(&provider.Response{}))
	p := &scriptedStreamProvider{events: events}
	c := NewLlmCaller(p, cost.Pricing{})

	ctx, cancel := context.WithCancel(context.Background())
	sink := make(chan provider.StreamEvent) // unbuffered, no reader

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _, _ = c.Stream(ctx, provider.Request{}, cost.Tracker{}, sink)
	}()

	cancel() // immediately abandon

	select {
	case <-done:
		// Good: caller returned promptly after cancellation.
	case <-context.Background().Done():
		t.Fatal("Stream did not return after ctx cancellation")
	}
}
