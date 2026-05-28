package agents

import (
	"context"
	"errors"
	"fmt"

	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/provider"
)

// ErrStreamIncomplete is returned by Stream when the provider's event
// channel closed without emitting a terminal StreamEventDone or
// StreamEventError. A well-behaved provider always closes with one of
// the two; getting here means the provider misbehaved.
var ErrStreamIncomplete = errors.New("agents: stream ended without Done or Error event")

// LlmCaller pairs a provider.Provider with model pricing so every call
// updates a cost.Tracker. Thin wrapper: one responsibility, "call the
// model and record what it cost."
//
// The Tracker flows through Call as a value (immutable) — callers must
// reassign with the returned Tracker. This mirrors how cost.Tracker
// itself works and prevents hidden mutation across goroutines.
//
// Intentionally simple in v1 — no retries, no provider fallback, no
// streaming. Each of those grows the surface in a focused way when it
// lands.
type LlmCaller struct {
	// Provider is the wire-level transport (e.g. *openai.Client).
	Provider provider.Provider

	// Pricing is used by the cost tracker to compute USD per call.
	// Zero Pricing keeps token totals correct but cost stays at 0.
	Pricing cost.Pricing
}

// NewLlmCaller constructs a caller. Pricing is optional — pass a zero
// cost.Pricing to disable cost computation (token totals still flow).
func NewLlmCaller(p provider.Provider, pricing cost.Pricing) *LlmCaller {
	return &LlmCaller{Provider: p, Pricing: pricing}
}

// Call sends one Request, records token usage on the tracker, and
// returns the provider response + the updated tracker. The original
// tracker is unchanged (immutable update pattern).
//
// Provider errors flow back unwrapped — the caller (ReactLoop.Run)
// decides how to wrap them into agent-layer errors.
func (c *LlmCaller) Call(
	ctx context.Context,
	req provider.Request,
	tracker cost.Tracker,
) (provider.Response, cost.Tracker, error) {
	resp, err := c.Provider.Call(ctx, req)
	if err != nil {
		return provider.Response{}, tracker, err
	}

	usage := cost.TokenUsage{
		PromptTokens:     int64(resp.Usage.PromptTokens),
		CompletionTokens: int64(resp.Usage.CompletionTokens),
		CacheReadTokens:  int64(resp.Usage.CachedTokens),
	}
	newTracker, _ := tracker.RecordUsage(usage, c.Pricing)
	return resp, newTracker, nil
}

// Stream sends one Request via the provider's streaming path, forwards
// every StreamEvent to sink as it arrives, assembles the final
// Response from the terminal StreamEventDone, and returns it along
// with the updated cost tracker. Same return shape as Call so the
// loop can swap between them with one branch.
//
// sink may be nil — events are still consumed (assembly still works);
// they just don't go anywhere. This lets callers exercise the
// streaming path without subscribing to the events.
//
// Forwarding uses select-with-ctx so a stalled consumer doesn't pin
// the producer goroutine indefinitely: when ctx is cancelled, we
// abandon the rest of the stream and return ctx.Err(). The provider's
// own goroutine will see the cancellation through its ctx wiring and
// close its event channel; on return we don't need to drain the rest.
//
// Mid-stream errors arrive on the event channel as StreamEventError;
// we surface those as the function's return error so they look the
// same to ReactLoop as a Call error. Setup errors from
// Provider.Stream itself return directly (channel was never created).
func (c *LlmCaller) Stream(
	ctx context.Context,
	req provider.Request,
	tracker cost.Tracker,
	sink chan<- provider.StreamEvent,
) (provider.Response, cost.Tracker, error) {
	events, err := c.Provider.Stream(ctx, req)
	if err != nil {
		return provider.Response{}, tracker, err
	}

	var (
		assembled *provider.Response
		streamErr error
	)
	for ev := range events {
		// Forward to the consumer first. select-with-ctx avoids a
		// deadlock when the consumer stops reading: ctx cancellation
		// unblocks the send and we exit cleanly.
		if sink != nil {
			select {
			case sink <- ev:
			case <-ctx.Done():
				return provider.Response{}, tracker, ctx.Err()
			}
		}
		switch ev.Kind {
		case provider.StreamEventDone:
			assembled = ev.Response
		case provider.StreamEventError:
			streamErr = ev.Err
		}
	}

	if streamErr != nil {
		return provider.Response{}, tracker, streamErr
	}
	if assembled == nil {
		return provider.Response{}, tracker, fmt.Errorf("%w", ErrStreamIncomplete)
	}

	usage := cost.TokenUsage{
		PromptTokens:     int64(assembled.Usage.PromptTokens),
		CompletionTokens: int64(assembled.Usage.CompletionTokens),
		CacheReadTokens:  int64(assembled.Usage.CachedTokens),
	}
	newTracker, _ := tracker.RecordUsage(usage, c.Pricing)
	return *assembled, newTracker, nil
}
