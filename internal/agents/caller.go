package agents

import (
	"context"

	"github.com/ashishgupta/opendev-go/internal/cost"
	"github.com/ashishgupta/opendev-go/internal/provider"
)

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
