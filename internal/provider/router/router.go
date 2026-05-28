// Package router picks a provider implementation based on the model
// name and exposes the pricing table both binaries share.
//
// Two consumers (cmd/opendev and cmd/opendev-tui) used to hold
// near-identical "if model starts with X, construct openai.NewClient"
// blocks plus duplicated pricing tables. This package owns both
// decisions so the wiring lives in one place and adding a third
// provider is a single-file change.
package router

import (
	"errors"
	"fmt"
	"strings"

	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/provider/anthropic"
	"github.com/ashish-work/opendev-go/internal/provider/openai"
)

// ErrEmptyModel is returned by New when the caller passes an empty
// model name. We don't try to substitute a default — the caller's
// flag default already supplies one, so empty here is a real bug.
var ErrEmptyModel = errors.New("router: model name is empty")

// ErrEmptyAPIKey is returned by New when the caller passes an empty
// API key. The caller is responsible for sourcing the key from the
// right environment variable (see EnvVarFor).
var ErrEmptyAPIKey = errors.New("router: api key is empty")

// New constructs a provider.Provider for the given model. The
// returned provider is fully configured: BaseURL applied (or the
// provider's default), API key set, ready to call.
//
// Routing rule: model names prefixed with "claude-" go to Anthropic.
// Everything else goes to OpenAI. The catch-all matters: most
// OpenAI-compatible servers (local llama, mistral proxies, vLLM
// fronts) use arbitrary model names like "llama-3-70b" or
// "deepseek-coder-v2", and the -base-url flag is the standard escape
// hatch. A strict whitelist by prefix would break those out of the
// box.
//
// baseURL=="" uses the selected provider's default; non-empty
// overrides it.
func New(model, baseURL, apiKey string) (provider.Provider, error) {
	if model == "" {
		return nil, ErrEmptyModel
	}
	if apiKey == "" {
		return nil, ErrEmptyAPIKey
	}

	switch providerFor(model) {
	case providerAnthropic:
		client := anthropic.NewClient(apiKey)
		if baseURL != "" {
			client.Adapter.BaseURL = baseURL
		}
		return client, nil
	default:
		client := openai.NewClient(apiKey)
		if baseURL != "" {
			client.Adapter.BaseURL = baseURL
		}
		return client, nil
	}
}

// EnvVarFor returns the environment-variable name the caller should
// read to source the API key for the given model. main.go uses this
// to format an accurate error message when the variable is unset
// ("ANTHROPIC_API_KEY must be set" beats a generic "no key").
func EnvVarFor(model string) string {
	switch providerFor(model) {
	case providerAnthropic:
		return "ANTHROPIC_API_KEY"
	default:
		return "OPENAI_API_KEY"
	}
}

// PricingFor returns cost.Pricing for known models. Unknown models
// fall back to zero pricing — token counts still flow through the
// tracker; cost stays $0 so a model not in the table renders as
// "free" rather than failing or guessing.
//
// Model names are prefix-matched so dated aliases like
// "gpt-4o-2024-08-06" or "claude-sonnet-4-5-20250101" pick up the
// right rate without explicit entries per snapshot.
//
// Ordering matters when prefixes overlap: longer/more-specific
// prefixes come first. E.g., "gpt-4o-mini" before "gpt-4o" so a
// gpt-4o-mini model doesn't accidentally pick up gpt-4o pricing.
func PricingFor(model string) cost.Pricing {
	for _, entry := range pricingTable {
		if strings.HasPrefix(model, entry.prefix) {
			return entry.pricing
		}
	}
	return cost.Pricing{}
}

// providerSelector is the internal enum returned by providerFor. Not
// exported because callers should treat the choice as opaque — they
// see only the provider.Provider interface.
type providerSelector int

const (
	providerOpenAI providerSelector = iota
	providerAnthropic
)

// providerFor encodes the routing rule. Pulled out so both New and
// EnvVarFor agree on the answer without duplicating the prefix check.
func providerFor(model string) providerSelector {
	if strings.HasPrefix(model, "claude-") {
		return providerAnthropic
	}
	return providerOpenAI
}

// pricingEntry is one row in the price table.
type pricingEntry struct {
	prefix  string
	pricing cost.Pricing
}

// pricingTable is the master rate sheet. Rates are USD per million
// tokens, sourced from each provider's public list price at the time
// of this commit. When prices change, update this single table —
// both binaries pick up the new values without modification.
//
// Order is significant: longer prefixes precede their shorter
// counterparts so "gpt-4o-mini" matches before "gpt-4o".
var pricingTable = []pricingEntry{
	// Anthropic — Claude 4 family (current generation).
	{"claude-opus-4-1", cost.Pricing{InputPricePerMillion: 15.00, OutputPricePerMillion: 75.00}},
	{"claude-opus-4", cost.Pricing{InputPricePerMillion: 15.00, OutputPricePerMillion: 75.00}},
	{"claude-sonnet-4-5", cost.Pricing{InputPricePerMillion: 3.00, OutputPricePerMillion: 15.00}},
	{"claude-sonnet-4", cost.Pricing{InputPricePerMillion: 3.00, OutputPricePerMillion: 15.00}},
	{"claude-haiku-4-5", cost.Pricing{InputPricePerMillion: 1.00, OutputPricePerMillion: 5.00}},

	// Anthropic — Claude 3.5 family (previous generation, still active).
	{"claude-3-5-sonnet", cost.Pricing{InputPricePerMillion: 3.00, OutputPricePerMillion: 15.00}},
	{"claude-3-5-haiku", cost.Pricing{InputPricePerMillion: 0.80, OutputPricePerMillion: 4.00}},

	// OpenAI. Order: -mini before plain -4o so the mini rate applies.
	{"gpt-4o-mini", cost.Pricing{InputPricePerMillion: 0.15, OutputPricePerMillion: 0.60}},
	{"gpt-4o", cost.Pricing{InputPricePerMillion: 2.50, OutputPricePerMillion: 10.00}},
	{"gpt-4-turbo", cost.Pricing{InputPricePerMillion: 10.00, OutputPricePerMillion: 30.00}},
	{"gpt-3.5-turbo", cost.Pricing{InputPricePerMillion: 0.50, OutputPricePerMillion: 1.50}},
}

// FormatMissingKey returns a user-facing error string for the case
// where the API key environment variable is unset. Co-located with
// EnvVarFor so the message stays consistent across both binaries.
func FormatMissingKey(model string) string {
	return fmt.Sprintf("error: %s must be set (model %q selected)",
		EnvVarFor(model), model)
}
