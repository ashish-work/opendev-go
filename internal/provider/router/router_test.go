package router

import (
	"errors"
	"strings"
	"testing"

	"github.com/ashish-work/opendev-go/internal/provider/anthropic"
	"github.com/ashish-work/opendev-go/internal/provider/openai"
)

func TestNew_ClaudeModelReturnsAnthropic(t *testing.T) {
	p, err := New("claude-sonnet-4-5", "", "test-key")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, ok := p.(*anthropic.Client); !ok {
		t.Errorf("Provider concrete type = %T, want *anthropic.Client", p)
	}
	if p.Name() != "anthropic" {
		t.Errorf("Name = %q, want anthropic", p.Name())
	}
}

func TestNew_GPTModelReturnsOpenAI(t *testing.T) {
	p, err := New("gpt-4o", "", "test-key")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, ok := p.(*openai.Client); !ok {
		t.Errorf("Provider concrete type = %T, want *openai.Client", p)
	}
	if p.Name() != "openai" {
		t.Errorf("Name = %q, want openai", p.Name())
	}
}

func TestNew_OFamilyModelRoutesToOpenAI(t *testing.T) {
	// o-family reasoning models (o1, o3, o4-mini) are OpenAI.
	for _, model := range []string{"o1-preview", "o3-mini", "o4-mini"} {
		t.Run(model, func(t *testing.T) {
			p, err := New(model, "", "test-key")
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, ok := p.(*openai.Client); !ok {
				t.Errorf("%s → %T, want *openai.Client", model, p)
			}
		})
	}
}

func TestNew_UnknownModelDefaultsToOpenAI(t *testing.T) {
	// The catch-all matters: OpenAI-compatible servers use arbitrary
	// model names and the user reaches them via -base-url.
	for _, model := range []string{"llama-3-70b", "mistral-large", "deepseek-coder-v2"} {
		t.Run(model, func(t *testing.T) {
			p, err := New(model, "", "test-key")
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, ok := p.(*openai.Client); !ok {
				t.Errorf("%s → %T, want *openai.Client (catch-all)", model, p)
			}
		})
	}
}

func TestNew_EmptyModelErrors(t *testing.T) {
	_, err := New("", "", "test-key")
	if !errors.Is(err, ErrEmptyModel) {
		t.Errorf("err = %v, want ErrEmptyModel", err)
	}
}

func TestNew_EmptyAPIKeyErrors(t *testing.T) {
	_, err := New("gpt-4o", "", "")
	if !errors.Is(err, ErrEmptyAPIKey) {
		t.Errorf("err = %v, want ErrEmptyAPIKey", err)
	}
}

func TestNew_BaseURLOverride_OpenAI(t *testing.T) {
	const url = "https://proxy.example/v1"
	p, err := New("gpt-4o", url, "k")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	c := p.(*openai.Client)
	if c.Adapter.BaseURL != url {
		t.Errorf("BaseURL = %q, want %q", c.Adapter.BaseURL, url)
	}
}

func TestNew_BaseURLOverride_Anthropic(t *testing.T) {
	const url = "https://proxy.example/v1"
	p, err := New("claude-sonnet-4-5", url, "k")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	c := p.(*anthropic.Client)
	if c.Adapter.BaseURL != url {
		t.Errorf("BaseURL = %q, want %q", c.Adapter.BaseURL, url)
	}
}

func TestNew_BaseURLEmptyKeepsProviderDefault(t *testing.T) {
	pOpenAI, _ := New("gpt-4o", "", "k")
	if got := pOpenAI.(*openai.Client).Adapter.BaseURL; got != openai.DefaultBaseURL {
		t.Errorf("openai BaseURL = %q, want default %q", got, openai.DefaultBaseURL)
	}
	pAnth, _ := New("claude-sonnet-4-5", "", "k")
	if got := pAnth.(*anthropic.Client).Adapter.BaseURL; got != anthropic.DefaultBaseURL {
		t.Errorf("anthropic BaseURL = %q, want default %q", got, anthropic.DefaultBaseURL)
	}
}

func TestEnvVarFor(t *testing.T) {
	cases := []struct {
		model string
		want  string
	}{
		{"claude-sonnet-4-5", "ANTHROPIC_API_KEY"},
		{"claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY"},
		{"gpt-4o", "OPENAI_API_KEY"},
		{"gpt-4o-mini", "OPENAI_API_KEY"},
		{"o1-preview", "OPENAI_API_KEY"},
		{"llama-3-70b", "OPENAI_API_KEY"}, // catch-all
		{"", "OPENAI_API_KEY"},            // pathological, falls through to default
	}
	for _, c := range cases {
		if got := EnvVarFor(c.model); got != c.want {
			t.Errorf("EnvVarFor(%q) = %q, want %q", c.model, got, c.want)
		}
	}
}

func TestPricingFor_KnownAnthropicModels(t *testing.T) {
	cases := []struct {
		model      string
		wantInput  float64
		wantOutput float64
	}{
		{"claude-opus-4-1", 15.00, 75.00},
		{"claude-opus-4", 15.00, 75.00},
		{"claude-sonnet-4-5", 3.00, 15.00},
		{"claude-sonnet-4", 3.00, 15.00},
		{"claude-haiku-4-5", 1.00, 5.00},
		{"claude-3-5-sonnet-20241022", 3.00, 15.00}, // prefix-matched alias
		{"claude-3-5-haiku-20241022", 0.80, 4.00},
	}
	for _, c := range cases {
		t.Run(c.model, func(t *testing.T) {
			p := PricingFor(c.model)
			if p.InputPricePerMillion != c.wantInput {
				t.Errorf("InputPricePerMillion = %v, want %v", p.InputPricePerMillion, c.wantInput)
			}
			if p.OutputPricePerMillion != c.wantOutput {
				t.Errorf("OutputPricePerMillion = %v, want %v", p.OutputPricePerMillion, c.wantOutput)
			}
		})
	}
}

func TestPricingFor_KnownOpenAIModels(t *testing.T) {
	cases := []struct {
		model      string
		wantInput  float64
		wantOutput float64
	}{
		{"gpt-4o-mini", 0.15, 0.60},
		{"gpt-4o", 2.50, 10.00},
		{"gpt-4-turbo", 10.00, 30.00},
		{"gpt-3.5-turbo", 0.50, 1.50},
		// Prefix-matched alias: "gpt-4o-2024-08-06" should pick up
		// gpt-4o pricing, NOT gpt-4o-mini, because gpt-4o-mini is more
		// specific and the table is ordered most-specific first but
		// "gpt-4o-2024..." doesn't start with "gpt-4o-mini".
		{"gpt-4o-2024-08-06", 2.50, 10.00},
	}
	for _, c := range cases {
		t.Run(c.model, func(t *testing.T) {
			p := PricingFor(c.model)
			if p.InputPricePerMillion != c.wantInput || p.OutputPricePerMillion != c.wantOutput {
				t.Errorf("Pricing for %q = %+v, want input=%v output=%v",
					c.model, p, c.wantInput, c.wantOutput)
			}
		})
	}
}

func TestPricingFor_UnknownModelReturnsZero(t *testing.T) {
	cases := []string{
		"unknown-model",
		"llama-3-70b",
		"mistral-large",
		"", // pathological
	}
	for _, model := range cases {
		t.Run(model, func(t *testing.T) {
			p := PricingFor(model)
			if p.InputPricePerMillion != 0 || p.OutputPricePerMillion != 0 {
				t.Errorf("PricingFor(%q) = %+v, want zero Pricing", model, p)
			}
		})
	}
}

func TestPricingFor_MiniBeforeNonMini(t *testing.T) {
	// Regression: gpt-4o-mini must not accidentally pick up gpt-4o
	// pricing due to a re-ordering of the table.
	mini := PricingFor("gpt-4o-mini")
	full := PricingFor("gpt-4o")
	if mini.InputPricePerMillion == full.InputPricePerMillion {
		t.Errorf("gpt-4o-mini and gpt-4o should have different pricing; both = %v",
			mini.InputPricePerMillion)
	}
	if mini.InputPricePerMillion >= full.InputPricePerMillion {
		t.Errorf("gpt-4o-mini cheaper than gpt-4o expected; got mini=%v full=%v",
			mini.InputPricePerMillion, full.InputPricePerMillion)
	}
}

func TestFormatMissingKey(t *testing.T) {
	msg := FormatMissingKey("claude-sonnet-4-5")
	if !strings.Contains(msg, "ANTHROPIC_API_KEY") {
		t.Errorf("missing-key message %q should name the right env var", msg)
	}
	if !strings.Contains(msg, "claude-sonnet-4-5") {
		t.Errorf("missing-key message %q should name the model", msg)
	}

	msg = FormatMissingKey("gpt-4o")
	if !strings.Contains(msg, "OPENAI_API_KEY") {
		t.Errorf("missing-key message %q should name OPENAI_API_KEY", msg)
	}
}
