// Package openai is the OpenAI Chat Completions adapter. It converts
// normalized provider.Request/Response shapes to and from the OpenAI
// wire format, which is the lingua franca cloned by most OpenAI-compatible
// providers (Groq, Mistral, DeepInfra, OpenRouter, local servers).
//
// We deliberately target the older Chat Completions API
// (/v1/chat/completions) rather than the newer Responses API
// (/v1/responses). Responses API is OpenAI-specific; Chat Completions
// is the cross-vendor standard supported by every OpenAI-compatible
// server. A separate `openairesponses` adapter can be added later
// if/when reasoning models (o1, o3) are needed.
package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// ErrEmptyChoices is returned by ParseResponse when the API returns a
// well-formed envelope with an empty "choices" array — a contract
// violation we don't try to recover from.
var ErrEmptyChoices = errors.New("openai: no choices in response")

// DefaultBaseURL is the public OpenAI API root. Override via Adapter.BaseURL
// for proxies, Azure, or any OpenAI-compatible endpoint.
const DefaultBaseURL = "https://api.openai.com/v1"

// Adapter holds the configurable wire-level state for an OpenAI-shaped
// provider. Stateless except for BaseURL — safe to share across goroutines.
type Adapter struct {
	// BaseURL is the API root; "" falls back to DefaultBaseURL.
	BaseURL string
}

// New returns an Adapter targeting the public OpenAI API.
func New() Adapter {
	return Adapter{BaseURL: DefaultBaseURL}
}

// Name returns the provider identifier used in logs and routing.
func (a Adapter) Name() string { return "openai" }

// ChatCompletionsURL is the full URL for /chat/completions. Empty BaseURL
// falls back to the default so the zero Adapter is still usable.
func (a Adapter) ChatCompletionsURL() string {
	base := a.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	return base + "/chat/completions"
}

// BuildRequest serializes a normalized provider.Request into the OpenAI
// Chat Completions JSON payload. The output is ready to POST as the
// request body.
func (a Adapter) BuildRequest(req provider.Request) ([]byte, error) {
	return json.Marshal(buildPayload(req, false))
}

// BuildStreamRequest is the streaming twin of BuildRequest. Same payload
// plus "stream": true and "stream_options": {"include_usage": true} so
// the server emits a final usage chunk we can fold into Response.Usage.
// Separate method rather than a flag on BuildRequest so callers state
// their intent at the call site and the two paths can't accidentally
// swap.
func (a Adapter) BuildStreamRequest(req provider.Request) ([]byte, error) {
	return json.Marshal(buildPayload(req, true))
}

// buildPayload is the shared body of BuildRequest and BuildStreamRequest.
// Returns a fresh map per call so callers can't mutate each other's
// state — matches the immutability rule used everywhere else in this
// codebase.
func buildPayload(req provider.Request, stream bool) map[string]any {
	payload := map[string]any{
		"model":    req.Model,
		"messages": convertMessages(req.Messages),
	}
	if len(req.Tools) > 0 {
		payload["tools"] = convertTools(req.Tools)
	}
	if effort := openAIReasoningEffort(req.Model, req.ReasoningEffort); effort != "" {
		payload["reasoning_effort"] = effort
	}
	if stream {
		payload["stream"] = true
		// include_usage asks the server for a final chunk containing
		// token counts; without it the usage object is absent from
		// every chunk and cost tracking goes blind on streamed calls.
		payload["stream_options"] = map[string]any{"include_usage": true}
	}
	return payload
}

// reasoningEffortModelPattern matches the OpenAI families that
// support the reasoning_effort knob: the o-series (o1, o3, o4) and
// GPT-5. Anchored at the start so substring matches like
// "chatgpt-4-o1-tuned" don't trigger. Updating the pattern when
// OpenAI ships a new reasoning family is the path of record;
// OpenAI does not expose a capability-query endpoint we could call.
var reasoningEffortModelPattern = regexp.MustCompile(
	`^(o1|o3|o4|gpt-5)`,
)

// openAIReasoningEffort returns the level string to emit in the
// payload, or "" to omit the field. Rules:
//
//   - Unset            → "" (caller didn't configure it)
//   - None             → "" (OpenAI has no "disable" — silently omit)
//   - Low/Medium/High  → the level string, only on supporting models
//   - Non-supporting model → "" regardless of level
//
// Split out so tests can exercise the mapping without round-
// tripping a whole payload.
func openAIReasoningEffort(model string, effort provider.ReasoningEffort) string {
	if effort == provider.ReasoningEffortUnset || effort == provider.ReasoningEffortNone {
		return ""
	}
	if !reasoningEffortModelPattern.MatchString(model) {
		return ""
	}
	switch effort {
	case provider.ReasoningEffortLow:
		return "low"
	case provider.ReasoningEffortMedium:
		return "medium"
	case provider.ReasoningEffortHigh:
		return "high"
	default:
		// Unknown future value — be conservative and omit rather
		// than send a value the API will reject.
		return ""
	}
}

// convertMessages translates normalized Messages into the OpenAI Chat
// Completions message-array shape. Behavior per role:
//
//   - "system" / "user" — role + content (text blocks joined).
//   - "assistant"       — role + content (null when only tool_calls present)
//     plus a "tool_calls" array.
//   - "tool"            — role + tool_call_id + content (the result string).
//
// For v1 we collapse []ContentBlock to one string by joining ContentText
// blocks. ContentImage blocks are skipped — vision support will come
// with the VLM workflow slot.
func convertMessages(msgs []provider.Message) []map[string]any {
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		text := joinText(m.Content)
		msg := map[string]any{"role": m.Role}

		switch m.Role {
		case "assistant":
			if len(m.ToolCalls) > 0 {
				msg["tool_calls"] = convertToolCalls(m.ToolCalls)
				if text == "" {
					// OpenAI requires content to be present; null
					// signals "no text, only tool calls".
					msg["content"] = nil
				} else {
					msg["content"] = text
				}
			} else {
				msg["content"] = text
			}
		case "tool":
			msg["tool_call_id"] = m.ToolCallID
			msg["content"] = text
		default: // "system", "user", and any future roles
			msg["content"] = text
		}

		out = append(out, msg)
	}
	return out
}

// convertToolCalls wraps each ToolCall in the OpenAI function-call envelope.
//
// Subtle: OpenAI specifies the `arguments` field as a JSON-encoded string,
// NOT raw JSON — yes, a string containing JSON. Our ToolCall.Arguments is
// json.RawMessage, so string(c.Arguments) gives us the encoded form ready
// to drop in as a string value (json.Marshal will then escape it once
// when serializing the outer map).
func convertToolCalls(calls []provider.ToolCall) []map[string]any {
	out := make([]map[string]any, 0, len(calls))
	for _, c := range calls {
		args := string(c.Arguments)
		if args == "" {
			args = "{}"
		}
		out = append(out, map[string]any{
			"id":   c.ID,
			"type": "function",
			"function": map[string]any{
				"name":      c.Name,
				"arguments": args,
			},
		})
	}
	return out
}

// convertTools wraps each ToolSchema in the OpenAI function-tool envelope:
//
//	{"type": "function", "function": {name, description, parameters}}
//
// Parameters is json.RawMessage so the JSON Schema flows through verbatim
// without re-parsing. Empty Parameters defaults to {"type":"object"} —
// OpenAI rejects tools that lack a parameter schema entirely.
func convertTools(tools []provider.ToolSchema) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		params := t.Parameters
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object"}`)
		}
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  params,
			},
		})
	}
	return out
}

// chatCompletionResponse is the temp-struct shape we unmarshal an
// OpenAI Chat Completions response into. Kept private — outside callers
// see only the normalized provider.Response that ParseResponse returns.
//
// Content is a pointer string so JSON null is allowed (assistant tool-call
// turns have content: null). Arguments is plain string because that's
// how OpenAI encodes it on the wire (a JSON-escaped string containing
// JSON); we wrap it in json.RawMessage when building the public ToolCall.
type chatCompletionResponse struct {
	ID     string `json:"id"`
	Model  string `json:"model"`
	Object string `json:"object"`

	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role      string  `json:"role"`
			Content   *string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`

	Usage struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

// ParseResponse converts an OpenAI Chat Completions response body into
// the normalized provider.Response. Symmetric with BuildRequest:
//
//   - choices[0].message.content (string|null) → Response.Content (empty
//     string when null).
//   - choices[0].message.tool_calls[i].function.{name,arguments} →
//     provider.ToolCall{Name, Arguments}. arguments is a JSON-encoded
//     string on the wire; we wrap it verbatim into json.RawMessage so
//     downstream tool code can json.Unmarshal directly.
//   - usage.prompt_tokens_details.cached_tokens → Usage.CachedTokens
//     (drives the T3.5 prompt-cache feature).
//
// Errors: malformed JSON wraps the json.Unmarshal error with %w; an
// empty choices array returns ErrEmptyChoices unwrapped so callers can
// match with errors.Is.
func (a Adapter) ParseResponse(body []byte) (provider.Response, error) {
	var raw chatCompletionResponse
	if err := json.Unmarshal(body, &raw); err != nil {
		return provider.Response{}, fmt.Errorf("openai: parse response: %w", err)
	}
	if len(raw.Choices) == 0 {
		return provider.Response{}, ErrEmptyChoices
	}

	choice := raw.Choices[0]

	var toolCalls []provider.ToolCall
	if n := len(choice.Message.ToolCalls); n > 0 {
		toolCalls = make([]provider.ToolCall, 0, n)
		for _, tc := range choice.Message.ToolCalls {
			toolCalls = append(toolCalls, provider.ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			})
		}
	}

	content := ""
	if choice.Message.Content != nil {
		content = *choice.Message.Content
	}

	return provider.Response{
		Content:      content,
		ToolCalls:    toolCalls,
		FinishReason: choice.FinishReason,
		Usage: provider.Usage{
			PromptTokens:     raw.Usage.PromptTokens,
			CompletionTokens: raw.Usage.CompletionTokens,
			CachedTokens:     raw.Usage.PromptTokensDetails.CachedTokens,
		},
	}, nil
}

// joinText concatenates ContentText blocks; ContentImage blocks are
// dropped silently for v1. Fast path for the common single-block case.
func joinText(blocks []provider.ContentBlock) string {
	if len(blocks) == 0 {
		return ""
	}
	if len(blocks) == 1 && blocks[0].Kind == provider.ContentText {
		return blocks[0].Text
	}
	var s string
	for _, b := range blocks {
		if b.Kind == provider.ContentText {
			s += b.Text
		}
	}
	return s
}
