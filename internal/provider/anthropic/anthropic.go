// Package anthropic is the Anthropic Messages API adapter. It converts
// normalized provider.Request/Response shapes to and from the Anthropic
// wire format (https://docs.anthropic.com/en/api/messages).
//
// The Messages API diverges from OpenAI's Chat Completions in three
// significant places, and each shows up in the translation code below:
//
//  1. System prompts live at the top level (not in the messages array).
//  2. There are only two message roles — user and assistant. Tool
//     results are content blocks inside a synthetic user message.
//  3. Prompt caching is opt-in via per-block cache_control markers
//     rather than automatic.
//
// We target the Messages API directly rather than going through
// Anthropic's older Completions API; Messages is the documented path
// for tool use and the only one that supports streaming.
package anthropic

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// ErrEmptyContent is returned by ParseResponse when the API returns a
// well-formed envelope with an empty "content" array. Mirrors openai's
// ErrEmptyChoices so callers can match the analogous failure mode with
// the same intent.
var ErrEmptyContent = errors.New("anthropic: no content in response")

// DefaultBaseURL is the public Anthropic API root. Override via
// Adapter.BaseURL for proxies or Anthropic-compatible endpoints.
const DefaultBaseURL = "https://api.anthropic.com/v1"

// DefaultAPIVersion is the value sent in the anthropic-version header.
// Anthropic requires the header on every request; pinning to a stable
// version date keeps the wire format predictable.
const DefaultAPIVersion = "2023-06-01"

// DefaultMaxTokens is the default cap on output tokens. Anthropic
// rejects requests without max_tokens, unlike OpenAI which defaults
// server-side. 4096 fits typical agent turns; bump via the adapter if
// a workload truncates.
const DefaultMaxTokens = 4096

// Thinking budget constants for fixed-mode models (Claude 3.7 /
// Opus 4.0–4.5 / Sonnet 4.0–4.5 / Haiku 4.5). Values come from
// production-tuned defaults used in published agent loops, not
// theoretical sweet spots:
//
//   - 4096  = smallest budget that produces usefully different
//             output than no thinking at all.
//   - 16384 = strong reasoning headroom for non-trivial tasks.
//   - 31999 = one below Anthropic's 32K hard cap; the real "high".
//
// Claude 4.6+ models use adaptive thinking instead (the model
// self-regulates its own budget) — see adaptiveThinkingPattern.
const (
	thinkingBudgetLow    = 4096
	thinkingBudgetMedium = 16384
	thinkingBudgetHigh   = 31999
)

// Adapter holds the configurable wire-level state for an Anthropic
// provider. Stateless except for BaseURL — safe to share across
// goroutines.
type Adapter struct {
	// BaseURL is the API root; "" falls back to DefaultBaseURL.
	BaseURL string

	// APIVersion overrides the anthropic-version header value; "" falls
	// back to DefaultAPIVersion. Useful when pinning to a beta header
	// to access new features.
	APIVersion string

	// MaxTokens is the default cap for output tokens applied to every
	// request that doesn't already specify one. Zero falls back to
	// DefaultMaxTokens.
	MaxTokens int
}

// New returns an Adapter targeting the public Anthropic API.
func New() Adapter {
	return Adapter{
		BaseURL:    DefaultBaseURL,
		APIVersion: DefaultAPIVersion,
		MaxTokens:  DefaultMaxTokens,
	}
}

// Name returns the provider identifier used in logs and routing.
func (a Adapter) Name() string { return "anthropic" }

// MessagesURL is the full URL for the /messages endpoint. Empty
// BaseURL falls back to the default so the zero Adapter is usable.
func (a Adapter) MessagesURL() string {
	base := a.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	return base + "/messages"
}

// APIVersionHeader returns the value to send in the anthropic-version
// header. Empty Adapter.APIVersion falls back to the default.
func (a Adapter) APIVersionHeader() string {
	if a.APIVersion == "" {
		return DefaultAPIVersion
	}
	return a.APIVersion
}

// BuildRequest serializes a normalized provider.Request into the
// Anthropic Messages JSON payload. The output is ready to POST as the
// request body.
//
// Three translation passes happen here:
//
//  1. System messages are pulled out of req.Messages and folded into a
//     top-level "system" field (Anthropic's API requires this shape).
//  2. The remaining messages are walked and translated: user/assistant
//     pass through with their content blocks; role:"tool" messages
//     become tool_result content blocks inside a user message, with
//     consecutive tool messages coalesced into one user message (the
//     API requires strict user/assistant alternation).
//  3. Assistant ToolCalls are expanded into tool_use content blocks
//     alongside any text the assistant also emitted.
func (a Adapter) BuildRequest(req provider.Request) ([]byte, error) {
	return json.Marshal(a.buildPayload(req, false))
}

// BuildStreamRequest is the streaming twin of BuildRequest. Same
// payload plus "stream": true. Unlike OpenAI's streaming we don't need
// any extra options — Anthropic's usage data flows through the
// message_start / message_delta events automatically. Separate method
// rather than a flag on BuildRequest so callers state their intent at
// the call site.
func (a Adapter) BuildStreamRequest(req provider.Request) ([]byte, error) {
	return json.Marshal(a.buildPayload(req, true))
}

// buildPayload is the shared body of BuildRequest and BuildStreamRequest.
// Returns a fresh map per call so callers can't mutate each other's
// state — matches the immutability rule used everywhere else in this
// codebase.
func (a Adapter) buildPayload(req provider.Request, stream bool) map[string]any {
	systemText, others := extractSystem(req.Messages)
	messages := buildMessages(others)

	maxTokens := a.MaxTokens
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}

	// Resolve thinking block first so we can bump max_tokens if a
	// fixed budget is going to land in the payload — Anthropic
	// rejects requests where max_tokens <= budget_tokens.
	thinking, budgetTokens := thinkingBlock(req.Model, req.ReasoningEffort)
	if budgetTokens > 0 {
		needed := budgetTokens + DefaultMaxTokens
		if maxTokens < needed {
			maxTokens = needed
		}
	}

	payload := map[string]any{
		"model":      req.Model,
		"max_tokens": maxTokens,
		"messages":   messages,
	}

	// System prompt with cache_control on the block. The top-level
	// "system" field accepts either a string or an array of text
	// blocks; the array form is the only way to attach cache_control,
	// so we always emit the array shape when the prompt is non-empty.
	if systemText != "" {
		payload["system"] = []map[string]any{
			{
				"type":          "text",
				"text":          systemText,
				"cache_control": map[string]any{"type": "ephemeral"},
			},
		}
	}

	if len(req.Tools) > 0 {
		payload["tools"] = convertTools(req.Tools)
	}

	if thinking != nil {
		payload["thinking"] = thinking
	}

	if stream {
		payload["stream"] = true
	}

	return payload
}

// fixedThinkingPattern matches Claude families that take a fixed
// budget_tokens value on the thinking block — Claude 3.7, Opus
// 4.0–4.5, Sonnet 4.0–4.5, Haiku 4.5. Adding a new fixed-budget
// model is a one-line append here.
var fixedThinkingPattern = regexp.MustCompile(
	`^(claude-3-7|claude-opus-4-[0-5]|claude-sonnet-4-[0-5]|claude-haiku-4-5)`,
)

// adaptiveThinkingPattern matches Claude families that support the
// adaptive thinking mode — Claude 4.6+ (Opus, Sonnet, Haiku) and
// future Claude 5 lines. Adaptive mode lets the model self-regulate
// its budget per call, which beats a fixed setting for capable
// models. Pattern is checked BEFORE fixedThinkingPattern so a model
// matched by both falls into the adaptive path.
var adaptiveThinkingPattern = regexp.MustCompile(
	`^(claude-opus-4-[6-9]|claude-sonnet-4-[6-9]|claude-haiku-4-[6-9]|claude-(opus|sonnet|haiku)-[5-9])`,
)

// thinkingBlock returns the value to set on payload["thinking"]
// (or nil to omit) along with the fixed budget token count when
// one was chosen (0 for adaptive, disabled, or omitted). The
// budget is returned separately so buildPayload can bump
// max_tokens above it without re-parsing the map.
//
// Branches:
//
//  1. ReasoningEffortUnset → nil, 0. The caller didn't ask for
//     thinking; let the model's defaults apply.
//  2. ReasoningEffortNone → {"type":"disabled"} on supporting
//     models, nil otherwise. Explicit suppression for callers
//     who measured that thinking hurts on a specific workload.
//  3. Low/Medium/High on an adaptive model → {"type":"adaptive"}.
//     The model picks its own budget. Operator's level is honored
//     by the model rather than by a hardcoded budget here.
//  4. Low/Medium/High on a fixed-budget model →
//     {"type":"enabled","budget_tokens":N} with N from the
//     thinkingBudget* constants.
//  5. Any non-supporting model → nil. Silent omit; no error.
func thinkingBlock(model string, effort provider.ReasoningEffort) (map[string]any, int) {
	if effort == provider.ReasoningEffortUnset {
		return nil, 0
	}

	adaptive := adaptiveThinkingPattern.MatchString(model)
	fixed := fixedThinkingPattern.MatchString(model)
	if !adaptive && !fixed {
		// Not a thinking-capable model — silently omit even when
		// the caller asked for None. There's nothing to suppress
		// because nothing was going to happen anyway.
		return nil, 0
	}

	if effort == provider.ReasoningEffortNone {
		return map[string]any{"type": "disabled"}, 0
	}

	if adaptive {
		// 4.6+ models self-regulate. We don't translate the
		// operator's level into a budget here — that's the
		// model's job.
		return map[string]any{"type": "adaptive"}, 0
	}

	// Fixed-budget path. Unknown future ReasoningEffort values
	// fall through to medium so the call still works rather than
	// silently dropping the thinking directive.
	budget := thinkingBudgetMedium
	switch effort {
	case provider.ReasoningEffortLow:
		budget = thinkingBudgetLow
	case provider.ReasoningEffortMedium:
		budget = thinkingBudgetMedium
	case provider.ReasoningEffortHigh:
		budget = thinkingBudgetHigh
	}
	return map[string]any{
		"type":          "enabled",
		"budget_tokens": budget,
	}, budget
}

// extractSystem walks the messages, returning the joined system-prompt
// text and the non-system messages in their original order. Multiple
// system messages are joined with two newlines (matching OpenAI's
// implicit join behavior); typically there's at most one anyway.
func extractSystem(msgs []provider.Message) (string, []provider.Message) {
	var (
		sys    string
		others []provider.Message
	)
	for _, m := range msgs {
		if m.Role == "system" {
			text := joinText(m.Content)
			if text == "" {
				continue
			}
			if sys == "" {
				sys = text
			} else {
				sys = sys + "\n\n" + text
			}
			continue
		}
		others = append(others, m)
	}
	return sys, others
}

// buildMessages translates non-system messages into Anthropic's
// messages array. Tool-result messages are coalesced into the
// preceding user message when one exists, per Anthropic's alternation
// requirement (user/assistant/user/assistant…).
func buildMessages(msgs []provider.Message) []map[string]any {
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case "user":
			out = append(out, map[string]any{
				"role":    "user",
				"content": userTextBlocks(m.Content),
			})

		case "assistant":
			out = append(out, map[string]any{
				"role":    "assistant",
				"content": assistantBlocks(m.Content, m.ToolCalls),
			})

		case "tool":
			tr := toolResultBlock(m.ToolCallID, joinText(m.Content))
			if n := len(out); n > 0 && out[n-1]["role"] == "user" {
				// Append the new tool_result onto the existing user
				// message's content. The "is all tool_results"
				// invariant holds because we only put tool_results
				// into a synthesized user message — never mix with
				// real user text.
				existing := out[n-1]["content"].([]map[string]any)
				out[n-1]["content"] = append(existing, tr)
			} else {
				out = append(out, map[string]any{
					"role":    "user",
					"content": []map[string]any{tr},
				})
			}

		default:
			// Unknown role: drop silently. Strict validation belongs
			// at the loop level, not the adapter.
		}
	}
	return out
}

// userTextBlocks converts the user message's content blocks into
// Anthropic text blocks. ContentImage blocks are skipped for v2 (vision
// support comes with the VLM workflow slot).
func userTextBlocks(blocks []provider.ContentBlock) []map[string]any {
	out := make([]map[string]any, 0, len(blocks))
	for _, b := range blocks {
		if b.Kind != provider.ContentText {
			continue
		}
		if b.Text == "" {
			continue
		}
		out = append(out, map[string]any{
			"type": "text",
			"text": b.Text,
		})
	}
	if len(out) == 0 {
		// Anthropic requires a non-empty content array. Emit a single
		// empty text block as a defensive floor — better than a 400
		// from the server.
		out = append(out, map[string]any{"type": "text", "text": ""})
	}
	return out
}

// assistantBlocks builds the assistant message's content array. Text
// blocks come first (preserving order); tool_use blocks for any
// ToolCalls follow.
func assistantBlocks(content []provider.ContentBlock, calls []provider.ToolCall) []map[string]any {
	out := make([]map[string]any, 0, len(content)+len(calls))
	for _, b := range content {
		if b.Kind == provider.ContentText && b.Text != "" {
			out = append(out, map[string]any{
				"type": "text",
				"text": b.Text,
			})
		}
	}
	for _, c := range calls {
		// Anthropic's "input" field carries the parsed tool arguments
		// as a JSON OBJECT (not a JSON-encoded string like OpenAI).
		// Our ToolCall.Arguments is json.RawMessage — pass it
		// through verbatim so it serializes as an object.
		var input json.RawMessage
		if len(c.Arguments) > 0 {
			input = c.Arguments
		} else {
			input = json.RawMessage(`{}`)
		}
		out = append(out, map[string]any{
			"type":  "tool_use",
			"id":    c.ID,
			"name":  c.Name,
			"input": input,
		})
	}
	return out
}

// toolResultBlock builds a single tool_result content block to embed
// in a synthesized user message.
func toolResultBlock(toolUseID, text string) map[string]any {
	return map[string]any{
		"type":         "tool_result",
		"tool_use_id":  toolUseID,
		"content":      text,
	}
}

// convertTools wraps each ToolSchema in Anthropic's tool envelope.
// Field rename from "parameters" → "input_schema" is the only
// difference from OpenAI's tool format; the JSON Schema body is
// identical.
func convertTools(tools []provider.ToolSchema) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		params := t.Parameters
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object"}`)
		}
		out = append(out, map[string]any{
			"name":         t.Name,
			"description":  t.Description,
			"input_schema": params,
		})
	}
	return out
}

// messagesResponse is the temp-struct shape we unmarshal an Anthropic
// Messages response into. Kept private — outside callers see only the
// normalized provider.Response that ParseResponse returns.
type messagesResponse struct {
	ID         string `json:"id"`
	Type       string `json:"type"`
	Role       string `json:"role"`
	Model      string `json:"model"`
	StopReason string `json:"stop_reason"`

	Content []struct {
		Type string `json:"type"`
		// text blocks
		Text string `json:"text"`
		// tool_use blocks
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	} `json:"content"`

	Usage struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	} `json:"usage"`
}

// ParseResponse converts an Anthropic Messages response body into the
// normalized provider.Response. Text blocks are concatenated into
// Content; tool_use blocks become ToolCalls with Input passed through
// as json.RawMessage (Anthropic returns it as a parsed object, but
// json.RawMessage preserves the encoded bytes).
//
// Stop-reason translation maps Anthropic vocabulary to our
// OpenAI-flavored FinishReason field — see mapStopReason for the table.
//
// Usage translation: Anthropic separates input_tokens from
// cache_read_input_tokens, but for cost-tracking purposes we want the
// total (cache reads are billed at a lower rate but they're still
// charged). PromptTokens carries the sum; CachedTokens carries the
// cache-read portion so the cost tracker can apply the cached rate.
func (a Adapter) ParseResponse(body []byte) (provider.Response, error) {
	var raw messagesResponse
	if err := json.Unmarshal(body, &raw); err != nil {
		return provider.Response{}, fmt.Errorf("anthropic: parse response: %w", err)
	}
	if len(raw.Content) == 0 {
		return provider.Response{}, ErrEmptyContent
	}

	var (
		contentBuf string
		toolCalls  []provider.ToolCall
	)
	for _, block := range raw.Content {
		switch block.Type {
		case "text":
			contentBuf += block.Text
		case "tool_use":
			args := block.Input
			if len(args) == 0 {
				args = json.RawMessage(`{}`)
			}
			toolCalls = append(toolCalls, provider.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: args,
			})
		}
	}

	return provider.Response{
		Content:      contentBuf,
		ToolCalls:    toolCalls,
		FinishReason: mapStopReason(raw.StopReason),
		Usage: provider.Usage{
			PromptTokens:     raw.Usage.InputTokens + raw.Usage.CacheReadInputTokens,
			CompletionTokens: raw.Usage.OutputTokens,
			CachedTokens:     raw.Usage.CacheReadInputTokens,
		},
	}, nil
}

// mapStopReason translates Anthropic's stop_reason vocabulary into the
// OpenAI-flavored FinishReason values the rest of the codebase already
// understands. Unknown values pass through as-is rather than being
// silently coerced — easier to debug a surprising stop_reason than
// to wonder why it became "stop".
func mapStopReason(s string) string {
	switch s {
	case "end_turn", "stop_sequence", "pause_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "":
		return ""
	default:
		return s
	}
}

// joinText concatenates ContentText blocks; ContentImage blocks are
// dropped silently for v2. Fast path for the common single-block case.
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
