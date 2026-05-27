// Package provider defines the abstraction over LLM providers (OpenAI,
// Anthropic, etc.) used by the agent loop. It hides each vendor's wire
// format behind a single normalized Request/Response shape so the loop
// can talk to any provider uniformly.
//
// We use typed structs rather than raw JSON for the internal payload,
// so the compiler catches shape mistakes and IDEs autocomplete fields.
//
// Adapters (one per provider) live in subpackages, e.g. provider/openai.
// They convert between this normalized format and the wire format.
package provider

import (
	"context"
	"encoding/json"
)

// Provider is the contract every LLM adapter implements. The agent loop
// holds one Provider and never knows which vendor it's talking to.
//
// Kept intentionally small (2 methods) — see the Go rule "small
// interfaces, 1-3 methods". Streaming will be added later as a separate
// optional interface tools can type-assert for, not bolted onto Provider.
type Provider interface {
	// Name returns the provider's identifier (e.g. "openai", "anthropic").
	// Used for logging and provider-specific behavior in the loop.
	Name() string

	// Call sends a synchronous request and returns the full response.
	// Honors ctx for cancellation and timeouts.
	Call(ctx context.Context, req Request) (Response, error)
}

// ContentKind tags the variant of a ContentBlock.
//
// This is the project's tagged-struct pattern for representing "one of
// several variants" (Go has no native sum types). The same shape
// repeats for TurnResult, LoopAction, etc. Iota-based int constants
// give us exhaustive switch coverage and zero allocation.
type ContentKind int

const (
	// ContentText — plain text payload; valid field: Text.
	ContentText ContentKind = iota

	// ContentImage — image payload; valid fields: image-related ones
	// (deferred; vision support comes with the VLM workflow slot).
	ContentImage
)

// Request is the normalized LLM call payload. Adapters convert this into
// vendor-specific JSON. Mirrors the OpenAI Chat Completions schema
// because that's the lingua franca of LLM APIs.
type Request struct {
	// Model identifier passed through to the provider (e.g. "gpt-4o").
	Model string

	// Messages is the full conversation history sent to the model.
	// Order matters: system prompt first, then alternating user/assistant.
	Messages []Message

	// Tools is the set of tools the model is allowed to call this turn.
	// Empty means "no tool use, just respond".
	Tools []ToolSchema
}

// Message is one turn in the conversation. Role + content is the core;
// the other fields are populated only for specific roles (assistant
// emits ToolCalls, role="tool" messages carry ToolCallID + Name).
//
// Modeled after the OpenAI Chat Completions message object so we can
// pass it through to OpenAI-compatible providers with minimal massaging.
type Message struct {
	// Role: "system" | "user" | "assistant" | "tool".
	Role string

	// Content is the message body, as ordered blocks. Multiple blocks
	// support multimodal input (e.g. text + image). Single text block
	// is by far the common case.
	Content []ContentBlock

	// ToolCalls — populated only when Role == "assistant" and the model
	// chose to call one or more tools this turn.
	ToolCalls []ToolCall

	// ToolCallID — populated only when Role == "tool"; identifies which
	// assistant tool_call this message is the result of.
	ToolCallID string

	// Name — populated only when Role == "tool"; the tool that produced
	// this result. OpenAI-shaped responses include this for clarity.
	Name string
}

// ContentBlock represents one piece of message content (text, image,
// tool use, or tool result). Only the fields valid for the current
// Kind are populated; others are zero. Constructors will enforce the
// invariant once vision lands — callers should prefer them over raw
// struct literals at that point.
type ContentBlock struct {
	// Kind selects which other fields are meaningful.
	Kind ContentKind

	// Text — valid when Kind == ContentText.
	Text string
}

// ToolCall is the model's request to invoke a specific tool with given
// arguments. The loop sees these in a Response and dispatches each to
// the tool registry, then feeds the ToolResult back as a Message with
// Role == "tool".
type ToolCall struct {
	// ID — unique per call within a turn, used to match the tool result
	// back to the call when forming the next request.
	ID string

	// Name — the tool the model wants to invoke (must exist in registry).
	Name string

	// Arguments — raw JSON; each tool parses its own args. We don't
	// pre-decode because the schema varies per tool and would force a
	// switch in this package.
	Arguments json.RawMessage
}

// ToolSchema describes a tool to the model so it knows what's callable
// and how. We pass these in Request.Tools; the provider serializes them
// into the vendor's tool-use schema.
type ToolSchema struct {
	// Name — must match the tool's Name() in the registry; the model
	// echoes this back in ToolCall.Name.
	Name string

	// Description — human-readable hint shown to the model. Quality
	// matters: poor descriptions degrade tool selection.
	Description string

	// Parameters — JSON Schema for the tool's args. Raw so each tool
	// supplies its own without this package needing tool-specific types.
	Parameters json.RawMessage
}

// Response is the normalized model reply. Adapters convert vendor
// responses into this shape. Either Content is non-empty (final answer)
// or ToolCalls is non-empty (tool-use turn) — never both meaningfully,
// per OpenAI Chat Completions semantics.
type Response struct {
	// Content — the model's text reply, empty if this is a tool-use turn.
	Content string

	// ToolCalls — non-empty when the model chose to call tools instead
	// of replying. The loop dispatches each and continues iterating.
	ToolCalls []ToolCall

	// Usage — token accounting reported by the provider. Drives
	// cost.Tracker and the prompt-cache calibration step.
	Usage Usage

	// FinishReason: "stop" | "length" | "tool_calls" | "content_filter"
	// per OpenAI conventions. Used to detect truncation.
	FinishReason string
}

// Usage is the token accounting for one provider call. We trust the
// API-reported counts (per the brief's "anchor budget on reported
// prompt_tokens, not local estimates" rule) — local tiktoken estimates
// drift on system-injected content and lose accuracy.
type Usage struct {
	// PromptTokens — total input tokens this turn (including cached).
	PromptTokens int

	// CachedTokens — subset of PromptTokens served from the provider's
	// prompt cache. Drives the two-part caching feature (T3.5).
	CachedTokens int

	// CompletionTokens — output tokens generated by the model.
	CompletionTokens int
}
