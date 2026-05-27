package agents

import (
	"encoding/json"

	"github.com/ashishgupta/opendev-go/internal/provider"
	"github.com/ashishgupta/opendev-go/internal/tools"
)

// DefaultSystemPrompt is the v1 default. Intentionally short — fancy
// prompt composition (persona, project context, prefix-cache split) can
// come later. For now, just enough to tell the model it's a coding
// agent with tools.
const DefaultSystemPrompt = `You are a helpful coding assistant. ` +
	`You have access to tools to read files and run shell commands. ` +
	`Use them when needed to answer the user's question or complete their task. ` +
	`When you have the final answer, respond with text directly (no tool calls).`

// SchemasFor converts every tool in the registry into the wire-format
// ToolSchemas the provider request needs. The loop calls this once per
// iteration; cheap because the registry is in-process and tools are
// stateless. Names are returned sorted (Registry.Names already does
// this) so the resulting Tools array is byte-stable across turns —
// which keeps any future prefix-cached portion stable.
func SchemasFor(reg *tools.Registry) []provider.ToolSchema {
	names := reg.Names()
	out := make([]provider.ToolSchema, 0, len(names))
	for _, name := range names {
		tool, ok := reg.Get(name)
		if !ok {
			// Name listed but tool gone — shouldn't happen since the
			// registry is single-process. Skip defensively.
			continue
		}
		out = append(out, provider.ToolSchema{
			Name:        tool.Name(),
			Description: tool.Description(),
			Parameters:  tool.Schema(),
		})
	}
	return out
}

// SystemMessage builds the leading system Message. Empty prompt falls
// back to DefaultSystemPrompt — keeps callers from accidentally sending
// an empty system message.
func SystemMessage(prompt string) provider.Message {
	if prompt == "" {
		prompt = DefaultSystemPrompt
	}
	return provider.Message{
		Role: "system",
		Content: []provider.ContentBlock{
			{Kind: provider.ContentText, Text: prompt},
		},
	}
}

// UserMessage wraps a plain text string as a user-role Message.
func UserMessage(text string) provider.Message {
	return provider.Message{
		Role: "user",
		Content: []provider.ContentBlock{
			{Kind: provider.ContentText, Text: text},
		},
	}
}

// ToolResultMessage builds the "role: tool" Message the loop appends
// after dispatching a ToolCall. Encodes both success and failure into
// the single text observation the model will see next turn.
//
// On failure, the error is prefixed with [ERROR] so the model can
// distinguish "this is the tool's output" from "this is what went
// wrong". OpenAI's API would let us pass structured error info, but
// our normalized layer is text-only for v1.
func ToolResultMessage(callID, name string, result tools.ToolResult) provider.Message {
	text := result.Output
	if !result.Success && result.Error != "" {
		if text != "" {
			text += "\n"
		}
		text += "[ERROR] " + result.Error
	}
	return provider.Message{
		Role:       "tool",
		ToolCallID: callID,
		Name:       name,
		Content: []provider.ContentBlock{
			{Kind: provider.ContentText, Text: text},
		},
	}
}

// ensureJSON returns args verbatim if non-empty, else the empty object
// "{}" — many tools call json.Unmarshal expecting at least valid JSON.
// Without this, a no-arg tool call would fail with a parse error
// before the tool ever sees ctx.
func ensureJSON(b json.RawMessage) json.RawMessage {
	if len(b) == 0 {
		return json.RawMessage("{}")
	}
	return b
}
