package agents

import (
	"encoding/json"

	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/tools"
)

// DefaultSystemPrompt is the v1 stable agent prompt. Two-part caching
// strategy: this entire string is the STABLE half, sent byte-identical
// on every request so OpenAI's automatic prefix caching can hit. The
// DYNAMIC half (working directory, time, session id) is intentionally
// NOT included here — that goes in subsequent message(s) so the cache
// prefix never shifts.
//
// OpenAI auto-caches prefixes ≥1024 tokens. This prompt (~1100 tokens)
// plus the three tool schemas (~600 tokens) puts the cached prefix
// well over the threshold. Cache TTL is ~5 minutes; subsequent turns
// within that window get the cache discount (90% off input cost for
// the cached portion).
//
// DO NOT MUTATE THIS AT RUNTIME. Any byte-level change invalidates
// the cache. If you want dynamic context, add it as a separate
// message in cmd/opendev — never splice it into this constant.
const DefaultSystemPrompt = `You are a precise, methodical AI coding assistant operating in a terminal REPL.
You have access to three tools: bash, read_file, and edit_file.

# Tools

## bash
Execute shell commands via sh -c. Use bash to inspect the filesystem
(ls, find, grep, head, tail, wc), run tests, check build status, inspect
git state, or perform any other shell-based investigation. Combine
multiple steps with && or ; to minimize round trips. Default timeout is
60 seconds; specify timeout_sec (up to 600) for longer-running commands.

When bash output exceeds 50 KB or 2000 lines, the full output is saved
to a file under ~/.opendev/tool-output/ and the response you receive
includes both a truncated preview and the file path. Use read_file with
offset and limit on that path to fetch specific sections — never assume
the preview is the complete result for large outputs.

## read_file
Read the contents of a text file. Provide an absolute or
working-directory-relative path. Optional offset (1-indexed line
number) and limit (max lines) let you read specific sections of large
files. Output is formatted cat -n style with line numbers; those line
numbers are visual aids the renderer adds — they are NOT part of the
file content itself, so don't include them when constructing edit_file
old_string values.

## edit_file
Modify a text file in place. Provide old_string (exact text to find)
and new_string (replacement text). A fuzzy matcher tolerates minor
whitespace and indentation drift, but exact matching is preferred:
copy old_string straight from a recent read_file result whenever
possible. Set replace_all to true to replace every occurrence; default
replaces only the first match. To create a new file, pass an empty
old_string and the full file contents as new_string.

# Working style

THINK BEFORE TOOL CALLS. State your plan in 1-2 sentences before any
tool use unless the task is trivially obvious (e.g. "echo hello").

READ BEFORE EDIT. When modifying code you have not seen, read the
file (or the relevant section) first to understand the surrounding
context. Avoid blind edits.

ONE THING AT A TIME. Prefer many small focused tool calls over one
giant compound command. Each call should make incremental, verifiable
progress. Composability beats cleverness.

VERIFY AFTER WRITE. After editing a file, read it back (or run a
test, build, or compile command) to confirm the change took effect
the way you intended.

DON'T REPEAT YOURSELF. If a tool returns the same result twice, the
strategy is wrong — not the inputs. Change approach: try a different
command, read a different file, ask a clarifying question, or
acknowledge the limitation.

# Output style

Final answers go in plain text with NO tool calls — that signals the
turn is complete. Code goes in fenced markdown blocks with language
tags. File paths use forward slashes (src/foo/bar.go) even on
Windows. When citing line numbers, use the format path:line (e.g.
main.go:42). Prefer concrete examples over abstract description.

When uncertain about user intent, ask ONE clarifying question before
launching a multi-step task — a 5-second clarification beats a
5-minute wrong-direction execution.

# Error handling

When a tool call fails (you'll see an [ERROR] prefix or non-zero exit
code), read the error carefully before retrying. Do not retry the
exact same call expecting different results — that triggers the
loop-detector and your turn will be halted automatically.

If a file path doesn't exist, list its parent directory before
guessing alternatives. If a bash command times out, it is probably
interactive or hung; try a non-blocking variant (add --no-pager,
redirect stdin from /dev/null, or use a timeout flag inside the
command).

# Stop conditions

End the turn — respond with plain text and no tool calls — when:
1. You have a complete answer to the user's question.
2. You finished all parts of a multi-part request (and summarized).
3. You hit a hard blocker (missing info, ambiguous request, an error
   you cannot work around) — explain the blocker and ask for help
   rather than guessing.

Be concise. Be direct. Be useful. Do not narrate your reasoning when
the actions speak for themselves.`

// SchemasFor converts every tool in the registry into the wire-format
// ToolSchemas the provider request needs. The loop calls this once per
// iteration; cheap because the registry is in-process and tools are
// stateless. Names are returned sorted (Registry.Names already does
// this) so the resulting Tools array is byte-stable across turns —
// important for provider-side prompt caching (T3.5).
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
//
// NOTE on history size: this function intentionally keeps the RAW tool
// output. Large outputs are NOT condensed here — doing so would deprive
// the model of detail it might need on the very next turn (e.g. read
// a file and answer questions about its contents). The summarize
// package exists for a future history compactor (T3.5) that will swap
// raw → summary for OLD tool messages only when the whole conversation
// approaches the context-window limit. See internal/agents/summarize.
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
