package anthropic

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// unmarshalRequest is a test helper that JSON-decodes a request body
// into a generic map. Returns t.Fatal on parse error so individual
// tests can assume the result is usable.
func unmarshalRequest(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("payload not valid JSON: %v\n%s", err, body)
	}
	return out
}

func TestBuildRequest_ExtractsSystemMessage(t *testing.T) {
	a := New()
	body, err := a.BuildRequest(provider.Request{
		Model: "claude-3-5-sonnet",
		Messages: []provider.Message{
			{Role: "system", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "You are a helpful assistant."},
			}},
			{Role: "user", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "hello"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}

	p := unmarshalRequest(t, body)

	// System field at top level — an array (so we can carry cache_control).
	sys, ok := p["system"].([]any)
	if !ok {
		t.Fatalf(`payload["system"] = %T, want []any`, p["system"])
	}
	if len(sys) != 1 {
		t.Fatalf("system len = %d, want 1", len(sys))
	}
	sysBlock := sys[0].(map[string]any)
	if sysBlock["type"] != "text" || sysBlock["text"] != "You are a helpful assistant." {
		t.Errorf("system block = %+v", sysBlock)
	}
	cc, ok := sysBlock["cache_control"].(map[string]any)
	if !ok {
		t.Fatalf("cache_control missing on system block: %+v", sysBlock)
	}
	if cc["type"] != "ephemeral" {
		t.Errorf("cache_control.type = %v, want ephemeral", cc["type"])
	}

	// Messages array should NOT contain the system message.
	msgs := p["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages len = %d, want 1 (user only — system extracted)", len(msgs))
	}
	if msgs[0].(map[string]any)["role"] != "user" {
		t.Errorf("messages[0].role = %v, want user", msgs[0].(map[string]any)["role"])
	}
}

func TestBuildRequest_NoSystemMessage_OmitsField(t *testing.T) {
	a := New()
	body, _ := a.BuildRequest(provider.Request{
		Model: "claude-3-5-sonnet",
		Messages: []provider.Message{
			{Role: "user", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "hi"},
			}},
		},
	})
	p := unmarshalRequest(t, body)
	if _, ok := p["system"]; ok {
		t.Errorf("system field should be omitted when no system message")
	}
}

func TestBuildRequest_MaxTokensDefault(t *testing.T) {
	a := New()
	body, _ := a.BuildRequest(provider.Request{Model: "claude"})
	p := unmarshalRequest(t, body)
	if mt, _ := p["max_tokens"].(float64); int(mt) != DefaultMaxTokens {
		t.Errorf("max_tokens = %v, want %d", p["max_tokens"], DefaultMaxTokens)
	}
}

func TestBuildRequest_MaxTokensCustomAdapter(t *testing.T) {
	a := New()
	a.MaxTokens = 8192
	body, _ := a.BuildRequest(provider.Request{Model: "claude"})
	p := unmarshalRequest(t, body)
	if mt, _ := p["max_tokens"].(float64); int(mt) != 8192 {
		t.Errorf("max_tokens = %v, want 8192", p["max_tokens"])
	}
}

func TestBuildRequest_ToolResultBecomesUserMessage(t *testing.T) {
	a := New()
	body, _ := a.BuildRequest(provider.Request{
		Model: "claude",
		Messages: []provider.Message{
			{Role: "user", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "do X"},
			}},
			{Role: "assistant", ToolCalls: []provider.ToolCall{{
				ID:        "toolu_1",
				Name:      "read_file",
				Arguments: json.RawMessage(`{"path":"main.go"}`),
			}}},
			{
				Role:       "tool",
				ToolCallID: "toolu_1",
				Name:       "read_file",
				Content: []provider.ContentBlock{
					{Kind: provider.ContentText, Text: "package main"},
				},
			},
		},
	})
	p := unmarshalRequest(t, body)

	msgs := p["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages len = %d, want 3 (user, assistant, synthesized user)", len(msgs))
	}

	// Third message: synthesized user with tool_result content block.
	third := msgs[2].(map[string]any)
	if third["role"] != "user" {
		t.Errorf("messages[2].role = %v, want user", third["role"])
	}
	content := third["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("messages[2].content len = %d, want 1", len(content))
	}
	tr := content[0].(map[string]any)
	if tr["type"] != "tool_result" {
		t.Errorf("content[0].type = %v, want tool_result", tr["type"])
	}
	if tr["tool_use_id"] != "toolu_1" {
		t.Errorf("tool_use_id = %v, want toolu_1", tr["tool_use_id"])
	}
	if tr["content"] != "package main" {
		t.Errorf("tool_result.content = %v, want %q", tr["content"], "package main")
	}
}

func TestBuildRequest_ConsecutiveToolResultsCoalesce(t *testing.T) {
	a := New()
	body, _ := a.BuildRequest(provider.Request{
		Model: "claude",
		Messages: []provider.Message{
			{Role: "assistant", ToolCalls: []provider.ToolCall{
				{ID: "toolu_1", Name: "f1"},
				{ID: "toolu_2", Name: "f2"},
			}},
			// Two tool results back-to-back must coalesce into one user message.
			{Role: "tool", ToolCallID: "toolu_1", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "result one"},
			}},
			{Role: "tool", ToolCallID: "toolu_2", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "result two"},
			}},
		},
	})
	p := unmarshalRequest(t, body)
	msgs := p["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages len = %d, want 2 (assistant + coalesced user)", len(msgs))
	}
	second := msgs[1].(map[string]any)
	if second["role"] != "user" {
		t.Errorf("messages[1].role = %v, want user", second["role"])
	}
	content := second["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("coalesced content len = %d, want 2", len(content))
	}
	if content[0].(map[string]any)["tool_use_id"] != "toolu_1" {
		t.Errorf("first tool_result id = %v", content[0].(map[string]any)["tool_use_id"])
	}
	if content[1].(map[string]any)["tool_use_id"] != "toolu_2" {
		t.Errorf("second tool_result id = %v", content[1].(map[string]any)["tool_use_id"])
	}
}

func TestBuildRequest_AssistantToolCallsBecomeContentBlocks(t *testing.T) {
	a := New()
	body, _ := a.BuildRequest(provider.Request{
		Model: "claude",
		Messages: []provider.Message{
			{Role: "user", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "hi"},
			}},
			{
				Role:    "assistant",
				Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "I'll help."}},
				ToolCalls: []provider.ToolCall{{
					ID:        "toolu_abc",
					Name:      "read_file",
					Arguments: json.RawMessage(`{"path":"x.go"}`),
				}},
			},
		},
	})
	p := unmarshalRequest(t, body)
	msgs := p["messages"].([]any)
	assistant := msgs[1].(map[string]any)
	if assistant["role"] != "assistant" {
		t.Fatalf("messages[1].role = %v, want assistant", assistant["role"])
	}
	blocks := assistant["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("assistant content len = %d, want 2 (text + tool_use)", len(blocks))
	}
	if blocks[0].(map[string]any)["type"] != "text" {
		t.Errorf("blocks[0].type = %v, want text", blocks[0].(map[string]any)["type"])
	}
	tu := blocks[1].(map[string]any)
	if tu["type"] != "tool_use" {
		t.Errorf("blocks[1].type = %v, want tool_use", tu["type"])
	}
	if tu["id"] != "toolu_abc" || tu["name"] != "read_file" {
		t.Errorf("tool_use id/name = %v/%v", tu["id"], tu["name"])
	}
	// "input" must be a JSON object, not a JSON-encoded string.
	input, ok := tu["input"].(map[string]any)
	if !ok {
		t.Fatalf("tool_use.input = %T, want object", tu["input"])
	}
	if input["path"] != "x.go" {
		t.Errorf("input.path = %v, want x.go", input["path"])
	}
}

func TestBuildRequest_ToolSchemaUsesInputSchema(t *testing.T) {
	a := New()
	body, _ := a.BuildRequest(provider.Request{
		Model: "claude",
		Tools: []provider.ToolSchema{{
			Name:        "read_file",
			Description: "Read a file",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
		}},
	})
	p := unmarshalRequest(t, body)
	tools, _ := p["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(tools))
	}
	tool := tools[0].(map[string]any)
	if tool["name"] != "read_file" {
		t.Errorf("name = %v", tool["name"])
	}
	// Anthropic field is "input_schema", not "parameters".
	if _, ok := tool["input_schema"]; !ok {
		t.Errorf("tool missing input_schema; got: %+v", tool)
	}
	if _, ok := tool["parameters"]; ok {
		t.Errorf("tool should not carry openai-style 'parameters'; got: %+v", tool)
	}
}

func TestBuildRequest_ToolSchemaEmptyParametersGetsDefault(t *testing.T) {
	a := New()
	body, _ := a.BuildRequest(provider.Request{
		Model: "claude",
		Tools: []provider.ToolSchema{{Name: "ping", Description: ""}},
	})
	p := unmarshalRequest(t, body)
	tool := p["tools"].([]any)[0].(map[string]any)
	schema, ok := tool["input_schema"].(map[string]any)
	if !ok {
		t.Fatalf("input_schema = %T, want object", tool["input_schema"])
	}
	if schema["type"] != "object" {
		t.Errorf("default input_schema = %+v, want {type:object}", schema)
	}
}

func TestBuildRequest_UserEmptyContentEmitsEmptyTextBlock(t *testing.T) {
	a := New()
	body, _ := a.BuildRequest(provider.Request{
		Model: "claude",
		Messages: []provider.Message{
			{Role: "user", Content: nil}, // pathological — defensive shape
		},
	})
	p := unmarshalRequest(t, body)
	msgs := p["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("content len = %d, want 1 floor block", len(content))
	}
	if content[0].(map[string]any)["text"] != "" {
		t.Errorf("floor block text = %v, want empty", content[0].(map[string]any)["text"])
	}
}

func TestParseResponse_TextOnly(t *testing.T) {
	body := []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5",
		"stop_reason":"end_turn",
		"content":[{"type":"text","text":"Hello world"}],
		"usage":{"input_tokens":10,"output_tokens":3}
	}`)
	resp, err := New().ParseResponse(body)
	if err != nil {
		t.Fatalf("ParseResponse: %v", err)
	}
	if resp.Content != "Hello world" {
		t.Errorf("Content = %q, want %q", resp.Content, "Hello world")
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop (mapped from end_turn)", resp.FinishReason)
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 3 {
		t.Errorf("Usage = %+v, want {Prompt:10 Completion:3}", resp.Usage)
	}
}

func TestParseResponse_ToolUse(t *testing.T) {
	body := []byte(`{
		"id":"msg_2","type":"message","role":"assistant","model":"claude-3-5",
		"stop_reason":"tool_use",
		"content":[
			{"type":"text","text":"I'll read it."},
			{"type":"tool_use","id":"toolu_xyz","name":"read_file","input":{"path":"x.go"}}
		],
		"usage":{"input_tokens":20,"output_tokens":15}
	}`)
	resp, err := New().ParseResponse(body)
	if err != nil {
		t.Fatalf("ParseResponse: %v", err)
	}
	if resp.Content != "I'll read it." {
		t.Errorf("Content = %q", resp.Content)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "toolu_xyz" || tc.Name != "read_file" {
		t.Errorf("ToolCall id/name = %q/%q", tc.ID, tc.Name)
	}
	// Arguments should round-trip — same path field preserved.
	var args map[string]string
	if err := json.Unmarshal(tc.Arguments, &args); err != nil {
		t.Fatalf("tool args not JSON: %v\n%s", err, tc.Arguments)
	}
	if args["path"] != "x.go" {
		t.Errorf("args.path = %q, want x.go", args["path"])
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want tool_calls", resp.FinishReason)
	}
}

func TestParseResponse_UsageWithCachedTokens(t *testing.T) {
	body := []byte(`{
		"content":[{"type":"text","text":"ok"}],
		"stop_reason":"end_turn",
		"usage":{
			"input_tokens":100,
			"output_tokens":10,
			"cache_read_input_tokens":40,
			"cache_creation_input_tokens":5
		}
	}`)
	resp, _ := New().ParseResponse(body)
	// PromptTokens carries the total — input + cache_read — so the cost
	// tracker sees the full input volume.
	if resp.Usage.PromptTokens != 140 {
		t.Errorf("PromptTokens = %d, want 140 (input_tokens + cache_read_input_tokens)", resp.Usage.PromptTokens)
	}
	if resp.Usage.CachedTokens != 40 {
		t.Errorf("CachedTokens = %d, want 40", resp.Usage.CachedTokens)
	}
	if resp.Usage.CompletionTokens != 10 {
		t.Errorf("CompletionTokens = %d, want 10", resp.Usage.CompletionTokens)
	}
}

func TestParseResponse_EmptyContentReturnsSentinel(t *testing.T) {
	body := []byte(`{"content":[],"stop_reason":"end_turn"}`)
	_, err := New().ParseResponse(body)
	if !errors.Is(err, ErrEmptyContent) {
		t.Errorf("err = %v, want ErrEmptyContent", err)
	}
}

func TestParseResponse_MalformedJSONReturnsError(t *testing.T) {
	_, err := New().ParseResponse([]byte(`{not json`))
	if err == nil {
		t.Fatal("expected parse error, got nil")
	}
	if !strings.Contains(err.Error(), "anthropic") {
		t.Errorf("error message %q should mention provider name", err)
	}
}

func TestMapStopReason(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"end_turn", "stop"},
		{"stop_sequence", "stop"},
		{"pause_turn", "stop"},
		{"max_tokens", "length"},
		{"tool_use", "tool_calls"},
		{"", ""},
		{"future_reason", "future_reason"}, // pass-through for unknowns
	}
	for _, c := range cases {
		if got := mapStopReason(c.in); got != c.want {
			t.Errorf("mapStopReason(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestAdapter_MessagesURL_FallbackOnEmptyBase(t *testing.T) {
	a := Adapter{} // zero value
	if got := a.MessagesURL(); got != DefaultBaseURL+"/messages" {
		t.Errorf("MessagesURL = %q, want default-derived URL", got)
	}
}

func TestAdapter_APIVersionHeader_FallbackOnEmpty(t *testing.T) {
	a := Adapter{}
	if got := a.APIVersionHeader(); got != DefaultAPIVersion {
		t.Errorf("APIVersionHeader = %q, want %q", got, DefaultAPIVersion)
	}
}

func TestAdapter_Name(t *testing.T) {
	if name := New().Name(); name != "anthropic" {
		t.Errorf("Name = %q, want anthropic", name)
	}
}

func TestBuildRequest_MultipleSystemMessagesJoinWithDoubleNewline(t *testing.T) {
	a := New()
	body, _ := a.BuildRequest(provider.Request{
		Model: "claude",
		Messages: []provider.Message{
			{Role: "system", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "rule 1"},
			}},
			{Role: "system", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "rule 2"},
			}},
			{Role: "user", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "go"},
			}},
		},
	})
	p := unmarshalRequest(t, body)
	sys := p["system"].([]any)
	if len(sys) != 1 {
		t.Fatalf("system len = %d, want 1 (multiple system messages joined)", len(sys))
	}
	if sys[0].(map[string]any)["text"] != "rule 1\n\nrule 2" {
		t.Errorf("joined system text = %v", sys[0].(map[string]any)["text"])
	}
}
