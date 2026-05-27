package openai

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// unmarshalJSON parses bytes back into a generic map so test assertions
// can be value-based, not byte-exact (more robust against JSON-encoder
// ordering and whitespace).
func unmarshalJSON(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("invalid JSON: %v\nraw: %s", err, b)
	}
	return m
}

func TestBuildRequest(t *testing.T) {
	cases := []struct {
		name string
		req  provider.Request
		want map[string]any
	}{
		{
			name: "user message only",
			req: provider.Request{
				Model: "gpt-4o",
				Messages: []provider.Message{
					{Role: "user", Content: []provider.ContentBlock{
						{Kind: provider.ContentText, Text: "hello"},
					}},
				},
			},
			want: map[string]any{
				"model": "gpt-4o",
				"messages": []any{
					map[string]any{"role": "user", "content": "hello"},
				},
			},
		},
		{
			name: "system + user",
			req: provider.Request{
				Model: "gpt-4o",
				Messages: []provider.Message{
					{Role: "system", Content: []provider.ContentBlock{
						{Kind: provider.ContentText, Text: "you are helpful"},
					}},
					{Role: "user", Content: []provider.ContentBlock{
						{Kind: provider.ContentText, Text: "hi"},
					}},
				},
			},
			want: map[string]any{
				"model": "gpt-4o",
				"messages": []any{
					map[string]any{"role": "system", "content": "you are helpful"},
					map[string]any{"role": "user", "content": "hi"},
				},
			},
		},
		{
			name: "assistant with tool_calls — content null",
			req: provider.Request{
				Model: "gpt-4o",
				Messages: []provider.Message{
					{Role: "assistant",
						ToolCalls: []provider.ToolCall{
							{
								ID:        "call_1",
								Name:      "read_file",
								Arguments: json.RawMessage(`{"path":"foo.txt"}`),
							},
						}},
				},
			},
			want: map[string]any{
				"model": "gpt-4o",
				"messages": []any{
					map[string]any{
						"role":    "assistant",
						"content": nil,
						"tool_calls": []any{
							map[string]any{
								"id":   "call_1",
								"type": "function",
								"function": map[string]any{
									"name":      "read_file",
									"arguments": `{"path":"foo.txt"}`,
								},
							},
						},
					},
				},
			},
		},
		{
			name: "assistant with both content and tool_calls",
			req: provider.Request{
				Model: "gpt-4o",
				Messages: []provider.Message{
					{Role: "assistant",
						Content: []provider.ContentBlock{
							{Kind: provider.ContentText, Text: "Let me check the file."},
						},
						ToolCalls: []provider.ToolCall{
							{ID: "call_1", Name: "read_file", Arguments: json.RawMessage(`{"path":"x"}`)},
						}},
				},
			},
			want: map[string]any{
				"model": "gpt-4o",
				"messages": []any{
					map[string]any{
						"role":    "assistant",
						"content": "Let me check the file.",
						"tool_calls": []any{
							map[string]any{
								"id":   "call_1",
								"type": "function",
								"function": map[string]any{
									"name":      "read_file",
									"arguments": `{"path":"x"}`,
								},
							},
						},
					},
				},
			},
		},
		{
			name: "tool result message",
			req: provider.Request{
				Model: "gpt-4o",
				Messages: []provider.Message{
					{Role: "tool", ToolCallID: "call_1",
						Content: []provider.ContentBlock{
							{Kind: provider.ContentText, Text: "file contents..."},
						}},
				},
			},
			want: map[string]any{
				"model": "gpt-4o",
				"messages": []any{
					map[string]any{
						"role":         "tool",
						"tool_call_id": "call_1",
						"content":      "file contents...",
					},
				},
			},
		},
		{
			name: "request with tools array",
			req: provider.Request{
				Model: "gpt-4o",
				Messages: []provider.Message{
					{Role: "user", Content: []provider.ContentBlock{
						{Kind: provider.ContentText, Text: "go"},
					}},
				},
				Tools: []provider.ToolSchema{
					{Name: "read_file", Description: "Reads a file.",
						Parameters: json.RawMessage(
							`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`,
						)},
				},
			},
			want: map[string]any{
				"model": "gpt-4o",
				"messages": []any{
					map[string]any{"role": "user", "content": "go"},
				},
				"tools": []any{
					map[string]any{
						"type": "function",
						"function": map[string]any{
							"name":        "read_file",
							"description": "Reads a file.",
							"parameters": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"path": map[string]any{"type": "string"},
								},
								"required": []any{"path"},
							},
						},
					},
				},
			},
		},
		{
			name: "multi-block text content joins",
			req: provider.Request{
				Model: "gpt-4o",
				Messages: []provider.Message{
					{Role: "user", Content: []provider.ContentBlock{
						{Kind: provider.ContentText, Text: "hello "},
						{Kind: provider.ContentText, Text: "world"},
					}},
				},
			},
			want: map[string]any{
				"model": "gpt-4o",
				"messages": []any{
					map[string]any{"role": "user", "content": "hello world"},
				},
			},
		},
		{
			name: "empty tool parameters default to object schema",
			req: provider.Request{
				Model: "gpt-4o",
				Messages: []provider.Message{
					{Role: "user", Content: []provider.ContentBlock{
						{Kind: provider.ContentText, Text: "go"},
					}},
				},
				Tools: []provider.ToolSchema{
					{Name: "noop", Description: "Does nothing."},
				},
			},
			want: map[string]any{
				"model": "gpt-4o",
				"messages": []any{
					map[string]any{"role": "user", "content": "go"},
				},
				"tools": []any{
					map[string]any{
						"type": "function",
						"function": map[string]any{
							"name":        "noop",
							"description": "Does nothing.",
							"parameters":  map[string]any{"type": "object"},
						},
					},
				},
			},
		},
		{
			name: "tool call with empty arguments defaults to {}",
			req: provider.Request{
				Model: "gpt-4o",
				Messages: []provider.Message{
					{Role: "assistant",
						ToolCalls: []provider.ToolCall{
							{ID: "call_x", Name: "noop"},
						}},
				},
			},
			want: map[string]any{
				"model": "gpt-4o",
				"messages": []any{
					map[string]any{
						"role":    "assistant",
						"content": nil,
						"tool_calls": []any{
							map[string]any{
								"id":   "call_x",
								"type": "function",
								"function": map[string]any{
									"name":      "noop",
									"arguments": "{}",
								},
							},
						},
					},
				},
			},
		},
	}

	a := New()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := a.BuildRequest(tc.req)
			if err != nil {
				t.Fatalf("BuildRequest: %v", err)
			}
			got := unmarshalJSON(t, b)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("payload mismatch\n  got:  %#v\n  want: %#v\n  raw:  %s", got, tc.want, b)
			}
		})
	}
}

func TestAdapterName(t *testing.T) {
	if got := New().Name(); got != "openai" {
		t.Errorf("Name() = %q, want %q", got, "openai")
	}
}

func TestParseResponse(t *testing.T) {
	cases := []struct {
		name string
		body string
		want provider.Response
	}{
		{
			name: "simple text response",
			body: `{
				"id": "chatcmpl-1",
				"object": "chat.completion",
				"model": "gpt-4o",
				"choices": [{
					"index": 0,
					"message": {"role": "assistant", "content": "Hello!"},
					"finish_reason": "stop"
				}],
				"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
			}`,
			want: provider.Response{
				Content:      "Hello!",
				FinishReason: "stop",
				Usage:        provider.Usage{PromptTokens: 10, CompletionTokens: 5},
			},
		},
		{
			name: "tool_calls with null content",
			body: `{
				"id": "chatcmpl-2",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": null,
						"tool_calls": [
							{
								"id": "call_1",
								"type": "function",
								"function": {
									"name": "read_file",
									"arguments": "{\"path\":\"foo.txt\"}"
								}
							}
						]
					},
					"finish_reason": "tool_calls"
				}],
				"usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
			}`,
			want: provider.Response{
				Content: "",
				ToolCalls: []provider.ToolCall{
					{
						ID:        "call_1",
						Name:      "read_file",
						Arguments: json.RawMessage(`{"path":"foo.txt"}`),
					},
				},
				FinishReason: "tool_calls",
				Usage:        provider.Usage{PromptTokens: 50, CompletionTokens: 20},
			},
		},
		{
			name: "multiple tool_calls",
			body: `{
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": null,
						"tool_calls": [
							{"id": "a", "type": "function",
							 "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}},
							{"id": "b", "type": "function",
							 "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}}
						]
					},
					"finish_reason": "tool_calls"
				}],
				"usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}
			}`,
			want: provider.Response{
				ToolCalls: []provider.ToolCall{
					{ID: "a", Name: "read_file", Arguments: json.RawMessage(`{"path":"a"}`)},
					{ID: "b", Name: "bash", Arguments: json.RawMessage(`{"cmd":"ls"}`)},
				},
				FinishReason: "tool_calls",
				Usage:        provider.Usage{PromptTokens: 100, CompletionTokens: 30},
			},
		},
		{
			name: "cached_tokens reported",
			body: `{
				"choices": [{
					"index": 0,
					"message": {"role": "assistant", "content": "ok"},
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 1200,
					"completion_tokens": 50,
					"total_tokens": 1250,
					"prompt_tokens_details": {"cached_tokens": 1024}
				}
			}`,
			want: provider.Response{
				Content:      "ok",
				FinishReason: "stop",
				Usage: provider.Usage{
					PromptTokens:     1200,
					CompletionTokens: 50,
					CachedTokens:     1024,
				},
			},
		},
		{
			name: "length finish_reason (truncation)",
			body: `{
				"choices": [{
					"index": 0,
					"message": {"role": "assistant", "content": "partial..."},
					"finish_reason": "length"
				}],
				"usage": {"prompt_tokens": 8000, "completion_tokens": 4096, "total_tokens": 12096}
			}`,
			want: provider.Response{
				Content:      "partial...",
				FinishReason: "length",
				Usage:        provider.Usage{PromptTokens: 8000, CompletionTokens: 4096},
			},
		},
	}

	a := New()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := a.ParseResponse([]byte(tc.body))
			if err != nil {
				t.Fatalf("ParseResponse: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("response mismatch\n  got:  %#v\n  want: %#v", got, tc.want)
			}
		})
	}
}

func TestParseResponseErrors(t *testing.T) {
	a := New()

	t.Run("empty choices returns ErrEmptyChoices", func(t *testing.T) {
		body := `{"choices": [], "usage": {}}`
		_, err := a.ParseResponse([]byte(body))
		if !errors.Is(err, ErrEmptyChoices) {
			t.Errorf("err = %v, want ErrEmptyChoices", err)
		}
	})

	t.Run("malformed JSON wraps unmarshal error", func(t *testing.T) {
		body := `{not valid json`
		_, err := a.ParseResponse([]byte(body))
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		// Should still be a json error somewhere in the chain.
		var syntaxErr *json.SyntaxError
		if !errors.As(err, &syntaxErr) {
			t.Errorf("err type = %T, want wrapping *json.SyntaxError", err)
		}
	})
}

// TestRoundTrip exercises BuildRequest + ParseResponse together against
// a canned end-to-end fixture, proving the two halves of the adapter
// stay in sync.
func TestRoundTrip(t *testing.T) {
	a := New()

	// Build a request — assert it serializes without error.
	req := provider.Request{
		Model: "gpt-4o",
		Messages: []provider.Message{
			{Role: "user", Content: []provider.ContentBlock{
				{Kind: provider.ContentText, Text: "hello"},
			}},
		},
	}
	if _, err := a.BuildRequest(req); err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}

	// Parse a canned response that uses the SAME model name + a tool
	// call whose arguments are JSON we can round-trip parse.
	body := `{
		"id": "chatcmpl-rt",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"tool_calls": [{
					"id": "call_rt",
					"type": "function",
					"function": {"name": "read_file", "arguments": "{\"path\":\"x\"}"}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
	}`
	resp, err := a.ParseResponse([]byte(body))
	if err != nil {
		t.Fatalf("ParseResponse: %v", err)
	}

	// Tool-call arguments should be a valid json.RawMessage that we
	// can json.Unmarshal into a typed struct downstream.
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(resp.ToolCalls))
	}
	var args struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(resp.ToolCalls[0].Arguments, &args); err != nil {
		t.Fatalf("tool arguments don't round-trip as JSON: %v", err)
	}
	if args.Path != "x" {
		t.Errorf("args.Path = %q, want %q", args.Path, "x")
	}
}

func TestChatCompletionsURL(t *testing.T) {
	cases := []struct {
		name string
		a    Adapter
		want string
	}{
		{"default constructor", New(), "https://api.openai.com/v1/chat/completions"},
		{"empty BaseURL falls back to default", Adapter{}, "https://api.openai.com/v1/chat/completions"},
		{"custom proxy BaseURL", Adapter{BaseURL: "https://example.com/proxy"}, "https://example.com/proxy/chat/completions"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.a.ChatCompletionsURL(); got != tc.want {
				t.Errorf("ChatCompletionsURL() = %q, want %q", got, tc.want)
			}
		})
	}
}
