package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// streamChannelBuffer is the capacity of the event channel returned by
// Stream. Small enough to back-pressure a stalled consumer (no runaway
// memory) but large enough that brief consumer pauses (e.g. a TUI
// rendering between TextDeltas) don't block the SSE-reading goroutine
// on every event. 8 is a teaching-friendly round number; production
// could profile this.
const streamChannelBuffer = 8

// streamScannerMaxBytes is the upper bound on a single SSE line.
// Default bufio.Scanner buffer is 64 KiB, which is fine for typical
// chunks but a tool_call with a very large argument payload (e.g.
// write_file with a 200 KB body delivered in one chunk) can blow past
// it. 1 MiB is a teaching constant — explicit, easy to bump if a real
// workload demands it.
const streamScannerMaxBytes = 1 << 20

// Stream implements provider.Provider. It POSTs a streaming Chat
// Completions request and returns a channel that receives StreamEvents
// as the server emits them.
//
// Setup errors (build failure, HTTP transport failure, non-2xx status)
// return (nil, err) before any channel work. Mid-stream errors arrive
// on the channel as StreamEventError, after which the channel closes.
// The producer goroutine guarantees exactly-once close via defer.
//
// ctx flows through http.NewRequestWithContext, so cancellation aborts
// the network read; the scanner returns an error, the goroutine emits
// StreamEventError, the channel closes. Consumers should drain the
// channel until close even when cancelling, or the goroutine leaks.
func (c *Client) Stream(ctx context.Context, req provider.Request) (<-chan provider.StreamEvent, error) {
	body, err := c.Adapter.BuildStreamRequest(req)
	if err != nil {
		return nil, fmt.Errorf("openai: build stream request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		c.Adapter.ChatCompletionsURL(),
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("openai: new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: HTTP: %w", err)
	}

	// Non-2xx is a setup failure: the server rejected the request
	// outright. Read and close the body so we can return the full error
	// payload, then bail with HTTPError — same shape Call uses, so
	// callers handle both paths identically.
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(resp.Body)
		return nil, &HTTPError{
			Status: resp.StatusCode,
			Body:   string(respBody),
		}
	}

	ch := make(chan provider.StreamEvent, streamChannelBuffer)
	go runStreamReader(resp.Body, ch)
	return ch, nil
}

// runStreamReader is the producer goroutine spawned by Stream. Owns the
// HTTP response body (closes it) and the event channel (closes it),
// both via defer at the top so every exit path honors the close
// contract — including panics.
//
// The reader loop: scan one SSE line at a time, ignore framing lines
// (blanks, event:-prefixed type markers we don't use), parse "data: "
// payloads as JSON chunks, drive a streamState that emits events.
func runStreamReader(body io.ReadCloser, ch chan<- provider.StreamEvent) {
	defer close(ch)
	defer body.Close()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), streamScannerMaxBytes)

	state := newStreamState()

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			// SSE framing: blank separator lines, "event: " type
			// prefixes (Chat Completions doesn't use them), comments.
			// All ignored.
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			state.finalize(ch)
			return
		}
		if err := state.handleChunk(payload, ch); err != nil {
			ch <- provider.NewError(err)
			return
		}
	}

	// Scanner ended. Either ctx cancellation aborted the read, the
	// server hung up mid-stream, or the body closed cleanly without
	// a [DONE] marker (some OpenAI-compatible servers omit it).
	if err := scanner.Err(); err != nil {
		ch <- provider.NewError(fmt.Errorf("openai: stream read: %w", err))
		return
	}
	state.finalize(ch)
}

// streamState accumulates text content + tool_call fragments + usage
// across SSE chunks. The reassembly logic lives here so handleChunk
// stays focused on parsing one chunk at a time.
//
// Tool_calls map (not slice) because OpenAI can emit chunks for index
// 0 and index 1 interleaved, and indices can be sparse. Lookup-by-index
// is the natural fit.
type streamState struct {
	contentBuf   strings.Builder
	finishReason string
	usage        provider.Usage
	toolCalls    map[int]*toolCallState
}

// toolCallState is one in-flight tool_call. Tracks id/name (received
// in the first chunk for this index, usually), accumulated args, and
// whether we've already emitted StreamEventToolCallStart.
type toolCallState struct {
	id      string
	name    string
	args    strings.Builder
	started bool
}

func newStreamState() *streamState {
	return &streamState{toolCalls: make(map[int]*toolCallState)}
}

// chatCompletionChunk is the per-chunk shape on the SSE wire. Pointer
// fields where null is meaningfully distinct from absent — content is
// often null on tool-call chunks, finish_reason is null until the
// final chunk, usage is present only on the final usage chunk.
type chatCompletionChunk struct {
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Content   *string `json:"content"`
			Reasoning *string `json:"reasoning_content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`

	Usage *struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

// handleChunk parses one SSE data payload and emits the events it
// implies. Returns an error if the JSON is malformed; the caller
// surfaces it as StreamEventError.
//
// Order of emission matters for downstream consumers: usage first (if
// present), then per-choice text/reasoning/tool_call events in the
// order they appear in the chunk. Tool_call events specifically:
// Start (once, when id+name first available) → Delta (per fragment)
// → Done (later, in finalize).
func (s *streamState) handleChunk(payload string, ch chan<- provider.StreamEvent) error {
	var chunk chatCompletionChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		return fmt.Errorf("openai: parse SSE chunk: %w", err)
	}

	if chunk.Usage != nil {
		s.usage = provider.Usage{
			PromptTokens:     chunk.Usage.PromptTokens,
			CompletionTokens: chunk.Usage.CompletionTokens,
			CachedTokens:     chunk.Usage.PromptTokensDetails.CachedTokens,
		}
		ch <- provider.NewUsage(s.usage)
	}

	for _, choice := range chunk.Choices {
		if choice.Delta.Content != nil && *choice.Delta.Content != "" {
			s.contentBuf.WriteString(*choice.Delta.Content)
			ch <- provider.NewTextDelta(*choice.Delta.Content)
		}

		// Reasoning deltas are non-standard on OpenAI's own Chat
		// Completions API but some compatible providers (DeepSeek R1
		// via OpenAI-compat mode) emit them. Pass through verbatim.
		if choice.Delta.Reasoning != nil && *choice.Delta.Reasoning != "" {
			ch <- provider.NewReasoningDelta(*choice.Delta.Reasoning)
		}

		for _, tc := range choice.Delta.ToolCalls {
			st, ok := s.toolCalls[tc.Index]
			if !ok {
				st = &toolCallState{}
				s.toolCalls[tc.Index] = st
			}
			// id/name arrive piecewise across compatible providers.
			// First non-empty value wins; later chunks just append
			// arguments.
			if tc.ID != "" {
				st.id = tc.ID
			}
			if tc.Function.Name != "" {
				st.name = tc.Function.Name
			}
			if !st.started && st.id != "" && st.name != "" {
				st.started = true
				ch <- provider.NewToolCallStart(tc.Index, st.id, st.name)
			}
			if tc.Function.Arguments != "" {
				st.args.WriteString(tc.Function.Arguments)
				ch <- provider.NewToolCallDelta(tc.Index, tc.Function.Arguments)
			}
		}

		if choice.FinishReason != nil {
			s.finishReason = *choice.FinishReason
		}
	}

	return nil
}

// finalize emits the terminal event sequence: a ToolCallDone for each
// accumulated tool_call (in index order so output is deterministic),
// then a single StreamEventDone carrying the assembled Response.
//
// Called from runStreamReader on either the [DONE] sentinel or a
// clean EOF without one — both are success paths.
func (s *streamState) finalize(ch chan<- provider.StreamEvent) {
	indices := make([]int, 0, len(s.toolCalls))
	for i := range s.toolCalls {
		indices = append(indices, i)
	}
	sort.Ints(indices)

	var toolCalls []provider.ToolCall
	for _, i := range indices {
		st := s.toolCalls[i]
		if !st.started {
			// A chunk emitted args for an index whose id/name never
			// arrived — drop it. Treating malformed slots as fatal
			// would be over-strict; the loop sees one fewer tool_call.
			continue
		}
		args := st.args.String()
		ch <- provider.NewToolCallDone(i, st.id, st.name, args)
		if args == "" {
			args = "{}"
		}
		toolCalls = append(toolCalls, provider.ToolCall{
			ID:        st.id,
			Name:      st.name,
			Arguments: json.RawMessage(args),
		})
	}

	ch <- provider.NewDone(&provider.Response{
		Content:      s.contentBuf.String(),
		ToolCalls:    toolCalls,
		Usage:        s.usage,
		FinishReason: s.finishReason,
	})
}
