package anthropic

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
// Stream. Same value as openai's (8) for cross-provider consistency —
// small enough to back-pressure a stalled consumer, large enough to
// absorb brief render pauses without blocking the SSE-reading
// goroutine.
const streamChannelBuffer = 8

// streamScannerMaxBytes caps a single SSE line. Anthropic chunks tend
// to be smaller than OpenAI's (one delta per event vs OpenAI's
// occasional chunk-with-everything), but a tool_use block's
// input_json_delta can carry a multi-KB partial_json on long argument
// payloads. 1 MiB matches openai's ceiling.
const streamScannerMaxBytes = 1 << 20

// Stream implements provider.Provider. It POSTs a streaming Messages
// request and returns a channel that receives StreamEvents as the
// server emits them.
//
// Setup vs in-stream error split matches the Provider contract: setup
// failures (build, transport, non-2xx) return (nil, err); mid-stream
// failures arrive on the channel as StreamEventError followed by
// close. The producer goroutine guarantees exactly-once close via
// defer.
//
// ctx flows through http.NewRequestWithContext so cancellation aborts
// the network read and terminates the goroutine. Consumers should
// drain the channel until close even when cancelling, or the
// goroutine leaks.
func (c *Client) Stream(ctx context.Context, req provider.Request) (<-chan provider.StreamEvent, error) {
	body, err := c.Adapter.BuildStreamRequest(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: build stream request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		c.Adapter.MessagesURL(),
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("anthropic: new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.APIKey)
	httpReq.Header.Set("anthropic-version", c.Adapter.APIVersionHeader())
	httpReq.Header.Set("Accept", "text/event-stream")

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: HTTP: %w", err)
	}

	// Non-2xx is a setup failure: read the full body so we can surface
	// it via HTTPError. Mirrors Call's behavior so callers handle both
	// paths identically.
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

// runStreamReader is the producer goroutine. Owns the HTTP body and
// the channel; closes both via defer so every exit path honors the
// close contract (including panics).
//
// Anthropic frames events as pairs of lines:
//
//	event: <event-type>
//	data: <json>
//	<blank>
//
// We ignore the "event:" line — the JSON payload always carries a
// "type" field that matches it, and switching off the inner type is
// simpler than threading both pieces of state. Blank lines are SSE
// framing; "data:" lines are the payload.
func runStreamReader(body io.ReadCloser, ch chan<- provider.StreamEvent) {
	defer close(ch)
	defer body.Close()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), streamScannerMaxBytes)

	state := newStreamState()

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			// SSE framing: blank separators, "event:" type prefixes
			// (we read off the inner type instead), comments. Drop.
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if err := state.handleEvent(payload, ch); err != nil {
			ch <- provider.NewError(err)
			return
		}
		if state.finished {
			return
		}
	}

	// Scanner ended without message_stop. Either ctx cancellation
	// aborted the read, the server hung up, or the body closed
	// cleanly. Treat the first two as failures and the third as a
	// degenerate success.
	if err := scanner.Err(); err != nil {
		ch <- provider.NewError(fmt.Errorf("anthropic: stream read: %w", err))
		return
	}
	state.finalize(ch)
}

// blockKind tags the content-block type so deltas can be dispatched
// correctly. Anthropic streams the kind in content_block_start; we
// remember it and route subsequent delta events accordingly.
type blockKind int

const (
	blockUnknown blockKind = iota
	blockText
	blockToolUse
	blockThinking
)

// blockState is one in-flight content block. Tracks the kind (decided
// at content_block_start), tool_use identity (when applicable), and
// the accumulated input_json fragments for tool_use blocks.
type blockState struct {
	kind blockKind

	// id/name populated for blockToolUse, captured from
	// content_block_start.
	id   string
	name string

	// argsBuf accumulates input_json_delta fragments for tool_use
	// blocks. Empty for other block kinds.
	argsBuf strings.Builder
}

// streamState holds the per-stream accumulators. Text content from
// every text-kind block concatenates into contentBuf; tool_use blocks
// are indexed by the block index and assembled separately; usage is
// updated across message_start (input) + message_delta (output).
type streamState struct {
	blocks       map[int]*blockState
	contentBuf   strings.Builder
	finishReason string
	usage        provider.Usage

	// finished is set when message_stop arrives. runStreamReader
	// checks this after each event and returns when true so the
	// finalize/Done sequence isn't followed by stray events.
	finished bool
}

func newStreamState() *streamState {
	return &streamState{blocks: make(map[int]*blockState)}
}

// sseEvent is the umbrella shape we unmarshal every Anthropic event
// payload into. Pointer fields where a missing field is meaningfully
// distinct from a zero value (usage may be on message_start only,
// content_block may carry a tool_use's id/name only, etc.).
type sseEvent struct {
	Type string `json:"type"`

	// content_block_start
	Index        int           `json:"index"`
	ContentBlock *contentBlock `json:"content_block"`

	// content_block_delta
	Delta *eventDelta `json:"delta"`

	// message_start
	Message *messageStart `json:"message"`

	// message_delta carries usage at the top level too.
	Usage *usageBlock `json:"usage"`

	// error event carries the error envelope under "error".
	Error *errorBlock `json:"error"`
}

type contentBlock struct {
	Type string `json:"type"`
	// tool_use fields
	ID   string `json:"id"`
	Name string `json:"name"`
}

// eventDelta is the inner delta object inside content_block_delta and
// message_delta events. The Type field discriminates which fields are
// populated; we switch on it.
type eventDelta struct {
	Type string `json:"type"`

	// text_delta
	Text string `json:"text"`

	// input_json_delta (tool args fragment)
	PartialJSON string `json:"partial_json"`

	// thinking_delta
	Thinking string `json:"thinking"`

	// message_delta fields (no "type" field on the delta object itself
	// for message_delta — just stop_reason).
	StopReason string `json:"stop_reason"`
}

type messageStart struct {
	Usage usageBlock `json:"usage"`
}

type usageBlock struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

type errorBlock struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// handleEvent dispatches one SSE event payload. Returns an error only
// if the JSON is malformed; semantic problems (unknown event type,
// delta for a missing block) are tolerated silently so server-side
// additions don't break our reader.
func (s *streamState) handleEvent(payload string, ch chan<- provider.StreamEvent) error {
	var ev sseEvent
	if err := json.Unmarshal([]byte(payload), &ev); err != nil {
		return fmt.Errorf("anthropic: parse SSE event: %w", err)
	}

	switch ev.Type {
	case "message_start":
		// Lock in the input-side token counts. output_tokens here is
		// always 0 or 1 (a "this many so far" snapshot); the real
		// count arrives at message_delta.
		if ev.Message != nil {
			s.usage.PromptTokens = ev.Message.Usage.InputTokens + ev.Message.Usage.CacheReadInputTokens
			s.usage.CachedTokens = ev.Message.Usage.CacheReadInputTokens
		}

	case "content_block_start":
		bs := &blockState{}
		if ev.ContentBlock != nil {
			switch ev.ContentBlock.Type {
			case "text":
				bs.kind = blockText
			case "tool_use":
				bs.kind = blockToolUse
				bs.id = ev.ContentBlock.ID
				bs.name = ev.ContentBlock.Name
				ch <- provider.NewToolCallStart(ev.Index, bs.id, bs.name)
			case "thinking":
				bs.kind = blockThinking
			default:
				// Unknown block kind — leave as blockUnknown so later
				// deltas drop silently. Forward-compat with new
				// Anthropic block types (e.g. vision output) without
				// failing the stream.
			}
		}
		s.blocks[ev.Index] = bs

	case "content_block_delta":
		bs, ok := s.blocks[ev.Index]
		if !ok || ev.Delta == nil {
			return nil
		}
		switch ev.Delta.Type {
		case "text_delta":
			if bs.kind == blockText && ev.Delta.Text != "" {
				s.contentBuf.WriteString(ev.Delta.Text)
				ch <- provider.NewTextDelta(ev.Delta.Text)
			}
		case "input_json_delta":
			if bs.kind == blockToolUse && ev.Delta.PartialJSON != "" {
				bs.argsBuf.WriteString(ev.Delta.PartialJSON)
				ch <- provider.NewToolCallDelta(ev.Index, ev.Delta.PartialJSON)
			}
		case "thinking_delta":
			if bs.kind == blockThinking && ev.Delta.Thinking != "" {
				ch <- provider.NewReasoningDelta(ev.Delta.Thinking)
			}
		case "signature_delta":
			// Cryptographic verification token attached to thinking
			// blocks. Not user-facing; needed only if we ever
			// replay thinking on a subsequent turn. Drop silently.
		}

	case "content_block_stop":
		// No emit here. ToolCallDone events come out in index order
		// during finalize() so the sequence stays deterministic
		// across multiple blocks. The block's argsBuf is already
		// fully assembled by this point.

	case "message_delta":
		// stop_reason and the final output_tokens count land here.
		if ev.Delta != nil {
			if ev.Delta.StopReason != "" {
				s.finishReason = mapStopReason(ev.Delta.StopReason)
			}
		}
		if ev.Usage != nil {
			// message_delta.usage.output_tokens is a CUMULATIVE
			// snapshot, not an incremental delta. Last value wins.
			s.usage.CompletionTokens = ev.Usage.OutputTokens
			ch <- provider.NewUsage(s.usage)
		}

	case "message_stop":
		s.finalize(ch)
		s.finished = true

	case "ping":
		// Keep-alive heartbeat. Drop silently.

	case "error":
		// Server-side mid-stream failure.
		msg := "anthropic: stream error"
		if ev.Error != nil && ev.Error.Message != "" {
			msg = "anthropic: stream error: " + ev.Error.Message
		}
		return fmt.Errorf("%s", msg)

	default:
		// Unknown event type — Anthropic may add new ones (e.g.
		// "message_pause"). Drop silently rather than failing.
	}

	return nil
}

// finalize emits the terminal sequence: a ToolCallDone for every
// tool_use block in index order (deterministic output independent of
// arrival order), then the assembled Response wrapped in
// StreamEventDone.
//
// Called from message_stop (the canonical success path) and from
// scanner-end-without-message_stop (a degenerate but recoverable
// success — some compatible servers may omit the final event).
func (s *streamState) finalize(ch chan<- provider.StreamEvent) {
	indices := make([]int, 0, len(s.blocks))
	for i := range s.blocks {
		indices = append(indices, i)
	}
	sort.Ints(indices)

	var toolCalls []provider.ToolCall
	for _, i := range indices {
		bs := s.blocks[i]
		if bs.kind != blockToolUse {
			continue
		}
		args := bs.argsBuf.String()
		ch <- provider.NewToolCallDone(i, bs.id, bs.name, args)
		if args == "" {
			args = "{}"
		}
		toolCalls = append(toolCalls, provider.ToolCall{
			ID:        bs.id,
			Name:      bs.name,
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
