package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// sseHandler builds an httptest handler that emits a list of Anthropic
// SSE events. Each entry is a (event-type, JSON-payload) pair; the
// helper writes them as proper "event: T\ndata: J\n\n" blocks and
// flushes after each one so the client sees them streamed.
//
// Newlines inside payloads are stripped before framing (SSE splits on
// "\n") so tests can write multi-line JSON literals for readability.
func sseHandler(events ...[2]string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for _, ev := range events {
			t := ev[0]
			payload := strings.ReplaceAll(ev[1], "\n", "")
			payload = strings.ReplaceAll(payload, "\t", "")
			_, _ = io.WriteString(w, "event: "+t+"\n")
			_, _ = io.WriteString(w, "data: "+payload+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
	}
}

// drainEvents reads until the channel closes. Times out via test
// deadline if the producer never closes — that catches a close-contract
// violation as a test hang rather than silent goroutine leak.
func drainEvents(t *testing.T, ch <-chan provider.StreamEvent) []provider.StreamEvent {
	t.Helper()
	var out []provider.StreamEvent
	for ev := range ch {
		out = append(out, ev)
	}
	return out
}

func TestStream_PlainText(t *testing.T) {
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":25,"output_tokens":1}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo "}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	c, _ := newTestClient(t, sseHandler(events...))

	ch, err := c.Stream(context.Background(), provider.Request{Model: "claude"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	evs := drainEvents(t, ch)

	// Expected: 3 TextDeltas, 1 Usage, 1 Done.
	wantKinds := []provider.StreamEventKind{
		provider.StreamEventTextDelta,
		provider.StreamEventTextDelta,
		provider.StreamEventTextDelta,
		provider.StreamEventUsage,
		provider.StreamEventDone,
	}
	gotKinds := kindsOf(evs)
	if !equalKinds(gotKinds, wantKinds) {
		t.Fatalf("event kinds = %v, want %v\n%s", gotKinds, wantKinds, dumpEvents(evs))
	}
	done := evs[len(evs)-1]
	if done.Response.Content != "Hello world" {
		t.Errorf("Done.Response.Content = %q, want %q", done.Response.Content, "Hello world")
	}
	if done.Response.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop", done.Response.FinishReason)
	}
	// Usage assembled across message_start (input_tokens=25) +
	// message_delta (output_tokens=3).
	if done.Response.Usage.PromptTokens != 25 || done.Response.Usage.CompletionTokens != 3 {
		t.Errorf("Usage = %+v, want {Prompt:25 Completion:3}", done.Response.Usage)
	}
}

func TestStream_SingleToolUse(t *testing.T) {
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":50,"output_tokens":1}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_abc","name":"read_file","input":{}}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"main.go\"}"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	c, _ := newTestClient(t, sseHandler(events...))

	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	evs := drainEvents(t, ch)

	// Expected: Start, Delta, Delta, Usage, ToolCallDone, Done.
	wantKinds := []provider.StreamEventKind{
		provider.StreamEventToolCallStart,
		provider.StreamEventToolCallDelta,
		provider.StreamEventToolCallDelta,
		provider.StreamEventUsage,
		provider.StreamEventToolCallDone,
		provider.StreamEventDone,
	}
	gotKinds := kindsOf(evs)
	if !equalKinds(gotKinds, wantKinds) {
		t.Fatalf("event kinds = %v, want %v\n%s", gotKinds, wantKinds, dumpEvents(evs))
	}

	start := evs[0]
	if start.ToolCall.ID != "toolu_abc" || start.ToolCall.Name != "read_file" {
		t.Errorf("Start.ToolCall = %+v", start.ToolCall)
	}
	if evs[1].ToolCall.Arguments != `{"path":` || evs[2].ToolCall.Arguments != `"main.go"}` {
		t.Errorf("Delta args mismatch: %+v, %+v", evs[1].ToolCall, evs[2].ToolCall)
	}
	tcDone := evs[4]
	if tcDone.ToolCall.Arguments != `{"path":"main.go"}` {
		t.Errorf("ToolCallDone.Arguments = %q, want assembled JSON", tcDone.ToolCall.Arguments)
	}
	done := evs[5]
	if len(done.Response.ToolCalls) != 1 {
		t.Fatalf("Response.ToolCalls len = %d, want 1", len(done.Response.ToolCalls))
	}
	if got := string(done.Response.ToolCalls[0].Arguments); got != `{"path":"main.go"}` {
		t.Errorf("Response.ToolCalls[0].Arguments = %q", got)
	}
	if done.Response.FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want tool_calls (mapped from tool_use)", done.Response.FinishReason)
	}
}

func TestStream_MixedTextAndToolUse(t *testing.T) {
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":10}}}`},
		// Block 0: text
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I'll read it."}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		// Block 1: tool_use
		{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_x","name":"read_file"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"x.go\"}"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	c, _ := newTestClient(t, sseHandler(events...))

	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	evs := drainEvents(t, ch)
	done := evs[len(evs)-1]
	if done.Response.Content != "I'll read it." {
		t.Errorf("Content = %q", done.Response.Content)
	}
	if len(done.Response.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(done.Response.ToolCalls))
	}
	if string(done.Response.ToolCalls[0].Arguments) != `{"path":"x.go"}` {
		t.Errorf("ToolCalls[0].Arguments = %q", done.Response.ToolCalls[0].Arguments)
	}
}

func TestStream_TwoToolUseBlocks_DeterministicDoneOrder(t *testing.T) {
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":10}}}`},
		{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_a","name":"f1"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		{"content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_b","name":"f2"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"y\":2}"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":2}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	c, _ := newTestClient(t, sseHandler(events...))

	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	evs := drainEvents(t, ch)

	// ToolCallDone events should appear in index order.
	var doneIndices []int
	for _, ev := range evs {
		if ev.Kind == provider.StreamEventToolCallDone {
			doneIndices = append(doneIndices, ev.ToolCall.Index)
		}
	}
	if !equalInts(doneIndices, []int{1, 2}) {
		t.Errorf("ToolCallDone index order = %v, want [1 2]", doneIndices)
	}

	done := evs[len(evs)-1]
	if len(done.Response.ToolCalls) != 2 {
		t.Fatalf("ToolCalls len = %d, want 2", len(done.Response.ToolCalls))
	}
	if done.Response.ToolCalls[0].ID != "toolu_a" || done.Response.ToolCalls[1].ID != "toolu_b" {
		t.Errorf("ToolCalls order = [%s %s], want [toolu_a toolu_b]",
			done.Response.ToolCalls[0].ID, done.Response.ToolCalls[1].ID)
	}
}

func TestStream_ThinkingDeltaEmitsReasoning_SignatureDropped(t *testing.T) {
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":10}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think..."}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc123"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"OK"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	c, _ := newTestClient(t, sseHandler(events...))
	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	evs := drainEvents(t, ch)

	var (
		reasoningCount  int
		reasoningText   string
		hasSignatureEv  bool
	)
	for _, ev := range evs {
		if ev.Kind == provider.StreamEventReasoningDelta {
			reasoningCount++
			reasoningText = ev.Text
		}
		// signature_delta should not produce any ReasoningDelta or
		// other event with the signature text.
		if strings.Contains(ev.Text, "abc123") {
			hasSignatureEv = true
		}
	}
	if reasoningCount != 1 {
		t.Errorf("ReasoningDelta count = %d, want 1", reasoningCount)
	}
	if reasoningText != "Let me think..." {
		t.Errorf("ReasoningDelta.Text = %q", reasoningText)
	}
	if hasSignatureEv {
		t.Errorf("signature_delta leaked into stream (token 'abc123' should never appear)")
	}
	// Final text content is just "OK" — the thinking text is reasoning,
	// not user-visible content.
	done := evs[len(evs)-1]
	if done.Response.Content != "OK" {
		t.Errorf("Done.Content = %q, want %q (reasoning excluded)", done.Response.Content, "OK")
	}
}

func TestStream_PingEventsIgnored(t *testing.T) {
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":10}}}`},
		{"ping", `{"type":"ping"}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text"}}`},
		{"ping", `{"type":"ping"}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`},
		{"ping", `{"type":"ping"}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	c, _ := newTestClient(t, sseHandler(events...))
	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	evs := drainEvents(t, ch)

	for _, ev := range evs {
		if ev.Kind == provider.StreamEventError {
			t.Errorf("ping events should not produce StreamEventError: %v", ev.Err)
		}
	}
	if evs[len(evs)-1].Kind != provider.StreamEventDone {
		t.Errorf("last event = %s, want done", evs[len(evs)-1].Kind)
	}
}

func TestStream_ErrorEventEmitsStreamError(t *testing.T) {
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":10}}}`},
		{"error", `{"type":"error","error":{"type":"overloaded_error","message":"server overloaded"}}`},
	}
	c, _ := newTestClient(t, sseHandler(events...))
	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	evs := drainEvents(t, ch)

	last := evs[len(evs)-1]
	if last.Kind != provider.StreamEventError {
		t.Fatalf("last event = %s, want error\n%s", last.Kind, dumpEvents(evs))
	}
	if last.Err == nil || !strings.Contains(last.Err.Error(), "server overloaded") {
		t.Errorf("Error.Err = %v, want to mention 'server overloaded'", last.Err)
	}
	// Done must NOT appear after Error.
	for _, ev := range evs {
		if ev.Kind == provider.StreamEventDone {
			t.Errorf("unexpected Done event after Error")
		}
	}
}

func TestStream_SetupHTTP401(t *testing.T) {
	c, _ := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"authentication_error","message":"invalid api key"}}`)
	})

	ch, err := c.Stream(context.Background(), provider.Request{Model: "claude"})
	if err == nil {
		t.Fatal("expected setup error, got nil")
	}
	if ch != nil {
		t.Errorf("expected nil channel, got %v", ch)
	}
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) {
		t.Fatalf("err type = %T, want *HTTPError", err)
	}
	if httpErr.Status != http.StatusUnauthorized {
		t.Errorf("Status = %d, want 401", httpErr.Status)
	}
	if !strings.Contains(httpErr.Body, "invalid api key") {
		t.Errorf("Body should mention invalid api key: %q", httpErr.Body)
	}
}

func TestStream_MalformedJSONEmitsErrorEvent(t *testing.T) {
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":10}}}`},
		{"content_block_start", `{this is not json}`},
	}
	c, _ := newTestClient(t, sseHandler(events...))
	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	evs := drainEvents(t, ch)

	last := evs[len(evs)-1]
	if last.Kind != provider.StreamEventError {
		t.Fatalf("last event = %s, want error\n%s", last.Kind, dumpEvents(evs))
	}
	for _, ev := range evs {
		if ev.Kind == provider.StreamEventDone {
			t.Errorf("unexpected Done after malformed JSON Error")
		}
	}
}

func TestStream_ContextCancellation(t *testing.T) {
	serverDone := make(chan struct{})
	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		defer close(serverDone)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n")
		flusher.Flush()
		<-r.Context().Done()
	})

	before := runtime.NumGoroutine()

	ctx, cancel := context.WithCancel(context.Background())
	ch, err := c.Stream(ctx, provider.Request{Model: "claude"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	cancel()
	_ = drainEvents(t, ch) // drain until close

	<-serverDone

	// Give the goroutine a beat to wind down.
	for i := 0; i < 20; i++ {
		if runtime.NumGoroutine() <= before+1 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if leaked := runtime.NumGoroutine() - before; leaked > 1 {
		t.Errorf("goroutine leak: +%d after cancel", leaked)
	}
}

func TestStream_RequestBodyHasStreamFlag(t *testing.T) {
	var gotBody []byte
	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	})

	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	_ = drainEvents(t, ch)

	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("body not JSON: %v\n%s", err, gotBody)
	}
	if payload["stream"] != true {
		t.Errorf(`body["stream"] = %v, want true`, payload["stream"])
	}
}

func TestStream_AcceptHeaderSet(t *testing.T) {
	var gotAccept string
	var gotAPIKey string
	var gotVersion string
	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotAccept = r.Header.Get("Accept")
		gotAPIKey = r.Header.Get("x-api-key")
		gotVersion = r.Header.Get("anthropic-version")
		_, _ = io.WriteString(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	})
	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	_ = drainEvents(t, ch)
	if gotAccept != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", gotAccept)
	}
	if gotAPIKey != "test-key" {
		t.Errorf("x-api-key = %q, want test-key", gotAPIKey)
	}
	if gotVersion != DefaultAPIVersion {
		t.Errorf("anthropic-version = %q, want default", gotVersion)
	}
}

func TestStream_UsageMergedFromStartAndDelta(t *testing.T) {
	// PromptTokens lands at message_start, CompletionTokens at message_delta.
	// Verify both flow into the final Done.Response.Usage.
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":100,"cache_read_input_tokens":30}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	c, _ := newTestClient(t, sseHandler(events...))
	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	evs := drainEvents(t, ch)
	done := evs[len(evs)-1]
	u := done.Response.Usage
	if u.PromptTokens != 130 { // 100 + 30 cache_read
		t.Errorf("PromptTokens = %d, want 130 (input + cache_read)", u.PromptTokens)
	}
	if u.CachedTokens != 30 {
		t.Errorf("CachedTokens = %d, want 30", u.CachedTokens)
	}
	if u.CompletionTokens != 42 {
		t.Errorf("CompletionTokens = %d, want 42", u.CompletionTokens)
	}
}

func TestStream_NoMessageStopFallback(t *testing.T) {
	// Scanner ends cleanly without a message_stop event — some
	// compatible servers may behave this way. We treat it as a
	// degenerate success: finalize the accumulated state and emit
	// Done anyway.
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":10}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`},
		// No content_block_stop, no message_delta, no message_stop.
	}
	c, _ := newTestClient(t, sseHandler(events...))
	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	evs := drainEvents(t, ch)
	last := evs[len(evs)-1]
	if last.Kind != provider.StreamEventDone {
		t.Errorf("last event = %s, want done\n%s", last.Kind, dumpEvents(evs))
	}
	if last.Response.Content != "hi" {
		t.Errorf("Content = %q", last.Response.Content)
	}
}

func TestStream_UnknownEventTypeIgnored(t *testing.T) {
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":1}}}`},
		// Hypothetical future event type — should be dropped.
		{"future_event", `{"type":"future_event","payload":{"anything":true}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	c, _ := newTestClient(t, sseHandler(events...))
	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	evs := drainEvents(t, ch)
	if evs[len(evs)-1].Kind != provider.StreamEventDone {
		t.Errorf("unknown event broke the stream: %s", dumpEvents(evs))
	}
}

func TestStream_UnknownBlockTypeIgnoredDeltas(t *testing.T) {
	// A future block kind (e.g. "image" output) we don't recognize.
	// Block-typed deltas targeting it should drop silently.
	events := [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":1}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"future_block_type"}}`},
		// Some delta we don't know how to handle for this block.
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"image_delta","data":"xyz"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	c, _ := newTestClient(t, sseHandler(events...))
	ch, _ := c.Stream(context.Background(), provider.Request{Model: "claude"})
	evs := drainEvents(t, ch)
	last := evs[len(evs)-1]
	if last.Kind != provider.StreamEventDone {
		t.Errorf("unknown block type broke the stream: %s", dumpEvents(evs))
	}
	// No tool_calls because there were no tool_use blocks.
	if len(last.Response.ToolCalls) != 0 {
		t.Errorf("ToolCalls = %v, want empty", last.Response.ToolCalls)
	}
}

// equalInts compares two int slices element-wise.
func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, x := range a {
		if x != b[i] {
			return false
		}
	}
	return true
}

// equalKinds compares two StreamEventKind slices element-wise.
func equalKinds(a, b []provider.StreamEventKind) bool {
	if len(a) != len(b) {
		return false
	}
	for i, x := range a {
		if x != b[i] {
			return false
		}
	}
	return true
}

// kindsOf extracts just the Kind field from each event for compact
// comparisons.
func kindsOf(events []provider.StreamEvent) []provider.StreamEventKind {
	out := make([]provider.StreamEventKind, 0, len(events))
	for _, ev := range events {
		out = append(out, ev.Kind)
	}
	return out
}

// dumpEvents pretty-prints events for failure messages.
func dumpEvents(events []provider.StreamEvent) string {
	var sb strings.Builder
	for i, ev := range events {
		sb.WriteString("  [")
		sb.WriteString(itoa(i))
		sb.WriteString("] ")
		sb.WriteString(ev.Kind.String())
		switch ev.Kind {
		case provider.StreamEventTextDelta, provider.StreamEventReasoningDelta:
			sb.WriteString(" text=")
			sb.WriteString(ev.Text)
		case provider.StreamEventToolCallStart, provider.StreamEventToolCallDelta, provider.StreamEventToolCallDone:
			sb.WriteString(" toolcall.name=")
			sb.WriteString(ev.ToolCall.Name)
			sb.WriteString(" args=")
			sb.WriteString(ev.ToolCall.Arguments)
		case provider.StreamEventError:
			sb.WriteString(" err=")
			if ev.Err != nil {
				sb.WriteString(ev.Err.Error())
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
