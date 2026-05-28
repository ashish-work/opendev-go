package openai

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

// sseHandler is an httptest handler that writes a list of SSE chunks
// to the response body with proper framing and flushing. Each entry is
// the JSON payload (or the literal "[DONE]"); the helper prepends
// "data: " and appends the "\n\n" separator and flushes after each
// chunk so the client sees them one at a time.
//
// Newlines inside chunks are stripped before framing because SSE uses
// "\n" as the line delimiter — embedded newlines would split a single
// JSON payload across multiple SSE lines and fail to parse. Tests can
// therefore write multi-line JSON literals for readability.
func sseHandler(chunks ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for _, c := range chunks {
			oneLine := strings.ReplaceAll(c, "\n", "")
			oneLine = strings.ReplaceAll(oneLine, "\t", "")
			_, _ = io.WriteString(w, "data: "+oneLine+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
	}
}

// drainEvents reads until the channel closes. Returns the events in
// arrival order. Times out via test deadline if the producer never
// closes the channel (catches close-contract violations).
func drainEvents(t *testing.T, ch <-chan provider.StreamEvent) []provider.StreamEvent {
	t.Helper()
	var out []provider.StreamEvent
	for ev := range ch {
		out = append(out, ev)
	}
	return out
}

func TestStream_PlainText(t *testing.T) {
	chunks := []string{
		`{"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"content":"lo "},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}`,
		`[DONE]`,
	}
	c, _ := newTestClient(t, sseHandler(chunks...))

	ch, err := c.Stream(context.Background(), provider.Request{
		Model: "gpt-4o",
		Messages: []provider.Message{
			{Role: "user", Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := drainEvents(t, ch)

	// Expected sequence: 3 TextDeltas, 1 Done.
	if len(events) != 4 {
		t.Fatalf("events = %d, want 4: %+v", len(events), events)
	}
	for i, want := range []string{"Hel", "lo ", "world"} {
		if events[i].Kind != provider.StreamEventTextDelta {
			t.Errorf("events[%d].Kind = %s, want text_delta", i, events[i].Kind)
		}
		if events[i].Text != want {
			t.Errorf("events[%d].Text = %q, want %q", i, events[i].Text, want)
		}
	}
	done := events[3]
	if done.Kind != provider.StreamEventDone {
		t.Fatalf("last event Kind = %s, want done", done.Kind)
	}
	if done.Response == nil {
		t.Fatal("Done.Response is nil")
	}
	if done.Response.Content != "Hello world" {
		t.Errorf("Done.Response.Content = %q, want %q", done.Response.Content, "Hello world")
	}
	if done.Response.FinishReason != "stop" {
		t.Errorf("Done.Response.FinishReason = %q, want %q", done.Response.FinishReason, "stop")
	}
}

func TestStream_SingleToolCall(t *testing.T) {
	chunks := []string{
		// First chunk for the tool_call: id + name + empty args.
		`{"choices":[{"index":0,"delta":{"tool_calls":[
			{"index":0,"id":"call_abc","type":"function",
			 "function":{"name":"read_file","arguments":""}}]}}]}`,
		// Two argument-fragment chunks.
		`{"choices":[{"index":0,"delta":{"tool_calls":[
			{"index":0,"function":{"arguments":"{\"path\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[
			{"index":0,"function":{"arguments":"\"main.go\"}"}}]}}]}`,
		// Final chunk with finish_reason.
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	}
	c, _ := newTestClient(t, sseHandler(chunks...))

	ch, err := c.Stream(context.Background(), provider.Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := drainEvents(t, ch)

	// Expected: Start, Delta, Delta, ToolCallDone, Done.
	if len(events) != 5 {
		t.Fatalf("events = %d, want 5:\n%s", len(events), dumpEvents(events))
	}
	if events[0].Kind != provider.StreamEventToolCallStart {
		t.Errorf("events[0].Kind = %s, want tool_call_start", events[0].Kind)
	}
	if events[0].ToolCall.ID != "call_abc" || events[0].ToolCall.Name != "read_file" {
		t.Errorf("Start.ToolCall = %+v", events[0].ToolCall)
	}
	for i, want := range []string{`{"path":`, `"main.go"}`} {
		ev := events[1+i]
		if ev.Kind != provider.StreamEventToolCallDelta {
			t.Errorf("events[%d].Kind = %s, want tool_call_delta", 1+i, ev.Kind)
		}
		if ev.ToolCall.Arguments != want {
			t.Errorf("events[%d].Arguments = %q, want %q", 1+i, ev.ToolCall.Arguments, want)
		}
	}
	if events[3].Kind != provider.StreamEventToolCallDone {
		t.Errorf("events[3].Kind = %s, want tool_call_done", events[3].Kind)
	}
	if events[3].ToolCall.Arguments != `{"path":"main.go"}` {
		t.Errorf("Done.Arguments = %q, want assembled JSON", events[3].ToolCall.Arguments)
	}
	if events[4].Kind != provider.StreamEventDone {
		t.Fatalf("events[4].Kind = %s, want done", events[4].Kind)
	}
	if got := events[4].Response.ToolCalls; len(got) != 1 {
		t.Fatalf("Response.ToolCalls len = %d, want 1", len(got))
	}
	if got := string(events[4].Response.ToolCalls[0].Arguments); got != `{"path":"main.go"}` {
		t.Errorf("Response.ToolCalls[0].Arguments = %q", got)
	}
}

func TestStream_MultipleToolCallsInterleaved(t *testing.T) {
	chunks := []string{
		// Open both tool_calls in one chunk.
		`{"choices":[{"index":0,"delta":{"tool_calls":[
			{"index":0,"id":"call_a","type":"function","function":{"name":"f1","arguments":""}},
			{"index":1,"id":"call_b","type":"function","function":{"name":"f2","arguments":""}}
		]}}]}`,
		// Interleave argument fragments across the two indices.
		`{"choices":[{"index":0,"delta":{"tool_calls":[
			{"index":0,"function":{"arguments":"{\"x\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[
			{"index":1,"function":{"arguments":"{\"y\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[
			{"index":0,"function":{"arguments":"1}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[
			{"index":1,"function":{"arguments":"2}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	}
	c, _ := newTestClient(t, sseHandler(chunks...))

	ch, err := c.Stream(context.Background(), provider.Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := drainEvents(t, ch)

	// Done events should arrive in index order regardless of arrival
	// order — that's the determinism guarantee finalize() makes.
	done := events[len(events)-1]
	if done.Kind != provider.StreamEventDone {
		t.Fatalf("last event Kind = %s, want done", done.Kind)
	}
	if len(done.Response.ToolCalls) != 2 {
		t.Fatalf("Response.ToolCalls len = %d, want 2", len(done.Response.ToolCalls))
	}
	tcs := done.Response.ToolCalls
	if tcs[0].ID != "call_a" || string(tcs[0].Arguments) != `{"x":1}` {
		t.Errorf("ToolCalls[0] = %+v", tcs[0])
	}
	if tcs[1].ID != "call_b" || string(tcs[1].Arguments) != `{"y":2}` {
		t.Errorf("ToolCalls[1] = %+v", tcs[1])
	}

	// ToolCallDone events should also come out in index order, before
	// the final StreamEventDone.
	var doneIndices []int
	for _, ev := range events {
		if ev.Kind == provider.StreamEventToolCallDone {
			doneIndices = append(doneIndices, ev.ToolCall.Index)
		}
	}
	if !equalInts(doneIndices, []int{0, 1}) {
		t.Errorf("ToolCallDone index order = %v, want [0 1]", doneIndices)
	}
}

func TestStream_ToolCallArgsInOneChunk(t *testing.T) {
	chunks := []string{
		`{"choices":[{"index":0,"delta":{"tool_calls":[
			{"index":0,"id":"call_x","type":"function",
			 "function":{"name":"f","arguments":"{\"k\":\"v\"}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	}
	c, _ := newTestClient(t, sseHandler(chunks...))

	ch, err := c.Stream(context.Background(), provider.Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := drainEvents(t, ch)

	var kinds []provider.StreamEventKind
	for _, ev := range events {
		kinds = append(kinds, ev.Kind)
	}
	want := []provider.StreamEventKind{
		provider.StreamEventToolCallStart,
		provider.StreamEventToolCallDelta, // single fragment arrived with id+name
		provider.StreamEventToolCallDone,
		provider.StreamEventDone,
	}
	if !equalKinds(kinds, want) {
		t.Errorf("event kinds = %v, want %v", kinds, want)
	}
	last := events[len(events)-1]
	if string(last.Response.ToolCalls[0].Arguments) != `{"k":"v"}` {
		t.Errorf("Response.ToolCalls[0].Arguments = %q", last.Response.ToolCalls[0].Arguments)
	}
}

func TestStream_UsageChunk(t *testing.T) {
	chunks := []string{
		`{"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
		// Usage-only chunk: zero choices, populated usage.
		`{"choices":[],"usage":{
			"prompt_tokens":42,"completion_tokens":7,"total_tokens":49,
			"prompt_tokens_details":{"cached_tokens":10}}}`,
		`[DONE]`,
	}
	c, _ := newTestClient(t, sseHandler(chunks...))

	ch, err := c.Stream(context.Background(), provider.Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := drainEvents(t, ch)

	var usageEv *provider.StreamEvent
	for i := range events {
		if events[i].Kind == provider.StreamEventUsage {
			usageEv = &events[i]
			break
		}
	}
	if usageEv == nil {
		t.Fatalf("no Usage event in stream:\n%s", dumpEvents(events))
	}
	if usageEv.Usage.PromptTokens != 42 || usageEv.Usage.CompletionTokens != 7 || usageEv.Usage.CachedTokens != 10 {
		t.Errorf("Usage = %+v, want {Prompt:42 Completion:7 Cached:10}", usageEv.Usage)
	}
	// Final Done event should carry the same usage in its Response.
	done := events[len(events)-1]
	if done.Response.Usage.PromptTokens != 42 {
		t.Errorf("Done.Response.Usage.PromptTokens = %d, want 42", done.Response.Usage.PromptTokens)
	}
}

func TestStream_NoDoneSentinelStillCompletes(t *testing.T) {
	// Some OpenAI-compatible servers close the stream cleanly without
	// emitting the literal [DONE] line. We treat that as success and
	// emit the assembled Done event anyway.
	chunks := []string{
		`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`,
	}
	c, _ := newTestClient(t, sseHandler(chunks...))

	ch, err := c.Stream(context.Background(), provider.Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := drainEvents(t, ch)
	last := events[len(events)-1]
	if last.Kind != provider.StreamEventDone {
		t.Fatalf("last event = %s, want done", last.Kind)
	}
	if last.Response.Content != "hi" {
		t.Errorf("Content = %q, want %q", last.Response.Content, "hi")
	}
}

func TestStream_SetupHTTP401(t *testing.T) {
	c, _ := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"bad key"}}`)
	})

	ch, err := c.Stream(context.Background(), provider.Request{Model: "gpt-4o"})
	if err == nil {
		t.Fatal("expected setup error, got nil")
	}
	if ch != nil {
		t.Errorf("expected nil channel on setup error, got %v", ch)
	}
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) {
		t.Fatalf("err type = %T, want *HTTPError", err)
	}
	if httpErr.Status != http.StatusUnauthorized {
		t.Errorf("Status = %d, want 401", httpErr.Status)
	}
	if !strings.Contains(httpErr.Body, "bad key") {
		t.Errorf("Body missing expected text: %q", httpErr.Body)
	}
}

func TestStream_MalformedJSONEmitsErrorEvent(t *testing.T) {
	chunks := []string{
		`{"choices":[{"index":0,"delta":{"content":"ok"}}]}`,
		`{this is not json}`,
		`[DONE]`,
	}
	c, _ := newTestClient(t, sseHandler(chunks...))

	ch, err := c.Stream(context.Background(), provider.Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Stream setup: %v", err)
	}
	events := drainEvents(t, ch)

	// Expected: text delta, then Error, then channel close (no Done).
	if len(events) < 2 {
		t.Fatalf("events = %d, want >=2:\n%s", len(events), dumpEvents(events))
	}
	last := events[len(events)-1]
	if last.Kind != provider.StreamEventError {
		t.Fatalf("last event Kind = %s, want error", last.Kind)
	}
	if last.Err == nil {
		t.Errorf("Error event has nil Err")
	}
	// Done must NOT appear after Error.
	for _, ev := range events {
		if ev.Kind == provider.StreamEventDone {
			t.Errorf("unexpected Done event after Error in stream")
		}
	}
}

func TestStream_ContextCancellation(t *testing.T) {
	// Server keeps writing chunks forever; client cancels after the
	// first one arrives. We verify: (a) the channel closes cleanly
	// after cancellation, (b) no goroutines leak.
	serverDone := make(chan struct{})
	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		defer close(serverDone)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		_, _ = io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"content":"x"}}]}`+"\n\n")
		flusher.Flush()
		<-r.Context().Done() // wait until client cancels
	})

	before := runtime.NumGoroutine()

	ctx, cancel := context.WithCancel(context.Background())
	ch, err := c.Stream(ctx, provider.Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	// Read the first event, then cancel and drain.
	<-ch
	cancel()
	_ = drainEvents(t, ch) // must terminate via channel close

	<-serverDone // ensure the server handler returned

	// Give the goroutine a beat to wind down.
	for i := 0; i < 20; i++ {
		if runtime.NumGoroutine() <= before+1 { // +1 for the handler if still cleaning up
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if leaked := runtime.NumGoroutine() - before; leaked > 1 {
		t.Errorf("goroutine leak: +%d after cancel", leaked)
	}
}

func TestStream_SendsStreamFlagInBody(t *testing.T) {
	// Verify Stream() POSTs a body with stream:true + stream_options.
	var gotBody []byte
	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	})

	ch, err := c.Stream(context.Background(), provider.Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	_ = drainEvents(t, ch)

	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("body not JSON: %v\n%s", err, gotBody)
	}
	if payload["stream"] != true {
		t.Errorf(`body["stream"] = %v, want true`, payload["stream"])
	}
	opts, ok := payload["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf(`body["stream_options"] = %v, want map`, payload["stream_options"])
	}
	if opts["include_usage"] != true {
		t.Errorf(`stream_options["include_usage"] = %v, want true`, opts["include_usage"])
	}
}

func TestStream_AcceptHeaderSet(t *testing.T) {
	var gotAccept string
	c, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotAccept = r.Header.Get("Accept")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	})
	ch, _ := c.Stream(context.Background(), provider.Request{Model: "gpt-4o"})
	_ = drainEvents(t, ch)
	if gotAccept != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", gotAccept)
	}
}

func TestStream_ReasoningDeltaPassthrough(t *testing.T) {
	// Standard OpenAI doesn't emit reasoning_content but some compat
	// providers do. Verify we pass them through as ReasoningDelta.
	chunks := []string{
		`{"choices":[{"index":0,"delta":{"reasoning_content":"thinking…"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}]}`,
		`[DONE]`,
	}
	c, _ := newTestClient(t, sseHandler(chunks...))

	ch, _ := c.Stream(context.Background(), provider.Request{Model: "deepseek-r1"})
	events := drainEvents(t, ch)

	var reasoningSeen bool
	for _, ev := range events {
		if ev.Kind == provider.StreamEventReasoningDelta && ev.Text == "thinking…" {
			reasoningSeen = true
		}
	}
	if !reasoningSeen {
		t.Errorf("no ReasoningDelta event in stream:\n%s", dumpEvents(events))
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

// dumpEvents pretty-prints a slice of events for test failure messages.
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
			sb.WriteString(" toolcall=")
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
	// strconv would do, but staying stdlib-light with a tiny inline
	// helper avoids an import just for one debug print.
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
