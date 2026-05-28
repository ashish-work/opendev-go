package tui

import (
	"context"
	"errors"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ashish-work/opendev-go/internal/agents"
	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/provider"
	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/workflow"
)

// stubProvider is a provider.Provider implementation that returns a
// canned Response. Used so TUI tests can exercise the agent loop
// without hitting the network. Set callErr to simulate API failures;
// set the response field to control what content / tool_calls come
// back.
type stubProvider struct {
	response provider.Response
	callErr  error
}

func (s stubProvider) Name() string { return "stub" }

func (s stubProvider) Call(ctx context.Context, req provider.Request) (provider.Response, error) {
	if err := ctx.Err(); err != nil {
		return provider.Response{}, err
	}
	if s.callErr != nil {
		return provider.Response{}, s.callErr
	}
	if s.response.Content == "" && len(s.response.ToolCalls) == 0 {
		// Default: a simple "ok" reply so the loop terminates.
		return provider.Response{Content: "ok"}, nil
	}
	return s.response, nil
}

// Stream synthesizes a deterministic event sequence from the stub's
// response field: a TextDelta for the content, Start/Delta/Done events
// per tool_call, then a terminal Done with the full response. Sticks
// to the same scripted state Call uses so individual tests don't have
// to script events separately.
func (s stubProvider) Stream(ctx context.Context, _ provider.Request) (<-chan provider.StreamEvent, error) {
	if s.callErr != nil {
		return nil, s.callErr
	}
	resp := s.response
	if resp.Content == "" && len(resp.ToolCalls) == 0 {
		resp = provider.Response{Content: "ok"}
	}

	ch := make(chan provider.StreamEvent, 8)
	go func() {
		defer close(ch)
		send := func(ev provider.StreamEvent) bool {
			select {
			case ch <- ev:
				return true
			case <-ctx.Done():
				return false
			}
		}
		if resp.Content != "" {
			if !send(provider.NewTextDelta(resp.Content)) {
				return
			}
		}
		for i, tc := range resp.ToolCalls {
			if !send(provider.NewToolCallStart(i, tc.ID, tc.Name)) {
				return
			}
			args := string(tc.Arguments)
			if args != "" {
				if !send(provider.NewToolCallDelta(i, args)) {
					return
				}
			}
			if !send(provider.NewToolCallDone(i, tc.ID, tc.Name, args)) {
				return
			}
		}
		respCopy := resp
		_ = send(provider.NewDone(&respCopy))
	}()
	return ch, nil
}

// newTestLoop builds a *agents.ReactLoop wired against the given stub
// provider. Used by both turn_test and tui_test (same package — these
// helpers are local to the test build only).
func newTestLoop(t *testing.T, p stubProvider) *agents.ReactLoop {
	t.Helper()
	caller := agents.NewLlmCaller(p, cost.Pricing{})
	registry := tools.NewRegistry()
	return agents.NewReactLoop(caller, registry, agents.Config{
		Workflow:      workflow.Config{Execution: workflow.SlotConfig{Model: "stub"}},
		MaxIterations: 5,
	})
}

func TestRunTurnCmd_ProducesTurnCompleteMsg(t *testing.T) {
	loop := newTestLoop(t, stubProvider{response: provider.Response{Content: "hi back"}})
	ctx := context.Background()
	cmd := runTurnCmd(ctx, loop, "hello", nil)
	if cmd == nil {
		t.Fatal("runTurnCmd should return a non-nil tea.Cmd")
	}
	msg := cmd()
	tcMsg, ok := msg.(turnCompleteMsg)
	if !ok {
		t.Fatalf("runTurnCmd produced %T, want turnCompleteMsg", msg)
	}
	if tcMsg.err != nil {
		t.Errorf("expected nil error on stub success, got %v", tcMsg.err)
	}
	if tcMsg.result.Content != "hi back" {
		t.Errorf("Result.Content = %q, want %q", tcMsg.result.Content, "hi back")
	}
}

func TestRunTurnCmd_PropagatesProviderError(t *testing.T) {
	loop := newTestLoop(t, stubProvider{callErr: errors.New("provider boom")})
	cmd := runTurnCmd(context.Background(), loop, "hello", nil)
	msg := cmd()
	tcMsg := msg.(turnCompleteMsg)
	if tcMsg.err == nil {
		t.Fatalf("expected error, got nil")
	}
	// The loop wraps the provider error with agents.ErrLLM; the
	// substring check tolerates that wrapping without pinning the
	// exact message.
	if !strings.Contains(tcMsg.err.Error(), "provider boom") {
		t.Errorf("error %q should contain the provider message", tcMsg.err)
	}
}

func TestRunTurnCmd_ContextCancellation(t *testing.T) {
	loop := newTestLoop(t, stubProvider{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled before the goroutine runs
	cmd := runTurnCmd(ctx, loop, "hello", nil)
	msg := cmd()
	tcMsg := msg.(turnCompleteMsg)
	if tcMsg.err == nil {
		t.Fatalf("expected error on cancelled context")
	}
	// The loop wraps ctx.Canceled with its own ErrInterrupted
	// sentinel (formatted as "%w: iter N: %v"). The wrapping uses
	// %w only on the sentinel, not on ctx.Err, so errors.Is needs
	// the sentinel to match.
	if !errors.Is(tcMsg.err, agents.ErrInterrupted) {
		t.Errorf("err = %v, want chain containing agents.ErrInterrupted", tcMsg.err)
	}
}

func TestUpdate_TurnCompleteSuccessRebuildsHistory(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m.thinking = true
	m.turnCancel = func() {} // dummy
	// Existing optimistic user message — should be replaced by the
	// translated loop messages.
	m.history = []viewMessage{{role: roleUser, content: "hello"}}

	loopMsgs := []provider.Message{
		{Role: "system", Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "ignored"}}},
		{Role: "user", Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "hello"}}},
		{Role: "assistant", Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "hi back"}}},
	}

	next, _ := m.Update(turnCompleteMsg{
		result: agents.Result{Messages: loopMsgs},
	})
	got := next.(model)

	if got.thinking {
		t.Errorf("thinking should reset to false after turn completes")
	}
	if got.turnCancel != nil {
		t.Errorf("turnCancel should be cleared after turn completes")
	}
	// System message dropped; user + assistant survive.
	if len(got.history) != 2 {
		t.Fatalf("history len = %d, want 2 (user + assistant; system skipped)", len(got.history))
	}
	if got.history[0].role != roleUser || got.history[0].content != "hello" {
		t.Errorf("history[0] = %+v, want user 'hello'", got.history[0])
	}
	if got.history[1].role != roleAssistant || got.history[1].content != "hi back" {
		t.Errorf("history[1] = %+v, want assistant 'hi back'", got.history[1])
	}
}

func TestUpdate_TurnCompleteCancellationAppendsNotice(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m.thinking = true
	m.history = []viewMessage{{role: roleUser, content: "hello"}}

	next, _ := m.Update(turnCompleteMsg{
		result: agents.Result{},
		err:    context.Canceled,
	})
	got := next.(model)

	if got.thinking {
		t.Errorf("thinking should reset to false")
	}
	// Should preserve the optimistic user message and append a
	// cancellation notice.
	last := got.history[len(got.history)-1]
	if last.role != roleAssistant {
		t.Errorf("last entry role = %d, want roleAssistant", last.role)
	}
	if !strings.Contains(last.content, "cancelled") {
		t.Errorf("last entry should mention cancelled, got %q", last.content)
	}
}

func TestUpdate_TurnCompleteErrorAppendsErrorMessage(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m.thinking = true
	m.history = []viewMessage{{role: roleUser, content: "hello"}}

	next, _ := m.Update(turnCompleteMsg{
		result: agents.Result{},
		err:    errors.New("API blew up"),
	})
	got := next.(model)

	if got.thinking {
		t.Errorf("thinking should reset to false")
	}
	last := got.history[len(got.history)-1]
	if last.role != roleAssistant {
		t.Errorf("error notice should be an assistant message, got role %d", last.role)
	}
	if !strings.Contains(last.content, "error") || !strings.Contains(last.content, "API blew up") {
		t.Errorf("error notice = %q, want it to mention the error message", last.content)
	}
}

func TestTranslateMessages_SkipsSystem(t *testing.T) {
	msgs := []provider.Message{
		{Role: "system", Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "sys"}}},
		{Role: "user", Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "hi"}}},
	}
	got := translateMessages(msgs)
	if len(got) != 1 {
		t.Fatalf("got %d, want 1 (system skipped)", len(got))
	}
	if got[0].role != roleUser {
		t.Errorf("kept role = %d, want roleUser", got[0].role)
	}
}

func TestTranslateMessages_AssistantTextBlocks(t *testing.T) {
	msgs := []provider.Message{
		{Role: "assistant", Content: []provider.ContentBlock{
			{Kind: provider.ContentText, Text: "part one"},
			{Kind: provider.ContentText, Text: "part two"},
		}},
	}
	got := translateMessages(msgs)
	if len(got) != 2 {
		t.Fatalf("got %d, want 2 assistant messages (one per text block)", len(got))
	}
	for i, v := range got {
		if v.role != roleAssistant {
			t.Errorf("got[%d].role = %d, want roleAssistant", i, v.role)
		}
	}
}

func TestTranslateMessages_ToolNameAndContent(t *testing.T) {
	msgs := []provider.Message{
		{
			Role:    "tool",
			Name:    "read_file",
			Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "file contents"}},
		},
	}
	got := translateMessages(msgs)
	if len(got) != 1 {
		t.Fatalf("got %d, want 1", len(got))
	}
	if got[0].role != roleTool {
		t.Errorf("role = %d, want roleTool", got[0].role)
	}
	if got[0].toolName != "read_file" {
		t.Errorf("toolName = %q, want 'read_file'", got[0].toolName)
	}
	if got[0].content != "file contents" {
		t.Errorf("content = %q, want 'file contents'", got[0].content)
	}
}

func TestTranslateMessages_EmptyTextBlocksSkipped(t *testing.T) {
	msgs := []provider.Message{
		{Role: "user", Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: ""}}},
		{Role: "assistant", Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: ""}}},
		{Role: "tool", Name: "x", Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: ""}}},
	}
	if got := translateMessages(msgs); len(got) != 0 {
		t.Errorf("empty content should produce no viewMessages, got %d", len(got))
	}
}

func TestTranslateMessages_UnknownRoleSkipped(t *testing.T) {
	msgs := []provider.Message{
		{Role: "developer", Content: []provider.ContentBlock{{Kind: provider.ContentText, Text: "x"}}},
	}
	if got := translateMessages(msgs); len(got) != 0 {
		t.Errorf("unknown role should be skipped, got %d", len(got))
	}
}

func TestApplyStreamEvent_TextDeltaAccumulatesIntoOneMessage(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m = applyStreamEvent(m, provider.NewTextDelta("Hel"))
	m = applyStreamEvent(m, provider.NewTextDelta("lo "))
	m = applyStreamEvent(m, provider.NewTextDelta("world"))
	if len(m.history) != 1 {
		t.Fatalf("history len = %d, want 1 (deltas should accumulate into one message)", len(m.history))
	}
	if m.history[0].role != roleAssistant || m.history[0].content != "Hello world" {
		t.Errorf("history[0] = %+v, want assistant 'Hello world'", m.history[0])
	}
	if m.pendingAssistantIdx != 0 {
		t.Errorf("pendingAssistantIdx = %d, want 0", m.pendingAssistantIdx)
	}
}

func TestApplyStreamEvent_ToolCallStartResetsPending(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	// First, accumulate some assistant text.
	m = applyStreamEvent(m, provider.NewTextDelta("Let me check"))
	if m.pendingAssistantIdx != 0 {
		t.Fatalf("pendingAssistantIdx after first delta = %d, want 0", m.pendingAssistantIdx)
	}
	// ToolCallStart should reset pending so the next TextDelta starts a NEW message.
	m = applyStreamEvent(m, provider.NewToolCallStart(0, "call_1", "read_file"))
	if m.pendingAssistantIdx != -1 {
		t.Errorf("pendingAssistantIdx after ToolCallStart = %d, want -1", m.pendingAssistantIdx)
	}
	if len(m.history) != 2 {
		t.Fatalf("history len = %d, want 2 (assistant text + tool placeholder)", len(m.history))
	}
	if m.history[1].role != roleTool || m.history[1].toolName != "read_file" {
		t.Errorf("history[1] = %+v, want tool 'read_file'", m.history[1])
	}
	if m.history[1].content != "(running...)" {
		t.Errorf("tool placeholder = %q, want '(running...)'", m.history[1].content)
	}
	// A subsequent TextDelta after the tool starts a fresh assistant message.
	m = applyStreamEvent(m, provider.NewTextDelta("done"))
	if len(m.history) != 3 {
		t.Fatalf("history len after next-iter text = %d, want 3", len(m.history))
	}
	if m.history[2].content != "done" {
		t.Errorf("new assistant message content = %q, want 'done'", m.history[2].content)
	}
}

func TestApplyStreamEvent_ToolCallDoneReplacesPlaceholder(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m = applyStreamEvent(m, provider.NewToolCallStart(0, "call_1", "read_file"))
	m = applyStreamEvent(m, provider.NewToolCallDone(0, "call_1", "read_file", `{"path":"x.go"}`))
	last := m.history[len(m.history)-1]
	if last.content != `{"path":"x.go"}` {
		t.Errorf("tool content after Done = %q, want assembled args", last.content)
	}
}

func TestApplyStreamEvent_ErrorAppendsAssistantMessage(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m = applyStreamEvent(m, provider.NewTextDelta("partial"))
	m = applyStreamEvent(m, provider.NewError(errors.New("api blew up")))
	last := m.history[len(m.history)-1]
	if last.role != roleAssistant {
		t.Errorf("last role = %d, want roleAssistant", last.role)
	}
	if !strings.Contains(last.content, "api blew up") {
		t.Errorf("error message = %q, want to mention 'api blew up'", last.content)
	}
	if m.pendingAssistantIdx != -1 {
		t.Errorf("pendingAssistantIdx after error = %d, want -1", m.pendingAssistantIdx)
	}
}

func TestApplyStreamEvent_DoneAndUsageAndDeltaIgnored(t *testing.T) {
	// These events flow through the loop's stream but have no
	// dedicated UI representation in this commit. Verify they don't
	// corrupt history.
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	before := len(m.history)
	m = applyStreamEvent(m, provider.NewDone(&provider.Response{Content: "ignored"}))
	m = applyStreamEvent(m, provider.NewUsage(provider.Usage{PromptTokens: 100}))
	m = applyStreamEvent(m, provider.NewToolCallDelta(0, `{"part":`))
	m = applyStreamEvent(m, provider.NewReasoningDelta("hmm"))
	if len(m.history) != before {
		t.Errorf("history len = %d, want %d (events should not append)", len(m.history), before)
	}
}

func TestUpdate_StreamEventMsg_AppendsTextAndRearmsCmd(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m.thinking = true
	m.streamCh = make(chan provider.StreamEvent, 4)
	m.pendingAssistantIdx = -1

	next, cmd := m.Update(streamEventMsg{event: provider.NewTextDelta("hello")})
	got := next.(model)
	if len(got.history) != 1 || got.history[0].content != "hello" {
		t.Errorf("history = %+v, want one assistant message 'hello'", got.history)
	}
	if cmd == nil {
		t.Errorf("Update should return the next read Cmd to re-arm the loop")
	}
}

func TestUpdate_StreamSinkClosedMsg_NoOp(t *testing.T) {
	// streamSinkClosedMsg arriving while idle (or after turnComplete)
	// should be a no-op: don't crash, don't return a Cmd.
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)

	next, cmd := m.Update(streamSinkClosedMsg{})
	if cmd != nil {
		t.Errorf("streamSinkClosedMsg should not return a Cmd, got %v", cmd)
	}
	if next.(model).history != nil && len(next.(model).history) != 0 {
		t.Errorf("history mutated unexpectedly: %+v", next.(model).history)
	}
}

func TestUpdate_TurnCompleteClosesStreamChannel(t *testing.T) {
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m.thinking = true
	m.streamCh = make(chan provider.StreamEvent, 4)
	m.pendingAssistantIdx = 3

	next, _ := m.Update(turnCompleteMsg{result: agents.Result{}})
	got := next.(model)
	if got.streamCh != nil {
		t.Errorf("streamCh should be nil after turnCompleteMsg")
	}
	if got.pendingAssistantIdx != -1 {
		t.Errorf("pendingAssistantIdx = %d, want -1 after turnCompleteMsg", got.pendingAssistantIdx)
	}
}

func TestUpdate_MidStreamCancellationKeepsPartialText(t *testing.T) {
	// Simulate: user submits, two text deltas land, user cancels, the
	// loop returns ErrInterrupted. The partial assistant message
	// should survive AND get a "(turn cancelled)" notice.
	m := initialModel(nil, "")
	m, _ = applyWindowSize(m, 100, 30)
	m.thinking = true
	m.history = []viewMessage{{role: roleUser, content: "long task"}}
	m.streamCh = make(chan provider.StreamEvent, 4)

	// Two deltas land via the streaming path.
	next, _ := m.Update(streamEventMsg{event: provider.NewTextDelta("I started")})
	m = next.(model)
	next, _ = m.Update(streamEventMsg{event: provider.NewTextDelta(" working")})
	m = next.(model)
	if len(m.history) != 2 || m.history[1].content != "I started working" {
		t.Fatalf("partial text not accumulated: %+v", m.history)
	}

	// User cancels mid-stream → loop returns ErrInterrupted in a
	// turnCompleteMsg with the loop's partial Result.Messages
	// (in real flow the loop captures more, but we simulate the empty
	// case to confirm appendOrReplaceHistory keeps the optimistic
	// streamed message).
	next, _ = m.Update(turnCompleteMsg{
		result: agents.Result{},
		err:    agents.ErrInterrupted,
	})
	got := next.(model)

	// The partial streamed text should still be visible somewhere in
	// the history, and a "cancelled" notice should be appended.
	var foundPartial, foundNotice bool
	for _, vm := range got.history {
		if strings.Contains(vm.content, "I started working") {
			foundPartial = true
		}
		if strings.Contains(vm.content, "cancelled") {
			foundNotice = true
		}
	}
	if !foundPartial {
		t.Errorf("partial streamed text lost after cancellation: %+v", got.history)
	}
	if !foundNotice {
		t.Errorf("cancellation notice missing: %+v", got.history)
	}
}

// Compile-time check: stubProvider satisfies provider.Provider.
var _ provider.Provider = stubProvider{}

// Reference the tea import so unused-import linting stays quiet
// even if a future commit removes the only use.
var _ tea.Cmd = nil
