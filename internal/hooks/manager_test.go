package hooks

import (
	"context"
	"errors"
	"regexp"
	"strings"
	"testing"
	"time"
)

// newTestManager builds a Manager whose settings register the given
// matchers for the given event. matchers should already have their
// `compiled` regex populated (use mustCompileMatcher) — these tests
// construct HookMatcher values by hand rather than going through
// the LoadFile path.
func newTestManager(t *testing.T, event HookEvent, matchers []HookMatcher) *Manager {
	t.Helper()
	settings := HookSettings{
		Hooks: map[HookEvent][]HookMatcher{event: matchers},
	}
	return NewManager(settings, NewExecutor(""))
}

// mustCompileMatcher returns a HookMatcher with the regex already
// compiled. Tests use this instead of LoadFile so they don't need
// to round-trip through temp files.
func mustCompileMatcher(t *testing.T, pattern, command string) HookMatcher {
	t.Helper()
	m := HookMatcher{Matcher: pattern, Command: command}
	if pattern != "" {
		re, err := regexp.Compile(pattern)
		if err != nil {
			t.Fatalf("compile regex %q: %v", pattern, err)
		}
		m.compiled = re
	}
	return m
}

func TestFire_NoHooksRegistered_FastEmptyResult(t *testing.T) {
	mgr := NewManager(HookSettings{}, NewExecutor(""))
	got, err := mgr.Fire(context.Background(), HookEventPreToolUse, "bash", struct{}{})
	if err != nil {
		t.Fatalf("Fire: %v", err)
	}
	if got == nil {
		t.Fatal("FireResult should not be nil")
	}
	if len(got.HookResults) != 0 {
		t.Errorf("no hooks should produce no HookResults; got %d", len(got.HookResults))
	}
	if got.PermissionDecision != "" {
		t.Errorf("no hooks should produce no permission verdict; got %q", got.PermissionDecision)
	}
}

func TestFire_SingleMatcherDecisionPropagates(t *testing.T) {
	matchers := []HookMatcher{
		mustCompileMatcher(t, "bash",
			`echo '{"reason":"audit complete","permissionDecision":"allow"}'`),
	}
	mgr := newTestManager(t, HookEventPreToolUse, matchers)

	got, err := mgr.Fire(context.Background(), HookEventPreToolUse, "bash", struct{}{})
	if err != nil {
		t.Fatalf("Fire: %v", err)
	}
	if !got.IsAllow() {
		t.Errorf("expected IsAllow; got perm=%q", got.PermissionDecision)
	}
	if got.Reason != "audit complete" {
		t.Errorf("Reason = %q, want 'audit complete'", got.Reason)
	}
	if len(got.HookResults) != 1 {
		t.Errorf("HookResults len = %d, want 1", len(got.HookResults))
	}
}

func TestFire_NonMatchingMatcherIsSkipped(t *testing.T) {
	matchers := []HookMatcher{
		mustCompileMatcher(t, "^bash$", `echo '{"reason":"bash matched"}'`),
	}
	mgr := newTestManager(t, HookEventPreToolUse, matchers)

	got, err := mgr.Fire(context.Background(), HookEventPreToolUse, "read_file", struct{}{})
	if err != nil {
		t.Fatalf("Fire: %v", err)
	}
	if len(got.HookResults) != 0 {
		t.Errorf("non-matching matcher should not produce HookOutcome; got %d entries",
			len(got.HookResults))
	}
}

func TestFire_AdditionalContextConcatsAcrossHooks(t *testing.T) {
	matchers := []HookMatcher{
		mustCompileMatcher(t, "", `echo '{"additionalContext":"first note"}'`),
		mustCompileMatcher(t, "", `echo '{"additionalContext":"second note"}'`),
	}
	mgr := newTestManager(t, HookEventUserPromptSubmit, matchers)

	got, err := mgr.Fire(context.Background(), HookEventUserPromptSubmit, "", struct{}{})
	if err != nil {
		t.Fatalf("Fire: %v", err)
	}
	want := "first note\n\nsecond note"
	if got.AdditionalContext != want {
		t.Errorf("AdditionalContext = %q, want %q", got.AdditionalContext, want)
	}
}

func TestFire_EmptyContextSkippedInConcat(t *testing.T) {
	matchers := []HookMatcher{
		mustCompileMatcher(t, "", `echo '{"additionalContext":""}'`),
		mustCompileMatcher(t, "", `echo '{"additionalContext":"only this"}'`),
	}
	mgr := newTestManager(t, HookEventUserPromptSubmit, matchers)

	got, _ := mgr.Fire(context.Background(), HookEventUserPromptSubmit, "", struct{}{})
	if got.AdditionalContext != "only this" {
		t.Errorf("AdditionalContext = %q, want 'only this'", got.AdditionalContext)
	}
}

func TestFire_UpdatedInputLastWins(t *testing.T) {
	matchers := []HookMatcher{
		mustCompileMatcher(t, "", `echo '{"updatedInput":{"v":1}}'`),
		mustCompileMatcher(t, "", `echo '{"updatedInput":{"v":2}}'`),
		mustCompileMatcher(t, "", `echo '{"updatedInput":{"v":3}}'`),
	}
	mgr := newTestManager(t, HookEventPreToolUse, matchers)

	got, _ := mgr.Fire(context.Background(), HookEventPreToolUse, "x", struct{}{})
	if !strings.Contains(string(got.UpdatedInput), `"v":3`) {
		t.Errorf("UpdatedInput = %q, want last writer's value (v:3)", got.UpdatedInput)
	}
}

func TestFire_DenyShortCircuitsRemainingHooks(t *testing.T) {
	matchers := []HookMatcher{
		mustCompileMatcher(t, "",
			`echo '{"permissionDecision":"deny","reason":"forbidden"}'`),
		mustCompileMatcher(t, "",
			`echo '{"additionalContext":"should not run"}'`),
	}
	mgr := newTestManager(t, HookEventPreToolUse, matchers)

	got, err := mgr.Fire(context.Background(), HookEventPreToolUse, "x", struct{}{})
	if err != nil {
		t.Fatalf("Fire: %v", err)
	}
	if !got.IsDeny() {
		t.Errorf("expected IsDeny; got perm=%q", got.PermissionDecision)
	}
	if got.Reason != "forbidden" {
		t.Errorf("Reason = %q, want 'forbidden'", got.Reason)
	}
	if len(got.HookResults) != 1 {
		t.Errorf("HookResults len = %d, want 1 (second hook should not run)",
			len(got.HookResults))
	}
	if strings.Contains(got.AdditionalContext, "should not run") {
		t.Errorf("second hook's context leaked into result: %q", got.AdditionalContext)
	}
}

func TestFire_AllowThenDeny_DenyWins(t *testing.T) {
	matchers := []HookMatcher{
		mustCompileMatcher(t, "",
			`echo '{"permissionDecision":"allow","reason":"first ok"}'`),
		mustCompileMatcher(t, "",
			`echo '{"permissionDecision":"deny","reason":"actually no"}'`),
	}
	mgr := newTestManager(t, HookEventPreToolUse, matchers)

	got, _ := mgr.Fire(context.Background(), HookEventPreToolUse, "x", struct{}{})
	if !got.IsDeny() {
		t.Errorf("deny after allow should win; got perm=%q", got.PermissionDecision)
	}
	if got.Reason != "actually no" {
		t.Errorf("Reason should pair with the winning deny; got %q", got.Reason)
	}
	if len(got.HookResults) != 2 {
		t.Errorf("HookResults len = %d, want 2 (both should have run)",
			len(got.HookResults))
	}
}

func TestFire_AllowThenAllow_FirstWins(t *testing.T) {
	matchers := []HookMatcher{
		mustCompileMatcher(t, "",
			`echo '{"permissionDecision":"allow","reason":"first allow"}'`),
		mustCompileMatcher(t, "",
			`echo '{"permissionDecision":"allow","reason":"second allow"}'`),
	}
	mgr := newTestManager(t, HookEventPreToolUse, matchers)

	got, _ := mgr.Fire(context.Background(), HookEventPreToolUse, "x", struct{}{})
	if !got.IsAllow() {
		t.Errorf("expected IsAllow")
	}
	if got.Reason != "first allow" {
		t.Errorf("Reason = %q, want 'first allow' (first opinion wins)", got.Reason)
	}
}

func TestFire_AskThenAllow_AskWins(t *testing.T) {
	matchers := []HookMatcher{
		mustCompileMatcher(t, "",
			`echo '{"permissionDecision":"ask","reason":"unsure"}'`),
		mustCompileMatcher(t, "",
			`echo '{"permissionDecision":"allow","reason":"fine"}'`),
	}
	mgr := newTestManager(t, HookEventPreToolUse, matchers)

	got, _ := mgr.Fire(context.Background(), HookEventPreToolUse, "x", struct{}{})
	if !got.IsAsk() {
		t.Errorf("first ask should win; got perm=%q", got.PermissionDecision)
	}
	if got.Reason != "unsure" {
		t.Errorf("Reason = %q, want 'unsure'", got.Reason)
	}
}

func TestFire_ReasonFallbackWhenNoPermDecision(t *testing.T) {
	// No hook expresses a perm decision, but the first hook
	// attaches a reason. That reason should surface so the user
	// sees the audit note.
	matchers := []HookMatcher{
		mustCompileMatcher(t, "",
			`echo '{"reason":"audit logged"}'`),
		mustCompileMatcher(t, "",
			`echo '{"additionalContext":"more info"}'`),
	}
	mgr := newTestManager(t, HookEventPostToolUse, matchers)

	got, _ := mgr.Fire(context.Background(), HookEventPostToolUse, "x", struct{}{})
	if got.PermissionDecision != "" {
		t.Errorf("perm should be empty; got %q", got.PermissionDecision)
	}
	if got.Reason != "audit logged" {
		t.Errorf("Reason fallback = %q, want 'audit logged'", got.Reason)
	}
}

func TestFire_PerHookTimeoutDoesNotHaltLoop(t *testing.T) {
	// First hook times out (200ms cap, sleeps 10s); the second
	// hook should still run and produce its decision. The timed-
	// out hook's outcome captures the err.
	matchers := []HookMatcher{
		mustCompileMatcher(t, "", "sleep 10"),
		mustCompileMatcher(t, "",
			`echo '{"permissionDecision":"allow","reason":"after timeout"}'`),
	}
	matchers[0].TimeoutMs = 200
	mgr := newTestManager(t, HookEventPreToolUse, matchers)

	start := time.Now()
	got, err := mgr.Fire(context.Background(), HookEventPreToolUse, "x", struct{}{})
	elapsed := time.Since(start)

	if err != nil {
		t.Errorf("Fire should not return error for per-hook timeout; got %v", err)
	}
	if len(got.HookResults) != 2 {
		t.Errorf("HookResults len = %d, want 2 (loop continued after timeout)",
			len(got.HookResults))
	}
	if got.HookResults[0].Err == nil {
		t.Errorf("first hook should have an error (timeout)")
	}
	if !errors.Is(got.HookResults[0].Err, context.DeadlineExceeded) {
		t.Errorf("first hook err = %v, want chain with DeadlineExceeded", got.HookResults[0].Err)
	}
	if !got.IsAllow() {
		t.Errorf("second hook's allow should still win; got perm=%q", got.PermissionDecision)
	}
	if elapsed > 5*time.Second {
		t.Errorf("Fire took %s — timeout apparently did not enforce; want <5s", elapsed)
	}
}

func TestFire_PerHookSpawnFailureDoesNotHaltLoop(t *testing.T) {
	// First hook gets a payload it can't marshal (channels in payload).
	// Second hook succeeds. This exercises infrastructure failure
	// continuation. We use the payload-marshal path because it's
	// the cleanest way to surface an err from Executor.Run.
	matchers := []HookMatcher{
		mustCompileMatcher(t, "", "true"),
		mustCompileMatcher(t, "",
			`echo '{"permissionDecision":"allow","reason":"still ran"}'`),
	}
	mgr := newTestManager(t, HookEventPreToolUse, matchers)

	// Payload contains a channel — every hook gets the same
	// marshal error. We expect BOTH outcomes to record the err
	// and the loop to complete without panicking.
	got, err := mgr.Fire(context.Background(), HookEventPreToolUse, "x",
		map[string]any{"bad": make(chan int)})

	if err != nil {
		t.Errorf("Fire should not return error for marshal failure; got %v", err)
	}
	if len(got.HookResults) != 2 {
		t.Errorf("HookResults len = %d, want 2", len(got.HookResults))
	}
	for i, h := range got.HookResults {
		if h.Err == nil {
			t.Errorf("hook %d should have marshal error", i)
		}
	}
}

func TestFire_CtxCancellationReturnsCtxErr(t *testing.T) {
	matchers := []HookMatcher{
		mustCompileMatcher(t, "", "sleep 10"),
		mustCompileMatcher(t, "", "echo '{}'"),
	}
	mgr := newTestManager(t, HookEventPreToolUse, matchers)

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel from a goroutine after a short delay so the first
	// hook is already running when the cancel fires.
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	got, err := mgr.Fire(ctx, HookEventPreToolUse, "x", struct{}{})
	if err == nil {
		t.Fatal("expected ctx error, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want chain with context.Canceled", err)
	}
	if got == nil {
		t.Fatal("FireResult should not be nil even on cancel")
	}
	// Partial HookResults available — at least the first hook
	// was started before cancel landed.
	if len(got.HookResults) == 0 {
		t.Errorf("no HookResults captured before cancel; got 0")
	}
}

func TestFireResult_HelperConsistency(t *testing.T) {
	cases := []struct {
		perm    PermissionDecision
		isDeny  bool
		isAllow bool
		isAsk   bool
	}{
		{PermissionDeny, true, false, false},
		{PermissionAllow, false, true, false},
		{PermissionAsk, false, false, true},
		{"", false, false, false},
		{"unknown", false, false, false},
	}
	for _, c := range cases {
		r := &FireResult{PermissionDecision: c.perm}
		if r.IsDeny() != c.isDeny {
			t.Errorf("perm=%q IsDeny()=%v, want %v", c.perm, r.IsDeny(), c.isDeny)
		}
		if r.IsAllow() != c.isAllow {
			t.Errorf("perm=%q IsAllow()=%v, want %v", c.perm, r.IsAllow(), c.isAllow)
		}
		if r.IsAsk() != c.isAsk {
			t.Errorf("perm=%q IsAsk()=%v, want %v", c.perm, r.IsAsk(), c.isAsk)
		}
	}
}

func TestFire_HookResultsCaptureOrder(t *testing.T) {
	matchers := []HookMatcher{
		mustCompileMatcher(t, "", `echo '{"reason":"alpha"}'`),
		mustCompileMatcher(t, "", `echo '{"reason":"beta"}'`),
		mustCompileMatcher(t, "", `echo '{"reason":"gamma"}'`),
	}
	mgr := newTestManager(t, HookEventStop, matchers)

	got, _ := mgr.Fire(context.Background(), HookEventStop, "", struct{}{})
	if len(got.HookResults) != 3 {
		t.Fatalf("HookResults len = %d, want 3", len(got.HookResults))
	}
	want := []string{"alpha", "beta", "gamma"}
	for i, w := range want {
		if got.HookResults[i].Decision.Reason != w {
			t.Errorf("HookResults[%d].Reason = %q, want %q",
				i, got.HookResults[i].Decision.Reason, w)
		}
	}
}

func TestFire_EmptyMatcherAlwaysMatches(t *testing.T) {
	// A matcher with empty pattern fires on any primaryIdentifier,
	// including the empty string. Useful for session-level events
	// that don't have a meaningful identifier.
	matchers := []HookMatcher{
		mustCompileMatcher(t, "", `echo '{"reason":"always"}'`),
	}
	mgr := newTestManager(t, HookEventSessionStart, matchers)

	got, _ := mgr.Fire(context.Background(), HookEventSessionStart, "", struct{}{})
	if len(got.HookResults) != 1 {
		t.Errorf("empty matcher should fire on empty identifier; got %d outcomes",
			len(got.HookResults))
	}
}
