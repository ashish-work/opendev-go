// Package session represents one REPL/TUI run from launch to exit.
// A Session carries identity (a short opaque ID + working directory
// + start time) and the four lifecycle-hook firing helpers both
// binaries use so they stay in lockstep on the user-facing hook
// contract.
//
// Phase 9 will extend this package with on-disk persistence;
// for v2 Phase 6 it's the in-memory session metadata + hook glue.
package session

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/ashish-work/opendev-go/internal/hooks"
)

// Session is the per-run identity record. Constructed once at
// binary startup and carried through every lifecycle event.
type Session struct {
	// ID is the opaque session identifier in
	// sess_<hex8>_<unix> shape. Random hex for uniqueness +
	// unix-second suffix for human-readable ordering in logs.
	ID string

	// WorkingDir is the binary's CWD at startup. Sent to hooks via
	// OPENDEV_WORKING_DIR env var and in the SessionStartPayload.
	WorkingDir string

	// StartedAt is the time New was called. Used for session
	// duration reporting in SessionEnd.
	StartedAt time.Time
}

// New constructs a Session for the given working directory. The ID
// is randomly generated; crypto/rand failures fall back to a
// deterministic-from-time fallback so a startup failure here can't
// crash the binary just for an audit ID.
func New(workingDir string) *Session {
	return &Session{
		ID:         generateID(time.Now()),
		WorkingDir: workingDir,
		StartedAt:  time.Now(),
	}
}

// generateID produces the sess_<hex8>_<unix> identifier. Exported
// only for tests; production code uses New.
func generateID(now time.Time) string {
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		// Fall back to a time-derived hex so we still produce an
		// ID. Collisions within a millisecond on a single binary
		// are vanishingly rare and not worth crashing for.
		nano := now.UnixNano()
		b[0] = byte(nano)
		b[1] = byte(nano >> 8)
		b[2] = byte(nano >> 16)
		b[3] = byte(nano >> 24)
	}
	return fmt.Sprintf("sess_%s_%d", hex.EncodeToString(b[:]), now.Unix())
}

// StartResult is the outcome of FireStart. The binary applies
// AdditionalContext by appending it to the system prompt before
// building ReactLoop; Denied causes a clean exit with the reason
// shown.
type StartResult struct {
	// AdditionalContext is what SessionStart hooks wanted to add
	// to the system prompt. Empty when no hook contributed.
	AdditionalContext string

	// Denied is true when any SessionStart hook returned a deny
	// decision. The binary should exit with the reason.
	Denied bool

	// Reason carries the deny message (or any non-empty Reason
	// when Denied is true). Empty when not denied.
	Reason string
}

// FireStart runs the SessionStart hook. nil manager is a no-op
// (returns the zero StartResult). Errors here are reserved for ctx
// cancellation — per-hook infrastructure failures are swallowed by
// the manager and only surface as logged warnings.
func (s *Session) FireStart(ctx context.Context, mgr *hooks.Manager) (StartResult, error) {
	if mgr == nil {
		return StartResult{}, nil
	}
	fire, err := mgr.Fire(ctx, hooks.HookEventSessionStart, "", hooks.SessionStartPayload{
		SessionID:  s.ID,
		WorkingDir: s.WorkingDir,
	})
	if err != nil {
		return StartResult{}, err
	}
	out := StartResult{
		AdditionalContext: fire.AdditionalContext,
	}
	if fire.IsDeny() {
		out.Denied = true
		out.Reason = fire.Reason
	}
	return out, nil
}

// PromptSubmitResult is the outcome of FirePromptSubmit. Prompt is
// the (possibly modified) string the binary should pass to
// loop.Run. Denied means the binary should skip the turn and
// surface Reason to the user.
type PromptSubmitResult struct {
	// Prompt is the effective text after applying any
	// AdditionalContext (prepended) and UpdatedInput (replaces
	// the prompt entirely). Equal to the input on the nil-manager
	// path or when no hook touched it.
	Prompt string

	// Denied is true when any UserPromptSubmit hook returned a
	// deny decision. The binary should NOT call loop.Run; the
	// REPL stays at the prompt, the TUI shows a notice and stays
	// idle.
	Denied bool

	// Reason is the deny message. Empty when not denied.
	Reason string
}

// FirePromptSubmit runs the UserPromptSubmit hook with the given
// prompt. The primary identifier is the prompt text itself so a
// matcher can match on content (e.g., regex "^@(.+)$" to gate
// magic-prefix commands).
//
// UpdatedInput, when present, is decoded as a UserPromptSubmitPayload
// — the hook emits {"updatedInput": {"prompt": "new text"}} to
// rewrite. Invalid JSON or missing Prompt field falls back to the
// original with a warning log.
//
// nil manager is a no-op (returns the input prompt unchanged).
func (s *Session) FirePromptSubmit(
	ctx context.Context,
	mgr *hooks.Manager,
	prompt string,
) (PromptSubmitResult, error) {
	if mgr == nil {
		return PromptSubmitResult{Prompt: prompt}, nil
	}
	fire, err := mgr.Fire(ctx, hooks.HookEventUserPromptSubmit, prompt,
		hooks.UserPromptSubmitPayload{Prompt: prompt})
	if err != nil {
		return PromptSubmitResult{Prompt: prompt}, err
	}
	if fire.IsDeny() {
		return PromptSubmitResult{
			Prompt: prompt,
			Denied: true,
			Reason: fire.Reason,
		}, nil
	}

	effective := prompt
	if len(fire.UpdatedInput) > 0 {
		var rewritten hooks.UserPromptSubmitPayload
		if err := json.Unmarshal(fire.UpdatedInput, &rewritten); err != nil {
			slog.Warn("session: invalid UpdatedInput from UserPromptSubmit hook; falling back",
				"error", err)
		} else if rewritten.Prompt != "" {
			effective = rewritten.Prompt
		}
	}
	if fire.AdditionalContext != "" {
		effective = fire.AdditionalContext + "\n\n" + effective
	}

	return PromptSubmitResult{Prompt: effective}, nil
}

// FireStop fires the Stop hook with the turn outcome. Fire-and-
// forget — there's nothing useful to gate at "turn complete," so
// the hook is a telemetry/observability signal. Errors are logged
// but not returned; the agent never pauses on Stop.
//
// resultText is the assistant's final reply (empty on error
// paths); errStr is the error string (empty on success).
func (s *Session) FireStop(
	ctx context.Context,
	mgr *hooks.Manager,
	resultText, errStr string,
) {
	if mgr == nil {
		return
	}
	_, err := mgr.Fire(ctx, hooks.HookEventStop, "", hooks.StopPayload{
		Result: resultText,
		Error:  errStr,
	})
	if err != nil {
		slog.Warn("session: Stop hook errored", "session_id", s.ID, "error", err)
	}
}

// FireEnd fires the SessionEnd hook with cumulative totals.
// Fire-and-forget like Stop; used for session-level telemetry,
// log shipping, cost reporting.
func (s *Session) FireEnd(
	ctx context.Context,
	mgr *hooks.Manager,
	costUSD float64,
	callCount int64,
) {
	if mgr == nil {
		return
	}
	_, err := mgr.Fire(ctx, hooks.HookEventSessionEnd, "", hooks.SessionEndPayload{
		SessionID: s.ID,
		CostUSD:   costUSD,
		CallCount: callCount,
	})
	if err != nil {
		slog.Warn("session: SessionEnd hook errored", "session_id", s.ID, "error", err)
	}
}
