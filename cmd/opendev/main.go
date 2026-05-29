// Command opendev is the v1 REPL entry point. Reads one user query per
// line on stdin, runs the ReAct loop against an LLM provider chosen
// from the -model flag (claude-* → Anthropic, otherwise OpenAI or any
// OpenAI-compatible server via -base-url), prints the assistant's
// reply and a per-turn cost line, repeats until EOF or "exit".
//
// This is the v1 closed-loop milestone: model requests tools, the loop
// executes them, results flow back, model gives a final answer.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/ashish-work/opendev-go/internal/agents"
	"github.com/ashish-work/opendev-go/internal/hooks"
	"github.com/ashish-work/opendev-go/internal/provider/router"
	"github.com/ashish-work/opendev-go/internal/runtime/permissions"
	"github.com/ashish-work/opendev-go/internal/session"
	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/tools/bash"
	"github.com/ashish-work/opendev-go/internal/tools/editfile"
	"github.com/ashish-work/opendev-go/internal/tools/listfiles"
	"github.com/ashish-work/opendev-go/internal/tools/readfile"
	"github.com/ashish-work/opendev-go/internal/tools/spawn"
	"github.com/ashish-work/opendev-go/internal/tools/todo"
	"github.com/ashish-work/opendev-go/internal/tools/truncation"
	"github.com/ashish-work/opendev-go/internal/tools/webfetch"
	"github.com/ashish-work/opendev-go/internal/tools/websearch"
	"github.com/ashish-work/opendev-go/internal/tools/writefile"
	"github.com/ashish-work/opendev-go/internal/workflow"
)

// Banner shown on startup. Kept short.
const banner = `opendev v0.2 — type "exit" or Ctrl-D to quit`

// inputBufferSize raises bufio.Scanner's default 64KB cap so long
// paste-ins (e.g. a stack trace) don't get silently truncated.
const inputBufferSize = 1 << 20 // 1 MB

func main() {
	model := flag.String("model", "gpt-4o-mini",
		"Model name. claude-* routes to Anthropic; everything else (including OpenAI-compatible servers via -base-url) routes to OpenAI.")
	maxIter := flag.Int("max-iter", agents.DefaultMaxIterations,
		"Maximum loop iterations per query before giving up.")
	systemPrompt := flag.String("system", "",
		"Override the default system prompt. Empty uses the built-in default.")
	baseURL := flag.String("base-url", "",
		"Provider base URL. Empty uses the selected provider's default; non-empty overrides for proxies or compatible servers.")
	maxContext := flag.Int("max-context", 128_000,
		"Context-window cap in tokens (for budget calibration). 0 disables the usage percentage.")
	flag.Parse()

	envVar := router.EnvVarFor(*model)
	apiKey := os.Getenv(envVar)
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, router.FormatMissingKey(*model))
		os.Exit(1)
	}

	workingDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: get working dir: %v\n", err)
		os.Exit(1)
	}

	// Sweep stale overflow files (>7 days) from previous sessions. Cheap;
	// silent when the dir doesn't exist.
	truncation.CleanupOldFiles()

	// Pick the right provider (openai/anthropic) based on the model name.
	client, err := router.New(*model, *baseURL, apiKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	caller := agents.NewLlmCaller(client, router.PricingFor(*model))

	registry := tools.NewRegistry()
	mustRegister(registry, readfile.New())
	mustRegister(registry, bash.New())
	mustRegister(registry, editfile.New())
	mustRegister(registry, writefile.New())
	mustRegister(registry, listfiles.New())
	mustRegister(registry, todo.New())
	mustRegister(registry, webfetch.New())
	mustRegister(registry, websearch.New())

	// Load hooks from ~/.opendev/settings.json + ./.opendev/settings.json.
	// Missing files are not errors; hooks are opt-in.
	hookSettings, err := hooks.Load(workingDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: load hooks: %v\n", err)
		os.Exit(1)
	}
	var hookManager *hooks.Manager
	if len(hookSettings.Hooks) > 0 {
		hookManager = hooks.NewManager(hookSettings, hooks.NewExecutor(workingDir))
	}

	// Load permissions from the same settings.json files (top-level
	// "permissions" key alongside "hooks"). Missing files / missing
	// key produce a zero Policy that allow-all, preserving the v1
	// default behavior.
	permPolicy, err := permissions.Load(workingDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: load permissions: %v\n", err)
		os.Exit(1)
	}
	if len(permPolicy.Tools) > 0 {
		slog.Info("permissions: loaded policy",
			"tool_entries", len(permPolicy.Tools))
	}

	// Construct the session and fire SessionStart. A deny here exits
	// the binary with the reason — admin-controlled session block.
	sess := session.New(workingDir)
	startResult, err := sess.FireStart(context.Background(), hookManager)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: SessionStart hook: %v\n", err)
		os.Exit(1)
	}
	if startResult.Denied {
		fmt.Fprintf(os.Stderr, "session blocked: %s\n", startResult.Reason)
		os.Exit(1)
	}
	// Apply SessionStart's AdditionalContext to the system prompt.
	effectiveSystemPrompt := *systemPrompt
	if startResult.AdditionalContext != "" {
		if effectiveSystemPrompt != "" {
			effectiveSystemPrompt += "\n\n" + startResult.AdditionalContext
		} else {
			effectiveSystemPrompt = startResult.AdditionalContext
		}
	}

	loop := agents.NewReactLoop(caller, registry, agents.Config{
		Workflow: workflow.Config{
			Execution: workflow.SlotConfig{Model: *model},
		},
		MaxIterations:    *maxIter,
		SystemPrompt:     effectiveSystemPrompt,
		WorkingDir:       workingDir,
		MaxContextTokens: *maxContext,
	})
	loop.Hooks = hookManager
	loop.Permissions = permPolicy

	// Register spawn_subagent AFTER the other tools — it needs the
	// shared registry, caller, and hook manager to construct child
	// loops. The registry pointer means a nested spawn from inside
	// a subagent sees the registry including spawn itself, so
	// recursion (capped at DefaultMaxDepth) Just Works.
	mustRegister(registry, spawn.New(spawn.Config{
		Caller:      caller,
		Registry:    registry,
		Workflow:    workflow.Config{Execution: workflow.SlotConfig{Model: *model}},
		WorkingDir:  workingDir,
		MaxCtx:      *maxContext,
		Hooks:       hookManager,
		Permissions: permPolicy,
	}))

	// Persistent total across REPL turns so the final goodbye can show
	// cumulative cost. Each Run starts with a fresh Tracker for that
	// turn; we accumulate post-turn.
	var totalCost float64
	var totalCalls int64

	// SIGINT installation. We forward Ctrl-C through a channel so each
	// turn can be cancelled WITHOUT killing the process — Ctrl-C
	// during a turn aborts the turn and returns to the prompt.
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT)
	defer signal.Stop(sigs)

	fmt.Println(banner)
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, inputBufferSize), inputBufferSize)

	for {
		fmt.Print(">>> ")
		if !scanner.Scan() {
			break // EOF / Ctrl-D
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if line == "exit" || line == "quit" {
			break
		}

		// UserPromptSubmit hook: gate, transform, or annotate the
		// user's input before it reaches the loop. Deny prints the
		// reason and keeps the user at the prompt.
		submit, err := sess.FirePromptSubmit(context.Background(), hookManager, line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: UserPromptSubmit hook: %v\n", err)
			continue
		}
		if submit.Denied {
			fmt.Fprintf(os.Stderr, "denied by hook: %s\n", submit.Reason)
			continue
		}

		turnCost, calls, replyText, turnErr := runTurn(loop, submit.Prompt, sigs)
		totalCost += turnCost
		totalCalls += calls

		// Stop hook: fire-and-forget telemetry signal for each turn.
		var errStr string
		if turnErr != nil {
			errStr = turnErr.Error()
		}
		sess.FireStop(context.Background(), hookManager, replyText, errStr)
	}

	if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
		fmt.Fprintf(os.Stderr, "input error: %v\n", err)
	}

	// SessionEnd hook: cumulative telemetry. Fire-and-forget.
	sess.FireEnd(context.Background(), hookManager, totalCost, totalCalls)

	fmt.Printf("\ngoodbye. total cost: $%.4f over %d API calls\n",
		totalCost, totalCalls)
}

// runTurn drives one ReactLoop.Run for the given user query, prints
// the assistant's reply + a per-turn status line, and handles SIGINT
// by cancelling the per-turn ctx (NOT the program).
//
// Returns: cost incurred this turn, API call count, the assistant's
// reply text (for Stop hook payload), and any error (for Stop hook
// payload).
func runTurn(loop *agents.ReactLoop, query string, sigs <-chan os.Signal) (float64, int64, string, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Forward one SIGINT to cancel this turn. Multiple Ctrl-Cs in one
	// turn won't pile up — we re-arm in the next call to runTurn.
	go func() {
		select {
		case <-sigs:
			cancel()
		case <-ctx.Done():
			// Turn finished normally; the goroutine exits.
		}
	}()

	result, tracker, err := loop.Run(ctx, query)

	// Status line first, regardless of error path — so the user can
	// see what was consumed even when the turn failed.
	//
	// in     = cumulative prompt tokens billed this turn (sum across iters)
	// cached = subset of `in` that hit OpenAI's prefix cache (10% price)
	// out    = cumulative completion tokens billed this turn
	// ctx    = calibrator's reported (ground truth) / estimated (with delta)
	//          / pct of MaxContextTokens
	// cost   = USD this turn (already discounts cached tokens)
	fmt.Fprintf(os.Stderr, "[iter=%d in=%d cached=%d out=%d ctx=%d/%d (%.1f%%) cost=%s]\n",
		tracker.CallCount,
		tracker.TotalInputTokens,
		tracker.TotalCacheReadTokens,
		tracker.TotalOutputTokens,
		result.Budget.Reported,
		result.Budget.Estimated,
		result.Budget.UsagePct*100,
		tracker.FormatCost(),
	)

	switch {
	case err == nil:
		fmt.Println(result.Content)
	case errors.Is(err, agents.ErrInterrupted):
		fmt.Fprintln(os.Stderr, "(interrupted)")
	case errors.Is(err, agents.ErrMaxIterations):
		fmt.Fprintln(os.Stderr, "(hit max iterations; partial conversation kept)")
		if result.Content != "" {
			fmt.Println(result.Content)
		}
	case errors.Is(err, agents.ErrDoomLoop):
		fmt.Fprintln(os.Stderr, "(doom loop detected — model was repeating itself; halted)")
		fmt.Fprintln(os.Stderr, "  try rephrasing your request or breaking it into smaller steps")
	default:
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
	}

	return tracker.TotalCostUSD, tracker.CallCount, result.Content, err
}

// mustRegister fails fast on registry-wiring bugs. We only call this
// at startup with known-good tools, so a failure means a programming
// mistake (duplicate name, empty name).
func mustRegister(reg *tools.Registry, tool tools.Tool) {
	if err := reg.Register(tool); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: register tool: %v\n", err)
		os.Exit(1)
	}
}
