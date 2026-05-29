// Command opendev-tui is the Bubble Tea-based interactive front-end
// for the agent. It runs alongside cmd/opendev (the REPL) — both
// binaries operate against the same internal/agents core, so a
// reader of the repo can study either presentation layer in
// isolation.
//
// This package is the binary's wiring layer: parse flags, validate
// the API key for the right provider, build the Provider / Caller /
// Registry / ReactLoop via internal/provider/router, hand the loop to
// internal/tui's Run, surface its exit code. Provider construction is
// delegated to the router (claude-* → Anthropic, everything else →
// OpenAI) so cmd/opendev and this binary share one source of truth.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"

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
	"github.com/ashish-work/opendev-go/internal/tui"
	"github.com/ashish-work/opendev-go/internal/workflow"
)

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

	// Load hooks from settings.json (user + project). Missing files
	// are not errors. Construct the manager only when at least one
	// hook is registered so nil-Manager paths stay fast.
	hookSettings, err := hooks.Load(workingDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: load hooks: %v\n", err)
		os.Exit(1)
	}
	var hookManager *hooks.Manager
	if len(hookSettings.Hooks) > 0 {
		hookManager = hooks.NewManager(hookSettings, hooks.NewExecutor(workingDir))
	}

	// Load permissions from the same settings.json files. Missing
	// files / missing "permissions" key produce an allow-everything
	// zero Policy.
	permPolicy, err := permissions.Load(workingDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: load permissions: %v\n", err)
		os.Exit(1)
	}
	if len(permPolicy.Tools) > 0 {
		slog.Info("permissions: loaded policy",
			"tool_entries", len(permPolicy.Tools))
	}

	// Construct session + fire SessionStart. Deny exits with the
	// reason; allow's AdditionalContext appends to the system prompt.
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

	// Plan-of-record injection: when the user has a todos.json
	// under ~/.opendev/, every LLM request will see the current
	// plan as a synthetic system message. Falling back to no
	// injection on DefaultPath failure keeps the binary usable on
	// systems without a writable $HOME.
	if todoPath, err := todo.DefaultPath(); err == nil {
		loop.TodoStore = todo.NewStore(todoPath)
	} else {
		slog.Info("todo: skipping plan injection (no default path)",
			"err", err)
	}

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

	runErr := tui.Run(loop, *model, sess, hookManager)

	// SessionEnd hook (fire-and-forget) — runs even if the TUI
	// reported an error so audit hooks still see end-of-session.
	// Totals would come from the model's tracker but the TUI
	// doesn't surface them through Run yet; pass zeros for now.
	sess.FireEnd(context.Background(), hookManager, 0, 0)

	if runErr != nil {
		fmt.Fprintln(os.Stderr, "opendev-tui:", runErr)
		os.Exit(1)
	}
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
