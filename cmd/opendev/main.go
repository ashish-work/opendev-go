// Command opendev is the v1 REPL entry point. Reads one user query per
// line on stdin, runs the ReAct loop against an OpenAI-compatible
// provider, prints the assistant's reply and a per-turn cost line,
// repeats until EOF or "exit".
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
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/ashish-work/opendev-go/internal/agents"
	"github.com/ashish-work/opendev-go/internal/cost"
	"github.com/ashish-work/opendev-go/internal/provider/openai"
	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/tools/bash"
	"github.com/ashish-work/opendev-go/internal/tools/editfile"
	"github.com/ashish-work/opendev-go/internal/tools/listfiles"
	"github.com/ashish-work/opendev-go/internal/tools/readfile"
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
		"OpenAI model name (or any OpenAI-compatible model).")
	maxIter := flag.Int("max-iter", agents.DefaultMaxIterations,
		"Maximum loop iterations per query before giving up.")
	systemPrompt := flag.String("system", "",
		"Override the default system prompt. Empty uses the built-in default.")
	baseURL := flag.String("base-url", openai.DefaultBaseURL,
		"Provider base URL (override for proxies / OpenAI-compatible servers).")
	maxContext := flag.Int("max-context", 128_000,
		"Context-window cap in tokens (for budget calibration). 0 disables the usage percentage.")
	flag.Parse()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "error: OPENAI_API_KEY must be set")
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

	// Wire the closed loop: openai.Client → LlmCaller → ReactLoop.
	client := openai.NewClient(apiKey)
	client.Adapter.BaseURL = *baseURL
	caller := agents.NewLlmCaller(client, pricingFor(*model))

	registry := tools.NewRegistry()
	mustRegister(registry, readfile.New())
	mustRegister(registry, bash.New())
	mustRegister(registry, editfile.New())
	mustRegister(registry, writefile.New())
	mustRegister(registry, listfiles.New())
	mustRegister(registry, todo.New())
	mustRegister(registry, webfetch.New())
	mustRegister(registry, websearch.New())

	loop := agents.NewReactLoop(caller, registry, agents.Config{
		Workflow: workflow.Config{
			Execution: workflow.SlotConfig{Model: *model},
		},
		MaxIterations:    *maxIter,
		SystemPrompt:     *systemPrompt,
		WorkingDir:       workingDir,
		MaxContextTokens: *maxContext,
	})

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

		turnCost, calls := runTurn(loop, line, sigs)
		totalCost += turnCost
		totalCalls += calls
	}

	if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
		fmt.Fprintf(os.Stderr, "input error: %v\n", err)
	}

	fmt.Printf("\ngoodbye. total cost: $%.4f over %d API calls\n",
		totalCost, totalCalls)
}

// runTurn drives one ReactLoop.Run for the given user query, prints
// the assistant's reply + a per-turn status line, and handles SIGINT
// by cancelling the per-turn ctx (NOT the program).
//
// Returns the cost incurred and number of API calls for this turn so
// main can accumulate session totals.
func runTurn(loop *agents.ReactLoop, query string, sigs <-chan os.Signal) (float64, int64) {
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

	return tracker.TotalCostUSD, tracker.CallCount
}

// pricingFor returns Pricing for known models. Unknown models fall
// back to zero Pricing — token counts still flow, cost stays $0.
//
// The model name is matched as a prefix to handle "gpt-4o-2024-08-06"
// style aliases.
func pricingFor(model string) cost.Pricing {
	table := []struct {
		prefix  string
		pricing cost.Pricing
	}{
		{"gpt-4o-mini", cost.Pricing{InputPricePerMillion: 0.15, OutputPricePerMillion: 0.60}},
		{"gpt-4o", cost.Pricing{InputPricePerMillion: 2.50, OutputPricePerMillion: 10.00}},
		{"gpt-4-turbo", cost.Pricing{InputPricePerMillion: 10.00, OutputPricePerMillion: 30.00}},
		{"gpt-3.5-turbo", cost.Pricing{InputPricePerMillion: 0.50, OutputPricePerMillion: 1.50}},
	}
	for _, entry := range table {
		if strings.HasPrefix(model, entry.prefix) {
			return entry.pricing
		}
	}
	return cost.Pricing{}
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
