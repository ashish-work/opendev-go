// Command opendev-tui is the Bubble Tea-based interactive front-end
// for the agent. It runs alongside cmd/opendev (the REPL) — both
// binaries operate against the same internal/agents core, so a
// reader of the repo can study either presentation layer in
// isolation.
//
// This package is the binary's wiring layer: parse flags, validate
// the API key, build the Provider / Caller / Registry / ReactLoop,
// hand the loop to internal/tui's Run, surface its exit code. The
// loop construction mirrors cmd/opendev's setup — that duplication
// is small (~25 lines) and intentional. If a third caller emerges,
// extracting into internal/wiring/ would be the right move.
package main

import (
	"flag"
	"fmt"
	"os"

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
	"github.com/ashish-work/opendev-go/internal/tui"
	"github.com/ashish-work/opendev-go/internal/workflow"
)

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

	if err := tui.Run(loop, *model); err != nil {
		fmt.Fprintln(os.Stderr, "opendev-tui:", err)
		os.Exit(1)
	}
}

// pricingFor returns Pricing for known models. Unknown models fall
// back to zero Pricing — token counts still flow, cost stays $0.
// The model name is matched as a prefix to handle "gpt-4o-2024-08-06"
// style aliases.
//
// Duplicated from cmd/opendev/main.go on purpose. When a third
// binary wants this table, lift it to internal/cost/pricing/ or
// similar.
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
		if startsWith(model, entry.prefix) {
			return entry.pricing
		}
	}
	return cost.Pricing{}
}

// startsWith is a tiny prefix check pulled out so this file doesn't
// import "strings" only for HasPrefix. Same intent as
// strings.HasPrefix; avoids the import for clarity in a thin glue
// file.
func startsWith(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
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
