// Command opendev-tui is the Bubble Tea-based interactive front-end
// for the agent. It runs alongside cmd/opendev (the REPL) — both
// binaries operate against the same internal/agents core, so a
// reader of the repo can study either presentation layer in
// isolation.
//
// This package is intentionally tiny — it's pure glue. All TUI logic
// lives in internal/tui. The split keeps cmd/<binary>/ packages
// focused on dependency wiring and process exit codes, which is the
// Go convention.
package main

import (
	"fmt"
	"os"

	"github.com/ashish-work/opendev-go/internal/tui"
)

func main() {
	if err := tui.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "opendev-tui:", err)
		os.Exit(1)
	}
}
