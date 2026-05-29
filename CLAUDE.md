# opendev-go

A terminal-native AI coding agent built in idiomatic Go. The model
issues tool calls, the loop executes them, results flow back, the
model continues until it produces a final answer.

This repository is structured as a teaching journey — read the commit
history from oldest to newest to follow how each layer was built. Each
commit is a self-contained step that compiles and passes tests.

## v1 scope

A minimal closed loop with three tools:

- `read_file` — read text files, with optional offset/limit windowing
- `bash` — execute foreground shell commands with timeout + output spillover
- `edit_file` — modify files with a 9-pass fuzzy matcher

Plus the runtime machinery: an OpenAI-compatible provider adapter, a
ReAct loop with iteration cap and SIGINT cancellation, an immutable
cost tracker with tiered pricing, an API-anchored token-budget
calibrator, a doom-loop cycle detector, and prompt-caching support
via a stable long system prompt. UI is plain stdin/stdout REPL.

## Design conventions

These rules show up in nearly every package; keep them in mind when
reading and writing code here.

- **Immutability.** Mutating state in place is the exception, not
  the default. `cost.Tracker`, `budget.Calibrator`, and `workflow.Config`
  all use value receivers and return new values from update methods.
  This makes data flow through the agent loop easy to follow and
  catches accidental shared-state bugs at compile time.

- **Small, focused files.** Each package solves one concern. When a
  file approaches ~400 lines or starts mixing concerns, it gets split
  into siblings. The `tools/editfile/match/` subpackage isolates
  fuzzy-matching from the rest of `edit_file`; the `agents/doomloop/`
  and `agents/summarize/` subpackages isolate algorithms from the
  loop they serve.

- **Sentinel errors + wrapping.** Each package owns its error sentinels
  (`agents.ErrLLM`, `tools.ErrToolNotFound`, etc.). Callers wrap with
  `fmt.Errorf("...: %w", ...)` and match via `errors.Is`. Typed errors
  with data (e.g. `APIError{Status, Message}`) use `errors.As`.

- **Tagged structs for sum types.** Where one of several variants is
  expected (a content block, a turn result, a tool category), the
  shape is a struct with a `Kind` field naming the variant + a union
  of fields where only the active variant's fields are meaningful.
  This is the idiomatic Go alternative to tagged unions in other
  languages.

- **`context.Context` everywhere.** Cancellation propagates through
  every long-running call. SIGINT in the REPL cancels the per-turn
  ctx without killing the process. Tool execution honors both the
  outer ctx (user-driven cancellation) and an inner per-tool timeout.

## Package layout

```
opendev-go/
├── go.mod
├── cmd/opendev/             # CLI entry point: REPL (v1; minimal surface)
├── cmd/opendev-tui/         # CLI entry point: Bubble Tea TUI — thin glue
│                            #   that delegates to internal/tui
├── internal/
│   ├── provider/            # Provider interface + normalized types
│   │   ├── openai/          # OpenAI-compatible Chat Completions adapter
│   │   ├── anthropic/       # Anthropic Messages API adapter
│   │   └── router/          # model→provider routing + shared pricing table
│   ├── tools/               # Tool interface + registry
│   │   ├── readfile/        # read_file
│   │   ├── bash/            # foreground shell command runner
│   │   ├── editfile/        # edit_file
│   │   │   └── match/       # 9-pass fuzzy matching chain
│   │   └── truncation/      # output-spillover-to-disk helper
│   ├── agents/              # ReAct loop, prompt composition, LLM caller
│   │   ├── doomloop/        # cycle detector
│   │   └── summarize/       # rule-based tool-result summarizer
│   ├── tui/                 # Bubble Tea Model/Update/View (powers
│   │                        #   cmd/opendev-tui — v2 Phase 1.5+)
│   ├── hooks/               # lifecycle hook system (Phase 6)
│   ├── workflow/            # typed model-role slots (Execution/Thinking/...)
│   ├── budget/              # token heuristic + API-anchored calibrator
│   ├── cost/                # immutable cost tracker
│   └── session/             # session state (placeholder for now)
└── scripts/smoke.sh         # end-to-end smoke test against a real provider
```

The `internal/` boundary enforces encapsulation — only `cmd/opendev`,
`cmd/opendev-tui`, and tests inside the module can import these
packages.

## Dependencies

Stdlib-first: `net/http`, `encoding/json`, `log/slog`, `flag`,
`os/exec`, `context`, `sync`, `errors`, `testing`.

Third-party additions (each recorded with a justification):

- `github.com/sergi/go-diff` — generates unified-diff output for
  `edit_file` results. BSD-licensed, no transitive deps.

- `golang.org/x/net/html` — DOM-based HTML parser used by
  `web_fetch` to convert HTML responses to plain text. Go-team
  maintained, BSD-licensed, no transitive runtime deps. Picked
  over a regex-based approach because real-world HTML has
  unclosed tags, nested elements, and entity escapes that regex
  patterns handle poorly. Pinned to v0.43.0 to keep the module's
  Go directive at 1.24.1 (newer x/net releases require Go 1.25+).

- `github.com/charmbracelet/bubbletea` — Elm-style TUI framework
  (Model/Update/View) powering `cmd/opendev-tui`. MIT-licensed,
  mature, used by the GitHub CLI, glow, and many others. The
  canonical Go answer for terminal UI; the alternative (raw
  tcell/termbox) would mean reinventing message dispatch, focus
  management, and alt-screen handling ourselves.

- `github.com/charmbracelet/bubbles` — premade Bubble Tea widgets
  (`textarea`, `viewport`, `spinner`, `list`). Used as building
  blocks for `cmd/opendev-tui`. Same author/ecosystem; no
  transitive bloat beyond what bubbletea pulls in.

- `github.com/charmbracelet/lipgloss` — declarative terminal
  styling (borders, colors, padding, alignment, layout). Used
  by `cmd/opendev-tui` for every rendered panel.

  All three Charm libs together pull in ~10 transitive deps that
  are standard for any Bubble Tea program (ANSI rendering,
  Unicode width, termenv color detection). Documented here so the
  set doesn't drift unnoticed.

Adding another third-party dep requires a recorded decision in this
file first.

## Build, vet, test

```bash
go build ./...
go vet ./...
go test -race ./...
```

The race detector should be on for every test run; concurrent state
appears in the registry, the per-file edit mutex map, and the loop's
ctx-cancellation goroutines.

## Running the REPL

```bash
export OPENAI_API_KEY=sk-...
go run ./cmd/opendev
```

Flags:

- `-model` (default `gpt-4o-mini`)
- `-max-iter` (default 25)
- `-max-context` (default 128 000)
- `-system` to override the built-in system prompt
- `-base-url` to point at an OpenAI-compatible endpoint

## Smoke test

`scripts/smoke.sh` exercises the closed loop against a real provider.
Costs roughly $0.001-0.01 per full run on `gpt-4o-mini`:

```bash
OPENAI_API_KEY=sk-... ./scripts/smoke.sh
```

Seven scenarios cover each tool, multi-step flows, and error recovery.
