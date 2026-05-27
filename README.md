# opendev-go

A terminal-native AI coding agent in idiomatic Go. The model issues
tool calls, a loop executes them, results flow back, the model
continues until it produces a final answer.

## How to read this repository

This repo is structured as a teaching journey. The commit history is
the curriculum. Reading the commits in chronological order — from the
oldest to the newest — walks you through the build of a production-
grade agent step by step.

```bash
git log --oneline --reverse
```

Each commit:

- Compiles cleanly (`go build ./...`).
- Passes whatever tests exist at that point (`go test -race ./...`).
- Solves one specific problem, explained in the commit message body.
- Builds incrementally on the previous commit.

To study a specific step, check out that commit and explore:

```bash
git checkout <commit-sha>
go build ./...
go test -race ./...
```

When you're done exploring, return to `main`:

```bash
git checkout main
```

## Running the agent

```bash
export OPENAI_API_KEY=sk-...
go run ./cmd/opendev
```

Flags are documented in `cmd/opendev/main.go`. The defaults work for
`gpt-4o-mini`.

## Project layout

See `CLAUDE.md` for the package map, design conventions, and the
external dependency policy.

## Smoke test

```bash
OPENAI_API_KEY=sk-... ./scripts/smoke.sh
```

Runs seven end-to-end scenarios against a real provider. Cost is
roughly $0.001–$0.01 per full run on `gpt-4o-mini`.
