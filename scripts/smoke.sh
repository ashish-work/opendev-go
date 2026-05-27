#!/usr/bin/env bash
#
# scripts/smoke.sh — End-to-end smoke tests for the v1 opendev agent.
#
# Runs ./cmd/opendev against the real OpenAI API and verifies the
# closed loop works for each tool + a few multi-step scenarios.
#
# DESIGN: assertions check OBSERVABLE SIDE-EFFECTS, never exact LLM
# wording. The model phrases things differently each run, but:
#   - tool dispatches show up as iter≥2 in the status line
#   - file edits leave the file on disk in a known state
#   - bash commands' output flows through to the final reply
#
# USAGE:
#   OPENAI_API_KEY=sk-... ./scripts/smoke.sh
#   OPENAI_API_KEY=sk-... ./scripts/smoke.sh --model gpt-4o-mini
#   OPENAI_API_KEY=sk-... ./scripts/smoke.sh --keep    # don't clean up tmpdir
#
# EXIT: 0 if all scenarios pass; non-zero on any failure.
#
# COST: roughly $0.001-0.01 per full run on gpt-4o-mini.

set -euo pipefail

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MODEL="${MODEL:-gpt-4o-mini}"
KEEP_TMPDIR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --keep)  KEEP_TMPDIR=1; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FATAL: OPENAI_API_KEY must be set" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# cd to the repo root so relative paths work regardless of where the
# script is invoked from.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Build the binary fresh so the test reflects the current source.
echo "==> building opendev"
go build -o ./opendev ./cmd/opendev

# All scenarios run from this fixture dir.
WORK_DIR="$(mktemp -d -t opendev-smoke.XXXXXX)"
echo "==> fixture dir: $WORK_DIR"

cleanup() {
  if [[ $KEEP_TMPDIR -eq 1 ]]; then
    echo "==> keeping fixture: $WORK_DIR"
  else
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT

# Colors — fall back to plain text if not a TTY.
if [[ -t 1 ]]; then
  GREEN=$'\033[32m'; RED=$'\033[31m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
  GREEN=""; RED=""; DIM=""; RESET=""
fi

PASS=0
FAIL=0
FAILED_TESTS=()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# run_agent <query> [<extra-binary-args>...]
# Pipes the query as one REPL turn and prints the COMBINED stdout+stderr.
# Always returns success — caller inspects the output.
run_agent() {
  local query="$1"; shift
  # Append a blank line + EOF marker so the REPL exits cleanly after
  # one turn. The agent treats "" as a no-op iteration; EOF exits.
  (
    cd "$WORK_DIR"
    printf '%s\n' "$query" | "$REPO_ROOT/opendev" -model "$MODEL" "$@" 2>&1 || true
  )
}

# assert_iter_at_least <output> <min-iterations> <scenario-name>
# Parses "[iter=N ..." from the status line and asserts N >= min.
# A tool dispatch always produces iter ≥ 2 (1 to call the tool, 1 to
# respond after seeing the result).
assert_iter_at_least() {
  local output="$1" min_iter="$2" name="$3"
  local iter
  iter=$(printf '%s\n' "$output" | grep -oE '\[iter=[0-9]+' | tail -1 | grep -oE '[0-9]+' || echo "0")
  if [[ -z "$iter" || "$iter" -lt "$min_iter" ]]; then
    fail "$name" "expected iter ≥ $min_iter, got $iter
--- output ---
$output
---"
    return 1
  fi
  return 0
}

# assert_output_contains <output> <needle> <scenario-name>
assert_output_contains() {
  local output="$1" needle="$2" name="$3"
  if ! grep -qF "$needle" <<<"$output"; then
    fail "$name" "output missing '$needle'
--- output ---
$output
---"
    return 1
  fi
  return 0
}

# assert_file_contains <path> <needle> <scenario-name>
assert_file_contains() {
  local path="$1" needle="$2" name="$3"
  if [[ ! -f "$path" ]]; then
    fail "$name" "expected file $path to exist"
    return 1
  fi
  if ! grep -qF "$needle" "$path"; then
    fail "$name" "file $path does not contain '$needle'. Actual contents:
$(cat "$path")"
    return 1
  fi
  return 0
}

# assert_file_not_contains <path> <needle> <scenario-name>
assert_file_not_contains() {
  local path="$1" needle="$2" name="$3"
  if grep -qF "$needle" "$path"; then
    fail "$name" "file $path still contains '$needle' (expected removed). Actual:
$(cat "$path")"
    return 1
  fi
  return 0
}

pass() {
  PASS=$((PASS + 1))
  printf '%s✓%s %s\n' "$GREEN" "$RESET" "$1"
}

fail() {
  FAIL=$((FAIL + 1))
  FAILED_TESTS+=("$1")
  printf '%s✗%s %s\n' "$RED" "$RESET" "$1"
  printf '%s%s%s\n' "$DIM" "$2" "$RESET"
}

# -----------------------------------------------------------------------------
# Scenarios
# -----------------------------------------------------------------------------

# Scenario 1 — bash tool: model calls bash, output flows through.
scenario_bash() {
  local name="bash dispatch"
  echo "==> $name"
  local out
  out=$(run_agent "use the bash tool to print the exact string SMOKE_TOKEN_42")
  assert_iter_at_least "$out" 2 "$name" || return
  assert_output_contains "$out" "SMOKE_TOKEN_42" "$name" || return
  pass "$name"
}

# Scenario 2 — read_file tool: model reads a fixture, reports contents.
scenario_read() {
  local name="read_file dispatch"
  echo "==> $name"
  local file="$WORK_DIR/secret.txt"
  echo "the magic word is BANANAPHONE" > "$file"

  local out
  out=$(run_agent "use read_file to read secret.txt and tell me the magic word")
  assert_iter_at_least "$out" 2 "$name" || return
  assert_output_contains "$out" "BANANAPHONE" "$name" || return
  pass "$name"
}

# Scenario 3 — edit_file tool: model edits, file changes on disk.
scenario_edit() {
  local name="edit_file dispatch (exact match)"
  echo "==> $name"
  local file="$WORK_DIR/config.txt"
  printf 'mode=alpha\ntimeout=30\nretries=3\n' > "$file"

  local out
  out=$(run_agent "use edit_file to change mode=alpha to mode=beta in config.txt")
  assert_iter_at_least "$out" 2 "$name" || return
  assert_file_contains "$file" "mode=beta" "$name" || return
  assert_file_not_contains "$file" "mode=alpha" "$name" || return
  pass "$name"
}

# Scenario 4 — fuzzy match: model edits with whitespace drift.
# Tests that the fuzzy matcher chain (LineTrimmed at minimum) corrects
# the model's input even when it gets indentation slightly wrong.
scenario_fuzzy_edit() {
  local name="edit_file (fuzzy: indent drift)"
  echo "==> $name"
  local file="$WORK_DIR/code.go"
  printf 'func add(a, b int) int {\n\treturn a + b\n}\n' > "$file"

  # Ask the model to change "return a + b" to "return a * b". The
  # file has the line tab-indented; the model might or might not
  # include the tab. Either way the matcher should land the edit.
  local out
  out=$(run_agent "use edit_file to change the return statement in code.go from 'return a + b' to 'return a * b'")
  assert_iter_at_least "$out" 2 "$name" || return
  assert_file_contains "$file" "return a * b" "$name" || return
  assert_file_not_contains "$file" "return a + b" "$name" || return
  pass "$name"
}

# Scenario 5 — multi-step: read then edit based on what was read.
# Exercises the loop closing more than once per query.
scenario_multistep() {
  local name="multi-step: read then edit"
  echo "==> $name"
  local file="$WORK_DIR/notes.txt"
  printf 'item one\nitem two\n' > "$file"

  local out
  out=$(run_agent "read notes.txt with read_file, then use edit_file to append a line 'item three' by changing 'item two' to 'item two\nitem three'")
  assert_iter_at_least "$out" 3 "$name" || return  # read + edit + summary
  assert_file_contains "$file" "item three" "$name" || return
  assert_file_contains "$file" "item one" "$name" || return
  assert_file_contains "$file" "item two" "$name" || return
  pass "$name"
}

# Scenario 6 — error recovery: file doesn't exist, model should
# get the tool error as an observation and respond gracefully (not crash).
scenario_error_recovery() {
  local name="error recovery: file not found"
  echo "==> $name"

  local out
  out=$(run_agent "use read_file to read nonexistent.txt and tell me what's in it")
  # Loop must exit cleanly (no panic, no crash). We don't assert on
  # the model's wording — just that it ran and produced some output.
  assert_iter_at_least "$out" 2 "$name" || return
  # Some response should follow the failed tool call.
  if [[ -z "$out" ]]; then
    fail "$name" "empty output"
    return
  fi
  pass "$name"
}

# Scenario 7 — REPL hygiene: missing API key, expected exit code.
scenario_no_api_key() {
  local name="REPL refuses to start without OPENAI_API_KEY"
  echo "==> $name"
  local out
  local exit_code=0
  out=$(OPENAI_API_KEY="" "$REPO_ROOT/opendev" 2>&1) || exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    fail "$name" "expected non-zero exit, got 0
output: $out"
    return
  fi
  assert_output_contains "$out" "OPENAI_API_KEY" "$name" || return
  pass "$name"
}

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

START_TIME=$(date +%s)

scenario_no_api_key
scenario_bash
scenario_read
scenario_edit
scenario_fuzzy_edit
scenario_multistep
scenario_error_recovery

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo
echo "================================================"
printf 'smoke tests done in %ds: %s%d passed%s, %s%d failed%s\n' \
  "$ELAPSED" "$GREEN" "$PASS" "$RESET" "$RED" "$FAIL" "$RESET"

if [[ $FAIL -gt 0 ]]; then
  echo
  echo "failed scenarios:"
  for t in "${FAILED_TESTS[@]}"; do
    printf '  - %s\n' "$t"
  done
  exit 1
fi

exit 0
