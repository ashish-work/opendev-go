// Package todo implements the todo tool — a stateful, file-backed
// list the model uses to plan its own multi-step work.
//
// The persistence story is the part of this tool worth studying. Every
// other tool in the codebase is stateless (operates on args plus the
// filesystem, then forgets). todo keeps its own state at
// ~/.opendev/todos.json so the plan survives REPL restarts AND context
// compaction (Phase 5) — the plan lives on disk, not in the prompt.
//
// Concurrency: writes go through filelock.AtomicWrite for crash-safety
// and through filelock.For to serialize against concurrent callers.
// Reads are plain os.ReadFile; the lock guards the read-modify-write
// cycle in todo.Execute, not Load on its own.
//
// State semantics: every State method is a pure function — it takes a
// receiver and arguments and returns a new State. No in-place mutation.
// This matches the cost.Tracker and budget.Calibrator patterns
// established in v1; bugs from accidental shared-state aliasing are
// catchable at compile time when State values are passed around.
package todo

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ashish-work/opendev-go/internal/tools/filelock"
)

// Status constants — the only allowed Status values. Strings (not
// iota) so the JSON file is human-inspectable and stable.
const (
	StatusPending    = "pending"
	StatusInProgress = "in_progress"
	StatusCompleted  = "completed"
)

// MaxTodos caps the list size. Exceeding it silently truncates on
// write — chosen over erroring because the model occasionally lays out
// big optimistic plans and we want the first N to survive rather than
// the whole call to fail. 100 is well beyond any reasonable plan.
const MaxTodos = 100

// validStatuses powers Status validation. Map for O(1) lookup.
var validStatuses = map[string]bool{
	StatusPending:    true,
	StatusInProgress: true,
	StatusCompleted:  true,
}

// Todo is one entry in the persisted list. JSON tags drive the
// on-disk shape.
type Todo struct {
	ID        int       `json:"id"`
	Title     string    `json:"title"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// State is the full persisted document. NextID is the monotonically
// increasing counter used to assign new IDs — preserved across Clear
// and across full rewrites so completed and freshly written items
// don't share numbers (which would be confusing when the user is
// reading transcripts).
type State struct {
	Todos     []Todo    `json:"todos"`
	NextID    int       `json:"next_id"`
	UpdatedAt time.Time `json:"updated_at"`
}

// DefaultPath returns ~/.opendev/todos.json. Mirrors the location
// convention used by the truncation package for spillover files —
// everything user-scoped lives under ~/.opendev/.
func DefaultPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("user home dir: %w", err)
	}
	return filepath.Join(home, ".opendev", "todos.json"), nil
}

// Store wraps file I/O for State. Stateless apart from the path; safe
// for concurrent use because all writes go through
// filelock.AtomicWrite. The Tool grabs filelock.For around the
// read-modify-write cycle in Execute, so two concurrent tool calls
// against the same path serialize correctly.
type Store struct {
	Path string
}

// NewStore returns a Store rooted at path. Use DefaultPath() for the
// canonical location; tests pass a t.TempDir-based path.
func NewStore(path string) *Store {
	return &Store{Path: path}
}

// Load reads the state file. A missing file returns the empty State
// with NextID=1 and no error — that's the normal "first run" path.
// Parse failures and permission errors surface to the caller.
func (s *Store) Load() (State, error) {
	data, err := os.ReadFile(s.Path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return State{NextID: 1}, nil
		}
		return State{}, fmt.Errorf("read %s: %w", s.Path, err)
	}
	var st State
	if err := json.Unmarshal(data, &st); err != nil {
		return State{}, fmt.Errorf("parse %s: %w", s.Path, err)
	}
	if st.NextID < 1 {
		st.NextID = 1
	}
	return st, nil
}

// Save writes st to disk via atomic temp+rename. Creates parent
// directories if missing so the first call doesn't depend on the user
// having ever run any other tool that touches ~/.opendev/.
func (s *Store) Save(st State) error {
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", filepath.Dir(s.Path), err)
	}
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal state: %w", err)
	}
	return filelock.AtomicWrite(s.Path, data, 0o644)
}

// ---- State methods. Each is a pure function returning a new State. ----

// WithTodos replaces the whole todo list with items. Returns the new
// State plus a count of items dropped beyond MaxTodos (zero when
// nothing was truncated). IDs are assigned starting at NextID and
// NextID is advanced past the new tail so a subsequent write keeps
// numbering distinct from this batch's items.
func (st State) WithTodos(items []Todo, now time.Time) (State, int) {
	truncated := 0
	if len(items) > MaxTodos {
		truncated = len(items) - MaxTodos
		items = items[:MaxTodos]
	}
	next := st.NextID
	if next < 1 {
		next = 1
	}
	out := make([]Todo, len(items))
	for i, item := range items {
		item.ID = next + i
		if item.Status == "" {
			item.Status = StatusPending
		}
		item.CreatedAt = now
		item.UpdatedAt = now
		out[i] = item
	}
	return State{
		Todos:     out,
		NextID:    next + len(items),
		UpdatedAt: now,
	}, truncated
}

// UpdateOne modifies a single todo by ID. Empty title means "keep the
// existing title"; empty status means "keep the existing status".
// Returns an error when the ID is missing or the status is not one
// of the three valid constants.
func (st State) UpdateOne(id int, title, status string, now time.Time) (State, error) {
	if status != "" && !validStatuses[status] {
		return st, fmt.Errorf("invalid status %q (must be pending, in_progress, or completed)", status)
	}
	idx := -1
	for i, t := range st.Todos {
		if t.ID == id {
			idx = i
			break
		}
	}
	if idx < 0 {
		return st, fmt.Errorf("todo with id=%d not found", id)
	}
	out := make([]Todo, len(st.Todos))
	copy(out, st.Todos)
	if title != "" {
		out[idx].Title = title
	}
	if status != "" {
		out[idx].Status = status
	}
	out[idx].UpdatedAt = now
	return State{
		Todos:     out,
		NextID:    st.NextID,
		UpdatedAt: now,
	}, nil
}

// CompleteOne marks the todo with the given ID as completed. Sugar
// over UpdateOne; kept as a distinct entry point so the tool's action
// dispatch can route "complete" without recomposing args.
func (st State) CompleteOne(id int, now time.Time) (State, error) {
	return st.UpdateOne(id, "", StatusCompleted, now)
}

// Cleared returns an empty State, preserving NextID so freshly added
// items continue the numbering instead of recycling IDs of just-
// deleted items.
func (st State) Cleared(now time.Time) State {
	return State{
		Todos:     nil,
		NextID:    st.NextID,
		UpdatedAt: now,
	}
}

// Counts returns the number of todos in each status bucket. Used by
// the renderer and by Tool.Execute's metadata payload.
func (st State) Counts() (pending, inProgress, completed int) {
	for _, t := range st.Todos {
		switch t.Status {
		case StatusPending:
			pending++
		case StatusInProgress:
			inProgress++
		case StatusCompleted:
			completed++
		}
	}
	return
}

// Render produces the human/model-readable ASCII list. ASCII markers
// ([x]/[~]/[ ]) over Unicode glyphs because they render the same
// in every terminal, paste cleanly into emails or chat, and don't
// depend on a font that has the glyphs.
func (st State) Render() string {
	if len(st.Todos) == 0 {
		return `Todos: (empty). Use action="write" with a todos array to set a plan.`
	}
	p, ip, c := st.Counts()
	var b strings.Builder
	fmt.Fprintf(&b, "Todos (%d total — %d completed, %d in_progress, %d pending):\n\n",
		len(st.Todos), c, ip, p)
	for _, t := range st.Todos {
		fmt.Fprintf(&b, "%s %d. %s\n", statusMarker(t.Status), t.ID, t.Title)
	}
	b.WriteString(`
Use action="update" with id and status to change state, or action="complete" with id to mark done.`)
	return b.String()
}

func statusMarker(s string) string {
	switch s {
	case StatusCompleted:
		return "[x]"
	case StatusInProgress:
		return "[~]"
	default:
		return "[ ]"
	}
}
