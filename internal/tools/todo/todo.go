package todo

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/ashish-work/opendev-go/internal/tools"
	"github.com/ashish-work/opendev-go/internal/tools/filelock"
)

// ToolName is the canonical name the model uses to invoke this tool.
const ToolName = "todo"

// Tool implements tools.Tool for the todo operation. The Path and Now
// fields are escape hatches for tests; main.go constructs with New()
// and leaves both at the zero value (DefaultPath + time.Now).
type Tool struct {
	// Path overrides the on-disk state location. Empty uses DefaultPath().
	Path string

	// Now lets tests inject deterministic timestamps. Nil uses time.Now.
	Now func() time.Time
}

// New returns a ready-to-register Tool that uses DefaultPath() and
// time.Now under the hood.
func New() *Tool { return &Tool{} }

var (
	_ tools.Tool        = (*Tool)(nil)
	_ tools.Categorized = (*Tool)(nil)
)

// Name implements tools.Tool.
func (t *Tool) Name() string { return ToolName }

// Category implements tools.Categorized. CategoryMeta covers planning,
// task management, and todo-style tools per the taxonomy in tools/tool.go.
func (t *Tool) Category() tools.Category { return tools.CategoryMeta }

// Description is the model's only authoritative source for todo
// semantics; the system prompt no longer carries per-tool sections.
func (t *Tool) Description() string {
	return "Manage a persistent todo list you (the model) use to plan multi-step " +
		"work. State lives at ~/.opendev/todos.json and survives REPL restarts " +
		"AND context compaction (the plan is on disk, not in the prompt). " +
		"Use this whenever a task has more than 2-3 steps so the plan doesn't " +
		"get lost across long bash output or summarized turns. " +
		"Actions: write (replace the whole list — pass todos as an array of " +
		"title strings or {title, status} objects), list (show current state), " +
		"update (modify one todo by id; pass title and/or status), complete " +
		"(mark one done by id — sugar over update), clear (empty the list). " +
		"The only valid status values are pending, in_progress, and completed. " +
		"Max 100 items; extras are silently truncated. Output is ASCII with " +
		"[x] for completed, [~] for in_progress, [ ] for pending. " +
		"IDs are monotonically assigned and preserved across rewrites — " +
		"action=write does NOT recycle IDs of items it replaced, so the model " +
		"can refer to a numbered todo unambiguously even after re-planning. " +
		"Typical flow: write the plan up front; update one item to in_progress " +
		"as you start that step; complete it as you finish; list whenever you " +
		"need to remember where you are."
}

// Schema is the JSON Schema for the model's tool-call arguments.
func (t *Tool) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"action": {
				"type": "string",
				"enum": ["write", "list", "update", "complete", "clear"],
				"description": "Which operation to perform."
			},
			"todos": {
				"type": "array",
				"description": "For action=write: array of titles (strings) or {title, status} objects. Status defaults to pending. Max 100 items.",
				"items": {
					"oneOf": [
						{"type": "string", "minLength": 1},
						{
							"type": "object",
							"properties": {
								"title":  {"type": "string", "minLength": 1},
								"status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
							},
							"required": ["title"]
						}
					]
				}
			},
			"id": {
				"type": "integer",
				"description": "For action=update or action=complete: the numeric id of the todo (see action=list output)."
			},
			"title": {
				"type": "string",
				"description": "For action=update: replacement title. Omit to keep the existing title."
			},
			"status": {
				"type": "string",
				"enum": ["pending", "in_progress", "completed"],
				"description": "For action=update: new status. Omit to keep the existing status."
			}
		},
		"required": ["action"]
	}`)
}

// args is the parsed form of the tool-call arguments. todos is left as
// []json.RawMessage so we can disambiguate string vs object per item.
type args struct {
	Action string            `json:"action"`
	Todos  []json.RawMessage `json:"todos,omitempty"`
	ID     *int              `json:"id,omitempty"`
	Title  string            `json:"title,omitempty"`
	Status string            `json:"status,omitempty"`
}

// Execute dispatches on action. Tool-domain errors (unknown action,
// missing id, invalid status, todo not found) return Success:false.
// Infrastructure errors (ctx cancellation, disk write failure) surface
// as Go errors. The lock + load + mutate + save sequence is held under
// filelock.For(path) so two concurrent calls don't race.
func (t *Tool) Execute(ctx context.Context, tctx tools.ToolContext, raw json.RawMessage) (tools.ToolResult, error) {
	var a args
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &a); err != nil {
			return failf("invalid arguments: %v", err), nil
		}
	}
	if a.Action == "" {
		return failf("action is required (one of: write, list, update, complete, clear)"), nil
	}
	if err := ctx.Err(); err != nil {
		return tools.ToolResult{}, err
	}

	path := t.Path
	if path == "" {
		p, err := DefaultPath()
		if err != nil {
			return tools.ToolResult{}, fmt.Errorf("todo: %w", err)
		}
		path = p
	}

	now := time.Now
	if t.Now != nil {
		now = t.Now
	}

	// Lock + read-modify-write. The store layer uses atomic temp+rename
	// for crash safety, but the read+save pair still needs serializing
	// against another in-flight todo call.
	mu := filelock.For(path)
	mu.Lock()
	defer mu.Unlock()

	store := NewStore(path)
	state, err := store.Load()
	if err != nil {
		return failf("load state: %v", err), nil
	}

	switch strings.ToLower(a.Action) {

	case "list":
		// Pure read; no need to Save.
		return success(state.Render(), stateMeta(state, 0)), nil

	case "clear":
		newState := state.Cleared(now())
		if err := store.Save(newState); err != nil {
			return tools.ToolResult{}, fmt.Errorf("todo: save: %w", err)
		}
		return success("Cleared all todos.\n\n"+newState.Render(), stateMeta(newState, 0)), nil

	case "write":
		items, err := parseTodos(a.Todos)
		if err != nil {
			return failf("%v", err), nil
		}
		newState, truncated := state.WithTodos(items, now())
		if err := store.Save(newState); err != nil {
			return tools.ToolResult{}, fmt.Errorf("todo: save: %w", err)
		}
		header := fmt.Sprintf("Wrote %d todo(s).", len(newState.Todos))
		if truncated > 0 {
			header += fmt.Sprintf(" %d extra dropped (max %d).", truncated, MaxTodos)
		}
		return success(header+"\n\n"+newState.Render(), stateMeta(newState, truncated)), nil

	case "update":
		if a.ID == nil {
			return failf("id is required for action=update"), nil
		}
		if a.Title == "" && a.Status == "" {
			return failf("provide at least one of title or status for action=update"), nil
		}
		newState, err := state.UpdateOne(*a.ID, a.Title, a.Status, now())
		if err != nil {
			return failf("%v", err), nil
		}
		if err := store.Save(newState); err != nil {
			return tools.ToolResult{}, fmt.Errorf("todo: save: %w", err)
		}
		return success(fmt.Sprintf("Updated todo %d.\n\n%s", *a.ID, newState.Render()), stateMeta(newState, 0)), nil

	case "complete":
		if a.ID == nil {
			return failf("id is required for action=complete"), nil
		}
		newState, err := state.CompleteOne(*a.ID, now())
		if err != nil {
			return failf("%v", err), nil
		}
		if err := store.Save(newState); err != nil {
			return tools.ToolResult{}, fmt.Errorf("todo: save: %w", err)
		}
		msg := fmt.Sprintf("Completed todo %d.", *a.ID)
		if p, ip, _ := newState.Counts(); p == 0 && ip == 0 && len(newState.Todos) > 0 {
			msg += " All todos are done."
		}
		return success(msg+"\n\n"+newState.Render(), stateMeta(newState, 0)), nil

	default:
		return failf("unknown action %q (must be one of: write, list, update, complete, clear)", a.Action), nil
	}
}

// parseTodos converts the args.Todos union (each item is either a JSON
// string or a {title, status} object) into a []Todo. Returns an error
// at the first malformed entry so the model gets a precise message
// pointing at the offending index.
func parseTodos(raw []json.RawMessage) ([]Todo, error) {
	out := make([]Todo, 0, len(raw))
	for i, item := range raw {
		// Try string first; many models default to that for plain plans.
		var s string
		if err := json.Unmarshal(item, &s); err == nil {
			if strings.TrimSpace(s) == "" {
				continue // skip silently — empty strings are usually a bug, not intent
			}
			out = append(out, Todo{Title: s})
			continue
		}
		// Fall back to object form.
		var obj struct {
			Title  string `json:"title"`
			Status string `json:"status"`
		}
		if err := json.Unmarshal(item, &obj); err != nil {
			return nil, fmt.Errorf("todos[%d]: must be a string or {title, status} object", i)
		}
		if strings.TrimSpace(obj.Title) == "" {
			return nil, fmt.Errorf("todos[%d]: title is required and non-empty", i)
		}
		if obj.Status != "" && !validStatuses[obj.Status] {
			return nil, fmt.Errorf("todos[%d]: invalid status %q (must be pending, in_progress, or completed)", i, obj.Status)
		}
		out = append(out, Todo{Title: obj.Title, Status: obj.Status})
	}
	return out, nil
}

func stateMeta(st State, truncated int) map[string]any {
	p, ip, c := st.Counts()
	m := map[string]any{
		"total":             len(st.Todos),
		"pending_count":     p,
		"in_progress_count": ip,
		"completed_count":   c,
		"next_id":           st.NextID,
	}
	if truncated > 0 {
		m["truncated"] = truncated
	}
	return m
}

func success(output string, meta map[string]any) tools.ToolResult {
	return tools.ToolResult{Success: true, Output: output, Metadata: meta}
}

func failf(format string, args ...any) tools.ToolResult {
	return tools.ToolResult{Success: false, Error: fmt.Sprintf(format, args...)}
}
