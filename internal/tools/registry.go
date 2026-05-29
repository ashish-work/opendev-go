package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"
)

// Registry stores Tool implementations indexed by name and dispatches
// Execute calls by name. The zero value is NOT usable — call NewRegistry.
//
// Concurrency: safe for many concurrent readers and occasional writers.
// Registrations typically happen once at startup; Dispatch happens
// on every loop iteration. RWMutex matches that access pattern.
//
// v1 ships only the core register/lookup/dispatch path. Cross-cutting
// concerns that need to plug into dispatch — middleware, dedup caches,
// aliases, input sanitizers — land alongside the features that need
// them (doom-loop detection brings its own fingerprinting, etc.).
type Registry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

// NewRegistry returns an empty Registry ready for registration.
func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Tool)}
}

// Register adds a Tool to the registry. Returns wrapped ErrInvalidParams
// for empty names and duplicate registrations — both indicate wiring
// bugs we want to surface loudly rather than silently overwrite.
//
// Idempotent registration is intentionally NOT supported. If you have
// a legitimate need to replace a tool, build that as an explicit
// `Replace` method later; don't paper over the duplicate.
func (r *Registry) Register(t Tool) error {
	name := t.Name()
	if name == "" {
		return fmt.Errorf("%w: tool has empty name", ErrInvalidParams)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("%w: duplicate tool name %q", ErrInvalidParams, name)
	}
	r.tools[name] = t
	return nil
}

// Get returns the Tool registered under name. The second return is
// false when no such tool exists — mirrors Go's map comma-ok idiom so
// callers handle missing tools explicitly.
func (r *Registry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	return t, ok
}

// Names returns the sorted slice of registered tool names. Sort is
// deterministic so the LLM request's tools array is stable across
// turns — important for provider-side prompt caching (T3.5).
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.tools))
	for name := range r.tools {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// Filter returns a fresh *Registry containing only the tools whose
// names appear in allowedNames. The receiver is not modified — every
// call produces an independent peer registry with the same
// Register / Get / Dispatch surface.
//
// Semantics:
//
//   - nil allowedNames     → full passthrough. Every tool from the
//     receiver is included. Used by SubAgentSpec.Tools == nil
//     (no restriction; e.g. the Build subagent).
//
//   - empty allowedNames   → empty registry. No tools allowed.
//     Distinct from nil; useful for a hypothetical "summarize only"
//     spec that must answer from context alone.
//
//   - non-empty whitelist  → only the listed names are included.
//     Names not present in the receiver are silently dropped, with
//     a slog.Debug log. The "silently drop unknowns" path matters
//     because subagent specs can forward-reference tools that
//     don't exist yet (e.g. Planner's present_plan in v2). Filter
//     produces what it can without failing.
//
// Used by subagent dispatch (internal/tools/spawn) to scope each
// child loop to its spec's advertised tool set.
//
// Performance: O(len(receiver) + len(allowedNames)). Filter is
// called once per subagent spawn (not per tool call) and the
// registries are small, so the linear cost is invisible. The
// returned registry is independent — future Register calls on the
// new registry don't affect the original.
func (r *Registry) Filter(allowedNames []string) *Registry {
	out := NewRegistry()

	r.mu.RLock()
	defer r.mu.RUnlock()

	if allowedNames == nil {
		// Full passthrough: copy every entry.
		for name, tool := range r.tools {
			out.tools[name] = tool
		}
		return out
	}

	// Whitelist mode (including empty slice → empty result).
	for _, name := range allowedNames {
		tool, ok := r.tools[name]
		if !ok {
			slog.Debug("registry: Filter dropping unknown tool name",
				"name", name)
			continue
		}
		out.tools[name] = tool
	}
	return out
}

// Dispatch looks up the tool by name and runs Execute, populating
// DurationMS on the returned ToolResult regardless of success/failure.
// Unknown names return ErrToolNotFound wrapped with the name.
//
// ctx flows straight through to Execute — Dispatch does not impose its
// own timeout. Compose timeouts via context.WithTimeout at the call site.
//
// IMPORTANT: the registry lock is released before Execute runs. Tools
// can be slow (a bash command may take minutes); holding the lock
// during execution would block all other dispatches.
func (r *Registry) Dispatch(
	ctx context.Context,
	tctx ToolContext,
	name string,
	args json.RawMessage,
) (ToolResult, error) {
	tool, ok := r.Get(name)
	if !ok {
		return ToolResult{}, fmt.Errorf("%w: %q", ErrToolNotFound, name)
	}

	start := time.Now()
	result, err := tool.Execute(ctx, tctx, args)
	result.DurationMS = time.Since(start).Milliseconds()

	return result, err
}
