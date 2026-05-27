// Package listfiles implements the list_files tool — the model's way
// to discover files under a directory with glob filtering, .gitignore
// awareness, and a hardcoded skip list for common dependency/build/VCS
// noise (node_modules, .git, target, __pycache__, *.min.js, …).
//
// Listing is read-only and idempotent. The walker uses filepath.WalkDir,
// honoring the directory's own SkipDir signal to prune entire subtrees
// (rather than recursing into them and filtering everything out, which
// would be O(everything)).
//
// Design notes:
//   - We do NOT use ripgrep or any external tool. The walker is stdlib.
//   - The glob matcher (glob.go) handles "**" semantics that
//     filepath.Match lacks.
//   - .gitignore support is a practical subset (most real files work,
//     but negation and parent-dir .gitignore are out of scope). See
//     gitignore.go for the full list of limitations.
package listfiles

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ashish-work/opendev-go/internal/tools"
)

// ToolName is the canonical name the model uses to invoke this tool.
const ToolName = "list_files"

// MaxResults caps the number of paths returned in one call. Beyond this,
// the output is truncated with a footer noting how many were elided.
// Exported so tests / advanced callers can shrink the cap.
var MaxResults = 1000

// Tool implements tools.Tool for the list_files operation. Stateless.
type Tool struct{}

// New returns a ready-to-register Tool.
func New() *Tool { return &Tool{} }

// Compile-time guards.
var (
	_ tools.Tool        = (*Tool)(nil)
	_ tools.Categorized = (*Tool)(nil)
)

// Name implements tools.Tool.
func (t *Tool) Name() string { return ToolName }

// Category implements tools.Categorized.
func (t *Tool) Category() tools.Category { return tools.CategoryRead }

// Description is the model's only authoritative source for list_files
// semantics; the system prompt no longer carries per-tool sections.
func (t *Tool) Description() string {
	return "List files under a directory matching a glob pattern. Use this for " +
		"directory inspection (\"what's in cmd/?\") and file discovery " +
		"(\"where are the .go files?\") — preferred over `bash ls` or " +
		"`bash find` because it returns structured results, respects " +
		".gitignore by default, and automatically skips conventional " +
		"noise (node_modules, .git, target, __pycache__, *.min.js, " +
		"*.pyc, etc.). " +
		"Parameters: pattern (glob — e.g. **/*.go, *.md, **/test_*.py; " +
		"default ** matches everything), path (absolute or " +
		"working-directory-relative; default working dir), max_depth " +
		"(recursion cap; 0 = base only; default unlimited), ignore " +
		"(extra glob patterns to skip), respect_gitignore (default " +
		"true), sort (\"lex\" default, or \"mtime\" for newest-first). " +
		"Glob semantics: * matches any chars except /; ** matches zero " +
		"or more whole path segments — that's why **/*.go matches files " +
		"at any depth while *.go matches only top-level files. " +
		"Results are capped at " + maxResultsStr() + " entries; if more match, " +
		"a footer notes truncation and you should refine the pattern " +
		"or path. Output is one relative path per line, no other framing."
}

// maxResultsStr formats MaxResults for inclusion in the description.
// Kept as a tiny helper so Description() reads as a single sentence.
func maxResultsStr() string {
	return fmt.Sprintf("%d", MaxResults)
}

// Schema is the JSON Schema for the model's tool-call arguments.
func (t *Tool) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"pattern": {
				"type": "string",
				"description": "Glob to match. * matches any chars except /; ** matches zero or more path segments. Examples: **/*.go (recursive), *.md (top-level only), cmd/**/main.go. Default: ** (everything)."
			},
			"path": {
				"type": "string",
				"description": "Base directory to walk. Absolute or relative to the working directory. Default: working directory."
			},
			"max_depth": {
				"type": "integer",
				"description": "Maximum recursion depth from base (0 = base directory only). Default: unlimited."
			},
			"ignore": {
				"type": "array",
				"items": {"type": "string"},
				"description": "Additional glob patterns to skip. Trailing / marks a directory-only pattern."
			},
			"respect_gitignore": {
				"type": "boolean",
				"description": "Honor a .gitignore file at the base directory (subset support: no negation, no parent-dir .gitignore). Default: true."
			},
			"sort": {
				"type": "string",
				"enum": ["lex", "mtime"],
				"description": "Result order. lex (default) is alphabetical; mtime is newest-first."
			}
		},
		"required": []
	}`)
}

type args struct {
	Pattern          string   `json:"pattern,omitempty"`
	Path             string   `json:"path,omitempty"`
	MaxDepth         *int     `json:"max_depth,omitempty"`
	Ignore           []string `json:"ignore,omitempty"`
	RespectGitignore *bool    `json:"respect_gitignore,omitempty"`
	Sort             string   `json:"sort,omitempty"`
}

type fileMatch struct {
	relPath string
	mtime   int64
}

// Execute walks the directory tree and returns matching files.
// Tool-domain failures (missing dir, bad pattern, not a directory)
// return ToolResult{Success:false}. Infrastructure failures (ctx
// cancellation) surface as Go errors.
func (t *Tool) Execute(ctx context.Context, tctx tools.ToolContext, raw json.RawMessage) (tools.ToolResult, error) {
	var a args
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &a); err != nil {
			return failf("invalid arguments: %v", err), nil
		}
	}

	if a.Pattern == "" {
		a.Pattern = "**"
	}

	sortMode := strings.ToLower(a.Sort)
	if sortMode == "" {
		sortMode = "lex"
	}
	if sortMode != "lex" && sortMode != "mtime" {
		return failf("sort must be \"lex\" or \"mtime\", got %q", a.Sort), nil
	}

	base, err := resolveBase(a.Path, tctx.WorkingDir)
	if err != nil {
		return failf("%v", err), nil
	}

	info, err := os.Stat(base)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return failf("directory not found: %s", base), nil
		}
		return failf("stat %s: %v", base, err), nil
	}
	if !info.IsDir() {
		return failf("%q is not a directory", base), nil
	}

	respectGitignore := true
	if a.RespectGitignore != nil {
		respectGitignore = *a.RespectGitignore
	}
	var gi *Gitignore
	if respectGitignore {
		g, err := LoadGitignore(filepath.Join(base, ".gitignore"))
		if err != nil {
			// Parse error on .gitignore is unusual but not fatal — log into
			// metadata via a returned warning is overkill; we just skip
			// the gitignore in that case.
			gi = nil
		} else {
			gi = g
		}
	}

	results, truncated, walkErr := walk(ctx, base, a.Pattern, a.MaxDepth, a.Ignore, gi, sortMode == "mtime")
	if walkErr != nil {
		if errors.Is(walkErr, context.Canceled) || errors.Is(walkErr, context.DeadlineExceeded) {
			return tools.ToolResult{}, walkErr
		}
		return tools.ToolResult{}, fmt.Errorf("list_files: walk %s: %w", base, walkErr)
	}

	if sortMode == "mtime" {
		sort.Slice(results, func(i, j int) bool { return results[i].mtime > results[j].mtime })
	} else {
		sort.Slice(results, func(i, j int) bool { return results[i].relPath < results[j].relPath })
	}

	meta := map[string]any{
		"base_dir":    base,
		"pattern":     a.Pattern,
		"sort":        sortMode,
		"total_files": len(results),
		"truncated":   truncated,
	}

	if len(results) == 0 {
		hint := ""
		if strings.HasSuffix(a.Pattern, "**") && !strings.HasSuffix(a.Pattern, "**/*") {
			hint = "\nHint: '**' alone matches whole-path zero-or-more, so a trailing '**' often returns nothing useful. Try '**/*' or '**/*.ext' to match files at any depth."
		}
		return tools.ToolResult{
			Success:  true,
			Output:   fmt.Sprintf("No files matching %q in %s%s", a.Pattern, base, hint),
			Metadata: meta,
		}, nil
	}

	var b strings.Builder
	for _, r := range results {
		b.WriteString(r.relPath)
		b.WriteByte('\n')
	}
	if truncated {
		fmt.Fprintf(&b, "\n... results truncated at %d files; refine the pattern or path to see more.\n", MaxResults)
	}

	return tools.ToolResult{
		Success:  true,
		Output:   b.String(),
		Metadata: meta,
	}, nil
}

// resolveBase computes the base directory, applying tctx.WorkingDir to
// relative inputs. Returns an error string the caller surfaces via failf
// when both args are empty.
func resolveBase(path, workingDir string) (string, error) {
	if path == "" {
		if workingDir == "" {
			return "", errors.New("path is required when no working directory is configured")
		}
		return workingDir, nil
	}
	if filepath.IsAbs(path) {
		return path, nil
	}
	if workingDir == "" {
		// Relative path with no working dir — interpret as relative to cwd.
		// filepath.Abs handles that.
		abs, err := filepath.Abs(path)
		if err != nil {
			return "", fmt.Errorf("resolve path: %w", err)
		}
		return abs, nil
	}
	return filepath.Join(workingDir, path), nil
}

// walk does the actual directory traversal. Pulled out of Execute so
// tests can exercise it directly and so the main function reads as
// arg-parsing + walk + sort + format, three phases.
func walk(ctx context.Context, base, pattern string, maxDepth *int, userIgnore []string, gi *Gitignore, needsMtime bool) ([]fileMatch, bool, error) {
	var results []fileMatch
	truncated := false

	err := filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			// Skip unreadable entries (permissions, vanished files) — listing
			// shouldn't bail out the whole walk because one subtree is denied.
			if d != nil && d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if path == base {
			return nil
		}

		// Cooperative cancellation: cheap check at every entry.
		if cerr := ctx.Err(); cerr != nil {
			return cerr
		}

		rel, _ := filepath.Rel(base, path)
		rel = filepath.ToSlash(rel)

		// Default directory excludes (node_modules, .git, target, ...).
		if pathHasExcludedComponent(rel) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// .gitignore — apply BEFORE max_depth so an ignored subtree is
		// pruned at the directory level (cheap), not visited and filtered.
		if gi != nil && gi.Match(rel, d.IsDir()) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// User-supplied ignore patterns.
		for _, ig := range userIgnore {
			if matchIgnore(ig, rel, d.IsDir()) {
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
		}

		// Max-depth prune at directory boundaries.
		if maxDepth != nil {
			depth := strings.Count(rel, "/") // 0 for top-level entries
			if depth > *maxDepth {
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
		}

		if d.IsDir() {
			return nil
		}

		// Default file-glob excludes (*.min.js, *.pyc, ...).
		if isExcludedFileGlob(filepath.Base(rel)) {
			return nil
		}

		// Final pattern match.
		if !matchGlob(pattern, rel) {
			return nil
		}

		var mtime int64
		if needsMtime {
			if info, err := d.Info(); err == nil {
				mtime = info.ModTime().UnixNano()
			}
		}

		results = append(results, fileMatch{relPath: rel, mtime: mtime})

		if len(results) >= MaxResults {
			truncated = true
			return filepath.SkipAll
		}
		return nil
	})

	if errors.Is(err, filepath.SkipAll) {
		err = nil
	}
	return results, truncated, err
}

func pathHasExcludedComponent(rel string) bool {
	for _, comp := range strings.Split(rel, "/") {
		for _, exc := range DefaultExcludeDirs {
			if comp == exc {
				return true
			}
		}
	}
	return false
}

func isExcludedFileGlob(basename string) bool {
	for _, pat := range DefaultExcludeFileGlobs {
		if matched, _ := filepath.Match(pat, basename); matched {
			return true
		}
	}
	return false
}

// matchIgnore evaluates a single user-supplied ignore pattern against
// a path. Trailing "/" restricts to directories; otherwise the rule
// applies to either. A pattern without "/" matches the basename at any
// depth (gitignore-style); a pattern with "/" matches the full path.
func matchIgnore(pattern, relPath string, isDir bool) bool {
	if strings.HasSuffix(pattern, "/") {
		if !isDir {
			return false
		}
		pattern = strings.TrimSuffix(pattern, "/")
	}
	if !strings.Contains(pattern, "/") {
		if ok, _ := filepath.Match(pattern, filepath.Base(relPath)); ok {
			return true
		}
		return false
	}
	return matchGlob(pattern, relPath)
}

func failf(format string, args ...any) tools.ToolResult {
	return tools.ToolResult{
		Success: false,
		Error:   fmt.Sprintf(format, args...),
	}
}
