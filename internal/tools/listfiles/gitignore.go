package listfiles

import (
	"bufio"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Gitignore implements a practical subset of the .gitignore file format
// — enough for the common case (most .gitignore files in the wild) but
// not the full spec.
//
// SUPPORTED:
//   - Comments (#) and blank lines are skipped.
//   - Trailing "/" marks a directory-only rule.
//   - Leading "/" anchors a pattern to the gitignore's directory (root).
//   - "*", "?", "**" globs (via matchGlob).
//
// NOT SUPPORTED (silently ignored to keep the tool useful):
//   - "!pattern" negation. The parser skips these lines so the rest of
//     the file still applies; a future task can add proper negation.
//   - "[abc]" character classes (filepath.Match supports these, so they
//     work inside a segment; the limitation only matters for unusual
//     edge cases).
//   - .gitignore files in parent directories. We load only the .gitignore
//     at the base directory of the walk.
type Gitignore struct {
	rules []gitignoreRule
}

type gitignoreRule struct {
	pattern  string // glob (without trailing slash or leading slash)
	isDir    bool   // rule applies to directories only
	anchored bool   // rule is anchored to gitignore root
}

// LoadGitignore parses the .gitignore at the given path. Missing file
// returns (nil, nil) — "no rules" is normal, not an error.
func LoadGitignore(path string) (*Gitignore, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	return parseGitignore(f)
}

func parseGitignore(r io.Reader) (*Gitignore, error) {
	var rules []gitignoreRule
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "!") {
			// Negation: skip silently. Limitation documented on Gitignore.
			continue
		}
		rule := gitignoreRule{}
		if strings.HasSuffix(line, "/") {
			rule.isDir = true
			line = strings.TrimSuffix(line, "/")
		}
		if strings.HasPrefix(line, "/") {
			rule.anchored = true
			line = strings.TrimPrefix(line, "/")
		}
		if line == "" {
			continue
		}
		rule.pattern = line
		rules = append(rules, rule)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return &Gitignore{rules: rules}, nil
}

// Match reports whether relPath (forward-slash form, relative to the
// gitignore's directory) is ignored. isDir signals whether the entry
// is a directory — needed for trailing-slash rules.
//
// A nil receiver always returns false; saves the caller a nil check
// when "no gitignore file" is normal.
func (g *Gitignore) Match(relPath string, isDir bool) bool {
	if g == nil {
		return false
	}
	relPath = filepath.ToSlash(relPath)
	base := filepath.Base(relPath)
	for _, r := range g.rules {
		if r.isDir && !isDir {
			continue
		}
		if r.anchored {
			// Anchored: pattern matches only against the full relative path.
			if matchGlob(r.pattern, relPath) {
				return true
			}
			continue
		}
		// Unanchored, no "/" in pattern → matches basename at any depth.
		if !strings.Contains(r.pattern, "/") {
			if ok, _ := filepath.Match(r.pattern, base); ok {
				return true
			}
			continue
		}
		// Unanchored with "/" → matches the full path OR any path suffix.
		if matchGlob(r.pattern, relPath) {
			return true
		}
		parts := strings.Split(relPath, "/")
		for i := 1; i < len(parts); i++ {
			if matchGlob(r.pattern, strings.Join(parts[i:], "/")) {
				return true
			}
		}
	}
	return false
}
