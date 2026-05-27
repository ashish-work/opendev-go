package listfiles

import (
	"path/filepath"
	"strings"
)

// matchGlob reports whether path matches the glob pattern. Semantics:
//
//   - "*" matches zero or more characters within a single path segment
//     (it does NOT cross "/").
//   - "?" matches exactly one character within a single path segment.
//   - "**" matches zero or more whole path segments. So "**/*.go" matches
//     "a.go", "dir/a.go", and "dir/sub/a.go", but not the path "".
//   - Any other character matches itself.
//
// Both pattern and path use forward slashes. Caller is responsible
// for converting OS-native paths (filepath.ToSlash). Empty pattern
// matches only empty path; bare "**" matches anything (including the
// empty path — by convention, "**" alone is a "yes to anything"
// signal from the caller).
func matchGlob(pattern, path string) bool {
	switch pattern {
	case "":
		return path == ""
	case "**":
		return true
	}
	return matchSegments(strings.Split(pattern, "/"), strings.Split(path, "/"))
}

// matchSegments walks two segment lists in parallel, with "**" allowed
// to consume zero or more segments greedily. Pure recursion; the depth
// is bounded by the pattern's segment count, not the path's, so it's
// safe on long paths.
func matchSegments(p, s []string) bool {
	// Drop empty leading segments produced by leading "/" in either input.
	for len(p) > 0 && p[0] == "" {
		p = p[1:]
	}
	for len(s) > 0 && s[0] == "" {
		s = s[1:]
	}

	if len(p) == 0 {
		return len(s) == 0
	}

	if p[0] == "**" {
		rest := p[1:]
		// Collapse "**/**/**" runs — they're all equivalent to a single "**".
		for len(rest) > 0 && rest[0] == "**" {
			rest = rest[1:]
		}
		if len(rest) == 0 {
			return true // trailing "**" matches whatever is left
		}
		for k := 0; k <= len(s); k++ {
			if matchSegments(rest, s[k:]) {
				return true
			}
		}
		return false
	}

	if len(s) == 0 {
		return false
	}

	// filepath.Match handles "*", "?", "[chars]" within a single segment;
	// it returns an error only for malformed patterns ("[" with no "]"),
	// in which case we treat as no-match.
	matched, err := filepath.Match(p[0], s[0])
	if err != nil || !matched {
		return false
	}
	return matchSegments(p[1:], s[1:])
}
