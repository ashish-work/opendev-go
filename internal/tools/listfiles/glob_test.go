package listfiles

import "testing"

func TestMatchGlob(t *testing.T) {
	tests := []struct {
		pattern string
		path    string
		want    bool
	}{
		// Bare ** and empty
		{"**", "", true},
		{"**", "a.go", true},
		{"**", "dir/sub/a.go", true},
		{"", "", true},
		{"", "a", false},

		// Top-level only (no /)
		{"*.go", "a.go", true},
		{"*.go", "main.go", true},
		{"*.go", "main.py", false},
		{"*.go", "cmd/a.go", false}, // single segment pattern, multi-segment path

		// Recursive
		{"**/*.go", "a.go", true},
		{"**/*.go", "cmd/a.go", true},
		{"**/*.go", "cmd/sub/a.go", true},
		{"**/*.go", "a.py", false},

		// Anchored top-level subdir
		{"cmd/*.go", "cmd/a.go", true},
		{"cmd/*.go", "cmd/sub/a.go", false},
		{"cmd/*.go", "a.go", false},

		// Recursive inside a dir
		{"cmd/**/*.go", "cmd/a.go", true},
		{"cmd/**/*.go", "cmd/sub/a.go", true},
		{"cmd/**/*.go", "cmd/sub/deep/a.go", true},
		{"cmd/**/*.go", "internal/a.go", false},

		// ? matches single char inside a segment
		{"?.go", "a.go", true},
		{"?.go", "ab.go", false},
		{"?.go", "/a.go", true}, // leading / is stripped

		// **/ in middle
		{"a/**/c", "a/c", true},   // ** matches zero segments
		{"a/**/c", "a/b/c", true}, // matches one segment
		{"a/**/c", "a/b/d/c", true},
		{"a/**/c", "a/b/d", false},

		// Multiple **
		{"**/**/*.go", "a.go", true},
		{"**/**/*.go", "cmd/sub/a.go", true},

		// [chars] class via filepath.Match within a segment
		{"[abc].txt", "a.txt", true},
		{"[abc].txt", "d.txt", false},

		// Trailing **
		{"cmd/**", "cmd/a.go", true},
		{"cmd/**", "cmd/sub/a.go", true},
		{"cmd/**", "other/a.go", false},
	}

	for _, tt := range tests {
		t.Run(tt.pattern+"|"+tt.path, func(t *testing.T) {
			got := matchGlob(tt.pattern, tt.path)
			if got != tt.want {
				t.Errorf("matchGlob(%q, %q) = %v, want %v", tt.pattern, tt.path, got, tt.want)
			}
		})
	}
}
