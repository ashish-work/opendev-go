package listfiles

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseGitignore_SkipsCommentsAndBlanks(t *testing.T) {
	in := `# comment
*.log

build/
/dist
!important.log
`
	g, err := parseGitignore(strings.NewReader(in))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(g.rules) != 3 {
		t.Fatalf("got %d rules, want 3 (comment, blank, and negation should all skip)", len(g.rules))
	}

	// *.log → unanchored, not dir
	if g.rules[0].pattern != "*.log" || g.rules[0].anchored || g.rules[0].isDir {
		t.Errorf("rule 0 = %+v, want pattern=*.log, anchored=false, isDir=false", g.rules[0])
	}
	// build/ → unanchored, dir
	if g.rules[1].pattern != "build" || g.rules[1].anchored || !g.rules[1].isDir {
		t.Errorf("rule 1 = %+v, want pattern=build, anchored=false, isDir=true", g.rules[1])
	}
	// /dist → anchored, not dir
	if g.rules[2].pattern != "dist" || !g.rules[2].anchored || g.rules[2].isDir {
		t.Errorf("rule 2 = %+v, want pattern=dist, anchored=true, isDir=false", g.rules[2])
	}
}

func TestGitignore_Match(t *testing.T) {
	in := `# my .gitignore
*.log
build/
/dist
internal/secret/
node_modules
`
	g, err := parseGitignore(strings.NewReader(in))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	tests := []struct {
		path  string
		isDir bool
		want  bool
	}{
		// *.log: unanchored, basename match
		{"a.log", false, true},
		{"sub/a.log", false, true},
		{"a.log/", true, true}, // would match dir too
		{"a.txt", false, false},

		// build/: directory only, unanchored
		{"build", true, true},
		{"build", false, false}, // not dir → does not match
		{"sub/build", true, true},

		// /dist: anchored, file or dir
		{"dist", false, true},
		{"dist", true, true},
		{"sub/dist", false, false}, // anchored, so only top-level

		// internal/secret/: dir only, unanchored, has "/"
		{"internal/secret", true, true},
		{"other/internal/secret", true, true}, // unanchored path suffix
		{"internal/secret", false, false},     // not a directory

		// node_modules: bare name, matches at any depth
		{"node_modules", true, true},
		{"foo/node_modules", true, true},
		{"node_modules.bak", false, false},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got := g.Match(tt.path, tt.isDir)
			if got != tt.want {
				t.Errorf("Match(%q, isDir=%v) = %v, want %v", tt.path, tt.isDir, got, tt.want)
			}
		})
	}
}

func TestGitignore_NilSafe(t *testing.T) {
	var g *Gitignore
	if g.Match("anything", false) {
		t.Errorf("nil Gitignore should return false from Match")
	}
}

func TestLoadGitignore_MissingFileIsNotError(t *testing.T) {
	dir := t.TempDir()
	g, err := LoadGitignore(filepath.Join(dir, "nonexistent.gitignore"))
	if err != nil {
		t.Fatalf("missing file should not error, got %v", err)
	}
	if g != nil {
		t.Errorf("missing file should return nil, got %+v", g)
	}
}

func TestLoadGitignore_RealFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".gitignore")
	body := "*.log\nbuild/\n# a comment\n\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	g, err := LoadGitignore(path)
	if err != nil {
		t.Fatalf("LoadGitignore: %v", err)
	}
	if g == nil {
		t.Fatal("got nil, want a parsed Gitignore")
	}
	if !g.Match("a.log", false) {
		t.Errorf("expected a.log to be ignored")
	}
	if !g.Match("build", true) {
		t.Errorf("expected build/ to be ignored")
	}
	if g.Match("main.go", false) {
		t.Errorf("main.go should not be ignored")
	}
}
