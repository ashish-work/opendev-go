package listfiles

// DefaultExcludeDirs are directory names that almost always represent
// dependency, cache, build, or VCS noise. The walker skips any path
// that has a directory component matching one of these. The list
// covers 20+ ecosystems (Node, Python, Rust, Go, Java, Ruby, Erlang,
// Haskell, OCaml, .NET, mobile, frameworks). Exported so consumers or
// tests can extend or replace the policy.
var DefaultExcludeDirs = []string{
	// Package / dependency directories
	"node_modules",
	"bower_components",
	"jspm_packages",
	"vendor",
	"Pods",
	".bundle",
	"packages",
	".pub-cache",
	".pub",
	"deps",
	".nuget",
	".m2",

	// Virtual environments
	".venv",
	"venv",
	".virtualenvs",
	".conda",

	// Build output
	"build",
	"dist",
	"out",
	"target",
	"bin",
	"obj",
	"lib",
	"_build",
	"ebin",
	"dist-newstyle",
	".build",
	"DerivedData",
	"CMakeFiles",
	".cmake",

	// Framework-specific build/cache
	".next",
	".nuxt",
	".angular",
	".svelte-kit",
	".vuepress",
	".gatsby-cache",
	".parcel-cache",
	".turbo",
	"dist_electron",

	// Caches
	".cache",
	"__pycache__",
	".pytest_cache",
	".mypy_cache",
	".ruff_cache",
	".hypothesis",
	".tox",
	".nox",
	".eslintcache",
	".stylelintcache",
	".gradle",
	".dart_tool",
	".mix",
	".cpcache",
	".lsp",

	// IDE / editor
	".idea",
	".vscode",
	".vscode-test",
	".vs",
	".metadata",
	".settings",
	"xcuserdata",
	".netbeans",

	// VCS
	".git",
	".svn",
	".hg",

	// Coverage / testing output
	"coverage",
	"htmlcov",
	".nyc_output",

	// Language-specific metadata
	".eggs",
	".Rproj.user",
	".julia",
	"_opam",
	".cabal-sandbox",
	".stack-work",
	"blib",
}

// DefaultExcludeFileGlobs are file-name globs that match generated or
// machine-only artifacts. Matched per-basename (e.g. *.min.js matches
// foo.min.js regardless of directory). Exported for the same reason
// as DefaultExcludeDirs — callers can tune the policy.
var DefaultExcludeFileGlobs = []string{
	"*.min.js",
	"*.min.css",
	"*.bundle.js",
	"*.chunk.js",
	"*.map",
	"*.pyc",
	"*.pyo",
	"*.class",
	"*.o",
	"*.so",
	"*.dylib",
	"*.dll",
	"*.exe",
	"*.beam",
	"*.hi",
	"*.dyn_hi",
	"*.dyn_o",
	"*.egg-info",
}
