package hooks

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
)

// SettingsDirName is the directory under the user's home and under a
// project root that holds the agent's settings file. Kept as a
// constant so tests don't need to hard-code the path twice.
const SettingsDirName = ".opendev"

// SettingsFileName is the JSON file under SettingsDirName that
// configures hooks (and, in later commits, permissions and other
// runtime settings).
const SettingsFileName = "settings.json"

// HookMatcher is one row inside a "hooks: { event: [...] }" array.
// Holds the user's regex source, the shell command, and an optional
// per-hook timeout override.
//
// The compiled regex lives on a private field. Public access goes
// through Matches(s) so the executor doesn't poke internals — also
// guarantees compile-on-load semantics (the manager doesn't need to
// know to compile; it just calls Matches).
type HookMatcher struct {
	// Matcher is the regex source as written in settings.json.
	// Applied to the event's primary identifier (tool name for
	// PreToolUse, prompt text for UserPromptSubmit, etc.). Empty
	// means "always match" — Matches returns true for any input.
	Matcher string `json:"matcher,omitempty"`

	// Command is the shell command to run when this matcher fires.
	// Required; the loader rejects entries that omit it.
	Command string `json:"command"`

	// TimeoutMs is the per-hook timeout override in milliseconds.
	// Zero (or omitted) means "use the manager's default" — that
	// default lands in #33.
	TimeoutMs int `json:"timeout_ms,omitempty"`

	// compiled is the parsed Matcher regex. nil when Matcher is
	// empty (always-match path). Populated by LoadFile so the
	// executor never sees a raw string it has to compile.
	compiled *regexp.Regexp
}

// Matches reports whether the matcher's regex applies to s. Empty
// matchers always match; non-empty matchers use the compiled regex.
func (m HookMatcher) Matches(s string) bool {
	if m.compiled == nil {
		return true
	}
	return m.compiled.MatchString(s)
}

// HookSettings is the typed in-memory representation of a settings
// file's "hooks" section. Keys are HookEvent values from #30; values
// are the matchers registered for that event in declaration order.
//
// Unknown event names from the source JSON are logged and dropped at
// load time so the in-memory shape only ever has valid event keys —
// callers don't need to filter.
type HookSettings struct {
	Hooks map[HookEvent][]HookMatcher
}

// MatchersFor returns the matchers registered for the given event,
// or nil if no hooks were registered for it. Returning nil (rather
// than an empty slice) lets the manager skip the dispatch loop with
// one nil check instead of two.
func (s HookSettings) MatchersFor(event HookEvent) []HookMatcher {
	if s.Hooks == nil {
		return nil
	}
	return s.Hooks[event]
}

// fileShape mirrors the on-disk JSON. Top-level keys other than
// "hooks" are ignored — this loader is the hooks loader only.
// Permissions and other settings sections land in their own
// loaders (Phase 8 introduces permissions).
type fileShape struct {
	Hooks map[string][]rawMatcher `json:"hooks,omitempty"`
}

// rawMatcher is the file-shape twin of HookMatcher. Separate because
// HookMatcher carries the compiled regex (which can't unmarshal from
// JSON), and we want explicit control over the file → typed
// translation.
type rawMatcher struct {
	Matcher   string `json:"matcher,omitempty"`
	Command   string `json:"command"`
	TimeoutMs int    `json:"timeout_ms,omitempty"`
}

// UserSettingsPath returns the path of the user-wide settings file
// (~/.opendev/settings.json). Pure string assembly — no I/O. Returns
// the empty string if the home directory can't be determined, which
// the caller can treat as "no user settings."
func UserSettingsPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, SettingsDirName, SettingsFileName)
}

// ProjectSettingsPath returns the project-specific settings file
// path for the given working directory: <workingDir>/.opendev/
// settings.json. Pure string assembly — no I/O.
func ProjectSettingsPath(workingDir string) string {
	return filepath.Join(workingDir, SettingsDirName, SettingsFileName)
}

// LoadFile reads one settings file and returns the parsed
// HookSettings. Missing files are not errors — hooks are opt-in and
// most users never create the file. Permission errors, malformed
// JSON, invalid regex, missing commands, and negative timeouts all
// surface with messages naming the offending entry.
//
// Compile-on-load: each non-empty regex is compiled here so a bad
// pattern fails at startup rather than mid-tool-dispatch when the
// hook would otherwise fire.
func LoadFile(path string) (HookSettings, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return HookSettings{}, nil
	}
	if err != nil {
		return HookSettings{}, fmt.Errorf("hooks: read %s: %w", path, err)
	}

	var file fileShape
	if err := json.Unmarshal(data, &file); err != nil {
		return HookSettings{}, fmt.Errorf("hooks: parse %s: %w", path, err)
	}

	out := HookSettings{Hooks: map[HookEvent][]HookMatcher{}}
	for rawName, rawList := range file.Hooks {
		event, ok := Parse(rawName)
		if !ok {
			// Unknown event name — most likely the user has a
			// config from a newer version. Log and skip; don't
			// fail.
			slog.Warn("hooks: unknown event name in settings; skipping",
				"path", path, "event", rawName)
			continue
		}
		matchers, err := compileMatchers(path, rawName, rawList)
		if err != nil {
			return HookSettings{}, err
		}
		out.Hooks[event] = matchers
	}

	return out, nil
}

// compileMatchers validates and compiles a slice of raw matchers
// for one event. The path + event arguments are folded into error
// messages so a failure tells the user exactly which entry is bad.
func compileMatchers(path, eventName string, raws []rawMatcher) ([]HookMatcher, error) {
	out := make([]HookMatcher, 0, len(raws))
	for i, raw := range raws {
		if raw.Command == "" {
			return nil, fmt.Errorf(
				"hooks: %s: %s[%d]: command is required",
				path, eventName, i,
			)
		}
		if raw.TimeoutMs < 0 {
			return nil, fmt.Errorf(
				"hooks: %s: %s[%d]: timeout_ms must be >= 0 (got %d)",
				path, eventName, i, raw.TimeoutMs,
			)
		}
		m := HookMatcher{
			Matcher:   raw.Matcher,
			Command:   raw.Command,
			TimeoutMs: raw.TimeoutMs,
		}
		if raw.Matcher != "" {
			re, err := regexp.Compile(raw.Matcher)
			if err != nil {
				return nil, fmt.Errorf(
					"hooks: %s: %s[%d]: invalid matcher %q: %w",
					path, eventName, i, raw.Matcher, err,
				)
			}
			m.compiled = re
		}
		out = append(out, m)
	}
	return out, nil
}

// Merge combines two HookSettings. For each event, project matchers
// come AFTER user matchers — the matcher iteration order in #33's
// dispatch is the execution order, so a user-level "always audit
// bash" hook runs before a project's tighter rule. Either input
// being empty is fine.
//
// The result is a new HookSettings; neither input is mutated.
func Merge(user, project HookSettings) HookSettings {
	merged := HookSettings{Hooks: map[HookEvent][]HookMatcher{}}
	for event, matchers := range user.Hooks {
		merged.Hooks[event] = append([]HookMatcher{}, matchers...)
	}
	for event, matchers := range project.Hooks {
		merged.Hooks[event] = append(merged.Hooks[event], matchers...)
	}
	return merged
}

// Load is the convenience entry point both binaries use: read the
// user-wide settings, read the project settings, merge them. Returns
// an empty HookSettings if neither file exists.
func Load(workingDir string) (HookSettings, error) {
	user, err := LoadFile(UserSettingsPath())
	if err != nil {
		return HookSettings{}, err
	}
	project, err := LoadFile(ProjectSettingsPath(workingDir))
	if err != nil {
		return HookSettings{}, err
	}
	return Merge(user, project), nil
}
