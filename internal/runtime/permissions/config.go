package permissions

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"github.com/ashish-work/opendev-go/internal/hooks"
)

// ErrInvalidConfig wraps every loader failure so callers can
// errors.Is-classify "the settings file is bad" without coupling
// to the specific message. Returned for unreadable files, malformed
// JSON, and uncompilable regex patterns.
var ErrInvalidConfig = errors.New("permissions: invalid config")

// rawFile mirrors the slice of settings.json this package owns.
// The hooks package owns the "hooks" key; permissions owns
// "permissions". Both keys are independent — neither requires the
// other to be present.
type rawFile struct {
	Permissions map[string]rawToolPermission `json:"permissions,omitempty"`
}

// rawToolPermission is the JSON shape for one tool's policy entry.
// Enabled is a pointer so the loader can distinguish "field omitted"
// (defaults to true) from "explicitly false" (the tool is fully
// disabled). Without this distinction a user who wrote
// {"bash": {"deny_patterns": ["..."]}} would accidentally disable
// bash entirely via Go's zero value — a footgun worth two extra
// allocations.
type rawToolPermission struct {
	Enabled      *bool    `json:"enabled,omitempty"`
	AlwaysAllow  bool     `json:"always_allow,omitempty"`
	DenyPatterns []string `json:"deny_patterns,omitempty"`
}

// LoadFile reads a single settings.json file and returns the
// permissions section. Missing files are not errors — permissions
// are opt-in and most users won't have a settings file at all.
// Malformed JSON, invalid regexes, and unreadable files all surface
// with a wrapped ErrInvalidConfig naming the offending entry.
//
// Compile-on-load: every deny_pattern is regex-compiled here so a
// bad pattern fails at startup with the user's exact pattern source
// in the error, rather than failing silently or mid-tool-dispatch
// when an LLM happens to trigger it.
//
// Empty or absent "permissions" key returns an empty Policy with
// no error. That's the v1-compatible default.
func LoadFile(path string) (Policy, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return Policy{}, nil
	}
	if err != nil {
		return Policy{}, fmt.Errorf(
			"%w: read %s: %v", ErrInvalidConfig, path, err,
		)
	}

	var file rawFile
	if err := json.Unmarshal(data, &file); err != nil {
		return Policy{}, fmt.Errorf(
			"%w: parse %s: %v", ErrInvalidConfig, path, err,
		)
	}

	if len(file.Permissions) == 0 {
		return Policy{}, nil
	}

	out := Policy{Tools: make(map[string]ToolPermission, len(file.Permissions))}
	for toolName, raw := range file.Permissions {
		perm, err := buildToolPermission(path, toolName, raw)
		if err != nil {
			return Policy{}, err
		}
		out.Tools[toolName] = perm
	}
	return out, nil
}

// buildToolPermission validates and compiles one tool entry. The
// path and toolName arguments are folded into error messages so a
// failure tells the user exactly which entry is bad.
func buildToolPermission(
	path, toolName string, raw rawToolPermission,
) (ToolPermission, error) {
	// Enabled defaulting: nil pointer → true. The user who writes
	// just {"deny_patterns": [...]} clearly wants the tool to run
	// (otherwise why specify deny patterns at all). Explicit
	// "enabled": false is the only way to fully disable a tool.
	enabled := true
	if raw.Enabled != nil {
		enabled = *raw.Enabled
	}

	perm := ToolPermission{
		Enabled:     enabled,
		AlwaysAllow: raw.AlwaysAllow,
	}

	if len(raw.DenyPatterns) > 0 {
		perm.DenyPatterns = make([]*regexp.Regexp, 0, len(raw.DenyPatterns))
		perm.denyPatternSources = make([]string, 0, len(raw.DenyPatterns))
		for i, src := range raw.DenyPatterns {
			re, err := regexp.Compile(src)
			if err != nil {
				return ToolPermission{}, fmt.Errorf(
					"%w: %s: %s.deny_patterns[%d]: invalid pattern %q: %v",
					ErrInvalidConfig, path, toolName, i, src, err,
				)
			}
			perm.DenyPatterns = append(perm.DenyPatterns, re)
			perm.denyPatternSources = append(perm.denyPatternSources, src)
		}
	}

	return perm, nil
}

// Merge combines two policies. Project entries REPLACE user entries
// for the same tool name (whole-entry replacement, not field-merge).
// Unlike hooks (where project matchers append to user matchers), a
// permissions policy is a whole "this is how this tool is allowed
// to run" decision; partial inheritance would muddy intent — e.g.
// a user setting AlwaysAllow=true plus a project deny_pattern would
// have surprising precedence either way. Whole-entry replacement
// is the predictable rule.
//
// Tools present in only one policy pass through unchanged.
// The result is a new Policy; neither input is mutated.
func Merge(user, project Policy) Policy {
	merged := Policy{Tools: map[string]ToolPermission{}}
	for name, perm := range user.Tools {
		merged.Tools[name] = perm
	}
	for name, perm := range project.Tools {
		merged.Tools[name] = perm
	}
	return merged
}

// Load is the convenience entry point: read user-wide settings,
// read project settings, merge. Returns Policy{} when neither file
// exists or neither has a "permissions" key — the default-allow
// behavior the v1 loop already has.
//
// Settings paths are sourced from the hooks package so the
// .opendev/settings.json convention stays single-source. Permissions
// has no other coupling to hooks.
func Load(workingDir string) (Policy, error) {
	user, err := LoadFile(userSettingsPath())
	if err != nil {
		return Policy{}, err
	}
	project, err := LoadFile(projectSettingsPath(workingDir))
	if err != nil {
		return Policy{}, err
	}
	return Merge(user, project), nil
}

// userSettingsPath wraps the hooks-side helper so test code in this
// package can override the home-dir resolution without poking
// hooks's internals. Pure delegation today; gives us a seam later.
func userSettingsPath() string {
	return hooks.UserSettingsPath()
}

// projectSettingsPath mirrors userSettingsPath. The hooks package
// owns the constants (SettingsDirName, SettingsFileName) so the
// convention is defined in exactly one place.
func projectSettingsPath(workingDir string) string {
	return filepath.Join(workingDir, hooks.SettingsDirName, hooks.SettingsFileName)
}
