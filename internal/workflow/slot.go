// Package workflow defines the five "slots" — named model roles — the
// agent can target for different phases of work. Each slot can be
// bound to its own model; unset slots fall back to Execution's model.
//
// The slots are modeled as a typed iota enum rather than a string-keyed
// map, because:
//
//   - The slot universe is closed (5 known phases). A typo on a slot
//     name is a programming error, not a config error — catch at compile.
//   - Comma-ok lookups on a map would litter every callsite with
//     `if !ok { fallback }`. A method with built-in fallback is cleaner.
//
// v1 only wires SlotExecution into the ReactLoop. The other four are
// defined so that later features (tool-result summarization, critique
// and thinking phases, vision support) can request them without a
// config-shape migration.
package workflow

// Slot identifies one of the five model roles the agent can dispatch
// against. Use the constants below — never construct Slot values from
// arbitrary ints.
type Slot int

const (
	// SlotExecution is the primary slot. The ReactLoop calls this for
	// every tool-using turn. It is the only slot that MUST be configured;
	// every other slot falls back to it.
	SlotExecution Slot = iota

	// SlotThinking is reserved for a future "private reasoning" phase
	// (deferred from v1's collapsed single-phase loop). Distinct from
	// Execution because thinking models (e.g. o1) may differ from the
	// model that's good at tool use.
	SlotThinking

	// SlotCompact is used to summarize large tool results or compact
	// history before it grows past the context window. Often a cheap
	// model — accuracy matters less than throughput.
	SlotCompact

	// SlotCritique is used by the deferred dual-agent flow to second-
	// guess the Execution model's answer before returning to the user.
	SlotCritique

	// SlotVLM is the vision/multi-modal slot — used when a tool result
	// includes an image the agent needs to interpret.
	SlotVLM
)

// String returns the slot's lower-case name. Used in logs and tests.
// Unknown values render as "unknown" rather than panicking — Slot is
// not user-facing data, so a typo crashes nothing.
func (s Slot) String() string {
	switch s {
	case SlotExecution:
		return "execution"
	case SlotThinking:
		return "thinking"
	case SlotCompact:
		return "compact"
	case SlotCritique:
		return "critique"
	case SlotVLM:
		return "vlm"
	default:
		return "unknown"
	}
}

// SlotConfig holds the per-slot knobs. v1 carries only Model; future
// fields (BaseURL, Temperature, MaxTokens) are intentionally deferred
// until a concrete task needs them — adding a field is a one-line
// change that does not break existing zero-value Configs.
type SlotConfig struct {
	// Model is the provider-specific identifier (e.g. "gpt-4o-mini").
	// Empty means "fall back to Execution.Model" per Config.Resolve.
	Model string
}

// Config bundles all five slots. The zero value is usable but useless —
// Execution.Model must be set for the loop to issue any calls.
type Config struct {
	Execution SlotConfig
	Thinking  SlotConfig
	Compact   SlotConfig
	Critique  SlotConfig
	VLM       SlotConfig
}

// Resolve returns the SlotConfig for the given slot, with any empty
// field defaulted from Execution. SlotExecution returns its own config
// verbatim — there is nothing to fall back to.
//
// Examples:
//
//	cfg := Config{Execution: SlotConfig{Model: "gpt-4o"}}
//	cfg.Resolve(SlotCompact).Model // => "gpt-4o" (defaulted)
//
//	cfg := Config{
//	    Execution: SlotConfig{Model: "gpt-4o"},
//	    Compact:   SlotConfig{Model: "gpt-4o-mini"},
//	}
//	cfg.Resolve(SlotCompact).Model // => "gpt-4o-mini" (explicit)
//
// Unknown Slot values return Execution as a safe default.
func (c Config) Resolve(slot Slot) SlotConfig {
	var s SlotConfig
	switch slot {
	case SlotExecution:
		return c.Execution
	case SlotThinking:
		s = c.Thinking
	case SlotCompact:
		s = c.Compact
	case SlotCritique:
		s = c.Critique
	case SlotVLM:
		s = c.VLM
	default:
		return c.Execution
	}
	if s.Model == "" {
		s.Model = c.Execution.Model
	}
	return s
}
