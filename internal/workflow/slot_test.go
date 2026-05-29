package workflow

import (
	"testing"

	"github.com/ashish-work/opendev-go/internal/provider"
)

func TestResolve(t *testing.T) {
	exec := SlotConfig{Model: "gpt-4o"}

	tests := []struct {
		name string
		cfg  Config
		slot Slot
		want string
	}{
		{
			name: "execution returns its own model",
			cfg:  Config{Execution: exec},
			slot: SlotExecution,
			want: "gpt-4o",
		},
		{
			name: "compact unset falls back to execution",
			cfg:  Config{Execution: exec},
			slot: SlotCompact,
			want: "gpt-4o",
		},
		{
			name: "compact set overrides execution",
			cfg:  Config{Execution: exec, Compact: SlotConfig{Model: "gpt-4o-mini"}},
			slot: SlotCompact,
			want: "gpt-4o-mini",
		},
		{
			name: "thinking set, compact still falls back to execution",
			cfg:  Config{Execution: exec, Thinking: SlotConfig{Model: "o1"}},
			slot: SlotCompact,
			want: "gpt-4o",
		},
		{
			name: "critique set overrides",
			cfg:  Config{Execution: exec, Critique: SlotConfig{Model: "claude-sonnet"}},
			slot: SlotCritique,
			want: "claude-sonnet",
		},
		{
			name: "vlm set overrides",
			cfg:  Config{Execution: exec, VLM: SlotConfig{Model: "gpt-4o-vision"}},
			slot: SlotVLM,
			want: "gpt-4o-vision",
		},
		{
			name: "unknown slot falls back to execution",
			cfg:  Config{Execution: exec},
			slot: Slot(42),
			want: "gpt-4o",
		},
		{
			name: "zero config resolves to empty model",
			cfg:  Config{},
			slot: SlotCompact,
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.cfg.Resolve(tt.slot).Model
			if got != tt.want {
				t.Errorf("Resolve(%v).Model = %q, want %q", tt.slot, got, tt.want)
			}
		})
	}
}

func TestSlotString(t *testing.T) {
	cases := []struct {
		slot Slot
		want string
	}{
		{SlotExecution, "execution"},
		{SlotThinking, "thinking"},
		{SlotCompact, "compact"},
		{SlotCritique, "critique"},
		{SlotVLM, "vlm"},
		{Slot(99), "unknown"},
	}
	for _, c := range cases {
		if got := c.slot.String(); got != c.want {
			t.Errorf("Slot(%d).String() = %q, want %q", c.slot, got, c.want)
		}
	}
}

func TestResolve_ReasoningEffortFallback(t *testing.T) {
	// Verifies SlotConfig.ReasoningEffort follows the same
	// fall-back-to-Execution rule that Model does. A slot whose
	// effort is Unset inherits Execution's effort; an explicit
	// value (including ReasoningEffortNone) wins over the default.
	exec := SlotConfig{
		Model:           "claude-opus-4-7",
		ReasoningEffort: provider.ReasoningEffortHigh,
	}

	cases := []struct {
		name string
		cfg  Config
		slot Slot
		want provider.ReasoningEffort
	}{
		{
			name: "execution returns its own effort verbatim",
			cfg:  Config{Execution: exec},
			slot: SlotExecution,
			want: provider.ReasoningEffortHigh,
		},
		{
			name: "compact unset falls back to execution's high",
			cfg:  Config{Execution: exec},
			slot: SlotCompact,
			want: provider.ReasoningEffortHigh,
		},
		{
			name: "compact explicit low wins over execution high",
			cfg: Config{
				Execution: exec,
				Compact:   SlotConfig{ReasoningEffort: provider.ReasoningEffortLow},
			},
			slot: SlotCompact,
			want: provider.ReasoningEffortLow,
		},
		{
			name: "compact explicit None wins — explicit suppression",
			cfg: Config{
				Execution: exec,
				Compact:   SlotConfig{ReasoningEffort: provider.ReasoningEffortNone},
			},
			slot: SlotCompact,
			want: provider.ReasoningEffortNone,
		},
		{
			name: "unset Execution + unset slot resolves to unset",
			cfg:  Config{Execution: SlotConfig{Model: "x"}},
			slot: SlotThinking,
			want: provider.ReasoningEffortUnset,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.cfg.Resolve(tc.slot).ReasoningEffort
			if got != tc.want {
				t.Errorf("Resolve(%v).ReasoningEffort = %q, want %q",
					tc.slot, got, tc.want)
			}
		})
	}
}
