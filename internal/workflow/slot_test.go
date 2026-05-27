package workflow

import "testing"

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
