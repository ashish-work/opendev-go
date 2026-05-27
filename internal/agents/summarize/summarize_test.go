package summarize

import (
	"strings"
	"testing"
)

func TestSummarize_PassThroughUnderThreshold(t *testing.T) {
	tests := []struct {
		name string
		text string
	}{
		{"short read_file", "package main\nimport \"fmt\"\n"},
		{"short bash", "hello world\n"},
		{"short edit", "edited 1 occurrence"},
		{"exactly at threshold", strings.Repeat("a", DefaultThreshold)},
		{"error message verbatim", "[ERROR] file not found"},
		{"empty stays empty", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Summarize("read_file", tt.text)
			if got.Truncated {
				t.Errorf("Truncated = true, want false for input len=%d", len(tt.text))
			}
			if got.Text != tt.text {
				t.Errorf("Text = %q, want %q", got.Text, tt.text)
			}
		})
	}
}

func TestSummarize_SummarizesOverThreshold(t *testing.T) {
	// Build a payload comfortably above DefaultThreshold.
	big := strings.Repeat("line of output\n", 1000) // ~15KB, 1000 lines

	tests := []struct {
		toolName    string
		wantPrefix  string
		wantSuffix  string
		wantSubstr  string // alternative: substring check
		description string
	}{
		{
			toolName:    "read_file",
			wantPrefix:  "Read file (",
			wantSubstr:  "1000 lines",
			description: "read_file dispatches to file summary",
		},
		{
			toolName:    "bash",
			wantPrefix:  "Command executed (",
			wantSubstr:  "1000 lines",
			description: "bash dispatches to command summary",
		},
		{
			toolName:    "edit_file",
			wantPrefix:  "File edited successfully",
			description: "edit_file is always the same one-liner",
		},
		{
			toolName:    "mystery_tool",
			wantPrefix:  "Success (",
			wantSubstr:  "1000 lines",
			description: "unknown tools get generic Success summary",
		},
	}
	for _, tt := range tests {
		t.Run(tt.description, func(t *testing.T) {
			got := Summarize(tt.toolName, big)
			if !got.Truncated {
				t.Errorf("Truncated = false, want true for input len=%d", len(big))
			}
			if !strings.HasPrefix(got.Text, tt.wantPrefix) {
				t.Errorf("Text = %q, want prefix %q", got.Text, tt.wantPrefix)
			}
			if tt.wantSubstr != "" && !strings.Contains(got.Text, tt.wantSubstr) {
				t.Errorf("Text = %q, want substring %q", got.Text, tt.wantSubstr)
			}
			// Summary must be radically shorter than input — otherwise
			// what's the point.
			if len(got.Text) > 200 {
				t.Errorf("summary length = %d, want < 200", len(got.Text))
			}
		})
	}
}

func TestSummarizeWith_CustomThreshold(t *testing.T) {
	// threshold=0 forces summarization on any non-empty input.
	got := SummarizeWith("bash", "tiny output\n", 0)
	if !got.Truncated {
		t.Error("Truncated = false, want true (threshold=0)")
	}
	if !strings.HasPrefix(got.Text, "Command executed (") {
		t.Errorf("Text = %q, want bash summary", got.Text)
	}

	// huge threshold leaves everything verbatim.
	huge := strings.Repeat("x\n", 10000)
	got = SummarizeWith("read_file", huge, 1<<20)
	if got.Truncated {
		t.Error("Truncated = true, want false (threshold above input)")
	}
	if got.Text != huge {
		t.Error("Text differs from input — should be verbatim")
	}
}

func TestCountLines(t *testing.T) {
	cases := []struct {
		input string
		want  int
	}{
		{"", 0},
		{"a", 1},
		{"a\n", 1},
		{"a\nb", 2},
		{"a\nb\n", 2},
		{"a\nb\nc", 3},
		{"\n", 1},     // single newline → 1 empty line
		{"\n\n", 2},   // two newlines → 2 empty lines
		{"x\n\ny", 3}, // text with blank line between
	}
	for _, c := range cases {
		if got := countLines(c.input); got != c.want {
			t.Errorf("countLines(%q) = %d, want %d", c.input, got, c.want)
		}
	}
}

func TestSummary_DoesNotPanicOnUTF8(t *testing.T) {
	// Multi-byte chars in the input should not cause boundary errors,
	// because the summary is built fresh from format strings — it
	// never indexes into the input.
	huge := strings.Repeat("日本語\n", 2000) // ~18KB, all multi-byte
	got := Summarize("read_file", huge)
	if !got.Truncated {
		t.Error("Truncated = false, want true for huge UTF-8 input")
	}
	if !strings.Contains(got.Text, "Read file (") {
		t.Errorf("Text = %q, want read_file summary", got.Text)
	}
}
