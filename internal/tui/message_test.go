package tui

import (
	"strings"
	"testing"
)

func TestRoleStyle_Labels(t *testing.T) {
	cases := []struct {
		name     string
		role     messageRole
		toolName string
		want     string
	}{
		{"user", roleUser, "", "you"},
		{"assistant", roleAssistant, "", "assistant"},
		{"tool no name", roleTool, "", "tool"},
		{"tool with name", roleTool, "bash", "tool: bash"},
		{"unknown falls back", messageRole(99), "", "?"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			header, _, _ := roleStyle(c.role, c.toolName)
			if header != c.want {
				t.Errorf("roleStyle(%v, %q) header = %q, want %q",
					c.role, c.toolName, header, c.want)
			}
		})
	}
}

func TestRoleStyle_StylesAreDistinct(t *testing.T) {
	// The three roles should produce three different rendered
	// headers. We compare the rendered output (with ANSI codes) so
	// any color/bold/italic difference shows up. If all three roles
	// rendered identically, the user couldn't tell them apart.
	userHdr, userStyle, _ := roleStyle(roleUser, "")
	asstHdr, asstStyle, _ := roleStyle(roleAssistant, "")
	toolHdr, toolStyle, _ := roleStyle(roleTool, "bash")

	renderedUser := userStyle.Render("▎ " + userHdr)
	renderedAsst := asstStyle.Render("▎ " + asstHdr)
	renderedTool := toolStyle.Render("▎ " + toolHdr)

	if renderedUser == renderedAsst {
		t.Errorf("user and assistant rendered identically — colors not distinct")
	}
	if renderedUser == renderedTool {
		t.Errorf("user and tool rendered identically — colors not distinct")
	}
	if renderedAsst == renderedTool {
		t.Errorf("assistant and tool rendered identically — colors not distinct")
	}
}

func TestViewMessage_Render_IncludesHeaderAndBody(t *testing.T) {
	m := viewMessage{role: roleUser, content: "hello world"}
	out := m.render(80, false)
	if !strings.Contains(out, "you") {
		t.Errorf("output should contain role header, got:\n%s", out)
	}
	if !strings.Contains(out, "hello world") {
		t.Errorf("output should contain body content, got:\n%s", out)
	}
	if !strings.Contains(out, "\n") {
		t.Errorf("output should be multi-line (header \\n body), got:\n%s", out)
	}
}

func TestViewMessage_Render_ToolIncludesName(t *testing.T) {
	m := viewMessage{role: roleTool, toolName: "read_file", content: "..."}
	out := m.render(80, false)
	if !strings.Contains(out, "tool: read_file") {
		t.Errorf("tool message should show 'tool: <name>' in header, got:\n%s", out)
	}
}

func TestViewMessage_Render_WrapsLongContent(t *testing.T) {
	long := strings.Repeat("word ", 50) // ~250 chars
	m := viewMessage{role: roleUser, content: long}
	out := m.render(40, false)
	// The body must wrap — output should have at least 3 lines
	// (header + multiple body lines).
	lines := strings.Split(out, "\n")
	if len(lines) < 3 {
		t.Errorf("long content should wrap into multiple lines, got %d:\n%s",
			len(lines), out)
	}
	// Each line should fit within a reasonable bound. We can't check
	// the wrap width exactly (ANSI escape sequences inflate byte
	// length), but no body line should be wildly longer than the
	// target.
	for i, line := range lines[1:] { // skip header
		// Strip ANSI for length check. Quick-and-dirty: count
		// runes between ESC[...m sequences. For a sanity test
		// the rough bound below is enough.
		if len(line) > 200 {
			t.Errorf("body line %d unexpectedly long (%d chars): %q",
				i+1, len(line), line)
		}
	}
}

func TestViewMessage_Render_TinyWidthDoesNotPanic(t *testing.T) {
	// Width below the min body width should clamp, not panic, not
	// produce a 0-width style call.
	m := viewMessage{role: roleAssistant, content: "tiny"}
	out := m.render(3, false) // way below minBodyWidth
	if out == "" {
		t.Errorf("render should produce output even at tiny width")
	}
	if !strings.Contains(out, "tiny") {
		t.Errorf("body should still appear: %q", out)
	}
}

func TestViewMessage_Render_EmptyContent(t *testing.T) {
	m := viewMessage{role: roleUser, content: ""}
	out := m.render(80, false)
	// Header should still appear; body is empty but the function
	// shouldn't choke or omit the header.
	if !strings.Contains(out, "you") {
		t.Errorf("header should render even with empty content: %q", out)
	}
}

// ---- Tool-card rendering ----

func TestViewMessage_Render_ToolShortContentNoHint(t *testing.T) {
	m := viewMessage{
		role:     roleTool,
		toolName: "bash",
		content:  "one line of output",
	}
	out := m.render(80, false) // collapsed, but content fits
	if !strings.Contains(out, "tool: bash") {
		t.Errorf("expected tool header: %q", out)
	}
	if !strings.Contains(out, "one line of output") {
		t.Errorf("expected body: %q", out)
	}
	if strings.Contains(out, "more line") {
		t.Errorf("short content should NOT trigger truncation hint: %q", out)
	}
}

func TestViewMessage_Render_ToolLongContentTruncatesWithHint(t *testing.T) {
	body := strings.Join([]string{
		"line 1", "line 2", "line 3",
		"line 4", "line 5", "line 6",
	}, "\n")
	m := viewMessage{role: roleTool, toolName: "bash", content: body}
	out := m.render(80, false) // collapsed
	if !strings.Contains(out, "line 1") || !strings.Contains(out, "line 3") {
		t.Errorf("first 3 lines should be visible:\n%s", out)
	}
	if strings.Contains(out, "line 4") || strings.Contains(out, "line 6") {
		t.Errorf("collapsed render should NOT include lines past the limit:\n%s", out)
	}
	if !strings.Contains(out, "3 more lines") {
		t.Errorf("expected 'N more lines' hint mentioning 3:\n%s", out)
	}
	if !strings.Contains(out, "Ctrl-T") {
		t.Errorf("hint should mention Ctrl-T expand control:\n%s", out)
	}
}

func TestViewMessage_Render_ToolExpandedShowsAllContent(t *testing.T) {
	body := strings.Join([]string{
		"line 1", "line 2", "line 3", "line 4", "line 5",
	}, "\n")
	m := viewMessage{role: roleTool, toolName: "bash", content: body}
	out := m.render(80, true) // expanded
	for _, want := range []string{"line 1", "line 2", "line 3", "line 4", "line 5"} {
		if !strings.Contains(out, want) {
			t.Errorf("expanded render should include %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "more line") {
		t.Errorf("expanded render should NOT show truncation hint:\n%s", out)
	}
}

func TestViewMessage_Render_ToolHasBorder(t *testing.T) {
	// The card uses lipgloss's rounded border. The exact rendered
	// bytes are ANSI-heavy but the corner glyphs ╭ and ╰ should be
	// present in the output regardless of theme.
	m := viewMessage{role: roleTool, toolName: "x", content: "hi"}
	out := m.render(80, false)
	if !strings.Contains(out, "╭") || !strings.Contains(out, "╰") {
		t.Errorf("tool card should have rounded-border corners:\n%s", out)
	}
}

func TestViewMessage_Render_NonToolIgnoresExpanded(t *testing.T) {
	// User and assistant messages should look identical whether
	// expanded is true or false. Only tool messages respond to the
	// flag.
	m := viewMessage{role: roleAssistant, content: "long " + strings.Repeat("text ", 30)}
	collapsed := m.render(80, false)
	expanded := m.render(80, true)
	if collapsed != expanded {
		t.Errorf("non-tool render should not depend on expanded flag")
	}
}

func TestViewMessage_Render_ToolTinyWidthDoesNotPanic(t *testing.T) {
	m := viewMessage{role: roleTool, toolName: "x", content: "hi"}
	out := m.render(2, false) // way smaller than card overhead
	if out == "" {
		t.Errorf("tiny-width tool render should still produce something")
	}
}

// ---- truncateToLines ----

func TestTruncateToLines(t *testing.T) {
	cases := []struct {
		name      string
		in        string
		max       int
		want      string
		wantExtra int
	}{
		{"empty", "", 3, "", 0},
		{"one line under cap", "only", 3, "only", 0},
		{"exactly cap", "a\nb\nc", 3, "a\nb\nc", 0},
		{"one over cap", "a\nb\nc\nd", 3, "a\nb\nc", 1},
		{"many over cap", "a\nb\nc\nd\ne\nf\ng", 2, "a\nb", 5},
		{"max zero with content", "a\nb", 0, "", 2},
		{"max zero empty", "", 0, "", 0},
		{"max negative", "a", -5, "", 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, extra := truncateToLines(c.in, c.max)
			if got != c.want {
				t.Errorf("preview = %q, want %q", got, c.want)
			}
			if extra != c.wantExtra {
				t.Errorf("extra = %d, want %d", extra, c.wantExtra)
			}
		})
	}
}

func TestPlural(t *testing.T) {
	if plural(1) != "" {
		t.Errorf("plural(1) should be empty, got %q", plural(1))
	}
	if plural(0) != "s" || plural(2) != "s" || plural(99) != "s" {
		t.Errorf("plural(non-1) should be 's'")
	}
}
