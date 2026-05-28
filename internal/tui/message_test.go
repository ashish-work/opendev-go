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
	out := m.render(80)
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
	out := m.render(80)
	if !strings.Contains(out, "tool: read_file") {
		t.Errorf("tool message should show 'tool: <name>' in header, got:\n%s", out)
	}
}

func TestViewMessage_Render_WrapsLongContent(t *testing.T) {
	long := strings.Repeat("word ", 50) // ~250 chars
	m := viewMessage{role: roleUser, content: long}
	out := m.render(40)
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
	out := m.render(3) // way below minBodyWidth
	if out == "" {
		t.Errorf("render should produce output even at tiny width")
	}
	if !strings.Contains(out, "tiny") {
		t.Errorf("body should still appear: %q", out)
	}
}

func TestViewMessage_Render_EmptyContent(t *testing.T) {
	m := viewMessage{role: roleUser, content: ""}
	out := m.render(80)
	// Header should still appear; body is empty but the function
	// shouldn't choke or omit the header.
	if !strings.Contains(out, "you") {
		t.Errorf("header should render even with empty content: %q", out)
	}
}
