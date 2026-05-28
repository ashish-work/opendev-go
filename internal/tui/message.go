package tui

import (
	"github.com/charmbracelet/lipgloss"

	"github.com/ashish-work/opendev-go/internal/provider"
)

// messageRole identifies the speaker of a viewMessage. Three roles
// for now — user, assistant, tool — matching the agent loop's
// existing categorization. A fourth ("system") exists in
// provider.Message for the system prompt, but the system prompt
// isn't part of the chat transcript so it has no display role.
//
// Tagged-int pattern (same shape as provider.ContentKind,
// tools.Category). The associated styles are looked up in roleStyle
// rather than carried on the enum, so adding a role later is
// data-only.
type messageRole int

const (
	roleUser messageRole = iota
	roleAssistant
	roleTool
)

// viewMessage is the TUI's rendering-side message type. Distinct from
// provider.Message — the wire-format struct has tagged content blocks
// and tool-call payloads we'd have to unwrap on every render. By the
// time something reaches viewMessage, all that work is done; this is
// the cached, render-ready shape the viewport consumes directly.
//
// When the agent loop is wired in (next commit), the integration layer
// translates each provider.Message → one or more viewMessages once
// at append time. Repaints stay cheap.
type viewMessage struct {
	// role selects header label + body styling. See roleStyle.
	role messageRole

	// content is the body text. Plain prose for user/assistant;
	// for tool messages this is the tool's stdout / Output field.
	content string

	// toolName is populated only for role == roleTool. The header
	// renders as "tool: <toolName>" so the user can tell which
	// tool ran without expanding the message.
	toolName string
}

// roleStyle returns the header label and the lipgloss styles to use
// for a given role. Pulled out as a function so the per-role branching
// lives in exactly one place; viewMessage.render is then trivial.
//
// Color choices:
//   - User: cyan + bold. Stands out without screaming; the user is
//     the active participant.
//   - Assistant: green. Conventional "model response" color across
//     chat UIs.
//   - Tool: yellow. Distinct from user/assistant; signals "this is
//     a tool observation, not a conversation turn."
//
// Bodies are default foreground except tool messages, which dim +
// italicize because tool output is reference material, not voice.
func roleStyle(role messageRole, toolName string) (string, lipgloss.Style, lipgloss.Style) {
	switch role {
	case roleUser:
		return "you", userHeaderStyle, defaultBodyStyle
	case roleAssistant:
		return "assistant", assistantHeaderStyle, defaultBodyStyle
	case roleTool:
		label := "tool"
		if toolName != "" {
			label = "tool: " + toolName
		}
		return label, toolHeaderStyle, toolBodyStyle
	default:
		// Unknown role — render as plain text so callers don't get a
		// blank message. Catches programming errors rather than user
		// errors.
		return "?", defaultBodyStyle, defaultBodyStyle
	}
}

// minBodyWidth keeps the body wrap target from going too narrow on
// tiny terminals — text wrapped to 3 columns is unreadable.
const minBodyWidth = 10

// render produces the styled multi-line output for one message.
// width is the viewport's visible width; the body wraps to that
// width (with a minimum floor). Output is "header\nbody" where the
// header is the role label and the body is the content text.
//
// The leading "▎ " on the header is a thick left vertical bar — a
// Slack/Discord-style speaker indicator. Header-only, not duplicated
// on body lines, because doing the bar-per-line trick correctly
// requires splitting the wrapped output and re-prefixing, which is
// more rendering plumbing than the visual win justifies.
func (m viewMessage) render(width int) string {
	header, headerStyle, bodyStyle := roleStyle(m.role, m.toolName)
	if width < minBodyWidth {
		width = minBodyWidth
	}
	headerLine := headerStyle.Render("▎ " + header)
	body := bodyStyle.Width(width).Render(m.content)
	return headerLine + "\n" + body
}

// translateMessages converts a slice of provider.Message (the wire
// format the agent loop produces) into the TUI-local viewMessage
// shape used for rendering. The translation is intentionally lossy:
//
//   - "system" role is skipped entirely. The system prompt is set-
//     once configuration, not a transcript turn.
//   - "user" messages become one viewMessage per text content block.
//   - "assistant" messages become one viewMessage per text content
//     block. The assistant's tool_calls themselves do NOT render —
//     they're scaffolding for the next iteration; the user sees the
//     tool RESULTS (the "tool"-role messages that follow) instead.
//     Rendering the tool_call as a separate "the model decided to
//     call X" entry would double-count it visually.
//   - "tool" messages become a viewMessage with role=roleTool, with
//     toolName populated from the message's Name field and content
//     from its first text block.
//
// Unknown roles are skipped silently. Empty-content blocks are also
// skipped — a literal empty viewMessage would render as a styled
// header with nothing below it, which looks broken.
func translateMessages(msgs []provider.Message) []viewMessage {
	out := make([]viewMessage, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case "system":
			// not part of the visible transcript

		case "user":
			for _, c := range m.Content {
				if c.Kind == provider.ContentText && c.Text != "" {
					out = append(out, viewMessage{role: roleUser, content: c.Text})
				}
			}

		case "assistant":
			for _, c := range m.Content {
				if c.Kind == provider.ContentText && c.Text != "" {
					out = append(out, viewMessage{role: roleAssistant, content: c.Text})
				}
			}

		case "tool":
			text := firstTextBlock(m.Content)
			if text == "" {
				continue
			}
			out = append(out, viewMessage{
				role:     roleTool,
				toolName: m.Name,
				content:  text,
			})
		}
	}
	return out
}

// firstTextBlock returns the first text content from a message, or "".
// Tool messages typically carry a single text block (the tool's
// stdout); this helper just picks it out without assuming index 0
// is text.
func firstTextBlock(blocks []provider.ContentBlock) string {
	for _, b := range blocks {
		if b.Kind == provider.ContentText {
			return b.Text
		}
	}
	return ""
}

// Pre-built styles. Declared as package vars so each render call
// reuses the same Style values instead of constructing them per-frame.
// Lipgloss styles are cheap to copy but cheaper to share.
var (
	userHeaderStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("39")).
			Bold(true)

	assistantHeaderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("36"))

	toolHeaderStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("220"))

	defaultBodyStyle = lipgloss.NewStyle().
				PaddingLeft(2)

	toolBodyStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("245")).
			Italic(true).
			PaddingLeft(2)
)
