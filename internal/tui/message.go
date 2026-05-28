package tui

import (
	"fmt"
	"strings"

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

// collapsedToolLines is how many lines of tool output we show before
// truncating with a "(… N more lines)" hint. Three is enough to see
// what kind of output it is without burying the conversation.
const collapsedToolLines = 3

// render produces the styled multi-line output for one message.
// width is the viewport's visible width; the body wraps to that width
// (with a minimum floor). expanded only affects tool messages — they
// render as bordered cards that show the first collapsedToolLines of
// content by default, and the full body when expanded.
//
// For non-tool messages the leading "▎ " is a Slack/Discord-style
// speaker bar, header-only (the wrapped body sits below without the
// bar continuing — implementing per-line bars cleanly with the
// lipgloss wrap is more plumbing than the visual win justifies).
// Tool messages drop the bar; the bordered card is their visual
// marker instead.
func (m viewMessage) render(width int, expanded bool) string {
	if m.role == roleTool {
		return m.renderTool(width, expanded)
	}
	header, headerStyle, bodyStyle := roleStyle(m.role, m.toolName)
	if width < minBodyWidth {
		width = minBodyWidth
	}
	headerLine := headerStyle.Render("▎ " + header)
	body := bodyStyle.Width(width).Render(m.content)
	return headerLine + "\n" + body
}

// renderTool draws a tool message as a bordered card. The card's
// border is rounded yellow (matching the tool role's accent color
// throughout the TUI), padded one column on each side, sized to
// total `width` so it slots into the viewport without overflow.
//
// Width math: the rounded border eats 2 columns (one per side) AND
// the padding eats 2 more. Available content width inside the card
// is therefore width - 4. We set lipgloss.Width on the OUTER card
// style to width - 2; lipgloss treats that as "content area" and
// adds the border outside. The 1-col padding on each side then
// leaves width - 4 for the actual text. The math is unintuitive
// the first time but consistent once you internalize it.
func (m viewMessage) renderTool(width int, expanded bool) string {
	if width < minBodyWidth+4 {
		width = minBodyWidth + 4 // ensure the card has room for some content
	}
	// Border (1 col each side) is outside Width; padding (1 col each
	// side) is inside Width. So cardWidth (the "content area" lipgloss
	// sees) = width - 2 borders, and the actual text fits in
	// cardWidth - 2 padding columns.
	cardWidth := width - 2

	headerText := "tool"
	if m.toolName != "" {
		headerText = "tool: " + m.toolName
	}
	headerLine := toolCardHeaderStyle.Render(headerText)

	body := m.content
	var extra int
	if !expanded {
		body, extra = truncateToLines(body, collapsedToolLines)
	}
	bodyLine := toolCardBodyStyle.Render(body)

	parts := []string{headerLine, bodyLine}
	if extra > 0 {
		hint := fmt.Sprintf(
			"(… %d more line%s — Ctrl-T to expand all tool details)",
			extra, plural(extra),
		)
		parts = append(parts, toolCardHintStyle.Render(hint))
	}

	inner := strings.Join(parts, "\n")
	return toolCardStyle.Width(cardWidth).Render(inner)
}

// truncateToLines returns the first `max` lines of s, plus the count
// of lines that were dropped. When s fits within `max`, returns s
// unchanged with extra=0. A max < 1 returns an empty preview and
// reports every line as extra — defensive against bad callers.
func truncateToLines(s string, max int) (preview string, extra int) {
	if max < 1 {
		if s == "" {
			return "", 0
		}
		return "", strings.Count(s, "\n") + 1
	}
	lines := strings.Split(s, "\n")
	if len(lines) <= max {
		return s, 0
	}
	return strings.Join(lines[:max], "\n"), len(lines) - max
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
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

// Tool-card-specific styles. The card is a separate rendering path
// from the plain user/assistant treatment, so it gets its own style
// set instead of reusing the toolBodyStyle (which has PaddingLeft
// that would double up with the card's own padding).
var (
	// toolCardStyle is the bordered box itself. Rounded corners +
	// yellow border match the tool role's accent color used in the
	// non-card path's header. Padding(0, 1) leaves one column of
	// breathing room left and right of the content.
	toolCardStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("220")).
			Padding(0, 1)

	// toolCardHeaderStyle is the header line INSIDE the card —
	// "tool: <name>" in bold yellow. Stays distinct from the body
	// so even a small card has a clear identifier at the top.
	toolCardHeaderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("220")).
				Bold(true)

	// toolCardBodyStyle is the tool's output text inside the card.
	// Dimmer than body text (245 = mid gray) because tool output is
	// reference material that shouldn't compete with the conversation.
	// No italic — output is often code/paths and italics hurt
	// readability for monospace content.
	toolCardBodyStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("245"))

	// toolCardHintStyle is the "(… N more — Ctrl-T to expand)" hint
	// that appears only when content is truncated. Dimmer + italic
	// so it reads as meta-instruction, not output.
	toolCardHintStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("241")).
				Italic(true)
)
