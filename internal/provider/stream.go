package provider

// StreamEvent and friends are the tagged-struct representation of
// token-level output from a streaming LLM call. They are the surface
// the agent loop consumes; adapters in subpackages (provider/openai,
// provider/anthropic) parse vendor-specific SSE byte streams and
// emit StreamEvents.
//
// Why a tagged struct rather than separate types per variant? Same
// reason ContentBlock works that way: a single concrete type means
// the consumer reads one Kind field and switches on it. No type
// assertions, no interface allocation per event — important because
// streaming emits hundreds of TextDelta events per second.
//
// What lives here vs. elsewhere:
//   - This file: data types only, no I/O, no SSE parsing.
//   - Stream method on Provider: added separately so the interface
//     change is reviewable on its own.
//   - SSE parsing + delta reassembly: adapter packages, where
//     vendor-specific event shapes are translated into StreamEvents.

// StreamEventKind tags the variant of a StreamEvent. Iota-based int
// constants follow the same pattern as ContentKind: cheap to compare,
// exhaustive switches feel natural, easy to add a String method for
// logging.
type StreamEventKind int

const (
	// StreamEventTextDelta — an incremental chunk of assistant text.
	// Most frequent event; arrives many times per turn. Valid field: Text.
	StreamEventTextDelta StreamEventKind = iota

	// StreamEventReasoningDelta — an incremental chunk of reasoning
	// or "thinking" content. Emitted by reasoning-capable models (e.g.
	// OpenAI o-family, Anthropic with thinking enabled). The loop can
	// display this separately from final answer text. Valid field: Text.
	StreamEventReasoningDelta

	// StreamEventToolCallStart — a new tool_call slot is opening. The
	// model has decided which tool to invoke but the arguments are
	// still streaming. Valid fields: ToolCall.{Index, ID, Name}.
	StreamEventToolCallStart

	// StreamEventToolCallDelta — an incremental fragment of arguments
	// for an in-flight tool_call. Concatenate across all deltas with
	// the same Index to assemble the full JSON. Valid fields:
	// ToolCall.{Index, Arguments}.
	StreamEventToolCallDelta

	// StreamEventToolCallDone — a tool_call's arguments are fully
	// assembled and ready for dispatch. Valid fields:
	// ToolCall.{Index, ID, Name, Arguments} where Arguments is the
	// complete JSON string.
	StreamEventToolCallDone

	// StreamEventUsage — a token-accounting update. Some providers
	// (Anthropic) emit usage mid-stream; others (OpenAI) only at end.
	// Valid field: Usage.
	StreamEventUsage

	// StreamEventDone — the stream has ended successfully and the
	// full Response is materialized. The loop treats this as it would
	// the result of a non-streaming Call. Valid field: Response.
	StreamEventDone

	// StreamEventError — the stream failed. After this event the
	// channel is closed; no further events arrive. Valid field: Err.
	StreamEventError
)

// String returns a stable, lowercase identifier for the kind. Used
// for logs and debug prints. Keeping these short matters because
// they appear in slog output once per streamed token.
func (k StreamEventKind) String() string {
	switch k {
	case StreamEventTextDelta:
		return "text_delta"
	case StreamEventReasoningDelta:
		return "reasoning_delta"
	case StreamEventToolCallStart:
		return "tool_call_start"
	case StreamEventToolCallDelta:
		return "tool_call_delta"
	case StreamEventToolCallDone:
		return "tool_call_done"
	case StreamEventUsage:
		return "usage"
	case StreamEventDone:
		return "done"
	case StreamEventError:
		return "error"
	default:
		return "unknown"
	}
}

// StreamEvent is one item produced by a streaming provider call. Only
// the fields valid for the active Kind are populated; other fields
// are zero. Prefer the New* constructors over raw struct literals so
// the invariant is enforced at the call site.
//
// Why a single struct with all fields union'd together rather than an
// interface? Two reasons:
//
//  1. Zero allocation per event. An interface value boxes its
//     payload; in a hot streaming path that's hundreds of allocations
//     per second. A plain struct passed by value stays on the stack.
//
//  2. Symmetry with ContentBlock, TurnKind, and other tagged structs
//     in this codebase. Readers who learned the pattern once recognize
//     it everywhere.
type StreamEvent struct {
	// Kind selects which other fields are meaningful.
	Kind StreamEventKind

	// Text — valid for StreamEventTextDelta and StreamEventReasoningDelta.
	// The chunk text as the provider emitted it; no trimming.
	Text string

	// ToolCall — valid for any StreamEventToolCall* kind. Grouped
	// into a sub-struct so the related fields stay together and the
	// top-level StreamEvent doesn't sprawl into a dozen fields.
	ToolCall ToolCallDelta

	// Usage — valid for StreamEventUsage. Cumulative since the start
	// of the call, per provider convention (not a per-chunk delta).
	Usage Usage

	// Response — valid for StreamEventDone. The fully assembled message
	// after the stream closes. Pointer so the zero value of StreamEvent
	// (used for other kinds) doesn't carry an empty Response payload.
	Response *Response

	// Err — valid for StreamEventError. The failure that ended the
	// stream. Wrapped with provider-package context by the adapter.
	Err error
}

// ToolCallDelta carries the per-tool_call fields for the three
// tool-call streaming events. Grouped here so a TextDelta event
// doesn't carry four empty tool-call fields.
//
// Providers stream tool_call arguments incrementally, one chunk per
// "index" slot. Multiple tool_calls can be in flight at once with
// different indices; consumers reassemble by indexing into a map or
// slice keyed by Index.
type ToolCallDelta struct {
	// Index identifies the tool_call slot within this turn. Stable
	// across Start/Delta/Done events for the same call. Providers
	// emit it (OpenAI explicitly, Anthropic derives from
	// content_block index).
	Index int

	// ID — the tool_call's unique identifier. Populated on
	// StreamEventToolCallStart and StreamEventToolCallDone.
	ID string

	// Name — the tool the model wants to invoke. Populated on
	// StreamEventToolCallStart and StreamEventToolCallDone.
	Name string

	// Arguments — for StreamEventToolCallDelta, an incremental JSON
	// fragment (raw, not yet valid JSON on its own). For
	// StreamEventToolCallDone, the fully assembled JSON string ready
	// for the tool's argument parser.
	Arguments string
}

// NewTextDelta constructs a TextDelta event. Use this instead of a
// raw struct literal so adding fields to StreamEvent later doesn't
// silently break call sites.
func NewTextDelta(text string) StreamEvent {
	return StreamEvent{Kind: StreamEventTextDelta, Text: text}
}

// NewReasoningDelta constructs a ReasoningDelta event.
func NewReasoningDelta(text string) StreamEvent {
	return StreamEvent{Kind: StreamEventReasoningDelta, Text: text}
}

// NewToolCallStart constructs a ToolCallStart event. Arguments are
// left empty — they arrive later as Delta events.
func NewToolCallStart(index int, id, name string) StreamEvent {
	return StreamEvent{
		Kind:     StreamEventToolCallStart,
		ToolCall: ToolCallDelta{Index: index, ID: id, Name: name},
	}
}

// NewToolCallDelta constructs a ToolCallDelta event carrying an
// incremental arguments fragment.
func NewToolCallDelta(index int, argsFragment string) StreamEvent {
	return StreamEvent{
		Kind:     StreamEventToolCallDelta,
		ToolCall: ToolCallDelta{Index: index, Arguments: argsFragment},
	}
}

// NewToolCallDone constructs a ToolCallDone event with the fully
// assembled arguments JSON.
func NewToolCallDone(index int, id, name, args string) StreamEvent {
	return StreamEvent{
		Kind: StreamEventToolCallDone,
		ToolCall: ToolCallDelta{
			Index:     index,
			ID:        id,
			Name:      name,
			Arguments: args,
		},
	}
}

// NewUsage constructs a Usage event. Providers emit one or more of
// these during a turn; consumers should treat each as the cumulative
// counts so far, not a delta.
func NewUsage(u Usage) StreamEvent {
	return StreamEvent{Kind: StreamEventUsage, Usage: u}
}

// NewDone constructs a Done event carrying the fully assembled
// Response. Adapters call this exactly once per successful stream,
// after which the channel closes.
func NewDone(resp *Response) StreamEvent {
	return StreamEvent{Kind: StreamEventDone, Response: resp}
}

// NewError constructs an Error event. Adapters call this at most
// once per stream; the channel closes immediately after.
func NewError(err error) StreamEvent {
	return StreamEvent{Kind: StreamEventError, Err: err}
}
