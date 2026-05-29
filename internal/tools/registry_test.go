package tools

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"
)

// fakeTool is a configurable Tool implementation used by registry tests.
// Each test wires its exec field to the behavior it wants to exercise.
type fakeTool struct {
	name   string
	desc   string
	schema json.RawMessage
	exec   func(ctx context.Context, tctx ToolContext, args json.RawMessage) (ToolResult, error)
}

func (f *fakeTool) Name() string            { return f.name }
func (f *fakeTool) Description() string     { return f.desc }
func (f *fakeTool) Schema() json.RawMessage { return f.schema }
func (f *fakeTool) Execute(ctx context.Context, tctx ToolContext, args json.RawMessage) (ToolResult, error) {
	return f.exec(ctx, tctx, args)
}

// successTool returns a fixed Output. Convenience constructor.
func successTool(name string) *fakeTool {
	return &fakeTool{
		name: name,
		desc: name + " description",
		exec: func(_ context.Context, _ ToolContext, _ json.RawMessage) (ToolResult, error) {
			return ToolResult{Success: true, Output: "ran " + name}, nil
		},
	}
}

func TestRegisterAndDispatchHappyPath(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(successTool("read_file")); err != nil {
		t.Fatalf("Register: %v", err)
	}

	got, err := r.Dispatch(context.Background(), ToolContext{}, "read_file", nil)
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if !got.Success {
		t.Errorf("Success = false, want true")
	}
	if got.Output != "ran read_file" {
		t.Errorf("Output = %q, want %q", got.Output, "ran read_file")
	}
}

func TestMultipleToolsDispatchIsolation(t *testing.T) {
	r := NewRegistry()
	for _, name := range []string{"read_file", "bash"} {
		if err := r.Register(successTool(name)); err != nil {
			t.Fatalf("Register %s: %v", name, err)
		}
	}

	for _, name := range []string{"bash", "read_file"} {
		got, err := r.Dispatch(context.Background(), ToolContext{}, name, nil)
		if err != nil {
			t.Fatalf("Dispatch %s: %v", name, err)
		}
		if got.Output != "ran "+name {
			t.Errorf("Output = %q, want %q", got.Output, "ran "+name)
		}
	}
}

func TestGetUnknownReturnsFalse(t *testing.T) {
	r := NewRegistry()
	if _, ok := r.Get("nonexistent"); ok {
		t.Error("Get returned ok=true for missing tool")
	}
}

func TestDispatchUnknownReturnsToolNotFound(t *testing.T) {
	r := NewRegistry()
	_, err := r.Dispatch(context.Background(), ToolContext{}, "ghost", nil)
	if !errors.Is(err, ErrToolNotFound) {
		t.Errorf("err = %v, want wraps ErrToolNotFound", err)
	}
}

func TestRegisterEmptyName(t *testing.T) {
	r := NewRegistry()
	err := r.Register(&fakeTool{name: ""})
	if !errors.Is(err, ErrInvalidParams) {
		t.Errorf("err = %v, want wraps ErrInvalidParams", err)
	}
}

func TestRegisterDuplicate(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(successTool("dup")); err != nil {
		t.Fatalf("first Register: %v", err)
	}
	err := r.Register(successTool("dup"))
	if !errors.Is(err, ErrInvalidParams) {
		t.Errorf("err = %v, want wraps ErrInvalidParams", err)
	}
}

func TestNamesReturnsSorted(t *testing.T) {
	r := NewRegistry()
	// Register out of alphabetical order to verify sort, not insertion order.
	for _, name := range []string{"zebra", "apple", "mango", "banana"} {
		if err := r.Register(successTool(name)); err != nil {
			t.Fatalf("Register %s: %v", name, err)
		}
	}

	got := r.Names()
	want := []string{"apple", "banana", "mango", "zebra"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Names() = %v, want %v", got, want)
	}
}

func TestDispatchPopulatesDuration(t *testing.T) {
	const sleep = 10 * time.Millisecond

	r := NewRegistry()
	tool := &fakeTool{
		name: "slow",
		exec: func(_ context.Context, _ ToolContext, _ json.RawMessage) (ToolResult, error) {
			time.Sleep(sleep)
			return ToolResult{Success: true}, nil
		},
	}
	if err := r.Register(tool); err != nil {
		t.Fatalf("Register: %v", err)
	}

	got, err := r.Dispatch(context.Background(), ToolContext{}, "slow", nil)
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	// Allow some scheduling slack but assert it's at least the sleep.
	if got.DurationMS < sleep.Milliseconds() {
		t.Errorf("DurationMS = %d, want >= %d", got.DurationMS, sleep.Milliseconds())
	}
}

func TestDispatchPopulatesDurationOnError(t *testing.T) {
	r := NewRegistry()
	wantErr := errors.New("bang")
	tool := &fakeTool{
		name: "boom",
		exec: func(_ context.Context, _ ToolContext, _ json.RawMessage) (ToolResult, error) {
			time.Sleep(2 * time.Millisecond)
			return ToolResult{Success: false, Error: "bang"}, wantErr
		},
	}
	if err := r.Register(tool); err != nil {
		t.Fatalf("Register: %v", err)
	}

	got, err := r.Dispatch(context.Background(), ToolContext{}, "boom", nil)
	if !errors.Is(err, wantErr) {
		t.Errorf("err = %v, want %v", err, wantErr)
	}
	if got.DurationMS <= 0 {
		t.Errorf("DurationMS = %d, want > 0 even on error", got.DurationMS)
	}
}

func TestDispatchContextCancellation(t *testing.T) {
	r := NewRegistry()

	// Tool that blocks until ctx is cancelled, then reports the ctx error.
	tool := &fakeTool{
		name: "blocker",
		exec: func(ctx context.Context, _ ToolContext, _ json.RawMessage) (ToolResult, error) {
			<-ctx.Done()
			return ToolResult{}, ctx.Err()
		},
	}
	if err := r.Register(tool); err != nil {
		t.Fatalf("Register: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	_, err := r.Dispatch(ctx, ToolContext{}, "blocker", nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want context.DeadlineExceeded", err)
	}
}

func TestConcurrentDispatchIsRaceFree(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(successTool("noop")); err != nil {
		t.Fatalf("Register: %v", err)
	}

	const goroutines = 50
	const callsPerGoroutine = 20

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < callsPerGoroutine; j++ {
				if _, err := r.Dispatch(context.Background(), ToolContext{}, "noop", nil); err != nil {
					t.Errorf("Dispatch: %v", err)
					return
				}
				_ = r.Names() // Mix in reads of another method.
			}
		}()
	}
	wg.Wait()
}

func TestConcurrentRegistrationIsRaceFree(t *testing.T) {
	r := NewRegistry()

	const goroutines = 20
	names := make([]string, goroutines)
	for i := range names {
		names[i] = "tool_" + string(rune('a'+i))
	}

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for _, name := range names {
		go func(n string) {
			defer wg.Done()
			if err := r.Register(successTool(n)); err != nil {
				t.Errorf("Register %s: %v", n, err)
			}
		}(name)
	}
	wg.Wait()

	if got := len(r.Names()); got != goroutines {
		t.Errorf("len(Names()) = %d, want %d", got, goroutines)
	}
}

// makeFilledRegistry returns a Registry pre-loaded with the named
// fake tools. Convenience for the Filter tests below.
func makeFilledRegistry(t *testing.T, names ...string) *Registry {
	t.Helper()
	r := NewRegistry()
	for _, n := range names {
		if err := r.Register(successTool(n)); err != nil {
			t.Fatalf("register %s: %v", n, err)
		}
	}
	return r
}

func TestFilter_NilAllowedNamesReturnsFullPassthrough(t *testing.T) {
	r := makeFilledRegistry(t, "alpha", "beta", "gamma")
	got := r.Filter(nil)
	want := []string{"alpha", "beta", "gamma"}
	if names := got.Names(); !reflect.DeepEqual(names, want) {
		t.Errorf("Filter(nil).Names() = %v, want %v", names, want)
	}
}

func TestFilter_EmptySliceReturnsEmptyRegistry(t *testing.T) {
	r := makeFilledRegistry(t, "alpha", "beta")
	got := r.Filter([]string{})
	if names := got.Names(); len(names) != 0 {
		t.Errorf("Filter([]string{}).Names() = %v, want empty", names)
	}
}

func TestFilter_WhitelistKeepsOnlyNamedTools(t *testing.T) {
	r := makeFilledRegistry(t, "alpha", "beta", "gamma", "delta")
	got := r.Filter([]string{"alpha", "gamma"})
	want := []string{"alpha", "gamma"}
	if names := got.Names(); !reflect.DeepEqual(names, want) {
		t.Errorf("Filter([alpha,gamma]).Names() = %v, want %v", names, want)
	}
	// Confirm dispatch works for kept tool, fails for dropped tool.
	if _, err := got.Dispatch(context.Background(), ToolContext{},
		"alpha", json.RawMessage(`{}`)); err != nil {
		t.Errorf("Dispatch alpha (kept) should succeed; got %v", err)
	}
	_, err := got.Dispatch(context.Background(), ToolContext{},
		"beta", json.RawMessage(`{}`))
	if !errors.Is(err, ErrToolNotFound) {
		t.Errorf("Dispatch beta (dropped) should return ErrToolNotFound; got %v", err)
	}
}

func TestFilter_UnknownNamesSilentlyDropped(t *testing.T) {
	// Matches Planner's forward-reference pattern: present_plan
	// doesn't exist as a tool yet, so it should be silently
	// dropped. The known name still flows through.
	r := makeFilledRegistry(t, "read_file", "list_files")
	got := r.Filter([]string{"read_file", "present_plan", "list_files"})
	want := []string{"list_files", "read_file"}
	if names := got.Names(); !reflect.DeepEqual(names, want) {
		t.Errorf("Filter dropped unknowns: got %v, want %v", names, want)
	}
}

func TestFilter_DoesNotMutateOriginal(t *testing.T) {
	// Critical: Filter is supposed to return a fresh registry
	// without touching the receiver. If the source got mutated,
	// the parent ReactLoop would lose tools mid-session.
	r := makeFilledRegistry(t, "alpha", "beta", "gamma")
	_ = r.Filter([]string{"alpha"})

	want := []string{"alpha", "beta", "gamma"}
	if names := r.Names(); !reflect.DeepEqual(names, want) {
		t.Errorf("original registry mutated by Filter: got %v, want %v",
			names, want)
	}
	// Original still dispatches non-filtered tools.
	if _, err := r.Dispatch(context.Background(), ToolContext{},
		"beta", json.RawMessage(`{}`)); err != nil {
		t.Errorf("Original Dispatch beta should still work; got %v", err)
	}
}

func TestFilter_ResultRegistryAcceptsFurtherRegistration(t *testing.T) {
	// The returned registry should be a fully functional peer —
	// callers can Register additional tools onto it without
	// touching the original.
	r := makeFilledRegistry(t, "alpha")
	scoped := r.Filter([]string{"alpha"})
	if err := scoped.Register(successTool("zeta")); err != nil {
		t.Errorf("Register on filtered registry should succeed; got %v", err)
	}
	if _, ok := scoped.Get("zeta"); !ok {
		t.Errorf("zeta should be registered on the filtered registry")
	}
	// Original shouldn't have zeta.
	if _, ok := r.Get("zeta"); ok {
		t.Errorf("zeta leaked into the original registry")
	}
}

func TestFilter_EmptySourceRegistry(t *testing.T) {
	// Calling Filter on an empty registry should return an empty
	// registry regardless of the whitelist.
	r := NewRegistry()
	for _, allowed := range [][]string{
		nil,
		{},
		{"unknown"},
	} {
		got := r.Filter(allowed)
		if names := got.Names(); len(names) != 0 {
			t.Errorf("Filter on empty registry with %v should be empty; got %v",
				allowed, names)
		}
	}
}

func TestFilter_DuplicateNamesInWhitelistDedupNaturally(t *testing.T) {
	// Listing the same name twice in the whitelist should be a
	// no-op for the duplicate. Map insert in the implementation
	// handles this naturally; we pin the behavior.
	r := makeFilledRegistry(t, "alpha")
	got := r.Filter([]string{"alpha", "alpha", "alpha"})
	if names := got.Names(); !reflect.DeepEqual(names, []string{"alpha"}) {
		t.Errorf("duplicate names should dedup; got %v", names)
	}
}

