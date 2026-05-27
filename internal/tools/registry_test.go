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
