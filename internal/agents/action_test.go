package agents

import (
	"errors"
	"testing"

	"github.com/ashish-work/opendev-go/internal/cost"
)

func TestLoopActionKind_String(t *testing.T) {
	cases := []struct {
		kind LoopActionKind
		want string
	}{
		{LoopActionContinue, "continue"},
		{LoopActionReturn, "return"},
		{LoopActionKind(-1), "unknown"},
		{LoopActionKind(99), "unknown"},
	}
	for _, c := range cases {
		if got := c.kind.String(); got != c.want {
			t.Errorf("LoopActionKind(%d).String() = %q, want %q", c.kind, got, c.want)
		}
	}
}

// Defensive: catches an iota re-order that accidentally collapses two
// kinds onto the same value.
func TestLoopActionKind_DistinctValues(t *testing.T) {
	if LoopActionContinue == LoopActionReturn {
		t.Fatal("LoopActionContinue and LoopActionReturn share a value")
	}
}

func TestNewLoopActionContinue(t *testing.T) {
	tracker := cost.Tracker{TotalInputTokens: 42, CallCount: 1}
	a := NewLoopActionContinue(tracker)
	if a.Kind != LoopActionContinue {
		t.Errorf("Kind = %s, want continue", a.Kind)
	}
	// Continue carries the tracker forward; the driver passes it to
	// the next phase unchanged.
	if a.Tracker != tracker {
		t.Errorf("Tracker = %+v, want %+v", a.Tracker, tracker)
	}
	// Result and Err must be zero for Continue. Result contains a
	// slice (Messages) so it's not == comparable; check the fields
	// that callers actually look at on the exit path.
	if a.Result.Content != "" || a.Result.Success || a.Result.Interrupted {
		t.Errorf("Continue.Result should be zero, got %+v", a.Result)
	}
	if a.Result.Messages != nil {
		t.Errorf("Continue.Result.Messages should be nil, got %+v", a.Result.Messages)
	}
	if a.Err != nil {
		t.Errorf("Continue.Err should be nil, got %v", a.Err)
	}
}

func TestNewLoopActionReturn_SuccessPath(t *testing.T) {
	tracker := cost.Tracker{CallCount: 3}
	result := Result{Content: "final answer", Success: true}
	a := NewLoopActionReturn(result, nil, tracker)
	if a.Kind != LoopActionReturn {
		t.Errorf("Kind = %s, want return", a.Kind)
	}
	if a.Err != nil {
		t.Errorf("success Return should have nil Err, got %v", a.Err)
	}
	if a.Result.Content != "final answer" {
		t.Errorf("Result.Content = %q, want %q", a.Result.Content, "final answer")
	}
	if !a.Result.Success {
		t.Errorf("Result.Success = false, want true")
	}
	if a.Tracker != tracker {
		t.Errorf("Tracker = %+v, want %+v", a.Tracker, tracker)
	}
}

func TestNewLoopActionReturn_FailurePath(t *testing.T) {
	wantErr := errors.New("boom")
	a := NewLoopActionReturn(Result{}, wantErr, cost.Tracker{})
	if a.Kind != LoopActionReturn {
		t.Errorf("Kind = %s, want return", a.Kind)
	}
	if !errors.Is(a.Err, wantErr) {
		t.Errorf("Err = %v, want %v", a.Err, wantErr)
	}
}

// Exhaustive-switch sentinel: matches the pattern from
// StreamEventKind_AllKindsHaveStringArm. A new Kind value without a
// corresponding String() arm will fall through to "unknown" and fail
// this test, alerting the future committer to update both places.
func TestLoopActionKind_AllKindsHaveStringArm(t *testing.T) {
	all := []LoopActionKind{
		LoopActionContinue,
		LoopActionReturn,
	}
	for _, k := range all {
		if s := k.String(); s == "unknown" {
			t.Errorf("LoopActionKind(%d) returned %q — add a case to String()", k, s)
		}
	}
}

// Compile-time check that LoopAction values can be passed by value
// through the loop driver. The type isn't == comparable (Result
// contains a slice), but Kind comparison is the discriminator the
// driver uses and that has to be cheap.
func TestLoopAction_KindIsDirectlyComparable(t *testing.T) {
	a := NewLoopActionContinue(cost.Tracker{})
	b := NewLoopActionContinue(cost.Tracker{})
	if a.Kind != b.Kind {
		t.Errorf("two Continue actions should be Kind-equal")
	}
}
