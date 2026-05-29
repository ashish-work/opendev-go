package hooks

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPermissionDecision_ConstantValues(t *testing.T) {
	// These string values are the wire vocabulary — every hook
	// process emits them on stdout. Pin them defensively.
	if PermissionAllow != "allow" {
		t.Errorf("PermissionAllow = %q, want %q", PermissionAllow, "allow")
	}
	if PermissionDeny != "deny" {
		t.Errorf("PermissionDeny = %q, want %q", PermissionDeny, "deny")
	}
	if PermissionAsk != "ask" {
		t.Errorf("PermissionAsk = %q, want %q", PermissionAsk, "ask")
	}
}

func TestHookDecision_EmptyMarshalsToEmptyObject(t *testing.T) {
	// omitempty on every field means a zero-value decision encodes
	// to the empty JSON object. Hooks that produce no opinion
	// should land here.
	b, err := json.Marshal(HookDecision{})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(b) != "{}" {
		t.Errorf("empty decision marshals to %q, want %q", b, "{}")
	}
}

func TestHookDecision_RoundTrip(t *testing.T) {
	original := HookDecision{
		AdditionalContext:  "Note: workspace is in dirty git state",
		UpdatedInput:       json.RawMessage(`{"path":"safer/place.txt"}`),
		PermissionDecision: PermissionAllow,
		Reason:             "Path moved into the sandboxed directory",
	}
	b, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got HookDecision
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.AdditionalContext != original.AdditionalContext {
		t.Errorf("AdditionalContext lost: got %q, want %q",
			got.AdditionalContext, original.AdditionalContext)
	}
	if string(got.UpdatedInput) != string(original.UpdatedInput) {
		t.Errorf("UpdatedInput lost: got %q, want %q",
			got.UpdatedInput, original.UpdatedInput)
	}
	if got.PermissionDecision != original.PermissionDecision {
		t.Errorf("PermissionDecision lost: got %q, want %q",
			got.PermissionDecision, original.PermissionDecision)
	}
	if got.Reason != original.Reason {
		t.Errorf("Reason lost: got %q, want %q", got.Reason, original.Reason)
	}
}

func TestHookDecision_OnlyOneFieldSurfacesInJSON(t *testing.T) {
	// A hook that only sets AdditionalContext should emit a JSON
	// payload containing ONLY that field. omitempty controls this.
	d := HookDecision{AdditionalContext: "extra context"}
	b, _ := json.Marshal(d)
	if !strings.Contains(string(b), "additionalContext") {
		t.Errorf("expected additionalContext in JSON, got %s", b)
	}
	if strings.Contains(string(b), "permissionDecision") {
		t.Errorf("permissionDecision should be omitted; got %s", b)
	}
	if strings.Contains(string(b), "updatedInput") {
		t.Errorf("updatedInput should be omitted; got %s", b)
	}
	if strings.Contains(string(b), "reason") {
		t.Errorf("reason should be omitted; got %s", b)
	}
}

func TestHookDecision_IsDeny(t *testing.T) {
	if !(HookDecision{PermissionDecision: PermissionDeny}).IsDeny() {
		t.Errorf("Deny decision should report IsDeny=true")
	}
	if (HookDecision{PermissionDecision: PermissionAllow}).IsDeny() {
		t.Errorf("Allow decision should not report IsDeny=true")
	}
	if (HookDecision{}).IsDeny() {
		t.Errorf("Empty decision (no opinion) should not be IsDeny")
	}
}

func TestHookDecision_IsAllow(t *testing.T) {
	if !(HookDecision{PermissionDecision: PermissionAllow}).IsAllow() {
		t.Errorf("Allow decision should report IsAllow=true")
	}
	if (HookDecision{PermissionDecision: PermissionDeny}).IsAllow() {
		t.Errorf("Deny decision should not report IsAllow=true")
	}
	if (HookDecision{}).IsAllow() {
		t.Errorf("Empty decision should not be IsAllow (no opinion ≠ allow)")
	}
}

func TestHookDecision_IsAsk(t *testing.T) {
	if !(HookDecision{PermissionDecision: PermissionAsk}).IsAsk() {
		t.Errorf("Ask decision should report IsAsk=true")
	}
	if (HookDecision{}).IsAsk() {
		t.Errorf("Empty decision should not be IsAsk")
	}
}

func TestHookDecision_UnknownPermissionValuePassesThrough(t *testing.T) {
	// A hook process could emit an unrecognized permissionDecision
	// value. Unmarshaling shouldn't fail — the manager will decide
	// what to do with unrecognized verdicts at the integration
	// layer rather than here at the type layer.
	body := []byte(`{"permissionDecision":"maybe"}`)
	var d HookDecision
	if err := json.Unmarshal(body, &d); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if d.PermissionDecision != "maybe" {
		t.Errorf("PermissionDecision = %q, want %q (pass-through)",
			d.PermissionDecision, "maybe")
	}
	// Unknown values are not Allow/Deny/Ask.
	if d.IsAllow() || d.IsDeny() || d.IsAsk() {
		t.Errorf("unknown permission value should not match any Is helper")
	}
}

func TestHookDecision_RawJSONInputPreserved(t *testing.T) {
	// UpdatedInput is raw JSON; the manager should hand it
	// downstream without re-parsing. Complex nested input should
	// round-trip byte-identically (after whitespace normalization).
	body := []byte(`{"updatedInput":{"path":"x.go","options":{"backup":true}}}`)
	var d HookDecision
	if err := json.Unmarshal(body, &d); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(d.UpdatedInput) == 0 {
		t.Fatal("UpdatedInput should be populated")
	}
	// Re-parse what we got and confirm the nested structure
	// survived intact.
	var inner map[string]any
	if err := json.Unmarshal(d.UpdatedInput, &inner); err != nil {
		t.Fatalf("UpdatedInput not valid JSON: %v\n%s", err, d.UpdatedInput)
	}
	if inner["path"] != "x.go" {
		t.Errorf("inner.path = %v, want x.go", inner["path"])
	}
}
