package pathutil

import (
	"strings"
	"testing"
)

func TestSensitiveReason(t *testing.T) {
	tests := []struct {
		name string
		path string
		// want is a substring the reason must contain. Empty want
		// means the path must be classified as safe (reason "").
		want string
	}{
		// .env family
		{"dot-env exact", ".env", "environment"},
		{"dot-env with path", "/project/.env", "environment"},
		{"dot-env-local", ".env.local", "environment"},
		{"dot-env-production", ".env.production", "environment"},
		{"dot-env-example allowed", ".env.example", ""},
		{"dot-env-sample allowed", ".env.sample", ""},
		{"uppercase ENV is sensitive", "/PROJECT/.ENV", "environment"},
		{"mixed case env.local", "/p/.Env.LOCAL", "environment"},

		// Private keys: basenames (location-independent).
		{"id_rsa", "id_rsa", "private key"},
		{"id_ed25519", "id_ed25519", "private key"},
		{"id_ecdsa", "id_ecdsa", "private key"},
		{"id_rsa in ssh dir", "/home/user/.ssh/id_rsa", "private key"},
		{"id_rsa in tmp", "/tmp/id_rsa", "private key"},

		// Private keys: extensions.
		{"server.pem", "server.pem", "private key"},
		{"private.key", "private.key", "private key"},
		{"upper PEM extension", "SERVER.PEM", "private key"},

		// Credentials filenames.
		{"credentials no ext", "credentials", "credentials"},
		{"credentials json", "credentials.json", "credentials"},
		{"credentials yaml", "credentials.yaml", "credentials"},
		{"service-account json", "service-account.json", "credentials"},
		{"npmrc", "/home/user/.npmrc", "credentials"},
		{"pypirc", ".pypirc", "credentials"},
		{"netrc", "/home/user/.netrc", "credentials"},
		{"htpasswd at etc", "/etc/.htpasswd", "credentials"},

		// Generic secret data files.
		{"app-secret json", "app-secret.json", "secrets"},
		{"secret yaml", "secret.yaml", "secrets"},
		{"secret yml", "secret.yml", "secrets"},
		{"secrets-store json", "secrets-store.json", "secrets"},

		// Safe (negative cases).
		{"go source", "main.go", ""},
		{"readme", "README.md", ""},
		{"toml config", "config.toml", ""},
		{"package json", "package.json", ""},
		{"cargo lock", "Cargo.lock", ""},
		{"log file", "/var/log/app.log", ""},
		{"deep go source", "/home/user/project/main.go", ""},
		{"empty path", "", ""},
		{"dot path", ".", ""},

		// "secret" in name but not a data file -> safe.
		{"secret readme", "secrets-readme.md", ""},
		{"secret docs", "/docs/secret-handling.md", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SensitiveReason(tt.path)
			if tt.want == "" {
				if got != "" {
					t.Errorf("SensitiveReason(%q) = %q, want empty (safe)", tt.path, got)
				}
				return
			}
			if got == "" {
				t.Errorf("SensitiveReason(%q) returned empty; want reason containing %q",
					tt.path, tt.want)
				return
			}
			if !strings.Contains(got, tt.want) {
				t.Errorf("SensitiveReason(%q) = %q, want it to contain %q",
					tt.path, got, tt.want)
			}
		})
	}
}

// TestSensitiveReason_OverrideAllowsExtension demonstrates the package
// vars are tunable for tests/consumers that need a tighter or looser
// policy temporarily.
func TestSensitiveReason_OverrideAllowsExtension(t *testing.T) {
	saved := EnvExampleSuffixes
	defer func() { EnvExampleSuffixes = saved }()

	// Treat .env.template as a safe sample too.
	EnvExampleSuffixes = append([]string{".template"}, saved...)
	if got := SensitiveReason(".env.template"); got != "" {
		t.Errorf("with .template appended, SensitiveReason(.env.template) = %q, want empty", got)
	}
	// Sanity: the real .env.local is still sensitive.
	if got := SensitiveReason(".env.local"); got == "" {
		t.Errorf("SensitiveReason(.env.local) should still be sensitive after override")
	}
}
