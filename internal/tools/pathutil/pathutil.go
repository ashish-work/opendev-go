// Package pathutil holds path-classification helpers shared across
// tool implementations. The current scope is credential/secret-file
// detection — write_file calls SensitiveReason to refuse writes that
// would overwrite likely-secret files.
//
// Detection is filename-based (case-insensitive), not directory-based.
// A file named id_rsa is treated as a private key whether it lives in
// ~/.ssh/, /tmp/, or anywhere else. The argument: filenames are a
// stronger signal than directories. /etc/ contains many files agents
// can legitimately edit (nginx.conf is not a security concern); the
// existence of id_rsa anywhere is.
//
// Directory-level deny rules belong to the future permissions
// subsystem, where users can configure them per project. This package
// is for the hardcoded set of "no agent should ever touch these"
// filenames.
package pathutil

import (
	"path/filepath"
	"strings"
)

// EnvExampleSuffixes mark a .env.* file as a template/sample rather
// than a real secrets file — .env.example and .env.sample are safe to
// write because they're meant to be committed to repos.
var EnvExampleSuffixes = []string{".example", ".sample"}

// CredentialNames are exact basenames (case-insensitive) classified
// as credentials files.
var CredentialNames = []string{
	"credentials",
	"credentials.json",
	"credentials.yaml",
	"credentials.yml",
	"service-account.json",
	".npmrc",
	".pypirc",
	".netrc",
	".htpasswd",
}

// PrivateKeyNames are exact basenames (case-insensitive) classified
// as private keys regardless of extension.
var PrivateKeyNames = []string{
	"id_rsa",
	"id_ed25519",
	"id_ecdsa",
}

// PrivateKeyExtensions are file extensions (leading dot) classified
// as private keys regardless of basename.
var PrivateKeyExtensions = []string{".pem", ".key"}

// SecretFileExtensions are extensions that, combined with "secret" in
// the basename, signal a secrets file (e.g. app-secret.json).
var SecretFileExtensions = []string{".json", ".yaml", ".yml"}

// SensitiveReason inspects the basename of path and returns a short
// human-readable reason when the file looks like it contains secrets,
// credentials, or keys. Returns "" when the path is safe to write.
// Detection is case-insensitive and ignores directory context.
//
// The reason string is intended to be embedded verbatim into the
// tool's refusal message, so it's phrased as a noun phrase
// ("environment file (may contain secrets)", not "this is an env
// file"). Callers add the framing.
func SensitiveReason(path string) string {
	name := strings.ToLower(filepath.Base(path))
	if name == "" || name == "." || name == "/" {
		return ""
	}

	// .env family: .env, .env.local, .env.production — but not
	// .env.example or .env.sample (those are template files meant
	// to be committed to a repo).
	if name == ".env" {
		return "environment file (may contain secrets)"
	}
	if strings.HasPrefix(name, ".env.") {
		safe := false
		for _, suf := range EnvExampleSuffixes {
			if strings.HasSuffix(name, suf) {
				safe = true
				break
			}
		}
		if !safe {
			return "environment file (may contain secrets)"
		}
	}

	// Private keys: known basenames first, then known extensions.
	for _, kn := range PrivateKeyNames {
		if name == kn {
			return "private key file"
		}
	}
	for _, ext := range PrivateKeyExtensions {
		if strings.HasSuffix(name, ext) {
			return "private key file"
		}
	}

	// Exact credentials filenames.
	for _, cn := range CredentialNames {
		if name == cn {
			return "credentials file"
		}
	}

	// Generic secret files: basename contains "secret" and has a
	// data-format extension. Catches app-secret.json, secret.yaml,
	// secrets-store.json without false-positiving on a README that
	// happens to say "secret" in the filename.
	if strings.Contains(name, "secret") {
		for _, ext := range SecretFileExtensions {
			if strings.HasSuffix(name, ext) {
				return "secrets file"
			}
		}
	}

	return ""
}
