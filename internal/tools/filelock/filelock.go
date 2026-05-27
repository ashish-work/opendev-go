// Package filelock holds the per-file mutex map and the atomic-write
// helper shared by every tool that mutates files on disk.
//
// The motivating constraint: write_file and edit_file may both touch
// the same path. If each tool keeps its own mutex map, they don't
// coordinate and two concurrent writes to the same file can race. A
// single package-level map keyed by absolute path is the smallest
// abstraction that solves it.
//
// Locks are never deleted from the map — they're cheap (24 bytes each)
// and the set of files touched in a session is bounded. Garbage-
// collecting would require ref-counting, which is not worth the
// complexity here.
//
// Usage:
//
//	mu := filelock.For(absPath)
//	mu.Lock()
//	defer mu.Unlock()
//	// ... read/modify/write the file ...
//	if err := filelock.AtomicWrite(absPath, newContent, mode); err != nil { ... }
package filelock

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// locks maps absolute paths to per-file mutexes. sync.Map is the right
// shape here: many keys (one per touched file), each touched briefly
// during read/modify/write. LoadOrStore atomically creates the mutex
// the first time a path is locked.
var locks sync.Map

// For returns the (singleton) mutex for the given absolute path. Always
// paired with Lock / defer Unlock at the call site. Pass an absolute
// path — relative paths would yield a different mutex per working
// directory, defeating the coordination point.
func For(absPath string) *sync.Mutex {
	actual, _ := locks.LoadOrStore(absPath, &sync.Mutex{})
	return actual.(*sync.Mutex)
}

// AtomicWrite writes data to a temp file in the same directory, then
// renames it over the target. rename(2) is atomic on POSIX within a
// single filesystem, so the file is either fully updated or fully
// preserved — never half-written. mode is applied to the new file
// (or matches the existing file's permissions when the caller stat'd
// the original before locking).
func AtomicWrite(path string, data []byte, mode os.FileMode) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".write-*")
	if err != nil {
		return fmt.Errorf("create temp: %w", err)
	}
	tmpName := tmp.Name()

	cleanup := func() { _ = os.Remove(tmpName) }

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		cleanup()
		return fmt.Errorf("write temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		cleanup()
		return fmt.Errorf("close temp: %w", err)
	}
	if err := os.Chmod(tmpName, mode); err != nil {
		cleanup()
		return fmt.Errorf("chmod temp: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		cleanup()
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}
