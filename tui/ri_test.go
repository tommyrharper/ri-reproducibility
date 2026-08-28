// The checks for the parts of the TUI that are not just drawing: what order
// runs come out in, which flags a launch actually sends, and finding the repo.
package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textinput"
)

func TestParseRunsPutsLiveRunsFirst(t *testing.T) {
	runs, err := parseRuns([]byte(`[
		{"name": "done", "status": "complete", "algorithm": "wsclean", "evaluations": 9322},
		{"name": "stopped", "status": "resumable", "algorithm": "r2d2", "evaluations": 12},
		{"name": "going", "status": "running", "algorithm": "r2d2", "evaluations": 3}
	]`))
	if err != nil {
		t.Fatal(err)
	}
	var order []string
	for _, run := range runs {
		order = append(order, run.Name)
	}
	if got := strings.Join(order, ","); got != "going,stopped,done" {
		t.Errorf("runs out of order: %s", got)
	}
	if runs[0].Evaluations != 3 || runs[0].Algorithm != "r2d2" {
		t.Errorf("run fields lost: %+v", runs[0])
	}
}

func TestSearchArgsOmitsEmptyFields(t *testing.T) {
	fields := []field{
		{flag: "nlive", input: textinput.New()},
		{flag: "mpi-procs", input: textinput.New()},
		{flag: "metric", input: textinput.New()},
	}
	fields[0].input.SetValue("50")
	fields[2].input.SetValue("  ") // whitespace is not a value

	got := strings.Join(searchArgs("wsclean", fields), " ")
	if got != "search wsclean --nlive 50" {
		t.Errorf("unexpected search arguments: %q", got)
	}
}

func TestRepoRootWalksUp(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "ri"), []byte("#!/usr/bin/env python3\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	deep := filepath.Join(root, "tui", "nested")
	if err := os.MkdirAll(deep, 0o755); err != nil {
		t.Fatal(err)
	}
	if got, err := repoRoot(deep); err != nil || got != root {
		t.Errorf("repoRoot(%q) = %q, %v", deep, got, err)
	}
	if _, err := repoRoot(t.TempDir()); err == nil {
		t.Error("repoRoot found a repository that is not there")
	}
}

func TestTailKeepsWholeLinesFromTheEnd(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run.log")
	if err := os.WriteFile(path, []byte("first\nsecond\nthird\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A limit that lands mid-line drops that partial line rather than showing
	// its tail, which is what makes a megabyte run.log readable.
	if got, err := tail(path, 12); err != nil || got != "third\n" {
		t.Errorf("tail = %q, %v", got, err)
	}
	if got, err := tail(path, 1000); err != nil || got != "first\nsecond\nthird\n" {
		t.Errorf("tail of a short file = %q, %v", got, err)
	}
}
