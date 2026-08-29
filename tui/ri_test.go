// The checks for the parts of the TUI that are not just drawing: what order
// runs come out in, which flags a launch actually sends, and finding the repo.
package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
)

func TestParseRunsKeepsNewestFirst(t *testing.T) {
	// ./ri runs --json in the order it prints, which is newest first. The
	// newest run here is a finished one and the oldest is still going, so
	// anything reordering by status would show up.
	runs, err := parseRuns([]byte(`[
		{"name": "newest", "status": "complete", "algorithm": "wsclean", "evaluations": 9322,
		 "started_label": "today 18:38 (2h ago)"},
		{"name": "middle", "status": "resumable", "algorithm": "r2d2", "evaluations": 12,
		 "started_label": "yesterday 22:54 (22h ago)"},
		{"name": "oldest", "status": "running", "algorithm": "r2d2", "evaluations": 3,
		 "started_label": "Wed 26 Aug 18:46 (2d ago)"}
	]`))
	if err != nil {
		t.Fatal(err)
	}
	var order []string
	for _, run := range runs {
		order = append(order, run.Name)
	}
	if got := strings.Join(order, ","); got != "newest,middle,oldest" {
		t.Errorf("runs out of order: %s", got)
	}
	if runs[0].Evaluations != 9322 || runs[0].Algorithm != "wsclean" {
		t.Errorf("run fields lost: %+v", runs[0])
	}
	// The column is drawn from this string, so losing it empties the column
	// rather than failing anywhere.
	if runs[0].StartedLabel != "today 18:38 (2h ago)" {
		t.Errorf("started label lost: %+v", runs[0])
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

	got := strings.Join(searchArgs("wsclean", fields, "results/nested-sampling/x"), " ")
	if got != "search wsclean --output-dir results/nested-sampling/x --nlive 50" {
		t.Errorf("unexpected search arguments: %q", got)
	}
}

func TestClaimRunDirNamesARunTheScriptsRecognise(t *testing.T) {
	r := ri{root: t.TempDir()}
	dir, err := r.claimRunDir("r2d2")
	if err != nil {
		t.Fatal(err)
	}
	// The shape ./ri runs sorts by and nested-sampling-health.py parses: the
	// prefix ns_claim_run_dir would have used, and a whole-second UTC stamp.
	if !regexp.MustCompile(`^results/nested-sampling/r2d2-vlaa-\d{8}T\d{6}Z$`).MatchString(dir) {
		t.Errorf("run directory not named like a run: %q", dir)
	}
	// Created, and created as a claim: the second search of the same second
	// must not be handed the directory the first one is about to fill.
	if info, err := os.Stat(filepath.Join(r.root, dir)); err != nil || !info.IsDir() {
		t.Fatalf("run directory not created: %v", err)
	}
	again, err := r.claimRunDir("r2d2")
	if err != nil || again == dir {
		t.Errorf("claimRunDir handed out %q twice (%v)", again, err)
	}
}

// A launched run belongs in the table before ./ri runs can see it, and has to
// leave it again - once, not twice - when the listing catches up.
func TestVisibleShowsALaunchedRunUntilTheListingHasIt(t *testing.T) {
	pending := Run{Name: "wsclean-vlaa-20260101T000000Z", Status: "starting"}
	m := model{
		launches: []launch{{run: pending, log: "/tmp/launch.log"}},
		runs:     []Run{{Name: "older", Status: "complete"}},
	}
	if got := m.visible(); len(got) != 2 || got[0].Name != pending.Name {
		t.Fatalf("launched run missing from the table: %+v", got)
	}
	// Running-only is the one view most likely to hide it, and the one a run
	// just started is most likely to be watched from.
	m.runningOnly = true
	if got := m.visible(); len(got) != 1 || got[0].Name != pending.Name {
		t.Errorf("launched run hidden by the running-only filter: %+v", got)
	}
	m.runningOnly = false
	m.runs = append(m.runs, Run{Name: pending.Name, Status: "running"})
	got := m.visible()
	if len(got) != 2 {
		t.Fatalf("run listed twice once ./ri runs saw it: %+v", got)
	}
	if got[1].Status != "running" {
		t.Errorf("stale pending row shown instead of the real one: %+v", got)
	}
	// The log stays with the run: it holds the build output, which the run's
	// own run.log never had.
	if m.launchLogFor(pending.Name) != "/tmp/launch.log" {
		t.Error("launch log lost once the run was listed")
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

// The loop the interface exists for: into a run, around its three views, back
// out to the table, and in again. The commands returned would shell out to
// ./ri, so they are dropped - what is checked here is where each key lands.
func TestWatchingARunLoopsBackToTheTable(t *testing.T) {
	m := newModel(ri{root: t.TempDir()})
	m.runs = []Run{{Name: "wsclean-vlaa-20260101T000000Z", Path: "results/nested-sampling/x",
		Status: "running"}}
	m.setRows()

	press := func(key string) {
		t.Helper()
		next, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(key)})
		m = next.(model)
	}
	escape := func() {
		t.Helper()
		next, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
		m = next.(model)
	}

	for round := 0; round < 2; round++ {
		next, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
		m = next.(model)
		if m.screen != screenLog || m.pane != paneHealth {
			t.Fatalf("round %d: enter did not open the health report (screen %v, pane %v)",
				round, m.screen, m.pane)
		}
		if m.logRun.Name != m.runs[0].Name {
			t.Fatalf("round %d: log pane is showing %q", round, m.logRun.Name)
		}
		// `l` all the way around: health, log, profile, and back to health.
		for _, want := range []pane{paneLog, paneProfile, paneHealth} {
			press("l")
			if m.pane != want {
				t.Fatalf("round %d: l landed on %v, wanted %v", round, m.pane, want)
			}
		}
		// The profile is a finished-run report, so it opens paused; the two
		// live views must not stay that way behind it.
		press("l")
		press("l")
		if m.pane != paneProfile || !m.paused {
			t.Fatalf("round %d: the profile pane did not open paused (pane %v, paused %v)",
				round, m.pane, m.paused)
		}
		press("l")
		if m.pane != paneHealth || m.paused {
			t.Fatalf("round %d: the health pane stayed paused behind the profile", round)
		}
		escape()
		if m.screen != screenRuns {
			t.Fatalf("round %d: esc did not return to the table", round)
		}
	}
}
