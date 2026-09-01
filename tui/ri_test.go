package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
)

func TestParseRunsKeepsNewestFirst(t *testing.T) {
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
	if runs[0].StartedLabel != "today 18:38 (2h ago)" {
		t.Errorf("started label lost: %+v", runs[0])
	}
}

func TestRunCondensesItsObjectiveAndBoxIntoARow(t *testing.T) {
	runs, err := parseRuns([]byte(`[{"name": "r2d2-vlaa-20260827T143426Z",
		"started_label": "Wed 26 Aug 18:46 (2d ago)",
		"settings": {"NS_METRIC": "total_rms_jy"},
		"parameter_space": [
			{"name": "log10_dynamic_range", "min": 1.0, "max": 6.0},
			{"name": "channel_count", "min": 1, "max": 8, "kind": "integer"},
			{"name": "start_frequency_hz", "kind": "band_start",
			 "min": 54000000.0, "max": 50000000000.0},
			{"name": "channel_width_hz", "min": 100000.0, "max": 2000000.0}]}]`))
	if err != nil {
		t.Fatal(err)
	}
	run := runs[0]
	if got := run.age(); got != "2d" {
		t.Errorf("age = %q, wanted the bracketed tail without its ago", got)
	}
	if got := run.objective(); got != "total_rms_jy" {
		t.Errorf("objective = %q", got)
	}
	if got := run.ranges(true); got != "ldr 1-6 cc 1-8 sfh 54M-50G cwh 100k-2M" {
		t.Errorf("abbreviated ranges = %q", got)
	}
	if got := run.ranges(false); !strings.Contains(got, "log10_dynamic_range 1-6") ||
		!strings.Contains(got, "channel_width_hz 100k-2M") {
		t.Errorf("full ranges = %q", got)
	}

	// A run directory from before parameter-space.json, or a hand-made one.
	bare := Run{StartedLabel: "just now"}
	if bare.age() != "now" || bare.objective() != "-" || bare.ranges(true) != "-" {
		t.Errorf("a run with nothing recorded reads as %+v", bare)
	}
	half := Run{Space: []Param{{Name: "source_offset_fraction"}}}
	if got := half.ranges(true); got != "sof" {
		t.Errorf("a dimension without bounds = %q", got)
	}

	// A run that died before recording its box borrows defaults.toml's, which
	// must never read as the run's own record.
	stand := run
	stand.SpaceFromDefaults = true
	if got := stand.ranges(true); got != "ldr 1-6 cc 1-8 sfh 54M-50G cwh 100k-2M ?" {
		t.Errorf("a borrowed box must be marked in the cell: %q", got)
	}
	if got := stand.ranges(false); !strings.Contains(got, "not recorded") {
		t.Errorf("a borrowed box must be spelled out in the detail line: %q", got)
	}
	// Absent from the JSON means recorded, so an old ./ri cannot mislabel.
	if runs[0].SpaceFromDefaults {
		t.Error("parameter_space_from_defaults must default to false")
	}
}

func TestDeleteRunOnlyEverRemovesARunDirectory(t *testing.T) {
	r := ri{root: t.TempDir()}
	dir, err := r.claimRunDir("r2d2")
	if err != nil {
		t.Fatal(err)
	}
	run := Run{Name: filepath.Base(dir), Path: dir, Status: "complete"}
	log := filepath.Join(r.root, "results", "tui-"+run.Name+".log")
	if err := os.WriteFile(log, []byte("launched\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, blocked := range []Run{
		{Name: run.Name, Path: run.Path, Status: "running"},
		{Name: run.Name, Path: run.Path, Status: "starting"},
		{Name: "results", Path: "results"},
		{Name: "root", Path: ""},
		{Name: "escape", Path: "results/nested-sampling/../.."},
	} {
		if err := r.deleteRun(blocked); err == nil {
			t.Errorf("deleted %+v, which is not a stopped run directory", blocked)
		}
	}
	if _, err := os.Stat(filepath.Join(r.root, dir)); err != nil {
		t.Fatalf("a refused delete removed the run anyway: %v", err)
	}

	if err := r.deleteRun(run); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(r.root, dir)); !os.IsNotExist(err) {
		t.Errorf("run directory survived the delete: %v", err)
	}
	if _, err := os.Stat(log); !os.IsNotExist(err) {
		t.Errorf("launch log outlived the run it describes: %v", err)
	}
	if _, err := os.Stat(filepath.Join(r.root, "results", "nested-sampling")); err != nil {
		t.Errorf("deleting a run took its parent with it: %v", err)
	}
}

func TestDeletingFromTheTableTakesAConfirmation(t *testing.T) {
	r := ri{root: t.TempDir()}
	dir, err := r.claimRunDir("wsclean")
	if err != nil {
		t.Fatal(err)
	}
	kept, err := r.claimRunDir("r2d2")
	if err != nil {
		t.Fatal(err)
	}
	m := newModel(r)
	m.runs = []Run{
		{Name: filepath.Base(dir), Path: dir, Status: "complete"},
		{Name: filepath.Base(kept), Path: kept, Status: "complete"},
	}
	m.setRows()
	m.table.SetCursor(1)

	press := func(key string) {
		t.Helper()
		next, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(key)})
		m = next.(model)
	}

	press("d")
	if m.doomed != m.runs[1].Name {
		t.Fatalf("d armed %q, not the selected run", m.doomed)
	}
	press("n")
	if m.doomed != "" || len(m.runs) != 2 {
		t.Fatalf("anything but y must keep the run: %+v", m.runs)
	}
	if _, err := os.Stat(filepath.Join(r.root, kept)); err != nil {
		t.Fatalf("declining still deleted the run: %v", err)
	}

	// A run resumed elsewhere while the prompt was up must survive the y.
	press("d")
	m.runs[1].Status = "running"
	press("y")
	if _, err := os.Stat(filepath.Join(r.root, kept)); err != nil {
		t.Fatalf("a run that went live under the prompt was deleted: %v", err)
	}
	m.runs[1].Status = "complete"

	// And d refuses a live run outright rather than asking first.
	m.runs[1].Status = "running"
	press("d")
	if m.doomed != "" || m.err == "" {
		t.Errorf("d offered to delete a running run (doomed %q, err %q)", m.doomed, m.err)
	}
	m.runs[1].Status = "complete"

	press("d")
	press("y")
	if len(m.runs) != 1 || m.runs[0].Path != dir {
		t.Fatalf("confirmed delete left %+v", m.runs)
	}
	if _, err := os.Stat(filepath.Join(r.root, kept)); !os.IsNotExist(err) {
		t.Errorf("confirmed delete did not remove the directory: %v", err)
	}
	if m.table.Cursor() != 0 {
		t.Errorf("cursor left past the end of a shorter table: %d", m.table.Cursor())
	}
	if !strings.Contains(m.runsView(), "deleted "+filepath.Base(kept)) {
		t.Error("the table said nothing about the delete")
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

	got := strings.Join(searchArgs(target{first: "wsclean"}, fields, "results/nested-sampling/x"), " ")
	if got != "search wsclean --output-dir results/nested-sampling/x --nlive 50" {
		t.Errorf("unexpected search arguments: %q", got)
	}
}

// --output-dir names the first search's directory only, so it has to stay
// ahead of --then: the chained search claims its own.
func TestSearchArgsChainsTheSecondImager(t *testing.T) {
	fields := []field{{flag: "nlive", input: textinput.New()}}
	fields[0].input.SetValue("125")

	pair := target{first: "wsclean", then: "r2d2"}
	got := strings.Join(searchArgs(pair, fields, "results/nested-sampling/x"), " ")
	want := "search wsclean --output-dir results/nested-sampling/x --then r2d2 --nlive 125"
	if got != want {
		t.Errorf("unexpected chained arguments:\n got %q\nwant %q", got, want)
	}
	if pair.label() != "wsclean then r2d2" {
		t.Errorf("unexpected label: %q", pair.label())
	}
	if (target{first: "r2d2"}).label() != "r2d2" {
		t.Errorf("a single imager labels itself: %q", (target{first: "r2d2"}).label())
	}
}

// Both arrows used to step forward, which is invisible with two choices and
// wrong with four.
func TestFormArrowsWalkTheTargetsBothWays(t *testing.T) {
	m := newModel(ri{root: t.TempDir()})
	press := func(key tea.KeyType) {
		t.Helper()
		next, _ := m.Update(tea.KeyMsg{Type: key})
		m = next.(model)
	}
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("n")})
	m = next.(model)

	press(tea.KeyRight)
	if targets[m.imager].label() != "r2d2" {
		t.Fatalf("right went to %q", targets[m.imager].label())
	}
	press(tea.KeyLeft)
	if targets[m.imager].label() != "wsclean" {
		t.Fatalf("left went to %q", targets[m.imager].label())
	}
	press(tea.KeyLeft)
	if want := "r2d2 then wsclean"; targets[m.imager].label() != want {
		t.Fatalf("left off the start went to %q, want %q", targets[m.imager].label(), want)
	}
	press(tea.KeyRight)
	if targets[m.imager].label() != "wsclean" {
		t.Fatalf("right off the end went to %q", targets[m.imager].label())
	}
}

func TestClaimRunDirNamesARunTheScriptsRecognise(t *testing.T) {
	r := ri{root: t.TempDir()}
	dir, err := r.claimRunDir("r2d2")
	if err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`^results/nested-sampling/r2d2-vlaa-\d{8}T\d{6}Z$`).MatchString(dir) {
		t.Errorf("run directory not named like a run: %q", dir)
	}
	if info, err := os.Stat(filepath.Join(r.root, dir)); err != nil || !info.IsDir() {
		t.Fatalf("run directory not created: %v", err)
	}
	again, err := r.claimRunDir("r2d2")
	if err != nil || again == dir {
		t.Errorf("claimRunDir handed out %q twice (%v)", again, err)
	}
}

func TestVisibleShowsALaunchedRunUntilTheListingHasIt(t *testing.T) {
	pending := Run{Name: "wsclean-vlaa-20260101T000000Z", Status: "starting"}
	m := model{
		launches: []launch{{run: pending, log: "/tmp/launch.log"}},
		runs:     []Run{{Name: "older", Status: "complete"}},
	}
	if got := m.visible(); len(got) != 2 || got[0].Name != pending.Name {
		t.Fatalf("launched run missing from the table: %+v", got)
	}
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
	if got, err := tail(path, 12); err != nil || got != "third\n" {
		t.Errorf("tail = %q, %v", got, err)
	}
	if got, err := tail(path, 1000); err != nil || got != "first\nsecond\nthird\n" {
		t.Errorf("tail of a short file = %q, %v", got, err)
	}
}

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
		for _, want := range []pane{paneLog, paneProfile, paneHealth} {
			press("l")
			if m.pane != want {
				t.Fatalf("round %d: l landed on %v, wanted %v", round, m.pane, want)
			}
		}
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

func TestScanRunsReadsNamesAndAgesOffDisk(t *testing.T) {
	root := t.TempDir()
	parent := filepath.Join(root, "results", "nested-sampling")
	for name, artifact := range map[string]string{
		"wsclean-vlaa-20260101T000000Z": "run.env",
		"r2d2-vlaa-20260103T000000Z":    "chains",
		"r2d2-vlaa-20260102T000000Z":    "summary.json",
	} {
		if err := os.MkdirAll(filepath.Join(parent, name), 0o755); err != nil {
			t.Fatal(err)
		}
		if artifact == "chains" {
			if err := os.Mkdir(filepath.Join(parent, name, artifact), 0o755); err != nil {
				t.Fatal(err)
			}
			continue
		}
		if err := os.WriteFile(filepath.Join(parent, name, artifact), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// A directory with none of the run artifacts is not a run, exactly as
	// find_runs decides it in scripts/nested-sampling-runs.py.
	if err := os.MkdirAll(filepath.Join(parent, "notes-20260104T000000Z"), 0o755); err != nil {
		t.Fatal(err)
	}

	runs := ri{root: root}.scanRuns()
	var order []string
	for _, run := range runs {
		order = append(order, run.Name)
	}
	want := "r2d2-vlaa-20260103T000000Z,r2d2-vlaa-20260102T000000Z,wsclean-vlaa-20260101T000000Z"
	if got := strings.Join(order, ","); got != want {
		t.Errorf("scan out of order:\n got %s\nwant %s", got, want)
	}
	for _, run := range runs {
		if !run.Preliminary {
			t.Errorf("%s is not marked preliminary: %+v", run.Name, run)
		}
		if run.Path != filepath.Join("results", "nested-sampling", run.Name) {
			t.Errorf("%s has the wrong path: %q", run.Name, run.Path)
		}
		if run.age() == "" {
			t.Errorf("%s has no age: %q", run.Name, run.StartedLabel)
		}
	}
}

func TestScannedRunRefusesDeletion(t *testing.T) {
	// A scan cannot tell a live run from a finished one, and "not known to be
	// running" must never open the door to deleting one that is.
	scanned := Run{Name: "r2d2-vlaa-20260101T000000Z", Preliminary: true}
	err := scanned.deletable()
	if err == nil {
		t.Fatal("a preliminary run agreed to be deleted")
	}
	if !strings.Contains(err.Error(), "still loading") {
		t.Errorf("unhelpful refusal: %v", err)
	}
	settled := Run{Name: "r2d2-vlaa-20260101T000000Z", Status: "complete"}
	if err := settled.deletable(); err != nil {
		t.Errorf("a listed complete run refused deletion: %v", err)
	}
}

func TestScanFillsTheTableThenTheListingReplacesIt(t *testing.T) {
	m := newModel(ri{root: t.TempDir()})
	scanned := []Run{{Name: "r2d2-vlaa-20260101T000000Z", Preliminary: true,
		StartedLabel: "(2h ago)"}}

	updated, _ := m.Update(scanMsg{runs: scanned})
	m = updated.(model)
	if len(m.table.Rows()) != 1 {
		t.Fatalf("the scan did not reach the table: %v", m.table.Rows())
	}
	row := m.table.Rows()[0]
	if row[0] != "r2d2-vlaa-20260101T000000Z" || row[3] != "2h" {
		t.Errorf("scanned row lost its name or age: %v", row)
	}
	// Everything the scan cannot know stays blank; a 0 in evals would read as
	// a run that scored nothing.
	for _, i := range []int{1, 2, 4, 5} {
		if row[i] != "" {
			t.Errorf("column %d should be blank until the listing lands: %q", i, row[i])
		}
	}

	listed := []Run{{Name: "r2d2-vlaa-20260101T000000Z", Status: "complete",
		Evaluations: 12, StartedLabel: "today 09:00 (2h ago)"}}
	updated, _ = m.Update(runsMsg{runs: listed})
	m = updated.(model)
	row = m.table.Rows()[0]
	if row[1] != "complete" || row[2] != "12" {
		t.Errorf("the listing did not replace the scanned row: %v", row)
	}

	// A scan that lost the race must not put half-known rows back.
	updated, _ = m.Update(scanMsg{runs: scanned})
	m = updated.(model)
	if row = m.table.Rows()[0]; row[1] != "complete" || row[2] != "12" {
		t.Errorf("a late scan overwrote the listing: %v", row)
	}
}

func TestAgoMatchesTheListingsThresholds(t *testing.T) {
	now := time.Date(2026, 6, 15, 12, 0, 0, 0, time.UTC)
	for _, c := range []struct {
		back time.Duration
		want string
	}{
		{0, "just now"},
		{89 * time.Second, "just now"},
		{-time.Hour, "just now"}, // a stamp from the future is a skewed clock
		{20 * time.Minute, "20m ago"},
		{2 * time.Hour, "2h ago"},
		{30 * 24 * time.Hour, "30d ago"},
	} {
		if got := ago(now.Add(-c.back), now); got != c.want {
			t.Errorf("%v back: got %q, want %q", c.back, got, c.want)
		}
	}
}
