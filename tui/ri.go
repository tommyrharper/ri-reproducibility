// TUI data comes from ./ri commands; scripts remain authoritative.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func repoRoot(dir string) (string, error) {
	for {
		if info, err := os.Stat(filepath.Join(dir, "ri")); err == nil && !info.IsDir() {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("no ./ri found above the working directory")
		}
		dir = parent
	}
}

// Run is one entry of ./ri runs --json.
type Run struct {
	Name         string            `json:"name"`
	Path         string            `json:"path"`
	Algorithm    string            `json:"algorithm"`
	Status       string            `json:"status"`
	Evaluations  int               `json:"evaluations"`
	StartedLabel string            `json:"started_label"`
	Settings     map[string]string `json:"settings"`
	Space        []Param           `json:"parameter_space"`
	// SpaceFromDefaults marks a Space that is defaults.toml standing in for a
	// run that died before recording its own. defaults.toml is edited between
	// runs, so this box is the repository's now, not necessarily the run's.
	SpaceFromDefaults bool `json:"parameter_space_from_defaults"`
	// Preliminary marks a run scanned off disk rather than listed by ./ri: its
	// name and age are real, and every other field is simply not known yet.
	// Never set from JSON - the listing is the thing that settles a run.
	Preliminary bool `json:"-"`
}

// Param is one searched dimension of a run's parameter-space.json. A
// band_start dimension has its resolved bounds, so min/max cover every kind,
// but a hand-made run directory may have neither.
type Param struct {
	Name string   `json:"name"`
	Min  *float64 `json:"min"`
	Max  *float64 `json:"max"`
}

// objective is what the sampler was maximizing, from the run's run.env.
func (r Run) objective() string {
	if metric := r.Settings["NS_METRIC"]; metric != "" {
		return metric
	}
	return "-"
}

// age is the "(2h ago)" tail of the started label, as 2h: the run name already
// carries the start timestamp, and the column header already says age.
func (r Run) age() string {
	label := r.StartedLabel
	if open := strings.LastIndex(label, "("); open >= 0 {
		label = strings.TrimSuffix(label[open+1:], ")")
	}
	if label == "just now" {
		return "now"
	}
	return strings.TrimSuffix(label, " ago")
}

// ranges renders the searched box. Abbreviated it is a table cell; in full it
// is the detail line under the table, which is where the initials are decoded.
func (r Run) ranges(abbreviated bool) string {
	var parts []string
	for _, p := range r.Space {
		name := p.Name
		if abbreviated {
			name = initials(p.Name)
		}
		if p.Min == nil || p.Max == nil {
			parts = append(parts, name)
			continue
		}
		parts = append(parts, name+" "+round(*p.Min)+"-"+round(*p.Max))
	}
	if len(parts) == 0 {
		return "-"
	}
	if abbreviated {
		if r.SpaceFromDefaults {
			// A trailing ? rather than a longer note: the cell is already the
			// widest column, and the detail line below spells it out.
			return strings.Join(parts, " ") + " ?"
		}
		return strings.Join(parts, " ")
	}
	if r.SpaceFromDefaults {
		return strings.Join(parts, " · ") + "  (not recorded - showing defaults.toml)"
	}
	return strings.Join(parts, " · ")
}

// initials turns log10_dynamic_range into ldr, so five dimensions fit a cell.
func initials(name string) string {
	var short []byte
	for _, word := range strings.Split(name, "_") {
		if word != "" {
			short = append(short, word[0])
		}
	}
	return string(short)
}

// round keeps a bound readable in a cell: 5.4e+07 becomes 54M.
func round(v float64) string {
	for _, unit := range []struct {
		scale  float64
		suffix string
	}{{1e9, "G"}, {1e6, "M"}, {1e3, "k"}} {
		if math.Abs(v) >= unit.scale {
			return strconv.FormatFloat(v/unit.scale, 'g', 3, 64) + unit.suffix
		}
	}
	return strconv.FormatFloat(v, 'g', 3, 64)
}

// deletable reports why a run cannot be removed: pulling the directory out
// from under live ranks is what ns_refuse_live_run exists to prevent.
func (r Run) deletable() error {
	if r.Preliminary {
		// A scanned row does not know whether the run is live, and "not known
		// to be running" must never be read as "safe to delete".
		return fmt.Errorf("%s is still loading - wait for the table to fill in", r.Name)
	}
	if r.Status == "running" || r.Status == "starting" {
		return fmt.Errorf("%s is still going - stop it before deleting it", r.Name)
	}
	return nil
}

// runArtifacts mirrors RUN_ARTIFACTS in scripts/nested-sampling-runs.py: what
// makes a directory under results/nested-sampling a run rather than a note.
var runArtifacts = []string{"run.env", "run.log", "summary.json", "evaluations", "chains"}

// runStamp matches the UTC timestamp a run's name ends with.
var runStamp = regexp.MustCompile(`(\d{8}T\d{6}Z)$`)

// scanRuns is what can be known about the runs from a readdir and a handful of
// stats. ./ri runs --json is authoritative and replaces every row it returns,
// but it counts every evaluation and reads every run's settings first, so the
// table would otherwise sit empty for as long as that takes.
func (r ri) scanRuns() []Run {
	parent := filepath.Join(r.root, "results", "nested-sampling")
	entries, err := os.ReadDir(parent)
	if err != nil {
		return nil
	}
	type scanned struct {
		run     Run
		started time.Time
	}
	var found []scanned
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dir := filepath.Join(parent, entry.Name())
		if !isRunDir(dir) {
			continue
		}
		started := startedAt(entry, dir)
		found = append(found, scanned{
			run: Run{
				Name:         entry.Name(),
				Path:         filepath.Join("results", "nested-sampling", entry.Name()),
				StartedLabel: "(" + ago(started, time.Now()) + ")",
				Preliminary:  true,
			},
			started: started,
		})
	}
	// The same order the listing comes back in, so no row moves under the
	// cursor when it lands.
	sort.Slice(found, func(i, j int) bool {
		if !found[i].started.Equal(found[j].started) {
			return found[i].started.After(found[j].started)
		}
		return found[i].run.Name > found[j].run.Name
	})
	runs := make([]Run, 0, len(found))
	for _, f := range found {
		runs = append(runs, f.run)
	}
	return runs
}

func isRunDir(dir string) bool {
	for _, artifact := range runArtifacts {
		if _, err := os.Stat(filepath.Join(dir, artifact)); err == nil {
			return true
		}
	}
	return false
}

// startedAt reads the run's start from its name, as the listing does, and falls
// back to the directory's mtime for a hand-made one that has no stamp.
func startedAt(entry os.DirEntry, dir string) time.Time {
	if match := runStamp.FindStringSubmatch(entry.Name()); match != nil {
		if at, err := time.ParseInLocation("20060102T150405Z", match[1], time.UTC); err == nil {
			return at
		}
	}
	if info, err := entry.Info(); err == nil {
		return info.ModTime()
	}
	return time.Time{}
}

// ago is the tail of the listing's started label, with the same thresholds as
// format_started in scripts/nested-sampling-runs.py, so a scanned row's age
// reads the same as the one that replaces it.
func ago(started, now time.Time) string {
	age := now.Sub(started)
	if age < 0 {
		age = 0 // a stamp from the future is a skewed clock
	}
	switch {
	case age < 90*time.Second:
		return "just now"
	case age < time.Hour:
		return strconv.Itoa(int(age.Minutes())) + "m ago"
	case age < 24*time.Hour:
		return strconv.Itoa(int(age.Hours())) + "h ago"
	default:
		return strconv.Itoa(int(age.Hours()/24)) + "d ago"
	}
}

// deleteRun removes a run and everything scored under it, which is why it
// refuses anything but a directory ./ri runs itself would list.
func (r ri) deleteRun(run Run) error {
	if err := run.deletable(); err != nil {
		return err
	}
	dir := filepath.Join(r.root, run.Path)
	if filepath.Dir(dir) != filepath.Join(r.root, "results", "nested-sampling") {
		return fmt.Errorf("%q is not a run directory", run.Path)
	}
	if err := os.RemoveAll(dir); err != nil {
		return err
	}
	// The launch log outlives the run directory it describes.
	os.Remove(filepath.Join(r.root, "results", "tui-"+filepath.Base(dir)+".log"))
	return nil
}

func parseRuns(out []byte) ([]Run, error) {
	// Not `return runs, json.Unmarshal(out, &runs)`: the spec orders the call
	// against the other operands only for calls, so `runs` may be read before
	// Unmarshal fills it.
	var runs []Run
	err := json.Unmarshal(out, &runs)
	return runs, err
}

func searchArgs(t target, fields []field, outputDir string) []string {
	args := []string{"search", t.first, "--output-dir", outputDir}
	if t.then != "" {
		args = append(args, "--then", t.then)
	}
	for _, f := range fields {
		if value := strings.TrimSpace(f.value()); value != "" {
			args = append(args, "--"+f.flag, value)
		}
	}
	return args
}

func (r ri) claimRunDir(imager string) (string, error) {
	parent := filepath.Join(r.root, "results", "nested-sampling")
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return "", err
	}
	for i := 0; i < 10; i++ {
		name := imager + "-vlaa-" + time.Now().UTC().Format("20060102T150405Z")
		err := os.Mkdir(filepath.Join(parent, name), 0o755)
		if err == nil {
			return filepath.Join("results", "nested-sampling", name), nil
		}
		if !os.IsExist(err) {
			return "", err
		}
		time.Sleep(time.Second)
	}
	return "", fmt.Errorf("no free run directory under %s", parent)
}

type ri struct{ root string }

func (r ri) run(args ...string) (string, error) {
	cmd := exec.Command("./ri", args...)
	cmd.Dir = r.root
	var out, errs strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &errs
	err := cmd.Run()
	if strings.TrimSpace(out.String()) == "" {
		return errs.String(), err
	}
	return out.String(), err
}

func (r ri) launch(runDir string, args []string) (string, error) {
	log := filepath.Join(r.root, "results", "tui-"+filepath.Base(runDir)+".log")
	f, err := os.Create(log)
	if err != nil {
		return "", err
	}
	cmd := exec.Command("./ri", args...)
	cmd.Dir = r.root
	cmd.Stdout, cmd.Stderr = f, f
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		f.Close()
		return "", err
	}
	// Reap launcher in background so ./ri runs does not count its zombie.
	go func() {
		cmd.Wait()
		f.Close()
	}()
	return log, nil
}

func tail(path string, limit int64) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return "", err
	}
	start := info.Size() - limit
	if start < 0 {
		start = 0
	}
	buf := make([]byte, info.Size()-start)
	if _, err := f.ReadAt(buf, start); err != nil && len(buf) > 0 {
		return "", err
	}
	text := string(buf)
	if start > 0 {
		if cut := strings.IndexByte(text, '\n'); cut >= 0 {
			text = text[cut+1:]
		}
	}
	return text, nil
}
