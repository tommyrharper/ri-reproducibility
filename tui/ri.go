// TUI data comes from ./ri commands; scripts remain authoritative.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
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
		return strings.Join(parts, " ")
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
	if r.Status == "running" || r.Status == "starting" {
		return fmt.Errorf("%s is still going - stop it before deleting it", r.Name)
	}
	return nil
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

func searchArgs(imager string, fields []field, outputDir string) []string {
	args := []string{"search", imager, "--output-dir", outputDir}
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
