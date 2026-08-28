// Everything the TUI knows about this repo it learns by running ./ri.
//
// Nothing here reimplements a listing, a health report or a launch: the run
// table is ./ri runs --json, the health pane is the text ./ri health already
// prints, and a new search is ./ri search with the flags the form filled in.
// That is deliberate - those scripts are the authority on what a run is, and a
// second implementation in Go would drift from them the first time one changed.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"
)

// repoRoot walks up from dir to the checkout that holds ./ri, so the TUI works
// whether it was started by `./ri tui` (cwd tui/) or by hand from anywhere
// inside the repository.
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

// Run is one entry of ./ri runs --json. Fields it does not use are dropped.
type Run struct {
	Name        string            `json:"name"`
	Path        string            `json:"path"`
	Algorithm   string            `json:"algorithm"`
	Status      string            `json:"status"`
	Evaluations int               `json:"evaluations"`
	Settings    map[string]string `json:"settings"`
}

// statusRank puts the runs worth looking at first: what is going now, then what
// could be resumed, then the finished ones - newest first within each group,
// which is the order ./ri runs already lists them in.
var statusRank = map[string]int{"running": 0, "resumable": 1, "incomplete": 2, "complete": 3}

func rank(status string) int {
	if r, ok := statusRank[status]; ok {
		return r
	}
	return len(statusRank)
}

func parseRuns(out []byte) ([]Run, error) {
	var runs []Run
	if err := json.Unmarshal(out, &runs); err != nil {
		return nil, err
	}
	sort.SliceStable(runs, func(i, j int) bool { return rank(runs[i].Status) < rank(runs[j].Status) })
	return runs, nil
}

// searchArgs turns the new-run form into ./ri search arguments. An empty field
// is left out entirely, so the run falls through to the environment and then to
// defaults.toml exactly as it would from the shell.
func searchArgs(imager string, fields []field) []string {
	args := []string{"search", imager}
	for _, f := range fields {
		if value := strings.TrimSpace(f.value()); value != "" {
			args = append(args, "--"+f.flag, value)
		}
	}
	return args
}

type ri struct{ root string }

// run shells out to ./ri and returns what it printed: stdout when there is
// any, and stderr only when there is not.
//
// The two are kept apart because both streams carry noise for the other's
// purpose - `uv` writes its own progress to stderr, and an install line in the
// middle of --json output is not JSON any more - and because a non-zero exit
// is not an error here: ./ri health exits 1 for a run it has a warning about,
// having printed the whole report anyone wants to read. Stdout is a pipe, so
// that report arrives as plain text.
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

// launch starts a search detached, and returns the log it is writing to.
//
// Detached rather than in the foreground because a search runs for hours and
// the point of this program is to watch it: setsid means quitting the TUI, or
// closing the terminal, does not take the run with it. The run's own run.log
// only exists once the run directory is claimed, so the build output and any
// early failure land here instead.
func (r ri) launch(args []string) (string, error) {
	log := filepath.Join(r.root, "results",
		"tui-search-"+time.Now().UTC().Format("20060102T150405Z")+".log")
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
	// Reaped in the background: without this the launcher stays a zombie for as
	// long as the TUI is up, and ./ri runs would count it as a live process.
	go func() {
		cmd.Wait()
		f.Close()
	}()
	return log, nil
}

// tail returns the last few kilobytes of a file, whole lines only. A run.log
// grows to megabytes over a multi-day search, and only its end is ever read.
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
