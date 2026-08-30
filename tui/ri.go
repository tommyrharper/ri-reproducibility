// TUI data comes from ./ri commands; scripts remain authoritative.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

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
