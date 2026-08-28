// ri-tui - a terminal front end for the failure-mode search: which runs exist,
// which are going, how healthy they are, and start another one.
//
// Started with `./ri tui`. Three screens, and nothing else: the run table, a
// pane that re-runs ./ri health on a timer, and a form that launches a search.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/table"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// How often a shown health report or log is re-read. A health report costs a
// process spawn and a walk of the run's evaluation directories, so this is slow
// enough not to sit on the host and quick enough to watch a run by.
const refreshEvery = 5 * time.Second

var (
	titleStyle  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("6"))
	helpStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	errStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("1"))
	noticeStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("2"))
	focusStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("6"))
)

type screen int

const (
	screenRuns screen = iota
	screenLog
	screenForm
)

type runsMsg struct {
	runs []Run
	err  error
}

type logMsg struct {
	body string
	err  error
	at   time.Time
}

type tickMsg time.Time

// A field of the new-run form. flag is the ./ri search flag it fills in.
type field struct {
	flag  string
	hint  string
	input textinput.Model
}

func (f field) value() string { return f.input.Value() }

type model struct {
	ri     ri
	screen screen

	table       table.Model
	runs        []Run
	runningOnly bool

	view     viewport.Model
	logTitle string
	// What the log pane re-runs on a refresh. Set when the pane is opened.
	reload func(ri) (string, error)
	// Set when the pane is showing a run, so 'l' can swap health for run.log.
	logRun     string
	showingLog bool
	paused     bool
	refreshed  time.Time

	imager  int
	fields  []field
	focused int

	notice string
	err    string
	width  int
	height int
}

var imagers = []string{"wsclean", "r2d2"}

func newModel(r ri) model {
	columns := []table.Column{
		{Title: "run", Width: 40},
		{Title: "imager", Width: 8},
		{Title: "status", Width: 11},
		{Title: "evaluations", Width: 11},
	}
	t := table.New(table.WithColumns(columns), table.WithFocused(true))
	style := table.DefaultStyles()
	style.Header = style.Header.BorderForeground(lipgloss.Color("8")).Bold(true)
	style.Selected = style.Selected.Foreground(lipgloss.Color("0")).Background(lipgloss.Color("6")).Bold(true)
	t.SetStyles(style)

	m := model{ri: r, table: t, view: viewport.New(80, 20)}
	for _, f := range []field{
		{flag: "nlive", hint: "live points"},
		{flag: "num-repeats", hint: "exploration per live point"},
		{flag: "max-ndead", hint: "dead-point budget, -1 for none"},
		{flag: "mpi-procs", hint: "MPI ranks"},
		{flag: "metric", hint: "objective, e.g. total_rms_jy"},
		{flag: "seed", hint: "PolyChord seed"},
	} {
		f.input = textinput.New()
		f.input.Placeholder = "defaults.toml"
		f.input.Prompt = ""
		f.input.Width = 24
		m.fields = append(m.fields, f)
	}
	return m
}

func (m model) Init() tea.Cmd {
	return tea.Batch(loadRuns(m.ri), tick())
}

func tick() tea.Cmd {
	return tea.Tick(refreshEvery, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func loadRuns(r ri) tea.Cmd {
	return func() tea.Msg {
		out, err := r.run("runs", "--json")
		if err != nil {
			return runsMsg{err: fmt.Errorf("./ri runs --json: %s", strings.TrimSpace(out))}
		}
		runs, err := parseRuns([]byte(out))
		return runsMsg{runs: runs, err: err}
	}
}

func loadLog(r ri, load func(ri) (string, error)) tea.Cmd {
	return func() tea.Msg {
		body, err := load(r)
		return logMsg{body: body, err: err, at: time.Now()}
	}
}

func health(run string) func(ri) (string, error) {
	return func(r ri) (string, error) {
		// Errors and all: ./ri health explains a run it cannot report on
		// better than this program could.
		out, _ := r.run("health", run)
		return out, nil
	}
}

func runLog(path string) func(ri) (string, error) {
	return func(r ri) (string, error) {
		return tail(filepath.Join(r.root, path, "run.log"), 200_000)
	}
}

func launchLog(path string) func(ri) (string, error) {
	return func(ri) (string, error) { return tail(path, 200_000) }
}

func (m *model) openLog(title string, load func(ri) (string, error)) tea.Cmd {
	m.screen, m.logTitle, m.reload, m.paused = screenLog, title, load, false
	m.view.SetContent("loading...")
	m.view.GotoTop()
	return loadLog(m.ri, load)
}

func (m model) selected() (Run, bool) {
	rows := m.visible()
	if i := m.table.Cursor(); i >= 0 && i < len(rows) {
		return rows[i], true
	}
	return Run{}, false
}

func (m model) visible() []Run {
	if !m.runningOnly {
		return m.runs
	}
	var live []Run
	for _, run := range m.runs {
		if run.Status == "running" {
			live = append(live, run)
		}
	}
	return live
}

func (m *model) setRows() {
	rows := []table.Row{}
	for _, run := range m.visible() {
		rows = append(rows, table.Row{run.Name, run.Algorithm, run.Status, strconv.Itoa(run.Evaluations)})
	}
	m.table.SetRows(rows)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.table.SetHeight(max(3, msg.Height-6))
		columns := m.table.Columns()
		// Wide enough for a run name and no wider: the other columns are what
		// the eye is scanning, and a 40-character gap between them hurts.
		columns[0].Width = min(44, max(20, msg.Width-38))
		m.table.SetColumns(columns)
		m.view.Width, m.view.Height = msg.Width, max(3, msg.Height-4)

	case tickMsg:
		var cmd tea.Cmd
		switch {
		case m.screen == screenLog && !m.paused:
			cmd = loadLog(m.ri, m.reload)
		case m.screen == screenRuns:
			cmd = loadRuns(m.ri)
		}
		return m, tea.Batch(cmd, tick())

	case runsMsg:
		if msg.err != nil {
			m.err = msg.err.Error()
			return m, nil
		}
		// Follow the run the cursor was on rather than its index: a refresh
		// can reorder the table under it when a run starts or finishes.
		var was string
		if run, ok := m.selected(); ok {
			was = run.Name
		}
		m.err, m.runs = "", msg.runs
		m.setRows()
		for i, run := range m.visible() {
			if run.Name == was {
				m.table.SetCursor(i)
			}
		}

	case logMsg:
		if msg.err != nil {
			m.view.SetContent(msg.err.Error())
			return m, nil
		}
		atBottom := m.view.AtBottom()
		// Wrapped rather than cut off: a health report is wider than a narrow
		// terminal, and the line most likely to run over is the warning that
		// says what to do about the run.
		m.view.SetContent(lipgloss.NewStyle().Width(m.view.Width).Render(msg.body))
		if atBottom {
			m.view.GotoBottom()
		}
		m.refreshed = msg.at

	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC {
			return m, tea.Quit
		}
		switch m.screen {
		case screenRuns:
			return m.updateRuns(msg)
		case screenLog:
			return m.updateLog(msg)
		case screenForm:
			return m.updateForm(msg)
		}
	}

	var cmd tea.Cmd
	switch m.screen {
	case screenRuns:
		m.table, cmd = m.table.Update(msg)
	case screenLog:
		m.view, cmd = m.view.Update(msg)
	}
	return m, cmd
}

func (m model) updateRuns(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "q":
		return m, tea.Quit
	case "r":
		return m, loadRuns(m.ri)
	case "a":
		m.runningOnly = !m.runningOnly
		m.setRows()
		m.table.SetCursor(0)
		return m, nil
	case "n":
		// Opens on the imager row, which is the first choice to make and takes
		// no typing, so every text field starts blurred.
		m.screen, m.focused, m.notice = screenForm, 0, ""
		for i := range m.fields {
			m.fields[i].input.Blur()
		}
		return m, nil
	case "enter":
		run, ok := m.selected()
		if !ok {
			return m, nil
		}
		m.logRun, m.showingLog = run.Path, false
		return m, m.openLog(run.Name, health(run.Name))
	}
	var cmd tea.Cmd
	m.table, cmd = m.table.Update(msg)
	return m, cmd
}

func (m model) updateLog(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "q", "esc", "backspace":
		m.screen = screenRuns
		return m, loadRuns(m.ri)
	case "r":
		return m, loadLog(m.ri, m.reload)
	case "p":
		m.paused = !m.paused
		return m, nil
	case "l":
		if m.logRun == "" {
			return m, nil
		}
		m.showingLog = !m.showingLog
		name := filepath.Base(m.logRun)
		if m.showingLog {
			return m, m.openLog(name+" run.log", runLog(m.logRun))
		}
		return m, m.openLog(name, health(name))
	}
	var cmd tea.Cmd
	m.view, cmd = m.view.Update(msg)
	return m, cmd
}

func (m model) updateForm(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// Row 0 is the imager choice; the text fields follow it.
	switch msg.String() {
	case "esc":
		m.screen = screenRuns
		return m, nil
	case "tab", "down", "shift+tab", "up":
		step := 1
		if msg.String() == "shift+tab" || msg.String() == "up" {
			step = -1
		}
		m.focused = (m.focused + step + len(m.fields) + 1) % (len(m.fields) + 1)
		var cmds []tea.Cmd
		for i := range m.fields {
			if i+1 == m.focused {
				cmds = append(cmds, m.fields[i].input.Focus())
			} else {
				m.fields[i].input.Blur()
			}
		}
		return m, tea.Batch(cmds...)
	case "left", "right":
		if m.focused == 0 {
			m.imager = (m.imager + 1) % len(imagers)
			return m, nil
		}
	case "enter":
		args := searchArgs(imagers[m.imager], m.fields)
		log, err := m.ri.launch(args)
		if err != nil {
			m.err = err.Error()
			return m, nil
		}
		m.notice = "launched: ./ri " + strings.Join(args, " ")
		m.logRun, m.showingLog = "", false
		return m, m.openLog(filepath.Base(log), launchLog(log))
	}
	if m.focused == 0 {
		return m, nil
	}
	var cmd tea.Cmd
	m.fields[m.focused-1].input, cmd = m.fields[m.focused-1].input.Update(msg)
	return m, cmd
}

func (m model) View() string {
	switch m.screen {
	case screenLog:
		return m.logView()
	case screenForm:
		return m.formView()
	}
	return m.runsView()
}

func (m model) runsView() string {
	title := "ri runs"
	if m.runningOnly {
		title += "  (running only)"
	}
	lines := []string{titleStyle.Render(title), m.table.View()}
	if len(m.visible()) == 0 {
		empty := "No runs under results/nested-sampling yet - press n to start one."
		if m.runningOnly {
			empty = "Nothing running right now - press a for every run."
		}
		lines = append(lines, helpStyle.Render(empty))
	}
	if m.notice != "" {
		lines = append(lines, noticeStyle.Render(m.notice))
	}
	if m.err != "" {
		lines = append(lines, errStyle.Render(m.err))
	}
	lines = append(lines, helpStyle.Render(
		"enter health  ·  n new run  ·  a running only  ·  r refresh  ·  q quit"))
	return strings.Join(lines, "\n")
}

func (m model) logView() string {
	state := "refreshing every " + refreshEvery.String()
	if m.paused {
		state = "paused"
	}
	if !m.refreshed.IsZero() {
		state += ", read " + m.refreshed.Format("15:04:05")
	}
	help := "esc back  ·  r refresh  ·  p pause  ·  ↑/↓ scroll  ·  q back"
	if m.logRun != "" {
		help = "esc back  ·  l " + map[bool]string{true: "health", false: "run.log"}[m.showingLog] +
			"  ·  r refresh  ·  p pause  ·  ↑/↓ scroll"
	}
	return strings.Join([]string{
		titleStyle.Render(m.logTitle) + helpStyle.Render("  "+state),
		m.view.View(),
		helpStyle.Render(help),
	}, "\n")
}

func (m model) formView() string {
	choice := []string{}
	for i, name := range imagers {
		if i == m.imager {
			choice = append(choice, focusStyle.Render("["+name+"]"))
		} else {
			choice = append(choice, " "+name+" ")
		}
	}
	cursor := func(row int) string {
		if m.focused == row {
			return focusStyle.Render("> ")
		}
		return "  "
	}
	lines := []string{
		titleStyle.Render("new search"),
		"",
		cursor(0) + fmt.Sprintf("%-14s", "imager") + strings.Join(choice, " "),
	}
	for i, f := range m.fields {
		lines = append(lines, cursor(i+1)+fmt.Sprintf("%-14s", f.flag)+
			f.input.View()+helpStyle.Render("  "+f.hint))
	}
	lines = append(lines, "",
		helpStyle.Render("Empty fields fall through to the environment and defaults.toml."),
		helpStyle.Render("The search runs detached, so quitting here does not stop it."))
	if m.err != "" {
		lines = append(lines, errStyle.Render(m.err))
	}
	lines = append(lines, "",
		helpStyle.Render("tab/↑↓ move  ·  ←/→ imager  ·  enter launch  ·  esc back"))
	return strings.Join(lines, "\n")
}

func main() {
	cwd, err := os.Getwd()
	if err == nil {
		var root string
		if root, err = repoRoot(cwd); err == nil {
			_, err = tea.NewProgram(newModel(ri{root: root}), tea.WithAltScreen()).Run()
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "ri tui:", err)
		os.Exit(1)
	}
}
