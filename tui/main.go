// ri-tui - a terminal front end for the failure-mode search: which runs exist,
// which are going, how healthy they are, and start another one.
//
// Started with `./ri tui`. Three screens, and nothing else: the run table, a
// pane that re-runs ./ri health, its log or its profile on a timer, and a form
// that launches a search.
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

// The sides of a run the pane can show, in the order `l` loops through them:
// is it progressing, what is it saying, and where did its time go.
type pane int

const (
	paneHealth pane = iota
	paneLog
	paneProfile
	paneCount
)

var paneNames = [paneCount]string{"health", "log", "profile"}

func (p pane) next() pane { return (p + 1) % paneCount }

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
	// The run the log pane is showing, and which of its sides: 'l' loops
	// through them, esc goes to the table, and enter there comes back here -
	// which is the whole navigation.
	logRun    Run
	pane      pane
	paused    bool
	refreshed time.Time

	imager   int
	fields   []field
	launches []launch
	focused  int

	notice string
	err    string
	width  int
	height int
}

var imagers = []string{"wsclean", "r2d2"}

// A search this session started. It sits in the table as `starting` until
// `./ri runs` can see its run directory, which takes as long as the image
// builds and the memory guard in front of the run do - a run started here used
// to be the one run the table did not list.
//
// The log outlives the wait: it holds the build output and any failure from
// before the run directory existed, so it stays the log side of `l` for this
// run for as long as the interface is up.
type launch struct {
	run Run
	log string
}

func newModel(r ri) model {
	columns := []table.Column{
		{Title: "run", Width: 40},
		// Each as wide as its widest value ("wsclean", "incomplete"), because
		// what those two characters buy on an 80-column terminal is the clock
		// time in the column below.
		{Title: "imager", Width: 7},
		{Title: "status", Width: 10},
		{Title: "evals", Width: 7},
		// "yesterday 22:54 (22h ago)" is the longest this gets.
		{Title: "started", Width: 25},
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

// profile is the per-stage timing table for a run: where its time went, and
// how much of it was not an evaluation at all. Errors and all, for the same
// reason health takes them - a run still going has no summary.json yet, and
// ./ri profile says so better than this program could.
func profile(run string) func(ri) (string, error) {
	return func(r ri) (string, error) {
		out, _ := r.run("profile", run)
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

// showRun opens the pane on one of a run's sides: the health report, the log
// it is writing, or where its time went. Everything that opens the pane goes
// through here, so `l` loops through the same views the table's enter arrives
// on.
func (m *model) showRun(run Run, p pane) tea.Cmd {
	m.logRun, m.pane = run, p
	switch p {
	case paneLog:
		if log := m.launchLogFor(run.Name); log != "" {
			return m.openLog(run.Name+" launch", launchLog(log))
		}
		return m.openLog(run.Name+" run.log", runLog(run.Path))
	case paneProfile:
		cmd := m.openLog(run.Name+" profile", profile(run.Name))
		// Opened paused, unlike the other two: a profile is written when the
		// run ends, so there is nothing for a timer to pick up, and re-reading
		// a summary.json that grows to megabytes every five seconds would cost
		// the host more than the run it is watching. `r` and `p` still work.
		m.paused = true
		return cmd
	}
	return m.openLog(run.Name, health(run.Name))
}

func (m model) selected() (Run, bool) {
	rows := m.visible()
	if i := m.table.Cursor(); i >= 0 && i < len(rows) {
		return rows[i], true
	}
	return Run{}, false
}

func (m model) visible() []Run {
	listed := map[string]bool{}
	for _, run := range m.runs {
		listed[run.Name] = true
	}
	var runs []Run
	for _, l := range m.launches {
		if !listed[l.run.Name] {
			runs = append(runs, l.run)
		}
	}
	runs = append(runs, m.runs...)
	if !m.runningOnly {
		return runs
	}
	var live []Run
	for _, run := range runs {
		// `starting` is a run of ours that has not reached the process table
		// yet, which is the one thing a running-only table must not hide.
		if run.Status == "running" || run.Status == "starting" {
			live = append(live, run)
		}
	}
	return live
}

// launchLogFor is the log a run this session started is writing to, or "" for
// any other run.
func (m model) launchLogFor(name string) string {
	for _, l := range m.launches {
		if l.run.Name == name {
			return l.log
		}
	}
	return ""
}

func (m *model) setRows() {
	rows := []table.Row{}
	for _, run := range m.visible() {
		rows = append(rows, table.Row{
			run.Name, run.Algorithm, run.Status,
			strconv.Itoa(run.Evaluations), run.StartedLabel,
		})
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
		//
		// The name is the identity - two runs of the same imager differ only in
		// the timestamp at the end of it - so on a narrow terminal it keeps the
		// 29 characters that takes and `started`, which says the same thing in
		// a form that can be read, gives up its width instead of the name's
		// tail. 34 is what the other three columns cost, padding included.
		name := min(44, max(29, msg.Width-59))
		columns[0].Width = name
		columns[4].Width = min(25, max(12, msg.Width-34-name))
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
		// A health report and a log are both read from their end, and both
		// grow, so the pane follows them. A profile is a fixed table whose
		// first line is the biggest stage - following it would open every
		// profile on its footnote.
		if atBottom && m.pane != paneProfile {
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
		// A run that has not claimed its directory yet has nothing for
		// ./ri health to report on, and everything to say in its log. The
		// command is bound first because showRun sets up the model it returns.
		openOn := paneHealth
		if run.Status == "starting" {
			openOn = paneLog
		}
		cmd := m.showRun(run, openOn)
		return m, cmd
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
		cmd := m.showRun(m.logRun, m.pane.next())
		return m, cmd
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
		imager := imagers[m.imager]
		dir, err := m.ri.claimRunDir(imager)
		if err != nil {
			m.err = err.Error()
			return m, nil
		}
		args := searchArgs(imager, m.fields, dir)
		log, err := m.ri.launch(dir, args)
		if err != nil {
			// The directory was claimed for a run that never started, so give
			// it back rather than leave a name no run can be given again.
			os.Remove(filepath.Join(m.ri.root, dir))
			m.err = err.Error()
			return m, nil
		}
		run := Run{
			Name: filepath.Base(dir), Path: dir, Algorithm: imager,
			Status: "starting", StartedLabel: "just now",
		}
		// Newest first, like the listing it is prepended to.
		m.launches = append([]launch{{run: run, log: log}}, m.launches...)
		m.err, m.notice = "", "launched "+run.Name
		m.setRows()
		m.table.SetCursor(0)
		cmd := m.showRun(run, paneLog)
		return m, cmd
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
		"enter watch  ·  n new run  ·  a running only  ·  r refresh  ·  q quit"))
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
	help := "esc runs  ·  l " + paneNames[m.pane.next()] +
		"  ·  r refresh  ·  p pause  ·  ↑/↓ scroll"
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
