#!/usr/bin/env python3
"""Self-check for ./ri, the repo's command-line front door.

Covers the only non-trivial thing it does: turn arguments into an environment
override plus a list of commands, without losing the
flag > environment > defaults.toml precedence. Also renders every --help, which
is where a malformed parser shows up.

Run it directly - `uv run --no-project scripts/test_cli.py` - or via CI.
Needs no Docker and no project dependencies.
"""

import argparse
import importlib.machinery
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_PATH = REPO_ROOT / "ri"

spec = importlib.util.spec_from_loader(
    "ri", importlib.machinery.SourceFileLoader("ri", str(CLI_PATH))
)
ri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ri)

failures = []


def check(what, expected, actual):
    if actual == expected:
        print(f"ok   {what}")
    else:
        print(f"FAIL {what}: expected {expected!r}, got {actual!r}", file=sys.stderr)
        failures.append(what)


def plan(*argv):
    """The (env overrides, commands) ./ri would use for these arguments."""
    args = ri.build_parser().parse_args(argv)
    if args.handler is ri.do_record and args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args.handler(args)


# Flags set the environment variable the scripts already read. `--metric=-snr`
# is the = form on purpose: a negated metric starts with a dash, which argparse
# would otherwise read as a flag.
check(
    "search flags become NS_* overrides",
    {"NS_NLIVE": "8", "NS_METRIC": "-snr", "NS_MPI_PROCS": "1"},
    plan("search", "wsclean", "--nlive", "8", "--metric=-snr", "--mpi-procs", "1")[0],
)

# --retries 0 is the flag most likely to be dropped by env_from(), because 0 is
# falsy and turning off the self-restart has to reach the run script.
check(
    "search --retries 0 reaches the run script",
    {"NS_RETRIES": "0"},
    plan("search", "wsclean", "--retries", "0")[0],
)

# Same falsy-zero hazard, same reason: 0 turns the stall watchdog off, which
# is the setting somebody reaches for when it is misfiring on their run.
check(
    "search --stall-timeout 0 reaches the run script",
    {"NS_STALL_TIMEOUT": "0"},
    plan("search", "wsclean", "--stall-timeout", "0")[0],
)

# The booleans here reach shell scripts that test them the shell way, so they
# have to arrive as 1/0 rather than Python's True/False.
check(
    "search --synchronous/--no-synchronous become 1/0",
    ({"NS_SYNCHRONOUS": "1"}, {"NS_SYNCHRONOUS": "0"}),
    (plan("search", "wsclean", "--synchronous")[0],
     plan("search", "wsclean", "--no-synchronous")[0]),
)


check(
    "search --keep-measurement-sets/--no-... become 1/0",
    ({"NS_KEEP_MEASUREMENT_SETS": "1"}, {"NS_KEEP_MEASUREMENT_SETS": "0"}),
    (plan("search", "wsclean", "--keep-measurement-sets")[0],
     plan("search", "wsclean", "--no-keep-measurement-sets")[0]),
)


# ...and a flag that was not given contributes nothing, so an environment
# variable exported by hand survives and defaults.toml still fills the rest.
check(
    "unset flags leave the environment alone",
    {},
    plan("search", "r2d2")[0],
)

check(
    "a search builds its images, then runs",
    [
        ["scripts/build.sh", "r2d2"],
        ["scripts/build.sh", "meqtrees"],
        ["scripts/build.sh", "polychord"],
        ["scripts/run-nested-sampling-r2d2.sh"],
    ],
    plan("search", "r2d2")[1],
)

check(
    "--no-build skips straight to the run",
    [["scripts/run-nested-sampling.sh"]],
    plan("search", "wsclean", "--no-build")[1],
)

# --enable-param / --disable-param are repeatable and join into the same
# comma-separated names load_parameter_space() reads from the environment.
check(
    "search --enable-param/--disable-param join into NS_*_PARAMS",
    {"NS_ENABLE_PARAMS": "source_offset_fraction", "NS_DISABLE_PARAMS": "channel_count,observation_minutes"},
    plan(
        "search", "wsclean", "--enable-param", "source_offset_fraction",
        "--disable-param", "channel_count", "--disable-param", "observation_minutes",
    )[0],
)

check(
    "params lists the parameter space",
    ({}, [["uv", "run", "scripts/list-parameter-space.py"]]),
    plan("params"),
)

check(
    "params --enable-param/--disable-param preview the same override",
    {"NS_DISABLE_PARAMS": "source_offset_fraction"},
    plan("params", "--disable-param", "source_offset_fraction")[0],
)

# `runs` and `resume` are how an interrupted run is found and continued, so
# the flags have to reach the scripts that do the finding and the continuing.
check(
    "runs passes its selectors through",
    ({}, [["uv", "run", "scripts/nested-sampling-runs.py", "--incomplete", "--json"]]),
    plan("runs", "--incomplete", "--json"),
)

check(
    "runs with no selectors lists everything",
    ({}, [["uv", "run", "scripts/nested-sampling-runs.py"]]),
    plan("runs"),
)

# `health` takes either a run or --all, never both, and the run is a bare
# positional rather than a flag - so it is the one place a mistranslation would
# silently report on the wrong run.
check(
    "health with no arguments leaves the run choice to the script",
    ({}, [["uv", "run", "scripts/nested-sampling-health.py"]]),
    plan("health"),
)

check(
    "health passes a run name and its thresholds through",
    ({}, [["uv", "run", "scripts/nested-sampling-health.py",
           "r2d2-vlaa-20260827T101500Z", "--stale-seconds", "30.0", "--json"]]),
    plan("health", "r2d2-vlaa-20260827T101500Z", "--stale-seconds", "30", "--json"),
)

check(
    "health --all asks about every run",
    ({}, [["uv", "run", "scripts/nested-sampling-health.py", "--all"]]),
    plan("health", "--all"),
)

check(
    "health --monitor passes the redraw interval through",
    ({}, [["uv", "run", "scripts/nested-sampling-health.py",
           "--monitor", "--interval", "10.0"]]),
    plan("health", "--monitor", "--interval", "10"),
)

# Both at once is a mistake, and it has to reach the script to be rejected -
# dropping one here would report on the wrong thing without saying so.
# The TUI is the one command that is not Python or Docker: it shells back into
# this script for everything it shows, and `go -C tui` is what lets its module
# live in tui/ without a go.mod at the repository root.
check(
    "tui runs the Go module in tui/",
    ({}, [["go", "-C", "tui", "run", "."]]),
    plan("tui"),
)

check(
    "health sends a run and --all together rather than picking one",
    ({}, [["uv", "run", "scripts/nested-sampling-health.py", "somerun", "--all"]]),
    plan("health", "somerun", "--all"),
)

# No flags: resuming reads the settings back out of the run directory, so the
# run name is the only thing that has to travel. The image builds a resume does
# are the resume script's own, because only run.env knows which images the run
# needs - so nothing but the run name is on this command line either.
check(
    "resume passes the run through and nothing else",
    ({}, [["scripts/resume-nested-sampling-run.sh", "r2d2-vlaa-20260827T101500Z"]]),
    plan("resume", "r2d2-vlaa-20260827T101500Z"),
)

# ...and the one way to stop it rebuilding, for a working tree that has moved
# on since the run started.
check(
    "resume --no-build reaches the script as an environment override",
    ({"NS_NO_BUILD": "1"},
     [["scripts/resume-nested-sampling-run.sh", "r2d2-vlaa-20260827T101500Z"]]),
    plan("resume", "r2d2-vlaa-20260827T101500Z", "--no-build"),
)

check(
    "report selectors become the script's variables",
    {"LAST": "1", "FORCE": "1"},
    plan("report", "--last", "1", "--force")[0],
)

check(
    "the boolean report selectors stay out when not asked for",
    {},
    plan("report")[0],
)

# `serve` defaults live in the script, not here, so an unset flag must stay out
# of the environment rather than pin the port or the bind address.
check(
    "serve overrides only what was asked for",
    ({"REPORT_PORT": "9000"}, [["scripts/serve-report.sh"]]),
    plan("serve", "--port", "9000"),
)

check(
    "serve with no flags leaves the environment alone",
    {},
    plan("serve")[0],
)

check(
    "self-check with no target runs every set",
    [["scripts/self-check.sh", "all"]],
    plan("self-check")[1],
)

check(
    "self-check takes one set at a time",
    [["scripts/self-check.sh", "simulate"]],
    plan("self-check", "simulate")[1],
)

check(
    "self-check reaches the r2d2 imaging worker's own set",
    [["scripts/self-check.sh", "r2d2-serve"]],
    plan("self-check", "r2d2-serve")[1],
)

check(
    "smoke with no target runs both imagers",
    [["scripts/smoke-test-wsclean.sh"], ["scripts/smoke-test-r2d2.sh"]],
    plan("smoke")[1],
)

check(
    "--native builds the host-optimized WSClean",
    ({"WSCLEAN_PORTABLE": "OFF"}, [["scripts/build.sh", "wsclean"]]),
    plan("build", "wsclean", "--native"),
)

check(
    "search --native keeps its own build host-optimized",
    "OFF",
    plan("search", "wsclean", "--native")[0]["WSCLEAN_PORTABLE"],
)

check(
    "search without --native leaves WSCLEAN_PORTABLE alone",
    False,
    "WSCLEAN_PORTABLE" in plan("search", "wsclean")[0],
)

check(
    "host-side analysis goes through uv",
    [["uv", "run", "scripts/profile-nested-sampling-run.py", "results/x", "--json"]],
    plan("profile", "results/x", "--json")[1],
)

check(
    "nested plot subcommands dispatch",
    [["uv", "run", "scripts/anesthetic-gui.py", "results/x"]],
    plan("plot", "gui", "results/x")[1],
)

check(
    "record passes the recorded command through verbatim",
    [[
        "scripts/record-environment.sh", "--tool", "r2d2", "--image", "img:tag",
        "--", "docker", "run", "--rm", "img:tag",
    ]],
    plan("record", "--tool", "r2d2", "--image", "img:tag",
         "--", "docker", "run", "--rm", "img:tag")[1],
)

# --dry-run is the seam this file leans on, so check it end to end too.
dry = subprocess.run(
    [sys.executable, str(CLI_PATH), "--dry-run", "search", "wsclean",
     "--no-build", "--seed", "7"],
    capture_output=True, text=True, check=True,
)
check(
    "--dry-run prints the environment with the command",
    "NS_SEED=7 scripts/run-nested-sampling.sh\n",
    dry.stdout,
)

# Every --help renders: a bad parser (a duplicate flag, a stray %-format in a
# help string) raises here rather than in front of a user. The subparser
# choices are argparse-private, but that is the only place the command tree
# lives, and reading it beats hardcoding a second copy of the command list.
def command_paths(parser, prefix=()):
    yield list(prefix)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield from command_paths(sub, (*prefix, name))


for path in command_paths(ri.build_parser()):
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), *path, "--help"],
        capture_output=True, text=True, check=False,
    )
    label = " ".join(["./ri", *path, "--help"])
    check(f"{label} renders", (0, True), (result.returncode, bool(result.stdout)))

# The name in the first column of `./ri runs` is what a reader copies, and it
# has to work in every command that takes a run. `health` and `resume` resolve
# it in the shell/Python they dispatch to; these three each own a resolver, and
# each one used to accept only a path - `./ri profile <name>` answered "no
# summary.json found at <cwd>/<name>". Checked here rather than three new
# self-checks because "every run-taking command takes the name" is a property
# of the front door, not of any one script.
def load_script(name):
    path = REPO_ROOT / "scripts" / name
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(path))
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


with tempfile.TemporaryDirectory() as runs_dir:
    # Pointed at a temporary directory rather than the real one: a fixture run
    # under results/nested-sampling/ is a run to every glob in this repo.
    named = Path(runs_dir) / "wsclean-vlaa-20260101T000000Z"
    named.mkdir()
    elsewhere = Path(runs_dir) / "elsewhere"
    elsewhere.mkdir()
    for script, resolver in (("profile-nested-sampling-run.py", "resolve_run"),
                             ("merge-nested-sampling-runs.py", "resolve_run_dir"),
                             ("anesthetic-gui.py", "resolve_target")):
        module = load_script(script)
        module.NESTED_SAMPLING_DIR = Path(runs_dir)
        # Each resolver is fed what its own argparse hands it - a str for two
        # of them, a Path for the GUI's - rather than a normalised type.
        resolve = getattr(module, resolver)
        wants_path = script == "anesthetic-gui.py"

        def resolved(raw):
            return resolve(Path(raw) if wants_path else raw)

        check(f"{script} resolves a bare run name", named, resolved(named.name))
        # A real path of that name still wins, so nothing that worked stops.
        check(f"{script} keeps a path that exists", elsewhere.resolve(),
              resolved(str(elsewhere)))
        # "alone" means not rewritten into NESTED_SAMPLING_DIR; each resolver
        # still canonicalises it, and on macOS /tmp is itself a symlink to
        # /private/tmp, so the expectation has to be resolved too.
        unknown = "/tmp/ri-no-such-run"
        check(f"{script} leaves an unknown path alone", Path(unknown).resolve(),
              resolved(unknown))

if failures:
    print(f"{len(failures)} check(s) failed", file=sys.stderr)
    sys.exit(1)
print("all checks passed")
