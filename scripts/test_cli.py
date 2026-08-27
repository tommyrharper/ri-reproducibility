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

# No flags: resuming reads the settings back out of the run directory, so the
# run name is the only thing that has to travel.
check(
    "resume passes the run through and nothing else",
    ({}, [["scripts/resume-nested-sampling-run.sh", "r2d2-vlaa-20260827T101500Z"]]),
    plan("resume", "r2d2-vlaa-20260827T101500Z"),
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

if failures:
    print(f"{len(failures)} check(s) failed", file=sys.stderr)
    sys.exit(1)
print("all checks passed")
