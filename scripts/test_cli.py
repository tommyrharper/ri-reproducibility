#!/usr/bin/env python3
"""Self-check ./ri argument dispatch, environment precedence, and help output."""

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
    args = ri.build_parser().parse_args(argv)
    if args.handler is ri.do_record and args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args.handler(args)


check(
    "search flags become NS_* overrides",
    {"NS_NLIVE": "8", "NS_METRIC": "-snr", "NS_MPI_PROCS": "1"},
    plan("search", "wsclean", "--nlive", "8", "--metric=-snr", "--mpi-procs", "1")[0],
)

check(
    "search --mgain reaches the run script",
    {"NS_WSCLEAN_MGAIN": "0.9"},
    plan("search", "wsclean", "--mgain", "0.9")[0],
)

check(
    "search --retries 0 reaches the run script",
    {"NS_RETRIES": "0"},
    plan("search", "wsclean", "--retries", "0")[0],
)

check(
    "search --stall-timeout 0 reaches the run script",
    {"NS_STALL_TIMEOUT": "0"},
    plan("search", "wsclean", "--stall-timeout", "0")[0],
)

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

check(
    "resume passes the run through and nothing else",
    ({}, [["scripts/resume-nested-sampling-run.sh", "r2d2-vlaa-20260827T101500Z"]]),
    plan("resume", "r2d2-vlaa-20260827T101500Z"),
)

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
    "self-check reaches the wsclean fork server's own set",
    [["scripts/self-check.sh", "zygote"]],
    plan("self-check", "zygote")[1],
)

check(
    "smoke with no target runs both imagers",
    [["scripts/smoke-test-wsclean.sh"], ["scripts/smoke-test-r2d2.sh"]],
    plan("smoke")[1],
)

check(
    "--native builds the host-optimized WSClean",
    ({"WSCLEAN_TARGET_CPU": "native"}, [["scripts/build.sh", "wsclean"]]),
    plan("build", "wsclean", "--native"),
)

check(
    "search --native keeps its own build host-optimized",
    "native",
    plan("search", "wsclean", "--native")[0]["WSCLEAN_TARGET_CPU"],
)

check(
    "search without --native leaves WSCLEAN_TARGET_CPU alone",
    False,
    "WSCLEAN_TARGET_CPU" in plan("search", "wsclean")[0],
)

check(
    "host-side analysis goes through uv",
    [["uv", "run", "scripts/profile-nested-sampling-run.py", "results/x", "--json"]],
    plan("profile", "results/x", "--json")[1],
)

check(
    "profile --phases reaches the profiler",
    [["uv", "run", "scripts/profile-nested-sampling-run.py", "results/x", "--phases"]],
    plan("profile", "results/x", "--phases")[1],
)

check(
    "profile --over-time reaches the profiler",
    [["uv", "run", "scripts/profile-nested-sampling-run.py", "results/x", "--over-time"]],
    plan("profile", "results/x", "--over-time")[1],
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

def load_script(name):
    path = REPO_ROOT / "scripts" / name
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(path))
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


with tempfile.TemporaryDirectory() as runs_dir:
    named = Path(runs_dir) / "wsclean-vlaa-20260101T000000Z"
    named.mkdir()
    elsewhere = Path(runs_dir) / "elsewhere"
    elsewhere.mkdir()
    for script, resolver in (("profile-nested-sampling-run.py", "resolve_run"),
                             ("merge-nested-sampling-runs.py", "resolve_run_dir"),
                             ("anesthetic-gui.py", "resolve_target")):
        module = load_script(script)
        module.NESTED_SAMPLING_DIR = Path(runs_dir)
        resolve = getattr(module, resolver)
        wants_path = script == "anesthetic-gui.py"

        def resolved(raw):
            return resolve(Path(raw) if wants_path else raw)

        check(f"{script} resolves a bare run name", named, resolved(named.name))
        check(f"{script} keeps a path that exists", elsewhere.resolve(),
              resolved(str(elsewhere)))
        unknown = "/tmp/ri-no-such-run"
        check(f"{script} leaves an unknown path alone", Path(unknown).resolve(),
              resolved(unknown))

if failures:
    print(f"{len(failures)} check(s) failed", file=sys.stderr)
    sys.exit(1)
print("all checks passed")
