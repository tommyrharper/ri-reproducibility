#!/usr/bin/env python3
"""Run every nested-sampling self-check that needs no Docker image, in CI.

Most of these checks were written, wired into their module's own `--self-check`
entry point, and then run by nothing. `./ri self-check` (scripts/self-check.sh)
fixed that for the ones that genuinely need an image. This file is the other
half: the checks whose code is plain stdlib Python, which a CI runner can
execute directly on both Linux and macOS with no Docker at all.

The bug that motivated it: #62 gave `_connect_shell_started_worker` a second
argument and updated only one of its two call sites, so every R2D2 run died on
its first evaluation while WSClean - which never takes that path - kept
passing. `self_check_worker_pool_connect` catches it in milliseconds and needs
nothing but a temp directory, yet it lived in a suite that ran nowhere
automatic. A check that runs nowhere is the same as a check nobody wrote.

`assert_every_check_is_classified` is what stops that happening again: every
`def self_check_*` in the package has to be named below as either host-runnable
or image-only, so a new one cannot be added without someone deciding where it
runs. The module names are parsed out of the source with `ast` rather than
imported, because three of the five modules import numpy at module scope and so
cannot be imported here at all.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NESTED_SAMPLING = REPO_ROOT / "scripts" / "lib" / "nested_sampling"
sys.path.insert(0, str(NESTED_SAMPLING))
os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))

# Checks this file runs. Stdlib only: no numpy, no astropy, no casacore, no
# meqserver, no Docker. Two of the r2d2_serve entries (lanczos, nufft) skip
# themselves and say so when numpy is absent, which is what they do here - they
# are listed because they cost nothing and become real coverage in the image.
HOST_RUNNABLE = {
    "common": (
        "self_check_evaluation_pruning",
        "self_check_parameter_space",
        "self_check_parameter_toggle",
        "self_check_profiling",
        "self_check_r2d2_thread_env",
        "self_check_resume_adoption",
        "self_check_worker_pool_connect",
        "self_check_worker_timeout",
    ),
    "r2d2_serve": (
        "self_check_lanczos_largest_eigenvalue",
        "self_check_lazy_utils",
        "self_check_nufft_plan_reuse",
        "self_check_serve_fifo",
        "self_check_serve_pool",
        "self_check_serve_reply_stream",
    ),
}

# Checks that need one of this repo's images to mean anything - a real TDL
# compile, a live meqserver, numpy, astropy or the R2D2 measurement operator.
# `./ri self-check` runs these; CI cannot.
IMAGE_ONLY = {
    "common": (
        "self_check_fits_reader",
        "self_check_image_pixel_size",
        # Proves numpy is imported lazily and then rebound, which it can only
        # do somewhere numpy is installed. It re-runs itself in a subprocess to
        # get a clean sys.modules, so it also escapes any in-process attempt to
        # hide numpy from it - CI caught this listed as host-runnable.
        "self_check_lazy_numpy",
        "self_check_metric_resolution",
        "self_check_source_offset",
        "self_check_spectral_window",
    ),
    "r2d2_serve": (),
    # Imports numpy at module scope, so nothing here is reachable from a bare
    # interpreter even where the check body itself would be.
    "simulate_point_source_ms": (
        "self_check_forest_reuse",
        "self_check_meqserver_restart",
        "self_check_phase_centre_predict",
        "self_check_predict_timeout_recovery",
        "self_check_serve_fifo",
        "self_check_serve_reply_stream",
        "self_check_skeleton_cache",
        "self_check_skeleton_prebuild",
        "self_check_wedge_kills_worker",
    ),
    "polychord_r2d2": (
        "self_check_failure_record_persistence",
        "self_check_r2d2_config_thread_cap",
        "self_check_worker_death_is_not_scored",
    ),
    "polychord_wsclean": ("self_check_failure_record_persistence",),
}

MODULES = sorted(set(HOST_RUNNABLE) | set(IMAGE_ONLY))


def declared_self_checks(module: str) -> set[str]:
    """Every `def self_check_*` in the module, read from source, not imported.

    A `self_check_*` name that a module imports from common belongs to common
    and is counted there, so only top-level function definitions count here.
    """
    tree = ast.parse((NESTED_SAMPLING / f"{module}.py").read_text())
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("self_check_")
    }


def assert_every_check_is_classified() -> None:
    """No self-check may exist without a decision about where it runs."""
    for module in MODULES:
        declared = declared_self_checks(module)
        classified = set(HOST_RUNNABLE.get(module, ())) | set(IMAGE_ONLY.get(module, ()))
        unclassified = declared - classified
        assert not unclassified, (
            f"{module}.py defines {sorted(unclassified)}, which this file does not"
            " list. Add each name to HOST_RUNNABLE (stdlib only) or to IMAGE_ONLY"
            " (needs an image, and so to scripts/self-check.sh)."
        )
        stale = classified - declared
        assert not stale, (
            f"{module}.py no longer defines {sorted(stale)}; drop the name from"
            " HOST_RUNNABLE or IMAGE_ONLY."
        )
    print(f"every self-check across {len(MODULES)} modules is classified")


def run_host_runnable() -> int:
    """Run each host-runnable check, letting the first failure raise."""
    ran = 0
    for module in sorted(HOST_RUNNABLE):
        imported = importlib.import_module(module)
        for name in HOST_RUNNABLE[module]:
            getattr(imported, name)()
            ran += 1
    return ran


if __name__ == "__main__":
    assert_every_check_is_classified()
    count = run_host_runnable()
    image_only = sum(len(names) for names in IMAGE_ONLY.values())
    print(
        f"OK: {count} host-runnable self-checks passed; "
        f"{image_only} need an image (./ri self-check)"
    )
