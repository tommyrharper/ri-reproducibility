#!/usr/bin/env python3
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

HOST_RUNNABLE = {
    "common": (
        "self_check_backfilled_intervals",
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
        "self_check_checkpoint_cache",
        "self_check_lanczos_largest_eigenvalue",
        "self_check_lazy_utils",
        "self_check_nufft_plan_reuse",
        "self_check_serve_fifo",
        "self_check_serve_pool",
        "self_check_serve_reply_stream",
    ),
}

IMAGE_ONLY = {
    "common": (
        "self_check_fits_reader",
        "self_check_image_pixel_size",
        "self_check_lazy_numpy",
        "self_check_metric_resolution",
        "self_check_source_offset",
        "self_check_spectral_window",
    ),
    "simulate_point_source_ms": (
        "self_check_forest_reuse",
        "self_check_meqserver_restart",
        "self_check_dropped_subtables",
        "self_check_noise_weighting",
        "self_check_phase_centre_predict",
        "self_check_predict_timeout_recovery",
        "self_check_scratch_root",
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
    tree = ast.parse((NESTED_SAMPLING / f"{module}.py").read_text())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("self_check_")}


def assert_every_check_is_classified() -> None:
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
