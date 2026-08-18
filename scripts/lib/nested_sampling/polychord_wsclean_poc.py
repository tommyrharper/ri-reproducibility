#!/usr/bin/env python3
"""Cheap PolyChord search over WSClean failure-prone point-source runs."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pypolychord
from pypolychord.settings import PolyChordSettings

from poc_common import (
    DEFAULT_WSCLEAN_AUTO_THRESHOLD,
    DEFAULT_WSCLEAN_NITER,
    FAILURE_OBJECTIVE,
    PARAMETER_SPACE,
    cube_like_from_theta,
    cube_to_params,
    compute_image_metrics,
    load_evaluations_from_dir,
    mpi_rank,
    params_key,
    prior_vector,
    read_gnu_time_peak_memory,
    resolve_metric,
    run_docker_monitored,
    self_check_metric_resolution,
    simulate_measurement_set,
    stable_seed,
    write_polychord_paramnames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=os.environ.get("REPO_ROOT", os.getcwd()))
    parser.add_argument("--meqtrees-image", default=os.environ.get("MEQTREES_IMAGE", "ri-reproducibility/meqtrees:kern-10"))
    parser.add_argument("--wsclean-image", default=os.environ.get("WSCLEAN_IMAGE", "ri-reproducibility/wsclean:v3.7"))
    parser.add_argument("--nlive", type=int, default=8)
    parser.add_argument("--num-repeats", type=int, default=2)
    parser.add_argument("--max-ndead", type=int, default=12)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--metric", default="off_source_rms_jy", help="Objective metric: badness, a raw metric name, or an expression over metric names")
    parser.add_argument("--platform", default=os.environ.get("DOCKER_DEFAULT_PLATFORM", "linux/arm64"))
    return parser.parse_args()


def evaluate(
    params: dict[str, Any],
    args: argparse.Namespace,
    eval_dir: Path,
    eval_id: int,
    objective_from_metrics: Callable[[dict[str, float]], float],
) -> dict[str, Any]:
    ms_path, sim_cmd, sim_error = simulate_measurement_set(params, eval_dir, args.meqtrees_image, args.platform)
    if sim_error is not None:
        return {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"simulation failed with exit {sim_error.returncode}",
            "paths": {"eval_dir": str(eval_dir)},
        }

    wsclean_dir = eval_dir / "wsclean"
    wsclean_dir.mkdir()
    container_name = f"ri-ns-wsclean-{uuid.uuid4().hex[:12]}"
    wsclean_stdout = eval_dir / "wsclean.stdout.log"
    wsclean_stderr = eval_dir / "wsclean.stderr.log"
    wsclean_time = wsclean_dir / "time.txt"
    wsclean_cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--platform",
        args.platform,
        "-v",
        f"{eval_dir}:/work",
        "--entrypoint",
        "/usr/bin/time",
        args.wsclean_image,
        "-v",
        "-o",
        "/work/wsclean/time.txt",
        "wsclean",
        "-name",
        "/work/wsclean/recon",
        "-temp-dir",
        "/work/wsclean",
        "-size",
        "128",
        "128",
        "-scale",
        "1asec",
        "-niter",
        str(DEFAULT_WSCLEAN_NITER),
        "-mgain",
        "0.8",
        "-auto-threshold",
        f"{DEFAULT_WSCLEAN_AUTO_THRESHOLD:.6f}",
        "-weight",
        "natural",
        "-pol",
        "I",
        "-j",
        "1",
        "-no-update-model-required",
        "/work/sim.ms",
    ]
    run_result = run_docker_monitored(wsclean_cmd, container_name, wsclean_stdout, wsclean_stderr)
    peak_memory_bytes = max(run_result.peak_memory_bytes, read_gnu_time_peak_memory(wsclean_time))
    if run_result.returncode != 0:
        return {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"wsclean failed with exit {run_result.returncode}",
            "paths": {"eval_dir": str(eval_dir), "measurement_set": str(ms_path)},
            "wall_seconds": run_result.wall_seconds,
            "peak_memory_bytes": peak_memory_bytes,
        }

    image_path = wsclean_dir / "recon-image.fits"
    dirty_path = wsclean_dir / "recon-dirty.fits"
    residual_dirty_path = wsclean_dir / "recon-residual.fits"
    try:
        metrics = compute_image_metrics(
            image_path,
            params["source_flux_jy"],
            run_result.wall_seconds,
            peak_memory_bytes,
            dirty_path=dirty_path,
            residual_dirty_path=residual_dirty_path,
        )
        objective = objective_from_metrics(metrics)
    except Exception as exc:
        return {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"metric computation failed: {exc}",
            "paths": {"eval_dir": str(eval_dir), "measurement_set": str(ms_path), "image": str(image_path)},
        }

    record = {
        "eval_id": eval_id,
        "params": params,
        "metrics": metrics,
        "objective": objective,
        "paths": {
            "eval_dir": str(eval_dir),
            "measurement_set": str(ms_path),
            "simulation_metadata": str(eval_dir / "simulation.json"),
            "image": str(image_path),
            "dirty": str(dirty_path),
            "residual": str(residual_dirty_path),
            "time": str(wsclean_time),
        },
        "commands": {
            "simulate": sim_cmd,
            "wsclean": wsclean_cmd,
        },
    }
    (eval_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    args = parse_args()
    objective_from_metrics, likelihood_framing = resolve_metric(args.metric)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir = output_dir / "evaluations"
    evaluations_dir.mkdir(exist_ok=True)
    (output_dir / "parameter-space.json").write_text(json.dumps(PARAMETER_SPACE, indent=2) + "\n")

    cache: dict[str, dict[str, Any]] = {}
    evaluations: list[dict[str, Any]] = []

    def prior(cube: np.ndarray) -> np.ndarray:
        params = cube_to_params(cube)
        return prior_vector(cube, params)

    def likelihood(theta: np.ndarray) -> tuple[float, list[float]]:
        # ponytail: theta values are rounded back to the documented parameter
        # space here; a later science run should keep integer/discrete handling
        # in one sampler-aware transform instead of this PoC bridge.
        params = cube_to_params(cube_like_from_theta(theta))
        key = params_key(params)
        params["noise_seed"] = stable_seed(args.seed, key)
        key = params_key(params)
        if key not in cache:
            eval_id = len(evaluations) + 1
            eval_dir = evaluations_dir / f"eval-{eval_id:04d}-{key}"
            record = evaluate(params, args, eval_dir, eval_id, objective_from_metrics)
            cache[key] = record
            evaluations.append(record)
            print(json.dumps({"eval_id": eval_id, "objective": record["objective"], "params": params}), flush=True)
        return float(cache[key]["objective"]), []

    settings = PolyChordSettings(len(PARAMETER_SPACE), 0)
    settings.base_dir = str(output_dir / "chains")
    settings.file_root = "wsclean_vlaa_poc"
    settings.nlive = args.nlive
    settings.num_repeats = args.num_repeats
    settings.max_ndead = args.max_ndead
    settings.seed = args.seed
    settings.read_resume = False
    settings.write_resume = False
    settings.feedback = 1

    write_polychord_paramnames(output_dir / "chains", settings.file_root)
    run_start = time.monotonic()
    pypolychord.run_polychord(likelihood, len(PARAMETER_SPACE), 0, settings, prior)
    total_wall_seconds = time.monotonic() - run_start

    if mpi_rank() == 0:
        all_evaluations = load_evaluations_from_dir(evaluations_dir)
        best = max(all_evaluations, key=lambda item: item["objective"]) if all_evaluations else None
        mpi_procs = int(os.environ.get("NS_MPI_PROCS", "1"))
        summary = {
            "algorithm": "wsclean",
            "vla_config": "VLA.A",
            "run_type": "cheap infrastructure PoC",
            "metric": args.metric,
            "likelihood_framing": likelihood_framing,
            "polychord": {
                "nlive": args.nlive,
                "num_repeats": args.num_repeats,
                "max_ndead": args.max_ndead,
                "seed": args.seed,
                "mpi_procs": mpi_procs,
            },
            "wsclean_fixed_hyperparameters": {
                "niter": DEFAULT_WSCLEAN_NITER,
                "auto_threshold": DEFAULT_WSCLEAN_AUTO_THRESHOLD,
            },
            "parameter_space": PARAMETER_SPACE,
            "evaluations": all_evaluations,
            "worst_evaluation": best,
            "total_wall_seconds": total_wall_seconds,
        }
        summary_path = output_dir / "poc-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    if os.environ.get("POLYCHORD_WSCLEAN_POC_SELF_CHECK") == "1":
        self_check_metric_resolution()
        print("metric resolution self-check passed")
    else:
        main()
