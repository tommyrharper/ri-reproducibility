#!/usr/bin/env python3
"""Cheap PolyChord search over R2D2 failure-prone point-source runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    FAILURE_OBJECTIVE,
    cube_like_from_theta,
    cube_to_params,
    compute_image_metrics,
    convert_ms_to_mat,
    load_evaluations_from_dir,
    load_parameter_space,
    mpi_rank,
    params_key,
    prewarm,
    prior_vector,
    r2d2_thread_count,
    r2d2_worker,
    resolve_metric,
    run_r2d2_imaging,
    self_check_lazy_numpy,
    self_check_metric_resolution,
    self_check_parameter_space,
    self_check_profiling,
    self_check_r2d2_thread_env,
    simulate_measurement_set,
    simulate_worker,
    stable_seed,
    summarize_profiling,
    write_evaluation_record,
    write_polychord_paramnames,
)

DEFAULT_R2D2_IM_DIM = 128
DEFAULT_R2D2_NUM_ITER = 25
DEFAULT_R2D2_NUM_CHANS = 64
DEFAULT_R2D2_ARCHITECTURE = "unet"
DEFAULT_R2D2_SUPER_RESOLUTION = 1.52
DEFAULT_R2D2_CKPT_REALISATIONS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=os.environ.get("REPO_ROOT", os.getcwd()))
    parser.add_argument("--meqtrees-image", default=os.environ.get("MEQTREES_IMAGE", "ri-reproducibility/meqtrees:kern-10"))
    parser.add_argument("--r2d2-image", default=os.environ.get("R2D2_IMAGE", "ri-reproducibility/r2d2:cpu"))
    parser.add_argument("--checkpoints-dir", default=os.environ.get("CHECKPOINTS_DIR", ""))
    parser.add_argument("--nlive", type=int, default=8)
    parser.add_argument("--num-repeats", type=int, default=2)
    parser.add_argument("--max-ndead", type=int, default=12)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--metric", default="total_rms_jy", help="Objective metric: badness, a raw metric name, or an expression over metric names")
    parser.add_argument("--platform", default=os.environ.get("DOCKER_DEFAULT_PLATFORM", "linux/arm64"))
    return parser.parse_args()


def write_r2d2_config(config_path: Path, data_file: str, output_path: str) -> None:
    lines = [
        f"data_file: {data_file}",
        f"output_path: {output_path}",
        "save_all_outputs: False",
        "nufft_pkg: finufft",
        "meas_op_on_gpu: False",
        "meas_dtype: double",
        f"im_dim_x: {DEFAULT_R2D2_IM_DIM}",
        f"im_dim_y: {DEFAULT_R2D2_IM_DIM}",
        "data_weighting: True",
        "natural_weight: True",
        "weight_type: briggs",
        f"num_iter: {DEFAULT_R2D2_NUM_ITER}",
        f"num_chans: {DEFAULT_R2D2_NUM_CHANS}",
        "series: R2D2",
        "layers: 1",
        f"architecture: {DEFAULT_R2D2_ARCHITECTURE}",
        "prune: True",
        "sigma_res_tol: 1e-4",
        "ckpt_path: /checkpoints/R2D2_A1",
        f"ckpt_realisations: {DEFAULT_R2D2_CKPT_REALISATIONS}",
        # R2D2's set_common_args() calls torch.set_num_threads() itself, from
        # psutil's CPU affinity, and that overrides the OMP_NUM_THREADS the
        # worker's `docker exec` sets. Without this every rank asked torch for
        # all 20 host CPUs, so the 8 default ranks ran 160 threads on 20 cores.
        f"ncpus: {r2d2_thread_count()}",
        "",
    ]
    config_path.write_text("\n".join(lines))


def evaluate(
    params: dict[str, Any],
    args: argparse.Namespace,
    eval_dir: Path,
    eval_id: int,
    objective_from_metrics: Callable[[dict[str, float]], float],
) -> dict[str, Any]:
    sim_start = time.perf_counter()
    ms_path, sim_cmd, sim_error = simulate_measurement_set(params, eval_dir, args.meqtrees_image, args.platform)
    simulate_seconds = time.perf_counter() - sim_start
    if sim_error is not None:
        return write_evaluation_record(eval_dir, {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"simulation failed with exit {sim_error.returncode}",
            "paths": {"eval_dir": str(eval_dir)},
            "commands": {"simulate": sim_cmd},
            "timing": {"simulate_seconds": simulate_seconds},
        })

    mat_path = eval_dir / "r2d2_data.mat"
    # ms_to_r2d2_mat.py argv for the simulate worker that just wrote this MS (see
    # convert_ms_to_mat): its own `docker exec` cost ~0.15s, of which only ~0.01s
    # was the conversion and the rest a fresh interpreter and its imports.
    convert_cmd = ["--ms-path", str(ms_path), "--mat-path", str(mat_path)]
    convert_start = time.perf_counter()
    convert_returncode = convert_ms_to_mat(convert_cmd, eval_dir, args.meqtrees_image, args.platform)
    convert_seconds = time.perf_counter() - convert_start
    if convert_returncode != 0:
        return write_evaluation_record(eval_dir, {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"ms_to_r2d2_mat failed with exit {convert_returncode}",
            "paths": {"eval_dir": str(eval_dir), "measurement_set": str(ms_path)},
            "commands": {"simulate": sim_cmd, "ms_to_r2d2_mat": convert_cmd},
            "timing": {"simulate_seconds": simulate_seconds, "convert_seconds": convert_seconds},
        })

    r2d2_dir = eval_dir / "r2d2"
    r2d2_dir.mkdir()
    config_path = eval_dir / "r2d2_config.yaml"
    write_r2d2_config(config_path, str(mat_path), str(r2d2_dir))

    r2d2_stdout = eval_dir / "r2d2.stdout.log"
    r2d2_stderr = eval_dir / "r2d2.stderr.log"
    # `imager.py` argv for this rank's long-lived R2D2 worker (see
    # run_r2d2_imaging): a fresh `docker run` of this image cost ~2.4s warm, of
    # which ~1.8s was container start plus torch and R2D2 imports.
    r2d2_cmd = [
        "--config",
        str(config_path),
        "--ckpt_path",
        "/checkpoints/R2D2_A1",
    ]
    run_result = run_r2d2_imaging(
        args.r2d2_image, args.platform, args.checkpoints_dir, r2d2_cmd, r2d2_stdout, r2d2_stderr
    )
    peak_memory_bytes = run_result.peak_memory_bytes
    if run_result.returncode != 0:
        return write_evaluation_record(eval_dir, {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"r2d2 failed with exit {run_result.returncode}",
            "paths": {"eval_dir": str(eval_dir), "measurement_set": str(ms_path), "mat": str(mat_path)},
            "wall_seconds": run_result.wall_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "timing": {
                "simulate_seconds": simulate_seconds,
                "convert_seconds": convert_seconds,
                "image_container_seconds": run_result.wall_seconds,
            },
        })

    image_path = r2d2_dir / "r2d2_data" / "R2D2_model_image.fits"
    dirty_path = r2d2_dir / "r2d2_data" / "dirty_normalised.fits"
    residual_dirty_path = r2d2_dir / "r2d2_data" / "R2D2_residual_dirty_image.fits"
    metrics_start = time.perf_counter()
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
        metrics_seconds = time.perf_counter() - metrics_start
        return write_evaluation_record(eval_dir, {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"metric computation failed: {exc}",
            "paths": {
                "eval_dir": str(eval_dir),
                "measurement_set": str(ms_path),
                "mat": str(mat_path),
                "image": str(image_path),
            },
            "timing": {
                "simulate_seconds": simulate_seconds,
                "convert_seconds": convert_seconds,
                "image_container_seconds": run_result.wall_seconds,
                "metrics_seconds": metrics_seconds,
            },
        })
    metrics_seconds = time.perf_counter() - metrics_start

    record = {
        "eval_id": eval_id,
        "params": params,
        "metrics": metrics,
        "objective": objective,
        "paths": {
            "eval_dir": str(eval_dir),
            "measurement_set": str(ms_path),
            "simulation_metadata": str(eval_dir / "simulation.json"),
            "mat": str(mat_path),
            "image": str(image_path),
            "dirty": str(dirty_path),
            "residual": str(residual_dirty_path),
        },
        "commands": {
            "simulate": sim_cmd,
            "ms_to_r2d2_mat": convert_cmd,
            "r2d2": r2d2_cmd,
        },
        "timing": {
            "simulate_seconds": simulate_seconds,
            "convert_seconds": convert_seconds,
            "image_container_seconds": run_result.wall_seconds,
            "metrics_seconds": metrics_seconds,
        },
    }
    return write_evaluation_record(eval_dir, record)


def self_check_r2d2_config_thread_cap() -> None:
    """`ncpus` must reach the config, or torch takes every host CPU per rank."""
    import tempfile

    saved = os.environ.get("R2D2_OMP_THREADS")
    os.environ["R2D2_OMP_THREADS"] = "3"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "r2d2_config.yaml"
            write_r2d2_config(config, "/data.mat", tmp)
            assert "ncpus: 3" in config.read_text().splitlines()
    finally:
        if saved is None:
            del os.environ["R2D2_OMP_THREADS"]
        else:
            os.environ["R2D2_OMP_THREADS"] = saved


def self_check_failure_record_persistence() -> None:
    import tempfile

    original_compute_metrics = globals()["compute_image_metrics"]
    original_convert = globals()["convert_ms_to_mat"]
    original_run_r2d2 = globals()["run_r2d2_imaging"]
    original_simulate = globals()["simulate_measurement_set"]

    def failing_simulate(
        params: dict[str, Any],
        eval_dir: Path,
        meqtrees_image: str,
        platform: str,
    ) -> tuple[Path, list[str], subprocess.CalledProcessError]:
        eval_dir.mkdir(parents=True, exist_ok=False)
        return eval_dir / "sim.ms", ["simulate"], subprocess.CalledProcessError(7, ["simulate"])

    try:
        globals()["simulate_measurement_set"] = failing_simulate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(meqtrees_image="meqtrees", platform="linux/arm64")
            record = evaluate({}, args, root / "eval-0001-deadbeef", 1, lambda metrics: 0.0)
            loaded = load_evaluations_from_dir(root)
            assert loaded == [record]
            assert loaded[0]["objective"] == FAILURE_OBJECTIVE
            assert loaded[0]["timing"]["simulate_seconds"] >= 0.0

        def successful_simulate(
            params: dict[str, Any],
            eval_dir: Path,
            meqtrees_image: str,
            platform: str,
        ) -> tuple[Path, list[str], None]:
            eval_dir.mkdir(parents=True, exist_ok=False)
            return eval_dir / "sim.ms", ["simulate"], None

        def successful_convert(*args: Any, **kwargs: Any) -> int:
            return 0

        def successful_r2d2(*args: Any, **kwargs: Any) -> argparse.Namespace:
            return argparse.Namespace(returncode=0, wall_seconds=2.0, peak_memory_bytes=4096)

        def failing_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
            raise ValueError("bad fits")

        globals()["simulate_measurement_set"] = successful_simulate
        globals()["convert_ms_to_mat"] = successful_convert
        globals()["run_r2d2_imaging"] = successful_r2d2
        globals()["compute_image_metrics"] = failing_metrics
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params = {"source_flux_jy": 1.0}
            args = argparse.Namespace(
                meqtrees_image="meqtrees",
                r2d2_image="r2d2",
                checkpoints_dir="/checkpoints",
                platform="linux/arm64",
            )
            record = evaluate(params, args, root / "eval-0002-deadbeef", 2, lambda metrics: 0.0)
            loaded = load_evaluations_from_dir(root)
            assert loaded == [record]
            assert loaded[0]["objective"] == FAILURE_OBJECTIVE
            assert loaded[0]["timing"]["metrics_seconds"] >= 0.0
    finally:
        globals()["compute_image_metrics"] = original_compute_metrics
        globals()["convert_ms_to_mat"] = original_convert
        globals()["run_r2d2_imaging"] = original_run_r2d2
        globals()["simulate_measurement_set"] = original_simulate


def main() -> None:
    args = parse_args()
    args.repo_root = str(Path(args.repo_root).resolve())
    if not args.checkpoints_dir:
        args.checkpoints_dir = str(Path(args.repo_root) / "checkpoints")
    args.checkpoints_dir = str(Path(args.checkpoints_dir).resolve())

    # Before `import pypolychord`, so both workers do their startup - Timba plus
    # a meqserver on the simulate side, `import torch` and the R2D2 modules on
    # the imaging side - while the sampler is still loading. Joined just below,
    # right before the first evaluation can ask for one.
    warm = prewarm(
        lambda: simulate_worker(args.meqtrees_image, args.platform),
        lambda: r2d2_worker(args.r2d2_image, args.platform, args.checkpoints_dir),
    )

    import pypolychord
    from pypolychord.settings import PolyChordSettings

    objective_from_metrics, likelihood_framing = resolve_metric(args.metric)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir = output_dir / "evaluations"
    evaluations_dir.mkdir(exist_ok=True)
    (output_dir / "parameter-space.json").write_text(json.dumps(load_parameter_space(), indent=2) + "\n")

    cache: dict[str, dict[str, Any]] = {}
    evaluations: list[dict[str, Any]] = []

    def prior(cube: np.ndarray) -> np.ndarray:
        params = cube_to_params(cube)
        return prior_vector(cube, params)

    def likelihood(theta: np.ndarray) -> tuple[float, list[float]]:
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

    settings = PolyChordSettings(len(load_parameter_space()), 0)
    settings.base_dir = str(output_dir / "chains")
    settings.file_root = "r2d2_vlaa"
    settings.nlive = args.nlive
    settings.num_repeats = args.num_repeats
    settings.max_ndead = args.max_ndead
    settings.seed = args.seed
    settings.read_resume = False
    settings.write_resume = False
    settings.feedback = 1

    write_polychord_paramnames(output_dir / "chains", settings.file_root)
    warm()
    run_start = time.monotonic()
    pypolychord.run_polychord(likelihood, len(load_parameter_space()), 0, settings, prior)
    total_wall_seconds = time.monotonic() - run_start

    if mpi_rank() == 0:
        all_evaluations = load_evaluations_from_dir(evaluations_dir)
        best = max(all_evaluations, key=lambda item: item["objective"]) if all_evaluations else None
        mpi_procs = int(os.environ.get("NS_MPI_PROCS", "1"))
        summary = {
            "algorithm": "r2d2",
            "vla_config": "VLA.A",
            "run_type": "nested-sampling run",
            "metric": args.metric,
            "likelihood_framing": likelihood_framing,
            "polychord": {
                "nlive": args.nlive,
                "num_repeats": args.num_repeats,
                "max_ndead": args.max_ndead,
                "seed": args.seed,
                "mpi_procs": mpi_procs,
            },
            "r2d2_fixed_hyperparameters": {
                "im_dim_x": DEFAULT_R2D2_IM_DIM,
                "im_dim_y": DEFAULT_R2D2_IM_DIM,
                "num_iter": DEFAULT_R2D2_NUM_ITER,
                "num_chans": DEFAULT_R2D2_NUM_CHANS,
                "architecture": DEFAULT_R2D2_ARCHITECTURE,
                "super_resolution": DEFAULT_R2D2_SUPER_RESOLUTION,
                "ckpt_path": "/checkpoints/R2D2_A1",
                "ckpt_realisations": DEFAULT_R2D2_CKPT_REALISATIONS,
            },
            "parameter_space": load_parameter_space(),
            "evaluations": all_evaluations,
            "worst_evaluation": best,
            "total_wall_seconds": total_wall_seconds,
            "profiling": summarize_profiling(all_evaluations, total_wall_seconds, mpi_procs),
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    if os.environ.get("POLYCHORD_R2D2_SELF_CHECK") == "1":
        self_check_metric_resolution()
        self_check_parameter_space()
        self_check_lazy_numpy()
        self_check_r2d2_thread_env()
        self_check_r2d2_config_thread_cap()
        self_check_profiling()
        self_check_failure_record_persistence()
        print("metric resolution self-check passed")
    else:
        main()
