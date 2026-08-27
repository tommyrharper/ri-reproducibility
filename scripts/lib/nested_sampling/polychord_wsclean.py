#!/usr/bin/env python3
"""Cheap PolyChord search over WSClean failure-prone point-source runs."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_WSCLEAN_AUTO_THRESHOLD,
    DEFAULT_WSCLEAN_NITER,
    FAILURE_OBJECTIVE,
    WORKER_DIED,
    WorkerDied,
    abort_run,
    adopt_completed_evaluations,
    cube_like_from_theta,
    cube_to_params,
    compute_image_metrics,
    load_evaluations_from_dir,
    load_parameter_space,
    mpi_rank,
    params_key,
    prewarm,
    prior_vector,
    read_gnu_time_peak_memory,
    read_gnu_time_wall_seconds,
    resolve_metric,
    self_check_fits_reader,
    self_check_lazy_numpy,
    self_check_metric_resolution,
    self_check_parameter_space,
    self_check_profiling,
    self_check_resume_adoption,
    self_check_spectral_window,
    sidecar_command,
    sidecar_run,
    sidecar_shell,
    simulate_measurement_set,
    simulate_worker,
    stable_seed,
    summarize_profiling,
    write_evaluation_record,
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
    parser.add_argument("--metric", default="total_rms_jy", help="Objective metric: badness, a raw metric name, or an expression over metric names")
    parser.add_argument("--platform", default=os.environ.get("DOCKER_DEFAULT_PLATFORM", "linux/arm64"))
    return parser.parse_args()


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
    # A dead worker is the host failing, not the algorithm, so it is never
    # scored - see WORKER_DIED in common.py.
    if sim_error is not None and sim_error.returncode == WORKER_DIED:
        raise WorkerDied(f"simulate worker died on evaluation {eval_id} ({eval_dir})")
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

    wsclean_dir = eval_dir / "wsclean"
    wsclean_dir.mkdir()
    wsclean_stdout = eval_dir / "wsclean.stdout.log"
    wsclean_stderr = eval_dir / "wsclean.stderr.log"
    wsclean_time = wsclean_dir / "time.txt"
    wsclean_cmd = [
        *sidecar_command(args.wsclean_image, prefix=["/usr/bin/time", "-v", "-o", str(wsclean_time)]),
        "-name",
        str(wsclean_dir / "recon"),
        "-temp-dir",
        str(wsclean_dir),
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
        str(ms_path),
    ]
    # No `docker stats` polling loop here: GNU `time -v` inside the container
    # reports an exact peak RSS, where the 0.2s-interval stats sampler both
    # missed short peaks and delayed noticing the process had exited.
    run_result = sidecar_run(args.wsclean_image, args.platform, eval_dir, wsclean_cmd, wsclean_stdout, wsclean_stderr)
    peak_memory_bytes = read_gnu_time_peak_memory(wsclean_time)
    image_binary_seconds = read_gnu_time_wall_seconds(wsclean_time)
    if run_result.returncode == WORKER_DIED:
        raise WorkerDied(f"wsclean sidecar shell died on evaluation {eval_id} ({eval_dir})")
    if run_result.returncode != 0:
        return write_evaluation_record(eval_dir, {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"wsclean failed with exit {run_result.returncode}",
            "paths": {"eval_dir": str(eval_dir), "measurement_set": str(ms_path)},
            "wall_seconds": run_result.wall_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "timing": {
                "simulate_seconds": simulate_seconds,
                "image_container_seconds": run_result.wall_seconds,
                "image_binary_seconds": image_binary_seconds,
            },
        })

    image_path = wsclean_dir / "recon-image.fits"
    dirty_path = wsclean_dir / "recon-dirty.fits"
    residual_dirty_path = wsclean_dir / "recon-residual.fits"
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
            "paths": {"eval_dir": str(eval_dir), "measurement_set": str(ms_path), "image": str(image_path)},
            "timing": {
                "simulate_seconds": simulate_seconds,
                "image_container_seconds": run_result.wall_seconds,
                "image_binary_seconds": image_binary_seconds,
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
            "image": str(image_path),
            "dirty": str(dirty_path),
            "residual": str(residual_dirty_path),
            "time": str(wsclean_time),
        },
        "commands": {
            "simulate": sim_cmd,
            "wsclean": wsclean_cmd,
        },
        "timing": {
            "simulate_seconds": simulate_seconds,
            "image_container_seconds": run_result.wall_seconds,
            "image_binary_seconds": image_binary_seconds,
            "metrics_seconds": metrics_seconds,
        },
    }
    return write_evaluation_record(eval_dir, record)


def self_check_failure_record_persistence() -> None:
    import subprocess
    import tempfile

    original_compute_metrics = globals()["compute_image_metrics"]
    original_sidecar_run = globals()["sidecar_run"]
    original_simulate = globals()["simulate_measurement_set"]
    original_sidecar_command = globals()["sidecar_command"]

    def failing_simulate(
        params: dict[str, Any],
        eval_dir: Path,
        meqtrees_image: str,
        platform: str,
    ) -> tuple[Path, list[str], subprocess.CalledProcessError]:
        eval_dir.mkdir(parents=True, exist_ok=False)
        return eval_dir / "sim.ms", ["simulate"], subprocess.CalledProcessError(7, ["simulate"])

    try:
        globals()["sidecar_command"] = lambda image, prefix=None: ["stub-wsclean"]
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

        def successful_wsclean(
            image: str,
            platform: str,
            workdir: Path,
            cmd: list[str],
            stdout_path: Path,
            stderr_path: Path,
        ) -> argparse.Namespace:
            return argparse.Namespace(returncode=0, wall_seconds=2.0, peak_memory_bytes=4096)

        def failing_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
            raise ValueError("bad fits")

        globals()["simulate_measurement_set"] = successful_simulate
        globals()["sidecar_run"] = successful_wsclean
        globals()["compute_image_metrics"] = failing_metrics
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params = {"source_flux_jy": 1.0}
            args = argparse.Namespace(meqtrees_image="meqtrees", wsclean_image="wsclean", platform="linux/arm64")
            record = evaluate(params, args, root / "eval-0002-deadbeef", 2, lambda metrics: 0.0)
            loaded = load_evaluations_from_dir(root)
            assert loaded == [record]
            assert loaded[0]["objective"] == FAILURE_OBJECTIVE
            assert loaded[0]["timing"]["metrics_seconds"] >= 0.0
    finally:
        globals()["compute_image_metrics"] = original_compute_metrics
        globals()["sidecar_run"] = original_sidecar_run
        globals()["simulate_measurement_set"] = original_simulate
        globals()["sidecar_command"] = original_sidecar_command


def main() -> None:
    args = parse_args()

    def warm_wsclean() -> None:
        sidecar_command(args.wsclean_image)
        sidecar_shell(args.wsclean_image, args.platform)

    # Before `import pypolychord`, so the rank's sidecar attachments come up
    # while the sampler is still loading. Joined just below, right before the
    # first evaluation can ask for one.
    warm = prewarm(
        lambda: simulate_worker(args.meqtrees_image, args.platform),
        warm_wsclean,
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
        # ponytail: theta values are rounded back to the documented parameter
        # space here; a later science run should keep integer/discrete handling
        # in one sampler-aware transform instead of this bridge.
        params = cube_to_params(cube_like_from_theta(theta))
        key = params_key(params)
        params["noise_seed"] = stable_seed(args.seed, key)
        key = params_key(params)
        if key not in cache:
            eval_id = len(evaluations) + 1
            eval_dir = evaluations_dir / f"eval-{eval_id:04d}-{key}"
            try:
                record = evaluate(params, args, eval_dir, eval_id, objective_from_metrics)
            except WorkerDied as exc:
                # No honest likelihood exists for an evaluation the host never
                # ran, and any value invented here would steer the sampler.
                abort_run(str(exc))
            cache[key] = record
            evaluations.append(record)
            print(json.dumps({"eval_id": eval_id, "objective": record["objective"], "params": params}), flush=True)
        return float(cache[key]["objective"]), []

    settings = PolyChordSettings(len(load_parameter_space()), 0)
    settings.base_dir = str(output_dir / "chains")
    settings.file_root = "wsclean_vlaa"
    settings.nlive = args.nlive
    settings.num_repeats = args.num_repeats
    settings.max_ndead = args.max_ndead
    settings.seed = args.seed
    # PolyChord's own checkpointing, on so that an interrupted run is not a
    # wasted one. It was off, which was survivable when a run was 100s of toy
    # evaluations and is not once a run is hours long: any interruption - the
    # host running out of memory, a Ctrl-C, a reboot - threw away every
    # evaluation. Resuming is `--output-dir <the interrupted run>`, which finds
    # the resume file and continues; a fresh run has none and starts clean.
    resume_path = Path(settings.base_dir) / f"{settings.file_root}.resume"
    settings.write_resume = True
    settings.read_resume = resume_path.exists()
    if settings.read_resume:
        # Adopt what the interrupted attempt already evaluated, so eval ids
        # carry on rather than restarting at 1 and colliding with its
        # directories, and so a repeated point is served from the cache
        # instead of being recomputed.
        done = adopt_completed_evaluations(evaluations_dir, evaluations, cache)
        print(f"resuming from {resume_path}, {done} evaluations already done", flush=True)
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
            "algorithm": "wsclean",
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
            "wsclean_fixed_hyperparameters": {
                "niter": DEFAULT_WSCLEAN_NITER,
                "auto_threshold": DEFAULT_WSCLEAN_AUTO_THRESHOLD,
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
    if os.environ.get("POLYCHORD_WSCLEAN_SELF_CHECK") == "1":
        self_check_metric_resolution()
        self_check_parameter_space()
        self_check_spectral_window()
        self_check_lazy_numpy()
        self_check_fits_reader()
        self_check_profiling()
        self_check_failure_record_persistence()
        self_check_resume_adoption()
        print("metric resolution self-check passed")
    else:
        main()
