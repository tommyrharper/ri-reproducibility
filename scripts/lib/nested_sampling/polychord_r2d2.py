#!/usr/bin/env python3
"""Cheap PolyChord search over R2D2 failure-prone point-source runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_IMAGE_DIM,
    DEFAULT_SUPER_RESOLUTION,
    FAILURE_OBJECTIVE,
    WORKER_DIED,
    WorkerDied,
    abort_run,
    adopt_completed_evaluations,
    cube_like_from_theta,
    cube_to_params,
    gathered_window_fit_stats,
    compute_image_metrics,
    image_pixel_size_arcsec,
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
    self_check_parameter_toggle,
    self_check_profiling,
    self_check_resume_adoption,
    self_check_r2d2_thread_env,
    self_check_source_offset,
    self_check_spectral_window,
    self_check_worker_pool_connect,
    self_check_worker_timeout,
    simulate_measurement_set,
    simulate_worker,
    stable_seed,
    summarize_profiling,
    window_fit_summary_line,
    write_evaluation_record,
    write_polychord_paramnames,
)

DEFAULT_R2D2_NUM_ITER = 25
DEFAULT_R2D2_NUM_CHANS = 64
DEFAULT_R2D2_ARCHITECTURE = "unet"
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
        f"super_resolution: {DEFAULT_SUPER_RESOLUTION}",
        "meas_op_on_gpu: False",
        "meas_dtype: double",
        f"im_dim_x: {DEFAULT_IMAGE_DIM}",
        f"im_dim_y: {DEFAULT_IMAGE_DIM}",
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

    mat_path = eval_dir / "r2d2_data.mat"
    # ms_to_r2d2_mat.py argv for the simulate worker that just wrote this MS (see
    # convert_ms_to_mat): its own `docker exec` cost ~0.15s, of which only ~0.01s
    # was the conversion and the rest a fresh interpreter and its imports.
    convert_cmd = ["--ms-path", str(ms_path), "--mat-path", str(mat_path)]
    convert_start = time.perf_counter()
    convert_returncode = convert_ms_to_mat(convert_cmd, eval_dir, args.meqtrees_image, args.platform)
    convert_seconds = time.perf_counter() - convert_start
    if convert_returncode == WORKER_DIED:
        raise WorkerDied(f"simulate worker died converting evaluation {eval_id} ({eval_dir})")
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
    # The OOM killer's usual victim: retried against a fresh worker already,
    # so reaching here means the host cannot run this evaluation at all.
    if run_result.returncode == WORKER_DIED:
        raise WorkerDied(f"r2d2 imaging worker died on evaluation {eval_id} ({eval_dir})")
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

    # R2D2 sizes its own pixels from the data and writes no WCS, so the metrics
    # have to be told the cell size it used. Same figure WSClean is handed as
    # `-scale`, from the same recorded baseline, so the two score the same sky
    # - see image_pixel_size_arcsec() in common.py.
    simulation = json.loads((eval_dir / "simulation.json").read_text())
    if "max_proj_baseline_lambda" not in simulation["observation"]:
        raise SystemExit(
            "FATAL: simulation.json has no observation.max_proj_baseline_lambda - "
            "rebuild the meqtrees image (scripts/build.sh meqtrees), it bakes in a stale simulator"
        )
    scale_arcsec = image_pixel_size_arcsec(simulation["observation"]["max_proj_baseline_lambda"])

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
            source_l_arcsec=params["source_l_arcsec"],
            source_m_arcsec=params["source_m_arcsec"],
            pixel_size_arcsec=scale_arcsec,
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
        "image_pixel_size_arcsec": scale_arcsec,
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


def self_check_worker_death_is_not_scored() -> None:
    """A dead worker must stop the run, not become the search's best point.

    The two cases have to stay apart: R2D2 exiting non-zero on its parameters
    is a failure mode and is scored, while a worker the OOM killer took says
    nothing about the algorithm and must never reach the sampler.
    """
    import tempfile

    original_convert = globals()["convert_ms_to_mat"]
    original_run_r2d2 = globals()["run_r2d2_imaging"]
    original_simulate = globals()["simulate_measurement_set"]

    def successful_simulate(
        params: dict[str, Any],
        eval_dir: Path,
        meqtrees_image: str,
        platform: str,
    ) -> tuple[Path, list[str], None]:
        eval_dir.mkdir(parents=True, exist_ok=False)
        return eval_dir / "sim.ms", ["simulate"], None

    def imaging(returncode: int) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> argparse.Namespace:
            return argparse.Namespace(returncode=returncode, wall_seconds=0.1, peak_memory_bytes=0)

        return run

    args = argparse.Namespace(
        meqtrees_image="meqtrees",
        r2d2_image="r2d2",
        checkpoints_dir="/checkpoints",
        platform="linux/arm64",
    )
    try:
        globals()["simulate_measurement_set"] = successful_simulate
        globals()["convert_ms_to_mat"] = lambda *a, **k: 0

        globals()["run_r2d2_imaging"] = imaging(WORKER_DIED)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                evaluate({}, args, root / "eval-0001-deadbeef", 1, lambda metrics: 0.0)
            except WorkerDied:
                pass
            else:
                raise AssertionError("a dead imaging worker was scored instead of stopping the run")
            # And nothing was recorded, so no later merge or plot can pick up
            # an objective for an evaluation the host never actually ran.
            assert load_evaluations_from_dir(root) == []

        # A real non-zero exit is still a failure mode, and still scored.
        globals()["run_r2d2_imaging"] = imaging(3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = evaluate({}, args, root / "eval-0001-deadbeef", 1, lambda metrics: 0.0)
            assert record["objective"] == FAILURE_OBJECTIVE
            assert "exit 3" in record["error"]

        # The same split applies to a dead simulate worker.
        globals()["run_r2d2_imaging"] = imaging(0)
        globals()["convert_ms_to_mat"] = lambda *a, **k: WORKER_DIED
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                evaluate({}, args, root / "eval-0001-deadbeef", 1, lambda metrics: 0.0)
            except WorkerDied:
                pass
            else:
                raise AssertionError("a dead simulate worker was scored instead of stopping the run")
    finally:
        globals()["simulate_measurement_set"] = original_simulate
        globals()["convert_ms_to_mat"] = original_convert
        globals()["run_r2d2_imaging"] = original_run_r2d2


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
            # A real simulate always leaves this behind, and evaluate() reads it
            # for the cell size R2D2's images carry no header for.
            (eval_dir / "simulation.json").write_text(
                json.dumps({"observation": {"max_proj_baseline_lambda": 1.0e5}})
            )
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
        params = cube_to_params(cube, track=True)
        return prior_vector(cube, params)

    def likelihood(theta: np.ndarray) -> tuple[float, list[float]]:
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
            except (Exception, SystemExit):
                # Anything else is a bug in this file, and a bug here used to
                # hang the job rather than end it. PolyChord calls the
                # likelihood from Fortran, so the traceback unwinds this rank
                # only and every other rank waits forever in a collective that
                # never completes: every core busy, nothing landing, and
                # run_with_retries never even reached because nothing exited.
                abort_run(traceback.format_exc())
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
    # PolyChord's own checkpointing, on so that an interrupted run is not a
    # wasted one. It was off, which was survivable when a run was 100s of toy
    # evaluations and is not once a run is hours long: any interruption - the
    # host running out of memory, a Ctrl-C, a reboot - threw away every
    # evaluation. Resuming is `--output-dir <the interrupted run>`, which finds
    # the resume file and continues; a fresh run has none and starts clean.
    resume_path = Path(settings.base_dir) / f"{settings.file_root}.resume"
    settings.write_resume = True
    settings.read_resume = resume_path.exists()
    # Adopt what an earlier attempt already evaluated, so eval ids carry on
    # rather than restarting at 1 and colliding with its directories, and so a
    # repeated point is served from the cache instead of being recomputed.
    #
    # Not conditional on the resume file, because the two conditions are not
    # the same one: PolyChord writes `.resume` at its first checkpoint, so an
    # attempt killed before that - up to seventy minutes of R2D2 imaging, and
    # every kill on a short run - leaves evaluations on disk and no resume
    # file. Adopting only when resuming meant that restart began at eval id 1
    # on top of the previous attempt's directories, which
    # simulate_measurement_set creates with `exist_ok=False`: one rank died on
    # FileExistsError and the rest hung forever in a collective that never
    # completed. Sampling from scratch is cheap here anyway - PolyChord redraws
    # the same points from the same seed and the cache answers without imaging.
    done = adopt_completed_evaluations(evaluations_dir, evaluations, cache)
    if done:
        where = (f"resuming from {resume_path}" if settings.read_resume
                 else "no checkpoint to resume from, re-sampling from the cache")
        print(f"{where}, {done} evaluations already done", flush=True)
    settings.feedback = 1

    write_polychord_paramnames(output_dir / "chains", settings.file_root)
    warm()
    run_start = time.monotonic()
    pypolychord.run_polychord(likelihood, len(load_parameter_space()), 0, settings, prior)
    total_wall_seconds = time.monotonic() - run_start
    # Collective, so every rank calls it before rank 0 goes on alone.
    window_fit_stats = gathered_window_fit_stats()

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
                "im_dim_x": DEFAULT_IMAGE_DIM,
                "im_dim_y": DEFAULT_IMAGE_DIM,
                "num_iter": DEFAULT_R2D2_NUM_ITER,
                "num_chans": DEFAULT_R2D2_NUM_CHANS,
                "architecture": DEFAULT_R2D2_ARCHITECTURE,
                "super_resolution": DEFAULT_SUPER_RESOLUTION,
                "ckpt_path": "/checkpoints/R2D2_A1",
                "ckpt_realisations": DEFAULT_R2D2_CKPT_REALISATIONS,
            },
            "parameter_space": load_parameter_space(),
            "evaluations": all_evaluations,
            "worst_evaluation": best,
            "total_wall_seconds": total_wall_seconds,
            "profiling": summarize_profiling(all_evaluations, total_wall_seconds, mpi_procs),
            "spectral_window_fitting": window_fit_stats,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(window_fit_summary_line(window_fit_stats))
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    if os.environ.get("POLYCHORD_R2D2_SELF_CHECK") == "1":
        self_check_metric_resolution()
        self_check_parameter_space()
        self_check_parameter_toggle()
        self_check_spectral_window()
        self_check_lazy_numpy()
        self_check_r2d2_thread_env()
        self_check_r2d2_config_thread_cap()
        self_check_source_offset()
        self_check_profiling()
        self_check_failure_record_persistence()
        self_check_worker_death_is_not_scored()
        self_check_resume_adoption()
        self_check_worker_timeout()
        self_check_worker_pool_connect()
        print("metric resolution self-check passed")
    else:
        main()
