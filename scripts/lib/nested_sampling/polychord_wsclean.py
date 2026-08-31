#!/usr/bin/env python3
"""Cheap PolyChord search over WSClean failure-prone point-source runs."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_IMAGE_DIM,
    DEFAULT_SUPER_RESOLUTION,
    DEFAULT_WSCLEAN_AUTO_THRESHOLD,
    DEFAULT_WSCLEAN_MGAIN,
    DEFAULT_WSCLEAN_NITER,
    FAILURE_OBJECTIVE,
    WORKER_DIED,
    WorkerDied,
    ZYGOTE_COMMAND,
    abort_run,
    adopt_completed_evaluations,
    clean_convergence_from,
    cube_like_from_theta,
    cube_to_params,
    gathered_window_fit_stats,
    compute_image_metrics,
    image_pixel_size_arcsec,
    load_evaluations_from_dir,
    load_parameter_space,
    mark_evaluation_start,
    mpi_rank,
    params_key,
    prewarm,
    prune_run_artefacts,
    prior_vector,
    resolve_metric,
    self_check_fits_reader,
    self_check_image_pixel_size,
    self_check_lazy_numpy,
    self_check_metric_resolution,
    self_check_parameter_space,
    self_check_parameter_toggle,
    self_check_profiling,
    self_check_resume_adoption,
    self_check_source_offset,
    self_check_spectral_window,
    self_check_worker_pool_connect,
    self_check_worker_timeout,
    sidecar_command,
    sidecar_worker,
    zygote_run,
    simulate_measurement_set,
    simulate_worker,
    stable_seed,
    summarize_profiling,
    window_fit_summary_line,
    write_evaluation_record,
    write_json_atomic,
    write_polychord_paramnames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
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
    # Opens the wall-clock interval write_evaluation_record() closes, which is
    # what lets the profiler tell PolyChord's time from idle time.
    mark_evaluation_start()
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

    # R2D2 sizes its pixels from the data; WSClean has to be told. Reading the
    # figure the simulator recorded keeps the two imaging the same sky - see
    # image_pixel_size_arcsec() in common.py.
    simulation = json.loads((eval_dir / "simulation.json").read_text())
    if "max_proj_baseline_lambda" not in simulation["observation"]:
        raise SystemExit(
            "FATAL: simulation.json has no observation.max_proj_baseline_lambda - "
            "rebuild the meqtrees image (scripts/build.sh meqtrees), it bakes in a stale simulator"
        )
    scale_arcsec = image_pixel_size_arcsec(simulation["observation"]["max_proj_baseline_lambda"])

    wsclean_dir = eval_dir / "wsclean"
    wsclean_dir.mkdir()
    wsclean_stdout = eval_dir / "wsclean.stdout.log"
    wsclean_stderr = eval_dir / "wsclean.stderr.log"
    wsclean_cmd = [
        *sidecar_command(args.wsclean_image),
        "-name",
        str(wsclean_dir / "recon"),
        "-temp-dir",
        str(wsclean_dir),
        "-size",
        str(DEFAULT_IMAGE_DIM),
        str(DEFAULT_IMAGE_DIM),
        "-scale",
        f"{scale_arcsec:.6g}asec",
        "-niter",
        str(DEFAULT_WSCLEAN_NITER),
        "-mgain",
        f"{DEFAULT_WSCLEAN_MGAIN:g}",
        "-auto-threshold",
        f"{DEFAULT_WSCLEAN_AUTO_THRESHOLD:.6f}",
        "-weight",
        "natural",
        "-pol",
        "I",
        "-j",
        "1",
        "-no-update-model-required",
        *([] if "sigma_res" in getattr(args, "metric", "") else ["-no-dirty"]),
        # Avoids WSClean's unused CORRECTED_DATA probe; see the cost-model doc.
        "-data-column",
        "DATA",
        # Timestamps make wsclean.stdout.log usable by `./ri profile --phases`.
        *(["-log-time", str(ms_path)]
          if os.environ.get("NS_WSCLEAN_LOG_TIME", "1") != "0" else []),
    ]
    # zygote wait4() supplies exact child wall time and peak RSS.
    run_result = zygote_run(args.wsclean_image, args.platform, eval_dir, wsclean_cmd, wsclean_stdout, wsclean_stderr)
    peak_memory_bytes = run_result.peak_memory_bytes
    image_binary_seconds = run_result.binary_seconds
    if run_result.returncode == WORKER_DIED:
        raise WorkerDied(f"wsclean {ZYGOTE_COMMAND} died on evaluation {eval_id} ({eval_dir})")
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
            dirty_path=dirty_path if "sigma_res" in args.metric else None,
            residual_dirty_path=residual_dirty_path if "sigma_res" in args.metric else None,
            source_l_arcsec=params["source_l_arcsec"],
            source_m_arcsec=params["source_m_arcsec"],
        )
        objective = objective_from_metrics(metrics)
        # Why CLEAN stopped and how far it got. Read now, while the log is
        # still beside the evaluation: the retention policy keeps the log for
        # only a few hundred evaluations a run, and these three fields are the
        # part of it that has to outlive that.
        metrics.update(clean_convergence_from(wsclean_stdout))
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
        "image_pixel_size_arcsec": scale_arcsec,
        "metrics": metrics,
        "objective": objective,
        "paths": {
            "eval_dir": str(eval_dir),
            "measurement_set": str(ms_path),
            "simulation_metadata": str(eval_dir / "simulation.json"),
            "image": str(image_path),
            "dirty": str(dirty_path),
            "residual": str(residual_dirty_path),
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
    original_zygote_run = globals()["zygote_run"]
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
        seen_wsclean_argv: list[list[str]] = []
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
            # A real simulate always leaves this behind, and evaluate() reads it
            # for the WSClean cell size.
            (eval_dir / "simulation.json").write_text(
                json.dumps({"observation": {"max_proj_baseline_lambda": 1.0e5}})
            )
            return eval_dir / "sim.ms", ["simulate"], None

        def successful_wsclean(
            image: str,
            platform: str,
            workdir: Path,
            argv: list[str],
            stdout_path: Path,
            stderr_path: Path,
        ) -> argparse.Namespace:
            seen_wsclean_argv.append(argv)
            return argparse.Namespace(
                returncode=0, wall_seconds=2.0, peak_memory_bytes=4096, binary_seconds=1.5
            )

        def failing_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
            raise ValueError("bad fits")

        globals()["simulate_measurement_set"] = successful_simulate
        globals()["zygote_run"] = successful_wsclean
        globals()["compute_image_metrics"] = failing_metrics
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params = {"source_flux_jy": 1.0}
            args = argparse.Namespace(meqtrees_image="meqtrees", wsclean_image="wsclean", platform="linux/arm64")
            record = evaluate(params, args, root / "eval-0002-deadbeef", 2, lambda metrics: 0.0)
            assert "-no-dirty" in seen_wsclean_argv[0]
            loaded = load_evaluations_from_dir(root)
            assert loaded == [record]
            assert loaded[0]["objective"] == FAILURE_OBJECTIVE
            assert loaded[0]["timing"]["metrics_seconds"] >= 0.0
    finally:
        globals()["compute_image_metrics"] = original_compute_metrics
        globals()["zygote_run"] = original_zygote_run
        globals()["simulate_measurement_set"] = original_simulate
        globals()["sidecar_command"] = original_sidecar_command


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir = output_dir / "evaluations"
    evaluations_dir.mkdir(exist_ok=True)
    # Before the sidecars, because warming them is where a run is most likely
    # to die, and a run that dies with no record of its box cannot be told
    # apart afterwards from one that searched a different one. Rank 0 alone:
    # every rank ran this line, so twenty of them raced over one file.
    if mpi_rank() == 0:
        write_json_atomic(output_dir / "parameter-space.json", load_parameter_space())

    def warm_wsclean() -> None:
        sidecar_command(args.wsclean_image)
        sidecar_worker(args.wsclean_image, args.platform, [ZYGOTE_COMMAND])

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

    # key -> objective, not key -> record: nothing reads the record back (the
    # summary re-reads them all from disk below) and holding them is what made
    # a resume of a big run cost gigabytes on every rank. `scored` is the eval
    # id counter the list length used to supply.
    cache: dict[str, float] = {}
    scored = 0

    def prior(cube: np.ndarray) -> np.ndarray:
        params = cube_to_params(cube, track=True)
        return prior_vector(cube, params)

    def likelihood(theta: np.ndarray) -> tuple[float, list[float]]:
        nonlocal scored
        # ponytail: theta values are rounded back to the documented parameter
        # space here; a later science run should keep integer/discrete handling
        # in one sampler-aware transform instead of this bridge.
        params = cube_to_params(cube_like_from_theta(theta))
        key = params_key(params)
        params["noise_seed"] = stable_seed(args.seed, key)
        key = params_key(params)
        if key not in cache:
            scored += 1
            eval_id = scored
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
            cache[key] = float(record["objective"])
            print(json.dumps({"eval_id": eval_id, "objective": record["objective"], "params": params}), flush=True)
        return cache[key], []

    settings = PolyChordSettings(len(load_parameter_space()), 0)
    settings.base_dir = str(output_dir / "chains")
    settings.file_root = "wsclean_vlaa"
    settings.nlive = args.nlive
    settings.num_repeats = args.num_repeats
    settings.max_ndead = args.max_ndead
    settings.seed = args.seed
    settings.synchronous = os.environ.get("NS_SYNCHRONOUS", "0") != "0"
    # Checkpoint long runs; `--output-dir <interrupted run>` resumes, while a
    # fresh run without a resume file starts clean.
    resume_path = Path(settings.base_dir) / f"{settings.file_root}.resume"
    settings.write_resume = True
    settings.read_resume = resume_path.exists()
    # Always adopt disk evaluations: retries can die before PolyChord writes a
    # resume file, and reusing ids prevents directory collisions. A fresh
    # retry reuses the cache; a resumed sampler skips ahead, so it may not.
    done = scored = adopt_completed_evaluations(evaluations_dir, cache)
    if done:
        where = (f"resuming from {resume_path}" if settings.read_resume
                 else "no checkpoint to resume from, re-sampling from the cache")
        print(f"{where}, {done} evaluations already done", flush=True)
    settings.feedback = 1

    write_polychord_paramnames(output_dir / "chains", settings.file_root)
    warm()
    run_start, run_started_epoch = time.monotonic(), time.time()
    pypolychord.run_polychord(likelihood, len(load_parameter_space()), 0, settings, prior)
    total_wall_seconds = time.monotonic() - run_start
    # Collective, so every rank calls it before rank 0 goes on alone.
    window_fit_stats = gathered_window_fit_stats()

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
                # Which MPI scheduling this run used, because it changes how
                # much of the worker-time budget the profiling block can
                # account for and a run's numbers are unreadable without it.
                "synchronous": settings.synchronous,
            },
            "wsclean_fixed_hyperparameters": {
                "niter": DEFAULT_WSCLEAN_NITER,
                "mgain": DEFAULT_WSCLEAN_MGAIN,
                "auto_threshold": DEFAULT_WSCLEAN_AUTO_THRESHOLD,
                "image_dim": DEFAULT_IMAGE_DIM,
                # `-scale` is derived per evaluation from this and the sampled
                # sky, not fixed; each record carries its own image_pixel_size_arcsec.
                "super_resolution": DEFAULT_SUPER_RESOLUTION,
            },
            "parameter_space": load_parameter_space(),
            "evaluations": all_evaluations,
            "worst_evaluation": best,
            "total_wall_seconds": total_wall_seconds,
            "profiling": summarize_profiling(all_evaluations, total_wall_seconds, mpi_procs, run_started_epoch),
            "spectral_window_fitting": window_fit_stats,
        }
        # Rank-based, so it can only run now that every evaluation is scored.
        # Mutates the records the summary embeds, so summary.json never names
        # an image this just deleted.
        prune_run_artefacts(evaluations_dir, all_evaluations)
        summary_path = output_dir / "summary.json"
        # Atomic: every reader treats a run with a summary.json as finished,
        # so half of one is a finished run nobody can report on, merge or
        # resume. See write_json_atomic().
        write_json_atomic(summary_path, summary)
        print(window_fit_summary_line(window_fit_stats))
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    if os.environ.get("POLYCHORD_WSCLEAN_SELF_CHECK") == "1":
        self_check_metric_resolution()
        self_check_parameter_space()
        self_check_parameter_toggle()
        self_check_spectral_window()
        self_check_lazy_numpy()
        self_check_fits_reader()
        self_check_image_pixel_size()
        self_check_source_offset()
        self_check_profiling()
        self_check_failure_record_persistence()
        self_check_resume_adoption()
        self_check_worker_timeout()
        self_check_worker_pool_connect()
        print("metric resolution self-check passed")
    else:
        main()
