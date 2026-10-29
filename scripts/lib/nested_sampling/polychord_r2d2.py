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
    DEFAULT_SUPER_RESOLUTION,
    FAILURE_OBJECTIVE,
    WORKER_DIED,
    WorkerDied,
    abort_run,
    adopt_completed_evaluations,
    cube_like_from_theta,
    cube_to_params,
    drain_record_writes,
    evidence_record,
    PolyChordFinish,
    chains_base_dir,
    start_chains_mirror,
    finished_run_records,
    follow_r2d2_worker_node,
    pack_finished_run,
    start_record_packer,
    stop_record_packer,
    post_run_line,
    post_run_phase,
    gathered_window_fit_stats,
    compute_image_metrics,
    image_dim,
    image_pixel_size_arcsec,
    load_evaluations_from_dir,
    load_parameter_space,
    mark_evaluation_start,
    mpi_rank,
    params_key,
    prewarm,
    prune_run_artefacts,
    salvage_evaluation_logs,
    prior_vector,
    r2d2_thread_count,
    r2d2_worker,
    resolve_metric,
    run_r2d2_imaging,
    self_check_lazy_numpy,
    self_check_follow_worker_node,
    self_check_metric_resolution,
    self_check_parameter_space,
    self_check_parameter_toggle,
    self_check_evidence_record,
    self_check_profiling,
    self_check_resume_adoption,
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
    write_json_atomic,
    write_polychord_paramnames,
    write_records_in_background,
)

DEFAULT_R2D2_NUM_ITER = 25
DEFAULT_R2D2_NUM_CHANS = 64
DEFAULT_R2D2_ARCHITECTURE = "unet"
DEFAULT_R2D2_CKPT_REALISATIONS = 1

# Fixed container path for the checkpoint named by R2D2_CKPT_NAME; the run
# script and `ns_refuse_missing_checkpoints` use the same name on the host.
R2D2_CKPT_PATH = f"/checkpoints/{os.environ.get('R2D2_CKPT_NAME', 'R2D2_A1')}"


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


def r2d2_iterations(stdout_path: Path) -> int | None:
    """U-Net forwards this request ran: upstream prints one `RDR:` line per
    iteration, and `prune: True` stops the series early on the residual, so the
    count (1-25) follows the image and drifts with the search (round 317)."""
    try:
        text = stdout_path.read_text(errors="replace")
    except OSError:
        return None
    return sum(line.startswith("RDR:") for line in text.splitlines()) or None


def write_r2d2_config(config_path: Path, data_file: str, output_path: str) -> int:
    threads = r2d2_thread_count()
    lines = [
        f"data_file: {data_file}",
        f"output_path: {output_path}",
        "save_all_outputs: False",
        "nufft_pkg: finufft",
        f"super_resolution: {DEFAULT_SUPER_RESOLUTION}",
        "meas_op_on_gpu: False",
        "meas_dtype: double",
        f"im_dim_x: {image_dim()}",
        f"im_dim_y: {image_dim()}",
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
        # Only read against a ground truth (`gdth_file`) or across several
        # realisations, neither used here. Unset, imager.py estimates it from
        # the operator norm: 0.38s of a 1.8s request on CSD3.
        "target_dynamic_range: 0.0",
        f"ckpt_path: {R2D2_CKPT_PATH}",
        f"ckpt_realisations: {DEFAULT_R2D2_CKPT_REALISATIONS}",
        # R2D2's set_common_args() calls torch.set_num_threads() itself, from
        # psutil's CPU affinity, and that overrides the OMP_NUM_THREADS the
        # worker pool's `apptainer exec --env` sets. Without this every rank
        # asked torch for all 20 host CPUs, so the 8 default ranks ran 160
        # threads on 20 cores.
        f"ncpus: {threads}",
        "",
    ]
    config_path.write_text("\n".join(lines))
    return threads


def evaluate(
    params: dict[str, Any],
    args: argparse.Namespace,
    eval_dir: Path,
    eval_id: int,
    objective_from_metrics: Callable[[dict[str, float]], float],
) -> dict[str, Any]:
    # Open interval lets profiler separate PolyChord time from idle time.
    mark_evaluation_start()
    sim_start = time.perf_counter()
    ms_path, sim_cmd, sim_error = simulate_measurement_set(
        params, eval_dir, args.meqtrees_image, args.platform, r2d2_mat=True
    )
    simulate_seconds = time.perf_counter() - sim_start
    # Dead worker is host failure, not an algorithm score.
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

    # R2D2 sizes its own pixels from the data and writes no WCS, so the metrics
    # have to be told the cell size it used. Same figure WSClean is handed as
    # `-scale`, from the same recorded baseline, so the two score the same sky
    # - see image_pixel_size_arcsec() in common.py.
    # Everything below is written beside the MS: tmpfs scratch when the run
    # has one, which publish_evaluation_scratch() empties at scoring.
    work_dir = ms_path.parent
    simulation = json.loads((work_dir / "simulation.json").read_text())
    if "max_proj_baseline_lambda" not in simulation["observation"]:
        raise SystemExit(
            "FATAL: simulation.json has no observation.max_proj_baseline_lambda - "
            "rebuild the meqtrees image (scripts/build.sh meqtrees), it bakes in a stale simulator"
        )
    scale_arcsec = image_pixel_size_arcsec(simulation["observation"]["max_proj_baseline_lambda"])

    # The simulate request wrote it from the arrays it put in the MS
    # (--mat-path), so there is no convert stage any more.
    mat_path = work_dir / "r2d2_data.mat"

    r2d2_dir = work_dir / "r2d2"
    r2d2_dir.mkdir()
    config_path = work_dir / "r2d2_config.yaml"
    r2d2_threads = write_r2d2_config(config_path, str(mat_path), str(r2d2_dir))

    r2d2_stdout = work_dir / "r2d2.stdout.log"
    r2d2_stderr = work_dir / "r2d2.stderr.log"
    # `imager.py` argv for this rank's long-lived R2D2 worker (see
    # run_r2d2_imaging): a fresh `docker run` of this image cost ~2.4s warm, of
    # which ~1.8s was container start plus torch and R2D2 imports.
    r2d2_cmd = [
        "--config",
        str(config_path),
        "--ckpt_path",
        R2D2_CKPT_PATH,
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
            dirty_path=dirty_path if "sigma_res" in args.metric else None,
            residual_dirty_path=residual_dirty_path if "sigma_res" in args.metric else None,
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
                "image_container_seconds": run_result.wall_seconds,
                "metrics_seconds": metrics_seconds,
            },
        })
    metrics_seconds = time.perf_counter() - metrics_start

    record = {
        "eval_id": eval_id,
        "params": params,
        "image_pixel_size_arcsec": scale_arcsec,
        "r2d2_threads": r2d2_threads,
        "metrics": metrics,
        "objective": objective,
        "paths": {
            "eval_dir": str(eval_dir),
            "measurement_set": str(ms_path),
            "simulation_metadata": str(work_dir / "simulation.json"),
            "mat": str(mat_path),
            "image": str(image_path),
            "dirty": str(dirty_path),
            "residual": str(residual_dirty_path),
        },
        "commands": {
            "simulate": sim_cmd,
            "r2d2": r2d2_cmd,
        },
        "timing": {
            "simulate_seconds": simulate_seconds,
            "image_container_seconds": run_result.wall_seconds,
            "metrics_seconds": metrics_seconds,
            "r2d2_iterations": r2d2_iterations(r2d2_stdout),
        },
    }
    return write_evaluation_record(eval_dir, record)


def self_check_r2d2_iterations() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r2d2.stdout.log"
        assert r2d2_iterations(out) is None, "a missing log has no count"
        out.write_text("RDR: 1.0e-01\nRDR rel. diff: 2.0e-01\nRDR: 8.0e-02\nRDR rel. diff: 2.0e-05\n"
                       "Imaging finished in 2/25 iterations in 0.5 sec, \n")
        assert r2d2_iterations(out) == 2, r2d2_iterations(out)


def self_check_r2d2_config_thread_cap() -> None:
    """`ncpus` must reach the config, or torch takes every host CPU per rank.
    Under R2D2_MAX_THREADS it grows as the evaluations in flight fall."""
    import tempfile

    names = ("R2D2_OMP_THREADS", "R2D2_MAX_THREADS", "NS_SCRATCH_DIR")
    saved = {name: os.environ.pop(name, None) for name in names}
    os.environ["R2D2_OMP_THREADS"] = "3"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "r2d2_config.yaml"

            def ncpus() -> int:
                write_r2d2_config(config, "/data.mat", tmp)
                return int(next(line for line in config.read_text().splitlines()
                                if line.startswith("ncpus: "))[len("ncpus: "):])

            assert ncpus() == 3
            os.environ["R2D2_MAX_THREADS"] = "24"
            assert ncpus() == 3, "adaptive threads without a scratch dir"
            os.environ["NS_SCRATCH_DIR"] = str(Path(tmp) / "scratch")
            mark_evaluation_start()
            assert ncpus() == 24, "one evaluation in flight should get every CPU"
            busy = Path(tmp) / "scratch" / ".busy"
            for pid in range(1, 5):
                (busy / f"other-{pid}").touch()
            assert ncpus() == 4, "five in flight share 24 CPUs"
            for pid in range(5, 12):
                (busy / f"other-{pid}").touch()
            assert ncpus() == 3, "never below R2D2_OMP_THREADS"
            eval_dir = Path(tmp) / "eval-1"
            eval_dir.mkdir()
            write_evaluation_record(eval_dir, {"eval_id": 1, "objective": 0.0})
            drain_record_writes()
            assert not (busy / str(os.getpid())).exists(), "scoring must clear the busy marker"
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def self_check_worker_death_is_not_scored() -> None:
    """A dead worker must stop the run, not become its best point."""
    import tempfile

    original_run_r2d2 = globals()["run_r2d2_imaging"]
    original_simulate = globals()["simulate_measurement_set"]

    def successful_simulate(
        params: dict[str, Any],
        eval_dir: Path,
        meqtrees_image: str,
        platform: str,
        r2d2_mat: bool = False,
    ) -> tuple[Path, list[str], None]:
        eval_dir.mkdir(parents=True, exist_ok=False)
        # A real simulate always leaves this, and evaluate() reads it for the
        # cell size; a stub without one
        # takes the run out on FileNotFoundError before reaching what is being
        # checked here.
        (eval_dir / "simulation.json").write_text(
            json.dumps({
                "observation": {"max_proj_baseline_lambda": 1.0e5},
                "noise": {"complex_sigma_jy": 0.01},
            })
        )
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
        globals()["simulate_measurement_set"] = lambda params, eval_dir, *a, **k: (
            eval_dir / "sim.ms", ["simulate"], subprocess.CalledProcessError(WORKER_DIED, ["simulate"])
        )
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
        globals()["run_r2d2_imaging"] = original_run_r2d2


def self_check_failure_record_persistence() -> None:
    import tempfile

    original_compute_metrics = globals()["compute_image_metrics"]
    original_run_r2d2 = globals()["run_r2d2_imaging"]
    original_simulate = globals()["simulate_measurement_set"]

    def failing_simulate(
        params: dict[str, Any],
        eval_dir: Path,
        meqtrees_image: str,
        platform: str,
        r2d2_mat: bool = False,
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
            r2d2_mat: bool = False,
        ) -> tuple[Path, list[str], None]:
            eval_dir.mkdir(parents=True, exist_ok=False)
            # A real simulate always leaves this behind, and evaluate() reads it
            # for the cell size R2D2's images carry no header for.
            (eval_dir / "simulation.json").write_text(
                json.dumps({
                    "observation": {"max_proj_baseline_lambda": 1.0e5},
                    "noise": {"complex_sigma_jy": 0.01},
                })
            )
            return eval_dir / "sim.ms", ["simulate"], None

        def successful_r2d2(*args: Any, **kwargs: Any) -> argparse.Namespace:
            return argparse.Namespace(returncode=0, wall_seconds=2.0, peak_memory_bytes=4096)

        def failing_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
            raise ValueError("bad fits")

        globals()["simulate_measurement_set"] = successful_simulate
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
        globals()["run_r2d2_imaging"] = original_run_r2d2
        globals()["simulate_measurement_set"] = original_simulate


def share_resume(chains: Path, base_dir: Path, file_root: str, read_resume: bool) -> bool:
    """Every rank, before run_polychord: make sure this rank sees <root>.resume.

    PolyChord tests `base_dir/<root>.resume` on every rank (resume_file_exists
    in nested_sampling.F90), and a rank that finds none runs GenerateLivePoints
    while the root resumes, so the search deadlocks. base_dir is node tmpfs,
    which only rank 0's ChainsMirror seeds, so with NS_NODES > 1 every resumed
    attempt hung on the other nodes (round 289). Each rank copies under its own
    name and renames, so ranks sharing a node never see half a file. The
    barrier also keeps rank 0's own node from testing before it has seeded.
    Returns whether base_dir is a copy on another node, to remove at the end."""
    try:
        import socket

        from mpi4py import MPI
    except ImportError:
        return False
    comm = MPI.COMM_WORLD
    off_root = comm.bcast(socket.gethostname(), root=0) != socket.gethostname()
    if off_root and read_resume and base_dir != chains:
        seed_resume(chains, base_dir, file_root, comm.Get_rank())
    comm.Barrier()
    return off_root and base_dir != chains


def seed_resume(chains: Path, base_dir: Path, file_root: str, rank: int) -> None:
    import shutil

    target = base_dir / f"{file_root}.resume"
    if not target.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        partial = base_dir / f".{target.name}.{rank}"
        shutil.copyfile(chains / target.name, partial)
        partial.replace(target)


def self_check_share_resume() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        chains, shm = Path(tmp) / "chains", Path(tmp) / "shm"
        chains.mkdir()
        (chains / "r.resume").write_text("checkpoint")
        # One rank, no other node: nothing to copy and nothing to remove.
        assert share_resume(chains, shm, "r", True) is False
        assert not shm.exists()
        # Another node: a whole copy under the final name, no partial left.
        seed_resume(chains, shm, "r", 7)
        assert (shm / "r.resume").read_text() == "checkpoint"
        assert [p.name for p in shm.iterdir()] == ["r.resume"]
        # A rank on the same node that comes second keeps the first copy.
        (chains / "r.resume").write_text("newer")
        seed_resume(chains, shm, "r", 8)
        assert (shm / "r.resume").read_text() == "checkpoint"


def main() -> None:
    args = parse_args()
    args.repo_root = str(Path(args.repo_root).resolve())
    if not args.checkpoints_dir:
        args.checkpoints_dir = str(Path(args.repo_root) / "checkpoints")
    args.checkpoints_dir = str(Path(args.checkpoints_dir).resolve())

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

    # Start both workers before `import pypolychord`, then join before evaluation.
    warm = prewarm(
        lambda: simulate_worker(args.meqtrees_image, args.platform),
        lambda: r2d2_worker(args.r2d2_image, args.platform, args.checkpoints_dir),
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
    # After the first evaluation, not at connect: the pool holds both FIFO
    # ends, so a rank connects before its worker has forked onto its node.
    followed = False

    def prior(cube: np.ndarray) -> np.ndarray:
        params = cube_to_params(cube, track=True)
        return prior_vector(cube, params)

    def likelihood(theta: np.ndarray) -> tuple[float, list[float]]:
        nonlocal scored, followed
        if finish.finished():
            return finish.discarded(), []
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
                salvage_evaluation_logs(eval_dir)
                abort_run(str(exc))
            except (Exception, SystemExit):
                # Anything else is a bug in this file, and a bug here used to
                # hang the job rather than end it. PolyChord calls the
                # likelihood from Fortran, so the traceback unwinds this rank
                # only and every other rank waits forever in a collective that
                # never completes: every core busy, nothing landing, and
                # run_with_retries never even reached because nothing exited.
                salvage_evaluation_logs(eval_dir)
                abort_run(traceback.format_exc())
            cache[key] = float(record["objective"])
            if not followed:
                followed = True
                follow_r2d2_worker_node(args.r2d2_image, args.meqtrees_image)
            print(json.dumps({"eval_id": eval_id, "objective": record["objective"], "params": params}), flush=True)
        return cache[key], []

    settings = PolyChordSettings(len(load_parameter_space()), 0)
    settings.base_dir = str(chains_base_dir(output_dir))
    settings.file_root = "r2d2_vlaa"
    settings.nlive = args.nlive
    settings.num_repeats = args.num_repeats
    settings.max_ndead = args.max_ndead
    settings.seed = args.seed
    settings.synchronous = os.environ.get("NS_SYNCHRONOUS", "0") != "0"
    # Checkpoint long runs so interruptions are resumable via `--output-dir`.
    resume_path = output_dir / "chains" / f"{settings.file_root}.resume"
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

    if mpi_rank() == 0:  # one writer: rank 0's mirror seeds tmpfs from it, and another
        # rank's truncating rewrite once seeded an empty file (round 229)
        write_polychord_paramnames(output_dir / "chains", settings.file_root)
    warm()
    run_start, run_started_epoch = time.monotonic(), time.time()
    write_records_in_background()
    packer = start_record_packer(evaluations_dir) if mpi_rank() == 0 else None
    mirror = start_chains_mirror(output_dir, Path(settings.base_dir))
    node_copy = share_resume(output_dir / "chains", Path(settings.base_dir), settings.file_root, settings.read_resume)
    finish = PolyChordFinish()
    try:
        output = pypolychord.run_polychord(likelihood, len(load_parameter_space()), 0, settings, prior, finish.dumper)
    except FileNotFoundError:
        # pypolychord reads <root>.stats on every rank once the run is over, and
        # across nodes (NS_NODES) a rank off the root's node has no copy of its
        # tmpfs chains; only rank 0 uses the output (docs/csd3-speed.md, round 287).
        if mpi_rank() == 0 or (Path(settings.base_dir) / f"{settings.file_root}.stats").exists():
            raise
        output = None
    total_wall_seconds = time.monotonic() - run_start
    finish.close()
    if node_copy:
        import shutil

        shutil.rmtree(settings.base_dir, ignore_errors=True)
    if mirror is not None:
        mirror.stop()  # before summary.json: a finished run's chains are all on Lustre
    stop_record_packer(packer)
    # Collective, as is the one below, so every rank calls both before rank 0 goes on alone.
    all_evaluations = finished_run_records(evaluations_dir)
    window_fit_stats = gathered_window_fit_stats()

    if all_evaluations is not None:
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
                # Which MPI scheduling this run used, because it changes how
                # much of the worker-time budget the profiling block can
                # account for and a run's numbers are unreadable without it.
                "synchronous": settings.synchronous,
            },
            "r2d2_fixed_hyperparameters": {
                "im_dim_x": image_dim(),
                "im_dim_y": image_dim(),
                "num_iter": DEFAULT_R2D2_NUM_ITER,
                "num_chans": DEFAULT_R2D2_NUM_CHANS,
                "architecture": DEFAULT_R2D2_ARCHITECTURE,
                "super_resolution": DEFAULT_SUPER_RESOLUTION,
                "ckpt_path": R2D2_CKPT_PATH,
                "ckpt_realisations": DEFAULT_R2D2_CKPT_REALISATIONS,
            },
            "parameter_space": load_parameter_space(),
            # Before the bulk, so a listing reads it out of the head rather
            # than stat-ing every evaluation directory.
            "evaluation_count": len(all_evaluations),
            "evaluations": all_evaluations,
            "worst_evaluation": best,
            "total_wall_seconds": total_wall_seconds,
            "evidence": evidence_record(output),
            "profiling": summarize_profiling(all_evaluations, total_wall_seconds, mpi_procs, run_started_epoch),
            "spectral_window_fitting": window_fit_stats,
        }
        # Rank-based, so it can only run now that every evaluation is scored.
        # Mutates the records the summary embeds, so summary.json never names
        # an image this just deleted.
        started = time.monotonic()
        prune_run_artefacts(evaluations_dir, all_evaluations)
        started = post_run_phase("prune", started)
        summary_path = output_dir / "summary.json"
        # Atomic: every reader treats a run with a summary.json as finished,
        # so half of one is a finished run nobody can report on, merge or
        # resume. See write_json_atomic().
        write_json_atomic(summary_path, summary)
        started = post_run_phase("summary", started)
        pack_finished_run(evaluations_dir)
        post_run_phase("pack", started)
        print(post_run_line())
        print(window_fit_summary_line(window_fit_stats))
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    if os.environ.get("POLYCHORD_R2D2_SELF_CHECK") == "1":
        self_check_metric_resolution()
        self_check_parameter_space()
        self_check_parameter_toggle()
        self_check_spectral_window()
        self_check_lazy_numpy()
        self_check_r2d2_config_thread_cap()
        self_check_r2d2_iterations()
        self_check_source_offset()
        self_check_profiling()
        self_check_evidence_record()
        self_check_failure_record_persistence()
        self_check_worker_death_is_not_scored()
        self_check_resume_adoption()
        self_check_worker_timeout()
        self_check_worker_pool_connect()
        self_check_follow_worker_node()
        self_check_share_resume()
        print("metric resolution self-check passed")
    else:
        main()
