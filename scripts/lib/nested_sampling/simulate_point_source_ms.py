#!/usr/bin/env python3
"""Create a noisy VLA Measurement Set for a single point source."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from casacore.tables import table


Cattery_VLA_A = Path("/usr/share/doc/makems/VLAA_ANT.tar.gz")
ANTENNA_TABLE_NAME = "VLAA_ANT"
TDL_SCRIPT = Path("/opt/ri-nested-sampling/point_source_forest.py")
# Shared with prebuild_skeletons(), which enumerates evaluation NTimes.
DEFAULT_INTEGRATION_SECONDS = 120.0
# The declination every archived run was simulated at, and the only one the
# skeletons baked into the image were built for; `declination_deg` in
# defaults.toml pins back to this while it is disabled.
DEFAULT_DECLINATION_DEG = 65.0
SPEED_OF_LIGHT = 299792458.0

# RAM avoids ~0.5s bind-mount fsync cost per run; final ~1MB copy costs ~2ms.
# ponytail: Docker's 64MB /dev/shm is ~30x current largest MS; raise it if needed.
SCRATCH_ROOT = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None


def scratch_root_for(destination: Path) -> str | None:
    shared = os.environ.get("NS_SCRATCH_DIR", "")
    if shared and destination.is_relative_to(shared):
        return str(destination)
    return SCRATCH_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-ms", required=True, help="Measurement Set path to create")
    parser.add_argument("--metadata-json", help="Optional JSON metadata path")
    parser.add_argument("--vla-config", default="VLA.A", choices=["VLA.A"])
    parser.add_argument("--observation-minutes", type=float, required=True)
    parser.add_argument("--integration-seconds", type=float, default=DEFAULT_INTEGRATION_SECONDS)
    parser.add_argument("--channel-count", type=int, required=True)
    parser.add_argument("--start-frequency-hz", type=float, required=True)
    parser.add_argument("--channel-width-hz", type=float, required=True)
    parser.add_argument("--source-flux-jy", type=float, default=1.0)
    parser.add_argument("--source-l-arcsec", type=float, default=0.0)
    parser.add_argument("--source-m-arcsec", type=float, default=0.0)
    parser.add_argument("--declination-deg", type=float, default=DEFAULT_DECLINATION_DEG)
    parser.add_argument("--dynamic-range", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    # On the argv rather than in the worker request, so the command recorded in
    # metrics.json reproduces the evaluation's bound as well as its parameters.
    parser.add_argument("--predict-wait-seconds", type=float, default=None)
    return parser.parse_args(argv)


def require_clean_output(output_ms: Path) -> None:
    if output_ms.exists():
        raise SystemExit(f"FATAL: output Measurement Set already exists: {output_ms}")
    for suffix in ("_p0", "_p1", ".gds"):
        candidate = output_ms.parent / f"{output_ms.name}{suffix}"
        if candidate.exists():
            raise SystemExit(f"FATAL: makems scratch output already exists: {candidate}")


def makems_declination(degrees: float) -> str:
    """Decimal degrees as the dotted sexagesimal makems hands to casacore's MVAngle.

    Rounded to whole arcseconds, so the string is always three dotted fields:
    a fractional-second field would be a fourth, which MVAngle reads as
    something else entirely. `declination_deg` is a whole-degree dimension, so
    the rounding only bites on a hand-passed `--declination-deg`.
    """
    d, rest = divmod(round(abs(degrees) * 3600.0), 3600)
    m, s = divmod(rest, 60)
    return f"{'-' if degrees < 0 else ''}{d}.{m}.{s}"


def n_times_of(args: argparse.Namespace) -> int:
    return max(1, int(math.ceil(args.observation_minutes * 60.0 / args.integration_seconds)))


def write_makems_config(args: argparse.Namespace, output_ms: Path) -> Path:
    output_ms.parent.mkdir(parents=True, exist_ok=True)
    antenna_dst = output_ms.parent / ANTENNA_TABLE_NAME
    if not antenna_dst.exists():
        shutil.unpack_archive(Cattery_VLA_A, output_ms.parent)

    n_times = n_times_of(args)
    cfg = output_ms.parent / "makems.cfg"
    cfg.write_text(
        "\n".join(
            [
                f"StartFreq={args.start_frequency_hz:.8f}",
                f"StepFreq={args.channel_width_hz:.8f}",
                "StartTime=2005/02/09/21:21:40",
                f"StepTime={args.integration_seconds:.8f}",
                "RightAscension=0:0:0",
                f"Declination={makems_declination(args.declination_deg)}",
                "NBands=1",
                f"NFrequencies={args.channel_count}",
                f"NTimes={n_times}",
                "NParts=1",
                "WriteAutoCorr=F",
                f"AntennaTableName={ANTENNA_TABLE_NAME}",
                f"MSName={output_ms.name}",
                "WriteImagerColumns=F",
                "MSDesPath=.",
                "",
            ]
        )
    )
    return cfg


def run_makems(output_ms: Path) -> None:
    log_path = output_ms.parent / "makems.log"
    with log_path.open("w") as log:
        subprocess.run(
            ["makems"],
            cwd=output_ms.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    for suffix in ("_p0", "_p1"):
        part = output_ms.parent / f"{output_ms.name}{suffix}"
        if part.exists():
            shutil.move(str(part), output_ms)
            return
    if output_ms.exists():
        return
    raise SystemExit(f"FATAL: makems did not create {output_ms.name}_p0, {output_ms.name}_p1, or {output_ms}")


# Cache by (NTimes, NFrequencies): copy/patch ~0.002s vs makems ~0.05s;
# shared /dev/shm serves all ranks.
#
# Capped at NS_MS_SKELETON_CACHE_MAX shapes, then stops publishing: the default
# 5-parameter space has under 100 shapes (~20MB), but with integration_seconds
# and declination_deg searched nearly every evaluation is a new shape. Uncapped,
# a 112-rank run put 138GB in /dev/shm in 25 minutes and WSClean then died of
# bad_alloc (docs/csd3-experiments.md, E5).
_SKELETON_DIR: Path | None = None
_SKELETON_DIR_EXPLICIT = False

# `--prebuild-skeletons` puts default-run shapes here at image build time;
# see docker/meqtrees/Dockerfile. Read-only at run time: under Apptainer the
# image is, and under Docker it may as well have been.
BAKED_SKELETON_DIR = Path("/opt/ms-skeletons")


def skeleton_dir() -> Path:
    """Where unseen shapes are built and published: the run's shared MS
    scratch, which is what the container's own filesystem amounted to under
    Docker - a fresh cache per run, gone with it."""
    global _SKELETON_DIR
    if _SKELETON_DIR is None:
        root = os.environ.get("NS_SCRATCH_DIR") or SCRATCH_ROOT or tempfile.gettempdir()
        _SKELETON_DIR = Path(root) / f"ms-skeletons-{os.getuid()}"
        _SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    return _SKELETON_DIR


def use_skeleton_cache(directory: Path | None) -> None:
    """Name the one directory to read and publish; None restores the default."""
    global _SKELETON_DIR, _SKELETON_DIR_EXPLICIT
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    _SKELETON_DIR = directory
    _SKELETON_DIR_EXPLICIT = directory is not None


def cached_skeleton(key: str) -> Path:
    """The entry for `key`: published this run, else baked into the image,
    else the path to publish it at. Baked shapes are a head start only on the
    default cache; a directory named with use_skeleton_cache() is on its own."""
    name = hashlib.sha256(key.encode()).hexdigest()[:32]
    published = skeleton_dir() / name
    if not published.exists() and not _SKELETON_DIR_EXPLICIT and (BAKED_SKELETON_DIR / name).is_dir():
        return BAKED_SKELETON_DIR / name
    return published


def publish_skeleton(built_ms: Path, cached: Path) -> None:
    limit = int(os.environ.get("NS_MS_SKELETON_CACHE_MAX", "512"))
    # Staging directories are mkdtemp's `tmp*`; published shapes are hex names.
    if sum(1 for name in os.listdir(skeleton_dir()) if not name.startswith("tmp")) >= limit:
        return
    staging = Path(tempfile.mkdtemp(dir=skeleton_dir()))
    try:
        shutil.copytree(built_ms, staging / "ms", symlinks=True)
        try:
            os.rename(staging / "ms", cached)
        except OSError:
            pass
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def patch_spectral_window(output_ms: Path, start_frequency_hz: float, channel_width_hz: float) -> None:
    with table(str(output_ms / "SPECTRAL_WINDOW"), readonly=False, ack=False) as spw:
        n_chan = int(spw.getcol("NUM_CHAN")[0])
        spw.putcol("CHAN_FREQ", (start_frequency_hz + (np.arange(n_chan) + 0.5) * channel_width_hz)[None, :])
        widths = np.full((1, n_chan), channel_width_hz)
        for column in ("CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION"):
            spw.putcol(column, widths)
        spw.putcol("REF_FREQUENCY", np.array([start_frequency_hz + n_chan * channel_width_hz / 2.0]))
        spw.putcol("TOTAL_BANDWIDTH", np.array([n_chan * channel_width_hz]))


def skeleton_key(cfg_text: str) -> str:
    return "\n".join(line for line in cfg_text.splitlines() if not line.startswith(("StartFreq=", "StepFreq=")))


def make_ms_skeleton(cfg: Path, output_ms: Path, args: argparse.Namespace, prune_unused: bool = False) -> None:
    key = skeleton_key(cfg.read_text())
    cached = cached_skeleton(key)
    if not cached.exists():
        run_makems(output_ms)
        publish_skeleton(output_ms, cached)
        return
    ignore = shutil.ignore_patterns(*UNUSED_SUBTABLES) if prune_unused else None
    shutil.copytree(cached, output_ms, symlinks=True, ignore=ignore)
    patch_spectral_window(output_ms, args.start_frequency_hz, args.channel_width_hz)
    (output_ms.parent / "makems.log").write_text(f"reused a cached makems skeleton for:\n{key}\n")


# With declination_deg and integration_seconds searched, the whole-shape cache
# above almost never hits, and makems spends ~3ms a timestep converting antenna
# positions to J2000 UVW (docs/csd3-speed.md, round 2). But timestep k of an
# observation does not depend on its length, so one makems build per
# (declination, integration) serves every shorter observation: a one-timestep
# template for the channel count, extended with the cached rows. A row's UVW is
# makems' per-antenna UVW (relative to antenna 0) of ANTENNA2 minus ANTENNA1,
# which is what is cached; save_observation() refuses anything it cannot
# rebuild bit for bit.
def _config_key(cfg_text: str, dropped: tuple[str, ...]) -> str:
    key = "\n".join(line for line in cfg_text.splitlines() if not line.startswith(dropped))
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def _observation_path(cfg_text: str) -> Path:
    directory = skeleton_dir() / "observations"
    directory.mkdir(exist_ok=True)
    return directory / (_config_key(cfg_text, ("StartFreq=", "StepFreq=", "NFrequencies=", "NTimes=")) + ".npz")


def _template_path(cfg_text: str) -> Path:
    directory = skeleton_dir() / "templates"
    directory.mkdir(exist_ok=True)
    return directory / _config_key(cfg_text, ("StartFreq=", "StepFreq=", "NTimes=", "Declination=", "StepTime="))


def _atomic_publish(write, destination: Path) -> None:
    staging = Path(tempfile.mkdtemp(dir=destination.parent))
    try:
        write(staging / "entry")
        try:
            os.replace(staging / "entry", destination)
        except OSError:
            pass  # another rank published this directory first
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def save_observation(ms_path: Path, cfg_text: str) -> None:
    with table(str(ms_path), ack=False) as ms:
        cols = {c: ms.getcol(c) for c in ("TIME", "TIME_CENTROID", "UVW", "ANTENNA1", "ANTENNA2", "INTERVAL", "EXPOSURE")}
    with table(str(ms_path / "FIELD"), ack=False) as field:
        dirs = {c: field.getcol(c) for c in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR", "TIME")}
    with table(str(ms_path / "OBSERVATION"), ack=False) as obs:
        time_range = obs.getcol("TIME_RANGE")
    n_rows = len(cols["TIME"])
    a1, a2 = cols["ANTENNA1"], cols["ANTENNA2"]
    n_base = int(np.count_nonzero(cols["TIME"] == cols["TIME"][0]))
    n_times = n_rows // n_base
    if n_base * n_times != n_rows or not (np.array_equal(a1, np.tile(a1[:n_base], n_times))
                                           and np.array_equal(a2, np.tile(a2[:n_base], n_times))):
        return
    uvw = cols["UVW"].reshape(n_times, n_base, 3)
    first = a1[:n_base] == 0
    antenna_uvw = np.zeros((n_times, int(max(a1.max(), a2.max())) + 1, 3))
    antenna_uvw[:, a2[:n_base][first]] = uvw[:, first]
    per_time = {c: cols[c].reshape(n_times, n_base)[:, 0] for c in ("TIME", "TIME_CENTROID")}
    interval, exposure = cols["INTERVAL"][0], cols["EXPOSURE"][0]
    rebuilt = antenna_uvw[:, a2[:n_base]] - antenna_uvw[:, a1[:n_base]]
    if not (np.array_equal(rebuilt, uvw)
            and all(np.array_equal(np.repeat(per_time[c], n_base), cols[c]) for c in per_time)
            and np.all(cols["INTERVAL"] == interval) and np.all(cols["EXPOSURE"] == exposure)
            and time_range[0, 1] == time_range[0, 0] + n_times * interval):
        return
    destination = _observation_path(cfg_text)
    try:
        with np.load(destination) as cached:
            if len(cached["time"]) >= n_times:
                return
    except (OSError, ValueError, KeyError):
        pass

    def write(path: Path) -> None:
        with path.open("wb") as out:
            np.savez(out, time=per_time["TIME"], time_centroid=per_time["TIME_CENTROID"], antenna_uvw=antenna_uvw,
                     interval=interval, exposure=exposure, time_range_start=time_range[0, 0],
                     **{f"FIELD_{c}": v for c, v in dirs.items()})

    _atomic_publish(write, destination)


def load_observation(cfg_text: str, n_times: int) -> dict | None:
    try:
        with np.load(_observation_path(cfg_text)) as cached:
            if len(cached["time"]) < n_times:
                return None
            return {k: cached[k] for k in cached.files}
    except (OSError, ValueError, KeyError):
        return None


def one_timestep_template(cfg_text: str, args: argparse.Namespace, scratch: Path) -> Path:
    template = _template_path(cfg_text)
    if not template.exists():
        one = argparse.Namespace(**vars(args))
        one.observation_minutes = args.integration_seconds / 120.0  # NTimes=1
        with tempfile.TemporaryDirectory(dir=scratch) as build:
            ms = Path(build) / Path(args.output_ms).name
            write_makems_config(one, ms)
            run_makems(ms)
            _atomic_publish(lambda path: shutil.copytree(ms, path, symlinks=True), template)
    return template


def extend_template(template: Path, observation: dict, output_ms: Path, n_times: int, args: argparse.Namespace) -> None:
    shutil.copytree(template, output_ms, symlinks=True, ignore=shutil.ignore_patterns(*UNUSED_SUBTABLES))
    with table(str(output_ms), readonly=False, ack=False) as ms:
        a1, a2 = ms.getcol("ANTENNA1"), ms.getcol("ANTENNA2")
        n_base = len(a1)
        ms.addrows((n_times - 1) * n_base)
        antenna_uvw = observation["antenna_uvw"][:n_times]
        ms.putcol("UVW", (antenna_uvw[:, a2] - antenna_uvw[:, a1]).reshape(-1, 3))
        ms.putcol("ANTENNA1", np.tile(a1, n_times))
        ms.putcol("ANTENNA2", np.tile(a2, n_times))
        ms.putcol("TIME", np.repeat(observation["time"][:n_times], n_base))
        ms.putcol("TIME_CENTROID", np.repeat(observation["time_centroid"][:n_times], n_base))
        ms.putcol("INTERVAL", np.full(n_times * n_base, observation["interval"]))
        ms.putcol("EXPOSURE", np.full(n_times * n_base, observation["exposure"]))
    with table(str(output_ms / "FIELD"), readonly=False, ack=False) as field:
        for column in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR", "TIME"):
            field.putcol(column, observation[f"FIELD_{column}"])
    start = observation["time_range_start"]
    with table(str(output_ms / "OBSERVATION"), readonly=False, ack=False) as obs:
        obs.putcol("TIME_RANGE", np.array([[start, start + n_times * observation["interval"]]]))
    patch_spectral_window(output_ms, args.start_frequency_hz, args.channel_width_hz)


def make_ms(cfg: Path, output_ms: Path, args: argparse.Namespace) -> None:
    """Build the evaluation's MS skeleton, running makems only for an
    observation no earlier evaluation has covered."""
    cfg_text = cfg.read_text()
    if cached_skeleton(skeleton_key(cfg_text)).exists():
        make_ms_skeleton(cfg, output_ms, args, prune_unused=True)
        return
    n_times = n_times_of(args)
    observation = load_observation(cfg_text, n_times)
    if observation is None:
        make_ms_skeleton(cfg, output_ms, args, prune_unused=True)
        save_observation(output_ms, cfg_text)
        return
    template = one_timestep_template(cfg_text, args, output_ms.parent)
    extend_template(template, observation, output_ms, n_times, args)
    (output_ms.parent / "makems.log").write_text(f"extended a cached observation to {n_times} timesteps\n")


def prebuild_skeletons(space: dict) -> None:
    minutes_lo, minutes_hi = space["observation_minutes"]
    chan_lo, chan_hi = space["channel_count"]
    step = DEFAULT_INTEGRATION_SECONDS
    shapes = [
        (n_times * step / 60.0, n_chan)
        for n_times in range(
            max(1, math.ceil(minutes_lo * 60.0 / step)),
            max(1, math.ceil(minutes_hi * 60.0 / step)) + 1,
        )
        for n_chan in range(chan_lo, chan_hi + 1)
    ]
    for minutes, n_chan in shapes:
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
            ms = Path(scratch) / "sim.ms"
            # StartFreq/StepFreq are outside the cache key and are rewritten on
            # every hit, so any value builds a reusable entry.
            built = parse_args([
                "--output-ms", str(ms), "--observation-minutes", repr(minutes),
                "--channel-count", str(n_chan), "--start-frequency-hz", "1.0e9",
                "--channel-width-hz", "1.0e6", "--dynamic-range", "300",
            ])
            make_ms_skeleton(write_makems_config(built, ms), ms, built)


def determine_corr_selection(output_ms: Path) -> tuple[str, int]:
    corr_sel_by_count = {1: "1", 2: "2", 4: "2x2"}
    with table(str(output_ms), readonly=True, ack=False) as ms:
        n_corr = ms.getcol("DATA", startrow=0, nrow=1).shape[-1]
    corr_sel = corr_sel_by_count.get(n_corr)
    if corr_sel is None:
        raise SystemExit(f"FATAL: unsupported correlation count in {output_ms}: {n_corr}")
    return corr_sel, n_corr


@contextlib.contextmanager
def redirect_fds(out_path: Path, err_path: Path | None = None):
    """Redirect stdout and stderr for this block, merging stderr when omitted."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    out = out_path.open("w")
    err = err_path.open("w") if err_path else None
    try:
        os.dup2(out.fileno(), 1)
        os.dup2((err or out).fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        out.close()
        if err:
            err.close()


_MQS = None
_PREDICTS_SINCE_RESTART = 0

# The forest currently loaded into the meqserver, keyed on the tdlconf text
# with the MS name removed - see run_meqtrees_predict().
_FOREST: dict[str, object] = {}


def meqserver_session():
    global _MQS
    if _MQS is None:
        import Timba.utils

        # Timba parses sys.argv for verbosity flags unless told not to.
        Timba.utils.verbosity.disable_argv()
        from Timba.Apps import meqserver
        from Timba.TDL import TDLOptions

        TDLOptions.enable_save_config(False)
        _MQS = meqserver.default_mqs(wait_init=10, extra=["-mt", "1"])
    return _MQS


# Bound MeqTrees deadlocks so worker can restart its server, without blocking
# PolyChord's collective. The floor, for a run that does not say otherwise: the
# predict is linear in NTimes, so a rank asking for a large Measurement Set
# sends the bound it sized for that shape as --predict-wait-seconds. See
# predict_wait_seconds() in common.py and docs/robustness.md.
PREDICT_WAIT_SECONDS = 3.0


class MeqserverWedged(RuntimeError):
    """Predict exceeded its wait; restart server and retry."""


def restart_meqserver_session() -> None:
    """Kill and reap wedged meqserver, leaving worker alive for retry."""
    global _MQS
    from Timba.Apps import meqserver

    global _PREDICTS_SINCE_RESTART
    pid = getattr(_MQS, "serv_pid", None)
    _MQS = None
    _PREDICTS_SINCE_RESTART = 0
    _FOREST.clear()
    # default_mqs() hands back its own module global whenever that is already a
    # meqserver, so clearing it is what makes a restart possible at all.
    meqserver.mqs = None
    if pid:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(OSError):
            os.waitpid(pid, 0)


def stop_meqserver_session() -> None:
    """Stop explicitly so Timba's non-daemon threads cannot hang interpreter exit."""
    global _MQS
    if _MQS is not None:
        from Timba.Apps import meqserver

        _MQS = None
        # The forest lives in the server, not here.
        _FOREST.clear()
        meqserver.stop_default_mqs()


def point_to_measurement_set(module, output_ms: Path) -> None:
    mssel = module.mssel
    if not mssel._select_new_ms(str(output_ms)):
        raise SystemExit(f"FATAL: MeqTrees could not read {output_ms}")
    mssel.msname = str(output_ms)
    # _select_new_ms() re-lists the MS's data columns, which resets the output
    # column option that _define_forest set to DATA. Miss this and the sinks
    # write to CORRECTED_DATA instead - no error, just an all-zero DATA column.
    mssel.output_column = "DATA"


def run_meqtrees_predict(
    output_ms: Path,
    corr_sel: str,
    source_flux_jy: float,
    l_rad: float,
    m_rad: float,
    wait_seconds: float | None = None,
) -> None:
    # None, not the module constant as a default argument: the constant is a
    # global the self-checks replace, and a default argument would freeze the
    # value this module was imported with.
    wait_seconds = PREDICT_WAIT_SECONDS if wait_seconds is None else wait_seconds
    tdlconf = output_ms.parent / "point_source_forest.tdlconf"
    tdlconf.write_text(
        "\n".join(
            [
                "[predict]",
                f"ms_sel.msname = {output_ms}",
                f"ms_sel.ms_corr_sel = {corr_sel}",
                f"source_flux_jy = {source_flux_jy!r}",
                f"source_l_rad = {l_rad!r}",
                f"source_m_rad = {m_rad!r}",
                "",
            ]
        )
    )
    # Compiling the forest is ~0.034s of every evaluation and depends on every
    # tdlconf key except ms_sel.msname: the antenna layout and phase centre come
    # from the fixed antenna table and RightAscension/Declination that
    # write_makems_config() hardcodes, and the MS shape is runtime data. So the
    # forest is compiled once per distinct source/correlation setup and later
    # evaluations just point it at their own MS. self_check_forest_reuse()
    # is the guard on that claim.
    key = "\n".join(line for line in tdlconf.read_text().splitlines() if not line.startswith("ms_sel.msname"))
    # A wedged meqserver is replaced once and the predict retried against the
    # fresh one, which costs ~0.2s and the caller never sees. Twice in a row is
    # no longer a stuck server but something this worker cannot fix, so it goes
    # back to the rank as a dead worker rather than a failed evaluation - see
    # MeqserverWedged.
    #
    # Errors from a predict get the same one retry on a fresh meqserver: on
    # CSD3 they came from a newly compiled forest in a new server ("node
    # 'VisDataMux' not found"), and every failing evaluation replayed alone
    # succeeded (docs/csd3-experiments.md). Parameters that really break the
    # predict fail again and are scored as before.
    for attempt in range(2):
        try:
            errors = _compile_and_predict(tdlconf, key, output_ms, wait_seconds)
            if errors and not attempt:
                with (output_ms.parent / "meqserver-wedged.log").open("a") as note:
                    note.write(f"attempt 1: {len(errors)} predict error(s): {errors!r}\n")
                restart_meqserver_session()
                continue
            break
        except MeqserverWedged as exc:
            # Its own file, not meqtree-pipeliner.log: the retry reopens that
            # log and truncates it, so a wedge that was recovered from left no
            # trace of ever having happened. This is the only record that an
            # evaluation cost seconds instead of milliseconds.
            with (output_ms.parent / "meqserver-wedged.log").open("a") as note:
                note.write(f"attempt {attempt + 1}: {exc}\n")
            if attempt:
                raise
            restart_meqserver_session()
    # The meqserver keeps every MS it has predicted into open after the rank
    # deletes it, so a tmpfs scratch never gets that memory back: 13MB an
    # evaluation in the 9-parameter space, 96GB over a 112-rank node in 20
    # minutes (docs/csd3-experiments.md, E6). Replacing it closes them.
    global _PREDICTS_SINCE_RESTART
    _PREDICTS_SINCE_RESTART += 1
    if _PREDICTS_SINCE_RESTART >= int(os.environ.get("NS_MEQSERVER_RECYCLE", "20")):
        restart_meqserver_session()
    if errors:
        raise SystemExit(f"FATAL: meqserver reported {len(errors)} error(s) during the predict")


def _compile_and_predict(tdlconf: Path, key: str, output_ms: Path, wait_seconds: float | None = None) -> list:
    """Compile and run bounded predict; raise MeqserverWedged on timeout."""
    wait_seconds = PREDICT_WAIT_SECONDS if wait_seconds is None else wait_seconds
    mqs = meqserver_session()
    from Timba.TDL import Compile, TDLOptions

    with redirect_fds(output_ms.parent / "meqtree-pipeliner.log"):
        module = _FOREST.get(key)
        if module is None:
            TDLOptions.config.read(str(tdlconf))
            TDLOptions.config.set_save_filename(None)
            module, _ns, msg = Compile.compile_file(mqs, str(TDL_SCRIPT), config="predict")
            print("###", msg)
            # The meqserver holds one forest, so a new compile invalidates the
            # previous entry rather than adding to it.
            _FOREST.clear()
            _FOREST[key] = module
            # The compiled selector can still name the MS an earlier compile
            # read (the warm-up's, gone), whatever the tdlconf says.
            point_to_measurement_set(module, output_ms)
        else:
            point_to_measurement_set(module, output_ms)
            print("### reusing the compiled forest; only the Measurement Set changed")
        try:
            TDLOptions.get_job_func("predict")(mqs, None, wait=wait_seconds)
        except AttributeError as exc:
            # Timba's meq() ends in `return msg.payload`, and msg is None when
            # the wait expires, so a timeout arrives here as an AttributeError
            # off that line rather than as a return value. Anything else of the
            # same type is a real bug and is left to propagate.
            if "NoneType" not in str(exc):
                raise
            raise MeqserverWedged(f"no reply to the predict in {wait_seconds}s") from exc
        # get_error_log() flushes, so each request only sees its own errors.
        errors = mqs.get_error_log()
        for index, (_event, error) in enumerate(errors):
            # !r, not str(): Timba's DMI record __str__ is still py2
            # (`string.join`) and raises AttributeError, which would replace the
            # meqserver's error with a traceback from the reporting path itself.
            print(f"###   {index:03d}: {error!r}")
    return errors


def point_source_visibilities(
    uvw: np.ndarray, freqs_hz: np.ndarray, source_flux_jy: float, l_rad: float, m_rad: float, n_corr: int
) -> np.ndarray:
    """Stokes-I point source at (l, m): what the Meow forest predicts, to float32
    rounding, at ~1/30 of its cost. self_check_analytic_predict() is the guard."""
    n_minus_1 = math.sqrt(1.0 - l_rad * l_rad - m_rad * m_rad) - 1.0
    delay_m = uvw @ np.array([l_rad, m_rad, n_minus_1])
    vis = source_flux_jy * np.exp((2j * math.pi / SPEED_OF_LIGHT) * np.outer(delay_m, freqs_hz))
    data = np.zeros((len(uvw), len(freqs_hz), n_corr), dtype=np.complex64)
    data[:, :, 0] = vis
    data[:, :, -1] = vis
    return data


# makems writes the full MSv2 subtable set, and casacore attaches every subtable
# on every open of the parent table - which WSClean does once per gridding and
# degridding pass, ~16 times an evaluation, at ~0.21ms a subtable per open.
# These six are empty (FLAG_CMD, HISTORY) or carry nothing a single-field
# unpolarised point-source simulation depends on, so they are dropped once the
# visibilities are written - not in the cached skeleton, because casacore
# refuses to open an MS that is missing any of them and the MeqTrees predict
# (now only self_check_analytic_predict()'s) needs it opened that way. Runs
# never copy them out of the skeleton (make_ms_skeleton()'s `prune_unused`);
# the delete below covers a fresh makems build. Worth -13.8% on the wsclean binary and +14.9%
# evaluations per second for the first five, and FEED another ~3%, with
# bit-identical images; see docs/nested-sampling-ms-open.md. The six that
# stay were each tried and each kills WSClean, so this list is complete.
UNUSED_SUBTABLES = ("FEED", "FLAG_CMD", "HISTORY", "POINTING", "PROCESSOR", "STATE")


def fill_point_source_visibilities(args: argparse.Namespace, output_ms: Path) -> dict[str, object]:
    if args.dynamic_range <= 0:
        raise SystemExit("FATAL: --dynamic-range must be positive")
    if args.source_flux_jy <= 0:
        raise SystemExit("FATAL: --source-flux-jy must be positive")

    noise_sigma_jy = args.source_flux_jy / args.dynamic_range
    l_rad = math.radians(args.source_l_arcsec / 3600.0)
    m_rad = math.radians(args.source_m_arcsec / 3600.0)

    with table(str(output_ms / "SPECTRAL_WINDOW"), readonly=True, ack=False) as spw:
        freqs_hz = np.asarray(spw.getcol("CHAN_FREQ")[0], dtype=np.float64)

    # ponytail: this simulator supports one unpolarized point source; full Stokes
    # models and multi-source dynamic-range stress cases are a follow-up ceiling.
    # The MeqTrees predict this replaced was half the stage on CSD3 (0.79s of
    # 1.51s in the 9-parameter space); docs/csd3-speed.md.
    rng = np.random.default_rng(args.seed)
    with table(str(output_ms), readonly=False, ack=False) as ms:
        # One row for the shape: DATA is about to be overwritten whole.
        _, n_chan, n_corr = ms.getcol("DATA", startrow=0, nrow=1).shape
        if n_chan != len(freqs_hz):
            raise SystemExit(f"FATAL: DATA has {n_chan} channels, SPW has {len(freqs_hz)}")
        uvw = np.asarray(ms.getcol("UVW"), dtype=np.float64)
        data = point_source_visibilities(uvw, freqs_hz, args.source_flux_jy, l_rad, m_rad, n_corr)

        if noise_sigma_jy:
            per_component_sigma = noise_sigma_jy / math.sqrt(2.0)
            # Added in place to the two float32 halves rather than built as a
            # complex128 array and added out of place. Rounding each component
            # to float32 before the add is what the old `.astype(np.complex64)`
            # did, so the column is bit-identical for a given seed - it just
            # stops allocating three more copies of DATA to get there (0.69ms
            # an evaluation). The two draws stay separate calls in this order
            # because that, not the arithmetic, is what fixes the stream.
            data.real += rng.normal(0.0, per_component_sigma, data.shape).astype(np.float32)
            data.imag += rng.normal(0.0, per_component_sigma, data.shape).astype(np.float32)

        ms.putcol("DATA", data)
        for optional_col in ("MODEL_DATA", "CORRECTED_DATA"):
            if optional_col in ms.colnames():
                ms.putcol(optional_col, data)
        if "FLAG" in ms.colnames():
            ms.putcol("FLAG", np.zeros(ms.getcol("FLAG").shape, dtype=bool))
        # WEIGHT and SIGMA are deliberately left at makems' 1.0. This simulator's
        # noise is one sigma for every row and channel, so the pair carried a
        # single number written to every row - and they are variable-shaped
        # IncrementalStMan columns, the slowest thing in the whole stage to
        # write (a TaQL UPDATE, the cheapest of the three ways tried, was 31% of
        # it). The number itself is in this evaluation's simulation.json as
        # noise.complex_sigma_jy, which is what ms_to_r2d2_mat.py's
        # --noise-sigma-jy is handed. WSClean weights naturally, so a uniform
        # weight of 1.0 images identically to a uniform 1/sigma^2.

        attached = ms.getkeywords()
        for unused in UNUSED_SUBTABLES:
            if unused in attached:
                ms.removekeyword(unused)
    # After the close, so casacore is never holding a table whose files are gone.
    for unused in UNUSED_SUBTABLES:
        shutil.rmtree(output_ms / unused, ignore_errors=True)

    # The longest projected baseline in wavelengths, which is what both imagers
    # size their pixels from - R2D2 computes it itself from the .mat's u/v (see
    # image_pixel_size_arcsec() in common.py), and the WSClean runner reads this
    # to pass the matching `-scale`. u/v scale linearly with frequency, so the
    # maximum over (row, channel) is the longest baseline at the top channel.
    max_proj_baseline_lambda = float(np.max(np.hypot(uvw[:, 0], uvw[:, 1]))) * float(freqs_hz.max()) / SPEED_OF_LIGHT

    return {
        "measurement_set": str(output_ms),
        "vla_config": args.vla_config,
        "antenna_table_source": str(Cattery_VLA_A),
        "visibility_engine": "analytic point-source RIME (checked against the MeqTrees Meow predict) plus seeded thermal-noise fill",
        "source": {
            "flux_jy": args.source_flux_jy,
            "l_arcsec": args.source_l_arcsec,
            "m_arcsec": args.source_m_arcsec,
        },
        "observation": {
            "max_proj_baseline_lambda": max_proj_baseline_lambda,
            "requested_minutes": args.observation_minutes,
            "integration_seconds": args.integration_seconds,
            "time_samples": n_times_of(args),
            "channel_count": args.channel_count,
            "start_frequency_hz": args.start_frequency_hz,
            "channel_width_hz": args.channel_width_hz,
            "channel_frequencies_hz": freqs_hz.tolist(),
            "declination_deg": args.declination_deg,
        },
        "noise": {
            "dynamic_range": args.dynamic_range,
            "complex_sigma_jy": noise_sigma_jy,
            "seed": args.seed,
        },
    }


def simulate(args: argparse.Namespace) -> None:
    final_ms = Path(args.output_ms)
    require_clean_output(final_ms)
    final_ms.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_json) if args.metadata_json else final_ms.parent / "simulation.json"

    with tempfile.TemporaryDirectory(dir=scratch_root_for(final_ms.parent)) as scratch:
        scratch_ms = Path(scratch) / final_ms.name
        try:
            make_ms(write_makems_config(args, scratch_ms), scratch_ms, args)
            metadata = fill_point_source_visibilities(args, scratch_ms)
        except BaseException:
            # The meqserver's error text is only in these, and the temporary
            # directory is about to take them with it.
            for log in Path(scratch).glob("*.log"):
                shutil.copy2(log, final_ms.parent / log.name)
            raise
        metadata["measurement_set"] = str(final_ms)
        for produced in sorted(Path(scratch).iterdir()):
            destination = final_ms.parent / produced.name
            if destination.is_dir():
                # shutil.move() nests into an existing directory instead of
                # replacing it; the unpacked VLAA_ANT table can already be there.
                shutil.rmtree(destination)
            shutil.move(str(produced), destination)

    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def warm_up() -> None:
    """Build one skeleton before FIFO requests arrive, so makems and casacore
    are warm for the first evaluation."""
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        ms = Path(scratch) / "sim.ms"
        args = parse_args([
            "--output-ms", str(ms), "--observation-minutes", "4.0",
            "--channel-count", "2", "--start-frequency-hz", "1.0e9",
            "--channel-width-hz", "1.0e6", "--dynamic-range", "300",
        ])
        make_ms_skeleton(write_makems_config(args, ms), ms, args)


def handle_request(request: dict) -> None:
    """Run one simulation or conversion request."""
    if request.get("action") == "convert":
        # Imported on first use, not at module scope: only the R2D2 PoC
        # converts, and the WSClean PoC's worker should not pay for scipy.
        from ms_to_r2d2_mat import main as convert

        convert(request["argv"])
    else:
        simulate(parse_args(request["argv"]))


def serve(fifo_base: str | None = None) -> None:
    """Serve JSON requests over stdin/stdout or a shared FIFO pair."""
    # Nothing has been asked of this worker yet, so warming up here overlaps
    # with the caller's own sampler startup. Under redirect_fds because makems
    # prints to fd 1, which is the stdin path's reply pipe.
    if fifo_base is not None:
        with redirect_fds(Path(os.devnull)):
            try:
                warm_up()
            except Exception:
                traceback.print_exc()
    if fifo_base is None:
        requests, replies = sys.stdin, os.fdopen(os.dup(1), "w")
    else:
        # Same order the caller opens them in: opening a FIFO blocks until the
        # other end is opened, so a mismatch here deadlocks both processes.
        requests = open(f"{fifo_base}.in")
        replies = open(f"{fifo_base}.out", "w")
    for line in requests:
        request = json.loads(line)
        returncode = 0
        with redirect_fds(Path(request["stdout"]), Path(request["stderr"])):
            try:
                handle_request(request)
            except Exception:
                traceback.print_exc()
                returncode = 1
            except SystemExit as exc:
                print(exc, file=sys.stderr)
                returncode = exc.code if isinstance(exc.code, int) else 1
        replies.write(json.dumps({"returncode": returncode}) + "\n")
        replies.flush()


def self_check_scratch_root() -> None:
    """Check shared scratch paths avoid an unnecessary cross-tmpfs copy."""
    was = os.environ.get("NS_SCRATCH_DIR")
    with tempfile.TemporaryDirectory() as shared:
        try:
            os.environ["NS_SCRATCH_DIR"] = shared
            inside = Path(shared) / "eval-0001-abc"
            assert scratch_root_for(inside) == str(inside), scratch_root_for(inside)
            outside = Path(shared).parent / "not-the-scratch" / "eval-0001-abc"
            assert scratch_root_for(outside) == SCRATCH_ROOT, scratch_root_for(outside)
            # No shared scratch at all (a self-check, a host with no writable
            # /dev/shm): the container's own /dev/shm, exactly as before.
            del os.environ["NS_SCRATCH_DIR"]
            assert scratch_root_for(inside) == SCRATCH_ROOT
        finally:
            # Restored, not dropped: the checks after this one run in the same
            # process, and in a sidecar the run really does set it.
            os.environ.pop("NS_SCRATCH_DIR", None)
            if was is not None:
                os.environ["NS_SCRATCH_DIR"] = was
    print("OK: scratch_root_for assembles in place inside NS_SCRATCH_DIR")


def self_check_skeleton_cache() -> None:
    def build(ms: Path, start_hz: float, step_hz: float, n_chan: int, minutes: float) -> Path:
        built = parse_args([
            "--output-ms", str(ms), "--observation-minutes", str(minutes),
            "--channel-count", str(n_chan), "--start-frequency-hz", repr(start_hz),
            "--channel-width-hz", repr(step_hz), "--dynamic-range", "300",
        ])
        make_ms_skeleton(write_makems_config(built, ms), ms, built)
        return ms

    for n_chan, minutes, start_hz, step_hz in ((2, 4.0, 1.0374e9, 1.3331e6), (6, 10.0, 1.1e9, 2.0e6)):
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
            use_skeleton_cache(Path(scratch) / "cache")
            build(Path(scratch) / "seed" / "sim.ms", 1.0e9, 1.0e6, n_chan, minutes)
            assert list(skeleton_dir().iterdir()), "the seed build published no cache entry"
            reused = build(Path(scratch) / "reused" / "sim.ms", start_hz, step_hz, n_chan, minutes)
            assert "reused a cached" in (reused.parent / "makems.log").read_text(), "the second build missed the cache"
            use_skeleton_cache(Path(scratch) / "cache-fresh")
            fresh = build(Path(scratch) / "fresh" / "sim.ms", start_hz, step_hz, n_chan, minutes)
            for sub in ("", "SPECTRAL_WINDOW", "ANTENNA", "FIELD", "DATA_DESCRIPTION", "POLARIZATION", "OBSERVATION", "FEED", "POINTING", "PROCESSOR", "STATE"):
                with table(str(reused / sub if sub else reused), readonly=True, ack=False) as left, \
                     table(str(fresh / sub if sub else fresh), readonly=True, ack=False) as right:
                    for column in left.colnames():
                        try:
                            values, expected = np.asarray(left.getcol(column)), np.asarray(right.getcol(column))
                        except RuntimeError:
                            continue  # optional array column left unset by makems
                        assert np.array_equal(values, expected), f"{sub or 'MAIN'}.{column} differs after a cached skeleton reuse"
    use_skeleton_cache(None)
    print("MS skeleton cache self-check passed")


def self_check_observation_prefix() -> None:
    """An MS extended from a cached observation must be the MS makems writes."""
    def values(ms: Path, sub: str) -> dict:
        with table(str(ms / sub if sub else ms), readonly=True, ack=False) as t:
            out = {"__keywords": repr(t.getkeywords()) if sub else None, "__rows": t.nrows()}
            for column in t.colnames():
                try:
                    out[column] = np.asarray(t.getcol(column))
                except RuntimeError:
                    out[column] = None  # optional array column left unset by makems
            return out

    def simulate_with(scratch: Path, name: str, dec: int, step: int, n_times: int, n_chan: int, prefix: bool) -> Path:
        ms = scratch / name / "sim.ms"
        args = parse_args([
            "--output-ms", str(ms), "--observation-minutes", repr(n_times * step / 60.0 - 0.001),
            "--integration-seconds", str(step), "--channel-count", str(n_chan), f"--declination-deg={dec}",
            "--start-frequency-hz", "1.0374e9", "--channel-width-hz", "1.3331e6", "--dynamic-range", "300",
            f"--source-l-arcsec={0.2 * dec}", "--source-m-arcsec=-0.1",
        ])
        cfg = write_makems_config(args, ms)
        if prefix:
            make_ms(cfg, ms, args)
        else:
            run_makems(ms)
        fill_point_source_visibilities(args, ms)
        return ms

    # (declination, integration, NTimes, channels, served from the cache)
    steps = [(34, 2, 10, 1, False), (34, 2, 4, 3, True), (34, 2, 20, 8, False), (34, 2, 15, 2, True),
             (34, 2, 20, 4, True), (-15, 7, 3, 3, False), (-15, 7, 1, 3, True), (75, 2, 5, 1, False)]
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch_dir:
        scratch = Path(scratch_dir)
        use_skeleton_cache(scratch / "cache")
        for i, (dec, step, n_times, n_chan, served) in enumerate(steps):
            got = simulate_with(scratch, f"prefix{i}", dec, step, n_times, n_chan, prefix=True)
            log = (got.parent / "makems.log").read_text()
            assert ("extended a cached observation" in log) == served, f"step {i}: cache use was not {served}: {log}"
            expected = simulate_with(scratch, f"makems{i}", dec, step, n_times, n_chan, prefix=False)
            with table(str(got), ack=False) as left, table(str(expected), ack=False) as right:
                assert repr(left.getdminfo()) == repr(right.getdminfo()), f"step {i}: storage layout differs"
            for sub in ("", "SPECTRAL_WINDOW", "ANTENNA", "FIELD", "DATA_DESCRIPTION", "POLARIZATION", "OBSERVATION"):
                left, right = values(got, sub), values(expected, sub)
                assert left.keys() == right.keys(), f"step {i}: {sub or 'MAIN'} columns differ"
                for column in left:
                    same = (left[column] is None and right[column] is None) if left[column] is None or right[column] is None \
                        else np.array_equal(left[column], right[column])
                    assert same, f"step {i}: {sub or 'MAIN'}.{column} differs from a makems build"
    use_skeleton_cache(None)
    print("observation prefix self-check passed")


def self_check_skeleton_prebuild() -> None:
    """A prebuilt shape must be the entry a real evaluation of it looks up."""
    space = {"observation_minutes": [4.0, 6.0], "channel_count": [2, 3]}
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        use_skeleton_cache(Path(scratch) / "cache")
        prebuild_skeletons(space)
        # Shapes are (NTimes, NFrequencies) = (2,2) (2,3) (3,2) (3,3).
        built = list(skeleton_dir().iterdir())
        assert len(built) == 4, f"prebuild published {len(built)} entries, expected 4"
        ms = Path(scratch) / "hit" / "sim.ms"
        args = parse_args([
            "--output-ms", str(ms), "--observation-minutes", "4.0", "--channel-count", "2",
            "--start-frequency-hz", "1.0374e9", "--channel-width-hz", "1.3331e6",
            "--dynamic-range", "300",
        ])
        make_ms_skeleton(write_makems_config(args, ms), ms, args)
        assert "reused a cached" in (ms.parent / "makems.log").read_text(), \
            "a prebuilt shape was not reused by a real evaluation of it"
    use_skeleton_cache(None)
    print("MS skeleton prebuild self-check passed")


def self_check_declination_config() -> None:
    """Declination must reach makems, and must be part of the skeleton cache key."""
    assert makems_declination(DEFAULT_DECLINATION_DEG) == "65.0.0", makems_declination(DEFAULT_DECLINATION_DEG)
    assert makems_declination(-30.0) == "-30.0.0", makems_declination(-30.0)
    assert makems_declination(-0.5) == "-0.30.0", makems_declination(-0.5)
    assert makems_declination(20.505) == "20.30.18", makems_declination(20.505)

    def cfg_text(dec: str | None) -> str:
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
            ms = Path(scratch) / "sim.ms"
            argv = ["--output-ms", str(ms), "--observation-minutes", "4.0", "--channel-count", "2",
                    "--start-frequency-hz", "1.0e9", "--channel-width-hz", "1.0e6", "--dynamic-range", "300"]
            return write_makems_config(parse_args(argv + (["--declination-deg", dec] if dec else [])), ms).read_text()

    assert "Declination=65.0.0" in cfg_text(None), "the default declination no longer reaches makems"
    assert "Declination=-30.0.0" in cfg_text("-30"), "--declination-deg does not reach makems"
    # make_ms_skeleton() keys the cache on the config minus StartFreq/StepFreq,
    # so a declination that did not show up there would silently reuse +65.
    def cache_key(text: str) -> list[str]:
        return [line for line in text.splitlines() if not line.startswith(("StartFreq=", "StepFreq="))]

    assert cache_key(cfg_text(None)) != cache_key(cfg_text("-30")), "declination is outside the skeleton cache key"
    print("declination config self-check passed")


def self_check_forest_reuse() -> None:
    """Reusing a compiled forest must predict what a fresh compile predicts."""
    shapes = ((2, 4.0), (6, 10.0), (3, 8.0))

    def build(scratch: Path, name: str, n_chan: int, minutes: float) -> np.ndarray:
        ms = scratch / name / "sim.ms"
        built = parse_args([
            "--output-ms", str(ms), "--observation-minutes", str(minutes),
            "--channel-count", str(n_chan), "--start-frequency-hz", "1.0e9",
            "--channel-width-hz", "1.0e6", "--dynamic-range", "300",
        ])
        make_ms_skeleton(write_makems_config(built, ms), ms, built)
        corr_sel, _ = determine_corr_selection(ms)
        run_meqtrees_predict(ms, corr_sel, 1.0, 0.0, 0.0)
        with table(str(ms), readonly=True, ack=False) as opened:
            return np.asarray(opened.getcol("DATA"))

    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        # One compile up front, then every shape runs off the cached forest.
        _FOREST.clear()
        reused = [build(Path(scratch), f"reused-{i}", *shape) for i, shape in enumerate(shapes)]
        assert len(_FOREST) == 1, "the forest was recompiled for an identical source setup"
        fresh = []
        for i, shape in enumerate(shapes):
            _FOREST.clear()
            fresh.append(build(Path(scratch), f"fresh-{i}", *shape))
        for shape, values, expected in zip(shapes, reused, fresh):
            assert np.array_equal(values, expected), f"DATA differs after a forest reuse at {shape}"
    _FOREST.clear()
    print("forest reuse self-check passed")


def self_check_analytic_predict() -> None:
    """point_source_visibilities() must be what MeqTrees predicts, on and off
    the phase centre, across the searched shapes and declinations."""
    cases = (
        # n_chan, minutes, integration s, declination, start Hz, width Hz, flux, l", m"
        (1, 0.3, 120.0, 65.0, 5.4e10, 2.0e6, 1.0, 0.0, 0.0),
        (8, 20.0, 120.0, 65.0, 5.4e7, 0.1e6, 1.0, 0.0, 0.0),
        (5, 7.3, 120.0, 65.0, 1.4e9, 1.1e6, 2.5, 0.0, 0.0),
        (2, 9.6, 3.0, 1.0, 4.36e10, 6.4e5, 1.0, -0.152, -0.193),
        (6, 13.5, 6.0, 37.0, 4.27e10, 1.04e6, 1.0, 0.190, 0.212),
        (4, 30.0, 1.0, -15.0, 9.6e9, 1.4e6, 1.0, 0.212, -0.491),
        (3, 5.0, 10.0, 75.0, 1.1e9, 2.0e6, 1.0, -3.0, 5.0),
    )
    worst = 0.0
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        for index, (n_chan, minutes, step, dec, start_hz, width_hz, flux, l_as, m_as) in enumerate(cases):
            ms = Path(scratch) / f"predict-{index}" / "sim.ms"
            built = parse_args([
                "--output-ms", str(ms), "--observation-minutes", str(minutes),
                "--integration-seconds", str(step), f"--declination-deg={dec}",
                "--channel-count", str(n_chan), "--start-frequency-hz", repr(start_hz),
                "--channel-width-hz", repr(width_hz), "--dynamic-range", "300",
                "--source-flux-jy", repr(flux),
            ])
            make_ms_skeleton(write_makems_config(built, ms), ms, built)
            corr_sel, n_corr = determine_corr_selection(ms)
            l_rad, m_rad = math.radians(l_as / 3600.0), math.radians(m_as / 3600.0)
            run_meqtrees_predict(ms, corr_sel, flux, l_rad, m_rad, wait_seconds=300.0)
            with table(str(ms), readonly=True, ack=False) as opened:
                predicted = np.asarray(opened.getcol("DATA"))
                uvw = opened.getcol("UVW")
            with table(str(ms / "SPECTRAL_WINDOW"), readonly=True, ack=False) as spw:
                freqs_hz = spw.getcol("CHAN_FREQ")[0]
            analytic = point_source_visibilities(uvw, freqs_hz, flux, l_rad, m_rad, n_corr)
            if not (l_as or m_as):
                assert np.array_equal(predicted, analytic), f"phase-centre constant differs from MeqTrees in case {index}"
            # One float32 ulp at |V| = flux, per component: rounding, not physics.
            error = float(np.abs(predicted - analytic).max()) / flux
            worst = max(worst, error)
            assert error < 2.5e-7, f"analytic predict differs from MeqTrees by {error:.3g} in case {index}"
    _FOREST.clear()
    print(f"analytic predict self-check passed (worst |MeqTrees - analytic| / flux = {worst:.3g})")


def self_check_noise_weighting() -> None:
    """Check untouched WEIGHT and equivalent nW conversion from simulation sigma."""
    from ms_to_r2d2_mat import ms_to_r2d2_mat
    from scipy.io import loadmat

    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        for index, dynamic_range in enumerate((10.0, 1.0e6)):
            ms = Path(scratch) / f"weights-{index}" / "sim.ms"
            metadata_path = ms.parent / "simulation.json"
            # simulate() prints the whole metadata document, which is the
            # contract its callers rely on and only noise here.
            with contextlib.redirect_stdout(io.StringIO()):
                simulate(parse_args([
                    "--output-ms", str(ms), "--metadata-json", str(metadata_path),
                    "--observation-minutes", "4.0", "--channel-count", "2",
                    "--start-frequency-hz", "1.0e9", "--channel-width-hz", "1.0e6",
                    "--source-flux-jy", "1.0", "--dynamic-range", repr(dynamic_range),
                    "--seed", "42",
                ]))
            with table(str(ms), readonly=True, ack=False) as opened:
                weight = np.asarray(opened.getcol("WEIGHT"), dtype=np.float64)
            assert np.array_equal(weight, np.ones_like(weight)), \
                f"makems no longer leaves WEIGHT at 1.0 (dynamic range {dynamic_range})"
            sigma = json.loads(metadata_path.read_text())["noise"]["complex_sigma_jy"]
            mat = ms.parent / "r2d2_data.mat"
            ms_to_r2d2_mat(ms, mat, noise_sigma_jy=sigma)
            nW = np.asarray(loadmat(str(mat))["nW"], dtype=np.float64)
            expected = np.sqrt(1.0 / (sigma * sigma))
            assert np.allclose(nW, expected, rtol=0.0, atol=0.0), \
                f"nW is {nW.min()}..{nW.max()}, not the {expected} the WEIGHT column used to carry"
    print("noise weighting self-check passed")


def self_check_dropped_subtables() -> None:
    """Finished MS must retain required subtables on both predict paths."""
    kept = ("ANTENNA", "DATA_DESCRIPTION", "FIELD", "OBSERVATION",
            "POLARIZATION", "SPECTRAL_WINDOW")
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        for index, (l_arcsec, m_arcsec) in enumerate(((0.0, 0.0), (5.0, 3.0))):
            ms = Path(scratch) / f"subtables-{index}" / "sim.ms"
            with contextlib.redirect_stdout(io.StringIO()):
                simulate(parse_args([
                    "--output-ms", str(ms), "--observation-minutes", "4.0",
                    "--channel-count", "2", "--start-frequency-hz", "1.0e9",
                    "--channel-width-hz", "1.0e6", "--source-flux-jy", "1.0",
                    "--source-l-arcsec", repr(l_arcsec), "--source-m-arcsec", repr(m_arcsec),
                    "--dynamic-range", "300", "--seed", "42",
                ]))
            with table(str(ms), readonly=True, ack=False) as opened:
                keywords = opened.getkeywords()
                rows = opened.nrows()
                columns = opened.colnames()
                data = np.asarray(opened.getcol("DATA"))
            assert rows, f"{ms} came out empty"
            # The phase-centre path fills an uninitialised array from a one-row
            # shape probe rather than reading DATA back, so a wrong shape or a
            # missed fill would leave whatever malloc returned in the column.
            assert data.shape[:2] == (rows, 2), f"{ms} DATA is {data.shape}, not ({rows}, 2, ...)"
            if not (l_arcsec or m_arcsec):
                assert abs(complex(data[..., 0].mean()) - 1.0) < 0.05, \
                    f"{ms} DATA does not average the 1 Jy source: {data[..., 0].mean()}"
            # polychord_wsclean.py passes `-data-column DATA` rather than let
            # WSClean open the whole measurement set to decide - which is only
            # the same answer while the simulator writes no CORRECTED_DATA.
            assert "DATA" in columns, f"{ms} has no DATA column"
            assert "CORRECTED_DATA" not in columns, (
                f"{ms} has a CORRECTED_DATA column, so WSClean would image that "
                "one - drop the `-data-column DATA` in polychord_wsclean.py"
            )
            for name in UNUSED_SUBTABLES:
                assert name not in keywords, f"{name} is still a keyword of {ms}"
                assert not (ms / name).exists(), f"{name} is still on disk in {ms}"
            for name in kept:
                assert name in keywords and (ms / name).is_dir(), f"{name} went missing from {ms}"
    _FOREST.clear()
    print("dropped subtable self-check passed")


def self_check_meqserver_restart() -> None:
    """Check one meqserver restart is retried and a second wedge kills worker."""
    original_predict = globals()["_compile_and_predict"]
    original_restart = globals()["restart_meqserver_session"]
    calls: list[str] = []

    def restart() -> None:
        calls.append("restart")

    def predict(wedges: int):
        remaining = [wedges]

        def run(*_args: object, **_kwargs: object) -> list:
            calls.append("predict")
            if remaining[0]:
                remaining[0] -= 1
                raise MeqserverWedged("no reply to the predict")
            return []

        return run

    scratch = tempfile.TemporaryDirectory()
    ms = Path(scratch.name) / "sim.ms"
    try:
        globals()["restart_meqserver_session"] = restart

        # One wedge: restart, retry, and the caller sees an ordinary predict.
        calls.clear()
        globals()["_compile_and_predict"] = predict(wedges=1)
        run_meqtrees_predict(ms, "2x2", 1.0, 0.0, 0.0)
        assert calls == ["predict", "restart", "predict"], calls
        # The only record that this evaluation cost seconds rather than
        # milliseconds - meqtree-pipeliner.log cannot hold it, because the
        # retry reopens that file and truncates whatever the wedge wrote.
        note = ms.parent / "meqserver-wedged.log"
        assert note.exists() and "attempt 1" in note.read_text(), note
        note.unlink()

        # No wedge: nothing is restarted and the predict runs once.
        calls.clear()
        globals()["_compile_and_predict"] = predict(wedges=0)
        run_meqtrees_predict(ms, "2x2", 1.0, 0.0, 0.0)
        assert calls == ["predict"], calls
        assert not (ms.parent / "meqserver-wedged.log").exists()

        # Two in a row: raised, not swallowed and not turned into an exit
        # status, so serve() can kill the worker and let the rank retry it.
        calls.clear()
        globals()["_compile_and_predict"] = predict(wedges=2)
        try:
            run_meqtrees_predict(ms, "2x2", 1.0, 0.0, 0.0)
        except MeqserverWedged:
            pass
        else:
            raise AssertionError("a second wedge in a row must reach the caller")
        assert calls == ["predict", "restart", "predict"], calls
    finally:
        scratch.cleanup()
        globals()["_compile_and_predict"] = original_predict
        globals()["restart_meqserver_session"] = original_restart

    print("meqserver restart self-check passed")


def self_check_predict_timeout_recovery() -> None:
    """Verify bounded predict timeout, meqserver replacement, and recovery."""
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        ms = Path(scratch) / "sim.ms"
        args = parse_args([
            "--output-ms", str(ms), "--observation-minutes", "4.0",
            "--channel-count", "2", "--start-frequency-hz", "1.0e9",
            "--channel-width-hz", "1.0e6", "--dynamic-range", "300",
        ])
        make_ms_skeleton(write_makems_config(args, ms), ms, args)
        corr_sel, _ = determine_corr_selection(ms)

        with redirect_fds(Path(os.devnull)):
            meqserver_session()
        first_pid = _MQS.serv_pid

        # A bound nothing can meet, so both attempts expire: the caller has to
        # get control back rather than block, and it has to arrive as
        # MeqserverWedged so serve() can tell it apart from a failed evaluation.
        original_bound = PREDICT_WAIT_SECONDS
        globals()["PREDICT_WAIT_SECONDS"] = 0.001
        started = time.monotonic()
        try:
            run_meqtrees_predict(ms, corr_sel, 1.0, 0.0, 0.0)
        except MeqserverWedged:
            pass
        else:
            raise AssertionError("a predict that never answers must raise MeqserverWedged")
        finally:
            globals()["PREDICT_WAIT_SECONDS"] = original_bound
        bounded = time.monotonic() - started
        assert bounded < 60.0, f"a numeric wait did not bound the predict: {bounded:.1f}s"
        assert (ms.parent / "meqserver-wedged.log").exists(), "a wedge left no record"

        # The server was replaced along the way, and the session it left behind
        # still produces real visibilities rather than a half-dead forest.
        assert _MQS is None or _MQS.serv_pid != first_pid, "the wedged meqserver was not replaced"
        with redirect_fds(Path(os.devnull)):
            run_meqtrees_predict(ms, corr_sel, 1.0, 0.0, 0.0)
        data = table(str(ms), ack=False).getcol("DATA")
        assert abs(data[0, 0, 0] - 1.0) < 1e-6, f"XX after recovery is {data[0, 0, 0]}"
        assert abs(data[0, 0, -1] - 1.0) < 1e-6, f"YY after recovery is {data[0, 0, -1]}"
    print("predict timeout recovery self-check passed")


def self_check_serve_reply_stream() -> None:
    """A worker's stdout must carry replies only, never meqserver startup chatter."""
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        worker = subprocess.Popen(
            [sys.executable, __file__, "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        request = {
            "argv": ["--not-a-real-option"],
            "stdout": str(Path(scratch) / "out.log"),
            "stderr": str(Path(scratch) / "err.log"),
        }
        stdout, _ = worker.communicate(json.dumps(request) + "\n", timeout=120)
    lines = stdout.splitlines()
    assert len(lines) == 1, f"worker stdout was not one reply line: {lines!r}"
    assert json.loads(lines[0])["returncode"] != 0, "a bogus request should report failure"
    print("serve reply stream self-check passed")


def self_check_serve_fifo() -> None:
    """Check that a `--fifo` worker answers without deadlocking."""
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        base = Path(scratch) / "0"
        os.mkfifo(f"{base}.in")
        os.mkfifo(f"{base}.out")
        worker = subprocess.Popen([sys.executable, __file__, "--serve", "--fifo", str(base)])
        deadline = time.monotonic() + 120.0
        while True:
            try:
                write_fd = os.open(f"{base}.in", os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError:
                assert time.monotonic() < deadline, "the --fifo worker never opened its request pipe"
                time.sleep(0.01)
        os.set_blocking(write_fd, True)
        with os.fdopen(write_fd, "w") as requests, open(f"{base}.out") as replies:
            request = {
                "argv": ["--not-a-real-option"],
                "stdout": str(Path(scratch) / "out.log"),
                "stderr": str(Path(scratch) / "err.log"),
            }
            requests.write(json.dumps(request) + "\n")
            requests.flush()
            reply = replies.readline()
        assert json.loads(reply)["returncode"] != 0, f"a bogus request should report failure: {reply!r}"
        # Closing the request pipe is the only shutdown signal the worker gets.
        assert worker.wait(timeout=120) == 0, "the --fifo worker did not exit on EOF"
    print("serve fifo self-check passed")


if __name__ == "__main__":
    # --serve and --self-check take no other arguments, so they are checked before
    # argparse, which requires the full simulate argument set.
    try:
        if sys.argv[1:] == ["--prebuild-skeletons"]:
            # Build time only: fills BAKED_SKELETON_DIR from the one authoritative
            # copy of the parameter space, bind-mounted in for this step so the
            # runtime image still carries nothing but the three simulate scripts.
            from common import load_parameter_space

            use_skeleton_cache(BAKED_SKELETON_DIR)
            prebuild_skeletons({spec["name"]: [spec["min"], spec["max"]] for spec in load_parameter_space()})
        elif sys.argv[1:2] == ["--serve"]:
            # `--serve` / `--serve --fifo <base>`; neither takes the simulate
            # argument set, so they are dispatched before argparse.
            serve(sys.argv[3] if sys.argv[2:3] == ["--fifo"] else None)
        elif sys.argv[1:] == ["--self-check"]:
            self_check_scratch_root()
            self_check_skeleton_cache()
            self_check_skeleton_prebuild()
            self_check_observation_prefix()
            self_check_declination_config()
            self_check_forest_reuse()
            self_check_analytic_predict()
            self_check_noise_weighting()
            self_check_dropped_subtables()
            self_check_meqserver_restart()
            self_check_predict_timeout_recovery()
            self_check_serve_reply_stream()
            self_check_serve_fifo()
        else:
            simulate(parse_args())
    finally:
        stop_meqserver_session()
