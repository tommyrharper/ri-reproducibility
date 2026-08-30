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
    parser.add_argument("--dynamic-range", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def require_clean_output(output_ms: Path) -> None:
    if output_ms.exists():
        raise SystemExit(f"FATAL: output Measurement Set already exists: {output_ms}")
    for suffix in ("_p0", "_p1", ".gds"):
        candidate = output_ms.parent / f"{output_ms.name}{suffix}"
        if candidate.exists():
            raise SystemExit(f"FATAL: makems scratch output already exists: {candidate}")


def write_makems_config(args: argparse.Namespace, output_ms: Path) -> Path:
    output_ms.parent.mkdir(parents=True, exist_ok=True)
    antenna_dst = output_ms.parent / ANTENNA_TABLE_NAME
    if not antenna_dst.exists():
        shutil.unpack_archive(Cattery_VLA_A, output_ms.parent)

    n_times = max(1, int(math.ceil(args.observation_minutes * 60.0 / args.integration_seconds)))
    cfg = output_ms.parent / "makems.cfg"
    cfg.write_text(
        "\n".join(
            [
                f"StartFreq={args.start_frequency_hz:.8f}",
                f"StepFreq={args.channel_width_hz:.8f}",
                "StartTime=2005/02/09/21:21:40",
                f"StepTime={args.integration_seconds:.8f}",
                "RightAscension=0:0:0",
                "Declination=65.0.0",
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
# ponytail: no eviction - the sidecar ends with the run and the full parameter
# space is ~20MB. Add an LRU sweep if a longer-lived container reuses one.
_SKELETON_DIR: Path | None = None

# `--prebuild-skeletons` puts default-run shapes here at image build time;
# see docker/meqtrees/Dockerfile.
BAKED_SKELETON_DIR = Path("/opt/ms-skeletons")


def skeleton_dir() -> Path:
    global _SKELETON_DIR
    if _SKELETON_DIR is None:
        # Baked shapes are a head start; unseen shapes are built and published.
        _SKELETON_DIR = BAKED_SKELETON_DIR if BAKED_SKELETON_DIR.is_dir() else Path(SCRATCH_ROOT or tempfile.gettempdir()) / "ms-skeletons"
        _SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    return _SKELETON_DIR


def use_skeleton_cache(directory: Path | None) -> None:
    global _SKELETON_DIR
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    _SKELETON_DIR = directory


def publish_skeleton(built_ms: Path, cached: Path) -> None:
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


def make_ms_skeleton(cfg: Path, output_ms: Path, args: argparse.Namespace, prune_unused: bool = False) -> None:
    key = "\n".join(line for line in cfg.read_text().splitlines() if not line.startswith(("StartFreq=", "StepFreq=")))
    cached = skeleton_dir() / hashlib.sha256(key.encode()).hexdigest()[:32]
    if not cached.exists():
        run_makems(output_ms)
        publish_skeleton(output_ms, cached)
        return
    ignore = shutil.ignore_patterns(*UNUSED_SUBTABLES) if prune_unused else None
    shutil.copytree(cached, output_ms, symlinks=True, ignore=ignore)
    patch_spectral_window(output_ms, args.start_frequency_hz, args.channel_width_hz)
    (output_ms.parent / "makems.log").write_text(f"reused a cached makems skeleton for:\n{key}\n")


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


# Bound MeqTrees deadlocks so worker can restart its server; 3s is ~30x the
# slowest recorded simulate and avoids blocking PolyChord's collective.
PREDICT_WAIT_SECONDS = 3.0


class MeqserverWedged(RuntimeError):
    """Predict exceeded PREDICT_WAIT_SECONDS; restart server and retry."""


def restart_meqserver_session() -> None:
    """Kill and reap wedged meqserver, leaving worker alive for retry."""
    global _MQS
    from Timba.Apps import meqserver

    pid = getattr(_MQS, "serv_pid", None)
    _MQS = None
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


def run_meqtrees_predict(output_ms: Path, corr_sel: str, source_flux_jy: float, l_rad: float, m_rad: float) -> None:
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
    for attempt in range(2):
        try:
            errors = _compile_and_predict(tdlconf, key, output_ms)
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
    if errors:
        raise SystemExit(f"FATAL: meqserver reported {len(errors)} error(s) during the predict")


def _compile_and_predict(tdlconf: Path, key: str, output_ms: Path) -> list:
    """Compile and run bounded predict; raise MeqserverWedged on timeout."""
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
        else:
            point_to_measurement_set(module, output_ms)
            print("### reusing the compiled forest; only the Measurement Set changed")
        try:
            TDLOptions.get_job_func("predict")(mqs, None, wait=PREDICT_WAIT_SECONDS)
        except AttributeError as exc:
            # Timba's meq() ends in `return msg.payload`, and msg is None when
            # the wait expires, so a timeout arrives here as an AttributeError
            # off that line rather than as a return value. Anything else of the
            # same type is a real bug and is left to propagate.
            if "NoneType" not in str(exc):
                raise
            raise MeqserverWedged(f"no reply to the predict in {PREDICT_WAIT_SECONDS}s") from exc
        # get_error_log() flushes, so each request only sees its own errors.
        errors = mqs.get_error_log()
        for index, (_event, error) in enumerate(errors):
            # !r, not str(): Timba's DMI record __str__ is still py2
            # (`string.join`) and raises AttributeError, which would replace the
            # meqserver's error with a traceback from the reporting path itself.
            print(f"###   {index:03d}: {error!r}")
    return errors


def phase_centre_visibility(source_flux_jy: float, n_corr: int) -> np.ndarray:
    """Return constant Stokes-I visibilities for a phase-centre source."""
    model = np.zeros(n_corr, dtype=np.complex64)
    model[0] = model[-1] = source_flux_jy
    return model


# makems writes the full MSv2 subtable set, and casacore attaches every subtable
# on every open of the parent table - which WSClean does once per gridding and
# degridding pass, ~16 times an evaluation, at ~0.21ms a subtable per open.
# These six are empty (FLAG_CMD, HISTORY) or carry nothing a single-field
# unpolarised point-source simulation depends on, so they are dropped once the
# visibilities are written - not in the cached skeleton, because casacore
# refuses to open an MS that is missing any of them and the MeqTrees predict
# needs it opened that way. An evaluation that runs no predict never copies them
# out of the skeleton at all (make_ms_skeleton()'s `prune_unused`); the delete
# below is what covers the one that does, and stays unconditional because it
# already tolerates them being absent. Worth -13.8% on the wsclean binary and +14.9%
# evaluations per second for the first five, and FEED another ~3%, with
# bit-identical images; see docs/nested-sampling-ms-open.md. The six that
# stay were each tried and each kills WSClean, so this list is complete.
UNUSED_SUBTABLES = ("FEED", "FLAG_CMD", "HISTORY", "POINTING", "PROCESSOR", "STATE")


def meqtrees_predict_needed(args: argparse.Namespace) -> bool:
    return bool(args.source_l_arcsec or args.source_m_arcsec)


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
    # A source at the phase centre predicts a constant, so the meqserver is
    # not asked for it - see phase_centre_visibility().
    predicted = meqtrees_predict_needed(args)
    if predicted:
        corr_sel, n_corr = determine_corr_selection(output_ms)
        run_meqtrees_predict(output_ms, corr_sel, args.source_flux_jy, l_rad, m_rad)

    rng = np.random.default_rng(args.seed)
    with table(str(output_ms), readonly=False, ack=False) as ms:
        if predicted:
            data = np.asarray(ms.getcol("DATA"), dtype=np.complex64)
            _, n_chan, data_n_corr = data.shape
            if data_n_corr != n_corr:
                raise SystemExit(f"FATAL: DATA correlation count changed from {n_corr} to {data_n_corr} after MeqTrees predict")
        else:
            # Every value in DATA is about to be overwritten with the same
            # constant, so the skeleton's zeros are not read back: only the
            # shape is wanted, and one row of it costs 0.05ms against 0.38ms
            # for the whole column. determine_corr_selection()'s separate open
            # of the same table goes with it (another 0.43ms) - its correlation
            # count comes from this probe, and its `corr_sel` string is only
            # ever handed to a predict this path does not run.
            _, n_chan, n_corr = ms.getcol("DATA", startrow=0, nrow=1).shape
            data = np.empty((ms.nrows(), n_chan, n_corr), dtype=np.complex64)
            data[:] = phase_centre_visibility(args.source_flux_jy, n_corr)
        uvw = np.asarray(ms.getcol("UVW"), dtype=np.float64)
        if n_chan != len(freqs_hz):
            raise SystemExit(f"FATAL: DATA has {n_chan} channels, SPW has {len(freqs_hz)}")

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
        "visibility_engine": "MeqTrees Meow point-source RIME predict (meqtree-pipeliner.py) plus seeded thermal-noise fill",
        "source": {
            "flux_jy": args.source_flux_jy,
            "l_arcsec": args.source_l_arcsec,
            "m_arcsec": args.source_m_arcsec,
        },
        "observation": {
            "max_proj_baseline_lambda": max_proj_baseline_lambda,
            "requested_minutes": args.observation_minutes,
            "integration_seconds": args.integration_seconds,
            "time_samples": max(1, int(math.ceil(args.observation_minutes * 60.0 / args.integration_seconds))),
            "channel_count": args.channel_count,
            "start_frequency_hz": args.start_frequency_hz,
            "channel_width_hz": args.channel_width_hz,
            "channel_frequencies_hz": freqs_hz.tolist(),
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
        cfg = write_makems_config(args, scratch_ms)
        make_ms_skeleton(cfg, scratch_ms, args, prune_unused=not meqtrees_predict_needed(args))
        metadata = fill_point_source_visibilities(args, scratch_ms)
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


def warm_forest() -> None:
    """Prewarm forest and first predict before FIFO requests arrive."""
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        ms = Path(scratch) / "sim.ms"
        args = parse_args([
            "--output-ms", str(ms), "--observation-minutes", "4.0",
            "--channel-count", "2", "--start-frequency-hz", "1.0e9",
            "--channel-width-hz", "1.0e6", "--dynamic-range", "300",
        ])
        make_ms_skeleton(write_makems_config(args, ms), ms, args)
        corr_sel, _ = determine_corr_selection(ms)
        run_meqtrees_predict(ms, corr_sel, 1.0, 0.0, 0.0)


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
    # meqserver_session() is otherwise first called inside request one, so every
    # rank paid ~0.3s of Timba import and meqserver startup in front of its first
    # evaluation - and because PolyChord asks all ranks for their initial live
    # points at once, all of it landed on the wall clock. Nothing has been asked
    # of this worker yet, so starting the server here instead overlaps it with
    # the caller's own sampler startup. Under redirect_fds because Timba prints
    # to fd 1 on startup, which is the stdin path's reply pipe.
    with redirect_fds(Path(os.devnull)):
        try:
            meqserver_session()
            if fifo_base is not None:
                warm_forest()
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
            except MeqserverWedged:
                # Past what this worker can fix - it has already replaced its
                # meqserver once for this request. Die instead of answering, so
                # the rank sees the worker death it already knows how to retry
                # (common.py's WORKER_DIED) rather than an exit status, which it
                # would score as a failure of these parameters. The parameters
                # did nothing wrong, and a host fault the sampler chases is the
                # one outcome FAILURE_OBJECTIVE must never be spent on.
                #
                # os._exit() because Timba's octopussy event thread is not a
                # daemon, so a normal exit blocks joining it forever - the same
                # trap stop_meqserver_session() exists to avoid, and it cannot
                # help here because it needs a server that answers.
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
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


def self_check_phase_centre_predict() -> None:
    """The constant phase_centre_visibility() writes must be what MeqTrees predicts."""
    cases = ((1, 0.3, 5.4e10, 2.0e6, 1.0), (8, 20.0, 5.4e7, 0.1e6, 1.0), (5, 7.3, 1.4e9, 1.1e6, 2.5))
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        for index, (n_chan, minutes, start_hz, width_hz, flux) in enumerate(cases):
            ms = Path(scratch) / f"phase-centre-{index}" / "sim.ms"
            built = parse_args([
                "--output-ms", str(ms), "--observation-minutes", str(minutes),
                "--channel-count", str(n_chan), "--start-frequency-hz", repr(start_hz),
                "--channel-width-hz", repr(width_hz), "--dynamic-range", "300",
                "--source-flux-jy", repr(flux),
            ])
            make_ms_skeleton(write_makems_config(built, ms), ms, built)
            corr_sel, n_corr = determine_corr_selection(ms)
            run_meqtrees_predict(ms, corr_sel, flux, 0.0, 0.0)
            with table(str(ms), readonly=True, ack=False) as opened:
                predicted = np.asarray(opened.getcol("DATA"))
            expected = np.broadcast_to(phase_centre_visibility(flux, n_corr), predicted.shape)
            assert np.array_equal(predicted, expected), \
                f"MeqTrees predicts more than the phase-centre constant at {(n_chan, minutes, start_hz, width_hz, flux)}"
    _FOREST.clear()
    print("phase-centre predict self-check passed")


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


def self_check_wedge_kills_worker() -> None:
    """A wedged worker must die as WORKER_DIED, not answer with a status."""
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        ms = Path(scratch) / "sim.ms"
        # A worker whose predicts can never succeed, without a knob in the
        # production path: the bound is set in the child before serve() runs.
        bootstrap = (
            f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
            "import simulate_point_source_ms as s; s.PREDICT_WAIT_SECONDS = 0.001; s.serve()"
        )
        worker = subprocess.Popen(
            [sys.executable, "-c", bootstrap],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        request = {
            # Offset, so the request actually reaches the meqserver: a source at
            # the phase centre never asks it for anything (see
            # phase_centre_visibility()) and a wedged worker would answer it.
            "argv": [
                "--output-ms", str(ms), "--observation-minutes", "4.0",
                "--channel-count", "2", "--start-frequency-hz", "1.0e9",
                "--channel-width-hz", "1.0e6", "--dynamic-range", "300",
                "--source-l-arcsec", "0.5",
            ],
            "stdout": str(Path(scratch) / "out.log"),
            "stderr": str(Path(scratch) / "err.log"),
        }
        worker.stdin.write(json.dumps(request) + "\n")
        worker.stdin.flush()
        reply = worker.stdout.readline()
        assert reply == "", f"a wedged worker answered instead of dying: {reply!r}"
        assert worker.wait(timeout=300) != 0, "a wedged worker must not exit successfully"
    print("wedge kills worker self-check passed")


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

            BAKED_SKELETON_DIR.mkdir(parents=True, exist_ok=True)
            prebuild_skeletons({spec["name"]: [spec["min"], spec["max"]] for spec in load_parameter_space()})
        elif sys.argv[1:2] == ["--serve"]:
            # `--serve` / `--serve --fifo <base>`; neither takes the simulate
            # argument set, so they are dispatched before argparse.
            serve(sys.argv[3] if sys.argv[2:3] == ["--fifo"] else None)
        elif sys.argv[1:] == ["--self-check"]:
            self_check_scratch_root()
            self_check_skeleton_cache()
            self_check_skeleton_prebuild()
            self_check_forest_reuse()
            self_check_phase_centre_predict()
            self_check_noise_weighting()
            self_check_dropped_subtables()
            self_check_meqserver_restart()
            self_check_predict_timeout_recovery()
            self_check_wedge_kills_worker()
            self_check_serve_reply_stream()
            self_check_serve_fifo()
        else:
            simulate(parse_args())
    finally:
        stop_meqserver_session()
