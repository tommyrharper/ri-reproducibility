#!/usr/bin/env python3
"""Create a VLA Measurement Set for a noisy single point source.

The MS skeleton and VLA.A antenna table come from makems' bundled VLA-A
example (shipped by the `makems` KERN package). Visibilities are predicted by
an actual MeqTrees/Meow point-source RIME run (see point_source_forest.py),
driven non-interactively through meqtree-pipeliner.py; thermal noise is then
added on top of that clean MeqTrees prediction.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from casacore.tables import table, taql


Cattery_VLA_A = Path("/usr/share/doc/makems/VLAA_ANT.tar.gz")
ANTENNA_TABLE_NAME = "VLAA_ANT"
TDL_SCRIPT = Path("/opt/ri-nested-sampling/point_source_forest.py")
# The MS time grid step. Shared with prebuild_skeletons(), which has to
# enumerate the same NTimes values a run's evaluations will ask for.
DEFAULT_INTEGRATION_SECONDS = 120.0
SPEED_OF_LIGHT = 299792458.0

# makems and casacore fsync on nearly every table write. On the bind-mounted
# repo that is ~0.5s of ext4 journal wait per run (makems alone: 0.54s on the
# bind mount vs 0.05s here), so build everything in RAM and copy the finished
# artifacts out once at the end - ~2ms for a ~1MB Measurement Set.
# ponytail: Docker gives /dev/shm 64MB by default, which is ~30x this run's
# largest MS; raise --shm-size on the sidecar if the parameter space grows.
SCRATCH_ROOT = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None


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


# makems' output depends on every config field except StartFreq/StepFreq, which
# only move six SPECTRAL_WINDOW columns. Copying a cached skeleton and rewriting
# those columns costs ~0.002s against ~0.05s for a makems run, and reproduces a
# fresh run's tables exactly (see --self-check). The parameter space has 20
# (NTimes, NFrequencies) shapes, so a long-lived --serve worker hits this cache
# for most of its evaluations.
#
# The cache lives on disk, not in this process, because all of the run's ranks
# `docker exec` their --serve worker into one shared meqtrees sidecar: they see
# the same /dev/shm, so a shape any rank has built is a copy away for all the
# others. A run only visits ~12 distinct shapes but does ~41 evaluations, so
# per-process caches missed on most of them.
# ponytail: no eviction - the sidecar is torn down at the end of the run and the
# whole parameter space is ~20MB of skeletons. Add an LRU sweep if a longer-lived
# container ever reuses one.
_SKELETON_DIR: Path | None = None

# Where `--prebuild-skeletons` puts the whole parameter space at image build
# time; see docker/meqtrees/Dockerfile. Every evaluation of a default run hits
# it, so no run does any makems at all.
BAKED_SKELETON_DIR = Path("/opt/ms-skeletons")


def skeleton_dir() -> Path:
    global _SKELETON_DIR
    if _SKELETON_DIR is None:
        # The baked directory is a normal writable container path, so a shape
        # the image was not built with is still built and published into it -
        # it is a head start, not a fixed set.
        _SKELETON_DIR = BAKED_SKELETON_DIR if BAKED_SKELETON_DIR.is_dir() else Path(SCRATCH_ROOT or tempfile.gettempdir()) / "ms-skeletons"
        _SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    return _SKELETON_DIR


def use_skeleton_cache(directory: Path | None) -> None:
    """Point the cache at `directory`, or back at the default when None.

    A fresh directory is how the self-checks force a rebuild.
    """
    global _SKELETON_DIR
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    _SKELETON_DIR = directory


def publish_skeleton(built_ms: Path, cached: Path) -> None:
    """Copy a fresh makems run into the shared cache under its final name.

    Staged then renamed, so a concurrent worker either does not see the entry or
    sees a complete one. Losing the rename race is normal - the winner's copy is
    equivalent - so the loser just drops its own.
    """
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
    """Rewrite the only columns makems derives from StartFreq/StepFreq."""
    with table(str(output_ms / "SPECTRAL_WINDOW"), readonly=False, ack=False) as spw:
        n_chan = int(spw.getcol("NUM_CHAN")[0])
        spw.putcol("CHAN_FREQ", (start_frequency_hz + (np.arange(n_chan) + 0.5) * channel_width_hz)[None, :])
        widths = np.full((1, n_chan), channel_width_hz)
        for column in ("CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION"):
            spw.putcol(column, widths)
        spw.putcol("REF_FREQUENCY", np.array([start_frequency_hz + n_chan * channel_width_hz / 2.0]))
        spw.putcol("TOTAL_BANDWIDTH", np.array([n_chan * channel_width_hz]))


def make_ms_skeleton(cfg: Path, output_ms: Path, args: argparse.Namespace) -> None:
    """run_makems(), reusing a cached run for configs that differ only in frequency."""
    key = "\n".join(line for line in cfg.read_text().splitlines() if not line.startswith(("StartFreq=", "StepFreq=")))
    cached = skeleton_dir() / hashlib.sha256(key.encode()).hexdigest()[:32]
    if not cached.exists():
        run_makems(output_ms)
        publish_skeleton(output_ms, cached)
        return
    shutil.copytree(cached, output_ms, symlinks=True)
    patch_spectral_window(output_ms, args.start_frequency_hz, args.channel_width_hz)
    (output_ms.parent / "makems.log").write_text(f"reused a cached makems skeleton for:\n{key}\n")


def prebuild_skeletons(space: dict) -> None:
    """Build a cache entry for every (NTimes, NFrequencies) the space can ask for.

    A fresh makems run is ~0.11s against ~0.004s for a cache hit, and a default
    run touches ~12 of the parameter space's 20 shapes, first-touching most of
    them mid-sampler where the miss lands straight on the wall clock. Run once
    at image build time (see docker/meqtrees/Dockerfile) the whole space costs
    ~1.2s and ~18MB and no run ever calls makems again - including the workers'
    own warm_forest(), which all eight used to race on the same fresh build.

    The MS name is part of the cache key, so this has to build under the same
    name a real evaluation uses; self_check_skeleton_prebuild() is the guard.
    """
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
    """Point fds 1 and 2 - so child processes too - at files for this block.

    `err_path=None` sends stderr to the stdout file, the way
    `subprocess.run(stderr=STDOUT)` did when each stage was its own process.
    """
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
    """The one meqserver this process talks to, started on first predict.

    `meqtree-pipeliner.py` spent ~0.32s of its ~0.46s on Timba imports, starting
    a meqserver and reaping it again, against ~0.14s of actual RIME predict, and
    paid all of it again on every evaluation. Under `--serve` the imports and the
    meqserver survive between requests; only the per-evaluation compile and run
    remain. stop_meqserver_session() shuts it down again.
    """
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


def stop_meqserver_session() -> None:
    """Shut the meqserver down explicitly, the way meqtree-pipeliner.py does.

    Timba registers stop_default_mqs() with atexit, but CPython joins non-daemon
    threads - including octopussy's event thread, which only exits once the
    server is stopped - before it runs atexit handlers, so leaving it to atexit
    hangs the interpreter at exit forever.
    """
    global _MQS
    if _MQS is not None:
        from Timba.Apps import meqserver

        _MQS = None
        # The forest lives in the server, not here.
        _FOREST.clear()
        meqserver.stop_default_mqs()


def point_to_measurement_set(module, output_ms: Path) -> None:
    """Aim an already-compiled forest at a different Measurement Set."""
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
    mqs = meqserver_session()
    from Timba.TDL import Compile, TDLOptions

    # Compiling the forest is ~0.034s of every evaluation and depends on every
    # tdlconf key except ms_sel.msname: the antenna layout and phase centre come
    # from the fixed antenna table and RightAscension/Declination that
    # write_makems_config() hardcodes, and the MS shape is runtime data. So the
    # forest is compiled once per distinct source/correlation setup and later
    # evaluations just point it at their own MS. self_check_forest_reuse()
    # is the guard on that claim.
    key = "\n".join(line for line in tdlconf.read_text().splitlines() if not line.startswith("ms_sel.msname"))
    # Same sequence meqtree-pipeliner.py runs for
    # `-c <tdlconf> point_source_forest.py[predict] =predict`.
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
        TDLOptions.get_job_func("predict")(mqs, None, wait=True)
        # get_error_log() flushes, so each request only sees its own errors.
        errors = mqs.get_error_log()
        for index, (_event, error) in enumerate(errors):
            # !r, not str(): Timba's DMI record __str__ is still py2
            # (`string.join`) and raises AttributeError, which would replace the
            # meqserver's error with a traceback from the reporting path itself.
            print(f"###   {index:03d}: {error!r}")
    if errors:
        raise SystemExit(f"FATAL: meqserver reported {len(errors)} error(s) during the predict")


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
    corr_sel, n_corr = determine_corr_selection(output_ms)
    run_meqtrees_predict(output_ms, corr_sel, args.source_flux_jy, l_rad, m_rad)

    rng = np.random.default_rng(args.seed)
    with table(str(output_ms), readonly=False, ack=False) as ms:
        data = np.asarray(ms.getcol("DATA"), dtype=np.complex64)
        uvw = np.asarray(ms.getcol("UVW"), dtype=np.float64)
        n_rows, n_chan, data_n_corr = data.shape
        if n_chan != len(freqs_hz):
            raise SystemExit(f"FATAL: DATA has {n_chan} channels, SPW has {len(freqs_hz)}")
        if data_n_corr != n_corr:
            raise SystemExit(f"FATAL: DATA correlation count changed from {n_corr} to {data_n_corr} after MeqTrees predict")

        if noise_sigma_jy:
            per_component_sigma = noise_sigma_jy / math.sqrt(2.0)
            noise = rng.normal(0.0, per_component_sigma, data.shape) + 1j * rng.normal(0.0, per_component_sigma, data.shape)
            data = data + noise.astype(np.complex64)

        ms.putcol("DATA", data)
        for optional_col in ("MODEL_DATA", "CORRECTED_DATA"):
            if optional_col in ms.colnames():
                ms.putcol(optional_col, data)
        if "FLAG" in ms.colnames():
            ms.putcol("FLAG", np.zeros(ms.getcol("FLAG").shape, dtype=bool))
        # WEIGHT and SIGMA are variable-shaped IncrementalStMan columns in a
        # makems MS, and python-casacore's putcol on those is quadratic in rows:
        # 1755 rows cost 43ms for the pair, against 8ms for a putcell row loop
        # and 3ms for this TaQL UPDATE, which is the same row loop in C++.
        # They cannot be moved to a cheaper storage manager - removecols refuses
        # to break up the ISM group makems put them in with 15 other columns.
        if "WEIGHT" in ms.colnames():
            taql(
                "UPDATE $ms SET WEIGHT=%r, SIGMA=%r"
                % (1.0 / (noise_sigma_jy * noise_sigma_jy), noise_sigma_jy)
            )

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

    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as scratch:
        scratch_ms = Path(scratch) / final_ms.name
        cfg = write_makems_config(args, scratch_ms)
        make_ms_skeleton(cfg, scratch_ms, args)
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
    """Compile the forest and run one predict, so evaluation one does not.

    Even against a warm skeleton cache the first simulate in a fresh worker
    costs ~0.17s where the next one costs ~0.03s; the difference is the TDL
    compile and the meqserver's first predict, and the forest is identical for
    every evaluation in a run (see run_meqtrees_predict()). Only worth doing
    when there is a window to hide it in - which is why serve() does this under
    --fifo, where the worker is the meqtrees container's own startup command and
    the run has three containers, a manifest and mpirun still to go, and not on
    the stdin path, where the rank that spawned the worker is already waiting.

    All the workers warm on the same shape, so seven of the eight lose the
    skeleton cache's publish race; that costs ~0.11s of an idle core each,
    inside the window, and the winner leaves an entry the run goes on to use.
    Baking a ready-made Measurement Set into the image to skip those makems runs
    was measured and moves worker-ready time by nothing.
    """
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
    """Run one request's work in this process.

    The MS-to-`.mat` convert used to be its own `docker exec` of
    ms_to_r2d2_mat.py at ~0.15s, of which ~0.01s was the conversion - the rest
    was the exec, a fresh interpreter and the numpy/casacore/scipy imports.
    This worker has all of that live and has just written the MS, so the R2D2
    PoC asks it to convert too.
    """
    if request.get("action") == "convert":
        # Imported on first use, not at module scope: only the R2D2 PoC
        # converts, and the WSClean PoC's worker should not pay for scipy.
        from ms_to_r2d2_mat import main as convert

        convert(request["argv"])
    else:
        simulate(parse_args(request["argv"]))


def serve(fifo_base: str | None = None) -> None:
    """Run one simulate per JSON request line, reusing this process.

    A one-shot `docker exec` of this script spent ~0.45s of its ~0.7s on process
    and meqserver startup that every evaluation repeated. A request is
    `{"argv": [...], "stdout": path, "stderr": path}`; everything the run prints
    goes to those two files, exactly as the caller's `docker exec` redirection
    did, and the reply is one JSON line - `{"returncode": int}`. A request with
    `"action": "convert"` runs handle_request()'s other entry point instead.

    Requests arrive on stdin and replies go to the process's original stdout,
    unless `fifo_base` is given: then they arrive on `<fifo_base>.in` and the
    replies go to `<fifo_base>.out`, a pair of FIFOs on the bind mount both
    containers share. That is what lets run-nested-sampling.sh start this
    worker as the meqtrees container's command - and pay all of the warm-up
    below - before the rank that will use it even exists.
    """
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
            except Exception:
                traceback.print_exc()
                returncode = 1
            except SystemExit as exc:
                print(exc, file=sys.stderr)
                returncode = exc.code if isinstance(exc.code, int) else 1
        replies.write(json.dumps({"returncode": returncode}) + "\n")
        replies.flush()


def self_check_skeleton_cache() -> None:
    """A patched cache hit must reproduce a fresh makems run column for column."""

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
    """A `--fifo` worker must answer on its FIFO pair without deadlocking.

    Both ends block on open until the other side opens, so the request pipe has
    to be opened first by both processes; get that backwards and the run hangs
    with no error. The caller side here is the same O_NONBLOCK retry
    common._connect_shell_started_worker() uses.
    """
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
            self_check_skeleton_cache()
            self_check_skeleton_prebuild()
            self_check_forest_reuse()
            self_check_serve_reply_stream()
            self_check_serve_fifo()
        else:
            simulate(parse_args())
    finally:
        stop_meqserver_session()
