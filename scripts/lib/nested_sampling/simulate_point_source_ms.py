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
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from casacore.tables import table


Cattery_VLA_A = Path("/usr/share/doc/makems/VLAA_ANT.tar.gz")
ANTENNA_TABLE_NAME = "VLAA_ANT"
TDL_SCRIPT = Path("/opt/ri-nested-sampling/point_source_forest.py")

# makems and casacore fsync on nearly every table write. On the bind-mounted
# repo that is ~0.5s of ext4 journal wait per run (makems alone: 0.54s on the
# bind mount vs 0.05s here), so build everything in RAM and copy the finished
# artifacts out once at the end - ~2ms for a ~1MB Measurement Set.
# ponytail: Docker gives /dev/shm 64MB by default, which is ~30x this PoC's
# largest MS; raise --shm-size on the sidecar if the parameter space grows.
SCRATCH_ROOT = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-ms", required=True, help="Measurement Set path to create")
    parser.add_argument("--metadata-json", help="Optional JSON metadata path")
    parser.add_argument("--vla-config", default="VLA.A", choices=["VLA.A"])
    parser.add_argument("--observation-minutes", type=float, required=True)
    parser.add_argument("--integration-seconds", type=float, default=120.0)
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
        meqserver.stop_default_mqs()


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

    # Same sequence meqtree-pipeliner.py runs for
    # `-c <tdlconf> point_source_forest.py[predict] =predict`.
    with redirect_fds(output_ms.parent / "meqtree-pipeliner.log"):
        TDLOptions.config.read(str(tdlconf))
        TDLOptions.config.set_save_filename(None)
        _module, _ns, msg = Compile.compile_file(mqs, str(TDL_SCRIPT), config="predict")
        print("###", msg)
        TDLOptions.get_job_func("predict")(mqs, None, wait=True)
        # get_error_log() flushes, so each request only sees its own errors.
        errors = mqs.get_error_log()
        for index, (_event, error) in enumerate(errors):
            print(f"###   {index:03d}: {error}")
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

    # ponytail: this PoC supports one unpolarized point source; full Stokes
    # models and multi-source dynamic-range stress cases are a follow-up ceiling.
    corr_sel, n_corr = determine_corr_selection(output_ms)
    run_meqtrees_predict(output_ms, corr_sel, args.source_flux_jy, l_rad, m_rad)

    rng = np.random.default_rng(args.seed)
    with table(str(output_ms), readonly=False, ack=False) as ms:
        data = np.asarray(ms.getcol("DATA"), dtype=np.complex64)
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
        if "WEIGHT" in ms.colnames():
            weight = np.full(ms.getcol("WEIGHT").shape, 1.0 / (noise_sigma_jy * noise_sigma_jy), dtype=np.float32)
            ms.putcol("WEIGHT", weight)
        if "SIGMA" in ms.colnames():
            sigma = np.full(ms.getcol("SIGMA").shape, noise_sigma_jy, dtype=np.float32)
            ms.putcol("SIGMA", sigma)

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
        write_makems_config(args, scratch_ms)
        run_makems(scratch_ms)
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


def serve() -> None:
    """Run one simulate per JSON request line on stdin, reusing this process.

    A one-shot `docker exec` of this script spent ~0.45s of its ~0.7s on process
    and meqserver startup that every evaluation repeated. A request is
    `{"argv": [...], "stdout": path, "stderr": path}`; everything the run prints
    goes to those two files, exactly as the caller's `docker exec` redirection
    did, and the reply is one JSON line - `{"returncode": int}` - on the
    process's original stdout.
    """
    replies = os.fdopen(os.dup(1), "w")
    for line in sys.stdin:
        request = json.loads(line)
        returncode = 0
        with redirect_fds(Path(request["stdout"]), Path(request["stderr"])):
            try:
                simulate(parse_args(request["argv"]))
            except Exception:
                traceback.print_exc()
                returncode = 1
            except SystemExit as exc:
                print(exc, file=sys.stderr)
                returncode = exc.code if isinstance(exc.code, int) else 1
        replies.write(json.dumps({"returncode": returncode}) + "\n")
        replies.flush()


if __name__ == "__main__":
    # --serve takes no other arguments, so it is checked before argparse, which
    # requires the full simulate argument set.
    try:
        if sys.argv[1:] == ["--serve"]:
            serve()
        else:
            simulate(parse_args())
    finally:
        stop_meqserver_session()
