#!/usr/bin/env python3
"""Create a VLA Measurement Set for a noisy single point source.

The MS skeleton and VLA.A antenna table come from MeqTrees/Cattery's bundled
makems examples. The visibility fill is the closed-form point-source equation,
which is the smallest deterministic ceiling for this one-source PoC.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from casacore.tables import table


C_M_PER_S = 299_792_458.0
Cattery_VLA_A = Path("/usr/Cattery/Siamese/MS/VLAAA_ANTENNA")


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def require_clean_output(output_ms: Path) -> None:
    if output_ms.exists():
        raise SystemExit(f"FATAL: output Measurement Set already exists: {output_ms}")
    for suffix in ("_p0", "_p1", ".gds"):
        candidate = output_ms.parent / f"{output_ms.name}{suffix}"
        if candidate.exists():
            raise SystemExit(f"FATAL: makems scratch output already exists: {candidate}")


def write_makems_config(args: argparse.Namespace, output_ms: Path) -> Path:
    output_ms.parent.mkdir(parents=True, exist_ok=True)
    antenna_dst = output_ms.parent / "VLAAA_ANTENNA"
    if not antenna_dst.exists():
        shutil.copytree(Cattery_VLA_A, antenna_dst)

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
                "AntennaTableName=VLAAA_ANTENNA",
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


def fill_point_source_visibilities(args: argparse.Namespace, output_ms: Path) -> dict[str, object]:
    if args.dynamic_range <= 0:
        raise SystemExit("FATAL: --dynamic-range must be positive")
    if args.source_flux_jy <= 0:
        raise SystemExit("FATAL: --source-flux-jy must be positive")

    noise_sigma_jy = args.source_flux_jy / args.dynamic_range
    l_rad = math.radians(args.source_l_arcsec / 3600.0)
    m_rad = math.radians(args.source_m_arcsec / 3600.0)
    n_term = math.sqrt(max(0.0, 1.0 - l_rad * l_rad - m_rad * m_rad))

    with table(str(output_ms / "SPECTRAL_WINDOW"), readonly=True, ack=False) as spw:
        freqs_hz = np.asarray(spw.getcol("CHAN_FREQ")[0], dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    with table(str(output_ms), readonly=False, ack=False) as ms:
        data_shape = ms.getcol("DATA").shape
        if len(data_shape) != 3:
            raise SystemExit(f"FATAL: unexpected DATA shape in {output_ms}: {data_shape}")
        n_rows, n_chan, n_corr = data_shape
        if n_chan != len(freqs_hz):
            raise SystemExit(f"FATAL: DATA has {n_chan} channels, SPW has {len(freqs_hz)}")

        uvw_m = np.asarray(ms.getcol("UVW"), dtype=np.float64)
        uvw_lambda = uvw_m[:, None, :] * freqs_hz[None, :, None] / C_M_PER_S
        phase_arg = uvw_lambda[:, :, 0] * l_rad + uvw_lambda[:, :, 1] * m_rad + uvw_lambda[:, :, 2] * (n_term - 1.0)
        point_model = args.source_flux_jy * np.exp(-2j * np.pi * phase_arg)

        data = np.zeros((n_rows, n_chan, n_corr), dtype=np.complex64)
        # ponytail: this PoC supports one unpolarized point source; full Stokes
        # models and multi-source dynamic-range stress cases are a follow-up ceiling.
        if n_corr >= 1:
            data[:, :, 0] = point_model
        if n_corr >= 4:
            data[:, :, 3] = point_model
        elif n_corr >= 2:
            data[:, :, 1] = point_model

        if noise_sigma_jy:
            per_component_sigma = noise_sigma_jy / math.sqrt(2.0)
            noise = rng.normal(0.0, per_component_sigma, data.shape) + 1j * rng.normal(0.0, per_component_sigma, data.shape)
            data += noise.astype(np.complex64)

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
        "visibility_engine": "MeqTrees/Cattery makems uvw skeleton plus analytic single-point-source fill",
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


def main() -> None:
    args = parse_args()
    output_ms = Path(args.output_ms)
    require_clean_output(output_ms)
    write_makems_config(args, output_ms)
    run_makems(output_ms)
    metadata = fill_point_source_visibilities(args, output_ms)

    metadata_path = Path(args.metadata_json) if args.metadata_json else output_ms.parent / "simulation.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
