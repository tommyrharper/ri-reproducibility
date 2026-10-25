#!/usr/bin/env python3
"""Convert a CASA Measurement Set to the R2D2-RI MATLAB .mat schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from casacore.tables import table
from scipy.io import savemat

SPEED_OF_LIGHT = 299792458.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms-path", required=True, help="Input Measurement Set path")
    parser.add_argument("--mat-path", required=True, help="Output .mat path")
    parser.add_argument(
        "--corr-index",
        type=int,
        default=0,
        help="Correlation index for Stokes-I parallel-hand visibility (default: 0)",
    )
    parser.add_argument(
        "--noise-sigma-jy",
        type=float,
        default=1.0,
        help=(
            "Thermal-noise sigma the MS was simulated with, in Jy. The simulator "
            "leaves WEIGHT at makems' 1.0 rather than writing 1/sigma^2 to every "
            "row (see fill_point_source_visibilities), so this is what carries the "
            "noise level into nW. Its simulation.json records it as "
            "noise.complex_sigma_jy. Default 1.0 takes WEIGHT at face value, which "
            "is right for an MS that carries real weights."
        ),
    )
    return parser.parse_args(argv)


def write_r2d2_mat(
    mat_path: Path,
    vis: np.ndarray,
    uvw: np.ndarray,
    freqs_hz: np.ndarray,
    row_weight: np.ndarray,
    noise_sigma_jy: float,
) -> None:
    """The .mat from arrays in memory: `vis` is one correlation, rows x channels.
    The simulator calls this directly with what it just wrote to the MS, which
    skips re-reading it (docs/csd3-speed.md, round 13)."""
    n_chan = vis.shape[1]
    u_lambda = (uvw[:, 0, None] * freqs_hz / SPEED_OF_LIGHT).reshape(-1)
    v_lambda = (uvw[:, 1, None] * freqs_hz / SPEED_OF_LIGHT).reshape(-1)
    y = np.asarray(vis, dtype=np.complex128).reshape(-1)
    # Convert MS inverse-variance weights to R2D2 nW, applying simulated sigma.
    nW = np.repeat(np.sqrt(np.asarray(row_weight, dtype=np.float64)) / noise_sigma_jy, n_chan)

    mat_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        str(mat_path),
        {
            "u": u_lambda.reshape(-1, 1),
            "v": v_lambda.reshape(-1, 1),
            "y": y.reshape(-1, 1),
            "nW": nW.reshape(-1, 1),
        },
        # Small per-evaluation MAT; compression costs more CPU than its I/O saves.
        do_compression=False,
    )


def ms_to_r2d2_mat(
    ms_path: Path,
    mat_path: Path,
    corr_index: int = 0,
    noise_sigma_jy: float = 1.0,
) -> dict[str, int]:
    if noise_sigma_jy <= 0.0:
        # Silently dividing by it would put `inf` in nW and hand R2D2 a
        # reconstruction nothing downstream could tell was wrong.
        raise ValueError(f"noise_sigma_jy must be positive, got {noise_sigma_jy}")
    with table(str(ms_path), readonly=True, ack=False) as ms:
        n_corr = ms.getcol("DATA", startrow=0, nrow=1).shape[2]
        if corr_index < 0 or corr_index >= n_corr:
            raise ValueError(f"corr_index {corr_index} out of range for {n_corr} correlations")
        # Only the one correlation used: reading all four and widening them to
        # complex128 was 76ms of a 211ms convert on the heaviest corner
        # (docs/csd3-speed.md, round 12). Same values, bit for bit.
        vis = np.asarray(ms.getcolslice("DATA", [-1, corr_index], [-1, corr_index])[:, :, 0], dtype=np.complex128)
        uvw = np.asarray(ms.getcol("UVW"), dtype=np.float64)
        weight = np.asarray(ms.getcol("WEIGHT"), dtype=np.float64)

    with table(str(ms_path / "SPECTRAL_WINDOW"), readonly=True, ack=False) as spw:
        freqs_hz = np.asarray(spw.getcol("CHAN_FREQ")[0], dtype=np.float64)

    write_r2d2_mat(mat_path, vis, uvw, freqs_hz, weight[:, corr_index], noise_sigma_jy)
    n_rows, n_chan = vis.shape
    return {
        "visibility_count": int(vis.size),
        "channel_count": int(n_chan),
        "row_count": int(n_rows),
        "corr_index": int(corr_index),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stats = ms_to_r2d2_mat(
        Path(args.ms_path),
        Path(args.mat_path),
        corr_index=args.corr_index,
        noise_sigma_jy=args.noise_sigma_jy,
    )
    print(stats)


if __name__ == "__main__":
    main()
