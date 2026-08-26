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
    return parser.parse_args(argv)


def ms_to_r2d2_mat(ms_path: Path, mat_path: Path, corr_index: int = 0) -> dict[str, int]:
    with table(str(ms_path), readonly=True, ack=False) as ms:
        data = np.asarray(ms.getcol("DATA"), dtype=np.complex128)
        uvw = np.asarray(ms.getcol("UVW"), dtype=np.float64)
        weight = np.asarray(ms.getcol("WEIGHT"), dtype=np.float64)

    with table(str(ms_path / "SPECTRAL_WINDOW"), readonly=True, ack=False) as spw:
        freqs_hz = np.asarray(spw.getcol("CHAN_FREQ")[0], dtype=np.float64)

    n_rows, n_chan, n_corr = data.shape
    if corr_index < 0 or corr_index >= n_corr:
        raise ValueError(f"corr_index {corr_index} out of range for {n_corr} correlations")

    vis = data[:, :, corr_index]
    row_weight = weight[:, corr_index]

    u_m = np.repeat(uvw[:, 0], n_chan)
    v_m = np.repeat(uvw[:, 1], n_chan)
    freqs = np.tile(freqs_hz, n_rows)
    wavelength = SPEED_OF_LIGHT / freqs
    u_lambda = u_m / wavelength
    v_lambda = v_m / wavelength
    y = vis.reshape(-1)
    # MS WEIGHT is inverse variance; R2D2 nW is sqrt(inverse variance).
    nW = np.sqrt(np.repeat(row_weight, n_chan))

    mat_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        str(mat_path),
        {
            "u": u_lambda.reshape(-1, 1),
            "v": v_lambda.reshape(-1, 1),
            "y": y.reshape(-1, 1),
            "nW": nW.reshape(-1, 1),
        },
        do_compression=True,
    )
    return {
        "visibility_count": int(y.size),
        "channel_count": int(n_chan),
        "row_count": int(n_rows),
        "corr_index": int(corr_index),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stats = ms_to_r2d2_mat(Path(args.ms_path), Path(args.mat_path), corr_index=args.corr_index)
    print(stats)


if __name__ == "__main__":
    main()
