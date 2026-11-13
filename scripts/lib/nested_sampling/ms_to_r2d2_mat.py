#!/usr/bin/env python3
"""Convert a CASA Measurement Set to the R2D2-RI MATLAB .mat schema."""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
from casacore.tables import table

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
    # complex128(vis) split into its halves, without the complex temporary.
    y = tuple(np.ascontiguousarray(half, dtype=np.float64).reshape(-1) for half in (vis.real, vis.imag))
    # Convert MS inverse-variance weights to R2D2 nW, applying simulated sigma.
    nW = np.repeat(np.sqrt(np.asarray(row_weight, dtype=np.float64)) / noise_sigma_jy, n_chan)

    mat_path.parent.mkdir(parents=True, exist_ok=True)
    write_mat_columns(mat_path, {"u": (u_lambda,), "v": (v_lambda,), "y": y, "nW": (nW,)})


def write_mat_columns(mat_path: Path, columns: dict[str, tuple[np.ndarray, ...]]) -> None:
    """Byte for byte what scipy's savemat(do_compression=False) writes for
    these float64 (n, 1) columns, a (real, imag) pair being one complex column,
    straight from the arrays: savemat's per-element tobytes() copies (and the
    complex128 temporary) were a third of R2D2's simulate (docs/csd3-speed.md,
    round 187). self_check_write_mat_columns() is the guard."""
    # 116-byte description, subsystem offset, version 0x0100, 'IM' endian test.
    header = struct.pack("<116sqH2s", f"MATLAB 5.0 MAT-file Platform: {os.name}, Created on: {time.asctime()}".encode(),
                         0, 0x0100, b"IM")
    parts: list = [header]
    for name, halves in columns.items():
        n = len(halves[0])
        raw = name.encode("latin1")
        assert n and len(raw) <= 4 and all(h.dtype == np.float64 and h.flags.c_contiguous and len(h) == n for h in halves)
        # miMATRIX; array flags (miUINT32, mxDOUBLE_CLASS, complex bit); dims
        # (miINT32, n x 1); the name as a small miINT8 element; then miDOUBLE data.
        parts.append(struct.pack("<II IIII IIii I4s", 14, 40 + len(halves) * (8 + 8 * n),
                                 6, 8, 6 | (len(halves) == 2) << 11, 0, 5, 8, n, 1, len(raw) << 16 | 1, raw))
        for half in halves:
            parts += [struct.pack("<II", 9, 8 * n), memoryview(half).cast("B")]
    with open(mat_path, "wb") as stream:
        stream.writelines(parts)


def self_check_write_mat_columns() -> None:
    """write_r2d2_mat() against savemat's bytes, with both on one clock."""
    import tempfile
    from unittest import mock

    from scipy.io import loadmat, savemat

    rng = np.random.default_rng(1)
    # complex64 is the simulator's .mat; complex128 is ms_to_r2d2_mat()'s.
    for rows, n_chan, dtype in ((1, 1, np.complex64), (7, 3, np.complex64), (2000, 4, np.complex64),
                                (300, 5, np.complex128)):
        vis = (rng.normal(size=(rows, n_chan)) + 1j * rng.normal(size=(rows, n_chan))).astype(dtype)
        uvw = rng.normal(0.0, 300.0, (rows, 3))
        freqs = 1e9 + np.arange(n_chan) * 1e6
        weight = rng.uniform(0.5, 2.0, rows)
        with tempfile.TemporaryDirectory() as tmp:
            got, want = Path(tmp) / "got.mat", Path(tmp) / "want.mat"
            with mock.patch("time.asctime", return_value="Wed Oct 28 05:30:00 2026"):
                write_r2d2_mat(got, vis, uvw, freqs, weight, 0.3)
                savemat(str(want), {
                    "u": (uvw[:, 0, None] * freqs / SPEED_OF_LIGHT).reshape(-1, 1),
                    "v": (uvw[:, 1, None] * freqs / SPEED_OF_LIGHT).reshape(-1, 1),
                    "y": np.asarray(vis, dtype=np.complex128).reshape(-1, 1),
                    "nW": np.repeat(np.sqrt(weight) / 0.3, n_chan).reshape(-1, 1),
                }, do_compression=False)
            assert got.read_bytes() == want.read_bytes(), (rows, n_chan)
            assert np.array_equal(loadmat(str(got))["y"], np.asarray(vis, np.complex128).reshape(-1, 1))
    print("write_mat_columns: OK")


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
    if argv is None and sys.argv[1:] == ["--self-check"]:
        self_check_write_mat_columns()
        return
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
