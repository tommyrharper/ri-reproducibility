#!/usr/bin/env python3
"""Self-check: simulated MS -> .mat -> R2D2 load_data_to_tensor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=os.environ.get("REPO_ROOT", os.getcwd()))
    parser.add_argument("--meqtrees-image", default=os.environ.get("MEQTREES_IMAGE", "ri-reproducibility/meqtrees:kern-10"))
    parser.add_argument("--r2d2-image", default=os.environ.get("R2D2_IMAGE", "ri-reproducibility/r2d2:cpu"))
    parser.add_argument("--platform", default=os.environ.get("DOCKER_DEFAULT_PLATFORM", "linux/arm64"))
    parser.add_argument("--ms-path", help="Use an existing Measurement Set instead of simulating")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    work_dir = Path(tempfile.mkdtemp(prefix="ms-mat-bridge-", dir=repo_root / "results"))
    ms_path = work_dir / "sim.ms"
    mat_path = work_dir / "r2d2_data.mat"

    if args.ms_path:
        ms_path = Path(args.ms_path).resolve()
        work_dir = ms_path.parent
        mat_path = work_dir / "r2d2_data.mat"
    else:
        run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                args.platform,
                "-v",
                f"{work_dir}:/work",
                args.meqtrees_image,
                "--output-ms",
                "/work/sim.ms",
                "--metadata-json",
                "/work/simulation.json",
                "--vla-config",
                "VLA.A",
                "--observation-minutes",
                "4",
                "--channel-count",
                "2",
                "--start-frequency-hz",
                "1.0e9",
                "--channel-width-hz",
                "1.0e6",
                "--source-flux-jy",
                "1.0",
                "--dynamic-range",
                "100",
                "--seed",
                "42",
            ]
        )

    run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            args.platform,
            "-v",
            f"{work_dir}:/work",
            "--entrypoint",
            "python3",
            args.meqtrees_image,
            "/opt/ri-nested-sampling/ms_to_r2d2_mat.py",
            "--ms-path",
            "/work/sim.ms",
            "--mat-path",
            "/work/r2d2_data.mat",
        ]
    )

    load_cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        args.platform,
        "-v",
        f"{work_dir}:/work",
        "--entrypoint",
        "python3",
        args.r2d2_image,
        "-c",
        (
            "import sys; sys.path.insert(0, 'src'); "
            "from utils import load_data_to_tensor; "
            "data = load_data_to_tensor(uv_file_path='/work/r2d2_data.mat', super_resolution=1.52, verbose=False); "
            "print('visibility_count', data['y'].numel()); "
            "print('u_count', data['u'].numel()); "
            "print('nW_count', data['nW'].numel())"
        ),
    ]
    result = subprocess.run(load_cmd, check=True, capture_output=True, text=True)
    print(result.stdout.strip())
    summary = {
        "work_dir": str(work_dir),
        "ms_path": str(ms_path),
        "mat_path": str(mat_path),
        "load_stdout": result.stdout.strip(),
        "status": "ok",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout or str(exc), file=sys.stderr)
        sys.exit(exc.returncode)
