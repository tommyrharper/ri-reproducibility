"""This user's Slurm jobs, asked of squeue at most once every two minutes.

`./ri health` and `./ri runs` both ask squeue which runs are live, and
`./ri tui` runs them every few seconds. CSD3's administrators ask that nothing
calls a Slurm client command more often than every two minutes (a scheduling
cycle takes minutes, so a fresher answer says nothing new), so every reader
shares one cached answer per user, refreshed under a lock so two readers
starting together make one call between them. `scripts/lib/slurm.sh` drops
the cache after a submission, so a run just submitted is seen as queued at
once. RI_SQUEUE_TTL overrides the two minutes; 0 always asks.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_TTL_SECONDS = 120


def _user() -> str:
    return os.environ.get("USER") or str(os.getuid())


def cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "ri" / f"squeue-{_user()}.txt"


def _ttl() -> float:
    try:
        return max(0.0, float(os.environ.get("RI_SQUEUE_TTL", DEFAULT_TTL_SECONDS)))
    except ValueError:
        return DEFAULT_TTL_SECONDS


def _fresh(path: Path, ttl: float) -> str | None:
    try:
        if ttl and time.time() - path.stat().st_mtime < ttl:
            return path.read_text()
    except OSError:
        pass
    return None


def _ask() -> str | None:
    try:
        return subprocess.run(
            ["squeue", "-h", "-u", _user(), "-o", "%i %T %j"],
            capture_output=True, text=True, check=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def squeue_lines() -> list[str]:
    """`<job id> <state> <name>` for each of this user's jobs; [] without Slurm."""
    path, ttl = cache_path(), _ttl()
    cached = _fresh(path, ttl)
    if cached is not None:
        return cached.splitlines()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = open(path.with_suffix(".lock"), "w")
    except OSError:
        out = _ask()
        return out.splitlines() if out is not None else []
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Whoever held the lock may have just refreshed it.
        cached = _fresh(path, ttl)
        if cached is not None:
            return cached.splitlines()
        out = _ask()
        if out is None:
            return []
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".squeue-")
        with os.fdopen(fd, "w") as handle:
            handle.write(out)
        os.replace(tmp, path)
        return out.splitlines()


def slurm_jobs() -> dict[str, str]:
    """Run name -> job state for this user's jobs, other than the one asking."""
    me = os.environ.get("SLURM_JOB_ID")
    jobs = {}
    for line in squeue_lines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0] != me:
            jobs[parts[2]] = parts[1]
    return jobs


def _self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        calls = Path(tmp) / "calls"
        bin_dir = Path(tmp) / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "squeue"
        fake.write_text(f"#!/bin/sh\necho x >>{calls}\necho '11 RUNNING r2d2-vlaa-a'\necho '12 PENDING wsclean-vlaa-b'\n")
        fake.chmod(0o755)
        os.environ.update(PATH=f"{bin_dir}:{os.environ['PATH']}", XDG_CACHE_HOME=tmp, SLURM_JOB_ID="12")
        os.environ.pop("RI_SQUEUE_TTL", None)
        assert slurm_jobs() == {"r2d2-vlaa-a": "RUNNING"}, slurm_jobs()
        for _ in range(5):
            slurm_jobs()
        assert calls.read_text().count("x") == 1, "readers within the TTL must share one squeue call"
        os.environ["RI_SQUEUE_TTL"] = "0"
        slurm_jobs()
        assert calls.read_text().count("x") == 2, "RI_SQUEUE_TTL=0 must ask every time"
        os.environ.pop("RI_SQUEUE_TTL")
        cache_path().unlink()
        slurm_jobs()
        assert calls.read_text().count("x") == 3, "a dropped cache (after sbatch) must be asked again"
        fake.unlink()
        cache_path().unlink()
        os.environ["PATH"] = str(bin_dir)  # not the cluster's own squeue
        assert slurm_jobs() == {}, "no squeue means no jobs"
    print("slurm_queue self-check passed")


if __name__ == "__main__" and sys.argv[1:] == ["--self-check"]:
    _self_check()
