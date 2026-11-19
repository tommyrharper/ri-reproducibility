#!/usr/bin/env python3
"""Pack runs' PolyChord chains on one host and unpack them on another."""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

RUNS_DIR = Path("results/nested-sampling")
ARCHIVE = Path("results/chains-export.tar.gz")
PULLED_DIR = Path("results/chains")
# What reading the chains needs; summary.json and the evaluations are GBs.
RUN_FILES = ("run.env", "parameter-space.json")


def is_complete(run: Path) -> bool:
    try:
        with open(run / "summary.json", "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 64))
            return f.read().rstrip().endswith(b"}")
    except OSError:
        return False


def resolve(name: str) -> Path:
    path = Path(name)
    return path if path.is_dir() else RUNS_DIR / name


def ssh_host() -> str:
    if os.environ.get("RI_SSH_HOST"):
        return os.environ["RI_SSH_HOST"]
    host = socket.getfqdn()
    # CSD3 login nodes' own names are not reachable from outside.
    if host.endswith(".data.cluster"):
        host = "login.hpc.cam.ac.uk"
    return f"{getpass.getuser()}@{host}"


def export(names: list[str], archive: Path) -> list[str]:
    runs = [resolve(n) for n in names] if names else sorted(
        p for p in RUNS_DIR.glob("*") if is_complete(p))
    runs = [r for r in runs if (r / "chains").is_dir()]
    if not runs:
        sys.exit(f"no runs with chains/ found{'' if names else f' under {RUNS_DIR}'}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".partial")
    with tarfile.open(partial, "w:gz") as tar:
        for run in runs:
            tar.add(run / "chains", f"{run.name}/chains",
                    filter=lambda m: None if m.name.endswith(".resume") else m)
            for name in RUN_FILES:
                if (run / name).is_file():
                    tar.add(run / name, f"{run.name}/{name}")
    partial.replace(archive)
    return [r.name for r in runs]


def safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = tar.getmembers()
    for m in members:
        parts = Path(m.name).parts
        if m.name.startswith("/") or ".." in parts or not (m.isfile() or m.isdir()):
            sys.exit(f"refusing archive: unsafe member {m.name!r}")
    return members


def unpack(archive: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = safe_members(tar)
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(dest, members, **kwargs)
    return sorted({Path(m.name).parts[0] for m in members})


def pull(source: str, dest: Path) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "chains.tar.gz"
        if subprocess.run(["scp", "-q", source, str(local)]).returncode != 0:
            sys.exit(f"scp {source} failed")
        return unpack(local, dest)


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        done = RUNS_DIR / "r2d2-vlaa-20260101T000000Z"
        (done / "chains").mkdir(parents=True)
        (done / "chains/r_dead-birth.txt").write_text("1 2\n")
        (done / "chains/r.resume").write_text("big")
        (done / "run.env").write_text("NS_NLIVE=500\n")
        (done / "summary.json").write_text('{"evaluations": []}\n')
        live = RUNS_DIR / "wsclean-vlaa-20260101T000001Z"
        (live / "chains").mkdir(parents=True)
        (live / "summary.json").write_text('{"evaluations": [')
        assert export([], ARCHIVE) == [done.name], "only complete runs by default"
        assert export([live.name], ARCHIVE) == [live.name], "a named run is taken as is"
        export([], ARCHIVE)
        got = Path("pulled")
        assert unpack(ARCHIVE, got) == [done.name]
        assert (got / done.name / "chains/r_dead-birth.txt").read_text() == "1 2\n"
        assert (got / done.name / "run.env").is_file()
        assert not (got / done.name / "chains/r.resume").exists(), ".resume left out"
        assert not (got / done.name / "summary.json").exists(), "summary.json left out"
        evil = Path("evil.tar.gz")
        with tarfile.open(evil, "w:gz") as tar:
            info = tarfile.TarInfo("../escape")
            tar.addfile(info)
        try:
            unpack(evil, got)
        except SystemExit as e:
            assert "unsafe" in str(e)
        else:
            raise AssertionError("a ../ member was unpacked")
        assert not Path("escape").exists()
    print("chains self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    subs = parser.add_subparsers(dest="action")
    ex = subs.add_parser("export")
    ex.add_argument("runs", nargs="*")
    pl = subs.add_parser("pull")
    pl.add_argument("source", metavar="HOST:PATH")
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    if args.action == "export":
        names = export(args.runs, ARCHIVE)
        size = ARCHIVE.stat().st_size / 1e6
        print(f"wrote {ARCHIVE} ({size:.1f} MB): {len(names)} run(s)")
        for name in names:
            print(f"  {name}")
        source = f"{ssh_host()}:{ARCHIVE.resolve()}"
        print(f"\nOn the other machine, in its checkout:\n  ./ri chains pull {shlex.quote(source)}")
    elif args.action == "pull":
        names = pull(args.source, PULLED_DIR)
        print(f"unpacked {len(names)} run(s) into {PULLED_DIR}/:")
        for name in names:
            print(f"  {name}    ./ri plot gui {PULLED_DIR / name}")
    else:
        parser.error("pick export or pull")


if __name__ == "__main__":
    main()
