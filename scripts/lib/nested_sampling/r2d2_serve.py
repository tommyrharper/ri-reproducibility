#!/usr/bin/env python3
"""Long-lived R2D2 imaging worker: one `imager.py` run per JSON request line.

A per-evaluation `docker run` of the R2D2 image cost ~2.4s warm on this host and
only ~0.6s of that was science: ~0.5s of container create/start plus ~1.3s of
`import torch` and the R2D2 module imports, paid again on every evaluation. This
process keeps both alive inside the shared R2D2 sidecar. A request is
`{"argv": [...], "stdout": path, "stderr": path}` on stdin - everything the
imaging run prints goes to those two files, exactly as the caller's `docker run`
redirection did - and the reply is one JSON line on stdout,
`{"returncode": int, "peak_memory_bytes": int}`. With `--fifo <base>` the same
conversation happens over the FIFO pair `<base>.in` / `<base>.out` instead, and
`--fifo-dir <dir>` serves one such pair per rank from a single warm-up. Either
is what lets the run script start these workers as the R2D2 container's own
command, before the ranks that will use them exist.

Upstream's `src/imager.py` has no importable entry point (its whole body sits
under `if __name__ == "__main__"`), so each request re-runs that body with
runpy. Its imports are then served from `sys.modules`, which is where the saving
comes from.

This script runs from the repository bind mount, not from a copy baked into the
R2D2 image, so editing it needs no image rebuild.
"""

from __future__ import annotations

import json
import os
import resource
import runpy
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

R2D2_HOME = Path(os.environ.get("R2D2_HOME", "/opt/r2d2/R2D2-RI"))
IMAGER = Path(os.environ.get("R2D2_IMAGER", R2D2_HOME / "src" / "imager.py"))


@contextmanager
def redirect_fds(out_path: Path, err_path: Path):
    """Point fds 1 and 2 - so child processes too - at files for this block."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    with out_path.open("w") as out, err_path.open("w") as err:
        try:
            os.dup2(out.fileno(), 1)
            os.dup2(err.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


# The high-water RSS a forked worker inherits from the warm-up it was forked
# from. A child process starts a fresh ru_maxrss counter even though it starts
# holding all of the parent's resident pages, so without this a pool worker
# under-reports by the whole cost of the imports - measured 196MB against the
# 303MB the same request reports from a worker that imported for itself.
_PEAK_FLOOR = 0


def peak_memory_bytes() -> int:
    """This worker's high-water RSS.

    `docker stats` used to sample a container that held one evaluation; a worker
    holds the whole rank, so this is a running maximum across its evaluations
    rather than a per-evaluation peak. It still answers the question the metric
    exists for - how much memory an R2D2 reconstruction of this size needs - and
    the first evaluation on a worker reports exactly what the container did.
    """
    return max(_PEAK_FLOOR, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def warm_imports() -> None:
    """Pay `import torch` and the R2D2 modules before request one.

    ~0.9s, and it used to be paid inside evaluation one, where it landed on the
    sampler's wall clock. Run by the container's own command it happens while
    the PolyChord container, mpirun and PolyChord's own setup still have to.
    """
    os.chdir(R2D2_HOME)
    sys.path.insert(0, str(IMAGER.parent))
    # imager.py's own import block, minus the parts it re-imports for free.
    # Anything it imports that is missing here is simply paid on request one.
    # Under redirect_fds because a stray import-time print would otherwise land
    # in the reply stream and be read as a reply.
    with redirect_fds(Path(os.devnull), Path(os.devnull)):
        try:
            import optimiser  # noqa: F401
            import utils  # noqa: F401
        except Exception:
            traceback.print_exc()


def serve_pool(fifo_dir: str) -> None:
    """One worker per FIFO pair in `fifo_dir`, all forked from a single warm-up.

    A run wants one worker per rank, and as separate interpreters they all
    `import torch` at the same moment: 0.89s on its own becomes 1.05-1.61s
    across 8 of them on a 20-CPU host, and the sampler waits for the slowest.
    Importing once and forking pays it at the solo price, and the children then
    share the ~300MB of it copy-on-write instead of holding a copy each.
    """
    global _PEAK_FLOOR
    bases = sorted(str(path)[: -len(".in")] for path in Path(fifo_dir).glob("*.in"))
    warm_imports()
    # Set before the fork, so every child reports at least what the warm-up it
    # inherited already cost.
    _PEAK_FLOOR = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    children = []
    for base in bases:
        pid = os.fork()
        if pid:
            children.append(pid)
            continue
        status = 0
        try:
            answer(base)
        except Exception:
            traceback.print_exc()
            status = 1
        # _exit, not sys.exit: this child inherited the parent's atexit hooks and
        # stdio buffers and must run neither.
        os._exit(status)
    for pid in children:
        os.waitpid(pid, 0)


def serve(fifo_base: str | None = None) -> None:
    """Warm this process up and answer requests until the stream ends."""
    warm_imports()
    answer(fifo_base)


def answer(fifo_base: str | None) -> None:
    """Run one imager.py per request line until the request stream ends."""
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
            saved_argv = sys.argv
            try:
                sys.argv = [str(IMAGER), *request["argv"]]
                runpy.run_path(str(IMAGER), run_name="__main__")
            except SystemExit as exc:
                returncode = exc.code if isinstance(exc.code, int) else 1
            except Exception:
                traceback.print_exc()
                returncode = 1
            finally:
                sys.argv = saved_argv
        replies.write(json.dumps({"returncode": returncode, "peak_memory_bytes": peak_memory_bytes()}) + "\n")
        replies.flush()


def self_check_serve_reply_stream() -> None:
    """Replies must carry JSON only, and a failed request must not end the worker."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        imager = root / "imager.py"
        imager.write_text(
            "import sys\n"
            "print('chatter on stdout')\n"
            "print('chatter on stderr', file=sys.stderr)\n"
            "sys.exit(int(sys.argv[1]))\n"
        )
        worker = subprocess.Popen(
            [sys.executable, __file__],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "R2D2_IMAGER": str(imager), "R2D2_HOME": str(root)},
        )
        for code in (3, 0):
            request = {"argv": [str(code)], "stdout": str(root / "out.log"), "stderr": str(root / "err.log")}
            worker.stdin.write(json.dumps(request) + "\n")
            worker.stdin.flush()
            reply = json.loads(worker.stdout.readline())
            assert reply["returncode"] == code, reply
            assert reply["peak_memory_bytes"] > 0, reply
            assert (root / "out.log").read_text() == "chatter on stdout\n"
            assert (root / "err.log").read_text() == "chatter on stderr\n"
        worker.stdin.close()
        assert worker.wait(timeout=30) == 0
    print("r2d2 serve reply stream self-check passed")


def self_check_serve_fifo() -> None:
    """A `--fifo` worker must answer on its FIFO pair without deadlocking."""
    import subprocess
    import tempfile
    import time

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        imager = root / "imager.py"
        imager.write_text("import sys\nsys.exit(int(sys.argv[1]))\n")
        base = root / "0"
        os.mkfifo(f"{base}.in")
        os.mkfifo(f"{base}.out")
        worker = subprocess.Popen(
            [sys.executable, __file__, "--fifo", str(base)],
            env={**os.environ, "R2D2_IMAGER": str(imager), "R2D2_HOME": str(root)},
        )
        deadline = time.monotonic() + 60
        while True:
            try:
                write_fd = os.open(f"{base}.in", os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError:
                assert time.monotonic() < deadline, "the --fifo worker never opened its request pipe"
                time.sleep(0.01)
        os.set_blocking(write_fd, True)
        with os.fdopen(write_fd, "w") as requests, open(f"{base}.out") as replies:
            for code in (3, 0):
                request = {"argv": [str(code)], "stdout": str(root / "o.log"), "stderr": str(root / "e.log")}
                requests.write(json.dumps(request) + "\n")
                requests.flush()
                assert json.loads(replies.readline())["returncode"] == code
        assert worker.wait(timeout=30) == 0, "the --fifo worker did not exit on EOF"
    print("r2d2 serve fifo self-check passed")


def self_check_serve_pool() -> None:
    """`--fifo-dir` must serve every pair in the directory off one warm-up."""
    import subprocess
    import tempfile
    import time

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "imager.py").write_text("import sys\nsys.exit(int(sys.argv[1]))\n")
        (root / "optimiser.py").write_text("")
        # Stands in for `import torch`: one line per interpreter that imports it.
        marker = root / "imports.log"
        # The 64MB is dropped again before the fork, so it only shows up in a
        # reply if the warm-up's high-water mark crossed the fork with it.
        (root / "utils.py").write_text(
            f"open({str(marker)!r}, 'a').write('x\\n')\n"
            "_ = bytearray(64 * 1024 * 1024)\n"
            "del _\n"
        )
        pool = root / "pool"
        pool.mkdir()
        for rank in (0, 1):
            os.mkfifo(pool / f"{rank}.in")
            os.mkfifo(pool / f"{rank}.out")
        worker = subprocess.Popen(
            [sys.executable, __file__, "--fifo-dir", str(pool)],
            env={**os.environ, "R2D2_IMAGER": str(root / "imager.py"), "R2D2_HOME": str(root)},
        )
        deadline = time.monotonic() + 60
        for rank, code in ((0, 3), (1, 0)):
            base = pool / str(rank)
            while True:
                try:
                    write_fd = os.open(f"{base}.in", os.O_WRONLY | os.O_NONBLOCK)
                    break
                except OSError:
                    assert time.monotonic() < deadline, f"rank {rank} never got a worker"
                    time.sleep(0.01)
            os.set_blocking(write_fd, True)
            with os.fdopen(write_fd, "w") as requests, open(f"{base}.out") as replies:
                request = {"argv": [str(code)], "stdout": str(root / "o.log"), "stderr": str(root / "e.log")}
                requests.write(json.dumps(request) + "\n")
                requests.flush()
                reply = json.loads(replies.readline())
                assert reply["returncode"] == code
                assert reply["peak_memory_bytes"] > 64 * 1024 * 1024, reply
        assert worker.wait(timeout=30) == 0, "the pool did not exit once every worker saw EOF"
        # The point of forking rather than starting one interpreter per rank.
        assert marker.read_text() == "x\n", f"the warm-up ran more than once: {marker.read_text()!r}"
    print("r2d2 serve pool self-check passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check_serve_reply_stream()
        self_check_serve_fifo()
        self_check_serve_pool()
    elif sys.argv[1:2] == ["--fifo-dir"]:
        serve_pool(sys.argv[2])
    else:
        serve(sys.argv[2] if sys.argv[1:2] == ["--fifo"] else None)
