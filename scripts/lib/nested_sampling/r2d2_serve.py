#!/usr/bin/env python3
"""Long-lived R2D2 imaging worker: one `imager.py` run per JSON request line.

A per-evaluation `docker run` of the R2D2 image cost ~2.4s warm on this host and
only ~0.6s of that was science: ~0.5s of container create/start plus ~1.3s of
`import torch` and the R2D2 module imports, paid again on every evaluation. This
process keeps both alive inside the shared R2D2 sidecar. A request is
`{"argv": [...], "stdout": path, "stderr": path}` on stdin - everything the
imaging run prints goes to those two files, exactly as the caller's `docker run`
redirection did - and the reply is one JSON line on stdout,
`{"returncode": int, "peak_memory_bytes": int}`.

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


def peak_memory_bytes() -> int:
    """This worker's high-water RSS.

    `docker stats` used to sample a container that held one evaluation; a worker
    holds the whole rank, so this is a running maximum across its evaluations
    rather than a per-evaluation peak. It still answers the question the metric
    exists for - how much memory an R2D2 reconstruction of this size needs - and
    the first evaluation on a worker reports exactly what the container did.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def serve() -> None:
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
    replies = os.fdopen(os.dup(1), "w")
    for line in sys.stdin:
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


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check_serve_reply_stream()
    else:
        serve()
