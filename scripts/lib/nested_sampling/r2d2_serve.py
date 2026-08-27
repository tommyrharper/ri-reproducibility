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
command, before the ranks that will use them exist. `--fifo-dir` also opens
every pair before it warms up, so a rank attaches at ~0.3s rather than ~1.2s and
spends the difference on its own startup instead of on this one's.

The warm-up also patches `MeasOp.get_op_norm` to a Lanczos solve and gives each
measurement operator one FINUFFT plan per transform type; see `patch_op_norm`
and `patch_nufft_plans` below.

Upstream's `src/imager.py` has no importable entry point (its whole body sits
under `if __name__ == "__main__"`), so each request re-runs that body with
runpy. Its imports are then served from `sys.modules`, which is where the saving
comes from - and the same guard is what lets the warm-up run the file under a
different name to pay for those imports up front.

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
import types
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


# `utils/__init__.py` re-exports one name from each of these, so importing any
# part of `utils` imports all of them. The two at the end are the ones nothing
# on the imaging path uses - `util_training` pulls lightning and `noise` pulls
# scipy.optimize, 0.13s of the imaging worker's readiness between them - so the
# search below reaches them only for a name the rest do not define.
_UTILS_SUBMODULES = ("args", "data", "evaluate", "io", "meas_op", "misc", "util_model", "noise", "util_training")


def install_lazy_utils() -> None:
    """Make `utils` import its submodules on demand rather than all at once.

    A module `__getattr__` (PEP 562) that resolves a name by walking the
    submodules in order, so `from utils import vprint` costs the submodules up
    to the one that defines it and no more. Same names, same values - only the
    ones nothing asks for go unimported.
    """
    import importlib

    package = types.ModuleType("utils")
    package.__path__ = [str(IMAGER.parent / "utils")]

    def __getattr__(name: str):
        for submodule in _UTILS_SUBMODULES:
            value = getattr(importlib.import_module(f"utils.{submodule}"), name, None)
            if value is not None:
                setattr(package, name, value)
                return value
        raise AttributeError(f"module 'utils' has no attribute {name!r}")

    package.__getattr__ = __getattr__
    sys.modules["utils"] = package


def warm_imports() -> None:
    """Pay `import torch` and the R2D2 modules before request one.

    ~0.9s, and it used to be paid inside evaluation one, where it landed on the
    sampler's wall clock. Run by the container's own command it happens while
    the PolyChord container, mpirun and PolyChord's own setup still have to.
    """
    os.chdir(R2D2_HOME)
    sys.path.insert(0, str(IMAGER.parent))
    install_lazy_utils()
    # Run imager.py's own body under a name that is not "__main__": its imaging
    # work is all behind an `if __name__ == "__main__"` guard, so what executes
    # is exactly its import block - no hand-maintained copy of it to drift.
    # Under redirect_fds because a stray import-time print would otherwise land
    # in the reply stream and be read as a reply.
    with redirect_fds(Path(os.devnull), Path(os.devnull)):
        try:
            runpy.run_path(str(IMAGER), run_name="__warmup__")
            patch_op_norm()
            patch_nufft_plans()
            # `create_meas_op` imports its NUFFT backend inside the function,
            # so imager.py's import block does not reach it and every worker
            # paid it on request one instead: 0.165s against a 0.072s steady
            # state. This is the backend `write_r2d2_config` asks for, and it
            # goes after patch_op_norm so that picking another one costs a slow
            # request one rather than an unpatched operator norm.
            from ri_measurement_operator.pysrc.measOperator import (  # noqa: F401
                meas_op_nufft_pytorch_finufft,
            )
        except Exception:
            traceback.print_exc()


def patch_nufft_plans() -> None:
    """Build each operator's FINUFFT plans once instead of once per transform.

    `finufft`'s simple interface - the one pytorch_finufft calls - creates a
    plan, sets the sampling points on it and destroys it again on every single
    transform. An evaluation applies the same trajectory tens of times (~21
    Lanczos matvecs, ~45 transforms, before the UNet passes even start), so all
    of that setup is repetition: measured at 0.018s of the 0.063s an operator
    norm cost when this landed, at the 29 matvecs `tol=1e-5` then took.
    Keeping one plan per (operator, transform type) measured a forward/adjoint
    pair at 0.99ms against 1.89ms.

    Same library, same eps/isign/upsampfac/modeord, so this only removes
    `makeplan`/`setpts`/`destroy` plus the autograd `Function.apply` dispatch
    that eager inference has no use for; the transforms come back bit-identical
    at this problem size. Anything the cached plans do not cover - a non-CPU
    device, a batch of more than one image - falls back to upstream. Both are
    under `no_grad`: the plan path detaches through numpy anyway, and this
    worker only ever runs `imager.py`, which is inference.
    """
    import finufft
    import numpy as np
    import torch
    from ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import (
        MeasOpPytorchFinufft,
    )

    forward, adjoint = MeasOpPytorchFinufft._GA, MeasOpPytorchFinufft._AtGt

    def plan(self, nufft_type: int):
        """This operator's plan for `nufft_type`, or None if upstream must run."""
        if str(self._device or "cpu") != "cpu":
            return None
        plans = getattr(self, "_ri_nufft_plans", None)
        if plans is None:
            plans = {}
            setattr(self, "_ri_nufft_plans", plans)
        if nufft_type not in plans:
            points = np.ascontiguousarray(self._traj.detach().numpy())
            made = finufft.Plan(
                nufft_type,
                tuple(int(size) for size in self._img_size),
                1,
                # pytorch_finufft's defaults for both transform types, plus the
                # two the operator passes itself. isign is -1 for type 1 too:
                # pytorch_finufft overrides FINUFFT's +1 there.
                eps=1e-6,
                isign=-1,
                dtype=torch.empty(0, dtype=self._dtype_meas).numpy().dtype,
                upsampfac=2.0,
                modeord=0,
            )
            made.setpts(points[0], points[1])
            plans[nufft_type] = made
        return plans[nufft_type]

    @torch.no_grad()
    def _GA(self, x: torch.Tensor) -> torch.Tensor:
        image = x.view(x.shape[0], *x.shape[-2:]).squeeze(0)
        cached = plan(self, 2) if image.ndim == 2 else None
        if cached is None:
            return forward(self, x)
        values = cached.execute(np.ascontiguousarray(image.to(self._dtype_meas).numpy()))
        return torch.from_numpy(values) * self._data_weight

    @torch.no_grad()
    def _AtGt(self, y: torch.Tensor) -> torch.Tensor:
        values = y.conj() * self._data_weight
        flat = values.reshape(-1, values.shape[-1])
        cached = plan(self, 1) if flat.shape[0] == 1 else None
        if cached is None:
            return adjoint(self, y)
        image = cached.execute(np.ascontiguousarray(flat[0].numpy()))
        return torch.from_numpy(image).reshape(*values.shape[:-1], *self._img_size)

    MeasOpPytorchFinufft._GA = _GA
    MeasOpPytorchFinufft._AtGt = _AtGt


def lanczos_largest_eigenvalue(matvec, size: int, dtype, max_restarts: int = 100) -> float:
    """Largest eigenvalue of a Hermitian positive semi-definite operator.

    `matvec` maps a flat vector to `A x`. ARPACK's Lanczos builds a Krylov
    subspace, so it converges on a clustered spectrum where a power iteration
    crawls: measured on this parameter space it is 17-25 operator applications
    against the 39-305 the power iteration takes, and lands within ~3e-6 of the
    true value against the power iteration's ~1e-4.

    `tol` is 1e-3, not the 1e-5 this started at: the answer only scales the
    operator, so upstream's own ~1e-4 is the accuracy that has to be beaten, and
    over 24 real operators from this parameter space 1e-5 cost 28.7 applications
    on average (max 33) for a 9e-11 median error where 1e-3 costs 21.2 (max 25)
    for 2.9e-6. Going further is where it stops being free - 1e-2 is 16.8
    applications but a 4.9e-4 median error, i.e. worse than the power iteration.

    The start vector is `ones`, not a random draw: with no seeding the upstream
    power iteration gives a different answer, and takes a different number of
    iterations, on every run of the same evaluation.

    ponytail: `max_restarts` bounds the worst case at ~600 applications; the
    caller falls back to the power iteration if it is ever hit.
    """
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh

    operator = LinearOperator((size, size), matvec=matvec, dtype=dtype)
    eigenvalues = eigsh(
        operator,
        k=1,
        which="LA",
        ncv=8,
        tol=1e-3,
        maxiter=max_restarts,
        v0=np.ones(size, dtype=dtype),
        return_eigenvectors=False,
    )
    return float(eigenvalues[0])


def patch_op_norm() -> None:
    """Solve `MeasOp.get_op_norm` with Lanczos instead of a power iteration.

    `get_op_norm` is the whole cost of an R2D2 evaluation that stops before the
    UNet passes, and most of one that does not: the operator's spectrum is
    tightly clustered, so upstream's power iteration needs 39-305 forward/
    adjoint NUFFT pairs to meet its 1e-5 relative-change test - and how many is
    a lottery decided by the unseeded `torch.randn` it starts from. That tail is
    what the sampler waits on, because a PolyChord round costs the slowest
    rank's evaluation, not the median one.

    Same quantity, same caching contract, ~2.5x fewer operator applications and
    ~1e-10 relative accuracy instead of ~1e-4.
    """
    import numpy as np
    import torch
    from ri_measurement_operator.pysrc.measOperator.meas_op import MeasOp
    from scipy.sparse.linalg import ArpackNoConvergence

    power_iteration = MeasOp.get_op_norm

    @torch.no_grad()
    def get_op_norm(self, compute_flag=False, rel_tol=1e-5, max_iter=500, verbose=False):
        if self._op_norm is not None and not compute_flag:
            return self._op_norm
        size = tuple(self._img_size)
        dtype = torch.empty(0, dtype=self._dtype).numpy().dtype

        def matvec(vector):
            image = torch.from_numpy(np.ascontiguousarray(vector, dtype=dtype))
            image = image.to(self._device).view(1, 1, *size)
            return self.adjoint_op(self.forward_op(image)).reshape(-1).cpu().numpy()

        try:
            self._op_norm = lanczos_largest_eigenvalue(matvec, int(np.prod(size)), dtype)
        except ArpackNoConvergence:
            self._op_norm = None
            return power_iteration(self, True, rel_tol, max_iter, verbose)
        return self._op_norm

    MeasOp.get_op_norm = get_op_norm


def serve_pool(fifo_dir: str) -> None:
    """One worker per FIFO pair in `fifo_dir`, all forked from a single warm-up.

    A run wants one worker per rank, and as separate interpreters they all
    `import torch` at the same moment: 0.89s on its own becomes 1.05-1.61s
    across 8 of them on a 20-CPU host, and the sampler waits for the slowest.
    Importing once and forking pays it at the solo price, and the children then
    share the ~300MB of it copy-on-write instead of holding a copy each.

    This process holds both ends of every pair open from before the warm-up
    until the pair's child exits, so a rank can attach while the warm-up is
    still running and get on with its own startup.
    """
    global _PEAK_FLOOR
    bases = sorted(str(path)[: -len(".in")] for path in Path(fifo_dir).glob("*.in"))
    # A rank attaches by write-opening `<rank>.in` - ENXIO until someone is
    # reading it - and then read-opening `<rank>.out`, which blocks until
    # someone is writing it. Opened only by the children, that is not until the
    # warm-up below has finished, and the rank spends ~1.2s idle in a window
    # where it has its own sampler to load and evaluation one's simulate and
    # convert to run. Held here instead, both opens succeed immediately.
    #
    # `.out` is O_RDWR because a plain write-open would block until the rank
    # read-opens, which is the wait this is removing; on a FIFO O_RDWR never
    # blocks. Holding a writer on it is also what keeps the rank's reply read
    # from hitting a spurious EOF in the moment between the fork and the child
    # opening its own end.
    keeper = {base: (os.open(f"{base}.in", os.O_RDONLY | os.O_NONBLOCK), os.open(f"{base}.out", os.O_RDWR))
              for base in bases}
    warm_imports()
    # Set before the fork, so every child reports at least what the warm-up it
    # inherited already cost.
    _PEAK_FLOOR = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    children = {}
    for base in bases:
        pid = os.fork()
        if pid:
            children[pid] = base
            continue
        status = 0
        try:
            # The child re-opens its own pair inside answer() and wants none of
            # the inherited ends, its own included: holding the request pipe's
            # read end open in two processes would keep a request alive after
            # the one that should answer it has gone.
            for fds in keeper.values():
                for fd in fds:
                    os.close(fd)
            answer(base)
        except Exception:
            traceback.print_exc()
            status = 1
        # _exit, not sys.exit: this child inherited the parent's atexit hooks and
        # stdio buffers and must run neither.
        os._exit(status)
    while children:
        pid, _status = os.wait()
        base = children.pop(pid, None)
        # Dropped as its worker goes, not at the end: while this process holds
        # them a rank whose worker died would write into a pipe nobody reads
        # and then wait forever for a reply. Closing here gives it the same
        # broken pipe and empty reply it gets when there is no pool at all.
        for fd in keeper.pop(base, ()):
            os.close(fd)


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


# What the stub imagers below do: `sys.exit(argv[1])`, behind the same
# `if __name__ == "__main__"` guard the real imager.py puts its body behind - so
# the warm-up's runpy pass over it runs its imports and nothing else.
_GUARDED_EXIT_IMAGER = "import sys\nif __name__ == '__main__':\n    sys.exit(int(sys.argv[1]))\n"


def self_check_serve_reply_stream() -> None:
    """Replies must carry JSON only, and a failed request must not end the worker."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        imager = root / "imager.py"
        imager.write_text(
            "import sys\n"
            "if __name__ == '__main__':\n"
            "    print('chatter on stdout')\n"
            "    print('chatter on stderr', file=sys.stderr)\n"
            "    sys.exit(int(sys.argv[1]))\n"
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
        imager.write_text(_GUARDED_EXIT_IMAGER)
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
        (root / "imager.py").write_text("from utils import vprint\n" + _GUARDED_EXIT_IMAGER)
        # Stands in for `import torch`: one line per interpreter that imports it.
        marker = root / "imports.log"
        # The 64MB is dropped again before the fork, so it only shows up in a
        # reply if the warm-up's high-water mark crossed the fork with it.
        package = root / "utils"
        package.mkdir()
        for submodule in _UTILS_SUBMODULES:
            (package / f"{submodule}.py").write_text("")
        (package / "misc.py").write_text(
            # The sleep is what makes the pre-opened FIFO ends observable: a
            # rank must be able to attach while this is still running.
            "import time\n"
            "time.sleep(0.5)\n"
            f"open({str(marker)!r}, 'a').write('x\\n')\n"
            "_ = bytearray(64 * 1024 * 1024)\n"
            "del _\n"
            "vprint = print\n"
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
            # What the pre-opened ends buy: rank 0 attaches - both FIFOs, not
            # just the request one - while the warm-up is still running.
            if rank == 0:
                assert not marker.exists(), "rank 0's request pipe waited for the warm-up"
            with os.fdopen(write_fd, "w") as requests, open(f"{base}.out") as replies:
                if rank == 0:
                    assert not marker.exists(), "rank 0's reply pipe waited for the warm-up"
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


def self_check_lanczos_largest_eigenvalue() -> None:
    """The solver must beat the power iteration on a tightly clustered spectrum.

    Only runs where scipy is installed, which is the R2D2 image - the other
    checks here stub out everything the image provides, but this one is about
    the numerics themselves. Run it there with:

        docker run --rm -v "$PWD:$PWD" --entrypoint python3
        ri-reproducibility/r2d2:cpu
        "$PWD/scripts/lib/nested_sampling/r2d2_serve.py" --self-check
    """
    try:
        import numpy as np
    except ImportError:
        print("r2d2 op-norm self-check skipped: no numpy")
        return
    try:
        import scipy.sparse.linalg  # noqa: F401
    except ImportError:
        print("r2d2 op-norm self-check skipped: no scipy")
        return

    size = 400
    # Eigenvalues 1.0, 0.999, 0.998, ...: the ratio a power iteration converges
    # at, mirroring what the measurement operator's spectrum looks like.
    spectrum = 1.0 - 0.001 * np.arange(size)
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((size, size)))[0]
    matrix = (basis * spectrum) @ basis.T
    largest = lanczos_largest_eigenvalue(lambda v: matrix @ v, size, np.float64)

    # The same 1e-5 relative-change test upstream uses, for the comparison the
    # patch exists to make.
    vector = np.ones(size) / np.sqrt(size)
    previous, applications = 1.0, 0
    while applications < 500:
        vector = matrix @ vector
        value = float(np.linalg.norm(vector))
        applications += 1
        if abs(value - previous) / previous < 1e-5:
            break
        previous, vector = value, vector / value
    assert abs(value - 1.0) > 1e-4, f"the power iteration was accurate in {applications}; pick a harder spectrum"
    # Against the power iteration rather than against an absolute number: this
    # spectrum is the worst case the patch is aimed at, and `tol` is a knob that
    # trades applications for accuracy, so what has to hold is the comparison
    # the patch exists to win, not whichever digit today's `tol` happens to hit.
    assert abs(largest - 1.0) < abs(value - 1.0) / 10, (largest, value)
    print("r2d2 op-norm self-check passed")


def self_check_nufft_plan_reuse() -> None:
    """A cached plan must give bit-identical transforms to a per-call one.

    Only runs inside the R2D2 image, where the measurement operator lives -
    same command as the op-norm check above.
    """
    sys.path.insert(0, str(IMAGER.parent))
    try:
        import torch
        from ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import (
            MeasOpPytorchFinufft,
        )
    except ImportError:
        print("r2d2 nufft plan self-check skipped: no R2D2 measurement operator")
        return

    def operator():
        torch.manual_seed(0)
        points = (torch.rand(1, 1, 200, dtype=torch.float64) - 0.5) * 6.0
        return MeasOpPytorchFinufft(u=points, v=points.flip(-1), img_size=(16, 16), dtype=torch.float64)

    torch.manual_seed(1)
    image = torch.randn(1, 1, 16, 16, dtype=torch.float64)
    upstream = operator()
    visibilities = upstream.forward_op(image)
    adjoint = upstream.adjoint_op(visibilities)

    patch_nufft_plans()
    patched = operator()
    # The forward transform interpolates off the grid, one output per sampling
    # point, and is bitwise reproducible. The adjoint spreads onto the grid,
    # where FINUFFT's own thread partitioning makes the summation order - and so
    # the last bit or two - vary between two identical calls once the points are
    # this sparse: 4e-16 relative over ten calls of upstream against itself at
    # 200 points on 16x16, and exactly zero at the 5616-points-on-128x128 the
    # PoC actually runs. Hence the tolerance on this one and not the other.
    assert torch.equal(patched.forward_op(image), visibilities), "the cached plan changed the forward transform"
    assert torch.allclose(patched.adjoint_op(visibilities), adjoint, rtol=1e-12, atol=0.0), (
        "the cached plan changed the adjoint transform"
    )
    # A batch of more than one is not what the cached plan was built for, so it
    # has to come back from upstream rather than silently image the first row.
    batch = torch.cat((visibilities, -visibilities), dim=0)
    assert torch.allclose(patched.adjoint_op(batch), upstream.adjoint_op(batch), rtol=1e-12, atol=0.0)
    print("r2d2 nufft plan self-check passed")


def self_check_lazy_utils() -> None:
    """A name must resolve to the same value, and no later submodule than needed."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "src" / "utils"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("from .misc import vprint\nfrom .noise import compute_tau\n")
        for submodule in _UTILS_SUBMODULES:
            (package / f"{submodule}.py").write_text("")
        (package / "misc.py").write_text("vprint = 'from misc'\n")
        (package / "noise.py").write_text("compute_tau = 'from noise'\n")

        global IMAGER
        saved_path, saved_imager = list(sys.path), IMAGER
        try:
            IMAGER = package.parent / "imager.py"
            sys.path.insert(0, str(package.parent))
            for name in [key for key in sys.modules if key == "utils" or key.startswith("utils.")]:
                del sys.modules[name]
            install_lazy_utils()
            import utils

            assert utils.vprint == "from misc", utils.vprint
            # `misc` comes before `noise`, so asking for a name it defines must
            # not have pulled the expensive tail of the list in behind it.
            assert "utils.noise" not in sys.modules, "the lazy shim imported past the name it found"
            assert utils.compute_tau == "from noise", utils.compute_tau
            assert "utils.noise" in sys.modules
        finally:
            for name in [key for key in sys.modules if key == "utils" or key.startswith("utils.")]:
                del sys.modules[name]
            sys.path[:] = saved_path
            IMAGER = saved_imager
    print("r2d2 lazy utils self-check passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check_lanczos_largest_eigenvalue()
        self_check_nufft_plan_reuse()
        self_check_lazy_utils()
        self_check_serve_reply_stream()
        self_check_serve_fifo()
        self_check_serve_pool()
    elif sys.argv[1:2] == ["--fifo-dir"]:
        serve_pool(sys.argv[2])
    else:
        serve(sys.argv[2] if sys.argv[1:2] == ["--fifo"] else None)
