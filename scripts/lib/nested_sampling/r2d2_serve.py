#!/usr/bin/env python3
"""Serve warmed R2D2 imaging workers over JSON lines or rank-specific FIFOs."""

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


# Forked workers inherit warm-up pages but reset ru_maxrss, so retain the
# parent's high-water mark or memory reports omit import cost.
_PEAK_FLOOR = 0


def peak_memory_bytes() -> int:
    return max(_PEAK_FLOOR, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


# Avoid eager imports from `utils`; the last two pull unused heavy dependencies.
_UTILS_SUBMODULES = ("args", "data", "evaluate", "io", "meas_op", "misc", "util_model", "noise", "util_training")


def install_lazy_utils() -> None:
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
    os.chdir(R2D2_HOME)
    sys.path.insert(0, str(IMAGER.parent))
    install_lazy_utils()
    # Reuse imager.py's import block; suppress import-time output from the
    # JSON reply stream.
    with redirect_fds(Path(os.devnull), Path(os.devnull)):
        try:
            runpy.run_path(str(IMAGER), run_name="__warmup__")
            patch_op_norm()
            patch_nufft_plans()
            # `create_meas_op` imports this backend lazily; preload it after
            # patching the operator norm.
            from ri_measurement_operator.pysrc.measOperator import (  # noqa: F401
                meas_op_nufft_pytorch_finufft,
            )
        except Exception:
            traceback.print_exc()


# FINUFFT's upsampling factor for the operator-norm matvecs only; the imaging
# transforms keep upstream's 2.0. `get_op_norm` produces one number, the
# `1/sqrt(2L)` target-dynamic-range heuristic, and the Lanczos solve that
# produces it already stops at a 1e-3 relative tolerance - against which 1.25
# costs nothing measurable: over 12 real operators from this parameter space the
# eigenvalue moves 7.3e-8 at worst and the application count is unchanged at
# 19.7 (the per-transform error is 5.1e-6, and averaging over a 128x128
# eigenvector is what turns it into 1e-8).
#
# What it buys is the FFT: 1.25 makes the padded grid 160x160 instead of
# 256x256, measured at 0.598ms per forward/adjoint pair against 0.893ms solo and
# - because that FFT is what saturates memory bandwidth - a whole imaging
# request at 51ms against 69ms with eight of them running at once. Over those 12
# operators the whole solve is 18.1ms at 1.25, 21.4ms at 1.5 and 26.0ms at 2.0.
# Loosening `eps` instead is nearly free here (0.829ms per pair at 1e-3 against
# 0.893ms at 1e-6): with ~3000 visibilities against 128x128 modes this transform
# is FFT-bound, not spreading-bound.
OP_NORM_UPSAMPFAC = 1.25


def patch_nufft_plans() -> None:
    """Cache CPU FINUFFT plans; unsupported devices and batches use upstream."""
    import finufft
    import numpy as np
    import torch
    from ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import (
        MeasOpPytorchFinufft,
    )

    forward, adjoint = MeasOpPytorchFinufft._GA, MeasOpPytorchFinufft._AtGt

    def plan(self, nufft_type: int):
        if str(self._device or "cpu") != "cpu":
            return None
        plans = getattr(self, "_ri_nufft_plans", None)
        if plans is None:
            plans = {}
            setattr(self, "_ri_nufft_plans", plans)
        upsampfac = getattr(self, "_ri_upsampfac", 2.0)
        key = (nufft_type, upsampfac)
        if key not in plans:
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
                upsampfac=upsampfac,
                modeord=0,
            )
            made.setpts(points[0], points[1])
            plans[key] = made
        return plans[key]

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


def lanczos_largest_eigenvalue(matvec, size: int, dtype, v0=None, max_restarts: int = 100):
    """Return largest eigenpair; caller falls back on non-convergence."""
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh

    operator = LinearOperator((size, size), matvec=matvec, dtype=dtype)
    if v0 is None:
        v0 = np.ones(size, dtype=dtype)
    eigenvalues, eigenvectors = eigsh(
        operator,
        k=1,
        which="LA",
        ncv=8,
        tol=1e-3,
        maxiter=max_restarts,
        v0=v0,
        return_eigenvectors=True,
    )
    return float(eigenvalues[0]), np.ascontiguousarray(eigenvectors[:, 0], dtype=dtype)


# Reuse the first converged eigenvector for every operator in this worker.
# Operators share a dominant subspace, so this cuts real solves from 19.6ms to
# 14.0ms (24-operator median) while keeping eigenvalue variation below 4e-6,
# far below ARPACK's 1e-3 tolerance. Freeze the first vector: rolling starts
# have the same mean cost but a worse 25-application maximum versus 17.
_reused_start_vector = None


def patch_op_norm() -> None:
    """Patch `MeasOp.get_op_norm` with the cached Lanczos implementation."""
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

        global _reused_start_vector
        length = int(np.prod(size))
        v0 = _reused_start_vector
        if v0 is not None and (v0.size != length or v0.dtype != dtype):
            v0 = None

        # The Lanczos matvecs run on a coarser FINUFFT upsampling grid than the
        # imaging transforms do; see OP_NORM_UPSAMPFAC.
        self._ri_upsampfac = OP_NORM_UPSAMPFAC
        try:
            self._op_norm, eigenvector = lanczos_largest_eigenvalue(matvec, length, dtype, v0)
        except ArpackNoConvergence:
            self._op_norm = None
            return power_iteration(self, True, rel_tol, max_iter, verbose)
        finally:
            self._ri_upsampfac = 2.0
        if _reused_start_vector is None:
            _reused_start_vector = eigenvector
        return self._op_norm

    MeasOp.get_op_norm = get_op_norm


def serve_pool(fifo_dir: str) -> None:
    """Fork one warmed worker per FIFO pair in `fifo_dir`."""
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
    warm_imports()
    answer(fifo_base)


def answer(fifo_base: str | None) -> None:
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


# Stub imager exits under real imager's guard, so warm-up imports only.
_GUARDED_EXIT_IMAGER = "import sys\nif __name__ == '__main__':\n    sys.exit(int(sys.argv[1]))\n"


def self_check_serve_reply_stream() -> None:
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
    largest, eigenvector = lanczos_largest_eigenvalue(lambda v: matrix @ v, size, np.float64)

    # A neighbouring operator, the way one evaluation's is a neighbour of the
    # last one's: `matrix`'s converged eigenvector must start it in fewer
    # applications than `ones` does, which is the whole point of reusing it.
    nudged = matrix + 1e-3 * (basis * np.roll(spectrum, 1)) @ basis.T
    counts = {}
    for label, start in (("ones", None), ("reused", eigenvector)):
        applied = [0]

        def count(v, applied=applied):
            applied[0] += 1
            return nudged @ v

        counts[label] = (lanczos_largest_eigenvalue(count, size, np.float64, start)[0], applied[0])
    assert counts["reused"][1] < counts["ones"][1], counts
    assert abs(counts["reused"][0] - counts["ones"][0]) / counts["ones"][0] < 1e-3, counts

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

    # `get_op_norm` runs its matvecs on OP_NORM_UPSAMPFAC plans; the imaging
    # transforms must still come off the 2.0 ones afterwards. Cheap to get
    # wrong - one missed restore and every later transform silently changes.
    patch_op_norm()
    assert patched.get_op_norm(True) > 0.0
    assert torch.equal(patched.forward_op(image), visibilities), "the op-norm plan leaked into imaging"
    assert set(patched._ri_nufft_plans) == {(1, OP_NORM_UPSAMPFAC), (1, 2.0), (2, OP_NORM_UPSAMPFAC), (2, 2.0)}
    print("r2d2 nufft plan self-check passed")


def self_check_lazy_utils() -> None:
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
