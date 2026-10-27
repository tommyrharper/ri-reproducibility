#!/usr/bin/env python3
"""Serve warmed R2D2 imaging workers over JSON lines or rank-specific FIFOs."""

from __future__ import annotations

import json
import os
import resource
import signal
import runpy
import sys
import time
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
_CHECKPOINT_CACHE: dict[tuple[int, str], dict] = {}
_NORMALIZED_CHECKPOINT_CACHE: dict[int, tuple[dict, dict]] = {}
_NUFFT_PLAN_CACHE: dict[tuple[int, tuple[int, ...], str], object] = {}


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


def patch_checkpoint_loading() -> None:
    """Load each checkpoint series once and share it across forked workers."""
    import utils.util_model as util_model

    original = util_model.get_DNNs

    def load_state_dict_from_state_dict(net, state_dict):
        # Keyed by identity because a state dict is not hashable. The entry
        # holds the source dict as well as the normalised one, so the id
        # cannot be recycled onto a different dict and answer for it.
        key = id(state_dict)
        cached = _NORMALIZED_CHECKPOINT_CACHE.get(key)
        if cached is None:
            cached = (state_dict, {'.'.join(name.split('.')[1:]): state_dict[name]
                                   for name in state_dict})
            _NORMALIZED_CHECKPOINT_CACHE[key] = cached
        net.load_state_dict(cached[1], assign=True)
        return net

    def get_DNNs(num_iter: int, ckpt_path):
        key = (int(num_iter), os.path.abspath(str(ckpt_path)))
        if key not in _CHECKPOINT_CACHE:
            _CHECKPOINT_CACHE[key] = original(num_iter, ckpt_path)
        return _CHECKPOINT_CACHE[key]

    util_model.get_DNNs = get_DNNs
    util_model.load_state_dict_from_state_dict = load_state_dict_from_state_dict
    package = sys.modules.get("utils")
    if package is not None:
        package.get_DNNs = get_DNNs
    optimiser = sys.modules.get("optimiser.R2D2")
    if optimiser is not None:
        optimiser.get_DNNs = get_DNNs


_NETS: dict[str, object] = {}


def patch_net_reuse() -> None:
    """Build R2D2's U-Net once per worker instead of once per request.

    Upstream builds it with random Kaiming weights, then every iteration's
    `load_net` assigns a checkpoint's over all of them (strict, so none
    survive). The random init alone was 0.17s of a 1.8s request on CSD3.
    """
    optimiser = sys.modules["optimiser.R2D2"]
    create = optimiser.create_net_imaging

    def create_net_imaging(*args, **kwargs):
        key = repr((args, sorted(kwargs.items())))
        if key not in _NETS:
            _NETS[key] = create(*args, **kwargs)
        return _NETS[key]

    optimiser.create_net_imaging = create_net_imaging


def r2d2_device() -> str:
    """Where the U-Net runs: `cpu` (the default) or `cuda` (R2D2_DEVICE)."""
    return os.environ.get("R2D2_DEVICE", "cpu").strip().lower() or "cpu"


# Per worker process, keyed by (id of the cached checkpoint dict, iteration):
# that iteration's weights, already on the GPU.
_CUDA_WEIGHTS: dict[tuple[int, int], dict] = {}


def patch_cuda_network() -> None:
    """Run R2D2's U-Net passes on the GPU and leave everything else on the CPU.

    Upstream has one switch, `meas_op_on_gpu`, and it moves the measurement
    operator too - the NUFFTs, the operator norm and this file's FINUFFT plan
    cache, none of which gain from a GPU at 128x128 and ~3000 visibilities, and
    whose GPU path needs cufinufft. The U-Net is what the phase profiler put at
    98% of imaging time (docs/nested-sampling-speed.md), and upstream calls it
    in exactly one place, `model.forward`, with the weights assigned by
    `load_net` just before. So those two are replaced in `optimiser.R2D2`:
    `load_net` assigns weights that are already on the GPU (each worker copies
    each iteration's once, then reuses them), and `forward` takes its three
    image tensors to the GPU and brings the result back, so the residual and
    everything after it see the CPU tensor they always did.

    Patched in the pool's parent, which must never initialise CUDA itself: a
    CUDA context does not survive fork, so each forked worker makes its own on
    its first request. TF32 is turned off so the GPU computes the same float32
    convolutions the CPU does, rather than a faster, 10-bit-mantissa version.
    """
    import torch

    optimiser = sys.modules["optimiser.R2D2"]
    upstream_forward = optimiser.forward
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Wait for the U-Net in the kernel, not in a spin loop: the copy back alone
    # spin-waits, burning a CPU per waiting worker (R2D2_CUDA_BLOCKING_SYNC).
    blocking_sync = os.environ.get("R2D2_CUDA_BLOCKING_SYNC", "1") != "0"

    def load_net(net, cur_iter, layers, dnn_dict):
        if layers != 1:
            raise NotImplementedError("R2D2_DEVICE=cuda supports layers: 1 (R2D2), not R3D3")
        key = (id(dnn_dict), int(cur_iter))
        weights = _CUDA_WEIGHTS.get(key)
        if weights is None:
            weights = {".".join(name.split(".")[1:]): tensor.to(device)
                       for name, tensor in dnn_dict[f"N{cur_iter}"].items()}
            _CUDA_WEIGHTS[key] = weights
        net.load_state_dict(weights, assign=True)
        return net

    def to_gpu(tensor):
        return tensor.to(device, non_blocking=True) if isinstance(tensor, torch.Tensor) else tensor

    def forward(layers, i, net, res_n, output_n, mean, eps=1e-110, dirty=None, PSF=None,
                input_order="res_rec"):
        output = upstream_forward(layers, i, net, to_gpu(res_n), to_gpu(output_n), to_gpu(mean),
                                  eps, to_gpu(dirty), to_gpu(PSF), input_order)
        if blocking_sync:
            done = torch.cuda.Event(blocking=True)
            done.record()
            done.synchronize()
        return output.to(res_n.device)

    optimiser.load_net = load_net
    optimiser.forward = forward


def job_numa_cpus() -> dict[int, list[int]]:
    """This process's allowed CPUs, by the NUMA node holding them."""
    allowed = os.sched_getaffinity(0)
    nodes = {}
    for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
        cpus = set()
        for part in (path / "cpulist").read_text().strip().split(","):
            if part:
                low, _, high = part.partition("-")
                cpus.update(range(int(low), int(high or low) + 1))
        if cpus & allowed:
            nodes[int(path.name[4:])] = sorted(cpus & allowed)
    return nodes


def job_numa_nodes() -> list[int]:
    """NUMA nodes holding at least one CPU this process may run on."""
    return list(job_numa_cpus())


def set_mempolicy(mode: int, nodes: list[int]) -> bool:
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        return False
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    mask = ctypes.c_ulong(sum(1 << node for node in nodes))
    # 238 is x86_64's set_mempolicy; MPOL_DEFAULT takes no mask.
    return libc.syscall(238, mode, ctypes.byref(mask) if nodes else None, 64 if nodes else 0) == 0


def weights_per_node(nodes: dict[int, list[int]] | None = None) -> dict[int, list[int]]:
    """Nodes to give their own checkpoint copy, or {} for one shared copy.

    R2D2_WEIGHTS_PER_NODE=1 or 0 forces it. The default, auto, takes it when
    every node holds at least half its fair share of the job's CPUs: a whole
    sapphire node's search ran +19% (docs/csd3-speed.md, round 106). A
    lopsided split like 32/6 keeps the shared copy, since pinning a node's
    few CPUs cost round 98's per-node copies 5-6% there.
    """
    mode = os.environ.get("R2D2_WEIGHTS_PER_NODE", "auto")
    nodes = job_numa_cpus() if nodes is None else nodes
    if mode == "0" or len(nodes) < 2:
        return {}
    if mode != "1":
        smallest, total = min(len(cpus) for cpus in nodes.values()), sum(len(cpus) for cpus in nodes.values())
        if 2 * smallest * len(nodes) < total:
            return {}
    return nodes


def worker_nodes(count: int, nodes: dict[int, list[int]]) -> list[int]:
    """Worker i's node, in proportion to each node's share of the job's CPUs.

    Round-robin would put half of 38 workers on a 6-CPU node of a 32/6
    allocation; this puts 32 and 6.
    """
    slots = [node for node, cpus in nodes.items() for _ in cpus]
    return [slots[i * len(slots) // count] for i in range(count)] if slots else []


@contextmanager
def interleaved_pages():
    """Spread pages allocated inside across the job's NUMA nodes.

    The pool parent loads every checkpoint once and its forked workers all
    read those pages. Left to first touch, they sit on whichever socket the
    parent ran on, and on a whole two-socket node half the workers stream
    weights across the interconnect. R2D2_INTERLEAVE_WEIGHTS=0 turns it off.
    """
    nodes = job_numa_nodes() if os.environ.get("R2D2_INTERLEAVE_WEIGHTS", "1") != "0" else []
    interleaving = len(nodes) > 1 and set_mempolicy(3, nodes)  # MPOL_INTERLEAVE
    try:
        yield
    finally:
        if interleaving:
            set_mempolicy(0, [])  # MPOL_DEFAULT


# One checkpoint cache per NUMA node (R2D2_WEIGHTS_PER_NODE); a worker swaps
# its node's in for _CHECKPOINT_CACHE after the fork.
_NODE_CHECKPOINTS: dict[int, dict] = {}


def load_checkpoints(checkpoint_path: Path) -> None:
    def load():
        sys.modules["utils"].get_DNNs(int(os.environ.get("R2D2_NUM_ITER", "25")), str(checkpoint_path))

    # A CUDA worker reads its weights from the GPU, so only CPU runs pin.
    nodes = weights_per_node() if r2d2_device() != "cuda" else {}
    if not nodes:
        with interleaved_pages():
            load()
        return
    for node in nodes:
        _CHECKPOINT_CACHE.clear()
        bound = set_mempolicy(2, [node])  # MPOL_BIND: this copy's pages on its node
        try:
            load()
        finally:
            if bound:
                set_mempolicy(0, [])
        _NODE_CHECKPOINTS[node] = dict(_CHECKPOINT_CACHE)


def enter_node(node: int, nodes: dict[int, list[int]]) -> None:
    """Run a forked worker on its node's CPUs, reading its node's weights."""
    os.sched_setaffinity(0, nodes[node])
    _CHECKPOINT_CACHE.clear()
    _CHECKPOINT_CACHE.update(_NODE_CHECKPOINTS[node])


def warm_imports() -> None:
    os.chdir(R2D2_HOME)
    sys.path.insert(0, str(IMAGER.parent))
    install_lazy_utils()
    # Reuse imager.py's import block; suppress import-time output from the
    # JSON reply stream.
    with redirect_fds(Path(os.devnull), Path(os.devnull)):
        try:
            runpy.run_path(str(IMAGER), run_name="__warmup__")
            import torch

            # On the pinned CPU build this selects a faster MKLDNN convolution
            # path: 59.7-60.3ms versus 65.0-70.6ms for the 128x128 U-Net,
            # with bit-identical output across three fresh probes.
            torch.backends.mkldnn.deterministic = True
            interop_threads = int(os.environ.get("R2D2_INTEROP_THREADS", "0"))
            if interop_threads:
                torch.set_num_interop_threads(max(1, interop_threads))
            patch_nufft_plans()
            patch_normal_operator()
            patch_checkpoint_loading()
            patch_net_reuse()
            if r2d2_device() == "cuda":
                patch_cuda_network()
            checkpoint_path = Path(os.environ.get("R2D2_CKPT_PATH", "/checkpoints/R2D2_A1"))
            if checkpoint_path.is_dir():
                # Load before fork: tensor pages stay shared until a child
                # writes them, instead of paying torch.load once per rank.
                load_checkpoints(checkpoint_path)
            # `create_meas_op` imports this backend lazily; preload it.
            from ri_measurement_operator.pysrc.measOperator import (  # noqa: F401
                meas_op_nufft_pytorch_finufft,
            )
        except Exception:
            traceback.print_exc()


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
        if nufft_type not in plans:
            points = np.ascontiguousarray(self._traj.detach().numpy())
            shape = tuple(int(size) for size in self._img_size)
            dtype = torch.empty(0, dtype=self._dtype_meas).numpy().dtype
            # The request's `ncpus`, which R2D2 hands torch.set_num_threads().
            # FINUFFT's own OpenMP runtime would stay at OMP_NUM_THREADS.
            nthreads = torch.get_num_threads()
            cache_key = (nufft_type, shape, dtype.str, nthreads)
            made = _NUFFT_PLAN_CACHE.get(cache_key)
            if made is None:
                made = finufft.Plan(
                    nufft_type,
                    shape,
                    1,
                    # pytorch_finufft's defaults for both transform types, plus the
                    # two the operator passes itself. isign is -1 for type 1 too:
                    # pytorch_finufft overrides FINUFFT's +1 there.
                    eps=1e-6,
                    isign=-1,
                    dtype=dtype,
                    upsampfac=2.0,
                    modeord=0,
                    nthreads=nthreads,
                )
                _NUFFT_PLAN_CACHE[cache_key] = made
            # Workers process one request at a time, so one shared plan can be
            # retargeted for each new operator without concurrent users.
            made.setpts(points[0], points[1])
            plans[nufft_type] = made
        return plans[nufft_type]

    @torch.inference_mode()
    def _GA(self, x: torch.Tensor) -> torch.Tensor:
        image = x.view(x.shape[0], *x.shape[-2:]).squeeze(0)
        cached = plan(self, 2) if image.ndim == 2 else None
        if cached is None:
            return forward(self, x)
        values = cached.execute(np.ascontiguousarray(image.to(self._dtype_meas).numpy()))
        return torch.from_numpy(values) * self._data_weight

    @torch.inference_mode()
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


# Kernel accuracy. Built once per request, so it can be far tighter than the
# 1e-6 of the NUFFT pairs it replaces.
NORMAL_KERNEL_EPS = 1e-9


class _Deferred:
    def __init__(self, image):
        self.image = image


def normal_operator(op):
    """Re(A^H A) of `op` as a 2N x 2N FFT convolution, or None if unsupported.

    With x real, A^H A x (upstream's adjoint of its forward, weights applied
    in both) is x correlated with g(d) = sum_k w_k^2 exp(+i k.d), d in
    [-(N-1), N-1]^2. One type-1 NUFFT onto 2N modes gives g, after which each
    application is two tiny FFTs instead of two NUFFTs over every visibility.
    """
    import finufft
    import numpy as np
    import torch

    if hasattr(op, "_ri_normal"):
        return op._ri_normal
    op._ri_normal = None
    weight, precond = op._data_weight, op._precond_weight
    if (str(op._device or "cpu") != "cpu" or not op._real_flag or not torch.all(precond == 1)
            or torch.any(weight.imag != 0)):
        return None
    points = op._traj.detach().to(torch.float64).numpy()
    strengths = np.ascontiguousarray(
        np.broadcast_to(weight.real.to(torch.float64).numpy() ** 2, (*weight.shape[:-1], points.shape[-1]))
        .reshape(-1, points.shape[-1])[0], dtype=np.complex128)
    rows, cols = (int(size) for size in op._img_size)
    # modeord 0 puts mode m at index m + N, so ifftshift puts d at index d mod 2N.
    kernel = finufft.nufft2d1(points[0], points[1], strengths, (2 * rows, 2 * cols),
                              eps=NORMAL_KERNEL_EPS, isign=-1, modeord=0,
                              nthreads=torch.get_num_threads())
    spectrum = np.fft.rfft2(np.fft.ifftshift(kernel.real))
    dtype = op._dtype

    def apply(x):
        padded = np.zeros((2 * rows, 2 * cols))
        padded[:rows, :cols] = x.detach().reshape(rows, cols).to(torch.float64).numpy()
        out = np.fft.irfft2(np.fft.rfft2(padded) * spectrum, s=padded.shape)[:rows, :cols]
        return torch.from_numpy(np.ascontiguousarray(out)).reshape(x.shape).to(dtype)

    op._ri_normal = apply
    return apply


def patch_normal_operator() -> None:
    """Replace R2D2's per-iteration adjoint(forward(model)) with `normal_operator`.

    `R2D2.run` uses the operator only in that one expression, so for the run
    forward_op defers and adjoint_op applies the convolution. The PSF is the
    same operator on a dirac.
    """
    import torch
    from ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import (
        MeasOpPytorchFinufft,
    )

    R2D2 = sys.modules["optimiser.R2D2"].R2D2
    upstream_run, upstream_psf = R2D2.run, MeasOpPytorchFinufft.get_psf

    def get_psf(self):
        apply = normal_operator(self)
        if apply is None:
            return upstream_psf(self)
        dirac = torch.zeros(1, 1, *self._img_size, dtype=self._dtype)
        dirac[0, 0, self._img_size[0] // 2, self._img_size[1] // 2] = 1.0
        return apply(dirac)

    def run(self):
        op = self._meas_op
        apply = normal_operator(op) if isinstance(op, MeasOpPytorchFinufft) else None
        if apply is None:
            return upstream_run(self)
        op.forward_op = _Deferred
        op.adjoint_op = lambda deferred: apply(deferred.image)
        try:
            return upstream_run(self)
        finally:
            del op.forward_op, op.adjoint_op

    MeasOpPytorchFinufft.get_psf = get_psf
    R2D2.run = run


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
    nodes = weights_per_node() if _NODE_CHECKPOINTS else {}
    node_of = dict(zip(bases, worker_nodes(len(bases), nodes)))

    def fork(base: str, replacement: bool = False) -> None:
        pid = os.fork()
        if pid:
            children[pid] = base
            return
        status = 0
        try:
            # The child re-opens its own pair inside answer() and wants none of
            # the inherited ends, its own included: holding the request pipe's
            # read end open in two processes would keep a request alive after
            # the one that should answer it has gone. First thing, before any
            # pause: a copy of a sibling's ends held here is a reader a rank
            # can attach to while nobody is there to answer.
            for fds in keeper.values():
                for fd in fds:
                    os.close(fd)
            # In the child, so the parent stays at os.wait() and lets go of a
            # dead worker's ends the moment it dies. The pause bounds the churn
            # if the replacement cannot serve either (its FIFO directory gone,
            # say); the rank's own retry waits at least this long.
            if replacement:
                time.sleep(1.0)
            if base in node_of:
                enter_node(node_of[base], nodes)
            answer(base)
        except Exception:
            traceback.print_exc()
            status = 1
        # _exit, not sys.exit: this child inherited the parent's atexit hooks and
        # stdio buffers and must run neither.
        os._exit(status)

    for base in bases:
        fork(base)
    # Serves until killed: a worker that exits - on EOF when its rank goes, or
    # dying mid-request - is forked again from this warm parent, and the rank
    # (or the next attempt's rank) reconnects to the same FIFOs.
    while True:
        pid, _status = os.wait()
        base = children.pop(pid)
        # Dropped as its worker goes, not at the end: while this process holds
        # them a rank whose worker died would write into a pipe nobody reads
        # and then wait forever for a reply. Closing here gives it the same
        # broken pipe and empty reply it gets when there is no pool at all,
        # and the replacement's own opens then wait for the rank to come back.
        for fd in keeper.pop(base, ()):
            os.close(fd)
        fork(base, replacement=True)


def serve(fifo_base: str | None = None) -> None:
    warm_imports()
    answer(fifo_base)


def answer(fifo_base: str | None) -> None:
    if fifo_base is None:
        requests, replies = sys.stdin, os.fdopen(os.dup(1), "w")
    else:
        # How the rank kills this worker if it wedges (FifoWorker.kill in
        # common.py); the pool then forks a replacement.
        Path(f"{fifo_base}.pid").write_text(f"{os.getpid()}\n")
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
        # Its own session, so the pool and the workers it forked go together at
        # the end - the way the run script kills a pool's process group.
        worker = subprocess.Popen(
            [sys.executable, __file__, "--fifo-dir", str(pool)],
            env={**os.environ, "R2D2_IMAGER": str(root / "imager.py"), "R2D2_HOME": str(root)},
            start_new_session=True,
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
        # The point of forking rather than starting one interpreter per rank.
        assert marker.read_text() == "x\n", f"the warm-up ran more than once: {marker.read_text()!r}"
        # Both workers saw EOF above; the pool forks replacements on the same
        # FIFOs rather than exiting, so a rank that comes back is served - and
        # still by a fork of the one warm-up.
        base = pool / "1"
        pid_before = int((pool / "1.pid").read_text())
        # The replacement records its pid before it opens the FIFOs, and a
        # connect before that could still land on the worker that is exiting.
        deadline = time.monotonic() + 30
        while int((pool / "1.pid").read_text()) == pid_before:
            assert time.monotonic() < deadline, "rank 1 was not given a replacement worker"
            time.sleep(0.01)
        while True:
            try:
                write_fd = os.open(f"{base}.in", os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError:
                assert time.monotonic() < deadline, "the replacement never opened its request pipe"
                time.sleep(0.01)
        os.set_blocking(write_fd, True)
        with os.fdopen(write_fd, "w") as requests, open(f"{base}.out") as replies:
            request = {"argv": ["5"], "stdout": str(root / "o.log"), "stderr": str(root / "e.log")}
            requests.write(json.dumps(request) + "\n")
            requests.flush()
            assert json.loads(replies.readline())["returncode"] == 5
        assert marker.read_text() == "x\n", "the replacement was not a fork of the warm parent"
        assert worker.poll() is None, "the pool exited instead of serving until killed"
        os.killpg(worker.pid, signal.SIGKILL)
        worker.wait(timeout=30)
    print("r2d2 serve pool self-check passed")


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
    second = operator()
    second._traj = second._traj + 0.01
    expected_second = second.forward_op(image)

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

    assert torch.allclose(second.forward_op(image), expected_second, rtol=1e-12, atol=0.0)
    assert patched._ri_nufft_plans[2] is second._ri_nufft_plans[2]
    print("r2d2 nufft plan self-check passed")


def self_check_normal_operator() -> None:
    sys.path.insert(0, str(IMAGER.parent))
    try:
        import torch
        from ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import (
            MeasOpPytorchFinufft,
        )
    except ImportError:
        print("r2d2 normal operator self-check skipped: no R2D2 measurement operator")
        return

    torch.manual_seed(0)
    count = 5000
    points = (torch.rand(1, 1, count, dtype=torch.float64) - 0.5) * 6.0
    weights = torch.rand(1, 1, count, dtype=torch.float64) + 0.5
    image = torch.randn(1, 1, 24, 20, dtype=torch.float32)
    for natural in (torch.ones(1, 1), weights):
        op = MeasOpPytorchFinufft(u=points, v=points.flip(-1) * 0.7, img_size=(24, 20),
                                  natural_weight=natural, image_weight=weights.sqrt(), dtype=torch.float64)
        expected = op.adjoint_op(op.forward_op(image))
        dirac = torch.zeros(1, 1, 24, 20, dtype=torch.float64)
        dirac[0, 0, 12, 10] = 1.0
        expected_psf = op.adjoint_op(op.forward_op(dirac))
        apply = normal_operator(op)
        got = apply(image)
        assert got.shape == expected.shape and got.dtype == expected.dtype, (got.shape, got.dtype)
        # Upstream's two NUFFTs at eps 1e-6 are the error here, not the kernel.
        for result, reference in ((got, expected), (apply(dirac), expected_psf)):
            error = float((result - reference).abs().max() / reference.abs().max())
            assert error < 1e-5, error
    op._precond_weight = torch.full((1, 1), 2.0, dtype=torch.complex128)
    del op._ri_normal
    assert normal_operator(op) is None, "a preconditioned operator must stay on upstream's NUFFTs"
    print("r2d2 normal operator self-check passed")


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


def self_check_checkpoint_cache() -> None:
    calls = []
    package = types.ModuleType("utils")
    module = types.ModuleType("utils.util_model")

    def loader(num_iter, ckpt_path):
        calls.append((num_iter, ckpt_path))
        return {"N1": object()}

    module.get_DNNs = loader
    module.load_state_dict_from_state_dict = lambda net, state_dict: net
    optimiser = types.ModuleType("optimiser.R2D2")
    optimiser.get_DNNs = loader
    sys.modules["utils"] = package
    sys.modules["utils.util_model"] = module
    sys.modules["optimiser.R2D2"] = optimiser
    _CHECKPOINT_CACHE.clear()
    _NORMALIZED_CHECKPOINT_CACHE.clear()
    try:
        patch_checkpoint_loading()
        package.get_DNNs(1, "/checkpoints/R2D2_A1")
        optimiser.get_DNNs(1, "/checkpoints/R2D2_A1")
        assert calls == [(1, "/checkpoints/R2D2_A1")], calls
        state_dict = {"unet.layer.weight": object()}

        class Net:
            def load_state_dict(self, values, **kwargs):
                assert list(values) == ["layer.weight"]
                assert kwargs == {"assign": True}

        module.load_state_dict_from_state_dict(Net(), state_dict)
        module.load_state_dict_from_state_dict(Net(), state_dict)
        assert len(_NORMALIZED_CHECKPOINT_CACHE) == 1
        # The entry pins its source dict, so the id it is keyed by stays that
        # dict's for as long as the cache holds the answer.
        assert _NORMALIZED_CHECKPOINT_CACHE[id(state_dict)][0] is state_dict
    finally:
        _CHECKPOINT_CACHE.clear()
        _NORMALIZED_CHECKPOINT_CACHE.clear()
        del sys.modules["utils.util_model"]
        del sys.modules["utils"]
        del sys.modules["optimiser.R2D2"]
    print("r2d2 checkpoint cache self-check passed")


def self_check_net_reuse() -> None:
    calls = []
    optimiser = types.ModuleType("optimiser.R2D2")
    optimiser.create_net_imaging = lambda **kwargs: calls.append(kwargs) or object()
    sys.modules["optimiser.R2D2"] = optimiser
    _NETS.clear()
    try:
        patch_net_reuse()
        first = optimiser.create_net_imaging(layers=1, num_chans=64, architecture="unet", device="cpu")
        assert optimiser.create_net_imaging(layers=1, num_chans=64, architecture="unet", device="cpu") is first
        assert optimiser.create_net_imaging(layers=1, num_chans=32, architecture="unet", device="cpu") is not first
        assert len(calls) == 2, calls
    finally:
        _NETS.clear()
        del sys.modules["optimiser.R2D2"]
    print("r2d2 net reuse self-check passed")


def self_check_interleaved_pages() -> None:
    nodes = job_numa_nodes()
    if Path("/sys/devices/system/node/node0").is_dir():
        assert nodes, "no NUMA node holds an allowed CPU"
    with interleaved_pages():
        block = bytearray(1 << 20)
    assert len(block) == 1 << 20
    print(f"r2d2 interleaved pages self-check passed ({len(nodes)} node(s))")


def self_check_worker_nodes() -> None:
    lopsided = {0: list(range(32)), 1: list(range(32, 38))}
    placed = worker_nodes(38, lopsided)
    assert (placed.count(0), placed.count(1)) == (32, 6), placed
    assert worker_nodes(4, {0: [0, 2], 1: [1, 3]}) == [0, 0, 1, 1]
    assert worker_nodes(3, {}) == []
    os.environ.pop("R2D2_WEIGHTS_PER_NODE", None)
    whole = {0: [0, 2], 1: [1, 3]}
    assert weights_per_node(whole) == whole, "a balanced split gets per-node copies"
    assert weights_per_node(lopsided) == {}, "a 32/6 split keeps one shared copy"
    assert weights_per_node({0: [0, 1]}) == {}
    os.environ["R2D2_WEIGHTS_PER_NODE"] = "1"
    assert weights_per_node(lopsided) == lopsided
    os.environ["R2D2_WEIGHTS_PER_NODE"] = "0"
    assert weights_per_node(whole) == {}
    os.environ.pop("R2D2_WEIGHTS_PER_NODE")
    print("r2d2 worker nodes self-check passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check_net_reuse()
        self_check_nufft_plan_reuse()
        self_check_normal_operator()
        self_check_lazy_utils()
        self_check_checkpoint_cache()
        self_check_serve_reply_stream()
        self_check_serve_fifo()
        self_check_serve_pool()
        self_check_interleaved_pages()
        self_check_worker_nodes()
    elif sys.argv[1:2] == ["--fifo-dir"]:
        serve_pool(sys.argv[2])
    else:
        serve(sys.argv[2] if sys.argv[1:2] == ["--fifo"] else None)
