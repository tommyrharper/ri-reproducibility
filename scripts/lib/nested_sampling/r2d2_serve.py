#!/usr/bin/env python3
"""Serve warmed R2D2 imaging workers over JSON lines or rank-specific FIFOs."""

from __future__ import annotations

import json
import os
import resource
import signal
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


class LazyCheckpoints(dict):
    """A checkpoint series that loads an entry the first time something asks for it."""

    def __init__(self, load):
        super().__init__()
        self.load = load

    def __missing__(self, name: str):
        state_dict = self[name] = self.load(name)
        return state_dict


def load_checkpoint_series(num_iter: int, ckpt_path) -> dict:
    """Upstream `get_DNNs`, with its torch.loads on 4 threads.

    Same checks, same dict. torch.load releases the GIL while it reads, so
    the pool's 25 checkpoints (6.2 GB) load in 0.9s instead of 2.0s
    (docs/csd3-speed.md, round 203). Pool threads inherit the caller's NUMA
    policy, so `load_checkpoints` still places the pages.
    """
    import glob
    from concurrent.futures import ThreadPoolExecutor

    import torch

    paths = []
    for i in range(num_iter):
        dnn = glob.glob(os.path.join(ckpt_path, f'*N{i+1}.ckpt'))
        if len(dnn) == 0:
            raise ValueError(f'Checkpoint for N{i+1} not found')
        elif len(dnn) > 1:
            raise ValueError(f'Checkpoint conflict for N{i+1}')
        paths.append(dnn[0])

    weights = checkpoint_weights_dir(ckpt_path)

    def load(path):
        cached = weights / os.path.basename(path) if weights else None
        if cached is not None and cached.exists():
            return torch.load(cached, map_location=torch.device('cpu'), weights_only=True)
        state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=False)['state_dict']
        if cached is not None:
            partial = cached.with_suffix(f'.{os.getpid()}.tmp')
            try:
                torch.save(state_dict, partial)
                partial.replace(cached)
            except OSError:
                partial.unlink(missing_ok=True)  # a full quota: the next pool tries again
        return state_dict

    # A CPU onnx pool forwards through its sessions, so only an export (a new
    # torch or checkpoint) or a request ONNX cannot serve needs these weights
    # (round 367).
    if weights is not None:
        return LazyCheckpoints(lambda name: load(paths[int(name[1:]) - 1]))
    with ThreadPoolExecutor(max(1, min(4, len(paths)))) as pool:
        return {f'N{i+1}': state_dict for i, state_dict in enumerate(pool.map(load, paths))}


def checkpoint_weights_dir(ckpt_path) -> Path | None:
    """Where a CPU onnx pool keeps each checkpoint's state_dict alone, or None.

    A Lightning checkpoint is half optimiser state (248 MB, 124 MB of it
    weights), and a cold pool on a new node reads all 6.2 GB off Lustre
    before its first request (docs/csd3-speed.md, round 257). Keyed like the
    exported models, by the checkpoints' names, sizes and mtimes.
    """
    if r2d2_device() != "cpu" or r2d2_runtime() != "onnx" or not ONNX_DIR.is_dir():
        return None
    import hashlib

    ckpt_path = Path(ckpt_path)
    stats = sorted((p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in ckpt_path.glob("*.ckpt"))
    key = hashlib.sha1(repr((str(ckpt_path), stats)).encode()).hexdigest()[:16]
    weights = ONNX_DIR / "weights" / key
    try:
        weights.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return weights


def patch_checkpoint_loading() -> None:
    """Load each checkpoint series once and share it across forked workers."""
    import utils.util_model as util_model

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
            _CHECKPOINT_CACHE[key] = load_checkpoint_series(num_iter, ckpt_path)
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


def r2d2_runtime() -> str:
    """What runs a U-Net forward: `torch`, `onnx` (CPU; the CPU default) or `trt` (GPU) (R2D2_RUNTIME)."""
    return os.environ.get("R2D2_RUNTIME", "torch").strip().lower() or "torch"


# Where the run script binds onnxruntime and the exported checkpoints.
ONNX_DIR = Path(os.environ.get("R2D2_ONNX_DIR", "/opt/r2d2-onnx"))
# One session per R2D2 iteration, built in the pool parent and shared by fork;
# with per-node checkpoints, one such set per NUMA node.
_ORT_SESSIONS: dict[int, object] = {}
_NODE_ORT_SESSIONS: dict[int, dict[int, object]] = {}
# For requests on more than one thread (round 258): per iteration, the
# optimised graph without its weights and the weights as 64-byte-aligned
# arrays, made in the pool parent (per NUMA node, like the sessions) so a
# worker's multi-thread sessions share them. A pool thread does not survive
# fork, so those sessions are built in the worker and kept.
_ORT_SHARED: dict[int, tuple] = {}
_NODE_ORT_SHARED: dict[int, dict[int, tuple]] = {}
_ORT_THREADED: dict[tuple[int, int], object] = {}


class OrtNet:
    """Stands in for the U-Net in `model.inference.forward`, which only calls it."""

    def __init__(self, session):
        self.session = session

    def __call__(self, x):
        import torch

        return torch.from_numpy(self.session.run(None, {"x": x.numpy()})[0])


def export_onnx(checkpoint_path: Path) -> Path:
    """Export each checkpoint once to `ONNX_DIR/models/<key>/N<i>.onnx`, height and width dynamic."""
    import hashlib

    import torch

    sys.path.insert(0, str(ONNX_DIR / "py"))
    import onnxruntime as ort

    util_model = sys.modules["utils.util_model"]
    num_iter = int(os.environ.get("R2D2_NUM_ITER", "25"))
    dnn_dict = util_model.get_DNNs(num_iter, str(checkpoint_path))
    stats = sorted((p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in checkpoint_path.iterdir())
    key = hashlib.sha1(repr((str(checkpoint_path), torch.__version__, ort.__version__, stats)).encode()).hexdigest()[:16]
    models = ONNX_DIR / "models" / key
    models.mkdir(parents=True, exist_ok=True)
    net = None
    for cur_iter in range(1, num_iter + 1):
        path = models / f"N{cur_iter}.onnx"
        if not path.exists():
            if net is None:
                net = util_model.create_net_imaging().eval()
            util_model.load_state_dict_from_state_dict(net, dnn_dict[f"N{cur_iter}"])
            partial = path.with_suffix(f".{os.getpid()}.tmp")
            # One thread: export runs a forward here in the pool parent, and an
            # OpenMP pool started before the fork deadlocks every worker's
            # first torch op (a GPU run's 3 threads a rank did, round 171).
            threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with torch.no_grad():
                    torch.onnx.export(net, (torch.zeros(1, 2, 32, 32),), str(partial), dynamo=False, opset_version=17,
                                      input_names=["x"], output_names=["y"],
                                      dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}})
            finally:
                torch.set_num_threads(threads)
            partial.replace(path)
    return models


def onnx_cpu_key() -> str:
    import hashlib

    cpuinfo = Path("/proc/cpuinfo").read_text().split("\n\n")[0]
    cpu = [line for line in cpuinfo.splitlines() if line.startswith(("model name", "flags"))]
    return hashlib.sha1(repr(cpu).encode()).hexdigest()[:12]


def prefetch_weights() -> int:
    """Fork a child that reads the pool's weights into the page cache, 16 files at a time; return its pid.

    A cold pool reads 5.8GB of checkpoints or (onnx, round 367) 2.9GB of
    optimised graphs one file at a time off Lustre: 8.9s and 10.3s of a 22-34s
    warm-up, against 5.8s warm (round 228). Forked before any import, so the
    parent's peak RSS, which every worker reports as its floor, does not see the reads.
    """
    pid = os.fork()
    if pid:
        return pid
    try:
        import concurrent.futures

        files = sorted(Path(os.environ.get("R2D2_CKPT_PATH", "/checkpoints/R2D2_A1")).glob("*.ckpt"))
        if r2d2_device() == "cpu" and r2d2_runtime() == "onnx":
            files = sorted(ONNX_DIR.glob(f"models/*/opt-{onnx_cpu_key()}/N*.onnx"))

        def read(path: Path) -> None:
            chunk = bytearray(1 << 22)
            with open(path, "rb", buffering=0) as stream:
                while stream.readinto(chunk):
                    pass

        with concurrent.futures.ThreadPoolExecutor(16) as reader:
            list(reader.map(read, files))
    except Exception:
        pass  # only ever a head start: the loads read whatever this missed
    os._exit(0)


def patch_onnx_network(checkpoint_path: Path) -> None:
    """Run each CPU U-Net forward through ONNX Runtime instead of torch.

    MLAS does this float32 network in 26ms against torch's 39.6ms with 36
    forked workers sharing one copy (docs/csd3-speed.md, round 140); outputs
    differ by up to 1.5e-5 of the peak over 25 chained checkpoints, so it is
    not bit-identical to torch, but converged log(Z) agrees (round 144) and it
    is the CPU default; R2D2_RUNTIME=torch keeps torch. Each checkpoint is
    exported once to `ONNX_DIR/models/<key>/N<i>.onnx`, with height and width dynamic,
    and its session built here, before the fork: one intra-op thread starts no
    pool threads, so it survives the fork and every worker shares its pages.
    A request running more than one torch thread (the tail of an async run,
    or R2D2_OMP_THREADS > 1) gets a session of that many threads built in the
    worker over the parent's weights (threaded_session, round 258); R3D3
    (layers > 1) keeps torch.
    """
    import concurrent.futures

    import torch

    models = export_onnx(checkpoint_path)
    import onnxruntime as ort

    num_iter = int(os.environ.get("R2D2_NUM_ITER", "25"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    # No CPU arena: bit-identical, same speed, ~8GB less PSS at 38 ranks (round 143).
    options.enable_cpu_mem_arena = False

    # Graph optimisation is ~3 of the 25 sessions' ~4.5s build, so the first
    # build on a CPU saves each optimised graph and later pools load that with
    # optimisation off: bit-identical outputs and forward time (round 202). It
    # may hold CPU-specific layouts (NCHWc), so it is keyed by the CPU.
    optimised = models / f"opt-{onnx_cpu_key()}"
    optimised.mkdir(exist_ok=True)

    def session(cur_iter: int):
        path = optimised / f"N{cur_iter}.onnx"
        if path.exists():
            loaded = ort.SessionOptions()
            for name in ("intra_op_num_threads", "inter_op_num_threads", "enable_cpu_mem_arena"):
                setattr(loaded, name, getattr(options, name))
            loaded.add_session_config_entry("session.intra_op.allow_spinning", "0")
            loaded.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            return ort.InferenceSession(str(path), loaded, providers=["CPUExecutionProvider"])
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        options.optimized_model_filepath = str(partial)
        try:
            built = ort.InferenceSession(str(models / f"N{cur_iter}.onnx"), options,
                                         providers=["CPUExecutionProvider"])
        finally:
            options.optimized_model_filepath = ""
        try:
            partial.replace(path)
        except OSError:  # not saved (read-only cache, say): the next pool optimises again
            pass
        return built

    iters = range(1, num_iter + 1)
    # Only a run whose requests can take more than one thread (async, or
    # R2D2_OMP_THREADS > 1) pays the arrays' memory, ~2.9GB per set.
    threaded = max(int(os.environ.get(name) or 1) for name in ("R2D2_OMP_THREADS", "R2D2_MAX_THREADS")) > 1

    def build() -> tuple[dict[int, object], dict[int, tuple]]:
        sessions = {}
        if not threaded or not all((optimised / f"N{cur_iter}.onnx").exists() for cur_iter in iters):
            sessions = {cur_iter: session(cur_iter) for cur_iter in iters}  # saves the optimised graphs
        if not threaded:
            return sessions, {}
        # A fresh pool per set: its threads take the caller's NUMA policy as
        # they start. The weights' copy and alignment release the GIL, so 8
        # threads take 1.4s against 2.5s (round 272).
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            shared = dict(zip(iters, pool.map(lambda cur_iter: ort_shared_parts(optimised / f"N{cur_iter}.onnx"), iters)))
        # The 1-thread sessions over those arrays too: bit-identical, as fast,
        # 0.3s against 1.9s for 25 file loads and one copy of the weights, not
        # two: first request 8.5 -> 5.7s in a search (rounds 272, 274).
        return sessions or {cur_iter: parts_session(shared[cur_iter], 1) for cur_iter in iters}, shared

    # Where the torch checkpoints went, the sessions' weights go too: a set
    # per NUMA node when load_checkpoints made one (enter_node swaps it in),
    # else one set interleaved across the job's nodes (round 149).
    if _NODE_CHECKPOINTS:
        for node in _NODE_CHECKPOINTS:
            bound = set_mempolicy(2, [node])  # MPOL_BIND
            try:
                _NODE_ORT_SESSIONS[node], _NODE_ORT_SHARED[node] = build()
            finally:
                if bound:
                    set_mempolicy(0, [])
        # Like _CHECKPOINT_CACHE, the parent keeps the last node's set.
        _ORT_SESSIONS.update(_NODE_ORT_SESSIONS[node])
        _ORT_SHARED.update(_NODE_ORT_SHARED[node])
    else:
        with interleaved_pages():
            sessions, shared = build()
            _ORT_SESSIONS.update(sessions)
            _ORT_SHARED.update(shared)

    optimiser = sys.modules["optimiser.R2D2"]
    upstream_load_net = optimiser.load_net

    def load_net(net, cur_iter, layers, dnn_dict):
        if isinstance(net, OrtNet):
            net = net.torch_net
        threads = torch.get_num_threads()
        session = _ORT_SESSIONS.get(int(cur_iter)) if threads == 1 else threaded_session(int(cur_iter), threads)
        if layers != 1 or session is None:
            return upstream_load_net(net, cur_iter, layers, dnn_dict)
        wrapper = OrtNet(session)
        wrapper.torch_net = net
        return wrapper

    optimiser.load_net = load_net


def ort_shared_parts(path: Path) -> tuple:
    """`path`'s graph with its weights declared external, and the weights as aligned OrtValues."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    def aligned(array):
        # numpy's 16-byte alignment costs MLAS 15% at 1 thread; 64 matches a file-loaded session.
        buffer = np.empty(array.nbytes + 64, np.uint8)
        offset = -buffer.ctypes.data % 64
        out = buffer[offset:offset + array.nbytes].view(array.dtype).reshape(array.shape)
        out[...] = array
        return out

    model = onnx.load(str(path))
    names, values = [], []
    for tensor in model.graph.initializer:
        names.append(tensor.name)
        values.append(ort.OrtValue.ortvalue_from_numpy(aligned(numpy_helper.to_array(tensor))))
        for field in ("raw_data", "float_data", "int32_data", "int64_data", "double_data"):
            tensor.ClearField(field)
        tensor.data_location = onnx.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key, entry.value = "location", "shared"
    return model.SerializeToString(), names, values


def threaded_session(cur_iter: int, threads: int):
    """A `threads`-thread session over the parent's weights, or None (no shared parts).

    ORT's output is bit-identical at 1-16 intra-op threads, and at 2 threads
    its forward takes 9.2ms against torch's 16.5ms (round 258). add_initializer
    uses the arrays in place: 25 sessions build in 0.34s for ~0.25GB. An
    async tail changes the count request by request, so it is floored to a
    power of two and every count's sessions are kept: at most four sets.
    """
    parts = _ORT_SHARED.get(cur_iter)
    if parts is None:
        return None
    threads = 1 << (threads.bit_length() - 1)
    session = _ORT_THREADED.get((cur_iter, threads))
    if session is None:
        session = _ORT_THREADED[(cur_iter, threads)] = parts_session(parts, threads)
    return session


def parts_session(parts: tuple, threads: int):
    """A `threads`-thread session over `ort_shared_parts`' arrays, used in place."""
    import onnxruntime as ort

    graph, names, values = parts
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for name, value in zip(names, values):
        options.add_initializer(name, value)
    return ort.InferenceSession(graph, options, providers=["CPUExecutionProvider"])


# Where the run script binds tensorrt-cu13 and the GPU's plans (R2D2_RUNTIME=trt).
TRT_DIR = Path(os.environ.get("R2D2_TRT_DIR", "/opt/r2d2-trt"))
# Per iteration, its serialised plan's path (set in the pool parent) and, per
# worker, its deserialised engine and context (made on first use: CUDA does
# not survive fork).
_TRT_PLANS: dict[int, Path] = {}
_TRT_CONTEXTS: dict[int, tuple] = {}


def build_trt_plans(models: Path, dim: int, num_iter: int, batch: int = 1) -> Path:
    """Build each missing fp32 plan for this GPU and image size; return their directory.

    Runs in a child of the pool parent (`--build-trt-plans`), which must not
    make a CUDA context. TF32 is cleared, as for eager. One fixed input shape,
    1x2xDIMxDIM: a forward of any other shape runs eager instead. With `batch`
    above 1 (R2D2_TRT_BATCH), for the gathering server, each plan has a fixed
    profile per batch size in trt_batch_sizes(batch), in a directory of its
    own: one dynamic 1..8 profile ran 27-28% slower at every size (round 224).
    """
    import torch

    sys.path.insert(0, str(TRT_DIR / "py"))
    import tensorrt as trt

    gpu = "".join(c if c.isalnum() else "_" for c in torch.cuda.get_device_name())
    sizes = trt_batch_sizes(batch)
    suffix = "-b" + "-".join(map(str, sizes)) if batch > 1 else ""
    plans = TRT_DIR / "plans" / f"{models.name}-{trt.__version__}-{gpu}-{dim}{suffix}"
    plans.mkdir(parents=True, exist_ok=True)
    log = trt.Logger(trt.Logger.ERROR)
    shape = (1, 2, dim, dim)
    for cur_iter in range(1, num_iter + 1):
        path = plans / f"N{cur_iter}.plan"
        if path.exists():
            continue
        builder = trt.Builder(log)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, log)
        if not parser.parse((models / f"N{cur_iter}.onnx").read_bytes()):
            raise RuntimeError(f"TensorRT could not parse N{cur_iter}.onnx: {parser.get_error(0)}")
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        if batch > 1:
            # The ONNX is exported with batch 1; this rebuild is no further
            # from the production plan than a batch-1 rebuild (round 223).
            network.get_input(0).shape = (-1, *shape[1:])
        for size in sizes:
            profile = builder.create_optimization_profile()
            fixed = (size, *shape[1:])
            profile.set_shape("x", fixed, fixed, fixed)
            config.add_optimization_profile(profile)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(f"TensorRT could not build N{cur_iter}")
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        partial.write_bytes(bytes(plan))
        partial.replace(path)
    return plans


class TrtNet:
    """Stands in for the U-Net on the GPU, as OrtNet does on the CPU; other shapes run eager."""

    def __init__(self, cur_iter, eager):
        self.cur_iter = cur_iter
        self.eager = eager

    def __call__(self, x):
        import torch

        context, shape, out_shape = trt_context(self.cur_iter)
        if tuple(x.shape) != shape or x.dtype != torch.float32:
            return self.eager()(x)
        x = x.contiguous()
        y = torch.empty(out_shape, dtype=torch.float32, device=x.device)
        context.set_tensor_address("x", x.data_ptr())
        context.set_tensor_address("y", y.data_ptr())
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(f"TensorRT N{self.cur_iter} failed")
        return y


def trt_context(cur_iter: int) -> tuple:
    entry = _TRT_CONTEXTS.get(cur_iter)
    if entry is None:
        import tensorrt as trt

        runtime = _TRT_CONTEXTS.get(0)
        if runtime is None:
            runtime = _TRT_CONTEXTS[0] = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(_TRT_PLANS[cur_iter].read_bytes())
        context = engine.create_execution_context()
        shape = tuple(engine.get_tensor_profile_shape("x", 0)[1])
        context.set_input_shape("x", shape)
        entry = _TRT_CONTEXTS[cur_iter] = (context, shape, tuple(context.get_tensor_shape("y")), engine)
    return entry[:3]


def read_chunk(path: Path, offset: int) -> None:
    """Read 64MB of `path` from `offset` into the page cache; a file gone or unreadable is skipped."""
    try:
        with open(path, "rb", buffering=0) as handle:
            handle.seek(offset)
            handle.read(64 << 20)
    except OSError:
        pass


def patch_trt_network(checkpoint_path: Path) -> None:
    """Run each GPU U-Net forward through a TensorRT fp32 engine instead of torch eager.

    On an A100 under MPS, 12 workers' engines put 2,700 forwards/s through the
    card against eager's best of 1,470 at 16 (docs/csd3-speed.md, rounds
    168-170), at ~1e-5 of the peak from the CPU path. An engine holds ~206MB
    on the card against eager's 126MB, so the run script sizes GPU ranks from
    NS_R2D2_CUDA_TRT_MB_PER_RANK. Plans are exported (export_onnx) and built
    (build_trt_plans, in a child) once per checkpoint set, TensorRT version,
    GPU model and image size, and cached under TRT_DIR/plans. A plan that
    cannot be built leaves the run on eager; R2D2_TRT_STATUS says which ran.
    """
    import concurrent.futures
    import subprocess

    # The child's `import tensorrt` page-faults ~0.8GB of libraries in off
    # Lustre, 26s on a cold node; big parallel reads first take ~1s (round 217).
    # Joined before anything forks.
    prefetch = concurrent.futures.ThreadPoolExecutor(16)
    libraries = [path for path in (TRT_DIR / "py").rglob("*.so*") if "builder_resource" not in path.name]
    prefetched = [prefetch.submit(read_chunk, path, offset)
                  for path in libraries for offset in range(0, path.stat().st_size, 64 << 20)]
    models = export_onnx(checkpoint_path)
    num_iter = int(os.environ.get("R2D2_NUM_ITER", "25"))
    dim = int(os.environ.get("NS_IMAGE_DIM", "32"))
    built = subprocess.run([sys.executable, __file__, "--build-trt-plans", str(models), str(dim), str(num_iter),
                            str(trt_batch())],
                           capture_output=True, text=True)
    concurrent.futures.wait(prefetched)
    prefetch.shutdown()
    # Warm-up output goes to /dev/null, so which engine ran is recorded here.
    status = Path(os.environ.get("R2D2_TRT_STATUS", os.devnull))
    if built.returncode != 0:
        status.write_text(f"eager: TensorRT plans not built\n{built.stderr[-4000:]}\n")
        return
    plans = Path(built.stdout.strip().splitlines()[-1])
    status.write_text(f"trt: {plans}\n")
    _TRT_PLANS.update({cur_iter: plans / f"N{cur_iter}.plan" for cur_iter in range(1, num_iter + 1)})
    sys.path.insert(0, str(TRT_DIR / "py"))

    optimiser = sys.modules["optimiser.R2D2"]
    upstream_load_net = optimiser.load_net

    wrap = TrtSharedNet if trt_shared() else TrtNet

    def load_net(net, cur_iter, layers, dnn_dict):
        if isinstance(net, (TrtNet, TrtSharedNet)):
            net = net.torch_net
        if layers != 1 or int(cur_iter) not in _TRT_PLANS:
            return upstream_load_net(net, cur_iter, layers, dnn_dict)
        wrapper = wrap(int(cur_iter), lambda: upstream_load_net(net, cur_iter, layers, dnn_dict))
        wrapper.torch_net = net
        return wrapper

    optimiser.load_net = load_net


def trt_shared() -> bool:
    """R2D2_TRT_SHARED=1 (with R2D2_RUNTIME=trt): one process holds the engines for every worker.

    A worker's 25 engines take 5.1GB, which caps an A100 at 12 TensorRT
    workers; one process holding them once, with an execution context per
    worker, puts 90% of 12 processes' forwards through the card on 6GB, and a
    pipe round trip per forward costs nothing measurable (csd3-speed.md, round
    207). Workers then stay off the GPU entirely: everything but the forward
    runs on the CPU, as it does for R2D2_DEVICE=cpu (round 208).
    """
    return (r2d2_device() == "cuda" and r2d2_runtime() == "trt"
            and os.environ.get("R2D2_TRT_SHARED", "0") == "1")


def trt_batch() -> int:
    """R2D2_TRT_BATCH=N (shared server only): run up to N workers' forwards of one net as one launch.

    One 32x32 forward leaves most of an A100 idle: batches of 8 on 4 streams
    put 3.05x the batch-1 ceiling through it (csd3-speed.md, round 223). The
    gathering server (serve_trt_gathered) serves whichever net has the most
    forwards waiting, so batches form without a waiting window.
    """
    return max(1, int(os.environ.get("R2D2_TRT_BATCH") or 1)) if trt_shared() else 1


def trt_batch_sizes(batch: int) -> list[int]:
    """The batch sizes a gathering plan has a profile for: powers of two below `batch`, then `batch`."""
    return sorted({1 << k for k in range(batch.bit_length()) if 1 << k < batch} | {batch})


# A forward request is (iteration, worker pid, sequence) then the input's
# bytes; the reply echoes the header then the output's. A SOCK_SEQPACKET pair
# keeps each one whole, and the echo lets a replacement worker drop a reply
# meant for the worker it replaced.
_TRT_HEADER = __import__("struct").Struct("<iiq")
# In the pool parent, each worker base's end of its pair; in a worker, its own.
_TRT_WORKER_SOCKETS: dict[str, object] = {}
_TRT_SOCKET: list = [None, 0]  # [socket, last sequence number]


def trt_shared_forward(sock, cur_iter: int, x, seq: int):
    """Send `x` (float32, CPU) to the server for net `cur_iter`; None if it did not answer."""
    import torch

    sock.send(_TRT_HEADER.pack(cur_iter, os.getpid(), seq) + x.numpy().tobytes())
    want = _TRT_HEADER.pack(cur_iter, os.getpid(), seq)
    size = _TRT_HEADER.size + x.numel() // 2 * 4
    while True:
        reply = sock.recv(size)
        if not reply:
            raise ConnectionError("TensorRT forward server closed")
        if reply[:_TRT_HEADER.size] == want:
            break
    if len(reply) != size:  # the server could not run this net
        return None
    return torch.frombuffer(bytearray(reply[_TRT_HEADER.size:]), dtype=torch.float32).view(1, 1, *x.shape[2:])


class TrtSharedNet:
    """Stands in for the U-Net in a worker, sending its forwards to the shared server; others run on the CPU."""

    def __init__(self, cur_iter, eager):
        self.cur_iter = cur_iter
        self.eager = eager

    def __call__(self, x):
        import torch

        sock = _TRT_SOCKET[0]
        dim = int(os.environ.get("NS_IMAGE_DIM", "32"))
        if sock is not None and tuple(x.shape) == (1, 2, dim, dim) and x.dtype == torch.float32:
            _TRT_SOCKET[1] += 1
            try:
                y = trt_shared_forward(sock, self.cur_iter, x.contiguous(), _TRT_SOCKET[1])
            except OSError:
                # Rather than fail every evaluation from here on, the worker
                # runs the CPU U-Net, and says so in the run's stderr.
                traceback.print_exc(file=sys.__stderr__)
                _TRT_SOCKET[0] = None
                y = None
            if y is not None:
                return y
        return self.eager()(x)


def serve_trt_shared(socks: list, dim: int) -> None:
    """The forward server: load each plan once, one lane (contexts, stream, buffers) a worker, a thread a lane."""
    import concurrent.futures
    import threading

    import numpy as np
    import torch

    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    # 5.3GB of plans: read eight at a time, deserialized in order as they land.
    # One at a time off Lustre took 11.6s cold and 4.0s warm, eight 3s and
    # 0.7s (docs/csd3-speed.md, rounds 216-217).
    with concurrent.futures.ThreadPoolExecutor(8) as reader:
        blobs = reader.map(Path.read_bytes, _TRT_PLANS.values())
        engines = {cur_iter: runtime.deserialize_cuda_engine(blob) for cur_iter, blob in zip(_TRT_PLANS, blobs)}
    scratch = max(engine.device_memory_size_v2 for engine in engines.values())
    device = torch.device("cuda")
    size = _TRT_HEADER.size + 2 * dim * dim * 4

    def lane(sock):
        stream = torch.cuda.Stream()
        x = torch.empty(1, 2, dim, dim, device=device)
        y = torch.empty(1, 1, dim, dim, device=device)
        hx = torch.empty(1, 2, dim, dim, pin_memory=True)
        hy = torch.empty(1, 1, dim, dim, pin_memory=True)
        memory = torch.empty(scratch, dtype=torch.uint8, device=device)
        contexts = {}
        for cur_iter, engine in engines.items():
            context = engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            context.set_input_shape("x", tuple(x.shape))
            context.set_tensor_address("x", x.data_ptr())
            context.set_tensor_address("y", y.data_ptr())
            context.set_device_memory(memory.data_ptr(), scratch)
            contexts[cur_iter] = context
        hx_bytes = hx.numpy().reshape(-1).view(np.uint8)
        # `memory` is returned only to stay alive: dropped here, torch's
        # allocator hands the same block to the next lane, every lane's
        # contexts then share one scratch, and concurrent forwards corrupt
        # each other (3,798 of 3,800 wrong at 19 lanes; round 208).
        return sock, stream, x, y, hx, hy, hx_bytes, contexts, memory

    def run(sock, stream, x, y, hx, hy, hx_bytes, contexts, _memory):
        while True:
            request = sock.recv(size)
            if not request:
                return
            header = request[:_TRT_HEADER.size]
            context = contexts.get(_TRT_HEADER.unpack(header)[0])
            if context is None or len(request) != size:
                sock.send(header)
                continue
            hx_bytes[:] = np.frombuffer(request, np.uint8, offset=_TRT_HEADER.size)
            with torch.cuda.stream(stream):
                x.copy_(hx, non_blocking=True)
                ok = context.execute_async_v3(stream.cuda_stream)
                hy.copy_(y, non_blocking=True)
                done = torch.cuda.Event(blocking=True)
                done.record(stream)
            done.synchronize()
            sock.send(header + hy.numpy().tobytes() if ok else header)

    # Each lane serves as soon as its 25 contexts exist: making all 19 took
    # 12.2s, which every worker used to wait out (round 221).
    threads = []
    for sock in socks:
        args = lane(sock)
        torch.cuda.synchronize()
        threads.append(threading.Thread(target=run, args=args, daemon=True))
        threads[-1].start()
    with open(os.environ.get("R2D2_TRT_STATUS", os.devnull), "a") as status:
        status.write(f"shared: one server, {len(threads)} lanes\n")
    for thread in threads:
        thread.join()


def gather_forwards(socks: list, size: int, batch: int, lanes: int, forward, launches: dict | None = None) -> None:
    """Serve workers' forward requests in batches: one reader thread, `lanes` threads running batches.

    The reader queues each request by net; a free lane takes up to `batch` of
    the net with most waiting and calls forward(lane, cur_iter, requests),
    which returns each request's output bytes, or None if it cannot run that
    net (each worker is then answered with the bare header and runs it on the
    CPU). Returns once every worker's socket has closed. `launches`, if
    given, counts the batches run by size.
    """
    import selectors
    import threading

    pending: dict[int, list] = {}
    ready = threading.Condition()
    reading = [True]

    def read():
        try:
            receive()
        except Exception:
            # As for a lane: a dead reader would leave every worker waiting.
            traceback.print_exc()
            os._exit(1)

    def receive():
        selector = selectors.DefaultSelector()
        for sock in socks:
            selector.register(sock, selectors.EVENT_READ)
        while selector.get_map():
            for key, _ in selector.select():
                try:
                    request = key.fileobj.recv(size)
                except OSError:
                    request = b""
                if not request:
                    selector.unregister(key.fileobj)
                    continue
                header = request[:_TRT_HEADER.size]
                if len(request) != size:
                    try:
                        key.fileobj.send(header)
                    except OSError:
                        pass
                    continue
                with ready:
                    pending.setdefault(_TRT_HEADER.unpack(header)[0], []).append((key.fileobj, request))
                    ready.notify()
        with ready:
            reading[0] = False
            ready.notify_all()

    def run(lane):
        while True:
            with ready:
                while not pending and reading[0]:
                    ready.wait()
                if not pending:
                    return
                cur_iter = max(pending, key=lambda net: len(pending[net]))
                group, pending[cur_iter] = pending[cur_iter][:batch], pending[cur_iter][batch:]
                if not pending[cur_iter]:
                    del pending[cur_iter]
                if launches is not None:
                    launches[len(group)] = launches.get(len(group), 0) + 1
            try:
                outputs = forward(lane, cur_iter, [request for _, request in group])
            except Exception:
                # A lane gone would leave its workers waiting for ever; the
                # whole server going shows each a closed socket instead.
                traceback.print_exc()
                os._exit(1)
            for i, (sock, request) in enumerate(group):
                try:
                    sock.send(request[:_TRT_HEADER.size] + (b"" if outputs is None else outputs[i]))
                except OSError:
                    pass

    threads = [threading.Thread(target=read, daemon=True)]
    threads += [threading.Thread(target=run, args=(lane,), daemon=True) for lane in range(lanes)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def serve_trt_gathered(socks: list, dim: int, batch: int) -> None:
    """The forward server for R2D2_TRT_BATCH: each plan loaded once, a few lanes running batched forwards.

    A lane is a stream, buffers for `batch` images and a context per net and
    batch size; R2D2_TRT_LANES (default 2) of them keep the card busy while
    the others copy (round 223's simulation put 2 ahead of 1 and 4). A group
    runs at the smallest size that holds it; the rows past it are whatever
    the buffer held, since a row's output does not depend on the others'
    (round 224).
    """
    import concurrent.futures
    import threading

    import numpy as np
    import torch

    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with concurrent.futures.ThreadPoolExecutor(8) as reader:
        blobs = reader.map(Path.read_bytes, _TRT_PLANS.values())
        engines = {cur_iter: runtime.deserialize_cuda_engine(blob) for cur_iter, blob in zip(_TRT_PLANS, blobs)}
    scratch = max(engine.device_memory_size_v2 for engine in engines.values())
    device = torch.device("cuda")
    size = _TRT_HEADER.size + 2 * dim * dim * 4
    row = 2 * dim * dim * 4
    sizes = trt_batch_sizes(batch)
    lanes = []
    for _ in range(max(1, int(os.environ.get("R2D2_TRT_LANES") or 2))):
        x = torch.empty(batch, 2, dim, dim, device=device)
        y = torch.empty(batch, 1, dim, dim, device=device)
        hx = torch.empty(batch, 2, dim, dim, pin_memory=True)
        hy = torch.empty(batch, 1, dim, dim, pin_memory=True)
        # Each lane's own scratch, kept alive in the tuple (see serve_trt_shared).
        memory = torch.empty(scratch, dtype=torch.uint8, device=device)
        stream = torch.cuda.Stream()
        contexts = {}
        for cur_iter, engine in engines.items():
            for index, rows in enumerate(sizes):
                context = engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
                context.set_optimization_profile_async(index, stream.cuda_stream)
                context.set_input_shape("x", (rows, 2, dim, dim))
                context.set_tensor_address("x", x.data_ptr())
                context.set_tensor_address("y", y.data_ptr())
                context.set_device_memory(memory.data_ptr(), scratch)
                contexts[cur_iter, rows] = context
        lanes.append((stream, x, y, hx, hy, hx.numpy().reshape(-1).view(np.uint8),
                      hy.numpy().reshape(batch, -1), contexts, memory))
    torch.cuda.synchronize()

    def forward(lane, cur_iter, requests):
        stream, x, y, hx, hy, hx_bytes, hy_rows, contexts, _memory = lanes[lane]
        n = len(requests)
        padded = next(size for size in sizes if size >= n)
        context = contexts.get((cur_iter, padded))
        if context is None:
            return None
        for i, request in enumerate(requests):
            hx_bytes[i * row:(i + 1) * row] = np.frombuffer(request, np.uint8, offset=_TRT_HEADER.size)
        with torch.cuda.stream(stream):
            x[:n].copy_(hx[:n], non_blocking=True)
            ok = context.execute_async_v3(stream.cuda_stream)
            hy[:n].copy_(y[:n], non_blocking=True)
            done = torch.cuda.Event(blocking=True)
            done.record(stream)
        done.synchronize()
        return [hy_rows[i].tobytes() for i in range(n)] if ok else None

    with open(os.environ.get("R2D2_TRT_STATUS", os.devnull), "a") as status:
        status.write(f"shared: one server, {len(socks)} workers gathered into batches of up to {batch} "
                     f"on {len(lanes)} lanes\n")
    # Batches form only among workers at the same R2D2 iteration, so the size
    # they reach is what the card's batching buys; r2d2-trt-batches.txt says
    # (bench rows carry its mean as trt_mean_batch).
    launches: dict[int, int] = {}
    if os.environ.get("R2D2_TRT_STATUS"):
        report = Path(os.environ["R2D2_TRT_STATUS"]).with_name("r2d2-trt-batches.txt")
        threading.Thread(target=report_launches, args=(report, launches), daemon=True).start()
    gather_forwards(socks, size, batch, len(lanes), forward, launches)
    if os.environ.get("R2D2_TRT_STATUS"):
        try:
            write_launches(report, launches)
        except OSError:
            pass


def write_launches(path: Path, launches: dict) -> None:
    """One line: launches, forwards, their mean batch and the count at each size."""
    counts = dict(launches)
    runs = sum(counts.values())
    forwards = sum(size * n for size, n in counts.items())
    sizes = " ".join(f"{size}:{counts[size]}" for size in sorted(counts))
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(f"launches {runs} forwards {forwards} mean {forwards / max(runs, 1):.3f} sizes {sizes}\n")
    os.replace(temporary, path)


def report_launches(path: Path, launches: dict) -> None:
    while True:
        time.sleep(30)
        try:
            write_launches(path, launches)
        except (OSError, RuntimeError):  # RuntimeError: a new size added mid-copy
            pass


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
    if node in _NODE_ORT_SESSIONS:
        _ORT_SESSIONS.clear()
        _ORT_SESSIONS.update(_NODE_ORT_SESSIONS[node])
        _ORT_SHARED.clear()
        _ORT_SHARED.update(_NODE_ORT_SHARED.get(node, {}))


_imager_code: dict[Path, Any] = {}


def run_imager(run_name: str) -> None:
    """`runpy.run_path` without re-reading and compiling imager.py each request (round 357)."""
    code = _imager_code.get(IMAGER)
    if code is None:
        code = _imager_code[IMAGER] = compile(IMAGER.read_text(), str(IMAGER), "exec")
    exec(code, {"__name__": run_name, "__file__": str(IMAGER)})


def warm_imports() -> None:
    os.chdir(R2D2_HOME)
    sys.path.insert(0, str(IMAGER.parent))
    install_lazy_utils()
    # Reuse imager.py's import block; suppress import-time output from the
    # JSON reply stream.
    with redirect_fds(Path(os.devnull), Path(os.devnull)):
        try:
            run_imager("__warmup__")
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
            patch_imaging_weights()
            patch_request_io()
            patch_checkpoint_loading()
            patch_early_stop()
            patch_net_reuse()
            if r2d2_device() == "cuda" and not trt_shared():
                patch_cuda_network()
            checkpoint_path = Path(os.environ.get("R2D2_CKPT_PATH", "/checkpoints/R2D2_A1"))
            if checkpoint_path.is_dir():
                # Load before fork: tensor pages stay shared until a child
                # writes them, instead of paying torch.load once per rank.
                load_checkpoints(checkpoint_path)
                if r2d2_device() == "cpu" and r2d2_runtime() == "onnx":
                    patch_onnx_network(checkpoint_path)
                if r2d2_device() == "cuda" and r2d2_runtime() == "trt":
                    patch_trt_network(checkpoint_path)
            # `create_meas_op` imports this backend lazily; preload it.
            from ri_measurement_operator.pysrc.measOperator import (  # noqa: F401
                meas_op_nufft_pytorch_finufft,
            )
        except Exception:
            traceback.print_exc()


def patch_imaging_weights() -> None:
    """Grid Briggs weights with `np.bincount` and apply the tail per grid cell.

    Byte-identical to upstream (same summation order at one thread, and torch's
    complex tail gives each value the same bits per cell as per visibility),
    without its masked-index copies and its per-visibility complex sqrt.
    R2D2_NUMPY_WEIGHTS=0 keeps upstream.
    """
    if os.environ.get("R2D2_NUMPY_WEIGHTS", "1") == "0":
        return
    import importlib

    import numpy as np
    import torch

    # The package re-exports the function under the module's own name, so
    # the module is only reachable through sys.modules.
    importlib.import_module("ri_measurement_operator.pysrc.utils.gen_imaging_weights")
    module = sys.modules["ri_measurement_operator.pysrc.utils.gen_imaging_weights"]
    upstream = module.gen_imaging_weights

    def gen_imaging_weights(u, v, nW, im_size, weight_type="briggs", weight_gridsize=2.0, weight_robustness=0.0):
        if (weight_type != "briggs" or u.device.type != "cpu" or u.dtype != torch.float64
                or torch.get_num_threads() != 1):
            return upstream(u, v, nW, im_size, weight_type, weight_gridsize, weight_robustness)
        rows, cols = (int(np.floor(i * weight_gridsize)) for i in im_size)
        nW2 = nW.view(-1) ** 2
        if nW2.size(-1) == 1:
            nW2 = nW2 * torch.ones(u.numel())
        u, v = u.reshape(-1).numpy(), v.reshape(-1).numpy()
        flip = v < 0
        u, v = np.where(flip, -u, u), np.where(flip, -v, v)
        p = np.floor((v + np.pi) * rows / 2 / np.pi).astype(np.int64) - 1
        q = np.floor((u + np.pi) * cols / 2 / np.pi).astype(np.int64) - 1
        index = p * cols + q
        grid = torch.from_numpy(np.bincount(index, weights=nW2.to(torch.float64).numpy(), minlength=rows * cols))
        robust_scale = (torch.sum(nW2) / torch.sum(grid**2)) * (5 * 10 ** (-weight_robustness)) ** 2
        return (1 / torch.sqrt(1 + robust_scale * grid))[torch.from_numpy(index)].view(1, 1, -1)

    gen_imaging_weights.upstream = upstream
    module.gen_imaging_weights = gen_imaging_weights


_FITS_HEADERS: dict[tuple, bytes] = {}


def patch_request_io() -> None:
    """Write a request's four FITS images and parse its config without astropy's
    and PyYAML's pure-Python paths (~1.3ms an image, ~1.2ms a config, solo).

    The FITS header depends only on shape and dtype, so astropy writes it once
    per shape and every later image is that header, the big-endian data and
    the zero padding: byte-identical to `fits.writeto`. Anything but a plain
    float array to a fresh path with overwrite=True goes to upstream. The
    config goes through libyaml's CSafeLoader, which reads the run's config
    to the same dict. R2D2_FAST_IO=0 keeps upstream.
    """
    if os.environ.get("R2D2_FAST_IO", "1") == "0":
        return
    import io

    import numpy as np
    import yaml
    from astropy.io import fits

    upstream_writeto = fits.writeto

    def writeto(filename, data, header=None, output_verify="exception", overwrite=False, checksum=False):
        if (header is not None or checksum or not overwrite or output_verify != "exception"
                or not isinstance(filename, (str, os.PathLike)) or not isinstance(data, np.ndarray)
                or data.dtype not in (np.float32, np.float64) or not 1 <= data.ndim <= 3):
            return upstream_writeto(filename, data, header, output_verify, overwrite, checksum)
        key = (data.shape, data.dtype.str)
        padding = -data.nbytes % 2880
        head = _FITS_HEADERS.get(key)
        if head is None:
            buffer = io.BytesIO()
            upstream_writeto(buffer, np.zeros(data.shape, data.dtype))
            head = _FITS_HEADERS[key] = buffer.getvalue()[: -(data.nbytes + padding)]
        with open(filename, "wb") as handle:
            handle.write(head)
            handle.write(data.astype(data.dtype.newbyteorder(">"), copy=False).tobytes())
            handle.write(bytes(padding))

    writeto.upstream = upstream_writeto
    fits.writeto = writeto
    if getattr(yaml, "__with_libyaml__", False):
        yaml.safe_load = lambda stream: yaml.load(stream, Loader=yaml.CSafeLoader)


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
                    spread_sort=nufft_sort(),
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


def nufft_sort() -> int:
    """FINUFFT's spread_sort for a request's two type-1 NUFFTs. 0 skips the bin sort,
    which on R2D2's tiny grids costs more than it saves: -18% a call under load, and
    images within one float32 ulp of the peak (docs/csd3-speed.md, round 153).
    R2D2_NUFFT_SORT=2 is upstream's."""
    return int(os.environ.get("R2D2_NUFFT_SORT", "0"))


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
                              nthreads=torch.get_num_threads(), spread_sort=nufft_sort())
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
    prefetcher = prefetch_weights()  # first, so it holds none of the FIFO ends below
    keeper = {base: (os.open(f"{base}.in", os.O_RDONLY | os.O_NONBLOCK), os.open(f"{base}.out", os.O_RDWR))
              for base in bases}
    warm_imports()
    # Whatever it has not read by now the loads already have. Reaped here, not
    # by the os.wait() below, which forks a replacement for every pid it gets.
    os.kill(prefetcher, signal.SIGKILL)
    os.waitpid(prefetcher, 0)
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
            for other, sock in _TRT_WORKER_SOCKETS.items():
                if other == base:
                    _TRT_SOCKET[0] = sock
                else:
                    sock.close()
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

    server = start_trt_server(bases) if trt_shared() and _TRT_PLANS else None
    for base in bases:
        fork(base)
    # Serves until killed: a worker that exits - on EOF when its rank goes, or
    # dying mid-request - is forked again from this warm parent, and the rank
    # (or the next attempt's rank) reconnects to the same FIFOs.
    while True:
        pid, _status = os.wait()
        if pid == server:
            # Its workers notice (closed socket) and run the CPU U-Net: right
            # answers at a sixth of the rate, which nothing else shows (round
            # 219), so it is recorded where `./ri health` looks.
            record_trt_server_exit(_status)
            continue
        base = children.pop(pid)
        # Dropped as its worker goes, not at the end: while this process holds
        # them a rank whose worker died would write into a pipe nobody reads
        # and then wait forever for a reply. Closing here gives it the same
        # broken pipe and empty reply it gets when there is no pool at all,
        # and the replacement's own opens then wait for the rank to come back.
        for fd in keeper.pop(base, ()):
            os.close(fd)
        fork(base, replacement=True)


def record_trt_server_exit(wait_status: int) -> None:
    print(f"r2d2 TensorRT forward server exited ({wait_status})", file=sys.stderr)
    with open(os.environ.get("R2D2_TRT_STATUS", os.devnull), "a") as status:
        status.write(f"fallback: server exited ({wait_status}) at "
                     f"{time.strftime('%Y-%m-%dT%H:%M:%S')}, CPU U-Net from here\n")


def start_trt_server(bases: list[str]) -> int:
    """Fork the shared TensorRT forward server, with a socket pair per worker; return its pid.

    The pool parent keeps each worker's end, so a replacement worker inherits
    it; the server's ends are closed here, so the server going shows each
    worker a closed socket rather than a wait for ever.
    """
    import socket

    dim = int(os.environ.get("NS_IMAGE_DIM", "32"))
    server_ends = []
    for base in bases:
        worker_end, server_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        for end in (worker_end, server_end):
            end.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * (_TRT_HEADER.size + 8 * dim * dim))
        _TRT_WORKER_SOCKETS[base] = worker_end
        server_ends.append(server_end)
    pid = os.fork()
    if pid == 0:
        status = 0
        try:
            for sock in _TRT_WORKER_SOCKETS.values():
                sock.close()
            if trt_batch() > 1:
                serve_trt_gathered(server_ends, dim, trt_batch())
            else:
                serve_trt_shared(server_ends, dim)
        except Exception:
            traceback.print_exc()
            status = 1
        os._exit(status)
    for sock in server_ends:
        sock.close()
    return pid


# NS_EARLY_STOP (round 355): at iteration k a request stops once the iterate's
# RMS is under this share of its contour. Each is ~1% under the lowest ratio any
# above-contour evaluation of round 354's converged search reached, and below 1,
# so a stopped evaluation always scores below its contour. Iteration 2 has no
# check and iteration 3 is tightened (round 365): 78 of round 360's 91 wrong stops
# came from 2, and without it 58 of 60 came from 3, all at ratios over 0.82.
EARLY_STOP_RATIOS = {3: 0.82, 4: 0.86, 5: 0.84, 6: 0.82, 8: 0.84, 10: 0.87, 12: 0.90, 15: 0.93, 18: 0.95}
_early_stop: dict[str, Any] | None = None


def below_contour(model: Any, iteration: int, rule: dict[str, Any]) -> bool:
    """Whether iterate `model` (2-d) is far enough below `rule`'s contour to stop."""
    ratio = EARLY_STOP_RATIOS.get(iteration)
    if ratio is None:
        return False
    y, x = rule["y"], rule["x"]
    pixel = float(model[y, x])
    squares = float((model * model).sum()) - pixel * pixel + (pixel - rule["flux"]) ** 2
    rms = (max(squares, 0.0) / (model.shape[0] * model.shape[1])) ** 0.5
    if rule.get("shadow"):
        print(f"EARLY CHECK: {iteration} {rms:.6e}")
    if rms >= ratio * rule["contour"]:
        return False
    print(f"EARLY STOP: iteration {iteration} RMS {rms:.6e} contour {rule['contour']:.6e}")
    return not rule.get("shadow")


def patch_early_stop() -> None:
    R2D2 = sys.modules["optimiser.R2D2"].R2D2
    upstream = R2D2._stop_criteria

    def _stop_criteria(self):
        if upstream(self):
            return True
        return _early_stop is not None and below_contour(self._model.squeeze(), self._iter + 1, _early_stop)

    R2D2._stop_criteria = _stop_criteria


def serve(fifo_base: str | None = None) -> None:
    warm_imports()
    answer(fifo_base)


def answer(fifo_base: str | None) -> None:
    global _early_stop
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
        _early_stop = request.get("early_stop")
        returncode = 0
        with redirect_fds(Path(request["stdout"]), Path(request["stderr"])):
            saved_argv = sys.argv
            try:
                sys.argv = [str(IMAGER), *request["argv"]]
                run_imager("__main__")
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


def self_check_early_stop() -> None:
    import numpy as np

    rule = {"contour": 1.0, "flux": 8.0, "x": 1, "y": 2}
    model = np.zeros((4, 4), dtype=np.float32)
    model[2, 1] = 8.0
    model[0, 0] = 2.0 * 0.5 * 4 * 0.9  # RMS 0.9 x contour, so 0.82-0.90 decides
    assert below_contour(model, 7, rule) is False, "not a check iteration"
    assert below_contour(model, 6, rule) is False
    assert below_contour(model, 15, rule) is True
    assert below_contour(np.zeros((4, 4)), 2, {**rule, "flux": 0.0}) is False, "iteration 2 has no check"
    assert below_contour(np.zeros((4, 4)), 3, {**rule, "flux": 0.0}) is True
    assert below_contour(np.zeros((4, 4)), 3, {**rule, "flux": 0.0, "shadow": True}) is False, "shadow never stops"
    assert all(ratio < 1.0 for ratio in EARLY_STOP_RATIOS.values())
    print("r2d2 early stop self-check passed")


def self_check_run_imager() -> None:
    import tempfile

    global IMAGER
    saved_imager, saved_argv = IMAGER, sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        IMAGER = Path(tmp) / "imager.py"
        IMAGER.write_text("import sys\nassert 'seen' not in globals()\nseen = 1\nsys.exit(int(sys.argv[1]))\n")
        sys.argv = ["imager", "3"]
        try:
            for _ in range(2):
                try:
                    run_imager("__main__")
                except SystemExit as exc:
                    assert exc.code == 3
                else:
                    raise AssertionError("the imager's exit did not reach the caller")
                IMAGER.unlink(missing_ok=True)  # the second run must not read the file
        finally:
            _imager_code.pop(IMAGER, None)
            IMAGER, sys.argv = saved_imager, saved_argv
    print("run_imager self-check passed")


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


def self_check_imaging_weights() -> None:
    sys.path.insert(0, str(IMAGER.parent))
    try:
        import torch
        from ri_measurement_operator.pysrc.utils import gen_imaging_weights  # noqa: F401
    except ImportError:
        print("r2d2 imaging weights self-check skipped: no R2D2 measurement operator")
        return

    patch_imaging_weights()
    module = sys.modules["ri_measurement_operator.pysrc.utils.gen_imaging_weights"]
    patched = module.gen_imaging_weights
    upstream = getattr(patched, "upstream", None)
    assert upstream is not None, "R2D2_NUMPY_WEIGHTS must patch by default"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        count = 30011
        for robustness in (0.0, -1.0, 1.5):
            u = (torch.rand(1, 1, count, dtype=torch.float64) - 0.5) * 6.0
            v = (torch.rand(1, 1, count, dtype=torch.float64) - 0.5) * 6.0
            for nW in (torch.full((1, 1), 3.7 + 0j, dtype=torch.complex128),
                       (torch.rand(1, 1, count, dtype=torch.float64) + 0.5).to(torch.complex128)):
                expected = upstream(u.clone(), v.clone(), nW, (32, 32), "briggs", 2.0, robustness)
                got = patched(u.clone(), v.clone(), nW, (32, 32), "briggs", 2.0, robustness)
                assert got.dtype == expected.dtype and got.shape == expected.shape, (got.dtype, expected.dtype)
                assert torch.equal(got, expected), "imaging weights must be byte-identical"
    finally:
        torch.set_num_threads(threads)
        module.gen_imaging_weights = upstream
    print("r2d2 imaging weights self-check passed")


def self_check_request_io() -> None:
    import tempfile

    try:
        import numpy as np
        import yaml
        from astropy.io import fits
    except ImportError:
        print("r2d2 request io self-check skipped: no astropy")
        return

    upstream_writeto, upstream_safe_load = fits.writeto, yaml.safe_load
    patch_request_io()
    assert getattr(fits.writeto, "upstream", None) is upstream_writeto, "R2D2_FAST_IO must patch by default"
    rng = np.random.default_rng(0)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            arrays = [rng.standard_normal((32, 32)), rng.standard_normal((32, 32)).astype(np.float32),
                      np.zeros((32, 32)), rng.standard_normal((7, 5)), rng.standard_normal(3).astype(np.float32),
                      rng.standard_normal((32, 32)).T, np.array([[np.nan, np.inf], [-np.inf, -0.0]]),
                      rng.standard_normal((32, 32)).astype(">f8")]
            for index, array in enumerate(arrays * 2):
                fast, slow = Path(tmp) / f"fast{index}.fits", Path(tmp) / f"slow{index}.fits"
                slow.write_bytes(b"stale" * 9000)
                fast.write_bytes(b"stale" * 9000)
                upstream_writeto(slow, array, overwrite=True)
                fits.writeto(str(fast) if index % 2 else fast, array, overwrite=True)
                assert fast.read_bytes() == slow.read_bytes(), f"FITS image {index} must be byte-identical"
            # Upstream's own refusals stay upstream's.
            try:
                fits.writeto(fast, arrays[0])
            except OSError:
                pass
            else:
                raise AssertionError("an existing file without overwrite must still be refused")
        config = "\n".join(["data_file: /tmp/a.mat", "im_dim_x: 32", "data_weighting: True",
                             "weight_robustness: -0.5", "target_dynamic_range: 1e+4", "series: R2D2",
                             "imweight_name: nWimag", "num_iter: 15", "ncpus: 1", "gdth_file: null"])
        assert yaml.safe_load(config) == upstream_safe_load(config), "the config must parse to the same dict"
    finally:
        fits.writeto, yaml.safe_load = upstream_writeto, upstream_safe_load
    print("r2d2 request io self-check passed")


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

    global load_checkpoint_series
    series = load_checkpoint_series
    load_checkpoint_series = loader
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
        load_checkpoint_series = series
        _CHECKPOINT_CACHE.clear()
        _NORMALIZED_CHECKPOINT_CACHE.clear()
        del sys.modules["utils.util_model"]
        del sys.modules["utils"]
        del sys.modules["optimiser.R2D2"]
    print("r2d2 checkpoint cache self-check passed")


def self_check_checkpoint_series() -> None:
    import tempfile

    import torch

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(1, 7):
            torch.save({"state_dict": {"unet.w": torch.full((3,), float(i))}, "optimizer": [i]},
                       Path(tmp) / f"R2D2_UNet_N{i}.ckpt")
        loaded = load_checkpoint_series(6, tmp)
        assert list(loaded) == [f"N{i}" for i in range(1, 7)], list(loaded)
        for i in range(1, 7):
            assert torch.equal(loaded[f"N{i}"]["unet.w"], torch.full((3,), float(i)))
        # A CPU onnx pool writes each state_dict alone once, then reads that (round 257).
        global ONNX_DIR
        onnx_dir, runtime = ONNX_DIR, os.environ.get("R2D2_RUNTIME")
        with tempfile.TemporaryDirectory() as cache:
            ONNX_DIR, os.environ["R2D2_RUNTIME"] = Path(cache), "onnx"
            try:
                assert not load_checkpoint_series(6, tmp), "a CPU onnx series loads nothing up front"
                assert all(torch.equal(load_checkpoint_series(6, tmp)[name]["unet.w"], loaded[name]["unet.w"])
                           for _ in range(2) for name in loaded)
                written = sorted(p.name for p in checkpoint_weights_dir(tmp).iterdir())
                assert written == [f"R2D2_UNet_N{i}.ckpt" for i in range(1, 7)], written
            finally:
                ONNX_DIR = onnx_dir
                if runtime is None:
                    os.environ.pop("R2D2_RUNTIME")
                else:
                    os.environ["R2D2_RUNTIME"] = runtime
        for num_iter, message in ((7, "Checkpoint for N7 not found"), (6, "Checkpoint conflict for N2")):
            if num_iter == 6:
                torch.save({"state_dict": {}}, Path(tmp) / "copy_N2.ckpt")
            try:
                load_checkpoint_series(num_iter, tmp)
            except ValueError as exc:
                assert str(exc) == message, exc
            else:
                raise AssertionError(f"no {message!r}")
    print("r2d2 checkpoint series self-check passed")


def self_check_threaded_session() -> None:
    """A multi-thread session over shared weights matches a file-loaded one bit for bit."""
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import helper, numpy_helper
    except ImportError:  # r2d2.sif without the run script's onnxruntime
        print("threaded_session: skipped, no onnxruntime")
        return
    import tempfile

    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(rng.standard_normal((8, 2, 3, 3)).astype(np.float32), "w")
    graph = helper.make_graph([helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])], "g",
                              [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2, 32, 32])],
                              [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)], [weight])
    x = rng.standard_normal((1, 2, 32, 32)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "N1.onnx"
        onnx.save(helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)]), str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        # As patch_onnx_network loads its saved optimised graphs.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        expected = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        _ORT_SHARED.clear()
        _ORT_THREADED.clear()
        try:
            assert threaded_session(1, 2) is None
            _ORT_SHARED[1] = ort_shared_parts(path)
            # The pool's own 1-thread sessions are built this way in async runs.
            assert np.array_equal(parts_session(_ORT_SHARED[1], 1).run(None, {"x": x})[0], expected)
            for threads in (2, 3, 4):
                session = threaded_session(1, threads)
                assert threaded_session(1, threads) is session
                assert np.array_equal(session.run(None, {"x": x})[0], expected)
            assert list(_ORT_THREADED) == [(1, 2), (1, 4)]  # 3 runs on 2
        finally:
            _ORT_SHARED.clear()
            _ORT_THREADED.clear()
    print("threaded_session: ok")


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


def self_check_trt_fallback() -> None:
    """A forward of a shape (or dtype) the engine was not built for runs eager."""
    import torch

    _TRT_CONTEXTS[7] = (None, (1, 2, 32, 32), (1, 1, 32, 32), None)
    try:
        net = TrtNet(7, lambda: (lambda x: x * 2))
        x = torch.ones(1, 2, 64, 64)
        assert torch.equal(net(x), x * 2)
        assert torch.equal(net(x[:, :, :32, :32].double()), x[:, :, :32, :32].double() * 2)
    finally:
        _TRT_CONTEXTS.clear()
    print("r2d2 trt fallback self-check passed")


def self_check_trt_shared_protocol() -> None:
    """A worker's forwards round-trip, drop stale replies, and fall back to the CPU net."""
    import socket
    import tempfile
    import threading

    import torch

    worker_end, server_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    os.environ["NS_IMAGE_DIM"] = "4"
    size = _TRT_HEADER.size + 2 * 4 * 4 * 4

    def fake_server():
        while request := server_end.recv(size):
            header, x = request[:_TRT_HEADER.size], torch.frombuffer(bytearray(request[_TRT_HEADER.size:]),
                                                                     dtype=torch.float32)
            cur_iter, pid, seq = _TRT_HEADER.unpack(header)
            server_end.send(_TRT_HEADER.pack(cur_iter, pid + 1, seq) + bytes(64))  # someone else's
            if cur_iter == 9:  # a net the server has no engine for
                server_end.send(header)
                continue
            server_end.send(header + (x[:16] * 3).numpy().tobytes())

    thread = threading.Thread(target=fake_server, daemon=True)
    thread.start()
    _TRT_SOCKET[:] = [worker_end, 0]
    try:
        x = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)
        net = TrtSharedNet(3, lambda: (lambda x: x[:, :1] * 2))
        assert torch.equal(net(x), x[:, :1] * 3), "served forward"
        assert torch.equal(net(x), x[:, :1] * 3), "second forward skips the stale reply"
        assert torch.equal(TrtSharedNet(9, lambda: (lambda x: x[:, :1] * 2))(x), x[:, :1] * 2), "no engine"
        assert torch.equal(net(torch.ones(1, 2, 8, 8)), torch.ones(1, 1, 8, 8) * 2), "other shape"
        server_end.shutdown(socket.SHUT_RDWR)  # wakes the fake server's recv; close() would not
        thread.join()
        server_end.close()
        with redirect_fds(Path(os.devnull), Path(os.devnull)):
            assert torch.equal(net(x), x[:, :1] * 2), "server gone"
        assert _TRT_SOCKET[0] is None
    finally:
        _TRT_SOCKET[:] = [None, 0]
        worker_end.close()
        os.environ.pop("NS_IMAGE_DIM")
    # The server's exit lands in the status file `./ri health` reads.
    with tempfile.TemporaryDirectory() as tmp:
        engine = Path(tmp, "r2d2-engine.txt")
        engine.write_text("shared: one server, 19 lanes\n")
        os.environ["R2D2_TRT_STATUS"] = str(engine)
        try:
            with redirect_fds(Path(os.devnull), Path(os.devnull)):
                record_trt_server_exit(139)
        finally:
            os.environ.pop("R2D2_TRT_STATUS")
        last = engine.read_text().splitlines()[-1]
        assert last.startswith("fallback: server exited (139) at "), last
    print("r2d2 trt shared protocol self-check passed")


def self_check_trt_gathered() -> None:
    """Gathered forwards reach the right worker, in batches no larger than asked, and the server ends."""
    import array
    import socket
    import tempfile
    import threading

    dim, batch, workers, nets = 2, 4, 9, 5
    row = 2 * dim * dim * 4
    size = _TRT_HEADER.size + row
    pairs = [socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET) for _ in range(workers)]
    batches = []
    lock = threading.Lock()

    def forward(lane, cur_iter, requests):
        if cur_iter == nets:  # a net the server has no engine for
            return None
        time.sleep(0.002)  # long enough that others' requests queue behind this one
        with lock:
            batches.append(len(requests))
        # Each output is the first half of the input times the net's number.
        return [array.array("f", [v * cur_iter for v in array.array("f", request[_TRT_HEADER.size:])[:dim * dim]]
                            ).tobytes() for request in requests]

    launches: dict[int, int] = {}
    server = threading.Thread(target=gather_forwards, args=([b for _, b in pairs], size, batch, 2, forward, launches),
                              daemon=True)
    server.start()
    failures = []

    def worker(index, sock):
        for seq in range(3 * nets):
            cur_iter = seq % nets + 1
            x = array.array("f", [index * 1000 + seq + k for k in range(2 * dim * dim)])
            header = _TRT_HEADER.pack(cur_iter, index, seq)
            sock.send(header + x.tobytes())
            reply = sock.recv(size)
            want = b"" if cur_iter == nets else array.array("f", [v * cur_iter for v in x[:dim * dim]]).tobytes()
            if reply != header + want:
                failures.append((index, seq))
        sock.send(b"short")
        if sock.recv(size) != b"short"[:_TRT_HEADER.size]:
            failures.append((index, "short"))
        sock.close()

    clients = [threading.Thread(target=worker, args=(i, a)) for i, (a, _) in enumerate(pairs)]
    for client in clients:
        client.start()
    for client in clients:
        client.join()
    server.join(10)
    assert not server.is_alive(), "server did not end when every worker's socket closed"
    for _, b in pairs:
        b.close()
    assert not failures, failures
    assert sum(batches) == workers * 3 * (nets - 1), batches
    assert max(batches) <= batch, batches
    assert max(batches) > 1, f"nothing was gathered: {batches}"
    # Launches count every batch, the net the server cannot run included.
    assert sum(launches.values()) >= len(batches) and sum(k * n for k, n in launches.items()) == workers * 3 * nets
    with tempfile.TemporaryDirectory() as tmp:
        write_launches(Path(tmp, "r2d2-trt-batches.txt"), {1: 2, 4: 1})
        assert Path(tmp, "r2d2-trt-batches.txt").read_text() == "launches 3 forwards 6 mean 2.000 sizes 1:2 4:1\n"
    print(f"r2d2 trt gathered self-check passed (mean batch {sum(batches) / len(batches):.1f})")


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


def self_check_prefetch_weights() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        # FIFOs: a write-open succeeds only once the prefetcher has opened one to read it.
        ckpts = [Path(tmp) / f"R2D2_UNet_N{i}.ckpt" for i in (1, 2)]
        for path in ckpts:
            os.mkfifo(path)
        os.environ["R2D2_CKPT_PATH"] = tmp
        try:
            pid = prefetch_weights()
            deadline = time.monotonic() + 10
            for path in ckpts:
                while True:
                    try:
                        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
                        break
                    except OSError:  # ENXIO: no reader yet
                        assert time.monotonic() < deadline, f"{path.name} never read"
                        time.sleep(0.01)
                os.write(fd, b"weights")
                os.close(fd)
            assert os.waitpid(pid, 0)[1] == 0
            os.environ["R2D2_CKPT_PATH"] = str(Path(tmp) / "missing")
            assert os.waitpid(prefetch_weights(), 0)[1] == 0, "no checkpoints is not an error"
        finally:
            os.environ.pop("R2D2_CKPT_PATH")
    print("r2d2 prefetch weights self-check passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check_net_reuse()
        self_check_nufft_plan_reuse()
        self_check_normal_operator()
        self_check_imaging_weights()
        self_check_request_io()
        self_check_lazy_utils()
        self_check_checkpoint_cache()
        self_check_checkpoint_series()
        self_check_threaded_session()
        self_check_early_stop()
        self_check_run_imager()
        self_check_serve_reply_stream()
        self_check_serve_fifo()
        self_check_serve_pool()
        self_check_prefetch_weights()
        self_check_interleaved_pages()
        self_check_worker_nodes()
        self_check_trt_fallback()
        self_check_trt_shared_protocol()
        self_check_trt_gathered()
    elif sys.argv[1:2] == ["--build-trt-plans"]:
        print(build_trt_plans(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])))
    elif sys.argv[1:2] == ["--fifo-dir"]:
        serve_pool(sys.argv[2])
    else:
        serve(sys.argv[2] if sys.argv[1:2] == ["--fifo"] else None)
