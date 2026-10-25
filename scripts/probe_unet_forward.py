"""Time R2D2's U-Net forward alone, 32x32, one checkpoint (docs/csd3-speed.md, round 11).

Inside r2d2.sif with the checkpoints at /checkpoints:
    python3 scripts/probe_unet_forward.py
B batches B images per forward (prints ms per image). DIM, N (forwards) and
T (torch threads) as named. Run P copies at once to see what a full node does:
    for i in $(seq 24); do apptainer exec ... python3 probe_unet_forward.py & done; wait
"""
import os, sys, time
import torch
sys.path.insert(0, "/opt/r2d2/R2D2-RI/src")
from utils.util_model import create_net_imaging

torch.backends.mkldnn.deterministic = True  # as r2d2_serve.warm_imports
torch.set_num_threads(int(os.environ.get("T", "1")))
net = create_net_imaging()
ck = torch.load("/checkpoints/R2D2_A1/R2D2_UNet_N1.ckpt", map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
net.load_state_dict({'.'.join(k.split('.')[1:]): v for k, v in sd.items()}, assign=True)
net.eval()
b, dim, n = int(os.environ.get("B", "1")), int(os.environ.get("DIM", "32")), int(os.environ.get("N", "200"))
torch.manual_seed(0)
x = torch.randn(b, 2, dim, dim)
with torch.no_grad():
    ref = torch.cat([net(x[i:i + 1]) for i in range(b)])
    y = net(x)
    for _ in range(5):
        net(x)
    t = time.perf_counter()
    for _ in range(n):
        net(x)
    dt = (time.perf_counter() - t) / n
print(f"{dt / b * 1e3:.2f} ms/image B={b} params={sum(p.numel() for p in net.parameters()) * 4 / 1e6:.0f}MB "
      f"batch_identical={torch.equal(y, ref)} maxrel={float((y - ref).abs().max() / ref.abs().max()):.2g}")
