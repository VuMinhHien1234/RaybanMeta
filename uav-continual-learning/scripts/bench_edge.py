"""#28 — Đo chi phí biên (edge): optimizer + head, cho câu chuyện UAV on-board.

Chỉ cần torch (không data, không timm) — chạy được trên Mac lẫn VM:
    python scripts/bench_edge.py            # bảng markdown ra stdout
    python scripts/bench_edge.py --steps 100 --device cpu

Đo 3 nhóm:
(a) TRAIN per-step (model proxy ~ViT-S matrix shapes): AdamW | M3-clip ns=5 | M3 ns=0
    (tách chi phí Newton–Schulz) | M3 f=32 (thưa ký ức chậm) — ms/bước + floats state optimizer.
(b) INFERENCE batch=1 trên feature 384-d: head linear | cosine | NCM-lookup — µs/ảnh.
(c) Peak RSS của tiến trình (MB).

Model proxy: stack Linear tái tạo phân bố shape tham số ViT-S (12 block × [384×1536,
1536×384, 384×384 attn]) — đủ để so optimizer TƯƠNG ĐỐI; không thay số đo end-to-end.
"""
from __future__ import annotations

import argparse
import resource
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")  # chạy từ gốc repo không cần pip install -e

from uavcl.optim.m3 import M3                      # noqa: E402
from uavcl.models.classifier import CosineHead     # noqa: E402


def build_proxy(depth: int = 12, dim: int = 384) -> nn.Module:
    """Stack các Linear có shape giống phân bố tham số ViT-S (xấp xỉ đủ dùng cho so optimizer)."""
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers += [nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Linear(dim, dim)]
    layers += [nn.Linear(dim, 45)]
    return nn.Sequential(*layers)


def opt_state_floats(opt: torch.optim.Optimizer) -> int:
    n = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                n += v.numel()
    return n


def bench_train(make_opt, steps: int, device: str, dim: int = 384) -> tuple[float, int]:
    """Trả (ms/bước trung bình sau warmup, floats state optimizer)."""
    torch.manual_seed(0)
    model = build_proxy(dim=dim).to(device)
    opt = make_opt(model.parameters())
    x = torch.randn(32, dim, device=device)
    y = torch.randint(0, 45, (32,), device=device)
    warmup = 5
    t0 = None
    for i in range(steps + warmup):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i + 1 == warmup:
            t0 = time.perf_counter()
    ms = (time.perf_counter() - t0) * 1000.0 / steps
    return ms, opt_state_floats(opt)


@torch.no_grad()
def bench_infer_head(head: nn.Module, steps: int, device: str, dim: int = 384) -> float:
    """µs cho 1 ảnh (batch=1) qua head, sau warmup."""
    head = head.to(device).eval()
    z = torch.randn(1, dim, device=device)
    for _ in range(20):
        head(z)
    t0 = time.perf_counter()
    for _ in range(steps):
        head(z)
    return (time.perf_counter() - t0) * 1e6 / steps


class NCMLookup(nn.Module):
    """Prototype lookup: cosine(z, proto) — mô phỏng chi phí đọc NCM-head (C=45, D=384)."""

    def __init__(self, num_classes: int = 45, dim: int = 384):
        super().__init__()
        self.register_buffer("proto", F.normalize(torch.randn(num_classes, dim), dim=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z, dim=1) @ self.proto.t()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50, help="số bước đo (sau 5 bước warmup)")
    ap.add_argument("--infer-steps", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = args.device
    torch.set_num_threads(torch.get_num_threads())  # dùng mặc định máy — ghi vào header

    rows = []
    configs = [
        ("AdamW", lambda p: torch.optim.AdamW(p, lr=1e-3)),
        ("M3 clip ns=5 f=16", lambda p: M3(p, lr=1e-3, update_norm="clip")),
        ("M3 clip ns=0 (tắt NS)", lambda p: M3(p, lr=1e-3, update_norm="clip", ns_steps=0)),
        ("M3 clip ns=3", lambda p: M3(p, lr=1e-3, update_norm="clip", ns_steps=3)),
        ("M3 clip f=32", lambda p: M3(p, lr=1e-3, update_norm="clip", frequency=32)),
    ]
    for name, mk in configs:
        ms, floats = bench_train(mk, args.steps, dev)
        rows.append((name, ms, floats))

    base = rows[0][1]
    print(f"\n### (a) Train per-step — proxy ViT-S, batch 32, device={dev}, threads={torch.get_num_threads()}\n")
    print("| Optimizer | ms/bước | so AdamW | state (floats) |")
    print("|---|---|---|---|")
    for name, ms, fl in rows:
        print(f"| {name} | {ms:.1f} | {ms / base:.2f}x | {fl:,} |")

    print(f"\n### (b) Inference head, batch=1, D=384, C=45 (µs/ảnh, {args.infer_steps} lần)\n")
    print("| Head | µs/ảnh |")
    print("|---|---|")
    for name, head in [("linear", nn.Linear(384, 45)),
                       ("cosine", CosineHead(384, 45)),
                       ("ncm-lookup", NCMLookup())]:
        us = bench_infer_head(head, args.infer_steps, dev)
        print(f"| {name} | {us:.1f} |")

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 if sys.platform == "linux" else 1024.0 * 1024.0)
    print(f"\n### (c) Peak RSS tiến trình: {peak_mb:.0f} MB")
    print("\nGhi kết quả 2 máy (Mac + VM) vào docs/EDGE_COST.md theo task #28.")


if __name__ == "__main__":
    main()
