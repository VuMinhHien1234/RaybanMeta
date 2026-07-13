#!/usr/bin/env python3
"""G0 smoke: verify the Titans neural-memory module runs (forward + backward).

Run from project root:  python scripts/smoke_titans.py
Confirms the 'memorize at test time' module works before we wire it to UAV data (G2).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    try:
        import torch
        from titans_pytorch import NeuralMemory
    except ImportError as e:
        print(f"[SKIP] missing dependency: {e.name}. Install per README (torch + titans-pytorch).")
        return 1

    from uavcl.utils import get_device, seed_everything

    seed_everything(0)
    device = get_device()
    dim, chunk = 384, 64

    mem = NeuralMemory(dim=dim, chunk_size=chunk).to(device)
    seq = torch.randn(1, 256, dim, device=device)          # (batch, seq_len, dim)
    retrieved, state = mem(seq)                             # forward
    assert retrieved.shape == seq.shape, (retrieved.shape, seq.shape)
    retrieved.float().sum().backward()                     # backward (gradients flow)

    print(f"device: {device}")
    print(f"[OK] NeuralMemory forward {tuple(seq.shape)} -> {tuple(retrieved.shape)}; backward ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
