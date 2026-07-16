"""Optimizers của dự án. Chọn qua config:

    train:
      optimizer: adamw | m3      # mặc định adamw
      m3:                        # tham số riêng khi dùng m3 (tùy chọn)
        alpha: 0.5
        frequency: 16
        ns_steps: 5
        beta_style: ema          # ema | paper (nguyên văn Algorithm 1)

CMSOptimizer (G3) sẽ bọc quanh optimizer trả về từ đây — tức CMS chạy được
trên cả AdamW lẫn M3.
"""
from __future__ import annotations

import torch

from .m3 import M3, newton_schulz


def build_optimizer(params, train_cfg: dict) -> torch.optim.Optimizer:
    params = list(params)
    if params and isinstance(params[0], dict):
        # param-groups (CMSOptimizer truyền vào, lr per-tier đã scale) — giữ nguyên
        for g in params:
            g["params"] = [p for p in g["params"] if p.requires_grad]
    else:
        params = [p for p in params if p.requires_grad]
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("lr", 3e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "m3":
        m3_cfg = dict(train_cfg.get("m3", {}) or {})
        delta_cfg = dict(m3_cfg.get("delta", {}) or {})
        return M3(
            params,
            lr=lr,
            weight_decay=wd,
            betas=tuple(m3_cfg.get("betas", (0.9, 0.999, 0.95))),
            alpha=float(m3_cfg.get("alpha", 0.5)),
            frequency=int(m3_cfg.get("frequency", 16)),
            ns_steps=int(m3_cfg.get("ns_steps", 5)),
            beta_style=str(m3_cfg.get("beta_style", "delta")),
            delta_alpha=tuple(delta_cfg.get("alpha", (0.999, 0.9999))),
            delta_eta=tuple(delta_cfg.get("eta", (0.1, 0.05))),
        )
    raise KeyError(f"Unknown optimizer '{name}' (chọn: adamw | m3)")


__all__ = ["M3", "newton_schulz", "build_optimizer"]
