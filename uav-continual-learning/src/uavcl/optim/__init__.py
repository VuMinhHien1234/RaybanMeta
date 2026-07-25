"""Optimizers của dự án. Chọn qua config:

    train:
      optimizer: adamw | m3      # mặc định adamw
      m3:                        # tham số riêng khi dùng m3 (tùy chọn)
        alpha: 0.5
        frequency: 16
        ns_steps: 5
        beta_style: delta        # delta (xấp xỉ dự án) | ema | paper (Algorithm 1)
        key_proj_eta: 0.0        # phép quên rank-1 thử nghiệm của dự án; 0=tắt

CMSOptimizer (G3) sẽ bọc quanh optimizer trả về từ đây — tức CMS chạy được
trên cả AdamW lẫn M3.
"""
# ↳ GIẢI THÍCH TỔNG QUAN: __init__ của package `optim`. Nó có 1 "nhà máy"
#   build_optimizer đọc config rồi trả về AdamW (chuẩn) hoặc M3 (của paper).
#   Nhờ tách ở đây, CMSOptimizer chỉ cần gọi build_optimizer là bọc được cả hai.
from __future__ import annotations

import torch

from .m3 import M3, newton_schulz


def build_optimizer(params, train_cfg: dict) -> torch.optim.Optimizer:
    # ↳ params: có thể là danh sách tham số thường, HOẶC danh sách param-group (CMS truyền vào).
    params = list(params)
    if params and isinstance(params[0], dict):
        # param-groups (CMSOptimizer truyền vào, lr per-tier đã scale) — giữ nguyên
        for g in params:
            g["params"] = [p for p in g["params"] if p.requires_grad]  # ↳ Lọc bỏ tham số bị đóng băng.
    else:
        params = [p for p in params if p.requires_grad]                # ↳ Trường hợp thường: chỉ giữ tham số còn học.
    name = str(train_cfg.get("optimizer", "adamw")).lower()            # ↳ Chọn loại optimizer từ config.
    lr = float(train_cfg.get("lr", 3e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)       # ↳ Optimizer chuẩn (đối chứng).
    if name == "m3":
        m3_cfg = dict(train_cfg.get("m3", {}) or {})                   # ↳ Đọc khối cấu hình riêng của M3.
        delta_cfg = dict(m3_cfg.get("delta", {}) or {})
        return M3(
            params,
            lr=lr,
            weight_decay=wd,
            betas=tuple(m3_cfg.get("betas", (0.9, 0.999, 0.95))),      # ↳ Chuyển từng tham số config -> M3.
            alpha=float(m3_cfg.get("alpha", 0.5)),
            frequency=int(m3_cfg.get("frequency", 16)),
            ns_steps=int(m3_cfg.get("ns_steps", 5)),
            beta_style=str(m3_cfg.get("beta_style", "delta")),
            delta_alpha=tuple(delta_cfg.get("alpha", (0.999, 0.9999))),
            delta_eta=tuple(delta_cfg.get("eta", (0.1, 0.05))),
            update_norm=str(m3_cfg.get("update_norm", "clip")),
            key_proj_eta=float(m3_cfg.get("key_proj_eta", 0.0)),
        )
    raise KeyError(f"Unknown optimizer '{name}' (chọn: adamw | m3)")   # ↳ Tên lạ -> báo lỗi rõ.


__all__ = ["M3", "newton_schulz", "build_optimizer"]  # ↳ Tên công khai của package optim.
