"""Classifier cho class-incremental: backbone + 1 head Linear đủ cột cho MỌI class.

Quy ước của cả dự án (G1 -> G4 giữ nguyên):
- Nhãn là id toàn cục, head luôn có `num_classes` cột.
- Khi TRAIN task t: mask logits về đúng các class của task t.
- Khi EVAL sau task t: mask logits về tập class ĐÃ THẤY (task 0..t).
Nhờ vậy đổi model (thêm Titans/CMS/HOPE) không phải đổi giao thức đo.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

MASK_FILL = -1.0e4  # đủ nhỏ cho fp16/fp32 mà không sinh NaN như -inf


class ContinualClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def mask_logits(logits: torch.Tensor, allowed: Sequence[int]) -> torch.Tensor:
    """Giữ nguyên cột trong `allowed`, đè các cột khác = MASK_FILL."""
    out = logits.new_full(logits.shape, MASK_FILL)
    idx = torch.as_tensor(list(allowed), dtype=torch.long, device=logits.device)
    out[:, idx] = logits[:, idx]
    return out
