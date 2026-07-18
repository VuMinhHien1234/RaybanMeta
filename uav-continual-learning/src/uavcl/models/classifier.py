"""Classifier cho class-incremental: backbone + 1 head Linear đủ cột cho MỌI class.

Quy ước của cả dự án (G1 -> G4 giữ nguyên):
- Nhãn là id toàn cục, head luôn có `num_classes` cột.
- Khi TRAIN task t: mask logits về đúng các class của task t.
- Khi EVAL sau task t: mask logits về tập class ĐÃ THẤY (task 0..t).
Nhờ vậy đổi model (thêm Titans/CMS/HOPE) không phải đổi giao thức đo.
"""
# ↳ GIẢI THÍCH TỔNG QUAN: Đây là model đơn giản nhất (baseline G1): backbone + 1 head.
#   "mask_logits" là mẹo quan trọng cả dự án dùng chung: head luôn có đủ cột cho MỌI
#   class, nhưng ta CHE (đặt điểm rất thấp) những class không được phép xét ở bước đó.
#   -> train chỉ tính lỗi trên class của task hiện tại; eval chỉ so trên class đã học.
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

MASK_FILL = -1.0e4  # đủ nhỏ cho fp16/fp32 mà không sinh NaN như -inf
# ↳ Giá trị "che": rất thấp nên sau softmax ~0, nhưng KHÔNG dùng -inf (dễ tạo NaN).


class ContinualClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone                       # ↳ Con mắt.
        self.head = nn.Linear(feat_dim, num_classes)   # ↳ Head: feature -> điểm số từng class.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))             # ↳ ảnh -> feature -> logits (B, num_classes).


def mask_logits(logits: torch.Tensor, allowed: Sequence[int]) -> torch.Tensor:
    """Giữ nguyên cột trong `allowed`, đè các cột khác = MASK_FILL."""
    # ↳ Che mọi class KHÔNG nằm trong `allowed`. Trả về logits mới cùng kích thước.
    out = logits.new_full(logits.shape, MASK_FILL)  # ↳ Tạo tensor toàn giá trị "che" (cùng shape/device/kiểu).
    idx = torch.as_tensor(list(allowed), dtype=torch.long, device=logits.device)  # ↳ Chỉ số cột được phép.
    out[:, idx] = logits[:, idx]                    # ↳ Chép lại điểm thật cho đúng các cột được phép.
    return out
