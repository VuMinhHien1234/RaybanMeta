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
import torch.nn.functional as F  # ↳ dùng cho chuẩn hoá (đầu cosine).

MASK_FILL = -1.0e4  # đủ nhỏ cho fp16/fp32 mà không sinh NaN như -inf
# ↳ Giá trị "che": rất thấp nên sau softmax ~0, nhưng KHÔNG dùng -inf (dễ tạo NaN).


class CosineHead(nn.Module):
    """Đầu phân loại COSINE (LUCIR-style): chuẩn hoá feature + trọng số trước khi nhân
    -> logit chỉ phụ thuộc HƯỚNG, bất biến ĐỘ LỚN. Nhờ vậy tránh 'recency bias' của đầu
    Linear (độ lớn trọng số class mới phình to lấn class cũ = nguyên nhân forgetting mà
    NCM-head đã lộ ra: Titans RESISC45 seed0 Linear 0.583/F0.274 -> NCM 0.758/F0.078).
    `scale` học được giữ logit đủ 'sắc' cho cross-entropy (~16 chuẩn).

    Đây là cách BIẾN phát hiện của NCM-head thành fix cố định trong model: 'prototype' giờ
    chính là vector trọng số, được gradient cập nhật -> KHÔNG cần dựng prototype / đọc lại
    data cũ / readout riêng. Giao diện y hệt nn.Linear (out_features/in_features) nên
    mask_logits, NCM-head, metrics dùng chung không đổi."""

    def __init__(self, feat_dim: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)  # ↳ 'prototype' học được.
        self.scale = nn.Parameter(torch.tensor(float(scale)))                  # ↳ nhiệt độ học được.
        self.out_features = int(num_classes)  # ↳ để mask_logits / NCM-head đọc như nn.Linear.
        self.in_features = int(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = F.normalize(x, dim=1)                 # ↳ feature -> vector đơn vị.
        wn = F.normalize(self.weight, dim=1)       # ↳ trọng số class -> vector đơn vị.
        return self.scale * (xn @ wn.t())          # ↳ cosine × scale -> logits (B, num_classes).


def build_head(kind: str, feat_dim: int, num_classes: int, scale: float = 16.0) -> nn.Module:
    """Nhà máy chọn đầu phân loại: 'linear' (mặc định, nn.Linear) | 'cosine' (CosineHead)."""
    kind = (kind or "linear").lower()
    if kind == "linear":
        return nn.Linear(feat_dim, num_classes)
    if kind == "cosine":
        return CosineHead(feat_dim, num_classes, scale=scale)
    raise ValueError(f"head '{kind}' không hợp lệ, chọn 'linear' | 'cosine'")


class ContinualClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int, head: str = "linear"):
        super().__init__()
        self.backbone = backbone                       # ↳ Con mắt.
        self.head = build_head(head, feat_dim, num_classes)  # ↳ linear (mặc định) | cosine (chống recency bias).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))             # ↳ ảnh -> feature -> logits (B, num_classes).

    def forward_from_feats(self, feats: torch.Tensor) -> torch.Tensor:
        """Logits từ feature-sau-backbone đã lưu (latent replay #22): bỏ qua backbone."""
        return self.head(feats)


def mask_logits(logits: torch.Tensor, allowed: Sequence[int]) -> torch.Tensor:
    """Giữ nguyên cột trong `allowed`, đè các cột khác = MASK_FILL."""
    # ↳ Che mọi class KHÔNG nằm trong `allowed`. Trả về logits mới cùng kích thước.
    out = logits.new_full(logits.shape, MASK_FILL)  # ↳ Tạo tensor toàn giá trị "che" (cùng shape/device/kiểu).
    idx = torch.as_tensor(list(allowed), dtype=torch.long, device=logits.device)  # ↳ Chỉ số cột được phép.
    out[:, idx] = logits[:, idx]                    # ↳ Chép lại điểm thật cho đúng các cột được phép.
    return out
