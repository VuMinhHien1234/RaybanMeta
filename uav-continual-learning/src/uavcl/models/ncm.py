"""NCM — Nearest Class Mean trên backbone ĐÓNG BĂNG (baseline thứ 3 của G1).

Vì sao phải có (bài học lấy từ project Meta-Rayban/CPM): trên đặc trưng của
backbone pretrained đóng băng, mỗi class chỉ cần một *prototype trung bình*;
prototype của class cũ KHÔNG BAO GIỜ bị ghi đè khi học class mới -> gần như
không quên, không cần gradient, bộ nhớ bị chặn O(C·D). Nếu Titans (G2) / CMS
(G3) không thắng nổi baseline "ngây thơ" này thì chưa có gì để báo cáo.

Giao diện giữ đúng quy ước dự án: forward(x) -> "logits" (B, num_classes)
= cosine(feature, prototype), nên engine/mask/metrics dùng chung không đổi.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCMClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # NCM định nghĩa trên đặc trưng cố định -> luôn đóng băng backbone.
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        # buffer (không phải Parameter): không có gradient, đi theo state_dict/device
        self.register_buffer("proto_sum", torch.zeros(num_classes, feat_dim))
        self.register_buffer("proto_count", torch.zeros(num_classes))

    def train(self, mode: bool = True):  # noqa: D401
        """Giữ backbone ở eval kể cả khi .train() (BatchNorm không được cập nhật)."""
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def update_prototypes(self, feats: torch.Tensor, ys: torch.Tensor) -> None:
        """Cộng dồn tổng đặc trưng theo class (trung bình online, bounded)."""
        feats = F.normalize(feats.detach().float(), dim=1)
        self.proto_sum.index_add_(0, ys, feats)
        self.proto_count.index_add_(0, ys, torch.ones_like(ys, dtype=torch.float))

    def prototypes(self) -> torch.Tensor:
        cnt = self.proto_count.clamp(min=1.0).unsqueeze(1)
        return F.normalize(self.proto_sum / cnt, dim=1)  # class chưa học -> vector 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(self.backbone(x).float(), dim=1)
        return feats @ self.prototypes().t()  # cosine scores (B, C) — dùng như logits

    def extra_floats(self) -> int:
        """Bộ nhớ thêm ngoài backbone (để so footprint giữa các method)."""
        return int(self.proto_sum.numel() + self.proto_count.numel())
