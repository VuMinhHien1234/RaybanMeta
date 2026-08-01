"""NCM — Nearest Class Mean trên backbone ĐÓNG BĂNG (baseline thứ 3 của G1).

Vì sao phải có (bài học lấy từ project Meta-Rayban/CPM): trên đặc trưng của
backbone pretrained đóng băng, mỗi class chỉ cần một *prototype trung bình*;
prototype của class cũ KHÔNG BAO GIỜ bị ghi đè khi học class mới -> gần như
không quên, không cần gradient, bộ nhớ bị chặn O(C·D). Nếu Titans (G2) / CMS
(G3) không thắng nổi baseline "ngây thơ" này thì chưa có gì để báo cáo.

Giao diện giữ đúng quy ước dự án: forward(x) -> "logits" (B, num_classes)
= cosine(feature, prototype), nên engine/mask/metrics dùng chung không đổi.
"""
# ↳ GIẢI THÍCH TỔNG QUAN: NCM là baseline "ngây thơ mà mạnh". Ý tưởng: mỗi class chỉ
#   giữ 1 vector trung bình (prototype) của các ảnh thuộc class đó. Phân loại = xem
#   feature ảnh mới GẦN prototype nào nhất (đo bằng cosine). Vì prototype class cũ
#   không bị đụng khi thêm class mới -> gần như KHÔNG QUÊN. Đây là "vạch chuẩn" mà
#   các phương pháp phức tạp (Titans/CMS/HOPE) phải vượt qua mới có ý nghĩa.
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F  # ↳ Chứa các hàm tiện ích như normalize.

from .classifier import MASK_FILL


class PrototypeHead(nn.Module):
    """Stateful cosine NCM head with bounded ``O(num_classes * feat_dim)`` memory."""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        if int(feat_dim) <= 0 or int(num_classes) <= 0:
            raise ValueError("feat_dim và num_classes phải là số nguyên dương")
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.register_buffer("proto_sum", torch.zeros(self.num_classes, self.feat_dim))
        self.register_buffer("proto_count", torch.zeros(self.num_classes))
        self.register_buffer("seen_mask", torch.zeros(self.num_classes, dtype=torch.bool))

    def _validate_features(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim != 2 or feats.shape[1] != self.feat_dim:
            raise ValueError(
                f"features phải có shape (B, {self.feat_dim}), nhận {tuple(feats.shape)}"
            )
        feats = feats.detach().to(device=self.proto_sum.device, dtype=torch.float32)
        if not torch.isfinite(feats).all():
            raise FloatingPointError("NCM features chứa NaN/Inf")
        return feats

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        feats = self._validate_features(feats)
        labels = labels.detach().to(device=self.proto_sum.device, dtype=torch.long)
        if labels.ndim != 1 or labels.shape[0] != feats.shape[0]:
            raise ValueError(
                f"labels phải có shape ({feats.shape[0]},), nhận {tuple(labels.shape)}"
            )
        if labels.numel() == 0:
            return
        if int(labels.min()) < 0 or int(labels.max()) >= self.num_classes:
            raise ValueError(f"NCM label phải nằm trong [0, {self.num_classes - 1}]")

        normalized = F.normalize(feats, dim=1)
        self.proto_sum.index_add_(0, labels, normalized)
        self.proto_count.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        self.seen_mask[labels.unique()] = True
        if not torch.isfinite(self.proto_sum).all() or not torch.isfinite(self.proto_count).all():
            raise FloatingPointError("NCM prototype state chứa NaN/Inf")

    def prototypes(self) -> torch.Tensor:
        counts = self.proto_count.clamp(min=1.0).unsqueeze(1)
        prototypes = F.normalize(self.proto_sum / counts, dim=1)
        if not torch.isfinite(prototypes).all():
            raise FloatingPointError("NCM prototypes chứa NaN/Inf")
        return prototypes

    @torch.no_grad()
    def replace_prototypes(
        self, class_mask: torch.Tensor, prototypes: torch.Tensor
    ) -> None:
        """Replace selected prototype directions while preserving class counts."""
        class_mask = torch.as_tensor(
            class_mask, device=self.seen_mask.device, dtype=torch.bool
        )
        if class_mask.shape != self.seen_mask.shape:
            raise ValueError(
                f"class_mask phải có shape {tuple(self.seen_mask.shape)}, "
                f"nhận {tuple(class_mask.shape)}"
            )
        if bool((class_mask & ~self.seen_mask).any()):
            raise ValueError("không thể transport prototype của class chưa thấy")
        values = self._validate_features(prototypes)
        if values.shape[0] != int(class_mask.sum()):
            raise ValueError("số prototype thay thế không khớp class_mask")
        normalized = F.normalize(values, dim=1)
        counts = self.proto_count[class_mask].clamp(min=1.0).unsqueeze(1)
        self.proto_sum[class_mask] = normalized * counts
        if not torch.isfinite(self.proto_sum).all():
            raise FloatingPointError("NCM prototype state chứa NaN/Inf sau transport")

    def logits(
        self, feats: torch.Tensor, allowed: Sequence[int] | None = None
    ) -> torch.Tensor:
        feats = F.normalize(self._validate_features(feats), dim=1)
        logits = feats @ self.prototypes().t()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("NCM cosine logits chứa NaN/Inf")

        usable = self.seen_mask.clone()
        if allowed is not None:
            allowed_mask = torch.zeros_like(usable)
            allowed_ids = torch.as_tensor(
                list(allowed), device=allowed_mask.device, dtype=torch.long
            )
            if allowed_ids.numel():
                if int(allowed_ids.min()) < 0 or int(allowed_ids.max()) >= self.num_classes:
                    raise ValueError("allowed chứa class id ngoài phạm vi NCM")
                allowed_mask[allowed_ids] = True
            usable &= allowed_mask
        if not bool(usable.any()):
            raise RuntimeError("NCM chưa có prototype cho class được phép đánh giá")
        return logits.masked_fill(~usable.unsqueeze(0), MASK_FILL)

    @torch.no_grad()
    def reset(self) -> None:
        self.proto_sum.zero_()
        self.proto_count.zero_()
        self.seen_mask.zero_()

    def extra_floats(self) -> int:
        return int(self.proto_sum.numel() + self.proto_count.numel())


class NCMClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # NCM định nghĩa trên đặc trưng cố định -> luôn đóng băng backbone.
        for p in self.backbone.parameters():
            p.requires_grad_(False)  # ↳ Không học backbone: NCM chỉ dựa trên feature có sẵn.
        self.backbone.eval()
        self.prototype_head = PrototypeHead(feat_dim, num_classes)

    @property
    def proto_sum(self) -> torch.Tensor:
        return self.prototype_head.proto_sum

    @property
    def proto_count(self) -> torch.Tensor:
        return self.prototype_head.proto_count

    def train(self, mode: bool = True):  # noqa: D401
        """Giữ backbone ở eval kể cả khi .train() (BatchNorm không được cập nhật)."""
        super().train(mode)
        self.backbone.eval()  # ↳ Ép backbone luôn eval để thống kê không trôi.
        return self

    @torch.no_grad()  # ↳ Cập nhật prototype không cần gradient.
    def update_prototypes(self, feats: torch.Tensor, ys: torch.Tensor) -> None:
        """Cộng dồn tổng đặc trưng theo class (trung bình online, bounded)."""
        self.prototype_head.update(feats, ys)

    def prototypes(self) -> torch.Tensor:
        return self.prototype_head.prototypes()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prototype_head.logits(self.backbone(x))

    def extra_floats(self) -> int:
        """Bộ nhớ thêm ngoài backbone (để so footprint giữa các method)."""
        return self.prototype_head.extra_floats()
