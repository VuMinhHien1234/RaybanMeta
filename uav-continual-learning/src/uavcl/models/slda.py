"""SLDA — Streaming Linear Discriminant Analysis trên backbone ĐÓNG BĂNG (task #21).

Vì sao cần (plans/TASKS_UAV_CL.md #21): baseline streaming rẻ + mạnh, chuẩn cho
continual learning trên feature cố định (Hayes & Kanan 2020). Khác NCM ở chỗ ngoài
trung bình mỗi class còn học MA TRẬN HIỆP PHƯƠNG SAI CHUNG của feature -> ranh giới
lớp xét cả "hình dạng" đám mây feature, không chỉ khoảng cách tới tâm.

Toán (streaming, KHÔNG đọc lại data cũ, mỗi mẫu chỉ thấy 1 lần):
    tích luỹ:  s_c  += f        (tổng feature theo class)
               n_c  += 1
               G    += f fᵀ     (tổng outer product toàn cục)
    suy ra:    μ_c  = s_c / n_c
               Σ_w  = (G − Σ_c n_c μ_c μ_cᵀ) / N          (within-class covariance, chính xác)
               Λ    = (Σ_w + ε I)⁻¹                        (shrinkage ε)
    dự đoán:   score_c(f) = (Λ μ_c)ᵀ f − ½ μ_cᵀ Λ μ_c      (LDA discriminant, prior đều)

Giao diện giữ đúng quy ước dự án: forward(x) -> "logits" (B, num_classes), nên
engine/mask_logits/metrics dùng chung không đổi. Prototype/covariance của class cũ
không bị ghi đè khi học class mới -> gần như không quên (như NCM), nhưng biết thêm
tương quan chiều feature. Bộ nhớ: C·D + D² + C float (D=384 -> ~150K float, không phình).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SLDAClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int,
                 shrinkage: float = 1e-4):
        super().__init__()
        if shrinkage <= 0.0:
            raise ValueError(f"shrinkage phải > 0 (nhận {shrinkage})")
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)          # ↳ SLDA định nghĩa trên feature CỐ ĐỊNH.
        self.backbone.eval()
        self.shrinkage = float(shrinkage)
        # buffer: đi theo state_dict/device, KHÔNG bị optimizer đụng vào.
        self.register_buffer("feat_sum", torch.zeros(num_classes, feat_dim))   # s_c
        self.register_buffer("count", torch.zeros(num_classes))                # n_c
        self.register_buffer("gram", torch.zeros(feat_dim, feat_dim))          # G = Σ f fᵀ
        self._cache_w = None       # (D, C) = Λ μ_cᵀ — cache để không nghịch đảo mỗi batch
        self._cache_b = None       # (C,)   = −½ μ_c Λ μ_c
        self._cache_version = -1   # bump theo _version mỗi lần update
        self._version = 0

    def train(self, mode: bool = True):  # noqa: D401
        """Backbone luôn eval (thống kê BatchNorm/LayerNorm không trôi)."""
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def update(self, feats: torch.Tensor, ys: torch.Tensor) -> None:
        """Hấp thụ 1 batch (streaming): cộng dồn s_c, n_c, G. Không giữ lại mẫu."""
        f = feats.detach().float()
        if not torch.isfinite(f).all():
            raise FloatingPointError("SLDA nhận feature chứa NaN/Inf")
        self.feat_sum.index_add_(0, ys, f)
        self.count.index_add_(0, ys, torch.ones_like(ys, dtype=torch.float))
        self.gram.add_(f.t() @ f)            # ↳ Σ f fᵀ cộng dồn theo batch (D×D).
        self._version += 1

    @torch.no_grad()
    def _refresh_cache(self) -> None:
        """Tính Λ, trọng số tuyến tính + bias từ thống kê tích luỹ (lazy, chỉ khi có update mới)."""
        if self._cache_version == self._version:
            return
        d = self.feat_sum.shape[1]
        n_total = float(self.count.sum())
        mu = self.feat_sum / self.count.clamp(min=1.0).unsqueeze(1)            # (C, D)
        # within-class covariance: (G − Σ_c n_c μ_c μ_cᵀ) / N   (class chưa học: n_c=0 -> không góp)
        between = (self.count.unsqueeze(1) * mu).t() @ mu                       # Σ n_c μ μᵀ (D×D)
        sigma = (self.gram - between) / max(n_total, 1.0)
        sigma = 0.5 * (sigma + sigma.t())                                       # ↳ ép đối xứng (sai số float)
        lam = torch.linalg.inv(sigma + self.shrinkage * torch.eye(d, device=sigma.device))
        w = lam @ mu.t()                                                        # (D, C) = Λ μᵀ
        b = -0.5 * (mu * (mu @ lam)).sum(dim=1)                                 # (C,)
        # class chưa có mẫu: score = 0 mọi nơi -> mask_logits sẽ che, không ảnh hưởng.
        empty = self.count < 1.0
        w[:, empty] = 0.0
        b[empty] = 0.0
        self._cache_w, self._cache_b = w, b
        self._cache_version = self._version

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x).float()
        self._refresh_cache()
        return feats @ self._cache_w + self._cache_b    # (B, C) discriminant scores — dùng như logits

    def extra_floats(self) -> int:
        return int(self.feat_sum.numel() + self.count.numel() + self.gram.numel())
