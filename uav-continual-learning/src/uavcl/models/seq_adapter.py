"""SeqAdapter (G2, task A3) — "người phiên dịch" ảnh -> chuỗi cho bộ nhớ Titans.

Titans chỉ ăn dữ liệu dạng chuỗi (1, L, D). Adapter quyết định "thời gian" nghĩa là gì:

- image_seq (MẶC ĐỊNH, quyết định Ngày 1): mỗi ẢNH = 1 bước thời gian.
    (B, D) hoặc (B, P, D) -> mean-pool token nếu cần -> (1, B, D).
    Batch giữ đúng thứ tự stream -> chuỗi = "từng lần UAV nhìn thấy cảnh".
- token_seq (để ablation): mỗi PATCH = 1 bước. (B, P, D) -> (1, B*P, D).
    Chuỗi dài, chi tiết trong ảnh, nhưng ý nghĩa "thời gian stream" yếu hơn.

Hàm restore() đưa output của memory về (B, D) cho head phân loại.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODES = ("image_seq", "token_seq")


class SeqAdapter(nn.Module):
    def __init__(self, mode: str = "image_seq"):
        super().__init__()
        mode = str(mode).lower()
        if mode not in MODES:
            raise ValueError(f"seq mode '{mode}' không hợp lệ, chọn {MODES}")
        self.mode = mode
        self._last_bp = None  # (B, P) của lần forward gần nhất, để restore

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.mode == "image_seq":
            if feats.dim() == 3:  # (B, P, D) -> gộp token thành 1 vector/ảnh
                feats = feats.mean(dim=1)
            assert feats.dim() == 2, f"image_seq cần (B,D)/(B,P,D), nhận {tuple(feats.shape)}"
            self._last_bp = (feats.shape[0], 1)
            return feats.unsqueeze(0)  # (1, B, D) — B bước thời gian
        # token_seq
        if feats.dim() != 3:
            raise ValueError(
                "token_seq cần token features (B,P,D) — backbone phải hỗ trợ "
                "forward_features (ViT của timm); tinycnn thì dùng image_seq."
            )
        B, P, D = feats.shape
        self._last_bp = (B, P)
        return feats.reshape(1, B * P, D)

    def restore(self, seq_out: torch.Tensor) -> torch.Tensor:
        """(1, L, D) từ memory -> (B, D) cho head (token_seq: trung bình P token/ảnh)."""
        B, P = self._last_bp
        out = seq_out.squeeze(0)  # (L, D)
        if P == 1:
            return out
        return out.reshape(B, P, -1).mean(dim=1)
