"""SeqAdapter (G2, task A3) — "người phiên dịch" ảnh -> chuỗi cho bộ nhớ Titans.

Titans chỉ ăn dữ liệu dạng chuỗi (1, L, D). Adapter quyết định "thời gian" nghĩa là gì:

- image_seq (MẶC ĐỊNH, quyết định Ngày 1): mỗi ẢNH = 1 bước thời gian.
    (B, D) hoặc (B, P, D) -> mean-pool token nếu cần -> (1, B, D).
    Batch giữ đúng thứ tự stream -> chuỗi = "từng lần UAV nhìn thấy cảnh".
- token_seq (để ablation): mỗi PATCH = 1 bước. (B, P, D) -> (1, B*P, D).
    Chuỗi dài, chi tiết trong ảnh, nhưng ý nghĩa "thời gian stream" yếu hơn.

Hàm restore() đưa output của memory về (B, D) cho head phân loại.
"""
# ↳ GIẢI THÍCH TỔNG QUAN: Bộ nhớ Titans nghĩ theo "chuỗi thời gian" (giống câu chữ).
#   Nhưng backbone lại cho ra ảnh rời rạc. Adapter là cầu nối: quy ước 1 bước thời
#   gian là 1 ẢNH (image_seq) hay 1 MẢNH ảnh/patch (token_seq).
#   Ký hiệu: B=số ảnh, P=số patch mỗi ảnh, D=độ dài vector, L=độ dài chuỗi.
from __future__ import annotations

import torch
import torch.nn as nn

MODES = ("image_seq", "token_seq")  # ↳ Hai chế độ hợp lệ; dùng để kiểm tra config.


class SeqAdapter(nn.Module):
    def __init__(self, mode: str = "image_seq"):
        super().__init__()
        mode = str(mode).lower()
        if mode not in MODES:                       # ↳ Chặn cấu hình sai ngay từ đầu.
            raise ValueError(f"seq mode '{mode}' không hợp lệ, chọn {MODES}")
        self.mode = mode
        self._last_bp = None  # (B, P) của lần forward gần nhất, để restore
        # ↳ Nhớ lại kích thước lần gần nhất để hàm restore() biết cách "gấp" chuỗi về lại ảnh.

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # ↳ Nhận feature từ backbone, trả về chuỗi (1, L, D) cho memory.
        if self.mode == "image_seq":
            if feats.dim() == 3:  # (B, P, D) -> gộp token thành 1 vector/ảnh
                feats = feats.mean(dim=1)           # ↳ Trung bình các patch -> mỗi ảnh còn 1 vector.
            assert feats.dim() == 2, f"image_seq cần (B,D)/(B,P,D), nhận {tuple(feats.shape)}"
            self._last_bp = (feats.shape[0], 1)     # ↳ Ghi nhớ B ảnh, mỗi ảnh 1 "token".
            return feats.unsqueeze(0)  # (1, B, D) — B bước thời gian
            # ↳ Thêm chiều batch=1 ở đầu: cả batch ảnh trở thành 1 chuỗi dài B bước.
        # token_seq
        if feats.dim() != 3:                        # ↳ token_seq bắt buộc có chiều patch (B,P,D).
            raise ValueError(
                "token_seq cần token features (B,P,D) — backbone phải hỗ trợ "
                "forward_features (ViT của timm); tinycnn thì dùng image_seq."
            )
        B, P, D = feats.shape                        # ↳ Tách kích thước.
        self._last_bp = (B, P)                       # ↳ Nhớ (B, P) để restore gấp lại đúng.
        return feats.reshape(1, B * P, D)            # ↳ Duỗi mọi patch của mọi ảnh thành 1 chuỗi dài B*P.

    def restore(self, seq_out: torch.Tensor) -> torch.Tensor:
        """(1, L, D) từ memory -> (B, D) cho head (token_seq: trung bình P token/ảnh)."""
        # ↳ Chiều ngược lại: sau khi memory xử lý xong chuỗi, gấp về mỗi ảnh 1 vector cho head.
        B, P = self._last_bp
        out = seq_out.squeeze(0)  # (L, D)           # ↳ Bỏ chiều batch=1 -> (L, D).
        if P == 1:
            return out                               # ↳ image_seq: L đã bằng B, trả thẳng.
        return out.reshape(B, P, -1).mean(dim=1)     # ↳ token_seq: gấp lại (B,P,D) rồi trung bình patch -> (B,D).
