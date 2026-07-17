"""TitansMemory (G2, task A1) — bọc `titans_pytorch.NeuralMemory` sau MỘT API cố định:

    out, state = memory(seq, state=None)     # seq: (1, L, D) -> out: (1, L, D)

Vì sao phải bọc: (1) cả dự án chỉ phụ thuộc chữ ký này — titans-pytorch đổi
version chỉ sửa file này; (2) chuẩn hoá giá trị trả về giữa các version
(có/không state kwarg, trả tuple hay tensor).

NeuralMemory là bộ nhớ "vừa đọc vừa tự ghi": trọng số nội bộ được cập nhật
ONLINE theo delta-rule ngay trong forward (test-time learning), theo từng
khúc `chunk_size` bước. `state` gói toàn bộ ký ức tích lũy — truyền state
của lần forward trước vào lần sau = trí nhớ nối dài (bậc B/C).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn


class TitansMemory(nn.Module):
    def __init__(self, dim: int, chunk_size: int = 64, **mem_kwargs):
        super().__init__()
        try:
            from titans_pytorch import NeuralMemory
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "G2 cần titans-pytorch: pip install titans-pytorch (xem README bước 4)"
            ) from e
        self.dim = int(dim)
        self.chunk_size = int(chunk_size)
        self.mem = NeuralMemory(dim=self.dim, chunk_size=self.chunk_size, **mem_kwargs)
        self._no_state_kwarg = False  # version quá cũ không nhận state -> chạy không nối ký ức

    def forward(self, seq: torch.Tensor, state=None):
        assert seq.dim() == 3, f"seq phải là (1, L, D), nhận {tuple(seq.shape)}"
        # ĐỆM chuỗi cho tròn bội số chunk_size (lặp lại frame cuối), xong CẮT về độ dài gốc.
        # Lý do: batch lẻ cuối epoch (vd 96 ảnh, chunk 64) làm titans-pytorch lệch sổ
        # chunk nội bộ -> RuntimeError "size of tensor a (2) must match b (3)".
        L = seq.shape[1]
        pad = (-L) % self.chunk_size
        if pad:
            seq = torch.cat([seq, seq[:, -1:, :].expand(-1, pad, -1)], dim=1)
        if self._no_state_kwarg:
            raw = self.mem(seq)
        else:
            try:
                raw = self.mem(seq, state=state)
            except TypeError:
                # version không hỗ trợ state kwarg — vẫn chạy được nhưng KHÔNG nối ký ức
                self._no_state_kwarg = True
                warnings.warn(
                    "titans-pytorch version này không nhận state= — bậc B/C sẽ không nối "
                    "ký ức. Nâng cấp: pip install -U titans-pytorch", stacklevel=2,
                )
                raw = self.mem(seq)

        # Chuẩn hoá output: (retrieved, next_state) | retrieved
        if isinstance(raw, tuple) and len(raw) == 2:
            out, next_state = raw
        else:
            out, next_state = raw, None
        if pad:
            out = out[:, :L, :]  # cắt phần đệm, trả đúng độ dài gốc
        return out, next_state

    def extra_floats(self) -> int:
        """Tham số của module bộ nhớ (chưa tính state — cộng riêng ở classifier)."""
        return sum(p.numel() for p in self.mem.parameters())
