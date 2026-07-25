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
# ↳ GIẢI THÍCH TỔNG QUAN (điểm cốt lõi của Titans): bộ nhớ này KHÁC mạng thường ở
#   chỗ nó TỰ SỬA trọng số ngay trong lúc chạy (forward), chứ không chỉ khi train.
#   Mỗi khi "đọc" 1 chuỗi, nó ghi luôn cái mới vào ký ức (delta-rule). Cục ký ức đó
#   là `state`; đưa state cũ vào lần sau = nhớ xuyên thời gian.
# ↳ Class này chỉ là lớp "áo khoác" quanh thư viện titans-pytorch để cả dự án gọi
#   qua 1 giao diện duy nhất; đổi version thư viện chỉ cần sửa mỗi file này.
from __future__ import annotations

import warnings  # ↳ Để cảnh báo (không dừng chương trình) khi version thư viện quá cũ.

import torch
import torch.nn as nn


def _install_zero_safe_internal_grad_clip(neural_memory_module) -> None:
    """Fix titans-pytorch<=0.4.22 soft clipping when an internal grad norm is zero."""
    current = neural_memory_module.softclamp_grad_norm
    if getattr(current, "_uavcl_zero_safe", False):
        return

    def zero_safe_softclamp_grad_norm(t: torch.Tensor, max_value: float) -> torch.Tensor:
        if t.numel() == 0:
            return t

        flat = t.reshape(*t.shape[:2], -1)
        norm = flat.norm(dim=-1, keepdim=True)
        half_max = max_value / 2
        clamped_norm = ((norm / half_max).tanh() * half_max) + half_max
        scale = torch.where(norm > 0, clamped_norm / norm.clamp_min(torch.finfo(norm.dtype).tiny), 0.0)
        return (flat * scale).reshape_as(t)

    zero_safe_softclamp_grad_norm._uavcl_zero_safe = True
    neural_memory_module.softclamp_grad_norm = zero_safe_softclamp_grad_norm


class TitansMemory(nn.Module):
    def __init__(self, dim: int, chunk_size: int = 64, **mem_kwargs):
        # ↳ dim = độ dài vector; chunk_size = cứ bao nhiêu bước thì ghi ký ức 1 lần;
        #   **mem_kwargs = các cờ ổn định (gated_transition...) truyền thẳng xuống thư viện.
        super().__init__()
        try:
            from titans_pytorch import NeuralMemory  # ↳ Lớp bộ nhớ thần kinh gốc của thư viện.
            from titans_pytorch import neural_memory as neural_memory_module
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "G2 cần titans-pytorch: pip install titans-pytorch (xem README bước 4)"
            ) from e
        if mem_kwargs.get("max_grad_norm") is not None:
            _install_zero_safe_internal_grad_clip(neural_memory_module)
        self.dim = int(dim)
        self.chunk_size = int(chunk_size)
        self.mem = NeuralMemory(dim=self.dim, chunk_size=self.chunk_size, **mem_kwargs)  # ↳ Tạo bộ nhớ thật.
        self._no_state_kwarg = False  # version quá cũ không nhận state -> chạy không nối ký ức
        # ↳ Cờ ghi nhớ: nếu phát hiện thư viện quá cũ (không nhận tham số state) thì bật True.

    def forward(self, seq: torch.Tensor, state=None):
        # ↳ seq: (1, L, D). state: ký ức trước đó (None = bắt đầu trắng).
        assert seq.dim() == 3, f"seq phải là (1, L, D), nhận {tuple(seq.shape)}"
        # ĐỆM chuỗi cho tròn bội số chunk_size (lặp lại frame cuối), xong CẮT về độ dài gốc.
        # Lý do: batch lẻ cuối epoch (vd 96 ảnh, chunk 64) làm titans-pytorch lệch sổ
        # chunk nội bộ -> RuntimeError "size of tensor a (2) must match b (3)".
        L = seq.shape[1]                          # ↳ Độ dài chuỗi thật.
        pad = (-L) % self.chunk_size              # ↳ Cần đệm thêm bao nhiêu bước cho tròn bội số chunk_size.
        if pad:
            seq = torch.cat([seq, seq[:, -1:, :].expand(-1, pad, -1)], dim=1)
            # ↳ Lặp lại frame cuối `pad` lần rồi nối vào đuôi -> chuỗi tròn chunk (giá trị đệm sẽ bị cắt sau).
        if self._no_state_kwarg:
            raw = self.mem(seq)                   # ↳ Thư viện cũ: gọi không kèm state.
        else:
            try:
                raw = self.mem(seq, state=state)  # ↳ Bình thường: truyền ký ức cũ vào để nối tiếp.
            except TypeError:
                # version không hỗ trợ state kwarg — vẫn chạy được nhưng KHÔNG nối ký ức
                self._no_state_kwarg = True       # ↳ Nhớ để lần sau khỏi thử lại.
                warnings.warn(
                    "titans-pytorch version này không nhận state= — bậc B/C sẽ không nối "
                    "ký ức. Nâng cấp: pip install -U titans-pytorch", stacklevel=2,
                )
                raw = self.mem(seq)

        # Chuẩn hoá output: (retrieved, next_state) | retrieved
        # ↳ Tùy version, thư viện trả về (kết quả, ký ức mới) HOẶC chỉ kết quả -> đồng bộ về 2 biến.
        if isinstance(raw, tuple) and len(raw) == 2:
            out, next_state = raw
        else:
            out, next_state = raw, None
        if pad:
            out = out[:, :L, :]  # cắt phần đệm, trả đúng độ dài gốc  ↳ Bỏ các bước đệm thêm lúc nãy.
        return out, next_state                    # ↳ Trả (kết quả đọc, ký ức cập nhật) cho classifier.

    def extra_floats(self) -> int:
        """Tham số của module bộ nhớ (chưa tính state — cộng riêng ở classifier)."""
        # ↳ Đếm số tham số của riêng bộ nhớ, để báo cáo "tốn thêm bao nhiêu" so với backbone trần.
        return sum(p.numel() for p in self.mem.parameters())
