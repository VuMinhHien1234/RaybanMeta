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


class TitansMemory(nn.Module):
    def __init__(self, dim: int, chunk_size: int = 64, self_referential: bool = False,
                 self_modifying: bool = False, self_modifying_readpath: bool = False, **mem_kwargs):
        # ↳ dim = độ dài vector; chunk_size = cứ bao nhiêu bước thì ghi ký ức 1 lần;
        #   **mem_kwargs = các cờ ổn định (gated_transition...) truyền thẳng xuống thư viện.
        #   self_referential (TASK 3, NL §8.1 Eq 79): tráo projection cố định -> context-adaptive.
        #   self_modifying (TASK 4, NL §8.1 cuối): value TỰ SINH theo M_{t-1} (bao trùm Task 3).
        super().__init__()
        self.dim = int(dim)
        self.chunk_size = int(chunk_size)
        self.self_modifying = bool(self_modifying)
        # Task 4 bao trùm Task 3: bật self_modifying -> coi như self_referential luôn.
        self.self_referential = bool(self_referential) or self.self_modifying
        if self.self_modifying:
            # bản self-modifying (TASK 4): dựng NeuralMemory rồi nâng value tự sinh theo M_{t-1}.
            # readpath=False (mặc định) = v2 tốt nhất (chỉ value); True = hướng 1 (thêm k/q, đo ra tệ hơn).
            from .self_ref_memory import build_self_modifying_neural_memory
            self.mem = build_self_modifying_neural_memory(
                self.dim, self.chunk_size, readpath=bool(self_modifying_readpath), **mem_kwargs)
        elif self.self_referential:
            # bản self-referential: dựng NeuralMemory rồi tráo to_keys/values/queries (self_ref_memory.py).
            from .self_ref_memory import build_self_ref_neural_memory
            self.mem = build_self_ref_neural_memory(self.dim, self.chunk_size, **mem_kwargs)
        else:
            try:
                from titans_pytorch import NeuralMemory  # ↳ Lớp bộ nhớ thần kinh gốc của thư viện.
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "G2 cần titans-pytorch: pip install titans-pytorch (xem README bước 4)"
                ) from e
            self.mem = NeuralMemory(dim=self.dim, chunk_size=self.chunk_size, **mem_kwargs)  # ↳ Tạo bộ nhớ thật.
        self._no_state_kwarg = False  # version quá cũ không nhận state -> chạy không nối ký ức
        # ↳ Cờ ghi nhớ: nếu phát hiện thư viện quá cũ (không nhận tham số state) thì bật True.
        # DIAGNOSTIC (vá lỗ hổng Task 2): đo η_t (tốc độ ghi, Eq 76) và α_t (cổng quên) — hai thứ
        # điều khiển vụ nổ norm(state). Dùng forward-hook (chỉ ĐỌC output, KHÔNG đổi hành vi model
        # -> zero-risk cho kết quả). Tích lũy trung bình trong 1 task, reset ở begin_task, log ở end_task.
        self._eta_sum = 0.0; self._eta_cnt = 0     # η_t = adaptive step (to_adaptive_step, trước transform)
        self._alpha_sum = 0.0; self._alpha_cnt = 0  # α_t = cổng quên = sigmoid(to_decay_factor) ∈ (0,1)
        self._install_eta_alpha_probes()

    def _install_eta_alpha_probes(self) -> None:
        """Gắn forward-hook lên to_adaptive_step (η) và to_decay_factor (α) để ghi trung bình."""
        def _eta_hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.numel() > 0:
                self._eta_sum += float(t.detach().float().mean()); self._eta_cnt += 1
        def _alpha_hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.numel() > 0:
                self._alpha_sum += float(t.detach().float().sigmoid().mean()); self._alpha_cnt += 1
        if hasattr(self.mem, "to_adaptive_step"):
            self.mem.to_adaptive_step.register_forward_hook(_eta_hook)
        if hasattr(self.mem, "to_decay_factor"):
            self.mem.to_decay_factor.register_forward_hook(_alpha_hook)

    def reset_eta_alpha(self) -> None:
        self._eta_sum = self._alpha_sum = 0.0; self._eta_cnt = self._alpha_cnt = 0

    def eta_alpha_stats(self):
        """(η_t trung bình, α_t trung bình) kể từ lần reset gần nhất; None nếu chưa đo được."""
        eta = self._eta_sum / self._eta_cnt if self._eta_cnt else None
        alpha = self._alpha_sum / self._alpha_cnt if self._alpha_cnt else None
        return eta, alpha

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
