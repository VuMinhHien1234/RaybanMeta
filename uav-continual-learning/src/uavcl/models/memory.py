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


# BẪY (2026-08-02): `def default_adaptive_step_transform(adaptive_step, max_lr = 1e-2)` ở
# neural_memory.py:255 KHÔNG dùng default của chính nó — dòng :457 luôn truyền
# `partial(..., max_lr = default_step_transform_max_lr)` với `default_step_transform_max_lr = 1.`
# (:272). Nên max_lr THẬT = 1.0, KHÔNG phải 1e-2. `_detect_eta_max_lr()` đọc động, hằng dưới
# chỉ là fallback khi không đọc được.
DEFAULT_ETA_MAX_LR = 1.0
# Trần logit mặc định khi bật `memory.gate_bound` (TASK 4):
#   α = sigmoid(logit)          -> ±2.9444 = ±log(19) -> α ∈ [0.05, 0.95]
#   η = sigmoid(logit) * max_lr -> ±2.1972 = ±log(9)  -> η ∈ [0.1, 0.9] * max_lr
DEFAULT_ALPHA_LOGIT_LIMIT = 2.9444
DEFAULT_ETA_LOGIT_LIMIT = 2.1972
# Vùng η lành mạnh biểu diễn bằng PHÂN SỐ của max_lr (không phụ thuộc thang đo) —
# đây chính là chỗ đã sai lần trước khi hardcode [1e-4, 9e-3] theo giả định max_lr=1e-2.
ETA_FRAC_LO, ETA_FRAC_HI = 0.01, 0.95


class TitansMemory(nn.Module):
    def __init__(self, dim: int, chunk_size: int = 64, self_referential: bool = False,
                 self_modifying: bool = False, self_modifying_readpath: bool = False,
                 gate_bound=None, **mem_kwargs):
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
        #
        # NGỮ NGHĨA α — xem neural_memory.py:634 + :813
        #     decay_factor = self.to_decay_factor(chunked_seq).sigmoid()
        #     update = self.assoc_scan(1. - decay_factor, update, ...)   # Eq (13)
        #   => hệ số truyền vào scan là (1 − α) = hệ số GIỮ LẠI, KHÔNG phải hệ số quên.
        #     α → 1 : giữ lại 0   = XOÁ SẠCH tích luỹ mỗi chunk (memory không tích luỹ, norm phẳng)
        #     α → 0 : giữ lại 100% = KHÔNG QUÊN GÌ  -> HƯỚNG NỔ norm
        #   (Bằng chứng 2026-08-02: run m3_s1 có α=0.0000 -> norm 1.757.552; mọi run α=1.0 có norm ~54.)
        #
        # NGỮ NGHĨA η — xem neural_memory.py:629-630 + :255
        #     adaptive_lr = self.to_adaptive_step(seq)                 <- hook nằm ở ĐÂY = logit THÔ
        #     adaptive_lr = self.adaptive_step_transform(adaptive_lr)  <- = sigmoid(x) * max_lr
        #   => logit thô KHÔNG phải learning rate. η THẬT = sigmoid(logit) * max_lr.
        self._eta_sum = 0.0; self._eta_cnt = 0      # Σ logit THÔ (giữ để so với run cũ)
        self._eta_real_sum = 0.0                    # Σ η THẬT = sigmoid(logit)*max_lr
        self._alpha_sum = 0.0; self._alpha_cnt = 0  # Σ α_t = sigmoid(to_decay_factor) ∈ (0,1)
        self._eta_max_lr = self._detect_eta_max_lr()
        # TASK 4 — chặn trôi cổng. Phải cài TRƯỚC probe để probe đo đúng giá trị SAU khi chặn.
        self._gate_bound = self._install_gate_bounds(gate_bound)
        self._install_eta_alpha_probes()

    def _detect_eta_max_lr(self) -> float:
        """Đọc max_lr thật từ `adaptive_step_transform` (functools.partial) — fallback 1e-2."""
        tf = getattr(self.mem, "adaptive_step_transform", None)
        kw = getattr(tf, "keywords", None)
        if isinstance(kw, dict) and "max_lr" in kw:
            try:
                return float(kw["max_lr"])
            except (TypeError, ValueError):
                pass
        return DEFAULT_ETA_MAX_LR

    def _install_gate_bounds(self, cfg):
        """TASK 4 — chặn logit của HAI cổng để chúng không trôi ra biên (Eq 76 phải sống).

        Vì sao cần: đo quỹ đạo 13 run (scripts/trace_gates.py, 2026-08-02) cho thấy α KHÔNG
        bão hoà từ đầu — nó khởi đầu lành mạnh (0.10–0.89) rồi TRÔI ĐƠN ĐIỆU ra biên trong
        2–4 task, kéo theo logit tăng gần tuyến tính (vd m3_s0: 5.8 → 55.6). Nguyên nhân:
        gradient descent thấy "quên nhiều hơn" làm giảm loss của task HIỆN TẠI, nên liên tục
        đẩy α → 1; không có gì đẩy ngược lại. Chuẩn hoá đầu vào (pre_norm) chỉ đổi thang đo
        lúc khởi tạo, KHÔNG chặn được đà trôi do gradient.

        Cách chặn (BỌC, không fork thư viện — cùng triết lý self_ref_memory.py):
            logit_bounded = tanh(logit / limit) * limit
        Hệ số góc tại 0 đúng bằng 1 (gần như identity ở vùng giữa), bão hoà mượt về ±limit.
        Gradient không bao giờ chết hẳn -> cổng vẫn PHỤ THUỘC DỮ LIỆU, chỉ không thoát ra biên.

        cfg=None/False -> không cài gì (hành vi cũ bất biến).
        """
        if not cfg:
            return None
        cfg = {} if cfg is True else dict(cfg)
        a_lim = float(cfg.get("alpha_logit_limit", DEFAULT_ALPHA_LOGIT_LIMIT))
        e_lim = float(cfg.get("eta_logit_limit", DEFAULT_ETA_LOGIT_LIMIT))
        if a_lim <= 0 or e_lim <= 0:
            raise ValueError("gate_bound.*_logit_limit phải > 0")

        def _make(limit):
            def _bound_hook(_m, _inp, out):
                if isinstance(out, tuple):
                    t = out[0]
                    if not torch.is_tensor(t):
                        return None
                    return (torch.tanh(t / limit) * limit,) + tuple(out[1:])
                if not torch.is_tensor(out):
                    return None
                return torch.tanh(out / limit) * limit
            return _bound_hook

        installed = {}
        if hasattr(self.mem, "to_adaptive_step"):
            self.mem.to_adaptive_step.register_forward_hook(_make(e_lim)); installed["eta"] = e_lim
        if hasattr(self.mem, "to_decay_factor"):
            self.mem.to_decay_factor.register_forward_hook(_make(a_lim)); installed["alpha"] = a_lim
        return installed or None

    def gate_bound_report(self):
        """Mô tả trần cổng đang áp (None nếu tắt) — để in 1 lần lúc dựng model."""
        if not self._gate_bound:
            return None
        import math
        parts = []
        if "alpha" in self._gate_bound:
            L = self._gate_bound["alpha"]; lo = 1 / (1 + math.exp(L))
            parts.append(f"α∈[{lo:.3f},{1 - lo:.3f}] (|logit|≤{L:.4g})")
        if "eta" in self._gate_bound:
            L = self._gate_bound["eta"]; lo = 1 / (1 + math.exp(L))
            parts.append(f"η∈[{lo * self._eta_max_lr:.2e},{(1 - lo) * self._eta_max_lr:.2e}] (|logit|≤{L:.4g})")
        return "  ".join(parts)

    def _install_eta_alpha_probes(self) -> None:
        """Gắn forward-hook lên to_adaptive_step (η) và to_decay_factor (α) để ghi trung bình.

        Đăng ký SAU `_install_gate_bounds` -> PyTorch gọi hook theo thứ tự đăng ký và hook
        trước có thể thay output, nên probe luôn đo giá trị THẬT SỰ được dùng.
        """
        def _eta_hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.numel() > 0:
                d = t.detach().float()
                self._eta_sum += float(d.mean())                                   # logit thô
                self._eta_real_sum += float(d.sigmoid().mean()) * self._eta_max_lr  # η THẬT
                self._eta_cnt += 1
        def _alpha_hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.numel() > 0:
                self._alpha_sum += float(t.detach().float().sigmoid().mean()); self._alpha_cnt += 1
        if hasattr(self.mem, "to_adaptive_step"):
            self.mem.to_adaptive_step.register_forward_hook(_eta_hook)
        if hasattr(self.mem, "to_decay_factor"):
            self.mem.to_decay_factor.register_forward_hook(_alpha_hook)

    def reset_eta_alpha(self) -> None:
        self._eta_sum = self._alpha_sum = self._eta_real_sum = 0.0
        self._eta_cnt = self._alpha_cnt = 0

    def eta_alpha_stats(self):
        """(η logit thô, η THẬT, α) trung bình kể từ lần reset; None nếu chưa đo được.

        η thô giữ lại để so với log của run cũ; η THẬT = sigmoid(logit)*max_lr là con số
        có ý nghĩa vật lý (learning-rate của memory). α: xem ngữ nghĩa ở __init__.
        """
        eta = self._eta_sum / self._eta_cnt if self._eta_cnt else None
        eta_real = self._eta_real_sum / self._eta_cnt if self._eta_cnt else None
        alpha = self._alpha_sum / self._alpha_cnt if self._alpha_cnt else None
        return eta, eta_real, alpha

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
