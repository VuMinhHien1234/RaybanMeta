"""SelfRefNeuralMemory (TASK 3 — Deep Self-Referential Titans, NL.pdf §8.1, Eq 79).

Ý tưởng paper: projection W_k, W_v, W_q bị ĐÓNG BĂNG sau pretrain (Eq 76) -> khả năng hiểu
ngữ cảnh bị chặn. Paper cho các projection này TỰ ĐIỀU CHỈNH theo ngữ cảnh (Eq 79) =
"self-referential". Bản này là "simple version": nhân output projection với một CỔNG tính
từ ngữ cảnh chuỗi -> projection tự điều biến theo context.

CÁCH LÀM (bản v2, version-agnostic): KHÔNG rebuild projection (dễ vỡ khi titans-pytorch đổi
cấu trúc giữa các version), mà BỌC nguyên module projection gốc lại:
    out = inner(x) * gate(mean_t(x))
- inner = projection gốc (giữ NGUYÊN mọi trọng số/khởi tạo của thư viện, bất kể cấu trúc).
- gate init = 1 -> khởi đầu TRÙNG hành vi cũ (ổn định), rồi mạng học cách điều biến.
Nhờ bọc thay vì rebuild, code chạy được với mọi phiên bản to_queries/to_keys/to_values.

Đây CHƯA phải bản đầy đủ (projection-là-memory-bền có init meta-học — TASK 3-full + TASK 5).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ContextGatedProjection(nn.Module):
    """Bọc một projection gốc (bất kể cấu trúc) và nhân output với cổng ngữ cảnh.

        base = inner(x)                          # projection gốc (giữ nguyên)
        gate = 2 * sigmoid(W_gate(mean_t(x)))    # (…,1,out) ∈ (0,2), init ~1
        out  = base * gate

    Giữ ĐÚNG call signature (…, n, in) -> (…, n, out) như module gốc. Init W_gate=0 ->
    gate=1 -> out=inner(x) (trùng hành vi cũ lúc khởi đầu, không phá train). O(dim), không
    tạo ma trận (dim×dim). out_dim được dò bằng 1 lần forward thử trong __init__ (mọi tham số
    được đăng ký NGAY để optimizer nhìn thấy).
    """

    def __init__(self, inner: nn.Module, in_dim: int):
        super().__init__()
        self.inner = inner
        param = next(inner.parameters())          # ↳ lấy device/dtype khớp module gốc.
        with torch.no_grad():                     # ↳ forward thử để biết out_dim (bất kể cấu trúc inner).
            probe = torch.zeros(1, 2, in_dim, device=param.device, dtype=param.dtype)
            out_dim = int(inner(probe).shape[-1])
        self.to_gate = nn.Linear(in_dim, out_dim, device=param.device, dtype=param.dtype)
        nn.init.zeros_(self.to_gate.weight)       # ↳ init 0 -> gate=1 -> khởi đầu = inner thuần.
        nn.init.zeros_(self.to_gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.inner(x)                              # ↳ (…, n, out) — projection gốc.
        ctx = x.mean(dim=-2, keepdim=True)                # ↳ (…, 1, in) tóm tắt ngữ cảnh chuỗi.
        gate = 2.0 * torch.sigmoid(self.to_gate(ctx))     # ↳ (…, 1, out) ∈ (0,2), init≈1.
        return base * gate                                # ↳ projection tự điều biến theo context.


def _first_linear_in_features(module: nn.Module) -> int:
    """Tìm in_features của Linear ĐẦU TIÊN trong projection = độ dài feature vào (= memory dim).
    Robust với mọi cấu trúc (Sequential / Linear trần / bọc thêm rearrange...)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    raise TypeError(
        "Không tìm thấy nn.Linear trong projection để suy in_dim — "
        "titans-pytorch đổi cấu trúc lạ? In thử: print(type(mem.to_queries), mem.to_queries)"
    )


def _wrap_projection(module: nn.Module, attr: str) -> None:
    """Bọc module.<attr> bằng ContextGatedProjection (giữ nguyên module gốc bên trong)."""
    old = getattr(module, attr)
    in_dim = _first_linear_in_features(old)
    setattr(module, attr, ContextGatedProjection(old, in_dim))  # ↳ đăng ký lại -> params vào graph.


def make_self_referential(mem: nn.Module, targets=("to_queries", "to_keys", "to_values")) -> nn.Module:
    """Biến một NeuralMemory (đã dựng) thành self-referential: bọc các projection k/v/q."""
    for attr in targets:
        if hasattr(mem, attr):
            _wrap_projection(mem, attr)
    return mem


def build_self_ref_neural_memory(dim: int, chunk_size: int, **mem_kwargs):
    """Dựng NeuralMemory chuẩn rồi bọc projection sang context-gated. Ném ImportError rõ nếu thiếu lib."""
    try:
        from titans_pytorch import NeuralMemory
    except ImportError as e:  # pragma: no cover
        raise ImportError("Task 3 cần titans-pytorch: pip install -U titans-pytorch") from e
    mem = NeuralMemory(dim=int(dim), chunk_size=int(chunk_size), **mem_kwargs)
    return make_self_referential(mem)


# =====================================================================================
# TASK 4 — Self-modifying value generation (NL.pdf §8.1 cuối; self-modifying kiểu
# Schmidhuber). Ở Eq 76/79 value vẫn là v_t = to_values(x_t) — TĨNH theo trạng thái memory.
# Task 4 cho value TỰ SINH theo trạng thái memory M_{t-1}: v_t = f(x_t, summary(M_{t-1})).
# Đây là mắt xích còn tĩnh cuối cùng trong chuỗi store; đóng nó lại tạo VÒNG TỰ THAM CHIẾU
# đầy đủ (value ghi vào M lại phụ thuộc chính M).
#
# CÁCH LÀM (bám triết lý Task 3: BỌC, không rebuild — version-agnostic):
#   1) to_values được thay bằng SelfModifyingValueProjection = ContextGatedProjection (Task 3)
#      + một nhánh phụ:  v += beta * tanh(W_state · summary(M_{t-1})).
#   2) summary(M_{t-1}) được BƠM vào projection ngay trước mỗi lần store, bằng cách BỌC
#      NeuralMemory.store_memories: nó nhận `weights` (chính là M_{t-1}) làm tham số vị trí
#      thứ 2 -> ta tóm tắt weights thành vài thống kê rồi set vào value-projection.
#
# AN TOÀN SỐ (roadmap nhấn mạnh "norm(state) không nổ"):
#   - W_state init = 0  -> lúc khởi đầu nhánh Task 4 = 0 -> output TRÙNG Task 3 (so 1-biến sạch).
#   - tanh() chặn nhánh state trong [-1,1]*beta -> bounded, không thể khuếch đại vô hạn.
#   - summary được nén log (log1p) và DETACH -> không mở thêm đường gradient chảy vào M_{t-1}
#     (giữ nguyên đường học của titans-pytorch qua per_sample_grad_fn), chỉ là tín hiệu điều kiện.
# =====================================================================================

_STATE_SUMMARY_DIM = 4  # [mean, rms, mean_abs, max_abs] (đã nén log) — cố định bất kể depth/heads.


def summarize_memory_state(weights) -> "torch.Tensor | None":
    """Tóm tắt M_{t-1} (TensorDict trọng số memory) -> tensor (bh, 4) đã nén log, DETACH.

    Mỗi tensor trọng số có shape (bh, *param) với bh = batch*heads (init_weights của titans).
    Ta rút [mean, rms, mean_abs, max_abs] trên mỗi tham số rồi trung bình qua các tham số ->
    (bh, 4). Nén log giữ được thứ tự độ lớn khi state phình (reset=never, chuỗi dài) mà không
    bão hoà tanh sớm. Trả None nếu không có tensor nào (không bơm nhánh state -> về Task 3).
    """
    stats = []
    for t in weights.values():
        if not torch.is_tensor(t):
            continue
        bh = t.shape[0]
        flat = t.reshape(bh, -1).detach().float()
        mean = flat.mean(dim=1)
        rms = flat.pow(2).mean(dim=1).clamp_min(1e-12).sqrt()
        mean_abs = flat.abs().mean(dim=1)
        max_abs = flat.abs().amax(dim=1)
        # nén log: giữ dấu cho mean; log1p cho các đại lượng độ lớn (>=0).
        comp = torch.stack([
            torch.sign(mean) * torch.log1p(mean.abs()),
            torch.log1p(rms),
            torch.log1p(mean_abs),
            torch.log1p(max_abs),
        ], dim=1)                                   # (bh, 4)
        stats.append(comp)
    if not stats:
        return None
    return torch.stack(stats, dim=0).mean(dim=0)    # trung bình qua các tham số -> (bh, 4)


class SelfModifyingValueProjection(ContextGatedProjection):
    """Value projection TỰ SINH theo trạng thái memory (Task 4).

        base = ContextGatedProjection.forward(x)         # = Task 3 (inner(x) * context-gate)
        v    = base + beta * tanh(W_state · summary(M_{t-1}))   # nhánh self-modifying (Task 4)

    W_state init = 0 -> nhánh Task 4 = 0 lúc đầu -> v = base (TRÙNG Task 3, init trung tính).
    summary được set qua set_state_summary() ngay trước mỗi store; None -> bỏ nhánh (an toàn).
    """

    def __init__(self, inner: nn.Module, in_dim: int, state_summary_dim: int = _STATE_SUMMARY_DIM):
        super().__init__(inner, in_dim)                       # dựng self.inner + self.to_gate (dò out_dim).
        out_dim = self.to_gate.out_features
        w = self.to_gate.weight                               # ↳ khớp device/dtype với phần Task 3.
        self.to_state_value = nn.Linear(state_summary_dim, out_dim, device=w.device, dtype=w.dtype)
        nn.init.zeros_(self.to_state_value.weight)            # ↳ init 0 -> nhánh state = 0 -> = Task 3.
        nn.init.zeros_(self.to_state_value.bias)
        self.state_beta = nn.Parameter(torch.ones((), device=w.device, dtype=w.dtype))  # biên độ học được.
        self._state_summary = None                           # (bh|1, sdim) hoặc None; set trước mỗi store.

    def set_state_summary(self, summary) -> None:
        """Bơm summary(M_{t-1}) (hoặc None để tắt nhánh cho lần forward tới)."""
        self._state_summary = summary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)                             # ↳ value theo Task 3 (context-gated).
        s = self._state_summary
        if s is None:
            return out
        # khớp batch: value_seq có batch b; summary có bh = b*heads. Nếu lệch -> gộp về (1,sdim) broadcast.
        if s.shape[0] != out.shape[0]:
            s = s.mean(dim=0, keepdim=True)
        term = torch.tanh(self.to_state_value(s.to(out.dtype)))   # (b|1, out) ∈ (-1,1), bounded.
        term = term.unsqueeze(-2)                            # (b|1, 1, out) -> broadcast theo chiều thời gian.
        return out + self.state_beta * term                 # value tự sinh theo M_{t-1}.


def _install_store_hook(mem: nn.Module, value_proj: "SelfModifyingValueProjection") -> None:
    """Bọc mem.store_memories: tóm tắt `weights` (=M_{t-1}) và bơm vào value_proj trước mỗi store.

    titans-pytorch gọi store_memories(store_seq, weights, seq_index=..., ...) — weights là VỊ TRÍ 2.
    Ta không đổi luồng: chỉ set summary trước, rồi gọi bản gốc; xong dọn về None (tránh rò summary cũ
    sang lần forward khác). weights=None (chunk đầu tiên, chưa có M) -> bỏ nhánh (về Task 3)."""
    orig_store = mem.store_memories                          # bound method gốc (giữ nguyên).

    def wrapped_store(seq, weights=None, *args, **kwargs):
        try:
            value_proj.set_state_summary(
                summarize_memory_state(weights) if weights is not None else None
            )
            return orig_store(seq, weights, *args, **kwargs)
        finally:
            value_proj.set_state_summary(None)               # dọn: chỉ sống trong đúng 1 store.

    mem.store_memories = wrapped_store                       # shadow method ở cấp instance.


def make_self_modifying(mem: nn.Module) -> nn.Module:
    """Biến NeuralMemory thành SELF-MODIFYING (Task 4), bao trùm cả self-referential (Task 3).

    - to_keys, to_queries: bọc context-gated (Task 3) như cũ.
    - to_values: thay bằng SelfModifyingValueProjection (Task 3 gate + nhánh self-modifying Task 4).
    - store_memories: bọc để bơm summary(M_{t-1}) vào value-projection mỗi lần store.
    """
    # k/q: giữ đúng Task 3 (context-gated).
    for attr in ("to_queries", "to_keys"):
        if hasattr(mem, attr):
            _wrap_projection(mem, attr)
    # v: nâng lên self-modifying (kế thừa Task 3 gate).
    if not hasattr(mem, "to_values"):
        raise TypeError("NeuralMemory không có to_values — titans-pytorch đổi cấu trúc lạ?")
    old_v = mem.to_values
    in_dim = _first_linear_in_features(old_v)
    value_proj = SelfModifyingValueProjection(old_v, in_dim)
    mem.to_values = value_proj                               # đăng ký lại -> params vào graph.
    _install_store_hook(mem, value_proj)                    # nối vòng tự tham chiếu qua M_{t-1}.
    return mem


def build_self_modifying_neural_memory(dim: int, chunk_size: int, **mem_kwargs):
    """Dựng NeuralMemory chuẩn rồi nâng lên self-modifying (Task 4). Ném ImportError rõ nếu thiếu lib."""
    try:
        from titans_pytorch import NeuralMemory
    except ImportError as e:  # pragma: no cover
        raise ImportError("Task 4 cần titans-pytorch: pip install -U titans-pytorch") from e
    mem = NeuralMemory(dim=int(dim), chunk_size=int(chunk_size), **mem_kwargs)
    return make_self_modifying(mem)
