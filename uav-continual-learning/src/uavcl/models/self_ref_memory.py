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
