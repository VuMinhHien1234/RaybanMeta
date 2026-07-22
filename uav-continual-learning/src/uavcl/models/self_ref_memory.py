"""SelfRefNeuralMemory (TASK 3 — Deep Self-Referential Titans, NL.pdf §8.1, Eq 79).

Ý tưởng paper: các projection W_k, W_v, W_q trong Transformer/associative-memory bị ĐÓNG
BĂNG sau pretrain (Eq 76) -> khả năng "hiểu ngữ cảnh" bị chặn. Paper cho các projection này
TỰ CẬP NHẬT theo ngữ cảnh (Eq 79) -> "self-referential".

Bản này = "simple version" của Eq 79 (paper cũng nêu một bản đơn giản, chia sẻ giá trị):
thay projection cố định bằng `ContextAdaptiveProjection` — projection có trọng số hiệu dụng
ĐIỀU BIẾN THEO NGỮ CẢNH của chuỗi hiện tại. KHÔNG đụng thuật toán store/retrieve của
titans-pytorch (chỉ tráo module projection, giữ đúng call signature) -> rủi ro thấp, dễ test.

Đây CHƯA phải bản đầy đủ (projection-là-memory-bền-vững có init meta-học — đó là TASK 3-full
kết hợp TASK 5). Bản này là bước đi được, kiểm chứng được, để đo self-referential có giúp không.

⚠️ CHƯA CHẠY TRAIN KIỂM CHỨNG (sandbox không có torch). Trước khi tin: trên GCP chạy
   `pytest tests/test_self_ref_memory.py -q` rồi 1 run quick, đọc log norm(state).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ContextAdaptiveProjection(nn.Module):
    """Thay cho Sequential(LinearNoBias(in,out), activation) trong NeuralMemory.

    Giữ NGUYÊN call signature: (b, n, in) -> (b, n, out). Khác biệt: trọng số hiệu dụng
    được nhân với một CỔNG tính từ ngữ cảnh chuỗi (mean theo thời gian) -> projection
    "tự điều chỉnh theo context" thay vì cố định.

        base = W_base(x)                         # projection nền (init = projection gốc)
        ctx  = mean_t(x)                         # tóm tắt ngữ cảnh chuỗi
        gate = 2 * sigmoid(W_gate(ctx))          # (0,2), init ~1  -> khởi đầu ≈ base thuần
        out  = activation(base * gate)

    Init W_gate = 0 -> gate ban đầu = 1 -> hành vi ban đầu TRÙNG projection cũ (ổn định),
    rồi mạng học cách điều biến. O(dim) chi phí, không tạo ma trận (dim×dim).
    """

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=False)  # ↳ projection nền (sẽ copy weight gốc).
        self.to_gate = nn.Linear(in_dim, out_dim)           # ↳ cổng điều biến theo ngữ cảnh.
        self.activation = activation
        nn.init.zeros_(self.to_gate.weight)  # ↳ init 0 -> gate=sigmoid(0)=0.5 -> 2*0.5=1 -> = base thuần.
        nn.init.zeros_(self.to_gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)                                 # ↳ (b, n, out) hoặc (n, out) — theo x.
        ctx = x.mean(dim=-2, keepdim=True)                  # ↳ trung bình theo trục thời gian = ngữ cảnh.
        gate = 2.0 * torch.sigmoid(self.to_gate(ctx))       # ↳ (…,1,out) ∈ (0,2), init≈1.
        return self.activation(base * gate)                 # ↳ projection tự điều biến theo context.


def _swap_projection(module: nn.Module, attr: str) -> None:
    """Tráo `module.<attr>` (Sequential(Linear, activation)) -> ContextAdaptiveProjection,
    ĐỌC shape/activation từ module gốc và GIỮ init trọng số nền (không phá khởi tạo của lib)."""
    old = getattr(module, attr)
    # titans-pytorch: to_queries/to_keys/to_values đều là Sequential(LinearNoBias, activation).
    if not (isinstance(old, nn.Sequential) and len(old) >= 2 and hasattr(old[0], "weight")):
        raise TypeError(
            f"{attr} không phải Sequential(Linear, activation) như kỳ vọng "
            f"(titans-pytorch đổi cấu trúc?) — kiểm tra lại neural_memory.py"
        )
    lin, activation = old[0], old[1]
    out_dim, in_dim = lin.weight.shape                      # nn.Linear weight = (out, in)
    new = ContextAdaptiveProjection(in_dim, out_dim, activation)
    with torch.no_grad():
        new.base.weight.copy_(lin.weight)                   # ↳ giữ đúng init projection gốc.
    setattr(module, attr, new)                              # ↳ đăng ký lại submodule (params tự vào graph).


def make_self_referential(mem: nn.Module, targets=("to_queries", "to_keys", "to_values")) -> nn.Module:
    """Biến một NeuralMemory (đã dựng) thành self-referential bằng cách tráo các projection."""
    for attr in targets:
        _swap_projection(mem, attr)
    return mem


def build_self_ref_neural_memory(dim: int, chunk_size: int, **mem_kwargs):
    """Dựng NeuralMemory chuẩn của titans-pytorch rồi tráo projection sang context-adaptive.

    Tách rời khỏi TitansMemory để dễ test độc lập. Ném ImportError rõ nếu thiếu titans-pytorch.
    """
    try:
        from titans_pytorch import NeuralMemory
    except ImportError as e:  # pragma: no cover
        raise ImportError("Task 3 cần titans-pytorch: pip install -U titans-pytorch") from e
    mem = NeuralMemory(dim=int(dim), chunk_size=int(chunk_size), **mem_kwargs)
    return make_self_referential(mem)
