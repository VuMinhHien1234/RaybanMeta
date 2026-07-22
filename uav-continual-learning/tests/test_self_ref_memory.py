"""TASK 3 — test self-referential projections (NL.pdf §8.1, Eq 79) — bản v2 (wrapper).

2 test KHÔNG cần titans-pytorch (chạy mọi nơi có torch):
  1. Init trung tính: gate khởi đầu = 1 -> output = inner(x) (không phá train lúc đầu).
  2. Self-referential: CÙNG token, NGỮ CẢNH khác -> output KHÁC (projection cố định không thể khác).
1 test tích hợp: bọc projection của NeuralMemory THẬT + forward giữ shape (version-agnostic).
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from uavcl.models.self_ref_memory import ContextGatedProjection, make_self_referential


def test_init_is_neutral_equals_inner():
    """to_gate init = 0 -> gate = 1 -> output = inner(x) (trùng projection gốc)."""
    torch.manual_seed(0)
    inner = nn.Linear(8, 16, bias=False)
    proj = ContextGatedProjection(inner, in_dim=8)
    x = torch.randn(2, 5, 8)
    out = proj(x)
    assert out.shape == (2, 5, 16)
    assert torch.allclose(out, inner(x), atol=1e-6), "init phải trung tính (= inner thuần)"


def test_projection_is_context_dependent():
    """Cùng token đầu, phần còn lại của chuỗi khác -> output token đó phải KHÁC."""
    torch.manual_seed(0)
    inner = nn.Linear(8, 16, bias=False)
    proj = ContextGatedProjection(inner, in_dim=8)
    nn.init.normal_(proj.to_gate.weight, std=0.5)   # để gate phụ thuộc ngữ cảnh
    tok = torch.randn(1, 1, 8)
    ctx_a = torch.cat([tok, torch.randn(1, 4, 8)], dim=1)
    ctx_b = torch.cat([tok, torch.randn(1, 4, 8) + 3.0], dim=1)  # ngữ cảnh rất khác
    out_a = proj(ctx_a)[:, 0]
    out_b = proj(ctx_b)[:, 0]   # CÙNG token 0, ngữ cảnh khác
    assert not torch.allclose(out_a, out_b, atol=1e-4), \
        "self-referential: cùng token, ngữ cảnh khác -> projection phải khác"


def test_make_self_referential_swaps_projection_and_forwards():
    """Tích hợp: bọc projection của NeuralMemory thật + forward giữ shape. Skip nếu thiếu lib."""
    nm = pytest.importorskip("titans_pytorch")
    mem = nm.NeuralMemory(dim=32, chunk_size=8)
    mem = make_self_referential(mem)
    assert isinstance(mem.to_queries, ContextGatedProjection)
    assert isinstance(mem.to_keys, ContextGatedProjection)
    assert isinstance(mem.to_values, ContextGatedProjection)
    seq = torch.randn(1, 16, 32)
    out = mem(seq)
    retrieved = out[0] if isinstance(out, tuple) else out
    assert retrieved.shape == (1, 16, 32), "forward sau khi bọc projection phải giữ shape (1,L,D)"
