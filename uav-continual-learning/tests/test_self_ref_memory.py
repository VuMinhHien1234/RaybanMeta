"""TASK 3 — test self-referential projections (NL.pdf §8.1, Eq 79).

Kiểm chứng 2 tính chất KHÔNG cần titans-pytorch (chạy được mọi nơi có torch):
  1. Init trung tính: gate khởi đầu = 1 -> output ≈ activation(base(x)) (không phá train lúc đầu).
  2. Self-referential: CÙNG một token nhưng NGỮ CẢNH khác -> output KHÁC (projection cố định thì
     không thể khác) -> đúng tinh thần "projection phụ thuộc ngữ cảnh" của Eq 79.
Và 1 test tích hợp (skip nếu thiếu titans-pytorch): tráo projection + forward giữ shape.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from uavcl.models.self_ref_memory import ContextAdaptiveProjection, make_self_referential


def test_init_is_neutral_equals_fixed_projection():
    """to_gate init = 0 -> gate = 1 -> output = activation(base(x)) (trùng projection cố định)."""
    torch.manual_seed(0)
    proj = ContextAdaptiveProjection(8, 16, nn.Identity())
    x = torch.randn(2, 5, 8)
    out = proj(x)
    expected = proj.base(x)  # activation=Identity, gate=1 lúc init
    assert out.shape == (2, 5, 16)
    assert torch.allclose(out, expected, atol=1e-6), "init phải trung tính (= base thuần)"


def test_projection_is_context_dependent():
    """Cùng token đầu tiên, nhưng phần còn lại của chuỗi khác -> output token đó phải KHÁC."""
    torch.manual_seed(0)
    proj = ContextAdaptiveProjection(8, 16, nn.Identity())
    # phá tính trung tính: cho to_gate học một chút để gate phụ thuộc ngữ cảnh
    nn.init.normal_(proj.to_gate.weight, std=0.5)
    tok = torch.randn(1, 1, 8)
    ctx_a = torch.cat([tok, torch.randn(1, 4, 8)], dim=1)
    ctx_b = torch.cat([tok, torch.randn(1, 4, 8) + 3.0], dim=1)  # ngữ cảnh rất khác
    out_a = proj(ctx_a)[:, 0]   # token 0 trong ngữ cảnh A
    out_b = proj(ctx_b)[:, 0]   # CÙNG token 0 trong ngữ cảnh B
    assert not torch.allclose(out_a, out_b, atol=1e-4), \
        "self-referential: cùng token, ngữ cảnh khác -> phải cho projection khác"


def test_make_self_referential_swaps_projection_and_forwards():
    """Tích hợp: tráo projection của NeuralMemory thật + forward giữ shape. Skip nếu thiếu lib."""
    nm = pytest.importorskip("titans_pytorch")
    mem = nm.NeuralMemory(dim=32, chunk_size=8)
    mem = make_self_referential(mem)
    assert isinstance(mem.to_keys, ContextAdaptiveProjection)
    assert isinstance(mem.to_values, ContextAdaptiveProjection)
    assert isinstance(mem.to_queries, ContextAdaptiveProjection)
    seq = torch.randn(1, 16, 32)
    out = mem(seq)
    retrieved = out[0] if isinstance(out, tuple) else out
    assert retrieved.shape == (1, 16, 32), "forward sau khi tráo projection phải giữ shape (1,L,D)"
