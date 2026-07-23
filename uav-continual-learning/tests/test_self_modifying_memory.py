"""TASK 4 — test self-modifying value generation (NL.pdf §8.1 cuối).

Value KHÔNG còn tĩnh: v_t = f(x_t, summary(M_{t-1})) -> vòng tự tham chiếu đầy đủ.

3 test KHÔNG cần titans-pytorch (chạy mọi nơi có torch):
  1. Init trung tính: W_state init = 0 -> nhánh Task 4 = 0 -> output = Task 3 (không phá train).
  2. Self-modifying: CÙNG x, ĐỔI summary(M_{t-1}) -> value KHÁC (cốt lõi "đổi state -> đổi value").
  3. summarize_memory_state: state khác -> summary khác (và không NaN, shape cố định (bh,4)).
2 test tích hợp (cần titans-pytorch):
  4. make_self_modifying: to_values thành SelfModifyingValueProjection + forward giữ shape, không NaN.
  5. Ổn định + hook: forward nối state >=5 lần, norm(state) hữu hạn (không nổ) và hook có bơm
     summary(M_{t-1}) THẬT (non-None) vào value-projection ít nhất 1 lần.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from uavcl.models.self_ref_memory import (
    ContextGatedProjection,
    SelfModifyingValueProjection,
    make_self_modifying,
    summarize_memory_state,
)


def test_init_is_neutral_equals_task3():
    """W_state init = 0 -> nhánh state = 0 -> output = inner(x) (trùng Task 3 lúc khởi đầu)."""
    torch.manual_seed(0)
    inner = nn.Linear(8, 16, bias=False)
    proj = SelfModifyingValueProjection(inner, in_dim=8)
    x = torch.randn(2, 5, 8)
    # kể cả khi ĐÃ bơm summary, nhánh Task 4 vẫn = 0 vì W_state = 0.
    proj.set_state_summary(torch.randn(2, 4))
    out = proj(x)
    assert out.shape == (2, 5, 16)
    assert torch.allclose(out, inner(x), atol=1e-6), "init phải trung tính (= inner thuần, = Task 3)"


def test_value_depends_on_memory_state():
    """CÙNG x, ĐỔI summary(M_{t-1}) -> value phải KHÁC (value cố định không thể khác)."""
    torch.manual_seed(0)
    inner = nn.Linear(8, 16, bias=False)
    proj = SelfModifyingValueProjection(inner, in_dim=8)
    nn.init.normal_(proj.to_state_value.weight, std=0.7)   # kích hoạt nhánh state (bỏ init 0).
    x = torch.randn(1, 5, 8)
    proj.set_state_summary(torch.zeros(1, 4))
    out_state_a = proj(x)
    proj.set_state_summary(torch.ones(1, 4) * 2.0)         # trạng thái memory KHÁC hẳn.
    out_state_b = proj(x)
    assert not torch.allclose(out_state_a, out_state_b, atol=1e-4), \
        "self-modifying: cùng x, khác M_{t-1} -> value phải khác"


def test_summarize_memory_state_shape_and_sensitivity():
    """summary: shape (bh,4), không NaN, và state khác -> summary khác."""
    w_a = {"l0.w": torch.zeros(1, 6, 6), "l0.b": torch.zeros(1, 6)}
    w_b = {"l0.w": torch.randn(1, 6, 6) * 5.0, "l0.b": torch.randn(1, 6)}
    s_a = summarize_memory_state(w_a)
    s_b = summarize_memory_state(w_b)
    assert s_a.shape == (1, 4) and s_b.shape == (1, 4)
    assert torch.isfinite(s_a).all() and torch.isfinite(s_b).all()
    assert not torch.allclose(s_a, s_b), "state khác nhau -> summary phải khác"
    assert summarize_memory_state({}) is None, "không có tensor -> None (bỏ nhánh, về Task 3)"


# ---------------------------------------------------------------- tích hợp (cần titans-pytorch)

def test_make_self_modifying_swaps_value_and_forwards():
    """to_values -> SelfModifyingValueProjection; forward giữ shape (1,L,D), không NaN."""
    nm = pytest.importorskip("titans_pytorch")
    mem = nm.NeuralMemory(dim=32, chunk_size=8)
    mem = make_self_modifying(mem)
    assert isinstance(mem.to_values, SelfModifyingValueProjection)
    # k/q vẫn context-gated (Task 3).
    assert isinstance(mem.to_queries, ContextGatedProjection)
    assert isinstance(mem.to_keys, ContextGatedProjection)
    seq = torch.randn(1, 16, 32)
    out = mem(seq)
    retrieved = out[0] if isinstance(out, tuple) else out
    assert retrieved.shape == (1, 16, 32)
    assert torch.isfinite(retrieved).all(), "forward không được ra NaN/Inf"


def test_state_feedback_stable_and_hook_fires():
    """Nối state >=5 lần: norm(state) hữu hạn (không nổ) + hook bơm summary(M_{t-1}) THẬT."""
    nm = pytest.importorskip("titans_pytorch")
    torch.manual_seed(0)
    mem = nm.NeuralMemory(dim=32, chunk_size=8)
    mem = make_self_modifying(mem)

    # ghi lại xem hook có set summary non-None (tức có M_{t-1} thật) hay không.
    seen_non_none = []
    orig_set = mem.to_values.set_state_summary
    def _rec(s):
        seen_non_none.append(s is not None)
        return orig_set(s)
    mem.to_values.set_state_summary = _rec

    def _state_norm(state) -> float:
        # state là NeuralMemState; gom norm mọi tensor trong updates/weights.
        total = 0.0
        for obj in (getattr(state, "weights", None), getattr(state, "updates", None)):
            if obj is None:
                continue
            for t in obj.values():
                if torch.is_tensor(t):
                    total += float(t.detach().float().norm())
        return total

    state = None
    for _ in range(6):
        seq = torch.randn(1, 16, 32)
        out, state = mem(seq, state=state)
        # detach giống truncated-BPTT trong titans_head để không tích graph.
        from uavcl.models.state_utils import detach_state
        state = detach_state(state)
        n = _state_norm(state)
        assert torch.isfinite(torch.tensor(n)), "norm(state) phải hữu hạn"
        assert n < 1e4, f"norm(state) nổ: {n}"
        assert torch.isfinite(out).all(), "output phải hữu hạn"

    assert any(seen_non_none), "hook phải bơm summary(M_{t-1}) non-None ít nhất 1 lần"
