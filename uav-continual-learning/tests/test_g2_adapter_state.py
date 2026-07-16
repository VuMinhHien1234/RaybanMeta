"""Test SeqAdapter + state_utils (chỉ cần torch, KHÔNG cần titans-pytorch)."""
import pytest

torch = pytest.importorskip("torch")

from uavcl.models.seq_adapter import SeqAdapter                     # noqa: E402
from uavcl.models.state_utils import (                              # noqa: E402
    clone_state, count_floats, detach_state, state_norm,
)


def test_adapter_image_seq_from_pooled_and_tokens():
    a = SeqAdapter("image_seq")
    seq = a(torch.randn(8, 384))            # (B, D)
    assert seq.shape == (1, 8, 384)
    assert a.restore(seq).shape == (8, 384)
    seq2 = a(torch.randn(8, 196, 384))      # (B, P, D) -> mean-pool
    assert seq2.shape == (1, 8, 384)


def test_adapter_token_seq():
    a = SeqAdapter("token_seq")
    seq = a(torch.randn(4, 196, 384))
    assert seq.shape == (1, 4 * 196, 384)
    assert a.restore(seq).shape == (4, 384)
    with pytest.raises(ValueError):
        a(torch.randn(4, 384))              # token_seq cần (B,P,D)


def test_adapter_bad_mode():
    with pytest.raises(ValueError):
        SeqAdapter("gibberish")


def _fake_state():
    # cấu trúc lồng nhau giả lập NeuralMemState: namedtuple + dict + tensor + int
    from collections import namedtuple

    NS = namedtuple("NS", ["seq_index", "weights", "cache"])
    return NS(seq_index=7,
              weights={"w1": torch.randn(4, 4, requires_grad=True)},
              cache=[torch.randn(3), None])


def test_state_utils_roundtrip():
    s = _fake_state()
    d = detach_state(s)
    assert d.seq_index == 7                                 # int giữ nguyên
    assert d.weights["w1"].requires_grad is False           # gradient bị cắt
    c = clone_state(s)
    c.weights["w1"].add_(100.0)                             # sửa bản sao...
    assert float(s.weights["w1"].detach().abs().max()) < 100.0  # ...bản gốc không đổi
    assert state_norm(s) > 0.0
    assert state_norm(None) == 0.0
    assert count_floats(s) == 4 * 4 + 3
