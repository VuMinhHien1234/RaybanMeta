"""Test đầu cosine (fix 'recency bias' của đầu Linear — nút thắt forgetting mà NCM-head
đã lộ ra). CHỈ cần torch; tự skip nếu thiếu.

Chạy: pytest tests/test_cosine_head.py -q
"""
import pytest

torch = pytest.importorskip("torch")

from uavcl.models.classifier import (  # noqa: E402
    ContinualClassifier,
    CosineHead,
    build_head,
    mask_logits,
)


def test_build_head_kinds():
    lin = build_head("linear", 8, 5)
    cos = build_head("cosine", 8, 5)
    assert isinstance(lin, torch.nn.Linear)
    assert isinstance(cos, CosineHead)
    with pytest.raises(ValueError):
        build_head("bogus", 8, 5)


def test_cosine_head_shape_and_interface():
    head = CosineHead(8, 5)
    out = head(torch.randn(4, 8))
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()
    # giao diện y hệt nn.Linear -> mask_logits / NCM-head đọc được
    assert head.out_features == 5 and head.in_features == 8


def test_cosine_head_is_magnitude_invariant():
    """Điểm cốt lõi: logit BẤT BIẾN theo độ lớn feature (chỉ phụ thuộc hướng)
    -> đây chính là thứ chống recency bias mà đầu Linear thiếu."""
    torch.manual_seed(0)
    head = CosineHead(8, 5)
    x = torch.randn(4, 8)
    out1 = head(x)
    out2 = head(x * 7.3)                 # nhân feature với hằng số dương
    assert torch.allclose(out1, out2, atol=1e-5)


def test_continual_classifier_head_switch():
    torch.manual_seed(0)
    backbone = torch.nn.Identity()                       # feature = input, khỏi cần timm
    m_lin = ContinualClassifier(backbone, 8, 5)          # mặc định linear (bất biến ngược)
    m_cos = ContinualClassifier(backbone, 8, 5, head="cosine")
    assert isinstance(m_lin.head, torch.nn.Linear)
    assert isinstance(m_cos.head, CosineHead)
    x = torch.randn(3, 8)
    logits = m_cos(x)
    assert logits.shape == (3, 5)
    # mask_logits vẫn chạy trên logit cosine: cột được phép giữ nguyên
    masked = mask_logits(logits, [1, 3])
    assert masked.shape == (3, 5)
    assert torch.allclose(masked[:, [1, 3]], logits[:, [1, 3]])


def test_cosine_head_trains():
    """Sanity: gradient chảy qua cả weight lẫn scale, học tách được 2 class."""
    torch.manual_seed(0)
    head = CosineHead(4, 2)
    opt = torch.optim.Adam(head.parameters(), lr=0.1)
    x0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(16, 1) + 0.01 * torch.randn(16, 4)
    x1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(16, 1) + 0.01 * torch.randn(16, 4)
    x = torch.cat([x0, x1])
    y = torch.tensor([0] * 16 + [1] * 16)
    for _ in range(100):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(head(x), y).backward()
        opt.step()
    acc = (head(x).argmax(1) == y).float().mean()
    assert acc > 0.9
