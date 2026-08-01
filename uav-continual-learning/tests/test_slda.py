"""Test #21 — SLDA (models/slda.py + method 'slda'). CHỈ cần torch; tự skip nếu thiếu.

Kiểm 4 điều:
1. Hai cụm Gaussian tách được -> acc ≈ 1 (LDA phải làm được bài dễ nhất).
2. Streaming = batch: update từng phần cho ĐÚNG cùng (w, b) như update một cục.
3. Class chưa học: score = 0 + mask_logits che được (không rò dự đoán).
4. fit_task chỉ duyệt loader đúng MỘT lần (không đọc lại data cũ).

Chạy: pytest tests/test_slda.py -q
"""
import pytest

torch = pytest.importorskip("torch")

from uavcl.methods import SLDA, build_method          # noqa: E402
from uavcl.models.classifier import mask_logits        # noqa: E402
from uavcl.models.slda import SLDAClassifier           # noqa: E402

FEAT = 8
C = 4


def _make(shrinkage=1e-4):
    return SLDAClassifier(torch.nn.Identity(), FEAT, C, shrinkage=shrinkage)


def _two_clusters(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, FEAT, generator=g) * 0.3 + torch.tensor([3.0] + [0.0] * (FEAT - 1))
    b = torch.randn(n, FEAT, generator=g) * 0.3 - torch.tensor([3.0] + [0.0] * (FEAT - 1))
    x = torch.cat([a, b])
    y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    return x, y


def test_slda_separates_two_gaussians():
    m = _make()
    x, y = _two_clusters()
    m.update(x, y)
    pred = m(x).argmax(dim=1)
    assert (pred == y).float().mean() > 0.99


def test_streaming_equals_batch():
    x, y = _two_clusters(seed=1)
    m1, m2 = _make(), _make()
    m1.update(x, y)                                   # một cục
    for i in range(0, len(y), 32):                    # streaming từng batch 32
        m2.update(x[i:i + 32], y[i:i + 32])
    m1._refresh_cache(); m2._refresh_cache()
    assert torch.allclose(m1._cache_w, m2._cache_w, atol=1e-4)
    assert torch.allclose(m1._cache_b, m2._cache_b, atol=1e-4)


def test_unseen_class_masked():
    m = _make()
    x, y = _two_clusters(seed=2)
    m.update(x, y)                                    # chỉ học class 0, 1
    logits = m(x[:5])
    assert torch.all(logits[:, 2:] == 0.0)            # class chưa học: score đúng 0
    pred = mask_logits(logits, [0, 1]).argmax(dim=1)  # mask về class đã học vẫn chạy
    assert int(pred.max()) <= 1


def test_fit_task_single_pass_and_validation():
    class CountingLoader:
        def __init__(self, x, y):
            self.x, self.y, self.n_iter = x, y, 0

        def __iter__(self):
            self.n_iter += 1
            yield self.x, self.y

    x, y = _two_clusters(seed=3)
    m = _make()
    method = build_method("slda", {})
    assert isinstance(method, SLDA) and method.gradient_free
    loader = CountingLoader(x, y)
    method.fit_task(m, loader, torch.device("cpu"))
    assert loader.n_iter == 1                          # đúng 1 lượt, không đọc lại
    assert method.footprint_floats(m) == m.extra_floats() > 0
    with pytest.raises(ValueError):
        _make(shrinkage=0.0)
