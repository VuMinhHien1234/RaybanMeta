"""Test #22 — LatentReplay (method 'latent_replay'). CHỈ cần torch; tự skip nếu thiếu.

Kiểm 3 điều trên ContinualClassifier (backbone Identity — không cần timm/titans):
1. Buffer tôn trọng quota mỗi class + lưu float16.
2. extra_batch_loss trả tensor hữu hạn sau task đầu (task đầu: None).
3. Chống quên thật trên stream giả 2 task: có replay -> acc task cũ cao hơn rõ rệt.

Chạy: pytest tests/test_latent_replay.py -q
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F                        # noqa: E402

from uavcl.methods import build_method                 # noqa: E402
from uavcl.models.classifier import ContinualClassifier, mask_logits  # noqa: E402

FEAT = 8
C = 4
K = 5   # buffer_per_class trong test


def _cluster(label, n=60, seed=0):
    g = torch.Generator().manual_seed(seed + label)
    center = torch.zeros(FEAT)
    center[label] = 4.0
    x = torch.randn(n, FEAT, generator=g) * 0.3 + center
    y = torch.full((n,), label, dtype=torch.long)
    return x, y


def _loader(pairs, batch=16):
    x = torch.cat([p[0] for p in pairs]); y = torch.cat([p[1] for p in pairs])
    return [(x[i:i + batch], y[i:i + batch]) for i in range(0, len(y), batch)]


class _ToyTrunkModel(torch.nn.Module):
    """Giả lập kiến trúc dự án: latent (sau backbone frozen) -> TRUNK CHIA SẺ trainable
    (vai memory) -> head. Quên xảy ra qua (1) trunk trôi theo task mới và (2) weight decay
    ăn mòn hàng head class cũ (đúng recency bias mà NCM-head đã lộ). Backbone Identity ->
    latent = x, forward_from_feats bỏ qua backbone như thật."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Identity()
        self.trunk = torch.nn.Linear(FEAT, FEAT)
        self.head = torch.nn.Linear(FEAT, C)

    def forward(self, x):
        return self.forward_from_feats(x)

    def forward_from_feats(self, z):
        return self.head(torch.tanh(self.trunk(z)))


def _train_stream(use_replay: bool, seed=0):
    torch.manual_seed(seed)
    model = _ToyTrunkModel()
    method = build_method("latent_replay", {"latent_replay": {
        "buffer_per_class": K, "replay_batch": 16, "weight": 1.0, "seed": 0}})
    tasks = [[_cluster(0), _cluster(1)], [_cluster(2), _cluster(3)]]
    for t, pairs in enumerate(tasks):
        allowed = [0, 1] if t == 0 else [2, 3]
        loader = _loader(pairs)
        # weight_decay: hàng head của class cũ không có gradient CE (mask) -> bị ăn mòn dần
        # nếu KHÔNG replay; có replay thì được "ôn" lại mỗi bước. lr/wd chọn cho quên rõ.
        opt = torch.optim.SGD(model.parameters(), lr=0.3, weight_decay=0.1)
        model.train()
        for _ in range(8):
            for x, y in loader:
                loss = F.cross_entropy(mask_logits(model(x), allowed), y)
                if use_replay:
                    extra = method.extra_batch_loss(model, x, model(x), torch.device("cpu"))
                    if extra is not None:
                        loss = loss + extra
                opt.zero_grad(); loss.backward(); opt.step()
        method.end_task(model, loader, torch.device("cpu"), allowed)
    # acc trên task 0 sau khi học xong task 1 (eval mask về mọi class đã thấy)
    model.eval()
    x0, y0 = _cluster(0, n=40, seed=99)[0], _cluster(0, n=40, seed=99)[1]
    with torch.no_grad():
        pred = mask_logits(model(x0), [0, 1, 2, 3]).argmax(dim=1)
    return (pred == y0).float().mean().item(), method


def test_buffer_respects_quota_and_dtype():
    _, method = _train_stream(use_replay=True)
    assert len(method._buf) == 4 * K                   # 4 class × quota K
    assert all(z.dtype == torch.float16 for z, _ in method._buf)
    n_floats = sum(z.numel() for z, _ in method._buf)
    assert n_floats == 4 * K * FEAT


def test_extra_loss_none_then_finite():
    model = ContinualClassifier(torch.nn.Identity(), FEAT, C)
    method = build_method("latent_replay", {"latent_replay": {"buffer_per_class": K}})
    x, y = _cluster(0)
    assert method.extra_batch_loss(model, x, model(x), torch.device("cpu")) is None  # task đầu
    method.end_task(model, _loader([(x, y)]), torch.device("cpu"), [0])
    loss = method.extra_batch_loss(model, x, model(x), torch.device("cpu"))
    assert loss is not None and torch.isfinite(loss)


def test_replay_reduces_forgetting():
    acc_with, _ = _train_stream(use_replay=True, seed=0)
    acc_without, _ = _train_stream(use_replay=False, seed=0)
    assert acc_with >= acc_without + 0.2               # chống quên phải RÕ RỆT trên bài toy
    assert acc_with > 0.85
