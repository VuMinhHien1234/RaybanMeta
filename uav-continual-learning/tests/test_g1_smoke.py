"""End-to-end smoke của G1 trên dữ liệu giả (tự skip nếu chưa có torch).

Chạy: pytest tests/test_g1_smoke.py -q   (CPU, ~1-2 phút)
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from uavcl.data import build_stream, get_source           # noqa: E402
from uavcl.data.loaders import build_task_loaders          # noqa: E402
from uavcl.engine import run_continual                     # noqa: E402
from uavcl.methods import build_method                     # noqa: E402
from uavcl.metrics import average_accuracy, average_forgetting  # noqa: E402
from uavcl.models import ContinualClassifier, build_backbone    # noqa: E402

DATA_CFG = {
    "name": "synthetic", "num_classes": 4, "train_per_class": 16,
    "val_per_class": 4, "test_per_class": 4, "split_seed": 0,
    "num_tasks": 2, "image_size": 32, "batch_size": 16, "num_workers": 0,
}
TRAIN_CFG = {"epochs_per_task": 2, "lr": 3e-3, "weight_decay": 0.0}


def _build():
    source = get_source(DATA_CFG)
    stream = build_stream(
        source.splits["train"].labels, source.splits["val"].labels,
        source.splits["test"].labels, num_classes=source.num_classes,
        num_tasks=2, seed=0,
    )
    loaders = build_task_loaders(source, stream, DATA_CFG)
    backbone, dim = build_backbone({"name": "tinycnn"})
    model = ContinualClassifier(backbone, dim, source.num_classes)
    return source, stream, loaders, model


def test_finetune_smoke():
    torch.manual_seed(0)
    _, stream, loaders, model = _build()
    method = build_method("finetune", {})
    R, _ = run_continual(model, method, stream, loaders, torch.device("cpu"), TRAIN_CFG, verbose=False)
    assert R.shape == (2, 2)
    assert 0.0 <= R.min() and R.max() <= 1.0
    assert R[0, 0] > 0.5                       # dữ liệu giả tách màu -> học được task đầu
    assert np.isfinite(average_accuracy(R))
    assert np.isfinite(average_forgetting(R))


def test_ewc_penalty_kicks_in():
    torch.manual_seed(0)
    _, stream, loaders, model = _build()
    method = build_method("ewc", {"ewc": {"ewc_lambda": 100.0, "max_batches": 4}})
    device = torch.device("cpu")
    assert method.penalty(model) is None       # trước task đầu: chưa có anchor
    R, _ = run_continual(model, method, stream, loaders, device, TRAIN_CFG, verbose=False)
    pen = method.penalty(model)
    assert pen is not None and torch.isfinite(pen) and pen.item() >= 0.0
    assert len(method._anchors) == 2           # mỗi task một anchor
    assert R.shape == (2, 2)
