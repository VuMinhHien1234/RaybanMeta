"""Smoke test Replay + LwF trên dữ liệu giả (tự skip nếu chưa có torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from uavcl.data import build_stream, get_source           # noqa: E402
from uavcl.data.loaders import build_task_loaders          # noqa: E402
from uavcl.engine import run_continual                     # noqa: E402
from uavcl.methods import build_method                     # noqa: E402
from uavcl.models import ContinualClassifier, build_backbone  # noqa: E402

DATA_CFG = {
    "name": "synthetic", "num_classes": 4, "train_per_class": 16,
    "val_per_class": 4, "test_per_class": 4, "split_seed": 0,
    "num_tasks": 2, "image_size": 32, "batch_size": 16, "num_workers": 0,
}
TRAIN_CFG = {"epochs_per_task": 2, "lr": 3e-3, "weight_decay": 0.0, "eval_future": True}


def _run(method_name, method_cfg):
    torch.manual_seed(0)
    source = get_source(DATA_CFG)
    stream = build_stream(
        source.splits["train"].labels, source.splits["val"].labels,
        source.splits["test"].labels, num_classes=source.num_classes, num_tasks=2, seed=0,
    )
    loaders = build_task_loaders(source, stream, DATA_CFG)
    backbone, dim = build_backbone({"name": "tinycnn"})
    model = ContinualClassifier(backbone, dim, source.num_classes)
    method = build_method(method_name, method_cfg)
    R, _ = run_continual(model, method, stream, loaders, torch.device("cpu"), TRAIN_CFG, verbose=False)
    return R, method, model


def test_replay_buffer_bounded_and_runs():
    R, method, model = _run("replay", {"replay": {"buffer_per_class": 4, "replay_batch": 8}})
    assert R.shape == (2, 2) and np.isfinite(R).all()
    # buffer bị chặn: <= 4 ảnh x 4 class
    assert 0 < len(method._buf) <= 4 * 4
    assert method.footprint_floats(model) == sum(x.numel() for x, _ in method._buf)
    # đã ghi cả 2 task vào danh sách class từng thấy
    assert method._seen == [0, 1, 2, 3] or len(method._seen) == 4
    # eval_future bật -> ô R[0,1] được đo (>= 0, dùng cho FWT)
    assert 0.0 <= R[0, 1] <= 1.0


def test_lwf_teacher_created_and_runs():
    R, method, model = _run("lwf", {"lwf": {"lwf_lambda": 0.5, "temperature": 2.0}})
    assert R.shape == (2, 2) and np.isfinite(R).all()
    # sau task 2: teacher tồn tại (được chụp trước task 2), old_classes = cả 4 class
    assert method._teacher is not None
    assert method.footprint_floats(model) > 0
    assert len(method._old_classes) == 4
