"""Test nhánh optimizer_per_task=false — ký ức optimizer sống xuyên task (tự skip nếu thiếu torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from uavcl.data import build_stream, get_source           # noqa: E402
from uavcl.data.loaders import build_task_loaders          # noqa: E402
from uavcl.engine import run_continual, train_one_task     # noqa: E402
from uavcl.methods import build_method                     # noqa: E402
from uavcl.models import ContinualClassifier, build_backbone  # noqa: E402

DATA_CFG = {
    "name": "synthetic", "num_classes": 4, "train_per_class": 12,
    "val_per_class": 4, "test_per_class": 4, "split_seed": 0,
    "num_tasks": 2, "image_size": 32, "batch_size": 8, "num_workers": 0,
}


def _setup():
    torch.manual_seed(0)
    source = get_source(DATA_CFG)
    stream = build_stream(
        source.splits["train"].labels, source.splits["val"].labels,
        source.splits["test"].labels, num_classes=source.num_classes, num_tasks=2, seed=0,
    )
    loaders = build_task_loaders(source, stream, DATA_CFG)
    backbone, dim = build_backbone({"name": "tinycnn"})
    model = ContinualClassifier(backbone, dim, source.num_classes)
    return stream, loaders, model


def test_train_one_task_reuses_given_optimizer():
    stream, loaders, model = _setup()
    method = build_method("finetune", {})
    cfg = {"epochs_per_task": 1, "lr": 1e-3, "optimizer": "m3"}
    _, opt1 = train_one_task(model, method, loaders[0]["train"], torch.device("cpu"),
                             stream[0].classes, cfg, opt=None)
    _, opt2 = train_one_task(model, method, loaders[1]["train"], torch.device("cpu"),
                             stream[1].classes, cfg, opt=opt1)
    assert opt2 is opt1                      # cùng MỘT optimizer -> ký ức M1/M2/V còn nguyên
    # state M3 có ký ức (step counter đã chạy qua cả 2 task)
    steps = [st["step"] for st in opt1.state.values() if "step" in st]
    assert steps and max(steps) > 5


def test_run_continual_with_persist_flag_runs_clean():
    stream, loaders, model = _setup()
    method = build_method("finetune", {})
    cfg = {"epochs_per_task": 1, "lr": 1e-3, "optimizer": "m3", "optimizer_per_task": False}
    R, _ = run_continual(model, method, stream, loaders, torch.device("cpu"), cfg, verbose=False)
    assert R.shape == (2, 2) and np.isfinite(R).all()
