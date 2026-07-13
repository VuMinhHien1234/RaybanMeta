"""Test baseline NCM (tự skip nếu chưa có torch): kỳ vọng cốt lõi = KHÔNG quên."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from uavcl.data import build_stream, get_source           # noqa: E402
from uavcl.data.loaders import build_task_loaders          # noqa: E402
from uavcl.engine import run_continual                     # noqa: E402
from uavcl.methods import build_method                     # noqa: E402
from uavcl.metrics import average_forgetting               # noqa: E402
from uavcl.models import NCMClassifier, build_backbone     # noqa: E402

DATA_CFG = {
    "name": "synthetic", "num_classes": 4, "train_per_class": 16,
    "val_per_class": 4, "test_per_class": 4, "split_seed": 0,
    "num_tasks": 2, "image_size": 32, "batch_size": 16, "num_workers": 0,
}


def test_ncm_no_forgetting_and_bounded():
    torch.manual_seed(0)
    source = get_source(DATA_CFG)
    stream = build_stream(
        source.splits["train"].labels, source.splits["val"].labels,
        source.splits["test"].labels, num_classes=source.num_classes, num_tasks=2, seed=0,
    )
    loaders = build_task_loaders(source, stream, DATA_CFG)
    backbone, dim = build_backbone({"name": "tinycnn"})
    model = NCMClassifier(backbone, dim, source.num_classes)
    method = build_method("ncm", {})

    # backbone bị đóng băng thật
    assert all(not p.requires_grad for p in model.backbone.parameters())

    R, _ = run_continual(model, method, stream, loaders, torch.device("cpu"),
                         {"epochs_per_task": 1}, verbose=False)
    assert R.shape == (2, 2)
    # prototype class cũ không bị ghi đè -> quên phải ~0 trên dữ liệu giả tách màu
    assert average_forgetting(R) <= 0.05
    assert R[1, 0] >= R[0, 0] - 0.05

    # bộ nhớ thêm bị chặn: C x D (+ C bộ đếm)
    expected = source.num_classes * dim + source.num_classes
    assert method.footprint_floats(model) == expected
    assert np.isfinite(R).all()
