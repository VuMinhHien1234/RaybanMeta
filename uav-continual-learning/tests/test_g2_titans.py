"""Test G2 end-to-end nhỏ (cần torch + titans-pytorch; tự skip nếu thiếu).

Chạy trên máy dev: pytest tests/test_g2_titans.py -q
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("titans_pytorch")

from uavcl.data import build_stream, get_source            # noqa: E402
from uavcl.data.loaders import build_task_loaders           # noqa: E402
from uavcl.engine import run_continual                      # noqa: E402
from uavcl.methods import build_method                      # noqa: E402
from uavcl.models import TitansClassifier, build_backbone   # noqa: E402
from uavcl.models.memory import TitansMemory                # noqa: E402

DATA_CFG = {
    "name": "synthetic", "num_classes": 4, "train_per_class": 16,
    "val_per_class": 4, "test_per_class": 4, "split_seed": 0,
    "num_tasks": 2, "image_size": 32, "batch_size": 8, "num_workers": 0,
}
TRAIN_CFG = {"epochs_per_task": 1, "lr": 3e-3, "weight_decay": 0.0}
MEM = {"enabled": True, "dim": "auto", "chunk_size": 4, "seq": "image_seq"}


def test_memory_forward_and_state_advances():
    torch.manual_seed(0)
    m = TitansMemory(dim=16, chunk_size=4)
    seq = torch.randn(1, 8, 16)
    out, st = m(seq, state=None)
    assert out.shape == (1, 8, 16)
    assert st is not None                      # state được trả về
    out2, st2 = m(seq, state=st)               # nối tiếp state
    assert out2.shape == (1, 8, 16)


def _run(reset_mode):
    torch.manual_seed(0)
    source = get_source(DATA_CFG)
    stream = build_stream(
        source.splits["train"].labels, source.splits["val"].labels,
        source.splits["test"].labels, num_classes=source.num_classes, num_tasks=2, seed=0,
    )
    loaders = build_task_loaders(source, stream, DATA_CFG)
    backbone, dim = build_backbone({"name": "tinycnn"})
    model = TitansClassifier(backbone, dim, source.num_classes, {**MEM, "reset": reset_mode})
    method = build_method("titans", {})
    R, _ = run_continual(model, method, stream, loaders, torch.device("cpu"), TRAIN_CFG, verbose=False)
    return R, model, method


def test_reset_image_keeps_no_state():
    R, model, _ = _run("image")
    assert R.shape == (2, 2) and np.isfinite(R).all()
    assert model._state is None                # mode image: không giữ ký ức
    assert model.state_norm() == 0.0


def test_state_is_graph_free_after_forward():
    """Chẩn đoán thẳng bug double-backward: state lưu lại phải ĐỨT HẲN graph
    (mọi tensor: grad_fn is None) — kể cả tensor nằm trong tensordict.TensorDict."""
    from uavcl.models.state_utils import _tree_tensors

    torch.manual_seed(0)
    backbone, dim = build_backbone({"name": "tinycnn"})
    model = TitansClassifier(backbone, dim, 4, {**MEM, "reset": "never"})
    model.train()
    model(torch.randn(4, 3, 32, 32))
    assert model._state is not None
    leaked = [t.shape for t in _tree_tensors(model._state) if t.grad_fn is not None]
    assert not leaked, f"state còn dính graph ở tensor: {leaked}"


def test_reset_never_carries_state_across_tasks():
    R, model, method = _run("never")
    assert np.isfinite(R).all()
    assert model._state is not None            # ký ức sống sau cả stream
    assert model.state_norm() > 0.0
    st = model.export_state()                  # S10: lưu được
    assert st is not None
    assert method.footprint_floats(model) > 0
    # backbone thật sự đóng băng
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_eval_is_deterministic_and_does_not_touch_state():
    from uavcl.engine import evaluate
    R, model, _ = _run("never")
    norm_before = model.state_norm()
    source = get_source(DATA_CFG)
    stream = build_stream(
        source.splits["train"].labels, source.splits["val"].labels,
        source.splits["test"].labels, num_classes=source.num_classes, num_tasks=2, seed=0,
    )
    loaders = build_task_loaders(source, stream, DATA_CFG)
    a1 = evaluate(model, loaders[0]["test"], torch.device("cpu"), [0, 1, 2, 3])
    a2 = evaluate(model, loaders[0]["test"], torch.device("cpu"), [0, 1, 2, 3])
    assert a1 == a2                            # chấm 2 lần -> y hệt
    assert model.state_norm() == pytest.approx(norm_before)  # thi không ghi trí nhớ
