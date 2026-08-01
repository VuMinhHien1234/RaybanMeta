"""Test #23 — checkpoint/resume của run_continual. CHỈ cần torch; tự skip nếu thiếu.

Kiểm 3 điều:
1. Chạy liền 3 task = chạy [task 0 -> dừng] rồi RESUME [task 1-2]: ma trận R GIỐNG HỆT.
2. Checkpoint khôi phục cả method (buffer latent replay sống lại sau resume).
3. Mặc định TẮT: không có checkpoint_path -> không tạo file nào.

Chạy: pytest tests/test_checkpoint.py -q
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uavcl.data.stream import TaskSpec                 # noqa: E402
from uavcl.engine import run_continual                 # noqa: E402
from uavcl.methods import build_method                 # noqa: E402
from uavcl.models.classifier import ContinualClassifier  # noqa: E402

FEAT = 8
C = 6
N_TASKS = 3


def _cluster(label, n=40, seed=0):
    g = torch.Generator().manual_seed(seed * 100 + label)
    center = torch.zeros(FEAT)
    center[label % FEAT] = 4.0
    return torch.randn(n, FEAT, generator=g) * 0.3 + center, torch.full((n,), label, dtype=torch.long)


def _loaders():
    """3 task × 2 class; loader = list batch (deterministic, không worker)."""
    stream, loaders = [], []
    for t in range(N_TASKS):
        cls = [2 * t, 2 * t + 1]
        xs, ys = zip(*[_cluster(c) for c in cls])
        x, y = torch.cat(xs), torch.cat(ys)
        batches = [(x[i:i + 16], y[i:i + 16]) for i in range(0, len(y), 16)]
        stream.append(TaskSpec(task_id=t, classes=cls, train_idx=list(range(len(y))),
                               val_idx=[0], test_idx=list(range(len(y)))))
        loaders.append({"train": batches, "val": batches[:1], "test": batches})
    return stream, loaders


def _cfg(extra=None):
    cfg = {"epochs_per_task": 2, "lr": 0.05, "optimizer": "adamw", "optimizer_per_task": True}
    cfg.update(extra or {})
    return cfg


def _fresh():
    torch.manual_seed(123)
    model = ContinualClassifier(torch.nn.Identity(), FEAT, C, head="linear")
    method = build_method("latent_replay", {"latent_replay": {"buffer_per_class": 3, "seed": 0}})
    return model, method


def test_resume_matches_straight_run(tmp_path):
    stream, loaders = _loaders()
    dev = torch.device("cpu")
    ck = str(tmp_path / "ck.pt")

    model_a, method_a = _fresh()
    R_full, _ = run_continual(model_a, method_a, stream, loaders, dev, _cfg(), verbose=False)

    model_b, method_b = _fresh()
    run_continual(model_b, method_b, stream, loaders, dev,
                  _cfg({"checkpoint_path": ck, "stop_after_task": 0}), verbose=False)  # "đứt" sau task 0
    model_c, method_c = _fresh()                       # tiến trình MỚI tinh, chỉ có file checkpoint
    R_res, _ = run_continual(model_c, method_c, stream, loaders, dev,
                             _cfg({"checkpoint_path": ck, "resume": True}), verbose=False)

    assert np.allclose(R_full, R_res, atol=1e-6), f"\nfull=\n{R_full}\nresume=\n{R_res}"


def test_resume_restores_method_buffer(tmp_path):
    stream, loaders = _loaders()
    dev = torch.device("cpu")
    ck = str(tmp_path / "ck.pt")
    model_b, method_b = _fresh()
    run_continual(model_b, method_b, stream, loaders, dev,
                  _cfg({"checkpoint_path": ck, "stop_after_task": 0}), verbose=False)
    from uavcl.checkpoint import load_run_checkpoint
    pay = load_run_checkpoint(ck)
    assert pay["task_done"] == 0
    assert len(pay["method"]._buf) == 2 * 3            # 2 class task 0 × buffer_per_class 3


def test_default_off_no_file(tmp_path):
    stream, loaders = _loaders()
    model, method = _fresh()
    run_continual(model, method, stream, loaders, torch.device("cpu"), _cfg(), verbose=False)
    assert list(tmp_path.iterdir()) == []              # không cờ -> không file nào được tạo
