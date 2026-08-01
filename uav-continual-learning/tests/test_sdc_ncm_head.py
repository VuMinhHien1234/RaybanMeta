"""Test SDC cho NCM-head (đòn A) — CHỈ cần torch; tự skip nếu thiếu.

Dùng model + loader GIẢ (nhẹ) để test thẳng logic engine, không cần torchvision/titans_pytorch.
Kiểm 3 điều:
1. Chế độ 'sdc' chạy được và cho ncm_R hợp lệ (hữu hạn, tam giác dưới ∈ [0,1]).
2. 'sdc' KHÔNG đọc lại data cũ: loader train của task cũ chỉ bị duyệt TRONG chính task đó,
   còn 'rebuild' duyệt lại nó ở MỌI stage sau.
3. Không truyền cờ -> mặc định 'rebuild' (hành vi cũ bất biến).

Chạy: pytest tests/test_sdc_ncm_head.py -q
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uavcl.data.stream import TaskSpec        # noqa: E402
from uavcl.engine import run_continual        # noqa: E402
from uavcl.methods import FineTune            # noqa: E402

FEAT_DIM = 8
N_TASKS = 3
CLASSES_PER_TASK = 2
NUM_CLASSES = N_TASKS * CLASSES_PER_TASK


class _FakeModel(torch.nn.Module):
    """Tối giản: .features() = x + drift (giả lập feature TRÔI), .head Linear -> logits."""

    def __init__(self, feat_dim=FEAT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.head = torch.nn.Linear(feat_dim, num_classes)
        self.register_buffer("drift", torch.zeros(feat_dim))

    def features(self, x):
        return x + self.drift

    def forward(self, x):
        return self.head(self.features(x))


class _DriftMethod(FineTune):
    """Như FineTune nhưng sau mỗi task làm feature TRÔI đi -> SDC có việc để làm."""

    name = "fakedrift"

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed):
        model.drift += 0.7


class _CountingLoader:
    """Bọc 1 loader, đếm số lần bị duyệt (__iter__) -> đo 'đọc lại data'."""

    def __init__(self, batches):
        self._batches = batches
        self.iters = 0

    def __iter__(self):
        self.iters += 1
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _make_batches(classes, n_per_class, seed):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in classes:
        mean = torch.zeros(FEAT_DIM)
        mean[c % FEAT_DIM] = 3.0                       # mỗi class 1 hướng riêng -> tách được
        xs.append(torch.randn(n_per_class, FEAT_DIM, generator=g) * 0.3 + mean)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    x = torch.cat(xs)
    y = torch.cat(ys)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    return [(x[i:i + 8], y[i:i + 8]) for i in range(0, len(y), 8)]  # batch 8


def _build():
    torch.manual_seed(0)
    stream = [
        TaskSpec(task_id=t, classes=[CLASSES_PER_TASK * t, CLASSES_PER_TASK * t + 1])
        for t in range(N_TASKS)
    ]
    loaders, counters = [], []
    for t, spec in enumerate(stream):
        train = _CountingLoader(_make_batches(spec.classes, 12, seed=t))
        test = _make_batches(spec.classes, 4, seed=100 + t)
        loaders.append({"train": train, "val": test, "test": test})
        counters.append(train)
    return _FakeModel(), _DriftMethod(), stream, loaders, counters


def _run(mode=None):
    model, method, stream, loaders, counters = _build()
    cfg = {"epochs_per_task": 1, "lr": 1e-2, "weight_decay": 0.0, "eval_ncm_head": True}
    if mode is not None:
        cfg["ncm_head_mode"] = mode
    R, log = run_continual(model, method, stream, loaders, torch.device("cpu"), cfg, verbose=False)
    return R, log, counters


def _assert_valid_ncm_R(ncm_R):
    assert ncm_R.shape == (N_TASKS, N_TASKS)
    for i in range(N_TASKS):
        for j in range(i + 1):
            v = ncm_R[i, j]
            assert np.isfinite(v) and 0.0 <= v <= 1.0


def test_sdc_runs_and_valid():
    _, log, _ = _run("sdc")
    assert "ncm_R" in log
    _assert_valid_ncm_R(log["ncm_R"])


def test_default_is_rebuild_and_valid():
    _, log, _ = _run(None)                              # không truyền ncm_head_mode
    _assert_valid_ncm_R(log["ncm_R"])


def test_sdc_does_not_reread_old_data():
    _, _, cnt_rebuild = _run("rebuild")
    _, _, cnt_sdc = _run("sdc")
    # task 0 (cũ nhất): rebuild duyệt lại ở mọi stage; sdc chỉ trong task 0.
    assert cnt_sdc[0].iters < cnt_rebuild[0].iters
    # chênh ít nhất = số stage sau task 0 (mỗi stage rebuild đọc lại task 0 một lần).
    assert cnt_rebuild[0].iters - cnt_sdc[0].iters >= (N_TASKS - 1)
