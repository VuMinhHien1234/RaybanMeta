"""Test #27 (open-set) + #24 (AAA). CHỈ cần torch/numpy; tự skip nếu thiếu torch.

Kiểm:
1. build_stream_with_holdout: đúng số class giữ lại, không rò vào task nào, validate tham số.
2. collect_openset_scores trên cụm Gaussian: quen gần prototype, lạ ở xa -> AUC > 0.95.
3. average_anytime_accuracy: khớp tính tay + phạt model "sập giữa hành trình".

Chạy: pytest tests/test_openset_eval.py -q
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uavcl.data.stream import build_stream_with_holdout   # noqa: E402
from uavcl.metrics import average_anytime_accuracy, open_set_summary  # noqa: E402
from uavcl.openset_eval import collect_openset_scores     # noqa: E402

FEAT = 8


class _FeatModel(torch.nn.Module):
    """Model giả: .features(x) = x (đủ cho scorer prototype)."""

    def features(self, x):
        return x

    def forward(self, x):
        return x


def _cluster(label, n=30, seed=0, spread=0.2):
    g = torch.Generator().manual_seed(seed + label)
    center = torch.zeros(FEAT)
    center[label % FEAT] = 4.0
    return torch.randn(n, FEAT, generator=g) * spread + center


def test_holdout_split_no_leak():
    labels = [c for c in range(10) for _ in range(4)]
    stream, held = build_stream_with_holdout(labels, labels, labels,
                                             num_classes=10, num_tasks=4, seed=0, holdout=2)
    assert len(held) == 2
    trained = {c for s in stream for c in s.classes}
    assert trained.isdisjoint(held)                    # class giữ lại KHÔNG xuất hiện trong task nào
    assert len(trained) == 8
    with pytest.raises(ValueError):
        build_stream_with_holdout(labels, labels, labels, num_classes=10, num_tasks=4, holdout=0)


def test_openset_auc_on_gaussians():
    model = _FeatModel()
    dev = torch.device("cpu")
    # 2 task × 1 class đã học (class 0, 1); class lạ = 5 (tâm ở chiều khác, xa prototype)
    task_loaders = []
    for c in (0, 1):
        x = _cluster(c)
        y = torch.full((len(x),), c, dtype=torch.long)
        task_loaders.append({"train": [(x, y)], "test": [(_cluster(c, seed=50), y)]})
    unseen = [(_cluster(5, seed=99), torch.full((30,), 5, dtype=torch.long))]
    genuine, impostor = collect_openset_scores(model, task_loaders, unseen, num_classes=6, device=dev)
    s = open_set_summary(genuine, impostor)
    assert s["auc"] > 0.95                             # quen/lạ phải tách rõ trên bài dễ
    assert np.mean(genuine) > np.mean(impostor)


def test_average_anytime_accuracy():
    # 3 mốc: [1.0] ; [0.5, 1.0] ; [0.9, 0.9, 0.9] -> AAA = mean(1.0, 0.75, 0.9)
    R = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.9, 0.9, 0.9]])
    assert abs(average_anytime_accuracy(R) - np.mean([1.0, 0.75, 0.9])) < 1e-9
    # model "sập giữa đường" phải bị AAA phạt so với model giữ vững, dù điểm cuối cao:
    R_dip = np.array([[1.0, 0.0], [0.2, 1.0]])         # sau task 1: task 0 sập còn 0.2
    R_keep = np.array([[1.0, 0.0], [1.0, 1.0]])        # giữ vững suốt hành trình
    assert average_anytime_accuracy(R_dip) < average_anytime_accuracy(R_keep)
