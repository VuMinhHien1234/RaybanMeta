"""Test metric open-set — thuần numpy, chạy được khi chưa cài torch."""
import numpy as np

from uavcl.metrics import auc, eer, open_set_summary, roc_points, tar_at_far


def test_perfect_separation():
    genuine = np.array([0.9, 0.95, 0.85, 0.99])
    impostor = np.array([0.1, 0.2, 0.05, 0.15])
    far, tar, _ = roc_points(genuine, impostor)
    assert auc(far, tar) >= 0.99
    e, _ = eer(genuine, impostor)
    assert e <= 0.01
    tar1, thr = tar_at_far(genuine, impostor, 0.01)
    assert tar1 == 1.0
    assert 0.2 < thr < 0.85  # ngưỡng nằm giữa 2 cụm


def test_random_scores_auc_near_half():
    rng = np.random.default_rng(0)
    genuine = rng.uniform(0, 1, 2000)
    impostor = rng.uniform(0, 1, 2000)
    far, tar, _ = roc_points(genuine, impostor)
    assert 0.45 < auc(far, tar) < 0.55
    e, _ = eer(genuine, impostor)
    assert 0.4 < e < 0.6


def test_summary_keys_and_monotonic_tar():
    rng = np.random.default_rng(1)
    genuine = rng.normal(0.7, 0.1, 500)
    impostor = rng.normal(0.3, 0.1, 500)
    s = open_set_summary(genuine, impostor)
    for k in ("auc", "eer", "tar@far=1%", "tar@far=10%"):
        assert k in s
    # nới FAR thì TAR không được giảm
    assert s["tar@far=10%"] >= s["tar@far=1%"]
    assert s["auc"] > 0.9
