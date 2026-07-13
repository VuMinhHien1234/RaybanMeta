"""Test logic chia task — thuần Python, chạy được cả khi CHƯA cài torch."""
import pytest

from uavcl.data.stream import (
    build_stream,
    indices_by_task,
    split_classes,
    stratified_split,
)


def test_split_classes_cover_disjoint():
    groups = split_classes(45, 9, seed=0)
    assert len(groups) == 9
    flat = [c for g in groups for c in g]
    assert sorted(flat) == list(range(45))          # phủ hết, không trùng
    assert all(len(g) == 5 for g in groups)          # 45/9 chia đều


def test_split_classes_deterministic_and_seed_sensitive():
    a = split_classes(10, 5, seed=0)
    b = split_classes(10, 5, seed=0)
    c = split_classes(10, 5, seed=1)
    assert a == b
    assert a != c


def test_split_classes_uneven():
    groups = split_classes(10, 3, seed=0)
    assert sorted(len(g) for g in groups) == [3, 3, 4]


def test_split_classes_invalid():
    with pytest.raises(ValueError):
        split_classes(5, 6)


def test_indices_by_task():
    labels = [0, 1, 2, 0, 1, 2, 3]
    groups = [[0, 1], [2, 3]]
    buckets = indices_by_task(labels, groups)
    assert buckets[0] == [0, 1, 3, 4]
    assert buckets[1] == [2, 5, 6]


def test_stratified_split_ratio_and_disjoint():
    labels = [0] * 50 + [1] * 50
    idx = list(range(100))
    big, small = stratified_split(idx, labels, fraction=0.2, seed=0)
    assert len(small) == 20 and len(big) == 80
    assert set(big) | set(small) == set(idx)
    assert set(big) & set(small) == set()
    # giữ tỉ lệ theo class
    assert sum(1 for i in small if labels[i] == 0) == 10


def test_build_stream_end_to_end():
    train = [c for c in range(6) for _ in range(10)]
    val = [c for c in range(6) for _ in range(3)]
    test = [c for c in range(6) for _ in range(3)]
    stream = build_stream(train, val, test, num_classes=6, num_tasks=3, seed=0)
    assert [s.task_id for s in stream] == [0, 1, 2]
    for s in stream:
        assert len(s.classes) == 2
        assert len(s.train_idx) == 20 and len(s.test_idx) == 6
        # chỉ số đúng class của task
        assert all(train[i] in s.classes for i in s.train_idx)
