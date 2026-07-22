"""Class-incremental stream utilities.

This module is intentionally torch-free so stream logic can be tested before the
ML stack is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import random


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    classes: list[int]
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]


def split_classes(
    num_classes: int,
    num_tasks: int,
    seed: int = 0,
    shuffle_classes: bool = True,
) -> list[list[int]]:
    """Split global class ids into deterministic, disjoint task groups."""
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if num_tasks > num_classes:
        raise ValueError("num_tasks cannot exceed num_classes")
    classes = list(range(int(num_classes)))
    if shuffle_classes:
        random.Random(seed).shuffle(classes)

    base, rem = divmod(num_classes, num_tasks)
    groups = []
    start = 0
    for t in range(num_tasks):
        n = base + (1 if t < rem else 0)
        groups.append(classes[start:start + n])
        start += n
    return groups


def indices_by_task(labels: Sequence[int], groups: Sequence[Sequence[int]]) -> list[list[int]]:
    """Return dataset indices whose labels belong to each task's class group."""
    buckets: list[list[int]] = []
    for group in groups:
        allowed = {int(c) for c in group}
        buckets.append([i for i, y in enumerate(labels) if int(y) in allowed])
    return buckets


def stratified_split(
    indices: Sequence[int],
    labels: Sequence[int],
    fraction: float,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Split indices into (large, small), preserving class proportions."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    by_class: dict[int, list[int]] = {}
    for i in indices:
        by_class.setdefault(int(labels[i]), []).append(int(i))

    rng = random.Random(seed)
    large: list[int] = []
    small: list[int] = []
    for cls in sorted(by_class):
        vals = list(by_class[cls])
        rng.shuffle(vals)
        n_small = int(round(len(vals) * fraction))
        small.extend(vals[:n_small])
        large.extend(vals[n_small:])
    return sorted(large), sorted(small)


def build_stream(
    train_labels: Sequence[int],
    val_labels: Sequence[int],
    test_labels: Sequence[int],
    num_classes: int,
    num_tasks: int,
    seed: int = 0,
    shuffle_classes: bool = True,
) -> list[TaskSpec]:
    groups = split_classes(num_classes, num_tasks, seed=seed, shuffle_classes=shuffle_classes)
    train_buckets = indices_by_task(train_labels, groups)
    val_buckets = indices_by_task(val_labels, groups)
    test_buckets = indices_by_task(test_labels, groups)
    return [
        TaskSpec(
            task_id=t,
            classes=[int(c) for c in groups[t]],
            train_idx=train_buckets[t],
            val_idx=val_buckets[t],
            test_idx=test_buckets[t],
        )
        for t in range(num_tasks)
    ]


def describe_stream(stream: Sequence[TaskSpec], class_names: Sequence[str] | None = None) -> str:
    lines = ["== Task stream =="]
    for spec in stream:
        if class_names:
            names = ", ".join(str(class_names[c]) for c in spec.classes)
        else:
            names = ", ".join(str(c) for c in spec.classes)
        lines.append(
            f"  task {spec.task_id}: [{names}] "
            f"train={len(spec.train_idx)} val={len(spec.val_idx)} test={len(spec.test_idx)}"
        )
    return "\n".join(lines)
