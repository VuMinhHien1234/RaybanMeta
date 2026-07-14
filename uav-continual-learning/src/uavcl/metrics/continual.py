"""Continual-learning metrics (pure NumPy, no torch needed).

All metrics take an accuracy matrix ``R`` of shape ``(T, T)`` where
``R[i, j]`` = accuracy on task ``j`` after finishing training on task ``i``
(row = training stage, column = evaluated task). The diagonal and
lower-triangle are what matter; the upper-triangle (j > i) is ignored.

These are the standard GEM-style metrics (Lopez-Paz & Ranzato, 2017)
that N1 uses to compare every model against the baselines.
"""
from __future__ import annotations

import numpy as np


def average_accuracy(R) -> float:
    """Mean accuracy over all tasks after the final training stage (higher = better)."""
    R = np.asarray(R, dtype=float)
    return float(np.mean(R[-1, :]))


def backward_transfer(R) -> float:
    """BWT: mean change on old tasks caused by later training (negative = forgetting)."""
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T < 2:
        return 0.0
    diffs = [R[-1, j] - R[j, j] for j in range(T - 1)]
    return float(np.mean(diffs))


def forward_transfer(R, chance: float = 0.0) -> float:
    """FWT (GEM-style): mean over j>=1 of ``R[j-1, j] - chance``.

    ``R[j-1, j]`` = accuracy trên task j NGAY TRƯỚC khi học nó (cần engine bật
    ``train.eval_future: true`` để ô này được đo; mặc định engine chỉ điền tam
    giác dưới). Với head phân loại khởi tạo mới, FWT thường xấp xỉ mức đoán mò
    (``chance``) — metric này chủ yếu có ý nghĩa từ G2+ khi model mang bộ nhớ.
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T < 2:
        return 0.0
    vals = [R[j - 1, j] - chance for j in range(1, T)]
    return float(np.mean(vals))


def average_forgetting(R) -> float:
    """Average forgetting: mean drop from each task's best-ever accuracy to its final
    accuracy (lower = better; this is the headline number the project tries to reduce)."""
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T < 2:
        return 0.0
    forgets = []
    for j in range(T - 1):
        prev_best = np.max(R[j:T - 1, j])  # best accuracy on task j before the final stage
        forgets.append(prev_best - R[-1, j])
    return float(np.mean(forgets))
