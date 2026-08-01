"""Continual-learning metrics (pure NumPy, no torch needed).

All metrics take an accuracy matrix ``R`` of shape ``(T, T)`` where
``R[i, j]`` = accuracy on task ``j`` after finishing training on task ``i``
(row = training stage, column = evaluated task). The diagonal and
lower-triangle are what matter; the upper-triangle (j > i) is ignored.

These are the standard GEM-style metrics (Lopez-Paz & Ranzato, 2017)
that N1 uses to compare every model against the baselines.
"""
# ↳ GIẢI THÍCH TỔNG QUAN: File này biến ma trận kết quả R (do engine trả về) thành các
#   CON SỐ TÓM TẮT để so sánh phương pháp. Nhắc lại R[i][j] = độ chính xác task j SAU
#   khi học xong task i. 4 chỉ số:
#     - average_accuracy: cuối cùng trung bình đúng bao nhiêu (càng CAO càng tốt).
#     - backward_transfer (BWT): học cái mới làm task cũ tốt lên hay tệ đi (ÂM = quên).
#     - forward_transfer (FWT): kiến thức cũ có giúp task mới CHƯA học không.
#     - average_forgetting: tụt bao nhiêu so với lúc đỉnh (càng THẤP càng tốt — số chính của dự án).
from __future__ import annotations

import numpy as np


def average_accuracy(R) -> float:
    """Mean accuracy over all tasks after the final training stage (higher = better)."""
    R = np.asarray(R, dtype=float)
    return float(np.mean(R[-1, :]))  # ↳ Lấy HÀNG CUỐI (sau khi học hết) rồi trung bình mọi task.


def backward_transfer(R) -> float:
    """BWT: mean change on old tasks caused by later training (negative = forgetting)."""
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T < 2:
        return 0.0  # ↳ Chỉ 1 task -> không có "cũ" để đo.
    diffs = [R[-1, j] - R[j, j] for j in range(T - 1)]
    # ↳ Với mỗi task cũ j: (điểm cuối cùng) − (điểm ngay khi vừa học xong j). Âm = tệ đi = quên.
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
    # ↳ Điểm trên task j TRƯỚC khi học nó, trừ mức đoán mò. Dương = kiến thức cũ giúp ích trước.
    return float(np.mean(vals))


def average_anytime_accuracy(R) -> float:
    """#24 — AAA (Average Anytime Accuracy) cho regime STREAMING/UAV.

    ``mean over t of mean_{j<=t} R[t, j]``: sau MỖI task, đo trung bình acc trên mọi
    task đã thấy, rồi trung bình qua các mốc. Khác average_accuracy (chỉ nhìn HÀNG CUỐI):
    UAV dùng model LIÊN TỤC trong lúc học, không chỉ lúc "học xong hết" — model sập ở
    giữa hành trình rồi hồi lại cuối kỳ vẫn bị AAA phạt, dù acc cuối đẹp."""
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    stage_means = [float(np.mean(R[t, : t + 1])) for t in range(T)]  # ↳ Trung bình phần đã thấy ở mỗi mốc t.
    return float(np.mean(stage_means))


def average_forgetting(R) -> float:
    """Average forgetting: mean drop from each task's best-ever accuracy to its final
    accuracy (lower = better; this is the headline number the project tries to reduce)."""
    # ↳ ĐÂY LÀ SỐ CHÍNH của dự án: trung bình mức TỤT từ đỉnh xuống cuối của mỗi task cũ.
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T < 2:
        return 0.0
    forgets = []
    for j in range(T - 1):
        prev_best = np.max(R[j:T - 1, j])  # best accuracy on task j before the final stage
        # ↳ Điểm CAO NHẤT từng đạt trên task j (trước giai đoạn cuối).
        forgets.append(prev_best - R[-1, j])  # ↳ Đỉnh − điểm cuối = đã quên bao nhiêu.
    return float(np.mean(forgets))
