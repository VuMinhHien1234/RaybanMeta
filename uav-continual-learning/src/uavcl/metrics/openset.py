"""Metric open-set "quen/lạ" — thuần numpy (chuyển thể từ project Meta-Rayban CPM).
Dùng khi UAV gặp class CHƯA HỌC: model phải biết nói "chưa biết" thay vì đoán bừa.
Quy ước: `genuine` = điểm tin cậy trên mẫu thuộc class ĐÃ học (mong cao),
`impostor` = điểm trên mẫu thuộc class CHƯA học (mong thấp).
- roc_points        : quét ngưỡng -> (FAR, TAR)
- auc               : diện tích dưới ROC (1.0 = phân biệt hoàn hảo, 0.5 = đoán mò)
- eer               : điểm ngưỡng nơi 2 loại lỗi bằng nhau (thấp = tốt)
- tar_at_far        : TAR tốt nhất khi ép FAR <= mục tiêu (chọn ngưỡng an toàn)
- open_set_summary  : gom tất cả vào 1 dict
"""
from __future__ import annotations

import numpy as np


def roc_points(genuine, impostor, n: int = 300):
    """Trả (far, tar, thresholds), threshold quét từ cao xuống thấp."""
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    lo = min(genuine.min(), impostor.min())
    hi = max(genuine.max(), impostor.max())
    thr = np.linspace(hi, lo, n)
    tar = np.array([(genuine >= t).mean() for t in thr])
    far = np.array([(impostor >= t).mean() for t in thr])
    return far, tar, thr


def auc(far: np.ndarray, tar: np.ndarray) -> float:
    order = np.argsort(far)
    integrate = getattr(np, "trapezoid", None)  # numpy >= 2.0
    if integrate is None:
        integrate = np.trapz  # numpy 1.x (trapz bị xoá ở 2.0)
    return float(integrate(tar[order], far[order]))


def eer(genuine, impostor) -> tuple[float, float]:
    """Trả (eer, threshold) — nơi FAR xấp xỉ FRR."""
    far, tar, thr = roc_points(genuine, impostor)
    frr = 1 - tar
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thr[i])


def tar_at_far(genuine, impostor, far_target: float) -> tuple[float, float]:
    """TAR lớn nhất sao cho FAR <= far_target. Trả (tar, threshold)."""
    far, tar, thr = roc_points(genuine, impostor)
    ok = far <= far_target
    if not ok.any():
        return 0.0, float(thr[0])
    i = int(np.argmax(tar[ok]))
    idx = np.where(ok)[0][i]
    return float(tar[idx]), float(thr[idx])


def open_set_summary(genuine, impostor) -> dict:
    far, tar, _ = roc_points(genuine, impostor)
    e, e_thr = eer(genuine, impostor)
    tar1, thr1 = tar_at_far(genuine, impostor, 0.01)
    tar10, thr10 = tar_at_far(genuine, impostor, 0.10)
    return {
        "auc": auc(far, tar),
        "eer": e,
        "eer_threshold": e_thr,
        "tar@far=1%": tar1,
        "threshold@far=1%": thr1,
        "tar@far=10%": tar10,
        "threshold@far=10%": thr10,
    }
