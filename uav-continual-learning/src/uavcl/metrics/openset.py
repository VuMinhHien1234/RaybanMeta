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
# ↳ GIẢI THÍCH TỔNG QUAN: "Open-set" = bài toán nhận biết mẫu LẠ (class chưa từng học).
#   Model cho ra 1 "điểm tin cậy"; ta cần 1 NGƯỠNG: trên ngưỡng = "quen", dưới = "lạ".
#   File này đo model đặt ngưỡng tốt tới đâu bằng các chỉ số chuẩn (ROC/AUC/EER/TAR@FAR).
#   Thuật ngữ: TAR = tỉ lệ nhận đúng mẫu quen; FAR = tỉ lệ nhầm mẫu lạ thành quen.
from __future__ import annotations

import numpy as np


def roc_points(genuine, impostor, n: int = 300):
    """Trả (far, tar, thresholds), threshold quét từ cao xuống thấp."""
    # ↳ Quét n ngưỡng từ cao xuống thấp; mỗi ngưỡng tính được 1 cặp (FAR, TAR) -> vẽ đường ROC.
    genuine = np.asarray(genuine, dtype=float)   # ↳ Điểm của mẫu QUEN (mong cao).
    impostor = np.asarray(impostor, dtype=float)  # ↳ Điểm của mẫu LẠ (mong thấp).
    lo = min(genuine.min(), impostor.min())      # ↳ Ngưỡng thấp nhất cần quét.
    hi = max(genuine.max(), impostor.max())      # ↳ Ngưỡng cao nhất cần quét.
    thr = np.linspace(hi, lo, n)                 # ↳ Dãy n ngưỡng đều nhau từ cao -> thấp.
    tar = np.array([(genuine >= t).mean() for t in thr])   # ↳ Tại mỗi ngưỡng: % mẫu quen được nhận đúng.
    far = np.array([(impostor >= t).mean() for t in thr])  # ↳ Tại mỗi ngưỡng: % mẫu lạ bị nhầm thành quen.
    return far, tar, thr


def auc(far: np.ndarray, tar: np.ndarray) -> float:
    # ↳ AUC = diện tích dưới đường ROC. 1.0 = phân biệt hoàn hảo, 0.5 = đoán mò.
    order = np.argsort(far)                       # ↳ Sắp theo FAR tăng dần để tích phân đúng.
    integrate = getattr(np, "trapezoid", None)  # numpy >= 2.0  ↳ Hàm tích phân hình thang.
    if integrate is None:
        integrate = np.trapz  # numpy 1.x (trapz bị xoá ở 2.0)  ↳ Tên cũ cho numpy bản thấp.
    return float(integrate(tar[order], far[order]))  # ↳ Tích phân TAR theo FAR.


def eer(genuine, impostor) -> tuple[float, float]:
    """Trả (eer, threshold) — nơi FAR xấp xỉ FRR."""
    # ↳ EER = mức lỗi khi 2 loại lỗi (nhận nhầm lạ / bỏ sót quen) BẰNG nhau. Thấp = tốt.
    far, tar, thr = roc_points(genuine, impostor)
    frr = 1 - tar                                 # ↳ FRR = tỉ lệ bỏ sót mẫu quen = 1 − TAR.
    i = int(np.argmin(np.abs(far - frr)))         # ↳ Tìm ngưỡng nơi FAR gần bằng FRR nhất.
    return float((far[i] + frr[i]) / 2), float(thr[i])  # ↳ EER ≈ trung bình 2 lỗi tại đó.


def tar_at_far(genuine, impostor, far_target: float) -> tuple[float, float]:
    """TAR lớn nhất sao cho FAR <= far_target. Trả (tar, threshold)."""
    # ↳ Trả lời: "nếu chỉ chấp nhận sai tối đa far_target, thì nhận đúng được bao nhiêu?".
    far, tar, thr = roc_points(genuine, impostor)
    ok = far <= far_target                        # ↳ Mặt nạ các ngưỡng thoả ràng buộc FAR.
    if not ok.any():
        return 0.0, float(thr[0])                 # ↳ Không ngưỡng nào đạt -> trả 0.
    i = int(np.argmax(tar[ok]))                   # ↳ Trong số hợp lệ, lấy TAR cao nhất.
    idx = np.where(ok)[0][i]                       # ↳ Đổi về chỉ số gốc để lấy ngưỡng tương ứng.
    return float(tar[idx]), float(thr[idx])


def open_set_summary(genuine, impostor) -> dict:
    # ↳ Gom mọi chỉ số open-set vào 1 dict cho tiện in/báo cáo.
    far, tar, _ = roc_points(genuine, impostor)
    e, e_thr = eer(genuine, impostor)
    tar1, thr1 = tar_at_far(genuine, impostor, 0.01)   # ↳ TAR khi chỉ cho phép 1% nhầm.
    tar10, thr10 = tar_at_far(genuine, impostor, 0.10) # ↳ TAR khi cho phép 10% nhầm.
    return {
        "auc": auc(far, tar),
        "eer": e,
        "eer_threshold": e_thr,
        "tar@far=1%": tar1,
        "threshold@far=1%": thr1,
        "tar@far=10%": tar10,
        "threshold@far=10%": thr10,
    }
