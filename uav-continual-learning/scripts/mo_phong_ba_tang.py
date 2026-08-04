#!/usr/bin/env python3
"""MÔ PHỎNG ĐẦU-CUỐI thuần numpy — kiểm THIẾT KẾ trước khi đốt giờ GPU (2026-08-04).

Chạy được ở MỌI máy (không cần torch, không cần RESISC45). Dùng THẬT hai mảnh thuần
Python của dự án: `revisit.lich_tu_cfg` (lịch bay) và `metrics.revisit` (O2/O3); phần
ba tầng là bản numpy PHẢN CHIẾU ĐÚNG công thức trong tang_nhanh.py / ngan_hang_che_do.py.

Trả lời 3 câu hỏi thiết kế:
  1. Chuỗi dataset -> trôi tuần hoàn -> pha 2 không nhãn -> ba tầng có tạo ra tín hiệu
     U2 > U1 > U0 không?
  2. Thước đo O3 đo trên ĐƯỜNG CHÉO R (chấm CUỐI chuyến) có nhìn thấy lợi ích của ngân
     hàng chế độ không? (Nghi ngờ: KHÔNG — tầng nhanh tự hội tụ trong vài batch, nên cuối
     chuyến U1 ≈ U2; lợi ích thật nằm ở ĐẦU chuyến -> phải đo trên prequential.)
  3. O2 (thời gian hồi phục) có phân biệt được ba arm không?

Mô hình dữ liệu: feature lớp c ~ N(M_c, 0.3²I); điều kiện mức m dịch TOÀN CỤC delta·m·4
(đúng giả định T1 "trục chung" — chính là regime mà trôi tổng hợp của dự án tạo ra).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from uavcl.data.revisit import bang_lich_bay, lich_tu_cfg  # noqa: E402
from uavcl.metrics.revisit import loi_ich_quay_lai, thoi_gian_hoi_phuc  # noqa: E402

D, C, BATCH, N_BATCH = 12, 5, 16, 40          # 40 batch/chuyến — như ~56 batch của RESISC45
DECAY = 0.997                                  # cửa sổ ~333 mẫu (~21 batch): hồi phục CHẬM, thấy rõ động học
NGUONG, CHO_KHOP = 1.5, 3
rng = np.random.default_rng(0)
# Tâm lớp SÁT nhau (σ=1) + trôi LỚN (8·muc): trôi VƯỢT khoảng cách lớp — regime mà
# hiệu chỉnh D8 của dự án đo được trên RESISC45 (NCM 0,69 -> 0,53). Tâm xa quá thì mọi arm
# đều đúng 100% và mô phỏng không nói được gì (bài học từ test_slda_cov_mode).
M = rng.normal(0, 1, (C, D))                   # tâm lớp
DELTA = rng.normal(0, 1, D); DELTA /= np.linalg.norm(DELTA)


def sinh_batch(muc, n=BATCH):
    ys = rng.integers(0, C, n)
    f = M[ys] + rng.normal(0, 0.5, (n, D)) + DELTA * (8.0 * muc)
    return f, ys


class TangNhanhNp:                             # phản chiếu tang_nhanh.py (bản day_du, chỉ m)
    def __init__(self):
        self.m = None; self.m0 = None

    def cap_nhat(self, f):
        mb = f.mean(0)
        r = DECAY ** len(f)
        self.m = mb.copy() if self.m is None else r * self.m + (1 - r) * mb

    def chot_moc(self):
        self.m0 = self.m.copy()

    def can_chinh(self, f):
        return f if self.m0 is None else f - (self.m - self.m0)


class NganHangNp:                              # phản chiếu ngan_hang_che_do.py (khớp L2)
    def __init__(self):
        self.modes = []; self.khop = None; self.so_nap = 0

    def khop_va_nap(self, tn, m_tuoi):
        """Khớp bằng trung bình TƯƠI của chuyến (bản vá 2026-08-04): tại batch thứ
        cho_khop_sau, EMA tn.m còn nhiễm ~67% (λ=0,995) điều kiện chuyến TRƯỚC — khớp
        bằng nó sẽ nạp nhầm chính chế độ vừa rời khỏi."""
        self.khop = None
        if not self.modes:
            return False
        d = [np.linalg.norm(m_tuoi - mk) for mk in self.modes]
        k = int(np.argmin(d))
        if d[k] > NGUONG:
            return False
        self.khop = k; self.so_nap += 1
        tn.m = self.modes[k].copy()            # NẠP snapshot -> thích nghi tức thời
        return True

    def ghi_lai(self, tn):
        if self.khop is not None:
            self.modes[self.khop] = 0.5 * self.modes[self.khop] + 0.5 * tn.m
        else:
            self.modes.append(tn.m.copy())
        self.khop = None


def chay_arm(ten, co_tn, co_nh, lich):
    """Trả về (R_cheo, preq_theo_chuyen). Chấm NCM: argmin ||f_căn_chỉnh − μ_c||."""
    mu = np.zeros((C, D)); dem = np.zeros(C)
    tn, nh = TangNhanhNp(), (NganHangNp() if co_nh else None)
    # ---- PHA 1 (chuyến 0, CÓ nhãn, điều kiện lich[0]) ---------------------------------
    for _ in range(N_BATCH):
        f, ys = sinh_batch(lich[0].muc_troi)
        if co_tn:
            tn.cap_nhat(f)
        for fi, yi in zip(f, ys):
            mu[yi] += fi; dem[yi] += 1
    mu /= np.maximum(dem, 1)[:, None]
    if co_tn:
        tn.chot_moc()

    def du_doan(f):
        fa = tn.can_chinh(f) if co_tn else f
        return np.argmin(((fa[:, None, :] - mu[None]) ** 2).sum(-1), axis=1)

    R_cheo, preq = [], {}
    f_test0, y_test0 = sinh_batch(lich[0].muc_troi, 400)
    R_cheo.append(float((du_doan(f_test0) == y_test0).mean()))
    # ---- PHA 2 (KHÔNG nhãn): prequential -> cập nhật; cuối chuyến chấm "đường chéo" ----
    for cb in lich[1:]:
        accs = []
        m_tuoi_sum = None
        for b in range(N_BATCH):
            f, ys = sinh_batch(cb.muc_troi)
            accs.append(float((du_doan(f) == ys).mean()))   # 1) dự đoán TRƯỚC
            if co_tn:
                tn.cap_nhat(f)                              # 2) rồi mới cập nhật (không nhãn)
                if nh is not None and b < CHO_KHOP:
                    mb = f.mean(0)
                    m_tuoi_sum = mb if m_tuoi_sum is None else m_tuoi_sum + mb
                    if b + 1 == CHO_KHOP:
                        nh.khop_va_nap(tn, m_tuoi_sum / CHO_KHOP)
        if nh is not None:
            nh.ghi_lai(tn)
        preq[cb.chi_so] = accs
        f_t, y_t = sinh_batch(cb.muc_troi, 400)             # test_chung: cùng phân bố, đk chuyến
        R_cheo.append(float((du_doan(f_t) == y_t).mean()))
    return R_cheo, preq


def main():
    lich = lich_tu_cfg(12, {"che_do": "tuan_hoan", "chu_ky": 4, "n_mode": 3})
    mode_that = [c.mode_that for c in lich]
    print("Lịch bay:", bang_lich_bay(lich), "\n")
    ket_qua = {}
    for ten, tn, nh in (("U0", False, False), ("U1", True, False), ("U2", True, True)):
        R_cheo, preq = chay_arm(ten, tn, nh, lich)
        R = [[R_cheo[j] if j <= i else 0.0 for j in range(len(R_cheo))]
             for i in range(len(R_cheo))]
        o3_cheo = loi_ich_quay_lai(R, mode_that)["loi_ich_quay_lai"]
        # O3 trên prequential (chỉ các chuyến pha 2 có trace): cả chuyến và 10 batch ĐẦU
        ts = sorted(preq)
        lam_R = lambda vals: [[vals[j] if j <= i else 0.0 for j in range(len(vals))]
                              for i in range(len(vals))]
        m_full = [float(np.mean(preq[t])) for t in ts]
        m_dau = [float(np.mean(preq[t][:10])) for t in ts]
        modes_p2 = [mode_that[t] for t in ts]
        o3_preq = loi_ich_quay_lai(lam_R(m_full), modes_p2)["loi_ich_quay_lai"]
        o3_preq10 = loi_ich_quay_lai(lam_R(m_dau), modes_p2)["loi_ich_quay_lai"]
        hp = [thoi_gian_hoi_phuc(preq[t]) for t in ts
              if mode_that[t] != mode_that[t - 1]]
        hp_tb = float(np.mean([h for h in hp if h != float("inf")])) if hp else 0.0
        ket_qua[ten] = (np.mean(R_cheo[1:]), o3_cheo, o3_preq, o3_preq10, hp_tb)

    print(f"{'':4} {'acc chéo TB':>12} {'O3 chéo':>9} {'O3 preq':>9} "
          f"{'O3 preq10 ⭐':>12} {'O2 hồi phục':>12}")
    for ten, (acc, oc, op, op10, hp) in ket_qua.items():
        print(f"{ten:4} {acc:12.4f} {oc:+9.4f} {op:+9.4f} {op10:+12.4f} {hp:12.1f}")

    print("""
Đọc kết quả (kiểm tra thiết kế):
  * U1 phải vượt xa U0 về acc chéo -> tầng nhanh có tác dụng (M1 ăn).
  * O3 CHÉO của U2 ≈ U1 ≈ 0 -> ĐÚNG NHƯ NGHI NGỜ: chấm cuối chuyến bị MÙ với ngân hàng
    (tầng nhanh tự hội tụ trước khi chấm). KHÔNG dùng cột này làm bằng chứng O3.
  * O3 PREQ10 ⭐ của U2 phải DƯƠNG rõ và vượt U1 -> "gặp lại thì nhận ra ngay" hiện ra
    ở 10 batch đầu chuyến. ĐÂY là con số O3 của dự án.
  * O2 của U1/U2 hữu hạn còn U0 thì không hồi phục ở chuyến trôi nặng.""")


if __name__ == "__main__":
    main()
