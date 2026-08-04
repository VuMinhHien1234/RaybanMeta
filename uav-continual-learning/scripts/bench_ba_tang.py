#!/usr/bin/env python3
"""P3 (KE_HOACH_SUA 2026-08-04) — đo O4 cho hệ BA TẦNG: MB thêm + ms mỗi khung hình.

Ràng buộc bài toán (BAI_TOAN §4): tổng bộ nhớ thêm ≤ 10 MB · độ trễ thêm ≤ 5% ngân sách
khung hình 30 fps (= 1,67 ms). Script này đo phần BA TẦNG cộng thêm SO VỚI SLDA trần
(backbone không tính — nó là chi phí chung của mọi phương án, xem B6).

Dùng:
    python scripts/bench_ba_tang.py                 # D=384 (ViT-S), C=45 (RESISC45)
    python scripts/bench_ba_tang.py --dim 1024 --num-classes 45 --k 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from uavcl.models.ngan_hang_che_do import NganHangCheDo  # noqa: E402
from uavcl.models.slda import SLDAClassifier  # noqa: E402
from uavcl.models.tang_nhanh import TangNhanh  # noqa: E402

NGAN_SACH_MS = 1000.0 / 30.0          # 33,33 ms mỗi khung hình
TRAN_TRE_MS = 0.05 * NGAN_SACH_MS     # ≤ 5% ngân sách
TRAN_BO_NHO_MB = 10.0


class _FakeBackbone(torch.nn.Module):
    def forward(self, x):
        return x


def _do_ms(fn, n=2000, warmup=200):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--num-classes", type=int, default=45)
    ap.add_argument("--k", type=int, default=8, help="số chế độ trong ngân hàng khi đo")
    a = ap.parse_args()
    D, C = a.dim, a.num_classes
    torch.manual_seed(0)

    # --- dựng hệ ba tầng, nạp K chế độ vào ngân hàng cho đúng chi phí lúc triển khai ----
    m = SLDAClassifier(_FakeBackbone(), D, C,
                       tang_nhanh={"enabled": True, "decay": 0.99},
                       ngan_hang={"enabled": True, "nguong": 0.5, "k_max": a.k})
    f, ys = torch.randn(512, D), torch.randint(0, C, (512,))
    m.update(f, ys)
    m.chot_moc_pha1()
    for k in range(a.k):
        m.tang_nhanh.m_t = m.tang_nhanh.m0 + torch.randn(D) * (k + 1)
        m.tang_nhanh.v_t = torch.ones(D)
        m.ngan_hang.ghi_lai(m.tang_nhanh)

    # --- bộ nhớ ------------------------------------------------------------------------
    b_tn = m.tang_nhanh.extra_bytes()
    b_nh = m.ngan_hang.extra_bytes()
    tong_mb = (b_tn + b_nh) / 1e6

    # --- độ trễ mỗi KHUNG HÌNH (batch=1, đúng chế độ bay) -------------------------------
    x1 = torch.randn(1, D)
    ms_can_chinh = _do_ms(lambda: m.tang_nhanh.can_chinh(x1))
    ms_cap_nhat = _do_ms(lambda: m.tang_nhanh.cap_nhat(x1))
    ms_khop = _do_ms(lambda: m.ngan_hang._gan_nhat(m.tang_nhanh.m_t, m.tang_nhanh.truc),
                     n=500)
    ms_moi_khung = ms_can_chinh + ms_cap_nhat      # khớp ngân hàng chỉ 1 lần/chuyến, không tính/khung

    print(f"\n== BENCH BA TẦNG (D={D}, C={C}, K={a.k}) — đối chiếu O4")
    print(f"  bộ nhớ tầng nhanh : {b_tn / 1024:8.1f} KB")
    print(f"  bộ nhớ tầng trung : {b_nh / 1024:8.1f} KB  ({a.k} chế độ)")
    print(f"  TỔNG thêm         : {tong_mb:8.4f} MB   (trần {TRAN_BO_NHO_MB} MB)"
          f"   {'✅ ĐẠT' if tong_mb <= TRAN_BO_NHO_MB else '❌ VƯỢT'}")
    print(f"  căn chỉnh/khung   : {ms_can_chinh:8.4f} ms")
    print(f"  cập nhật m_t/khung: {ms_cap_nhat:8.4f} ms")
    print(f"  khớp ngân hàng    : {ms_khop:8.4f} ms   (1 lần/chuyến — ngoài đường nóng)")
    print(f"  TỔNG thêm/khung   : {ms_moi_khung:8.4f} ms  (trần {TRAN_TRE_MS:.2f} ms = 5% của "
          f"{NGAN_SACH_MS:.1f} ms)   {'✅ ĐẠT' if ms_moi_khung <= TRAN_TRE_MS else '❌ VƯỢT'}")
    print("  (Tham chiếu B6: SLDA trần 0,0071 ms · Titans self-mod depth-3 2,1782 ms / 102,68 MB)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
