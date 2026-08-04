#!/usr/bin/env python3
"""Quét kết quả track chính U0/U1/U2 -> bảng + CON SỐ CÔNG BỐ. Chỉ dùng thư viện chuẩn.

Chạy được cả trên VM lẫn trên Mac (không cần venv, không cần torch):

    python3 scripts/tom_tat_ba_tang.py                 # bảng đầy đủ
    python3 scripts/tom_tat_ba_tang.py --cua-chan      # thêm: U1 có hơn U0 không (exit 3 nếu không)

Luật công bố (DA_TRIEN_KHAI_SUA 2026-08-04, mục audit):
  O3 = o3_preq10_loi_ich(U2) − o3_preq10_loi_ich(U1), GHÉP CẶP theo seed.
  KHÔNG dùng cột "O3 chéo" (R[t][t]) — nó chấm cuối chuyến, lúc tầng nhanh đã tự hội tụ
  nên mù với ngân hàng chế độ.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys

ARM_THU_TU = ["U0_dongbang", "U1_tangnhanh", "U2_nganhang", "arm1_lam1", "arm2_quen"]


def quet(goc: str) -> dict:
    """-> {(arm, seed): metrics_revisit dict}. Tên thư mục: artifacts_revisit_<arm>_s<seed>."""
    ra: dict = {}
    mau = os.path.join(goc, "artifacts_revisit_*", "results", "*", "metrics_revisit.json")
    for p in sorted(glob.glob(mau)):
        thu_muc = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))
        m = re.match(r"artifacts_revisit_(.+)_s(\d+)$", thu_muc)
        if not m:
            print(f"  (bỏ qua thư mục không đúng tên: {thu_muc})", file=sys.stderr)
            continue
        arm, seed = m.group(1), int(m.group(2))
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  (hỏng: {p} — {e})", file=sys.stderr)
            continue
        # ghép thêm average_accuracy từ metrics.json cạnh bên nếu có
        mp = os.path.join(os.path.dirname(p), "metrics.json")
        if os.path.exists(mp):
            try:
                d["_avg_acc"] = json.load(open(mp, encoding="utf-8")).get("average_accuracy")
            except (json.JSONDecodeError, OSError):
                pass
        ra[(arm, seed)] = d
    return ra


def _f(d: dict, *khoa, mac_dinh=None):
    for k in khoa:
        if k in d and d[k] is not None:
            return d[k]
    return mac_dinh


def _tb_sd(xs):
    xs = [x for x in xs if x is not None and not isinstance(x, str)]
    if not xs:
        return None, None, 0
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def in_bang(kq: dict) -> None:
    arms = sorted({a for a, _ in kq}, key=lambda a: (ARM_THU_TU.index(a) if a in ARM_THU_TU else 99, a))
    print(f"\n{'arm':<16}{'seed':>5}{'O1 acc':>9}{'O3 preq10 ⭐':>14}{'O3 chéo':>10}"
          f"{'O2 batch':>10}{'#chế độ':>9}{'#nạp':>6}")
    print("-" * 79)
    for arm in arms:
        for seed in sorted(s for a, s in kq if a == arm):
            d = kq[(arm, seed)]
            nh = d.get("ngan_hang") or {}
            o1 = _f(d, "acc_dieu_kien_hien_tai")
            o3p = _f(d, "o3_preq10_loi_ich")
            o3c = _f(d, "loi_ich_quay_lai")
            o2 = _f(d, "o2_hoi_phuc_tb_batch")
            print(f"{arm:<16}{seed:>5}"
                  f"{(f'{o1:.4f}' if o1 is not None else '—'):>9}"
                  f"{(f'{o3p:+.4f}' if o3p is not None else '—'):>14}"
                  f"{(f'{o3c:+.4f}' if o3c is not None else '—'):>10}"
                  f"{(f'{o2:.1f}' if isinstance(o2, (int, float)) else '—'):>10}"
                  f"{(str(nh.get('so_che_do')) if nh else '—'):>9}"
                  f"{(str(nh.get('so_lan_nap')) if nh else '—'):>6}")
        # dòng trung bình
        tb1, sd1, n = _tb_sd([_f(kq[(arm, s)], "acc_dieu_kien_hien_tai") for a, s in kq if a == arm])
        tb3, sd3, _ = _tb_sd([_f(kq[(arm, s)], "o3_preq10_loi_ich") for a, s in kq if a == arm])
        if n:
            print(f"{'  ↳ TB (' + str(n) + ' seed)':<21}"
                  f"{(f'{tb1:.4f}' if tb1 is not None else '—'):>9}"
                  f"{(f'{tb3:+.4f}' if tb3 is not None else '—'):>14}")
    print()


def con_so_cong_bo(kq: dict) -> None:
    seeds = sorted({s for a, s in kq if a == "U2_nganhang"} & {s for a, s in kq if a == "U1_tangnhanh"})
    if not seeds:
        print("⚠️ Chưa đủ cặp U1/U2 cùng seed — chưa tính được con số công bố.\n")
        return
    hieu = []
    print("CON SỐ CÔNG BỐ — O3 = preq10(U2) − preq10(U1), ghép cặp theo seed")
    for s in seeds:
        u1 = _f(kq[("U1_tangnhanh", s)], "o3_preq10_loi_ich")
        u2 = _f(kq[("U2_nganhang", s)], "o3_preq10_loi_ich")
        if u1 is None or u2 is None:
            continue
        hieu.append(u2 - u1)
        print(f"  seed {s}: {u2:+.4f} − {u1:+.4f} = {u2 - u1:+.4f}")
    tb, sd, n = _tb_sd(hieu)
    if tb is None:
        print("  (không đọc được o3_preq10_loi_ich)\n")
        return
    print(f"\n  ⭐ O3 = {tb:+.4f} ± {sd:.4f}  ({n} seed)")
    if tb <= 0:
        print("  ❌ Ngân hàng chế độ KHÔNG mang lại lợi ích quay lại — kết luận âm, "
              "vẫn phải báo cáo trung thực.")
    elif sd > abs(tb):
        print("  ⚠️ σ lớn hơn hiệu — chưa tách khỏi nhiễu. Cần thêm seed trước khi tuyên bố.")
    else:
        print("  ✅ Hiệu dương và vượt σ — có tín hiệu O3.")
    print()


def cua_chan_u1_hon_u0(kq: dict) -> int:
    a1, _, n1 = _tb_sd([_f(kq[(a, s)], "acc_dieu_kien_hien_tai") for a, s in kq if a == "U1_tangnhanh"])
    a0, _, n0 = _tb_sd([_f(kq[(a, s)], "acc_dieu_kien_hien_tai") for a, s in kq if a == "U0_dongbang"])
    if not n0 or not n1:
        print("⛔ CỬA CHẶN: thiếu kết quả U0 hoặc U1 — không kết luận được.")
        return 3
    print(f"CỬA CHẶN U1 vs U0 (O1 acc điều kiện hiện tại): U1={a1:.4f} ({n1} seed) · "
          f"U0={a0:.4f} ({n0} seed) · hiệu {a1 - a0:+.4f}")
    if a1 <= a0:
        print("⛔ U1 KHÔNG hơn U0 -> tầng nhanh không cứu được trôi. DỪNG, đừng chạy U2.\n")
        return 3
    print("✅ U1 hơn U0 — được phép chạy U2.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("goc", nargs="?", default=".", help="thư mục chứa artifacts_revisit_* (mặc định .)")
    ap.add_argument("--cua-chan", action="store_true", help="kiểm U1 > U0, exit 3 nếu không")
    a = ap.parse_args()

    kq = quet(a.goc)
    if not kq:
        print(f"Không thấy metrics_revisit.json nào dưới {a.goc}/artifacts_revisit_*/results/*/")
        return 1
    in_bang(kq)
    if a.cua_chan:
        return cua_chan_u1_hon_u0(kq)
    con_so_cong_bo(kq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
