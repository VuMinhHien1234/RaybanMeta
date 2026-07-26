#!/usr/bin/env python3
"""compare_all.py — gộp MỌI kết quả rải rác thành 1 bảng tổng.

Quét đệ quy từ 1 thư mục gốc:
  - artifacts_*/results/*/metrics.json      -> optimizer, seed, Linear-head Acc/Forget
  - .../metrics_ncm.json (nếu có)           -> NCM-head Acc/Forget (đòn A)
  - *.log                                    -> norm(state) cuối + eta_t/alpha_t cuối

Chỉ dùng thư viện CHUẨN (json/re/sys/pathlib) -> chạy bằng `python3` thường, không cần venv.

Dùng:
  python3 scripts/compare_all.py [thư_mục_gốc]     # mặc định: thư mục hiện tại
Ví dụ (trên Mac, sau khi giải nén các tarball vào result_test/):
  cd ~/Desktop/Raybanmeta
  python3 uav-continual-learning/scripts/compare_all.py result_test
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()


def _num(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "   -   "


# ---------------------------------------------------------------- 1) metrics.json
rows = []
for mj in sorted(ROOT.rglob("metrics.json")):
    try:
        m = json.loads(mj.read_text())
    except Exception:
        continue
    ncm = {}
    nj = mj.parent / "metrics_ncm.json"
    if nj.exists():
        try:
            ncm = json.loads(nj.read_text())
        except Exception:
            ncm = {}
    # artifacts_XXX/results/<run>/metrics.json -> parents[2] = artifacts_XXX
    folder = mj.parents[2].name if len(mj.parents) >= 3 else mj.parent.name
    rows.append({
        "folder": folder,
        "opt": str(m.get("optimizer", "?")),
        "seed": str(m.get("seed", "?")),
        "lin_acc": m.get("average_accuracy"),
        "lin_fgt": m.get("average_forgetting"),
        "ncm_acc": ncm.get("average_accuracy"),
        "ncm_fgt": ncm.get("average_forgetting"),
    })

rows.sort(key=lambda r: (r["opt"], r["seed"], r["folder"]))

print("\n================= BẢNG TỔNG (metrics.json) =================")
print(f"{'folder':30} {'opt':6} {'seed':4} | {'Lin Acc':8} {'Lin Fgt':8} | {'NCM Acc':8} {'NCM Fgt':8}")
print("-" * 90)
for r in rows:
    star = "  <-- đạt mốc thắng" if isinstance(r["ncm_acc"], (int, float)) and r["ncm_acc"] >= 0.72 \
        and isinstance(r["ncm_fgt"], (int, float)) and r["ncm_fgt"] <= 0.10 else ""
    print(f"{r['folder'][:30]:30} {r['opt']:6} {r['seed']:4} | "
          f"{_num(r['lin_acc']):8} {_num(r['lin_fgt']):8} | "
          f"{_num(r['ncm_acc']):8} {_num(r['ncm_fgt']):8}{star}")
if not rows:
    print("(không tìm thấy metrics.json nào dưới:", ROOT, ")")
print("\nMốc: NCM gốc 0.6933/0.10 · replay 0.7937/0.079 · THẮNG khi Acc>=0.72 & Forget<=0.10")

# ---------------------------------------------------------------- 2) log: norm + eta/alpha
print("\n============ norm(state) & eta/alpha CUỐI (từ *.log) ============")
_norm_re = re.compile(r"norm\(state\) sau task = ([\d.]+)")
_ea_re = re.compile(r"eta_t\(avg\)=([\w.eE+/-]+)\s+alpha_t/forget-gate\(avg\)=([\w.eE+/-]+)")
found_log = False
for lg in sorted(ROOT.rglob("*.log")):
    try:
        txt = lg.read_text(errors="ignore")
    except Exception:
        continue
    norms = _norm_re.findall(txt)
    eas = _ea_re.findall(txt)
    if not norms and not eas:
        continue
    found_log = True
    last_norm = norms[-1] if norms else "-"
    flag = " ⚠️NỔ" if norms and float(norms[-1]) > 1e4 else ""
    eta, alpha = (eas[-1] if eas else ("-", "-"))
    print(f"{lg.name[:34]:34}  norm={last_norm:>14}{flag:6}  eta={eta:>10}  alpha={alpha:>8}")
if not found_log:
    print("(không log nào có dòng norm(state)/eta — có thể chạy bằng code cũ chưa có phần log)")

print("\nĐọc nhanh: norm > 1e4 = NỔ · alpha(forget-gate)~1 = không quên = hướng nổ · "
      "NCM Acc cao & bền qua seed/opt = lời giải chốt.\n")
