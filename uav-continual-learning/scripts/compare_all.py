#!/usr/bin/env python3
"""compare_all.py — gộp MỌI kết quả rải rác thành 1 bảng tổng.

Quét đệ quy từ 1 thư mục gốc:
  - artifacts_*/results/*/metrics.json      -> method, head, optimizer, seed, Acc/Forget/AAA
  - .../metrics_ncm.json (nếu có)           -> NCM-head shadow Acc/Forget (rebuild HOẶC sdc — xem tên folder)
  - .../metrics_openset.json (nếu có)       -> AUC/EER/TAR@FAR (#27)
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
import math
import re
import sys
from pathlib import Path


def _keep(alpha_s):
    """Hệ số GIỮ LẠI = (1 − α) — con số có ý nghĩa trực tiếp [neural_memory.py:813]."""
    try:
        return f"{1.0 - float(alpha_s):.4f}"
    except (TypeError, ValueError):
        return "-"

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()

WIN_ACC, WIN_FGT = 0.72, 0.10   # mốc thắng của dự án (Acc >= / Forget <=)


def _num(x, w=7):
    return f"{x:{w}.4f}" if isinstance(x, (int, float)) else " " * (w - 3) + "-  "


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------- 1) metrics.json
rows = []
for mj in sorted(ROOT.rglob("metrics.json")):
    m = _load(mj)
    if not m:
        continue
    ncm = _load(mj.parent / "metrics_ncm.json")
    osr = _load(mj.parent / "metrics_openset.json")
    # artifacts_XXX/results/<run>/metrics.json -> parents[2] = artifacts_XXX
    folder = mj.parents[2].name if len(mj.parents) >= 3 else mj.parent.name
    rows.append({
        "folder": folder,
        "method": str(m.get("method", "?")),
        "head": str(m.get("head", "-")),
        "opt": str(m.get("optimizer", "?")),
        "seed": str(m.get("seed", "?")),
        "acc": m.get("average_accuracy"),
        "fgt": m.get("average_forgetting"),
        "aaa": m.get("average_anytime_accuracy"),
        "ncm_acc": ncm.get("average_accuracy"),
        "ncm_fgt": ncm.get("average_forgetting"),
        "os_auc": osr.get("auc"),
    })

rows.sort(key=lambda r: (r["method"], r["head"], r["opt"], r["seed"], r["folder"]))

print("\n======================== BẢNG TỔNG (metrics.json) ========================")
print(f"{'folder':26} {'method':13} {'head':6} {'opt':5} {'sd':2} | "
      f"{'Acc':7} {'Fgt':7} {'AAA':7} | {'NCMAcc':7} {'NCMFgt':7}")
print("-" * 108)
for r in rows:
    def _win(a, f):
        return isinstance(a, (int, float)) and isinstance(f, (int, float)) and a >= WIN_ACC and f <= WIN_FGT
    star = ""
    if _win(r["acc"], r["fgt"]):
        star = "  <-- ĐẠT MỐC (head chính)"     # ↳ cosine/linear tự thân đạt — ứng viên chốt
    elif _win(r["ncm_acc"], r["ncm_fgt"]):
        star = "  <-- đạt mốc (NCM shadow)"      # ↳ chỉ readout phụ đạt — xem tên folder: sdc hay rebuild
    print(f"{r['folder'][:26]:26} {r['method'][:13]:13} {r['head'][:6]:6} {r['opt'][:5]:5} {r['seed']:2} | "
          f"{_num(r['acc'])} {_num(r['fgt'])} {_num(r['aaa'])} | "
          f"{_num(r['ncm_acc'])} {_num(r['ncm_fgt'])}{star}")
if not rows:
    print("(không tìm thấy metrics.json nào dưới:", ROOT, ")")
print(f"\nMốc: THẮNG khi Acc>={WIN_ACC} & Forget<={WIN_FGT} (bền qua >=3 seed) · replay ảnh 0.7937 · "
      "NCM shadow của run sdc = chế độ SDC, của run cosine/rebuild = rebuild.")
print("AAA (#24) = trung bình acc TẠI MỌI MỐC — quan trọng cho regime streaming/UAV.")

# ---------------------------------------------------------------- 2) open-set (#27)
os_rows = [(r["folder"], r["head"], r["seed"], r["os_auc"]) for r in rows if r["os_auc"] is not None]
if os_rows:
    print("\n==================== OPEN-SET (#27, metrics_openset.json) ====================")
    for folder, head, seed, aucv in os_rows:
        print(f"{folder[:30]:30} head={head:6} seed={seed:2}  AUC={_num(aucv)}")
    print("(AUC ~1 = tách 'quen/lạ' tốt; so head linear vs cosine/NCM ở đây)")

# ---------------------------------------------------------------- 3) log: norm + eta/alpha
print("\n============ norm(state) & eta/alpha CUỐI (từ *.log) ============")
_norm_re = re.compile(r"norm\(state\) sau task = ([\d.]+)")
# Định dạng CŨ (vẫn được in song song để tương thích ngược)
_ea_re = re.compile(r"eta_t\(avg\)=([\w.eE+/-]+)\s+alpha_t/forget-gate\(avg\)=([\w.eE+/-]+)")
# Định dạng MỚI (2026-08-02): có thêm η THẬT = sigmoid(logit)*max_lr
_ea_new_re = re.compile(r"eta_raw=([\w.eE+/-]+)\s+eta_real=([\w.eE+/-]+)\s+alpha=([\w.eE+/-]+)")
# max_lr THẬT = 1.0 (neural_memory.py:272 `default_step_transform_max_lr = 1.`); giá trị
# 1e-2 ở :255 là default của hàm transform, KHÔNG bao giờ được dùng vì :457 luôn ghi đè.
ETA_MAX_LR = 1.0
ALPHA_SAT_HI, ALPHA_SAT_LO = 0.99, 0.01
ETA_OK_LO, ETA_OK_HI = 0.01 * ETA_MAX_LR, 0.95 * ETA_MAX_LR


def _eta_real_from_raw(raw_s):
    """Log cũ chỉ có logit thô -> quy đổi sang η thật để so sánh được với log mới."""
    try:
        x = float(raw_s)
    except (TypeError, ValueError):
        return None
    if x < -700:
        return 0.0
    if x > 700:
        return ETA_MAX_LR
    return (1.0 / (1.0 + math.exp(-x))) * ETA_MAX_LR


found_log = False
for lg in sorted(ROOT.rglob("*.log")):
    try:
        txt = lg.read_text(errors="ignore")
    except Exception:
        continue
    norms = _norm_re.findall(txt)
    new = _ea_new_re.findall(txt)
    eas = _ea_re.findall(txt)
    if not norms and not eas and not new:
        continue
    found_log = True
    last_norm = norms[-1] if norms else "-"
    flag = " ⚠️NỔ" if norms and float(norms[-1]) > 1e4 else ""
    if new:                                   # log MỚI: đọc thẳng η thật
        eta, eta_real_s, alpha = new[-1]
        try:
            eta_real = float(eta_real_s)
        except (TypeError, ValueError):
            eta_real = None
    elif eas:                                 # log CŨ: tự quy đổi
        eta, alpha = eas[-1]
        eta_real = _eta_real_from_raw(eta)
    else:
        eta, alpha, eta_real = "-", "-", None
    # cờ bão hoà: đây là thứ đáng lẽ đã bắt được lỗi α/η từ tháng trước
    try:
        a = float(alpha)
        sat = " ⚠️α-SAT" if (a > ALPHA_SAT_HI or a < ALPHA_SAT_LO) else ""
    except (TypeError, ValueError):
        sat = ""
    if eta_real is not None and not (ETA_OK_LO <= eta_real <= ETA_OK_HI):
        sat += " ⚠️η-SAT"
    er = f"{eta_real:.2e}" if eta_real is not None else "-"
    print(f"{lg.name[:30]:30}  norm={last_norm:>13}{flag:6}  eta_raw={eta:>9}  "
          f"eta_real={er:>9}  alpha={alpha:>7}  keep={_keep(alpha):>7}{sat}")
if not found_log:
    print("(không log nào có dòng norm(state)/eta — có thể chạy bằng code cũ chưa có phần log)")

print("\nĐọc nhanh: norm > 1e4 = NỔ · alpha=decay_factor, GIỮ LẠI=(1−alpha) [neural_memory.py:813]")
print("  alpha→0 = không quên gì  = HƯỚNG NỔ norm      (bằng chứng: m3_s1 α=0 → norm 1.75e6)")
print("  alpha→1 = xoá sạch mỗi chunk = memory KHÔNG tích luỹ (norm phẳng ~54)")
print(f"  eta_real = sigmoid(logit)*max_lr({ETA_MAX_LR:g}) — lành mạnh trong [{ETA_OK_LO:g}, {ETA_OK_HI:g}]; "
      "eta_raw là logit THÔ, không phải learning rate")
print("  ⚠️SAT = cổng bão hoà -> Eq 76 không còn phụ thuộc dữ liệu -> KHÔNG tin accuracy của run đó")
print("  Quỹ đạo η/α theo TỪNG task: python3 scripts/trace_gates.py <thư_mục>\n")
