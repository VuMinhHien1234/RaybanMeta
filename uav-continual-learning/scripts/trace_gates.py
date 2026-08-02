#!/usr/bin/env python3
"""TASK 0 — Trích QUỸ ĐẠO η/α theo từng task từ log ĐÃ CÓ (không chạy lại gì).

Vì sao cần: `methods.py` in η/α sau MỖI task và reset ở đầu task, nên mỗi log đã chứa
sẵn 1 giá trị / task. Nhưng `compare_all.py` chỉ lấy giá trị CUỐI CÙNG (`eas[-1]`)
-> toàn bộ quỹ đạo bị vứt đi. Script này lấy lại nó.

Câu hỏi cần trả lời (quyết định hướng vá của TASK 4):
  - α bão hoà ở 1.0 NGAY TỪ task 0  -> lỗi thang đo feature -> `pre_norm` là đúng thuốc.
  - α TRÔI DẦN lên 1.0 qua các task -> vòng phản hồi khi train -> cần thêm cơ chế chặn.

Ngữ nghĩa (xem neural_memory.py:813 `assoc_scan(1. - decay_factor, ...)`):
  α = decay_factor; hệ số GIỮ LẠI = (1 - α).
  α -> 1 : xoá sạch tích luỹ mỗi chunk (memory KHÔNG tích luỹ, norm phẳng)
  α -> 0 : giữ 100%, không quên gì -> HƯỚNG NỔ norm

η trong log CŨ là logit THÔ (probe gắn trước `adaptive_step_transform`).
η thật = sigmoid(logit) * max_lr, max_lr = 1.0 (neural_memory.py:272 — KHONG phai 1e-2).
Script tự quy đổi. Log MỚI (sau TASK 2) đã có sẵn `eta_real=` -> ưu tiên dùng.

Dùng:  python3 scripts/trace_gates.py [thư_mục_gốc]      (mặc định: result_test)
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "result_test")

# BAY: neural_memory.py:255 co `max_lr=1e-2` nhung :457 luon truyen
# `default_step_transform_max_lr = 1.` (:272) -> max_lr THAT = 1.0.
ETA_MAX_LR = 1.0
ALPHA_SAT_HI = 0.99        # >= : cổng quên bão hoà ở mức XOÁ SẠCH
ALPHA_SAT_LO = 0.01        # <= : cổng quên bão hoà ở mức GIỮ HẾT (hướng nổ)
ETA_FRAC_LO, ETA_FRAC_HI = 0.01, 0.95      # vung lanh manh, tinh theo PHAN SO cua max_lr
ETA_OK_LO, ETA_OK_HI = ETA_FRAC_LO * ETA_MAX_LR, ETA_FRAC_HI * ETA_MAX_LR
ALPHA_MOVE_MIN = 0.05      # α phải biến thiên ít nhất ngần này giữa các task (Eq 76)

# Định dạng CŨ:  eta_t(avg)=55.6201  alpha_t/forget-gate(avg)=1.0000
_RE_OLD = re.compile(r"eta_t\(avg\)=([-\w.eE+]+)\s+alpha_t/forget-gate\(avg\)=([-\w.eE+]+)")
# Định dạng MỚI (sau TASK 2): eta_raw=... eta_real=... alpha=... keep=...
_RE_NEW = re.compile(r"eta_raw=([-\w.eE+]+)\s+eta_real=([-\w.eE+]+)\s+alpha=([-\w.eE+]+)")
_RE_NORM = re.compile(r"norm\(state\) sau task = ([\d.]+)")


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _sigmoid(x):
    if x < -700:
        return 0.0
    if x > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def parse(txt):
    """-> (etas_raw, etas_real, alphas, norms) theo thứ tự task."""
    new = _RE_NEW.findall(txt)
    if new:
        raw = [_f(a) for a, _, _ in new]
        real = [_f(b) for _, b, _ in new]
        alpha = [_f(c) for _, _, c in new]
    else:
        old = _RE_OLD.findall(txt)
        raw = [_f(a) for a, _ in old]
        alpha = [_f(b) for _, b in old]
        real = [(_sigmoid(r) * ETA_MAX_LR if r is not None else None) for r in raw]
    norms = [_f(n) for n in _RE_NORM.findall(txt)]
    return raw, real, alpha, norms


def verdict(alphas):
    """Bão hoà NGAY TỪ ĐẦU hay TRÔI DẦN? -> quyết định hướng vá TASK 4."""
    vals = [a for a in alphas if a is not None]
    if not vals:
        return "khong-du-lieu", ""
    sat = [a >= ALPHA_SAT_HI or a <= ALPHA_SAT_LO for a in vals]
    span = max(vals) - min(vals)
    if all(sat) and sat[0]:
        return "BAO-HOA-TU-DAU", f"alpha bao hoa ngay task 0 va giu nguyen (span={span:.4f}) -> pre_norm (TASK 4)"
    if sat[-1] and not sat[0]:
        return "TROI-DAN", f"alpha troi tu {vals[0]:.4f} -> {vals[-1]:.4f} -> can them co che chan (TASK 5 + weight decay)"
    if span < ALPHA_MOVE_MIN:
        return "DUNG-YEN", f"alpha khong bao hoa nhung DUNG YEN (span={span:.4f} < {ALPHA_MOVE_MIN}) -> Eq 76 van chua song"
    return "LANH-MANH", f"alpha bien thien span={span:.4f} -> cong con phu thuoc du lieu"


def main():
    logs = sorted(p for p in ROOT.rglob("*.log"))
    if not logs:
        print(f"Khong tim thay *.log nao trong {ROOT}/")
        return
    seen, tally = set(), {}
    for lg in logs:
        try:
            txt = lg.read_text(errors="ignore")
        except OSError:
            continue
        raw, real, alphas, norms = parse(txt)
        if not alphas:
            continue
        key = (lg.name, tuple(alphas))         # bo qua ban sao trung (res1/res2/res3 chong nhau)
        if key in seen:
            continue
        seen.add(key)

        tag, why = verdict(alphas)
        tally[tag] = tally.get(tag, 0) + 1
        print(f"\n{'=' * 78}\n{lg.name}   [{tag}]\n{'=' * 78}")
        print(f"  {why}")
        print(f"  {'task':>5} {'eta_raw':>10} {'eta THAT':>10} {'':>4} {'alpha':>8} {'giu lai':>8} {'norm':>13}")
        n = max(len(alphas), len(norms))
        for i in range(n):
            a = alphas[i] if i < len(alphas) else None
            r = raw[i] if i < len(raw) else None
            e = real[i] if i < len(real) else None
            nm = norms[i] if i < len(norms) else None
            f_a = "SAT" if a is not None and (a >= ALPHA_SAT_HI or a <= ALPHA_SAT_LO) else ""
            f_e = "!" if e is not None and not (ETA_OK_LO <= e <= ETA_OK_HI) else ""
            print(f"  {i:>5} {_s(r, '10.3f'):>10} {_s(e, '10.2e'):>10} {f_e:>4} "
                  f"{_s(a, '8.4f'):>8} {_s(None if a is None else 1 - a, '8.4f'):>8} "
                  f"{_s(nm, '13.1f'):>13} {f_a}")

    print(f"\n{'=' * 78}\nTONG KET\n{'=' * 78}")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18} {v:3} run")
    print("\nKet luan huong va TASK 4:")
    if tally.get("BAO-HOA-TU-DAU", 0) >= max(tally.values(), default=0):
        print("  -> Chu dao la BAO HOA TU DAU: nguyen nhan la THANG DO feature dau vao memory.")
        print("     `pre_norm` (TASK 4) la dung thuoc. TASK 5 chi la bo tro.")
    elif tally.get("TROI-DAN", 0):
        print("  -> Co run TROI DAN: ngoai pre_norm can them chan (weight decay tren 2 cong / clamp logit).")
    print(f"\nNguong dung: alpha bao hoa neu >= {ALPHA_SAT_HI} hoac <= {ALPHA_SAT_LO} | "
          f"eta THAT lanh manh trong [{ETA_OK_LO:g}, {ETA_OK_HI:g}]\n")


def _s(v, fmt):
    return "-" if v is None else format(v, fmt)


if __name__ == "__main__":
    main()
