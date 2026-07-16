#!/usr/bin/env python3
"""MỘT LỆNH chạy toàn bộ chuỗi thí nghiệm G1→G4 + sinh báo cáo docx.

    python scripts/run_all.py                 # tất cả (đầy đủ, ~8–14h GPU T4)
    python scripts/run_all.py --quick         # chỉ EuroSAT (~2–4h) — chạy thử trước!
    python scripts/run_all.py --stages g3,g4,report
    python scripts/run_all.py --shutdown      # tự tắt máy GCP khi xong (đỡ tốn tiền)

Tính chất:
- RESUMABLE: run nào đã có metrics.json thì bỏ qua (--skip-existing) — đứt mạng/
  restart cứ chạy lại đúng lệnh cũ.
- Tự chọn cấu hình CMS THẮNG từ ablation G3 (điểm = Acc − Forgetting) để đưa vào G4.
- Dừng ngay khi 1 run lỗi (trừ khi --keep-going).
- Cuối cùng: compare_g1 + make_report -> artifacts/BAO_CAO_KET_QUA.docx
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
RESULTS = ROOT / "artifacts" / "results"


def sh(args_list, keep_going=False) -> bool:
    print(f"\n$ {' '.join(str(a) for a in args_list)}", flush=True)
    t0 = time.time()
    r = subprocess.run([str(a) for a in args_list], cwd=ROOT)
    print(f"  -> exit={r.returncode} ({time.time() - t0:.0f}s)", flush=True)
    if r.returncode != 0 and not keep_going:
        raise SystemExit(f"[STOP] Lệnh lỗi (exit {r.returncode}). Sửa rồi chạy lại — các run xong rồi sẽ được skip.")
    return r.returncode == 0


def run_g1_cmd(config, method=None, sets=None):
    cmd = [PY, "scripts/run_g1.py", "--config", config, "--skip-existing"]
    if method:
        cmd += ["--method", method]
    if sets:
        cmd += ["--set"] + list(sets)
    return cmd


def pick_g3_winner() -> dict | None:
    """Đọc các run cms trên EuroSAT, chọn cấu hình có (Acc − Forgetting) cao nhất."""
    best, best_score = None, -1e9
    for p in RESULTS.glob("eurosat_cms_*/metrics.json"):
        m = json.loads(p.read_text())
        score = float(m["average_accuracy"]) - float(m["average_forgetting"])
        if score > best_score:
            best_score, best = score, m
    if best is None:
        return None
    periods = [int(x) for x in str(best.get("cms_periods", "1-4-16")).split("-")]
    return {"order": best.get("cms_order", "late_slow"), "periods": periods,
            "score": best_score, "run": f"{best['dataset']}_{best['method']}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="g1,g2,g3,g4,report",
                    help="chọn stage, vd: g2,g3 (mặc định: tất cả)")
    ap.add_argument("--quick", action="store_true", help="chỉ EuroSAT (bỏ RESISC45) — để thử nhanh")
    ap.add_argument("--keep-going", action="store_true", help="1 run lỗi vẫn chạy tiếp run khác")
    ap.add_argument("--shutdown", action="store_true", help="tắt máy khi xong (cho GCP)")
    args = ap.parse_args()
    stages = {s.strip() for s in args.stages.split(",")}
    kg = args.keep_going
    t_start = time.time()

    # ---------------- G1: 5 baseline ----------------
    if "g1" in stages:
        print("\n========== G1 — 5 baseline ==========")
        for m in ["finetune", "ewc", "replay", "lwf", "ncm"]:
            sh(run_g1_cmd("configs/g1_eurosat.yaml", m), kg)
        if not args.quick:
            for m in ["finetune", "ewc", "replay", "lwf", "ncm"]:
                sh(run_g1_cmd("configs/g1_resisc45.yaml", m), kg)

    # ---------------- G2: Titans bậc A/B/C ----------------
    if "g2" in stages:
        print("\n========== G2 — Titans (A -> B -> C) ==========")
        sh(run_g1_cmd("configs/g2_titans_eurosat.yaml", sets=["memory.reset=image"]), kg)
        sh(run_g1_cmd("configs/g2_titans_eurosat.yaml", sets=["memory.reset=task"]), kg)
        sh(run_g1_cmd("configs/g2_titans_eurosat.yaml",
                      sets=["memory.reset=never", "train.eval_future=true"]), kg)
        # đối chứng optimizer trên bậc C
        sh(run_g1_cmd("configs/g2_titans_eurosat.yaml",
                      sets=["memory.reset=never", "train.eval_future=true", "train.optimizer=adamw"]), kg)
        if not args.quick:
            sh(run_g1_cmd("configs/g2_titans_resisc45.yaml",
                          sets=["memory.reset=never", "train.eval_future=true"]), kg)

    # ---------------- G3: CMS ablation + winner ----------------
    if "g3" in stages:
        print("\n========== G3 — CMS ablation (4 run EuroSAT) ==========")
        for order in ["late_slow", "early_slow"]:
            for tiers in ["[[4,1],[4,4],[4,16]]", "[[4,1],[4,8],[4,64]]"]:
                sh(run_g1_cmd("configs/g3_cms_eurosat.yaml",
                              sets=[f"cms.order={order}", f"cms.tiers={tiers}"]), kg)
        w = pick_g3_winner()
        if w:
            print(f"\n[G3] CẤU HÌNH THẮNG: order={w['order']} periods={w['periods']} (score={w['score']:.4f})")
            if not args.quick:
                tiers = str([[4, p] for p in w["periods"]]).replace(" ", "")
                sh(run_g1_cmd("configs/g3_cms_resisc45.yaml",
                              sets=[f"cms.order={w['order']}", f"cms.tiers={tiers}"]), kg)
                sh(run_g1_cmd("configs/g3_cms_resisc45.yaml",  # đối chứng optimizer
                              sets=[f"cms.order={w['order']}", f"cms.tiers={tiers}",
                                    "train.optimizer=adamw"]), kg)

    # ---------------- G4: HOPE (ghép cấu hình thắng) ----------------
    if "g4" in stages:
        print("\n========== G4 — HOPE (Titans + CMS-winner + M3-delta) ==========")
        w = pick_g3_winner()
        wsets = []
        if w:
            tiers = str([[4, p] for p in w["periods"]]).replace(" ", "")
            wsets = [f"cms.order={w['order']}", f"cms.tiers={tiers}"]
            print(f"[G4] dùng CMS winner: {wsets}")
        else:
            print("[G4] chưa có kết quả G3 -> dùng cấu hình mặc định trong config")
        sh(run_g1_cmd("configs/g4_hope_eurosat.yaml", sets=wsets or None), kg)
        if not args.quick:
            sh(run_g1_cmd("configs/g4_hope_resisc45.yaml", sets=wsets or None), kg)
            sh(run_g1_cmd("configs/g4_hope_resisc45.yaml",  # 2x2: HOPE + adamw
                          sets=(wsets + ["train.optimizer=adamw"])), kg)

    # ---------------- Báo cáo ----------------
    if "report" in stages:
        print("\n========== Báo cáo ==========")
        sh([PY, "scripts/compare_g1.py"], keep_going=True)
        sh([PY, "scripts/make_report.py"], kg)

    print(f"\n[DONE] Tổng thời gian: {(time.time() - t_start) / 3600:.2f} giờ."
          f" Kết quả: artifacts/results/ + artifacts/BAO_CAO_KET_QUA.docx")
    if args.shutdown:
        print("[SHUTDOWN] Tắt máy sau 1 phút (Ctrl+C để hủy)...")
        time.sleep(60)
        subprocess.run(["sudo", "shutdown", "-h", "now"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
