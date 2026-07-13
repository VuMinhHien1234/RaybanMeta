#!/usr/bin/env python3
"""Gộp mọi run trong <dir>/results/*/metrics.json thành MỘT bảng số mốc (baseline).

  python scripts/compare_g1.py                # đọc ./artifacts/results
  python scripts/compare_g1.py --dir path/khac

Bảng này là "cổng G1": từ G2 trở đi, Titans/CMS/HOPE phải thắng các dòng ở đây
(Average Accuracy cao hơn, Average Forgetting thấp hơn) mới được coi là có tác dụng.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./artifacts", help="thư mục log (chứa results/)")
    args = ap.parse_args()

    root = pathlib.Path(args.dir) / "results"
    rows = []
    for p in sorted(root.glob("*/metrics.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        print(f"Chưa có run nào trong {root}. Chạy scripts/run_g1.py trước.")
        return 1

    cols = ["dataset", "method", "backbone", "seed", "num_tasks",
            "average_accuracy", "average_forgetting", "backward_transfer",
            "trainable_params", "method_extra_floats", "runtime_sec"]
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]].sort_values(["dataset", "method", "seed"])
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = root / "baseline_table.md"
    out.write_text(df.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
