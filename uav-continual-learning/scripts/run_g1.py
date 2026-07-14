#!/usr/bin/env python3
"""G1 runner — chạy MỘT thí nghiệm học liên tục từ 1 file config.

Ví dụ:
  python scripts/run_g1.py --config configs/g1_smoke.yaml                     # kiểm tra pipeline (CPU, ~1 phút)
  python scripts/run_g1.py --config configs/g1_eurosat.yaml --method finetune
  python scripts/run_g1.py --config configs/g1_eurosat.yaml --method ewc
  python scripts/run_g1.py --config configs/g1_resisc45.yaml --method ewc --set train.lr=5e-5

Kết quả lưu ở  <log.dir>/results/<dataset>_<method>_seed<seed>/ :
  acc_matrix.csv   ma trận R[i,j] = acc task j sau khi học task i
  metrics.json     average_accuracy / average_forgetting / backward_transfer
  config.yaml      config đã dùng (tái lập được)
Sau khi chạy >=2 method, gộp bảng so sánh:  python scripts/compare_g1.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="đường dẫn file yaml (vd configs/g1_eurosat.yaml)")
    ap.add_argument("--method", default=None, help="finetune | ewc (mặc định lấy từ config)")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="k.sub=v",
                    help="override config, vd: train.lr=1e-4 data.num_tasks=5")
    args = ap.parse_args()

    from uavcl.utils.config import apply_overrides, load_config, save_config
    from uavcl.utils.seed import seed_everything

    cfg = apply_overrides(load_config(args.config), args.overrides)
    method_name = (args.method or cfg.get("train", {}).get("method", "finetune")).lower()
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    try:
        import torch  
    except ImportError:
        print("[ERR] Chưa có torch — xem README bước 3 để cài đúng nền tảng.")
        return 1
    from uavcl.data import build_stream, describe_stream, get_source
    from uavcl.data.loaders import build_task_loaders
    from uavcl.engine import resolve_device, run_continual
    from uavcl.methods import build_method
    from uavcl.metrics import average_accuracy, average_forgetting, backward_transfer, forward_transfer
    from uavcl.models import ContinualClassifier, build_backbone

    device = resolve_device(cfg.get("device", "auto"))
    print(f"== G1 run: dataset={cfg['data']['name']} method={method_name} seed={seed} device={device.type}")

    # --- data ---
    t0 = time.time()
    source = get_source(cfg["data"])
    stream = build_stream(
        source.splits["train"].labels,
        source.splits["val"].labels,
        source.splits["test"].labels,
        num_classes=source.num_classes,
        num_tasks=int(cfg["data"]["num_tasks"]),
        seed=seed,
        shuffle_classes=bool(cfg["data"].get("shuffle_classes", True)),
    )
    print(describe_stream(stream, source.class_names))
    loaders = build_task_loaders(source, stream, cfg["data"])

    # --- model + method ---
    backbone, feat_dim = build_backbone(cfg["backbone"])
    if method_name == "ncm":
        from uavcl.models.ncm import NCMClassifier

        model = NCMClassifier(backbone, feat_dim, source.num_classes).to(device)
    else:
        model = ContinualClassifier(backbone, feat_dim, source.num_classes).to(device)
    method = build_method(method_name, cfg)

    # --- run ---
    R, log = run_continual(model, method, stream, loaders, device, cfg["train"])

    # --- metrics + save ---
    metrics = {
        "dataset": source.name,
        "method": method_name,
        "seed": seed,
        "num_tasks": len(stream),
        "backbone": cfg["backbone"]["name"],
        "average_accuracy": average_accuracy(R),
        "average_forgetting": average_forgetting(R),
        "backward_transfer": backward_transfer(R),
        # FWT chỉ có nghĩa khi bật train.eval_future (đo acc task kế tiếp TRƯỚC khi học)
        "forward_transfer": forward_transfer(R) if cfg["train"].get("eval_future", False) else None,
        # chi phí — so cùng accuracy thì method rẻ hơn thắng:
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "method_extra_floats": int(method.footprint_floats(model)),
        "runtime_sec": round(time.time() - t0, 1),
    }
    run_name = f"{source.name}_{method_name}_seed{seed}"
    out = pathlib.Path(cfg.get("log", {}).get("dir", "./artifacts")) / "results" / run_name
    out.mkdir(parents=True, exist_ok=True)
    cols = [f"task{j}" for j in range(len(stream))]
    pd.DataFrame(R, index=[f"after_task{i}" for i in range(len(stream))], columns=cols) \
        .to_csv(out / "acc_matrix.csv", float_format="%.4f")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_config(cfg, out / "config.yaml")

    print("\n== Ma trận accuracy (hàng = sau khi học task i, cột = đánh giá task j)")
    with np.printoptions(precision=3, suppress=True):
        print(np.tril(R))
    print("\n== Kết quả")
    print(f"  Average Accuracy   : {metrics['average_accuracy']:.4f}  (cao = tốt)")
    print(f"  Average Forgetting : {metrics['average_forgetting']:.4f}  (thấp = tốt — con số dự án cần giảm)")
    print(f"  Backward Transfer  : {metrics['backward_transfer']:.4f}  (âm = quên)")
    print(f"  Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
