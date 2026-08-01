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


def run_dir_name(cfg: dict, method_name: str) -> str:
    """Tên thư mục kết quả — nguồn duy nhất, dùng cho cả ghi lẫn resume (--skip-existing)."""
    name = f"{str(cfg['data']['name']).lower()}_{method_name}_seed{int(cfg.get('seed', 0))}"
    opt = str(cfg.get("train", {}).get("optimizer", "adamw")).lower()
    if opt != "adamw":
        name += f"_{opt}"
    mem_cfg = cfg.get("memory") or {}
    if mem_cfg.get("enabled", False):  # 3 bậc reset A/B/C KHÔNG được ghi đè/skip lẫn nhau
        name += f"_r{str(mem_cfg.get('reset', 'image')).lower()}"
    if not bool(cfg.get("train", {}).get("optimizer_per_task", True)):
        name += "_optkeep"  # ablation "ký ức optimizer xuyên task" không đè run thường
    cms_cfg = cfg.get("cms") or {}
    if cms_cfg.get("enabled", False):
        periods = "-".join(str(t[1]) for t in cms_cfg.get("tiers", []))
        name += f"_{cms_cfg.get('order', 'late_slow')}_p{periods}"
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="đường dẫn file yaml (vd configs/g1_eurosat.yaml)")
    ap.add_argument("--method", default=None, help="finetune | ewc (mặc định lấy từ config)")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="k.sub=v",
                    help="override config, vd: train.lr=1e-4 data.num_tasks=5")
    ap.add_argument("--skip-existing", action="store_true",
                    help="đã có metrics.json cho run này thì bỏ qua (resume cho run_all)")
    args = ap.parse_args()

    from uavcl.utils.config import apply_overrides, load_config, save_config
    from uavcl.utils.seed import seed_everything

    cfg = apply_overrides(load_config(args.config), args.overrides)
    method_name = (args.method or cfg.get("train", {}).get("method", "finetune")).lower()
    seed = int(cfg.get("seed", 0))

    out = pathlib.Path(cfg.get("log", {}).get("dir", "./artifacts")) / "results" / run_dir_name(cfg, method_name)
    if args.skip_existing and (out / "metrics.json").exists():
        print(f"[SKIP] {out.name} — đã có kết quả.")
        return 0
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
    head_kind = str(cfg.get("head", "linear")).lower()  # ↳ 'linear' (mặc định) | 'cosine' (fix recency bias của head)
    mem_cfg = cfg.get("memory", {}) or {}
    cms_on = bool((cfg.get("cms") or {}).get("enabled", False))
    if mem_cfg.get("enabled", False) and cms_on:
        # G4 — HOPE: Titans (tầng nhanh) + backbone-CMS (tầng trung/chậm)
        from uavcl.models.hope import HOPEClassifier

        model = HOPEClassifier(backbone, feat_dim, source.num_classes, mem_cfg, head=head_kind).to(device)
        if method_name != "hope":
            print(f"[WARN] memory+cms cùng bật + method '{method_name}' — thường dùng --method hope")
        print(f"[hope] seq={model.seq_mode} reset={model.reset_mode} chunk={model.memory.chunk_size}")
    elif mem_cfg.get("enabled", False):
        from uavcl.models.titans_head import TitansClassifier

        model = TitansClassifier(backbone, feat_dim, source.num_classes, mem_cfg, head=head_kind).to(device)
        if method_name not in ("titans", "finetune"):
            print(f"[WARN] memory.enabled + method '{method_name}' — thường dùng --method titans")
        print(f"[titans] seq={model.seq_mode} reset={model.reset_mode} chunk={model.memory.chunk_size}")
    elif method_name == "ncm":
        from uavcl.models.ncm import NCMClassifier

        model = NCMClassifier(backbone, feat_dim, source.num_classes).to(device)
    else:
        model = ContinualClassifier(backbone, feat_dim, source.num_classes, head=head_kind).to(device)
    method = build_method(method_name, cfg)

    # --- run ---
    cms_cfg = cfg.get("cms") or {}
    if cms_cfg.get("enabled", False):
        cfg["train"]["cms"] = cms_cfg  # engine đọc từ train_cfg -> dùng CMSOptimizer (G3)
    R, log = run_continual(model, method, stream, loaders, device, cfg["train"])

    # --- metrics + save ---
    metrics = {
        "dataset": source.name,
        "method": method_name,
        "seed": seed,
        "num_tasks": len(stream),
        "backbone": cfg["backbone"]["name"],
        "head": head_kind,
        "optimizer": str(cfg["train"].get("optimizer", "adamw")).lower(),
        "optimizer_per_task": bool(cfg["train"].get("optimizer_per_task", True)),
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
    if mem_cfg.get("enabled", False):
        metrics["memory_reset"] = str(mem_cfg.get("reset", "image")).lower()
    if cms_cfg.get("enabled", False):
        metrics["cms_order"] = cms_cfg.get("order", "late_slow")
        metrics["cms_periods"] = "-".join(str(t[1]) for t in cms_cfg.get("tiers", []))
    out.mkdir(parents=True, exist_ok=True)  # out đã tính từ run_dir_name ở đầu main
    cols = [f"task{j}" for j in range(len(stream))]
    pd.DataFrame(R, index=[f"after_task{i}" for i in range(len(stream))], columns=cols) \
        .to_csv(out / "acc_matrix.csv", float_format="%.4f")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_config(cfg, out / "config.yaml")
    if hasattr(model, "export_state"):  # G2 (S10): lưu "cục ký ức" cuối stream
        st = model.export_state()
        if st is not None:
            import torch as _torch

            _torch.save(st, out / "memory_state.pt")
            print(f"  memory_state.pt đã lưu (norm={model.state_norm():.4f})")

    print("\n== Ma trận accuracy (hàng = sau khi học task i, cột = đánh giá task j)")
    with np.printoptions(precision=3, suppress=True):
        print(np.tril(R))
    print("\n== Kết quả")
    print(f"  Average Accuracy   : {metrics['average_accuracy']:.4f}  (cao = tốt)")
    print(f"  Average Forgetting : {metrics['average_forgetting']:.4f}  (thấp = tốt — con số dự án cần giảm)")
    print(f"  Backward Transfer  : {metrics['backward_transfer']:.4f}  (âm = quên)")

    # ĐÒN A: nếu bật train.eval_ncm_head, log có ma trận NCM-head -> tính + lưu + so sánh.
    if "ncm_R" in log:
        Rn = np.asarray(log["ncm_R"])
        ncm_metrics = {
            "readout": "ncm_head_posthoc",
            "average_accuracy": average_accuracy(Rn),
            "average_forgetting": average_forgetting(Rn),
            "backward_transfer": backward_transfer(Rn),
        }
        pd.DataFrame(Rn, index=[f"after_task{i}" for i in range(len(stream))], columns=cols) \
            .to_csv(out / "acc_matrix_ncm.csv", float_format="%.4f")
        (out / "metrics_ncm.json").write_text(json.dumps(ncm_metrics, indent=2), encoding="utf-8")
        print("\n== ĐÒN A — NCM-head (prototype trên feature sau memory, KHÔNG train lại)")
        print(f"  Linear-head : Acc {metrics['average_accuracy']:.4f} | Forget {metrics['average_forgetting']:.4f}")
        print(f"  NCM-head    : Acc {ncm_metrics['average_accuracy']:.4f} | Forget {ncm_metrics['average_forgetting']:.4f}")
        print(f"  Mốc NCM gốc : Acc 0.6933 | Forget 0.1000  (vượt được = head Linear là nút thắt)")

    print(f"  Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
