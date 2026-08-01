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
import hashlib
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def run_dir_name(cfg: dict, method_name: str) -> str:
    """Tên thư mục kết quả — nguồn duy nhất, dùng cho cả ghi lẫn resume (--skip-existing)."""
    name = f"{str(cfg['data']['name']).lower()}_{method_name}_seed{int(cfg.get('seed', 0))}"
    run_tag = str(cfg.get("log", {}).get("run_tag", "")).strip().lower()
    if run_tag:
        safe_tag = "".join(c for c in run_tag if c.isalnum() or c in "-_")
        if not safe_tag:
            raise ValueError("log.run_tag phải chứa ít nhất một ký tự chữ hoặc số")
        name += f"_{safe_tag}"
    opt = str(cfg.get("train", {}).get("optimizer", "adamw")).lower()
    if opt != "adamw":
        name += f"_{opt}"
    if opt == "m3":
        m3_cfg = dict(cfg.get("train", {}).get("m3", {}) or {})
        style = str(m3_cfg.get("beta_style", "delta")).lower()
        norm = str(m3_cfg.get("update_norm", "clip")).lower()
        frequency = int(m3_cfg.get("frequency", 16))
        alpha = float(m3_cfg.get("alpha", 0.5))
        lr = float(cfg.get("train", {}).get("lr", 3e-4))
        name += f"_{style}_{norm}_f{frequency}_lr{lr:g}"
        if alpha != 0.5:
            name += f"_a{alpha:g}"
        key_proj_eta = float(m3_cfg.get("key_proj_eta", 0.0))
        if key_proj_eta:
            name += f"_kp{key_proj_eta:g}"
    mem_cfg = cfg.get("memory") or {}
    if mem_cfg.get("enabled", False):  # 3 bậc reset A/B/C KHÔNG được ghi đè/skip lẫn nhau
        name += f"_r{str(mem_cfg.get('reset', 'image')).lower()}"
    if not bool(cfg.get("train", {}).get("optimizer_per_task", True)):
        name += "_optkeep"  # ablation "ký ức optimizer xuyên task" không đè run thường
    cms_cfg = cfg.get("cms") or {}
    if cms_cfg.get("enabled", False):
        periods = "-".join(str(t[1]) for t in cms_cfg.get("tiers", []))
        name += f"_{cms_cfg.get('order', 'late_slow')}_p{periods}"
    split_protocol = str(cfg.get("data", {}).get("split_protocol", "")).strip().lower()
    if split_protocol:
        safe_protocol = "".join(c for c in split_protocol if c.isalnum() or c in "-_")
        if not safe_protocol:
            raise ValueError("data.split_protocol phải chứa ít nhất một ký tự chữ hoặc số")
        name += f"_split{safe_protocol}"
    train_cfg = cfg.get("train", {}) or {}
    ncm_cfg = train_cfg.get("ncm") or {}
    adaptation = {
        "blend": ncm_cfg.get("blend"),
        "transport": ncm_cfg.get("transport"),
        "feature_distillation": train_cfg.get("feature_distillation"),
    }
    if any(value for value in adaptation.values()):
        digest = hashlib.sha1(
            json.dumps(adaptation, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:8]
        name += f"_ncmadapt{digest}"
    return name


def result_is_complete(out: pathlib.Path, cfg: dict) -> bool:
    required = ("metrics.json", "config.yaml", "acc_matrix.csv", "train_log.json")
    if not all((out / name).exists() for name in required):
        return False
    train_cfg = cfg.get("train", {})
    ncm_cfg = train_cfg.get("ncm")
    if isinstance(ncm_cfg, dict) and bool(ncm_cfg.get("enabled", True)):
        readouts = ncm_cfg.get(
            "readouts", ["online_current_task", "posthoc_full_seen_train"]
        )
        if isinstance(readouts, str):
            readouts = [readouts]
        expected = []
        if "online_current_task" in readouts:
            expected.extend(("metrics_ncm_online.json", "acc_matrix_ncm_online.csv"))
            blend_cfg = ncm_cfg.get("blend", {}) or {}
            if blend_cfg.get("enabled"):
                from uavcl.models.ncm_adaptation import gamma_key

                for gamma in blend_cfg.get("gammas", [1.0]):
                    key = gamma_key(float(gamma))
                    expected.extend(
                        (
                            f"metrics_ncm_blend_{key}.json",
                            f"acc_matrix_ncm_blend_{key}.csv",
                        )
                    )
            transport_cfg = ncm_cfg.get("transport", {}) or {}
            for index, candidate in enumerate(transport_cfg.get("candidates", []) or []):
                safe_name = "".join(
                    char if char.isalnum() else "_"
                    for char in str(candidate.get("name", f"candidate_{index}")).lower()
                ).strip("_")
                key = f"t{index}_{safe_name or 'candidate'}"
                expected.extend(
                    (
                        f"metrics_ncm_transport_{key}.json",
                        f"acc_matrix_ncm_transport_{key}.csv",
                    )
                )
        if "posthoc_full_seen_train" in readouts:
            expected.extend(("metrics_ncm_posthoc.json", "acc_matrix_ncm_posthoc.csv"))
        if bool(ncm_cfg.get("save_state", True)):
            expected.append("checkpoint.pt")
        return all((out / name).exists() for name in expected)
    if bool(train_cfg.get("eval_ncm_head", False)):
        return all((out / name).exists() for name in ("metrics_ncm.json", "acc_matrix_ncm.csv"))
    return True


def has_terminal_scientific_failure(out: pathlib.Path) -> bool:
    failure_path = out / "failure.json"
    if not failure_path.exists():
        return False
    try:
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return failure.get("exception") == "FloatingPointError"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="đường dẫn file yaml (vd configs/g1_eurosat.yaml)")
    ap.add_argument("--method", default=None, help="finetune | ewc (mặc định lấy từ config)")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="k.sub=v",
                    help="override config, vd: train.lr=1e-4 data.num_tasks=5")
    ap.add_argument("--skip-existing", action="store_true",
                    help="đã có metrics.json cho run này thì bỏ qua (resume cho run_all)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="chạy lại cả run đã dừng vì NaN/state-health FloatingPointError")
    args = ap.parse_args()

    from uavcl.utils.config import apply_overrides, load_config, save_config
    from uavcl.utils.seed import seed_everything

    cfg = apply_overrides(load_config(args.config), args.overrides)
    method_name = (args.method or cfg.get("train", {}).get("method", "finetune")).lower()
    seed = int(cfg.get("seed", 0))

    out = pathlib.Path(cfg.get("log", {}).get("dir", "./artifacts")) / "results" / run_dir_name(cfg, method_name)
    if args.skip_existing and result_is_complete(out, cfg):
        print(f"[SKIP] {out.name} — đã có kết quả.")
        return 0
    if args.skip_existing and not args.retry_failed and has_terminal_scientific_failure(out):
        print(f"[SKIP FAILED] {out.name} — FloatingPointError đã được ghi nhận.")
        return 2
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
    mem_cfg = cfg.get("memory", {}) or {}
    cms_on = bool((cfg.get("cms") or {}).get("enabled", False))
    if mem_cfg.get("enabled", False) and cms_on:
        # G4 — HOPE: Titans (tầng nhanh) + backbone-CMS (tầng trung/chậm)
        from uavcl.models.hope import HOPEClassifier

        model = HOPEClassifier(backbone, feat_dim, source.num_classes, mem_cfg).to(device)
        if method_name != "hope":
            print(f"[WARN] memory+cms cùng bật + method '{method_name}' — thường dùng --method hope")
        print(f"[hope] seq={model.seq_mode} reset={model.reset_mode} chunk={model.memory.chunk_size}")
    elif mem_cfg.get("enabled", False):
        from uavcl.models.titans_head import TitansClassifier

        model = TitansClassifier(backbone, feat_dim, source.num_classes, mem_cfg).to(device)
        if method_name not in ("titans", "finetune"):
            print(f"[WARN] memory.enabled + method '{method_name}' — thường dùng --method titans")
        print(f"[titans] seq={model.seq_mode} reset={model.reset_mode} chunk={model.memory.chunk_size}")
    elif method_name == "ncm":
        from uavcl.models.ncm import NCMClassifier

        model = NCMClassifier(backbone, feat_dim, source.num_classes).to(device)
    else:
        model = ContinualClassifier(backbone, feat_dim, source.num_classes).to(device)
    method = build_method(method_name, cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "failure.json").unlink(missing_ok=True)
    save_config(cfg, out / "config.yaml")
    progress_path = out / "progress_checkpoint.pt"
    cfg["train"]["_progress_checkpoint_path"] = str(progress_path)

    # --- run ---
    cms_cfg = cfg.get("cms") or {}
    if cms_cfg.get("enabled", False):
        cfg["train"]["cms"] = cms_cfg  # engine đọc từ train_cfg -> dùng CMSOptimizer (G3)
    try:
        R, log = run_continual(model, method, stream, loaders, device, cfg["train"])
    except Exception as exc:
        failure = {
            "status": "failed",
            "exception": type(exc).__name__,
            "message": str(exc),
            "runtime_sec": round(time.time() - t0, 1),
        }
        (out / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise

    # --- metrics + save ---
    metrics = {
        "dataset": source.name,
        "method": method_name,
        "seed": seed,
        "num_tasks": len(stream),
        "backbone": cfg["backbone"]["name"],
        "optimizer": str(cfg["train"].get("optimizer", "adamw")).lower(),
        "lr": float(cfg["train"].get("lr", 3e-4)),
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
        "device": str(device),
        "torch_version": str(torch.__version__),
    }
    if device.type == "cuda":
        metrics["cuda_device"] = torch.cuda.get_device_name(device)
    try:
        metrics["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=pathlib.Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        metrics["git_commit"] = None
    if mem_cfg.get("enabled", False):
        metrics["memory_reset"] = str(mem_cfg.get("reset", "image")).lower()
    if cms_cfg.get("enabled", False):
        metrics["cms_order"] = cms_cfg.get("order", "late_slow")
        metrics["cms_periods"] = "-".join(str(t[1]) for t in cms_cfg.get("tiers", []))
    if str(cfg["train"].get("optimizer", "adamw")).lower() == "m3":
        metrics["m3"] = dict(cfg["train"].get("m3", {}) or {})
    metrics["grad_clip_norm"] = cfg["train"].get("grad_clip_norm")
    cols = [f"task{j}" for j in range(len(stream))]
    pd.DataFrame(R, index=[f"after_task{i}" for i in range(len(stream))], columns=cols) \
        .to_csv(out / "acc_matrix.csv", float_format="%.4f")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    serializable_log = dict(log)
    for key, value in list(serializable_log.items()):
        if key.startswith("ncm_") and isinstance(value, np.ndarray):
            serializable_log[key] = value.tolist()
    (out / "train_log.json").write_text(
        json.dumps(serializable_log, indent=2), encoding="utf-8"
    )
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

    def save_ncm_readout(matrix_key: str, readout: str, stem: str, revisit: bool) -> dict | None:
        if matrix_key not in log:
            return None
        matrix = np.asarray(log[matrix_key])
        protocol = dict(log.get("ncm_protocol", {}))
        readout_metrics = {
            "readout": readout,
            "revisit_old_train": revisit,
            "feature_protocol": protocol.get("feature_protocol"),
            "prototype_loader": protocol.get("prototype_loader"),
            "average_accuracy": average_accuracy(matrix),
            "average_forgetting": average_forgetting(matrix),
            "backward_transfer": backward_transfer(matrix),
        }
        pd.DataFrame(
            matrix,
            index=[f"after_task{i}" for i in range(len(stream))],
            columns=cols,
        ).to_csv(out / f"acc_matrix_{stem}.csv", float_format="%.4f")
        (out / f"metrics_{stem}.json").write_text(
            json.dumps(readout_metrics, indent=2), encoding="utf-8"
        )
        return readout_metrics

    online_metrics = save_ncm_readout(
        "ncm_online_R", "ncm_online_current_task", "ncm_online", False
    )
    blend_metrics = {}
    from uavcl.models.ncm_adaptation import gamma_key

    protocol_gammas = log.get("ncm_protocol", {}).get("blend_gammas", [])
    for gamma in protocol_gammas:
        key = gamma_key(float(gamma))
        saved = save_ncm_readout(
            f"ncm_blend_R_{key}",
            f"ncm_online_blend_gamma_{float(gamma):g}",
            f"ncm_blend_{key}",
            False,
        )
        if saved is not None:
            saved["gamma"] = float(gamma)
            saved["transport"] = log.get("ncm_protocol", {}).get("transport")
            (out / f"metrics_ncm_blend_{key}.json").write_text(
                json.dumps(saved, indent=2), encoding="utf-8"
            )
            blend_metrics[key] = saved
    transport_metrics = {}
    transport_cfg = log.get("ncm_protocol", {}).get("transport", {}) or {}
    for index, candidate in enumerate(transport_cfg.get("candidates", []) or []):
        safe_name = "".join(
            char if char.isalnum() else "_"
            for char in str(candidate.get("name", f"candidate_{index}")).lower()
        ).strip("_")
        key = f"t{index}_{safe_name or 'candidate'}"
        saved = save_ncm_readout(
            f"ncm_transport_R_{key}",
            f"ncm_transport_candidate_{key}",
            f"ncm_transport_{key}",
            False,
        )
        if saved is not None:
            saved["candidate"] = candidate
            saved["gamma"] = float(transport_cfg.get("screen_gamma", 1.0))
            (out / f"metrics_ncm_transport_{key}.json").write_text(
                json.dumps(saved, indent=2), encoding="utf-8"
            )
            transport_metrics[key] = saved
    posthoc_metrics = save_ncm_readout(
        "ncm_posthoc_R", "ncm_posthoc_full_seen_train", "ncm_posthoc", True
    )

    # Backward-compatible files for historical train.eval_ncm_head campaigns.
    if "ncm_R" in log:
        Rn = np.asarray(log["ncm_R"])
        legacy_metrics = {
            "readout": "ncm_head_posthoc",
            "average_accuracy": average_accuracy(Rn),
            "average_forgetting": average_forgetting(Rn),
            "backward_transfer": backward_transfer(Rn),
        }
        pd.DataFrame(
            Rn,
            index=[f"after_task{i}" for i in range(len(stream))],
            columns=cols,
        ).to_csv(out / "acc_matrix_ncm.csv", float_format="%.4f")
        (out / "metrics_ncm.json").write_text(
            json.dumps(legacy_metrics, indent=2), encoding="utf-8"
        )

    if online_metrics or posthoc_metrics:
        print("\n== NCM feature readouts")
        print(
            f"  Linear              : Acc {metrics['average_accuracy']:.4f} | "
            f"Forget {metrics['average_forgetting']:.4f}"
        )
        if online_metrics:
            print(
                f"  NCM online          : Acc {online_metrics['average_accuracy']:.4f} | "
                f"Forget {online_metrics['average_forgetting']:.4f}"
            )
        if posthoc_metrics:
            print(
                f"  NCM post-hoc oracle : Acc {posthoc_metrics['average_accuracy']:.4f} | "
                f"Forget {posthoc_metrics['average_forgetting']:.4f}"
            )
        for key, item in blend_metrics.items():
            if key == gamma_key(1.0):
                continue
            print(
                f"  NCM blend {item['gamma']:g}       : Acc {item['average_accuracy']:.4f} | "
                f"Forget {item['average_forgetting']:.4f}"
            )

    configured_ncm = cfg.get("train", {}).get("ncm")
    if (
        isinstance(configured_ncm, dict)
        and bool(configured_ncm.get("enabled", True))
        and bool(configured_ncm.get("save_state", True))
    ):
        from uavcl.checkpoint import save_inference_checkpoint

        save_inference_checkpoint(
            out / "checkpoint.pt",
            model,
            config=cfg,
            seen_classes=sorted({class_id for spec in stream for class_id in spec.classes}),
            git_commit=metrics.get("git_commit"),
        )
        print("  checkpoint.pt đã lưu (model + Titans state + online NCM)")

    progress_path.unlink(missing_ok=True)
    print(f"  Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
