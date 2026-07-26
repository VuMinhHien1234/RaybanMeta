#!/usr/bin/env python3
"""Compare the practical M3 baseline with NL.pdf Algorithm 1 variants."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys
from collections import defaultdict

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "artifacts" / "m3_paper_study"
REPORT_PATH = ROOT.parent / "An" / "KET_QUA_SO_SANH_M3_PRACTICAL_VS_PAPER.md"

VARIANTS = {
    "p0": {
        "label": "M3-delta-approx-clip",
        "style": "delta",
        "norm": "clip",
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "lrs": (3e-3, 5e-3),
    },
    "p1": {
        "label": "M3-paper-stabilized",
        "style": "paper",
        "norm": "clip",
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "lrs": (1e-4, 3e-4, 1e-3, 3e-3, 5e-3),
    },
    "p2": {
        "label": "M3-paper-strict",
        "style": "paper",
        "norm": "none",
        "weight_decay": 0.0,
        "grad_clip": None,
        "lrs": (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
    },
    "p3": {
        "label": "M3-EMA-clip",
        "style": "ema",
        "norm": "clip",
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "lrs": (),
    },
}


def _tag(variant: str) -> str:
    return f"m3-paper-study-{variant}"


def _overrides(variant: str, seed: int, lr: float) -> list[str]:
    cfg = VARIANTS[variant]
    grad_clip = "null" if cfg["grad_clip"] is None else str(cfg["grad_clip"])
    return [
        f"experiment.tag={_tag(variant)}",
        f"seed={seed}",
        "train.optimizer=m3",
        "train.optimizer_per_task=false",
        "train.eval_validation=true",
        f"train.lr={lr}",
        f"train.weight_decay={cfg['weight_decay']}",
        f"train.grad_clip_norm={grad_clip}",
        f"train.m3.beta_style={cfg['style']}",
        f"train.m3.update_norm={cfg['norm']}",
        "train.m3.alpha=0.5",
        "train.m3.frequency=16",
        "train.m3.diagnostics=true",
        "train.m3.diagnostics_first_n=50",
        "train.m3.paper_timing=next_chunk",
    ]


def _run(
    config: pathlib.Path,
    log_dir: pathlib.Path,
    overrides: list[str],
    *,
    dry_run: bool,
    skip_existing: bool,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_g1.py"),
        "--config",
        str(config),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    cmd.extend(["--set", *overrides, f"log.dir={log_dir}"])
    print("\n$", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def _result_rows(log_dir: pathlib.Path, dataset: str | None = None) -> list[dict]:
    rows = []
    for metrics_path in sorted((log_dir / "results").glob("*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = yaml.safe_load(metrics_path.with_name("config.yaml").read_text(encoding="utf-8"))
            tag = str((config.get("experiment") or {}).get("tag", ""))
            if not tag.startswith("m3-paper-study-"):
                continue
            if dataset and str(config["data"]["name"]).lower() != dataset:
                continue
            train_log_path = metrics_path.with_name("train_log.json")
            if not train_log_path.exists():
                continue
            train_log = json.loads(train_log_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
            continue
        variant = tag.rsplit("-", 1)[-1]
        finite_state = all(bool(v) for v in train_log.get("state_finite", {}).values())
        diagnostics = train_log.get("optimizer_diagnostics", {})
        nonfinite = sum(int(v.get("nonfinite_count", 0)) for v in diagnostics.values())
        rows.append({
            "path": metrics_path.parent,
            "variant": variant,
            "seed": int(config.get("seed", 0)),
            "lr": float(config["train"]["lr"]),
            "metrics": metrics,
            "config": config,
            "train_log": train_log,
            "stable": finite_state and nonfinite == 0,
        })
    return rows


def _best(log_dir: pathlib.Path, dataset: str, variant: str, seed: int = 0) -> dict:
    candidates = [
        row for row in _result_rows(log_dir, dataset)
        if row["variant"] == variant and row["seed"] == seed and row["stable"]
        and row["metrics"].get("validation_average_accuracy") is not None
        and (variant != "p2" or _paper_strict_scale_ok(log_dir, row))
    ]
    if not candidates:
        raise RuntimeError(
            f"Không có run validation hữu hạn cho {dataset}/{variant}/seed{seed}."
        )
    return max(
        candidates,
        key=lambda row: (
            float(row["metrics"]["validation_average_accuracy"]),
            -float(row["metrics"].get("validation_average_forgetting") or 0.0),
        ),
    )


def _max_fast_drift(row: dict) -> float:
    values = []
    for task in row["train_log"].get("method_diagnostics", {}).values():
        fast = task.get("tier_drift", {}).get("fast", {})
        value = fast.get("relative_parameter_drift")
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return max(values, default=math.inf)


def _max_loss(row: dict) -> float:
    values = [
        float(loss)
        for task_losses in row["train_log"].get("train_loss", {}).values()
        for loss in task_losses
        if math.isfinite(float(loss))
    ]
    return max(values, default=math.inf)


def _paper_strict_scale_ok(log_dir: pathlib.Path, row: dict) -> bool:
    baselines = [
        candidate for candidate in _result_rows(log_dir, str(row["config"]["data"]["name"]).lower())
        if candidate["variant"] == "p0"
        and candidate["seed"] == row["seed"]
        and candidate["stable"]
    ]
    if not baselines:
        return False
    baseline_drift = min(_max_fast_drift(candidate) for candidate in baselines)
    baseline_loss = min(_max_loss(candidate) for candidate in baselines)
    drift_ok = _max_fast_drift(row) <= 100.0 * max(baseline_drift, 1e-12)
    loss_ok = _max_loss(row) <= 10.0 * max(baseline_loss, 1e-12)
    return drift_ok and loss_ok


def _best_or_none(log_dir: pathlib.Path, dataset: str, variant: str, seed: int = 0):
    try:
        return _best(log_dir, dataset, variant, seed)
    except RuntimeError:
        return None


def _three_seed_gate(log_dir: pathlib.Path, variant: str, lr: float) -> None:
    rows = [
        row for row in _result_rows(log_dir, "eurosat")
        if row["variant"] == variant and math.isclose(row["lr"], lr) and row["stable"]
        and (variant != "p2" or _paper_strict_scale_ok(log_dir, row))
    ]
    seeds = {row["seed"] for row in rows}
    missing = sorted({0, 1, 2} - seeds)
    if missing:
        raise RuntimeError(
            f"{variant} lr={lr:g} chưa qua stability gate EuroSAT; thiếu seed {missing}."
        )


def _run_grid(args, config: pathlib.Path, variants: tuple[str, ...], seeds=(0,)) -> None:
    for variant in variants:
        for lr in VARIANTS[variant]["lrs"]:
            for seed in seeds:
                _run(
                    config,
                    args.log_dir,
                    ["data.num_workers=0", *_overrides(variant, seed, lr)],
                    dry_run=args.dry_run,
                    skip_existing=args.skip_existing,
                )


def _summarize(log_dir: pathlib.Path) -> tuple[dict, str]:
    rows = _result_rows(log_dir)
    grouped = defaultdict(list)
    for row in rows:
        key = (
            str(row["config"]["data"]["name"]).lower(),
            row["variant"],
            row["lr"],
        )
        grouped[key].append(row)

    summary_rows = []
    for (dataset, variant, lr), group in sorted(grouped.items()):
        stable = [row for row in group if row["stable"]]
        accs = [float(row["metrics"]["average_accuracy"]) for row in stable]
        forgets = [float(row["metrics"]["average_forgetting"]) for row in stable]
        bwts = [float(row["metrics"]["backward_transfer"]) for row in stable]
        fwts = [
            float(row["metrics"]["forward_transfer"])
            for row in stable if row["metrics"].get("forward_transfer") is not None
        ]
        runtimes = [float(row["metrics"]["runtime_sec"]) for row in stable]
        summary_rows.append({
            "dataset": dataset,
            "variant": variant,
            "label": VARIANTS.get(variant, {}).get("label", variant),
            "lr": lr,
            "seeds": sorted(row["seed"] for row in stable),
            "runs": len(group),
            "stable_runs": len(stable),
            "accuracy_mean": sum(accs) / len(accs) if accs else None,
            "accuracy_std": statistics.stdev(accs) if len(accs) > 1 else 0.0 if accs else None,
            "forgetting_mean": sum(forgets) / len(forgets) if forgets else None,
            "forgetting_std": (
                statistics.stdev(forgets) if len(forgets) > 1 else 0.0 if forgets else None
            ),
            "bwt_mean": sum(bwts) / len(bwts) if bwts else None,
            "fwt_mean": sum(fwts) / len(fwts) if fwts else None,
            "runtime_mean_sec": sum(runtimes) / len(runtimes) if runtimes else None,
        })

    failures = []
    for path in sorted((log_dir / "results").glob("*/failure.json")):
        try:
            failures.append({"run": path.parent.name, **json.loads(path.read_text())})
        except (OSError, ValueError, TypeError):
            continue
    eurosat_seed_sets = [
        set(row["seeds"]) for row in summary_rows if row["dataset"] == "eurosat"
    ]
    resisc_rows = [row for row in summary_rows if row["dataset"] == "resisc45"]
    study_complete = (
        any({0, 1, 2} <= seeds for seeds in eurosat_seed_sets)
        and any({0, 1, 2} <= set(row["seeds"]) for row in resisc_rows)
    )
    lines = [
        "# Kết quả so sánh M3 practical và M3 paper",
        "",
        "Báo cáo này được tạo tự động bởi `scripts/run_m3_paper_study.py summarize`.",
        "LR được chọn bằng validation; test metrics chỉ dùng để báo cáo sau khi khóa cấu hình.",
        "",
        f"Trạng thái: **{'đã đủ EuroSAT + RESISC45' if study_complete else 'đang triển khai/chưa đủ run dataset thật'}**.",
        "",
        "## Biến thể",
        "",
        "- P0: `M3-delta-approx-clip`, baseline chính không đổi semantics.",
        "- P1: `M3-paper-stabilized`, paper accumulation/timing + cơ chế ổn định của project.",
        "- P2: `M3-paper-strict`, paper accumulation/timing không weight decay/gradient clip/update clip.",
        "- P3: `M3-EMA-clip`, đối chứng chẩn đoán.",
        "",
        "## Kết quả hiện có",
        "",
        "| Dataset | Variant | LR | Seeds hữu hạn | Accuracy mean +/- sd | Forgetting mean +/- sd | BWT | FWT | Runtime (s) |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        acc = (
            "n/a" if row["accuracy_mean"] is None
            else f"{row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}"
        )
        forgetting = (
            "n/a" if row["forgetting_mean"] is None
            else f"{row['forgetting_mean']:.4f} +/- {row['forgetting_std']:.4f}"
        )
        bwt = "n/a" if row["bwt_mean"] is None else f"{row['bwt_mean']:.4f}"
        fwt = "n/a" if row["fwt_mean"] is None else f"{row['fwt_mean']:.4f}"
        runtime = (
            "n/a" if row["runtime_mean_sec"] is None else f"{row['runtime_mean_sec']:.1f}"
        )
        seeds = ",".join(map(str, row["seeds"])) or "-"
        lines.append(
            f"| {row['dataset']} | {row['label']} | {row['lr']:g} | "
            f"{seeds} | {acc} | {forgetting} | {bwt} | {fwt} | {runtime} |"
        )
    lines.extend([
        "",
        "## Kiểm chứng triển khai",
        "",
        "- Unit/integration tests: công thức một bước, chunk timing, không bias-correct V, strict không clip, CMS clock và baseline regression.",
        "- P1/P2 dùng `paper_timing=next_chunk`; mode `legacy_boundary` chỉ giữ để tái lập artifact cũ.",
        "- Run identity dùng `experiment.tag`; P1/P2 không thể ghi đè nhau.",
        "- Mỗi run lưu config, git/environment metadata, validation metrics, failure record và optimizer diagnostics.",
        "- Diagnostics gồm gradient trước/sau clip, M1/M2/V/O1/O2, denominator, raw/post update, clip rate và drift theo CMS tier.",
        "",
        f"Run thất bại đã ghi nhận: **{len(failures)}**.",
        "",
        "## Lưu ý",
        "",
        "- `M3-paper-stabilized` giữ weight decay và hai lớp clipping của project.",
        "- `M3-paper-strict` bỏ weight decay, global gradient clipping và update clipping.",
        "- Mode paper là bản bám pseudocode với policy tensor 1D của project, không phải code gốc 100% của tác giả.",
        "- Không kết luận variant thắng từ kết quả synthetic smoke; cần hoàn tất EuroSAT và stability gate trước RESISC45.",
        "",
        "## Lệnh tiếp tục",
        "",
        "```bash",
        "python3 scripts/run_m3_paper_study.py eurosat-sweep",
        "python3 scripts/run_m3_paper_study.py eurosat-replicate",
        "python3 scripts/run_m3_paper_study.py resisc45",
        "python3 scripts/run_m3_paper_study.py summarize",
        "```",
        "",
        "Runner mặc định `--skip-existing`, nên có thể chạy lại cùng lệnh để resume.",
        "",
    ])
    return {"study_complete": study_complete, "rows": summary_rows, "failures": failures}, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("audit", "smoke", "eurosat-sweep", "eurosat-replicate", "resisc45", "summarize"),
    )
    parser.add_argument("--log-dir", type=pathlib.Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    eurosat = ROOT / "configs" / "g4_hope_eurosat.yaml"
    resisc45 = ROOT / "configs" / "g4_hope_resisc45.yaml"

    if args.phase == "audit":
        cmd = [
            sys.executable, "-m", "pytest", "-q",
            "tests/test_m3.py", "tests/test_g3_cms.py", "tests/test_run_logging.py",
        ]
        print("$", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
        return 0

    if args.phase == "smoke":
        common = [
            "device=cpu",
            "data.name=synthetic",
            "data.num_classes=4",
            "data.train_per_class=4",
            "data.val_per_class=2",
            "data.test_per_class=2",
            "data.num_tasks=2",
            "data.image_size=224",
            "data.batch_size=4",
            "data.num_workers=0",
            "backbone.pretrained=false",
            "train.epochs_per_task=1",
            "train.eval_future=false",
        ]
        for variant, lr in (("p0", 3e-3), ("p1", 1e-3), ("p2", 1e-6)):
            _run(
                eurosat,
                args.log_dir,
                [*common, *_overrides(variant, 0, lr)],
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )
        return 0

    if args.phase == "eurosat-sweep":
        _run_grid(args, eurosat, ("p0", "p1", "p2"))
        if not args.dry_run:
            best_p1 = _best(args.log_dir, "eurosat", "p1")
            _run(
                eurosat,
                args.log_dir,
                ["data.num_workers=0", *_overrides("p3", 0, best_p1["lr"])],
                dry_run=False,
                skip_existing=args.skip_existing,
            )
        return 0

    if args.phase == "eurosat-replicate":
        for variant in ("p0", "p1", "p2"):
            if variant == "p0":
                lrs = VARIANTS["p0"]["lrs"]
            else:
                selected = _best_or_none(args.log_dir, "eurosat", variant)
                if selected is None:
                    if variant == "p2":
                        print("[SKIP] P2 không có run seed 0 ổn định; không replicate.")
                        continue
                    raise RuntimeError("P1 chưa có run seed 0 ổn định.")
                lrs = (selected["lr"],)
            for lr in lrs:
                for seed in (0, 1, 2):
                    _run(
                        eurosat,
                        args.log_dir,
                        ["data.num_workers=0", *_overrides(variant, seed, lr)],
                        dry_run=args.dry_run,
                        skip_existing=args.skip_existing,
                    )
        return 0

    if args.phase == "resisc45":
        protocol = str(yaml.safe_load(resisc45.read_text())["data"].get("split_protocol", ""))
        if protocol != "combined31500_v2":
            raise RuntimeError("RESISC45 phải dùng split_protocol=combined31500_v2.")
        paper_candidates = []
        for variant in ("p1", "p2"):
            selected = _best_or_none(args.log_dir, "eurosat", variant)
            if selected is None:
                if variant == "p2":
                    print("[SKIP] P2 không có cấu hình EuroSAT ổn định.")
                    continue
                raise RuntimeError("P1 chưa có cấu hình EuroSAT ổn định.")
            try:
                _three_seed_gate(args.log_dir, variant, selected["lr"])
            except RuntimeError:
                if variant == "p2":
                    print("[SKIP] P2 chưa qua stability gate 3 seed EuroSAT.")
                    continue
                raise
            paper_candidates.append(selected)
        if not paper_candidates:
            raise RuntimeError("Không có paper variant nào qua stability gate EuroSAT.")
        winner = max(
            paper_candidates,
            key=lambda row: float(row["metrics"]["validation_average_accuracy"]),
        )
        jobs = [("p0", 5e-3), (winner["variant"], winner["lr"])]
        for variant, lr in jobs:
            for seed in (0, 1, 2):
                _run(
                    resisc45,
                    args.log_dir,
                    ["data.num_workers=0", *_overrides(variant, seed, lr)],
                    dry_run=args.dry_run,
                    skip_existing=args.skip_existing,
                )
        return 0

    summary, markdown = _summarize(args.log_dir)
    summary_dir = args.log_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (summary_dir / "summary.md").write_text(markdown, encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(markdown, encoding="utf-8")
    print(f"Saved {summary_dir / 'summary.md'}")
    print(f"Saved {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
