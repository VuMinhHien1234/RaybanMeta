#!/usr/bin/env python3
"""Run the gated Task-4 Titans + legacy/improved M3 comparison."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from typing import Iterable

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEGACY_CONFIG = ROOT / "configs" / "g2_titans_resisc45_selfmod_m3_legacy.yaml"
IMPROVED_CONFIG = ROOT / "configs" / "g2_titans_resisc45_selfmod_m3_improved.yaml"
LRS = (1e-4, 3e-4, 1e-3, 3e-3, 5e-3)
SEEDS = (0, 1, 2)
STRESS_SEED = 1
MAX_STATE_NORM = 10000.0
MAX_STATE_GROWTH = 10.0


def _run_g1_module():
    path = ROOT / "scripts" / "run_g1.py"
    spec = importlib.util.spec_from_file_location("titan_m3_run_g1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN_G1 = _run_g1_module()


def _flatten(value, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(child, child_prefix))
        return out
    return {prefix: value}


def validate_controlled_configs() -> None:
    legacy = yaml.safe_load(LEGACY_CONFIG.read_text(encoding="utf-8"))
    improved = yaml.safe_load(IMPROVED_CONFIG.read_text(encoding="utf-8"))
    left = _flatten(legacy)
    right = _flatten(improved)
    allowed = {
        "train.grad_clip_norm",
        "train.optimizer_per_task",
        "train.m3.update_norm",
        "log.dir",
    }
    differences = {
        key: (left.get(key, "<missing>"), right.get(key, "<missing>"))
        for key in sorted(set(left) | set(right))
        if left.get(key, "<missing>") != right.get(key, "<missing>")
    }
    unexpected = {key: value for key, value in differences.items() if key not in allowed}
    if unexpected:
        raise RuntimeError(f"Legacy/improved config lệch ngoài whitelist: {unexpected}")
    if set(differences) != allowed:
        raise RuntimeError(f"Config comparison thiếu/đổi biến dự kiến: {differences}")


def _final_config(config_path: pathlib.Path, overrides: list[str]) -> dict:
    from uavcl.utils.config import apply_overrides, load_config

    return apply_overrides(load_config(config_path), list(overrides))


def _result_dir(
    config_path: pathlib.Path, artifact_root: pathlib.Path, tag: str, overrides: list[str]
) -> pathlib.Path:
    cfg = _final_config(
        config_path,
        [*overrides, f"log.dir={artifact_root}", f"log.run_tag={tag}"],
    )
    method = str(cfg.get("train", {}).get("method", "titans")).lower()
    return artifact_root / "results" / RUN_G1.run_dir_name(cfg, method)


def _common_overrides(seed: int, lr: float) -> list[str]:
    return [
        f"seed={seed}",
        f"train.lr={lr}",
        "train.eval_ncm_head=true",
        f"train.max_state_norm={MAX_STATE_NORM:g}",
        f"train.max_state_norm_growth={MAX_STATE_GROWTH:g}",
    ]


def _stream_process(cmd: list[str], cwd: pathlib.Path, log_path: pathlib.Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {' '.join(cmd)}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return proc.wait()


def run_experiment(
    config_path: pathlib.Path,
    artifact_root: pathlib.Path,
    tag: str,
    seed: int,
    lr: float,
    extra_overrides: Iterable[str] = (),
    *,
    dry_run: bool = False,
) -> bool:
    overrides = [*_common_overrides(seed, lr), *extra_overrides]
    out = _result_dir(config_path, artifact_root, tag, overrides)
    metrics_path = out / "metrics.json"
    failure_path = out / "failure.json"
    label = out.name
    campaign_failure_path = artifact_root / "campaign_failures" / f"{label}.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_g1.py"),
        "--config",
        str(config_path),
        "--skip-existing",
        "--set",
        *overrides,
        f"log.dir={artifact_root}",
        f"log.run_tag={tag}",
    ]
    print("\n$", " ".join(cmd), flush=True)
    if dry_run:
        return True
    final_cfg = _final_config(
        config_path,
        [*overrides, f"log.dir={artifact_root}", f"log.run_tag={tag}"],
    )
    if RUN_G1.result_is_complete(out, final_cfg) and not failure_path.exists():
        print(f"[SKIP VALID] {label}", flush=True)
        return True

    code = _stream_process(cmd, ROOT, artifact_root / "campaign_logs" / f"{label}.log")
    if code == 0 and RUN_G1.result_is_complete(out, final_cfg):
        campaign_failure_path.unlink(missing_ok=True)
        print(f"[DONE] {label}", flush=True)
        return True

    failure_dir = artifact_root / "campaign_failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "run": label,
        "returncode": code,
        "command": cmd,
        "result_dir": str(out),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    campaign_failure_path.write_text(
        json.dumps(failure, indent=2), encoding="utf-8"
    )
    print(f"[FAILED] {label} (exit={code})", flush=True)
    return False


def _run_aux(cmd: list[str], artifact_root: pathlib.Path, name: str, dry_run: bool) -> None:
    print("\n$", " ".join(cmd), flush=True)
    if dry_run:
        return
    code = _stream_process(cmd, ROOT, artifact_root / "campaign_logs" / f"{name}.log")
    if code != 0:
        raise RuntimeError(f"{name} thất bại với exit={code}")


def phase_check(artifact_root: pathlib.Path, dry_run: bool, skip_dataset_check: bool) -> None:
    validate_controlled_configs()
    _run_aux([sys.executable, "scripts/check_env.py"], artifact_root, "check_env", dry_run)
    _run_aux([sys.executable, "-m", "pytest", "-q"], artifact_root, "pytest", dry_run)
    if dry_run or skip_dataset_check:
        return

    sys.path.insert(0, str(ROOT / "src"))
    from uavcl.data import get_source

    cfg = yaml.safe_load(IMPROVED_CONFIG.read_text(encoding="utf-8"))
    source = get_source(cfg["data"])
    counts = defaultdict(int)
    total = 0
    for split in source.splits.values():
        total += len(split.labels)
        for label in split.labels:
            counts[int(label)] += 1
    if total != 31500 or set(counts.values()) != {700}:
        raise RuntimeError(
            f"RESISC45 protocol sai: total={total}, class_counts={dict(sorted(counts.items()))}"
        )
    print("[DATA OK] RESISC45 31,500 samples, 700/class", flush=True)


def _smoke_overrides() -> list[str]:
    return [
        "data.name=synthetic",
        "data.split_protocol=synthetic_smoke",
        "data.num_classes=6",
        "data.train_per_class=8",
        "data.val_per_class=4",
        "data.test_per_class=4",
        "data.num_tasks=3",
        "data.image_size=32",
        "data.batch_size=8",
        "data.num_workers=0",
        "backbone.name=tinycnn",
        "backbone.pretrained=false",
        "memory.chunk_size=8",
        "train.epochs_per_task=1",
        "train.eval_future=false",
        "train.eval_ncm_head=false",
    ]


def phase_smoke(artifact_root: pathlib.Path, dry_run: bool) -> None:
    common = _smoke_overrides()
    run_experiment(
        LEGACY_CONFIG, artifact_root, "smoke-legacy", 0, 1e-3, common, dry_run=dry_run
    )
    run_experiment(
        IMPROVED_CONFIG, artifact_root, "smoke-improved", 0, 1e-3, common, dry_run=dry_run
    )


def phase_stress(artifact_root: pathlib.Path, dry_run: bool) -> None:
    run_experiment(LEGACY_CONFIG, artifact_root, "legacy", STRESS_SEED, 1e-3, dry_run=dry_run)
    run_experiment(
        IMPROVED_CONFIG, artifact_root, "improved", STRESS_SEED, 1e-3, dry_run=dry_run
    )


def phase_sweep(artifact_root: pathlib.Path, dry_run: bool) -> None:
    for lr in LRS:
        run_experiment(
            IMPROVED_CONFIG, artifact_root, "improved", STRESS_SEED, lr, dry_run=dry_run
        )


def _load_result(path: pathlib.Path) -> dict:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    train_log = json.loads((path / "train_log.json").read_text(encoding="utf-8"))
    norms = [float(value) for value in train_log.get("state_norm", {}).values()]
    finite = [bool(value) for value in train_log.get("state_finite", {}).values()]
    ncm_path = path / "metrics_ncm.json"
    ncm = json.loads(ncm_path.read_text(encoding="utf-8")) if ncm_path.exists() else {}
    max_growth = 1.0
    for previous, current in zip(norms, norms[1:]):
        if previous > 0.0:
            max_growth = max(max_growth, current / previous)
    valid = (
        bool(norms)
        and all(math.isfinite(value) for value in norms)
        and (not finite or all(finite))
        and max(norms) < MAX_STATE_NORM
        and max_growth <= MAX_STATE_GROWTH
    )
    return {
        "path": path,
        "tag": str(cfg.get("log", {}).get("run_tag", "")),
        "seed": int(cfg.get("seed", 0)),
        "lr": float(cfg["train"]["lr"]),
        "accuracy": float(metrics["average_accuracy"]),
        "forgetting": float(metrics["average_forgetting"]),
        "bwt": float(metrics["backward_transfer"]),
        "ncm_accuracy": float(ncm["average_accuracy"]) if ncm else None,
        "ncm_forgetting": float(ncm["average_forgetting"]) if ncm else None,
        "max_state_norm": max(norms) if norms else math.inf,
        "max_state_growth": max_growth,
        "valid": valid,
        "runtime_sec": float(metrics.get("runtime_sec", 0.0)),
        "device": metrics.get("device"),
        "git_commit": metrics.get("git_commit"),
    }


def collect_results(artifact_root: pathlib.Path) -> list[dict]:
    rows = []
    for metrics_path in sorted((artifact_root / "results").glob("*/metrics.json")):
        result_dir = metrics_path.parent
        if not (result_dir / "config.yaml").exists() or not (result_dir / "train_log.json").exists():
            continue
        rows.append(_load_result(result_dir))
    return rows


def select_improved_lr(artifact_root: pathlib.Path) -> float:
    candidates = [
        row
        for row in collect_results(artifact_root)
        if row["tag"] == "improved"
        and row["seed"] == STRESS_SEED
        and row["lr"] in LRS
        and row["valid"]
    ]
    if not candidates:
        raise RuntimeError("Không có LR M3 mới hữu hạn trên stress seed 1")
    winner = max(
        candidates,
        key=lambda row: (
            row["accuracy"] - max(row["forgetting"], 0.0),
            row["accuracy"],
            -row["max_state_norm"],
        ),
    )
    print(
        f"[SELECTED LR] {winner['lr']:g} | acc={winner['accuracy']:.4f} "
        f"fgt={winner['forgetting']:.4f} max_norm={winner['max_state_norm']:.2f}",
        flush=True,
    )
    return float(winner["lr"])


def phase_replicate(artifact_root: pathlib.Path, dry_run: bool) -> None:
    if dry_run:
        best_lr = 1e-3
        print("[DRY RUN] giả định best_lr=0.001", flush=True)
    else:
        best_lr = select_improved_lr(artifact_root)
    for seed in SEEDS:
        run_experiment(LEGACY_CONFIG, artifact_root, "legacy", seed, 1e-3, dry_run=dry_run)
        run_experiment(
            IMPROVED_CONFIG, artifact_root, "improved", seed, 1e-3, dry_run=dry_run
        )
        if best_lr != 1e-3:
            run_experiment(
                IMPROVED_CONFIG, artifact_root, "improved", seed, best_lr, dry_run=dry_run
            )


def phase_ablate(artifact_root: pathlib.Path, dry_run: bool) -> None:
    if dry_run:
        best_lr = 1e-3
    else:
        best_lr = select_improved_lr(artifact_root)
    run_experiment(
        LEGACY_CONFIG, artifact_root, "legacy", STRESS_SEED, best_lr, dry_run=dry_run
    )
    run_experiment(
        IMPROVED_CONFIG,
        artifact_root,
        "ablation-clip-only",
        STRESS_SEED,
        best_lr,
        ("train.optimizer_per_task=true", "train.grad_clip_norm=null"),
        dry_run=dry_run,
    )
    run_experiment(
        IMPROVED_CONFIG,
        artifact_root,
        "ablation-clip-carry",
        STRESS_SEED,
        best_lr,
        ("train.optimizer_per_task=false", "train.grad_clip_norm=null"),
        dry_run=dry_run,
    )
    run_experiment(
        IMPROVED_CONFIG, artifact_root, "improved", STRESS_SEED, best_lr, dry_run=dry_run
    )


def _mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} +/- {std:.4f}"


def phase_summarize(artifact_root: pathlib.Path, dry_run: bool = False) -> None:
    if dry_run:
        print("[DRY RUN] sẽ tổng hợp runs.csv và summary.md", flush=True)
        return
    rows = collect_results(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "tag",
        "seed",
        "lr",
        "valid",
        "accuracy",
        "forgetting",
        "bwt",
        "ncm_accuracy",
        "ncm_forgetting",
        "max_state_norm",
        "max_state_growth",
        "runtime_sec",
        "device",
        "git_commit",
        "path",
    ]
    with (artifact_root / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)

    groups = defaultdict(list)
    for row in rows:
        if row["tag"].startswith("smoke"):
            continue
        groups[(row["tag"], row["lr"])].append(row)

    lines = [
        "# Titan mới + M3 cũ/mới",
        "",
        "| Recipe | LR | Valid/Total | Linear Acc | Linear Fgt | NCM Acc | NCM Fgt | Max norm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (tag, lr), group in sorted(groups.items()):
        valid = [row for row in group if row["valid"]]
        lines.append(
            f"| {tag} | {lr:g} | {len(valid)}/{len(group)} | "
            f"{_mean_std([row['accuracy'] for row in valid])} | "
            f"{_mean_std([row['forgetting'] for row in valid])} | "
            f"{_mean_std([row['ncm_accuracy'] for row in valid if row['ncm_accuracy'] is not None])} | "
            f"{_mean_std([row['ncm_forgetting'] for row in valid if row['ncm_forgetting'] is not None])} | "
            f"{max((row['max_state_norm'] for row in group), default=math.nan):.2f} |"
        )

    matched = {
        (row["tag"], row["seed"]): row
        for row in rows
        if row["lr"] == 1e-3 and row["tag"] in {"legacy", "improved"}
    }
    lines.extend(
        [
            "",
            "## Paired difference tại LR 1e-3",
            "",
            "| Seed | New-Old Acc | New-Old Fgt | New-Old NCM Acc | Norm old | Norm new |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in SEEDS:
        old = matched.get(("legacy", seed))
        new = matched.get(("improved", seed))
        if old is None or new is None:
            lines.append(f"| {seed} | - | - | - | - | - |")
            continue
        ncm_delta = (
            new["ncm_accuracy"] - old["ncm_accuracy"]
            if new["ncm_accuracy"] is not None and old["ncm_accuracy"] is not None
            else math.nan
        )
        lines.append(
            f"| {seed} | {new['accuracy'] - old['accuracy']:+.4f} | "
            f"{new['forgetting'] - old['forgetting']:+.4f} | {ncm_delta:+.4f} | "
            f"{old['max_state_norm']:.2f} | {new['max_state_norm']:.2f} |"
        )
    (artifact_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def _is_google_compute_engine() -> bool:
    product = pathlib.Path("/sys/class/dmi/id/product_name")
    try:
        return "google compute engine" in product.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("check", "smoke", "stress", "sweep", "replicate", "ablate", "summarize", "all"),
    )
    parser.add_argument(
        "--artifact-root",
        type=pathlib.Path,
        default=ROOT / "artifacts" / "titan_m3_study",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dataset-check", action="store_true")
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()

    if args.shutdown and not args.dry_run and not _is_google_compute_engine():
        raise RuntimeError("--shutdown chỉ được phép trên Google Compute Engine")

    phases = {
        "check": lambda: phase_check(
            artifact_root, args.dry_run, args.skip_dataset_check
        ),
        "smoke": lambda: phase_smoke(artifact_root, args.dry_run),
        "stress": lambda: phase_stress(artifact_root, args.dry_run),
        "sweep": lambda: phase_sweep(artifact_root, args.dry_run),
        "replicate": lambda: phase_replicate(artifact_root, args.dry_run),
        "ablate": lambda: phase_ablate(artifact_root, args.dry_run),
        "summarize": lambda: phase_summarize(artifact_root, args.dry_run),
    }
    selected = (
        ("check", "smoke", "stress", "sweep", "replicate", "ablate", "summarize")
        if args.phase == "all"
        else (args.phase,)
    )
    try:
        for phase in selected:
            print(f"\n{'=' * 24} {phase.upper()} {'=' * 24}", flush=True)
            phases[phase]()
        return 0
    finally:
        if args.shutdown and not args.dry_run:
            print("[SHUTDOWN] Campaign kết thúc; đang tắt VM.", flush=True)
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
