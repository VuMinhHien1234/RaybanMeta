#!/usr/bin/env python3
"""Run the gated M3 study: sweep -> replicate -> RESISC45 -> focused optimization."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
LRS = (1e-4, 3e-4, 1e-3, 3e-3)
OPTIMIZE_CANDIDATES = (
    ("delta", 3e-3, 0.5, 16),
    ("ema", 3e-3, 0.5, 16),
    ("delta", 2e-3, 0.5, 16),
    ("delta", 5e-3, 0.5, 16),
    ("delta", 3e-3, 0.05, 16),
    ("delta", 3e-3, 0.1, 16),
    ("delta", 3e-3, 0.25, 16),
    ("delta", 3e-3, 0.5, 8),
    ("delta", 3e-3, 0.5, 32),
)


def _run(config: pathlib.Path, log_dir: pathlib.Path, overrides: list[str], dry_run: bool) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_g1.py"),
        "--config",
        str(config),
        "--skip-existing",
        "--set",
        *overrides,
        f"log.dir={log_dir}",
    ]
    print("\n$", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def _m3_overrides(
    seed: int, style: str, lr: float, alpha: float = 0.5, frequency: int = 16
) -> list[str]:
    return [
        f"seed={seed}",
        "train.optimizer=m3",
        f"train.m3.beta_style={style}",
        "train.m3.update_norm=clip",
        f"train.m3.alpha={alpha}",
        f"train.m3.frequency={frequency}",
        f"train.lr={lr}",
    ]


def _finite_metrics(
    log_dir: pathlib.Path, dataset: str, seed: int, update_norm: str = "clip"
) -> list[tuple[pathlib.Path, dict]]:
    rows = []
    for path in sorted((log_dir / "results").glob(f"{dataset}_hope_seed{seed}_m3_*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        cfg = yaml.safe_load(path.with_name("config.yaml").read_text(encoding="utf-8"))
        # Các run trước 07-24 không ghi trường này; mặc định khi đó là legacy "rms".
        norm = str(cfg["train"].get("m3", {}).get("update_norm", "rms")).lower()
        if norm != update_norm:
            continue
        train_log_path = path.with_name("train_log.json")
        if not train_log_path.exists():
            continue
        train_log = json.loads(train_log_path.read_text(encoding="utf-8"))
        finite = train_log.get("state_finite", {})
        if finite and all(bool(value) for value in finite.values()):
            rows.append((path, metrics))
    return rows


def _best_eurosat_m3(log_dir: pathlib.Path) -> tuple[str, float]:
    rows = _finite_metrics(log_dir, "eurosat", seed=0)
    if not rows:
        raise RuntimeError("Chưa có M3 EuroSAT seed 0 hữu hạn. Chạy phase=sweep trước.")
    path, _ = max(rows, key=lambda item: float(item[1]["average_accuracy"]))
    cfg = yaml.safe_load(path.with_name("config.yaml").read_text(encoding="utf-8"))
    return str(cfg["train"]["m3"]["beta_style"]), float(cfg["train"]["lr"])


def _require_three_finite_seeds(log_dir: pathlib.Path, style: str, lr: float) -> None:
    missing = []
    for seed in (0, 1, 2):
        matches = []
        for path, _ in _finite_metrics(log_dir, "eurosat", seed):
            cfg = yaml.safe_load(path.with_name("config.yaml").read_text(encoding="utf-8"))
            m3_cfg = cfg["train"]["m3"]
            if str(m3_cfg["beta_style"]) == style and float(cfg["train"]["lr"]) == lr:
                matches.append(path)
        if not matches:
            missing.append(seed)
    if missing:
        raise RuntimeError(
            f"Chưa đủ EuroSAT M3-{style} clip lr={lr:g} hữu hạn; thiếu seed {missing}. "
            "Chạy phase=replicate và xử lý lỗi trước RESISC45."
        )


def _candidate_metrics(
    log_dir: pathlib.Path,
    dataset: str,
    seed: int,
    candidate: tuple[str, float, float, int],
) -> tuple[pathlib.Path, dict] | None:
    style, lr, alpha, frequency = candidate
    for path, metrics in _finite_metrics(log_dir, dataset, seed):
        cfg = yaml.safe_load(path.with_name("config.yaml").read_text(encoding="utf-8"))
        m3_cfg = cfg["train"]["m3"]
        if (
            str(m3_cfg["beta_style"]) == style
            and float(cfg["train"]["lr"]) == lr
            and float(m3_cfg.get("alpha", 0.5)) == alpha
            and int(m3_cfg.get("frequency", 16)) == frequency
        ):
            return path, metrics
    return None


def _best_optimization_candidate(log_dir: pathlib.Path) -> tuple[str, float, float, int]:
    rows = []
    for candidate in OPTIMIZE_CANDIDATES:
        match = _candidate_metrics(log_dir, "eurosat", 0, candidate)
        if match is None:
            continue
        _, metrics = match
        accuracy = float(metrics["average_accuracy"])
        forgetting = float(metrics["average_forgetting"])
        score = accuracy - max(forgetting, 0.0)
        rows.append((score, accuracy, candidate))
    if len(rows) != len(OPTIMIZE_CANDIDATES):
        found = {row[2] for row in rows}
        missing = [candidate for candidate in OPTIMIZE_CANDIDATES if candidate not in found]
        raise RuntimeError(f"Optimize grid thiếu kết quả hữu hạn seed 0: {missing}")
    return max(rows, key=lambda row: (row[0], row[1]))[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("smoke", "sweep", "replicate", "resisc45", "optimize")
    )
    parser.add_argument("--log-dir", type=pathlib.Path, default=ROOT / "artifacts" / "m3_study")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    eurosat = ROOT / "configs" / "g4_hope_eurosat.yaml"
    resisc45 = ROOT / "configs" / "g4_hope_resisc45.yaml"

    if args.phase == "smoke":
        common = [
            "device=cpu",
            "data.name=synthetic",
            "data.num_classes=2",
            "data.train_per_class=4",
            "data.val_per_class=2",
            "data.test_per_class=2",
            "data.num_tasks=1",
            "data.image_size=224",
            "data.batch_size=4",
            "data.num_workers=0",
            "backbone.pretrained=false",
            "train.epochs_per_task=1",
            "train.eval_future=false",
        ]
        _run(eurosat, args.log_dir, common + ["train.optimizer=adamw"], args.dry_run)
        _run(eurosat, args.log_dir, common + _m3_overrides(0, "ema", 1e-3), args.dry_run)
        _run(eurosat, args.log_dir, common + _m3_overrides(0, "delta", 1e-3), args.dry_run)
        return 0

    if args.phase == "sweep":
        _run(
            eurosat,
            args.log_dir,
            ["seed=0", "data.num_workers=0", "train.optimizer=adamw"],
            args.dry_run,
        )
        for style in ("ema", "delta"):
            for lr in LRS:
                _run(
                    eurosat,
                    args.log_dir,
                    ["data.num_workers=0", *_m3_overrides(0, style, lr)],
                    args.dry_run,
                )
        return 0

    if args.phase == "optimize":
        # Seed 0 grid. Existing baseline candidates are skipped automatically.
        for style, lr, alpha, frequency in OPTIMIZE_CANDIDATES:
            _run(
                eurosat,
                args.log_dir,
                [
                    "data.num_workers=0",
                    *_m3_overrides(0, style, lr, alpha, frequency),
                ],
                args.dry_run,
            )
        if args.dry_run:
            return 0

        winner = _best_optimization_candidate(args.log_dir)
        style, lr, alpha, frequency = winner
        print(
            f"[optimize winner] style={style} lr={lr:g} "
            f"alpha={alpha:g} frequency={frequency}",
            flush=True,
        )
        # Replicate the selected candidate on EuroSAT.
        for seed in (1, 2):
            _run(
                eurosat,
                args.log_dir,
                [
                    "data.num_workers=0",
                    *_m3_overrides(seed, style, lr, alpha, frequency),
                ],
                args.dry_run,
            )
        for seed in (0, 1, 2):
            if _candidate_metrics(args.log_dir, "eurosat", seed, winner) is None:
                raise RuntimeError(f"Winner không hữu hạn trên EuroSAT seed {seed}")

        # Only spend RESISC45 time when the winner differs from the validated baseline.
        baseline = ("delta", 3e-3, 0.5, 16)
        if winner != baseline:
            for seed in (0, 1, 2):
                _run(
                    resisc45,
                    args.log_dir,
                    [
                        "data.num_workers=0",
                        *_m3_overrides(seed, style, lr, alpha, frequency),
                    ],
                    args.dry_run,
                )
        return 0

    style, lr = _best_eurosat_m3(args.log_dir)
    print(f"[selected] M3-{style.upper()} lr={lr:g}")

    if args.phase == "replicate":
        for seed in (1, 2):
            _run(
                eurosat,
                args.log_dir,
                [f"seed={seed}", "data.num_workers=0", "train.optimizer=adamw"],
                args.dry_run,
            )
            _run(
                eurosat,
                args.log_dir,
                ["data.num_workers=0", *_m3_overrides(seed, style, lr)],
                args.dry_run,
            )
        return 0

    _require_three_finite_seeds(args.log_dir, style, lr)
    for seed in (0, 1, 2):
        _run(
            resisc45,
            args.log_dir,
            [f"seed={seed}", "data.num_workers=0", "train.optimizer=adamw"],
            args.dry_run,
        )
        _run(
            resisc45,
            args.log_dir,
            ["data.num_workers=0", *_m3_overrides(seed, style, lr)],
            args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
