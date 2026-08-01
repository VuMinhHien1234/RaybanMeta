#!/usr/bin/env python3
"""Run and summarize the bounded-online versus post-hoc NCM Head study."""
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

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TITANS_CONFIG = ROOT / "configs" / "g2_titans_resisc45_selfmod_m3_ncm.yaml"
SMOKE_CONFIG = ROOT / "configs" / "g2_titans_ncm_smoke.yaml"
FROZEN_CONFIG = ROOT / "configs" / "g1_resisc45.yaml"
SEEDS = (0, 1, 2)


def _run_g1_module():
    path = ROOT / "scripts" / "run_g1.py"
    spec = importlib.util.spec_from_file_location("ncm_study_run_g1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN_G1 = _run_g1_module()


def _load_final_config(path: pathlib.Path, overrides: list[str]) -> dict:
    from uavcl.utils.config import apply_overrides, load_config

    return apply_overrides(load_config(path), list(overrides))


def _result_dir(
    config_path: pathlib.Path,
    method: str,
    artifact_root: pathlib.Path,
    overrides: list[str],
) -> pathlib.Path:
    cfg = _load_final_config(config_path, [*overrides, f"log.dir={artifact_root}"])
    return artifact_root / "results" / RUN_G1.run_dir_name(cfg, method)


def _stream(cmd: list[str], log_path: pathlib.Path, *, dry_run: bool) -> int:
    print("\n$", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {' '.join(cmd)}\n")
        handle.flush()
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return process.wait()


def _run_experiment(
    config_path: pathlib.Path,
    method: str,
    artifact_root: pathlib.Path,
    overrides: list[str],
    label: str,
    *,
    dry_run: bool,
) -> pathlib.Path:
    final_overrides = [*overrides, f"log.dir={artifact_root}"]
    out = _result_dir(config_path, method, artifact_root, overrides)
    cfg = _load_final_config(config_path, final_overrides)
    if RUN_G1.result_is_complete(out, cfg) and not (out / "failure.json").exists():
        print(f"[SKIP COMPLETE] {out.name}", flush=True)
        return out
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_g1.py"),
        "--config",
        str(config_path),
        "--method",
        method,
        "--skip-existing",
        "--set",
        *final_overrides,
    ]
    code = _stream(cmd, artifact_root / "campaign_logs" / f"{label}.log", dry_run=dry_run)
    if dry_run:
        return out
    if code != 0 or not RUN_G1.result_is_complete(out, cfg):
        raise RuntimeError(f"Experiment {label} thất bại (exit={code}): {out}")
    return out


def phase_check(artifact_root: pathlib.Path, *, dry_run: bool) -> None:
    for name, cmd in (
        ("check_env", [sys.executable, "scripts/check_env.py"]),
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
    ):
        code = _stream(cmd, artifact_root / "campaign_logs" / f"{name}.log", dry_run=dry_run)
        if code != 0:
            raise RuntimeError(f"{name} thất bại (exit={code})")


def phase_smoke(artifact_root: pathlib.Path, *, dry_run: bool) -> pathlib.Path:
    return _run_experiment(
        SMOKE_CONFIG,
        "titans",
        artifact_root,
        [],
        "smoke",
        dry_run=dry_run,
    )


def _titans_overrides(seed: int) -> list[str]:
    return [f"seed={seed}", "log.run_tag=ncm-head"]


def _frozen_overrides(seed: int) -> list[str]:
    return [
        f"seed={seed}",
        "data.split_protocol=combined31500_v2",
        "data.expected_total_samples=31500",
        "data.expected_samples_per_class=700",
        "data.prototype_batch_size=32",
        "log.run_tag=ncm-frozen",
    ]


def phase_seed(
    artifact_root: pathlib.Path, seed: int, *, dry_run: bool
) -> tuple[pathlib.Path, pathlib.Path]:
    titans = _run_experiment(
        TITANS_CONFIG,
        "titans",
        artifact_root,
        _titans_overrides(seed),
        f"titans_seed{seed}",
        dry_run=dry_run,
    )
    frozen = _run_experiment(
        FROZEN_CONFIG,
        "ncm",
        artifact_root,
        _frozen_overrides(seed),
        f"frozen_ncm_seed{seed}",
        dry_run=dry_run,
    )
    return titans, frozen


def _read_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _final_alignment(log: dict) -> float | None:
    diagnostics = log.get("ncm_diagnostics", {})
    if not diagnostics:
        return None
    final_key = max(diagnostics, key=lambda key: int(key))
    value = diagnostics[final_key].get("online_posthoc_alignment", {}).get("mean_cosine")
    return None if value is None else float(value)


def collect_rows(artifact_root: pathlib.Path) -> list[dict]:
    rows = []
    for seed in SEEDS:
        titans_cfg = _load_final_config(
            TITANS_CONFIG,
            [*_titans_overrides(seed), f"log.dir={artifact_root}"],
        )
        titans_dir = artifact_root / "results" / RUN_G1.run_dir_name(titans_cfg, "titans")
        frozen_cfg = _load_final_config(
            FROZEN_CONFIG,
            [*_frozen_overrides(seed), f"log.dir={artifact_root}"],
        )
        frozen_dir = artifact_root / "results" / RUN_G1.run_dir_name(frozen_cfg, "ncm")
        required = (
            titans_dir / "metrics.json",
            titans_dir / "metrics_ncm_online.json",
            titans_dir / "metrics_ncm_posthoc.json",
            titans_dir / "train_log.json",
            frozen_dir / "metrics.json",
        )
        if not all(path.exists() for path in required):
            continue
        linear = _read_json(titans_dir / "metrics.json")
        online = _read_json(titans_dir / "metrics_ncm_online.json")
        posthoc = _read_json(titans_dir / "metrics_ncm_posthoc.json")
        frozen = _read_json(frozen_dir / "metrics.json")
        log = _read_json(titans_dir / "train_log.json")
        norms = [float(value) for value in log.get("state_norm", {}).values()]
        values = [
            float(linear["average_accuracy"]),
            float(online["average_accuracy"]),
            float(posthoc["average_accuracy"]),
            float(frozen["average_accuracy"]),
        ]
        rows.append(
            {
                "seed": seed,
                "valid": all(math.isfinite(value) for value in values)
                and all(math.isfinite(value) for value in norms),
                "linear_accuracy": values[0],
                "linear_forgetting": float(linear["average_forgetting"]),
                "online_accuracy": values[1],
                "online_forgetting": float(online["average_forgetting"]),
                "posthoc_accuracy": values[2],
                "posthoc_forgetting": float(posthoc["average_forgetting"]),
                "frozen_accuracy": values[3],
                "frozen_forgetting": float(frozen["average_forgetting"]),
                "titans_gain_over_frozen": values[1] - values[3],
                "oracle_online_gap": values[2] - values[1],
                "final_prototype_alignment": _final_alignment(log),
                "max_state_norm": max(norms) if norms else None,
                "runtime_sec": float(linear.get("runtime_sec", 0.0)),
                "device": linear.get("device"),
                "git_commit": linear.get("git_commit"),
                "titans_path": str(titans_dir),
                "frozen_path": str(frozen_dir),
            }
        )
    return rows


def _mean_std(rows: list[dict], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return "-"
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.4f} +/- {std:.4f}"


def write_report(artifact_root: pathlib.Path) -> list[dict]:
    rows = collect_rows(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    columns = [
        "seed",
        "valid",
        "linear_accuracy",
        "linear_forgetting",
        "online_accuracy",
        "online_forgetting",
        "posthoc_accuracy",
        "posthoc_forgetting",
        "frozen_accuracy",
        "frozen_forgetting",
        "titans_gain_over_frozen",
        "oracle_online_gap",
        "final_prototype_alignment",
        "max_state_norm",
        "runtime_sec",
        "device",
        "git_commit",
        "titans_path",
        "frozen_path",
    ]
    with (artifact_root / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if row["valid"]]
    lines = [
        "# NCM Head study",
        "",
        "## Per-seed results",
        "",
        "| Seed | Frozen NCM Acc/Fgt | Titans Linear Acc/Fgt | Titans NCM Online Acc/Fgt | "
        "Titans NCM Post-hoc Acc/Fgt | Online-Frozen | Oracle-Online |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['frozen_accuracy']:.4f}/{row['frozen_forgetting']:.4f} | "
            f"{row['linear_accuracy']:.4f}/{row['linear_forgetting']:.4f} | "
            f"{row['online_accuracy']:.4f}/{row['online_forgetting']:.4f} | "
            f"{row['posthoc_accuracy']:.4f}/{row['posthoc_forgetting']:.4f} | "
            f"{row['titans_gain_over_frozen']:+.4f} | {row['oracle_online_gap']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Valid seeds: {len(valid)}/{len(SEEDS)}",
            f"- Frozen NCM accuracy: {_mean_std(valid, 'frozen_accuracy')}",
            f"- Titans Linear accuracy: {_mean_std(valid, 'linear_accuracy')}",
            f"- Titans NCM Online accuracy: {_mean_std(valid, 'online_accuracy')}",
            f"- Titans NCM Post-hoc accuracy: {_mean_std(valid, 'posthoc_accuracy')}",
            f"- Titans Online gain over Frozen NCM: {_mean_std(valid, 'titans_gain_over_frozen')}",
            f"- Post-hoc oracle minus Online gap: {_mean_std(valid, 'oracle_online_gap')}",
            f"- Final online/post-hoc prototype cosine: "
            f"{_mean_std(valid, 'final_prototype_alignment')}",
            "",
            "Post-hoc được phép đọc lại toàn bộ train data đã thấy; Online và Frozen NCM không được phép.",
            "Chỉ Titans NCM Online so với Frozen NCM là phép so chính cho head bounded/deployable.",
        ]
    )
    (artifact_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return rows


def _is_google_compute_engine() -> bool:
    product = pathlib.Path("/sys/class/dmi/id/product_name")
    try:
        return "google compute engine" in product.read_text(encoding="utf-8").lower()
    except OSError:
        return False


def shutdown_vm() -> None:
    if not _is_google_compute_engine():
        raise RuntimeError("Từ chối shutdown: máy hiện tại không phải Google Compute Engine")
    print("[SHUTDOWN] Campaign đã kết thúc; VM sẽ tắt ngay.", flush=True)
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("check", "smoke", "seed0", "replicate", "report", "all"),
    )
    parser.add_argument(
        "--artifact-root",
        type=pathlib.Path,
        default=ROOT / "artifacts_ncm_head_study",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    succeeded = False
    try:
        if args.phase in ("check", "all"):
            phase_check(root, dry_run=args.dry_run)
        if args.phase in ("smoke", "all"):
            phase_smoke(root, dry_run=args.dry_run)
        if args.phase in ("seed0", "all"):
            phase_seed(root, 0, dry_run=args.dry_run)
        if args.phase in ("replicate", "all"):
            for seed in SEEDS:
                phase_seed(root, seed, dry_run=args.dry_run)
        if args.phase in ("report", "all") and not args.dry_run:
            rows = write_report(root)
            if args.phase == "all" and len([row for row in rows if row["valid"]]) != len(SEEDS):
                raise RuntimeError("Campaign chưa có đủ 3 seed hợp lệ")
        succeeded = True
        return 0
    finally:
        if args.shutdown and not args.dry_run:
            status = "thành công" if succeeded else "thất bại"
            print(f"[FINAL] Campaign {status}; thực hiện shutdown theo yêu cầu.", flush=True)
            shutdown_vm()


if __name__ == "__main__":
    raise SystemExit(main())
