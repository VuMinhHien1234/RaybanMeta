#!/usr/bin/env python3
"""Run the no-replay Titans + NCM blend/transport/distillation campaign."""
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
FULL_CONFIG = ROOT / "configs" / "g2_titans_resisc45_ncm_adaptation.yaml"
SMOKE_CONFIG = ROOT / "configs" / "g2_titans_ncm_adaptation_smoke.yaml"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts_ncm_adaptation"
SEEDS = (0, 1, 2)
CONFIRMATORY_SEEDS = (3, 4, 5)
CONFIRMATORY_GAMMA = 0.25
DISTILL_WEIGHTS = (0.01, 0.05, 0.1)
TRANSPORT_CANDIDATES = (
    ("translation", 0.0, 1.0),
    ("diagonal", 10.0, 0.5),
    ("procrustes", 0.0, 0.5),
    ("identity_ridge", 1.0, 0.5),
    ("identity_ridge", 10.0, 0.5),
    ("identity_ridge", 100.0, 0.5),
    ("identity_ridge", 10.0, 0.25),
    ("identity_ridge", 10.0, 1.0),
)


def _run_g1_module():
    path = ROOT / "scripts" / "run_g1.py"
    spec = importlib.util.spec_from_file_location("ncm_adaptation_run_g1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN_G1 = _run_g1_module()


def _load_config(config_path: pathlib.Path, overrides: list[str]) -> dict:
    from uavcl.utils.config import apply_overrides, load_config

    return apply_overrides(load_config(config_path), overrides)


def _result_dir(
    config_path: pathlib.Path,
    artifact_root: pathlib.Path,
    tag: str,
    overrides: list[str],
) -> pathlib.Path:
    final = _load_config(
        config_path, [*overrides, f"log.dir={artifact_root}", f"log.run_tag={tag}"]
    )
    method = str(final.get("train", {}).get("method", "titans")).lower()
    return artifact_root / "results" / RUN_G1.run_dir_name(final, method)


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
    artifact_root: pathlib.Path,
    tag: str,
    overrides: list[str],
    *,
    dry_run: bool,
) -> pathlib.Path:
    final_overrides = [*overrides, f"log.dir={artifact_root}", f"log.run_tag={tag}"]
    cfg = _load_config(config_path, final_overrides)
    out = _result_dir(config_path, artifact_root, tag, overrides)
    if RUN_G1.result_is_complete(out, cfg) and not (out / "failure.json").exists():
        print(f"[SKIP COMPLETE] {out.name}", flush=True)
        return out
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_g1.py"),
        "--config",
        str(config_path),
        "--method",
        "titans",
        "--skip-existing",
        "--set",
        *final_overrides,
    ]
    code = _stream(cmd, artifact_root / "campaign_logs" / f"{tag}.log", dry_run=dry_run)
    if dry_run:
        return out
    if code != 0 or not RUN_G1.result_is_complete(out, cfg):
        raise RuntimeError(f"Experiment {tag} thất bại (exit={code}): {out}")
    return out


def _gamma_list(values: list[float]) -> str:
    unique = sorted({float(value) for value in values})
    return "[" + ",".join(f"{value:g}" for value in unique) + "]"


def _validation_scores(result_dir: pathlib.Path) -> dict[float, float]:
    from uavcl.models.ncm_adaptation import gamma_key

    log = json.loads((result_dir / "train_log.json").read_text(encoding="utf-8"))
    gammas = log["ncm_protocol"]["blend_gammas"]
    scores = {}
    for gamma in gammas:
        key = gamma_key(float(gamma))
        values = [
            float(task["online_blend"][key]["current_task_validation_accuracy"])
            for task in log["ncm_diagnostics"].values()
        ]
        scores[float(gamma)] = statistics.fmean(values)
    return scores


def _transport_gate_score(result_dir: pathlib.Path, gamma: float) -> tuple[float, int]:
    from uavcl.models.ncm_adaptation import gamma_key

    key = gamma_key(gamma)
    log = json.loads((result_dir / "train_log.json").read_text(encoding="utf-8"))
    diagnostics = [
        task["online_blend"][key]["transport"]
        for task in log["ncm_diagnostics"].values()
    ]
    accepted = [item for item in diagnostics if item.get("accepted")]
    improvement = statistics.fmean(
        float(item.get("improvement", 0.0)) for item in accepted
    ) if accepted else -math.inf
    return improvement, len(accepted)


def _selection_path(artifact_root: pathlib.Path) -> pathlib.Path:
    return artifact_root / "selection.json"


def _load_selection(artifact_root: pathlib.Path) -> dict:
    path = _selection_path(artifact_root)
    if not path.exists():
        raise RuntimeError("Thiếu selection.json; chạy phase screen trước")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_selection(artifact_root: pathlib.Path, selection: dict) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    _selection_path(artifact_root).write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )


def phase_check(artifact_root: pathlib.Path, *, dry_run: bool) -> None:
    for name, command in (
        ("check_env", [sys.executable, "scripts/check_env.py"]),
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
    ):
        code = _stream(
            command, artifact_root / "campaign_logs" / f"{name}.log", dry_run=dry_run
        )
        if code != 0:
            raise RuntimeError(f"{name} thất bại (exit={code})")


def phase_smoke(artifact_root: pathlib.Path, *, dry_run: bool) -> None:
    _run_experiment(
        SMOKE_CONFIG, artifact_root, "adaptation-smoke", [], dry_run=dry_run
    )


def phase_blend(artifact_root: pathlib.Path, *, dry_run: bool) -> pathlib.Path:
    return _run_experiment(
        FULL_CONFIG,
        artifact_root,
        "blend-screen-seed0",
        [
            "seed=0",
            "train.ncm.transport.enabled=false",
            "train.feature_distillation.enabled=false",
        ],
        dry_run=dry_run,
    )


def phase_transport(
    artifact_root: pathlib.Path, selection: dict, *, dry_run: bool
) -> pathlib.Path:
    gamma = float(selection["experimental_gamma"])
    candidates = [
        {
            "name": f"{kind}_r{regularization:g}_b{beta:g}",
            "type": kind,
            "regularization": regularization,
            "beta": beta,
        }
        for kind, regularization, beta in TRANSPORT_CANDIDATES
    ]
    return _run_experiment(
        FULL_CONFIG,
        artifact_root,
        "transport-candidates-seed0",
        [
            "seed=0",
            f"train.ncm.blend.gammas={_gamma_list([0.0, gamma, 1.0])}",
            "train.ncm.transport.enabled=true",
            f"train.ncm.transport.screen_gamma={gamma:g}",
            f"train.ncm.transport.candidates={json.dumps(candidates, separators=(',', ':'))}",
            "train.feature_distillation.enabled=false",
        ],
        dry_run=dry_run,
    )


def phase_distill(
    artifact_root: pathlib.Path, selection: dict, *, dry_run: bool
) -> list[tuple[float, pathlib.Path]]:
    gamma = float(selection["experimental_gamma"])
    runs = []
    for weight in DISTILL_WEIGHTS:
        out = _run_experiment(
            FULL_CONFIG,
            artifact_root,
            f"distill-fixed-w{weight:g}-seed0",
            [
                "seed=0",
                f"train.ncm.blend.gammas={_gamma_list([0.0, gamma, 1.0])}",
                "train.ncm.transport.enabled=false",
                "train.feature_distillation.enabled=true",
                "train.feature_distillation.teacher=frozen_backbone",
                f"train.feature_distillation.weight={weight:g}",
            ],
            dry_run=dry_run,
        )
        runs.append((weight, out))
    return runs


def phase_combine(
    artifact_root: pathlib.Path, selection: dict, seed: int, *, dry_run: bool
) -> pathlib.Path:
    gamma = float(selection["experimental_gamma"])
    transport = selection["transport"]
    weight = float(selection["distillation_weight"])
    return _run_experiment(
        FULL_CONFIG,
        artifact_root,
        f"full-combination-seed{seed}",
        [
            f"seed={seed}",
            f"train.ncm.blend.gammas={_gamma_list([0.0, gamma, 1.0])}",
            "train.ncm.transport.enabled=true",
            f"train.ncm.transport.type={transport['type']}",
            f"train.ncm.transport.regularization={transport['regularization']:g}",
            f"train.ncm.transport.beta={transport['beta']:g}",
            "train.feature_distillation.enabled=true",
            "train.feature_distillation.teacher=frozen_backbone",
            f"train.feature_distillation.weight={weight:g}",
        ],
        dry_run=dry_run,
    )


def phase_screen(artifact_root: pathlib.Path, *, dry_run: bool) -> None:
    blend = phase_blend(artifact_root, dry_run=dry_run)
    if dry_run:
        placeholder = {
            "experimental_gamma": 0.25,
            "transport": {"type": "identity_ridge", "regularization": 10.0, "beta": 0.5},
            "distillation_weight": 0.05,
        }
        phase_transport(artifact_root, placeholder, dry_run=True)
        phase_distill(artifact_root, placeholder, dry_run=True)
        phase_combine(artifact_root, placeholder, 0, dry_run=True)
        return

    blend_scores = _validation_scores(blend)
    positive = {gamma: score for gamma, score in blend_scores.items() if gamma > 0.0}
    experimental_gamma = max(positive, key=lambda gamma: (positive[gamma], -gamma))
    selection = {
        "selection_protocol": "mean current-task validation; no old validation/test revisit",
        "blend_validation_scores": blend_scores,
        "overall_gamma": max(blend_scores, key=lambda gamma: (blend_scores[gamma], -gamma)),
        "experimental_gamma": experimental_gamma,
    }

    transport_run = phase_transport(artifact_root, selection, dry_run=False)
    transport_log = json.loads(
        (transport_run / "train_log.json").read_text(encoding="utf-8")
    )
    transport_rows = []
    candidates = transport_log["ncm_protocol"]["transport"]["candidates"]
    for index, candidate in enumerate(candidates):
        safe_name = "".join(
            char if char.isalnum() else "_"
            for char in str(candidate["name"]).lower()
        ).strip("_")
        key = f"t{index}_{safe_name or 'candidate'}"
        task_values = [
            task["transport_candidates"][key]
            for task in transport_log["ncm_diagnostics"].values()
        ]
        validation = statistics.fmean(
            float(item["current_task_validation_accuracy"]) for item in task_values
        )
        accepted = [item["transport"] for item in task_values if item["transport"].get("accepted")]
        gate_improvement = statistics.fmean(
            float(item.get("improvement", 0.0)) for item in accepted
        ) if accepted else -math.inf
        transport_rows.append(
            (
                validation,
                gate_improvement,
                len(accepted),
                (candidate["type"], candidate["regularization"], candidate["beta"]),
                transport_run,
            )
        )
    valid_transport = [row for row in transport_rows if row[2] > 0]
    if not valid_transport:
        raise RuntimeError("Không transport candidate nào qua safety gate")
    winner_t = max(valid_transport, key=lambda row: (row[0], row[1], row[2]))
    kind, regularization, beta = winner_t[3]
    selection["transport"] = {
        "type": kind,
        "regularization": regularization,
        "beta": beta,
        "validation_score": winner_t[0],
        "mean_gate_improvement": winner_t[1],
        "accepted_tasks": winner_t[2],
    }

    distill_runs = phase_distill(artifact_root, selection, dry_run=False)
    distill_rows = [
        (_validation_scores(result)[experimental_gamma], weight, result)
        for weight, result in distill_runs
    ]
    winner_d = max(distill_rows, key=lambda row: (row[0], -row[1]))
    selection["distillation_weight"] = winner_d[1]
    selection["distillation_validation_score"] = winner_d[0]
    _save_selection(artifact_root, selection)
    phase_combine(artifact_root, selection, 0, dry_run=False)


def phase_replicate(artifact_root: pathlib.Path, *, dry_run: bool) -> None:
    selection = _load_selection(artifact_root) if not dry_run else {
        "experimental_gamma": 0.25,
        "transport": {"type": "identity_ridge", "regularization": 10.0, "beta": 0.5},
        "distillation_weight": 0.05,
    }
    for seed in (1, 2):
        _run_experiment(
            FULL_CONFIG,
            artifact_root,
            f"blend-screen-seed{seed}",
            [
                f"seed={seed}",
                "train.ncm.transport.enabled=false",
                "train.feature_distillation.enabled=false",
            ],
            dry_run=dry_run,
        )
    for seed in (1, 2):
        phase_combine(artifact_root, selection, seed, dry_run=dry_run)


def phase_confirm_gamma025(
    artifact_root: pathlib.Path, *, dry_run: bool
) -> list[pathlib.Path]:
    """Evaluate the pre-registered gamma on fresh seeds without another sweep."""
    results = []
    for seed in CONFIRMATORY_SEEDS:
        results.append(
            _run_experiment(
                FULL_CONFIG,
                artifact_root,
                f"gamma025-confirm-seed{seed}",
                [
                    f"seed={seed}",
                    f"train.ncm.blend.gammas={_gamma_list([0.0, CONFIRMATORY_GAMMA])}",
                    "train.ncm.transport.enabled=false",
                    "train.feature_distillation.enabled=false",
                ],
                dry_run=dry_run,
            )
        )
    return results


def _read_metric(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_confirm_report(artifact_root: pathlib.Path) -> pathlib.Path:
    from uavcl.models.ncm_adaptation import gamma_key

    gamma_key_value = gamma_key(CONFIRMATORY_GAMMA)
    rows = []
    for seed in CONFIRMATORY_SEEDS:
        matches = sorted(
            (artifact_root / "results").glob(
                f"resisc45_titans_seed{seed}_gamma025-confirm-seed{seed}_*"
            )
        )
        complete = [
            result
            for result in matches
            if (result / "metrics_ncm_blend_g0.json").exists()
            and (result / f"metrics_ncm_blend_{gamma_key_value}.json").exists()
            and (result / "train_log.json").exists()
            and not (result / "failure.json").exists()
        ]
        if len(complete) != 1:
            raise RuntimeError(
                f"Seed {seed} cần đúng 1 result hoàn chỉnh, tìm thấy {len(complete)}"
            )
        result = complete[0]
        frozen = _read_metric(result / "metrics_ncm_blend_g0.json")
        candidate = _read_metric(result / f"metrics_ncm_blend_{gamma_key_value}.json")
        train_log = _read_metric(result / "train_log.json")
        diagnostics = [
            task["online_blend"][gamma_key_value]
            for task in train_log["ncm_diagnostics"].values()
        ]
        no_replay = all(
            not bool(item.get("revisit_old_train"))
            and int(item.get("old_samples_revisited", 0)) == 0
            for item in diagnostics
        )
        state_finite = all(bool(value) for value in train_log["state_finite"].values())
        values = [
            float(candidate["average_accuracy"]),
            float(candidate["average_forgetting"]),
            float(candidate["backward_transfer"]),
            float(frozen["average_accuracy"]),
            float(frozen["average_forgetting"]),
        ]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Seed {seed} chứa NaN/Inf trong metrics")
        if not no_replay or not state_finite:
            raise RuntimeError(
                f"Seed {seed} vi phạm hậu kiểm: no_replay={no_replay}, "
                f"state_finite={state_finite}"
            )
        rows.append(
            {
                "seed": seed,
                "gamma": CONFIRMATORY_GAMMA,
                "accuracy": values[0],
                "forgetting": values[1],
                "backward_transfer": values[2],
                "frozen_accuracy": values[3],
                "frozen_forgetting": values[4],
                "gain_over_frozen": values[0] - values[3],
                "no_replay": no_replay,
                "state_finite": state_finite,
                "result_dir": str(result),
            }
        )

    artifact_root.mkdir(parents=True, exist_ok=True)
    summary = artifact_root / "gamma025_confirmatory_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def mean_std(field: str) -> tuple[float, float]:
        values = [float(row[field]) for row in rows]
        return statistics.fmean(values), statistics.stdev(values)

    acc_mean, acc_std = mean_std("accuracy")
    fgt_mean, fgt_std = mean_std("forgetting")
    gain_mean, gain_std = mean_std("gain_over_frozen")
    lines = [
        "# Confirmatory evaluation: Anchored Blend gamma 0.25",
        "",
        f"- Gamma was fixed to `{CONFIRMATORY_GAMMA:g}` before these runs.",
        f"- Fresh seeds: `{', '.join(map(str, CONFIRMATORY_SEEDS))}`.",
        "- Transport: disabled.",
        "- Feature distillation: disabled.",
        "- Old train replay: forbidden and audited.",
        "",
        "| Seed | Accuracy | Forgetting | Frozen control | Gain over Frozen |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['accuracy']:.4f} | {row['forgetting']:.4f} | "
            f"{row['frozen_accuracy']:.4f} | {row['gain_over_frozen']:+.4f} |"
        )
    lines.extend(
        [
            f"| **Mean +/- std** | **{acc_mean:.4f} +/- {acc_std:.4f}** | "
            f"**{fgt_mean:.4f} +/- {fgt_std:.4f}** | **0.7114 +/- 0.0000** | "
            f"**{gain_mean:+.4f} +/- {gain_std:.4f}** |",
            "",
            "This report is confirmatory for the pre-registered gamma only; no gamma "
            "was re-selected from these test results.",
        ]
    )
    report = artifact_root / "REPORT_GAMMA025_CONFIRMATORY.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[CONFIRM REPORT] {report}")
    return report


def phase_report(artifact_root: pathlib.Path) -> pathlib.Path:
    selection = _load_selection(artifact_root)
    gamma = float(selection["experimental_gamma"])
    from uavcl.models.ncm_adaptation import gamma_key

    key = gamma_key(gamma)
    rows = []
    for result in sorted((artifact_root / "results").glob("*")):
        config_path = result / "config.yaml"
        metric_path = result / f"metrics_ncm_blend_{key}.json"
        if not config_path.exists() or not metric_path.exists() or (result / "failure.json").exists():
            continue
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "tag": str(cfg.get("log", {}).get("run_tag", "")),
                "seed": int(cfg.get("seed", 0)),
                "gamma": gamma,
                **_read_metric(metric_path),
                "result_dir": str(result),
            }
        )
    if not rows:
        raise RuntimeError("Không có metrics hợp lệ để viết report")
    artifact_root.mkdir(parents=True, exist_ok=True)
    with (artifact_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    full = [row for row in rows if row["tag"].startswith("full-combination-seed")]
    blend = [row for row in rows if row["tag"].startswith("blend-screen-seed")]
    lines = [
        "# Titans + NCM no-replay adaptation campaign",
        "",
        f"- Selected experimental gamma: `{gamma:g}`",
        f"- Transport: `{selection['transport']}`",
        f"- Fixed-anchor distillation weight: `{selection['distillation_weight']:g}`",
        f"- Selection protocol: {selection['selection_protocol']}",
        "",
        "| Group | Valid seeds | Accuracy mean +/- std | Forgetting mean +/- std |",
        "|---|---:|---:|---:|",
    ]
    for name, group in (("Blend", blend), ("Full combination", full)):
        acc = [float(row["average_accuracy"]) for row in group]
        fgt = [float(row["average_forgetting"]) for row in group]
        acc_std = statistics.stdev(acc) if len(acc) > 1 else 0.0
        fgt_std = statistics.stdev(fgt) if len(fgt) > 1 else 0.0
        lines.append(
            f"| {name} | {len(group)} | {statistics.fmean(acc):.4f} +/- {acc_std:.4f} | "
            f"{statistics.fmean(fgt):.4f} +/- {fgt_std:.4f} |"
        )
    lines.extend(
        [
            "",
            "Baselines đã xác nhận trước campaign: Frozen NCM `0.7114`, Titans NCM Online "
            "`0.6190`, Post-hoc oracle `0.7459`.",
            "",
            "Kết luận khoa học cuối chỉ được viết sau khi đủ 3 seed và hậu kiểm no-replay/NaN.",
        ]
    )
    report = artifact_root / "REPORT_TITANS_NCM_NO_REPLAY.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[REPORT] {report}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "check",
            "smoke",
            "blend",
            "screen",
            "replicate",
            "report",
            "confirm-gamma025",
            "confirm-report",
            "confirm",
            "all",
        ),
    )
    parser.add_argument("--artifact-root", type=pathlib.Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()

    success = False
    try:
        if args.phase in ("check", "confirm", "all"):
            phase_check(args.artifact_root, dry_run=args.dry_run)
        if args.phase in ("smoke", "all"):
            phase_smoke(args.artifact_root, dry_run=args.dry_run)
        if args.phase == "blend":
            phase_blend(args.artifact_root, dry_run=args.dry_run)
        if args.phase in ("screen", "all"):
            phase_screen(args.artifact_root, dry_run=args.dry_run)
        if args.phase in ("replicate", "all"):
            phase_replicate(args.artifact_root, dry_run=args.dry_run)
        if args.phase in ("report", "all") and not args.dry_run:
            phase_report(args.artifact_root)
        if args.phase in ("confirm-gamma025", "confirm"):
            phase_confirm_gamma025(args.artifact_root, dry_run=args.dry_run)
        if args.phase in ("confirm-report", "confirm") and not args.dry_run:
            phase_confirm_report(args.artifact_root)
        success = True
        return 0
    finally:
        if args.shutdown and success and not args.dry_run:
            print("[SHUTDOWN] campaign hoàn tất; tắt VM để ngừng tính GPU/CPU", flush=True)
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
