import importlib.util
import json
from pathlib import Path

import pytest


def _load_campaign_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ncm_adaptation_study.py"
    spec = importlib.util.spec_from_file_location("run_ncm_adaptation_study", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CAMPAIGN = _load_campaign_module()


def _load_run_g1_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_g1.py"
    spec = importlib.util.spec_from_file_location("confirmatory_run_g1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN_G1 = _load_run_g1_module()


def test_confirmatory_phase_locks_gamma_and_fresh_seeds(monkeypatch, tmp_path):
    calls = []

    def fake_run(config_path, artifact_root, tag, overrides, *, dry_run):
        calls.append((tag, overrides, dry_run))
        return tmp_path / tag

    monkeypatch.setattr(CAMPAIGN, "_run_experiment", fake_run)

    results = CAMPAIGN.phase_confirm_gamma025(tmp_path, dry_run=True)

    assert len(results) == 3
    assert [tag for tag, _, _ in calls] == [
        "gamma025-confirm-seed3",
        "gamma025-confirm-seed4",
        "gamma025-confirm-seed5",
    ]
    for seed, (_, overrides, dry_run) in zip((3, 4, 5), calls):
        assert dry_run
        assert f"seed={seed}" in overrides
        assert "train.ncm.blend.gammas=[0,0.25]" in overrides
        assert "train.ncm.transport.enabled=false" in overrides
        assert "train.feature_distillation.enabled=false" in overrides


def test_completion_guard_allows_anchored_run_without_gamma_one(tmp_path):
    cfg = {
        "train": {
            "ncm": {
                "enabled": True,
                "readouts": ["online_current_task"],
                "save_state": True,
                "blend": {"enabled": True, "gammas": [0.0, 0.25]},
            }
        }
    }
    required = (
        "metrics.json",
        "config.yaml",
        "acc_matrix.csv",
        "train_log.json",
        "metrics_ncm_blend_g0.json",
        "acc_matrix_ncm_blend_g0.csv",
        "metrics_ncm_blend_g0p25.json",
        "acc_matrix_ncm_blend_g0p25.csv",
        "checkpoint.pt",
    )
    for name in required:
        (tmp_path / name).touch()

    assert RUN_G1.result_is_complete(tmp_path, cfg)


def test_confirmatory_report_audits_and_summarizes(tmp_path):
    results_root = tmp_path / "results"
    for seed, accuracy in ((3, 0.72), (4, 0.73), (5, 0.71)):
        result = results_root / (
            f"resisc45_titans_seed{seed}_gamma025-confirm-seed{seed}_fixture"
        )
        result.mkdir(parents=True)
        (result / "metrics_ncm_blend_g0.json").write_text(
            json.dumps({"average_accuracy": 0.7114, "average_forgetting": 0.0874}),
            encoding="utf-8",
        )
        (result / "metrics_ncm_blend_g0p25.json").write_text(
            json.dumps(
                {
                    "average_accuracy": accuracy,
                    "average_forgetting": 0.09,
                    "backward_transfer": -0.09,
                }
            ),
            encoding="utf-8",
        )
        diagnostics = {
            str(task): {
                "online_blend": {
                    "g0p25": {
                        "revisit_old_train": False,
                        "old_samples_revisited": 0,
                    }
                }
            }
            for task in range(9)
        }
        (result / "train_log.json").write_text(
            json.dumps(
                {
                    "ncm_diagnostics": diagnostics,
                    "state_finite": {str(task): True for task in range(9)},
                }
            ),
            encoding="utf-8",
        )

    report = CAMPAIGN.phase_confirm_report(tmp_path)

    assert report.exists()
    assert "0.7200 +/- 0.0100" in report.read_text(encoding="utf-8")
    assert (tmp_path / "gamma025_confirmatory_summary.csv").exists()


def test_confirmatory_report_rejects_replay(tmp_path):
    results_root = tmp_path / "results"
    for seed in (3, 4, 5):
        result = results_root / (
            f"resisc45_titans_seed{seed}_gamma025-confirm-seed{seed}_fixture"
        )
        result.mkdir(parents=True)
        for name, payload in (
            ("metrics_ncm_blend_g0.json", {"average_accuracy": 0.7114, "average_forgetting": 0.08}),
            (
                "metrics_ncm_blend_g0p25.json",
                {
                    "average_accuracy": 0.72,
                    "average_forgetting": 0.09,
                    "backward_transfer": -0.09,
                },
            ),
        ):
            (result / name).write_text(json.dumps(payload), encoding="utf-8")
        revisited = 1 if seed == 4 else 0
        (result / "train_log.json").write_text(
            json.dumps(
                {
                    "ncm_diagnostics": {
                        "0": {
                            "online_blend": {
                                "g0p25": {
                                    "revisit_old_train": bool(revisited),
                                    "old_samples_revisited": revisited,
                                }
                            }
                        }
                    },
                    "state_finite": {"0": True},
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match="no_replay=False"):
        CAMPAIGN.phase_confirm_report(tmp_path)
