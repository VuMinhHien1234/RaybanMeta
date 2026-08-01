import importlib.util
import json
from pathlib import Path


def _run_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_g1.py"
    spec = importlib.util.spec_from_file_location("run_g1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_m3_run_name_separates_learning_rates():
    run_g1 = _run_module()
    base = {
        "seed": 0,
        "data": {"name": "eurosat"},
        "memory": {"enabled": True, "reset": "never"},
        "train": {
            "optimizer": "m3",
            "optimizer_per_task": False,
            "lr": 1e-4,
            "m3": {"beta_style": "ema", "frequency": 16},
        },
    }
    name1 = run_g1.run_dir_name(base, "hope")
    base["train"]["lr"] = 3e-4
    name2 = run_g1.run_dir_name(base, "hope")
    assert name1 != name2
    assert "lr0.0001" in name1
    assert "lr0.0003" in name2


def test_m3_run_name_separates_nondefault_alpha():
    run_g1 = _run_module()
    base = {
        "seed": 0,
        "data": {"name": "eurosat"},
        "train": {
            "optimizer": "m3",
            "lr": 3e-3,
            "m3": {"beta_style": "delta", "frequency": 16, "alpha": 0.5},
        },
    }
    default_name = run_g1.run_dir_name(base, "hope")
    base["train"]["m3"]["alpha"] = 0.1
    ablation_name = run_g1.run_dir_name(base, "hope")
    assert default_name != ablation_name
    assert "_a0.1" in ablation_name


def test_run_name_separates_study_tags():
    run_g1 = _run_module()
    base = {
        "seed": 1,
        "data": {"name": "resisc45"},
        "log": {"run_tag": "legacy"},
        "train": {"optimizer": "m3", "lr": 1e-3, "m3": {"update_norm": "rms"}},
    }
    legacy = run_g1.run_dir_name(base, "titans")
    base["log"]["run_tag"] = "clip-carry"
    improved = run_g1.run_dir_name(base, "titans")
    assert legacy != improved
    assert "_legacy_" in legacy
    assert "_clip-carry_" in improved


def test_run_name_rejects_empty_sanitized_tag():
    import pytest

    run_g1 = _run_module()
    cfg = {
        "seed": 0,
        "data": {"name": "synthetic"},
        "log": {"run_tag": "..."},
        "train": {"optimizer": "adamw"},
    }
    with pytest.raises(ValueError, match="run_tag"):
        run_g1.run_dir_name(cfg, "titans")


def test_result_complete_requires_ncm_outputs_when_enabled(tmp_path):
    run_g1 = _run_module()
    cfg = {"train": {"eval_ncm_head": True}}
    for name in ("metrics.json", "config.yaml", "acc_matrix.csv", "train_log.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert not run_g1.result_is_complete(tmp_path, cfg)
    (tmp_path / "metrics_ncm.json").write_text("{}", encoding="utf-8")
    (tmp_path / "acc_matrix_ncm.csv").write_text("", encoding="utf-8")
    assert run_g1.result_is_complete(tmp_path, cfg)


def test_result_complete_requires_named_ncm_outputs_and_checkpoint(tmp_path):
    run_g1 = _run_module()
    cfg = {
        "train": {
            "ncm": {
                "enabled": True,
                "readouts": ["online_current_task", "posthoc_full_seen_train"],
                "save_state": True,
            }
        }
    }
    for name in ("metrics.json", "config.yaml", "acc_matrix.csv", "train_log.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert not run_g1.result_is_complete(tmp_path, cfg)
    for name in (
        "metrics_ncm_online.json",
        "acc_matrix_ncm_online.csv",
        "metrics_ncm_posthoc.json",
        "acc_matrix_ncm_posthoc.csv",
        "checkpoint.pt",
    ):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert run_g1.result_is_complete(tmp_path, cfg)


def test_terminal_scientific_failure_only_accepts_floating_point(tmp_path):
    run_g1 = _run_module()
    failure_path = tmp_path / "failure.json"
    failure_path.write_text(
        json.dumps({"exception": "FloatingPointError", "message": "state NaN"}),
        encoding="utf-8",
    )
    assert run_g1.has_terminal_scientific_failure(tmp_path)

    failure_path.write_text(
        json.dumps({"exception": "RuntimeError", "message": "temporary I/O"}),
        encoding="utf-8",
    )
    assert not run_g1.has_terminal_scientific_failure(tmp_path)
