import importlib.util
import json
from pathlib import Path

import yaml


def _study_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_titan_m3_study.py"
    spec = importlib.util.spec_from_file_location("run_titan_m3_study", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_control_and_treatment_only_differ_by_whitelist():
    study = _study_module()
    study.validate_controlled_configs()


def test_flatten_nested_config():
    study = _study_module()
    assert study._flatten({"train": {"lr": 1e-3}, "seed": 0}) == {
        "train.lr": 1e-3,
        "seed": 0,
    }


def test_shutdown_guard_rejects_local_machine():
    study = _study_module()
    assert isinstance(study._is_google_compute_engine(), bool)


def test_dry_run_summary_does_not_write(tmp_path):
    study = _study_module()
    study.phase_summarize(tmp_path, dry_run=True)
    assert not (tmp_path / "runs.csv").exists()
    assert not (tmp_path / "summary.md").exists()


def test_floating_point_failure_is_terminal(tmp_path):
    study = _study_module()
    failure_path = tmp_path / "failure.json"
    failure_path.write_text(
        json.dumps({"exception": "FloatingPointError", "message": "gradient NaN"}),
        encoding="utf-8",
    )
    assert study._is_terminal_scientific_failure(failure_path)

    failure_path.write_text(
        json.dumps({"exception": "RuntimeError", "message": "serialization bug"}),
        encoding="utf-8",
    )
    assert not study._is_terminal_scientific_failure(failure_path)


def test_collect_results_includes_scientific_failure(tmp_path):
    study = _study_module()
    result_dir = tmp_path / "results" / "failed-run"
    result_dir.mkdir(parents=True)
    (result_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": 1,
                "train": {"lr": 1e-3, "eval_ncm_head": True},
                "log": {"run_tag": "legacy"},
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "failure.json").write_text(
        json.dumps(
            {
                "exception": "FloatingPointError",
                "message": "M3 gradient NaN",
                "runtime_sec": 12.5,
            }
        ),
        encoding="utf-8",
    )

    rows = study.collect_results(tmp_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["valid"] is False
    assert rows[0]["failure"] == "FloatingPointError: M3 gradient NaN"

    study.phase_summarize(tmp_path)
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| legacy | 0.001 | 0/1 |" in summary
    assert "| 1 | - | - | - | - | - |" in summary
