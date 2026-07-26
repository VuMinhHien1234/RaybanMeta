import importlib.util
from pathlib import Path


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
