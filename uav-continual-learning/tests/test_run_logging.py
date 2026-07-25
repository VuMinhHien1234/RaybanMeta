import importlib.util
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
