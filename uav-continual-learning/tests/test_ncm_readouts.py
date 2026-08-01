"""Protocol tests for online and post-hoc NCM readouts on Titans features."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("titans_pytorch")

from uavcl.data import build_stream, get_source  # noqa: E402
from uavcl.data.loaders import build_task_loaders  # noqa: E402
from uavcl.engine import run_continual  # noqa: E402
from uavcl.methods import build_method  # noqa: E402
from uavcl.models import TitansClassifier, build_backbone  # noqa: E402


DATA_CFG = {
    "name": "synthetic",
    "num_classes": 4,
    "train_per_class": 8,
    "val_per_class": 2,
    "test_per_class": 2,
    "split_seed": 0,
    "num_tasks": 2,
    "image_size": 32,
    "batch_size": 4,
    "num_workers": 0,
}
MEM_CFG = {
    "enabled": True,
    "dim": "auto",
    "chunk_size": 4,
    "seq": "image_seq",
    "reset": "never",
}


class OnePassLoader:
    """Allow exactly one full iteration; catches accidental old-task revisits."""

    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.calls = 0

    def __iter__(self):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("old prototype loader was revisited")
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def _setup():
    torch.manual_seed(0)
    source = get_source(DATA_CFG)
    stream = build_stream(
        source.splits["train"].labels,
        source.splits["val"].labels,
        source.splits["test"].labels,
        num_classes=source.num_classes,
        num_tasks=2,
        seed=0,
    )
    loaders = build_task_loaders(source, stream, DATA_CFG)
    backbone, dim = build_backbone({"name": "tinycnn"})
    model = TitansClassifier(backbone, dim, source.num_classes, MEM_CFG)
    return source, stream, loaders, model


def test_online_ncm_reads_each_current_task_once_and_keeps_bounded_state():
    source, stream, loaders, model = _setup()
    guarded = [OnePassLoader(task["prototype"]) for task in loaders]
    for task, loader in zip(loaders, guarded):
        task["prototype"] = loader
    train_cfg = {
        "epochs_per_task": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "ncm": {
            "enabled": True,
            "readouts": ["online_current_task"],
            "feature_protocol": "independent_image",
        },
    }
    matrix, log = run_continual(
        model,
        build_method("titans", {}),
        stream,
        loaders,
        torch.device("cpu"),
        train_cfg,
        verbose=False,
    )

    assert matrix.shape == (2, 2)
    assert np.isfinite(log["ncm_online_R"]).all()
    assert "ncm_posthoc_R" not in log
    assert [loader.calls for loader in guarded] == [1, 1]
    assert int(model.ncm_online_head.proto_count.sum()) == len(source.splits["train"].labels)
    assert model.ncm_online_head.extra_floats() == source.num_classes * (
        model.head.in_features + 1
    )
    assert all(
        not task["online_current_task"]["revisit_old_train"]
        for task in log["ncm_diagnostics"].values()
    )


def test_posthoc_readout_revisits_old_task_loader_by_definition():
    _, stream, loaders, model = _setup()
    loaders[0]["prototype"] = OnePassLoader(loaders[0]["prototype"])
    train_cfg = {
        "epochs_per_task": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "ncm": {
            "enabled": True,
            "readouts": ["posthoc_full_seen_train"],
            "feature_protocol": "independent_image",
        },
    }
    with pytest.raises(RuntimeError, match="revisited"):
        run_continual(
            model,
            build_method("titans", {}),
            stream,
            loaders,
            torch.device("cpu"),
            train_cfg,
            verbose=False,
        )


def test_prototype_loader_is_deterministic_and_uses_train_indices():
    _, stream, loaders, _ = _setup()
    loader = loaders[0]["prototype"]
    first = [(x.clone(), y.clone()) for x, y in loader]
    second = [(x.clone(), y.clone()) for x, y in loader]
    assert loader.dataset.indices == list(stream[0].train_idx)
    assert len(first) == len(second)
    assert all(torch.equal(x1, x2) and torch.equal(y1, y2) for (x1, y1), (x2, y2) in zip(first, second))
    assert sum(y.numel() for _, y in first) == len(stream[0].train_idx)


def test_task_boundary_checkpoint_resumes_completed_stream(tmp_path):
    _, stream, loaders, model = _setup()
    progress = tmp_path / "progress.pt"
    train_cfg = {
        "epochs_per_task": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "optimizer_per_task": False,
        "_progress_checkpoint_path": str(progress),
        "ncm": {
            "enabled": True,
            "readouts": ["online_current_task"],
            "feature_protocol": "independent_image",
        },
    }
    first_matrix, first_log = run_continual(
        model,
        build_method("titans", {}),
        stream,
        loaders,
        torch.device("cpu"),
        train_cfg,
        verbose=False,
    )
    assert progress.exists()

    _, resumed_stream, resumed_loaders, resumed_model = _setup()
    resumed_matrix, resumed_log = run_continual(
        resumed_model,
        build_method("titans", {}),
        resumed_stream,
        resumed_loaders,
        torch.device("cpu"),
        train_cfg,
        verbose=False,
    )
    assert np.array_equal(resumed_matrix, first_matrix)
    assert np.array_equal(resumed_log["ncm_online_R"], first_log["ncm_online_R"])
    assert torch.equal(
        resumed_model.ncm_online_head.proto_count, model.ncm_online_head.proto_count
    )
