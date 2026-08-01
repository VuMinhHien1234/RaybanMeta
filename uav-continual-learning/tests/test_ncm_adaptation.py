"""Mechanism and protocol tests for no-replay Titans + NCM adaptation."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("titans_pytorch")

from uavcl.data import build_stream, get_source  # noqa: E402
from uavcl.data.loaders import build_task_loaders  # noqa: E402
from uavcl.engine import run_continual  # noqa: E402
from uavcl.methods import build_method  # noqa: E402
from uavcl.models import PrototypeHead, TitansClassifier, blend_features, build_backbone  # noqa: E402
from uavcl.models.ncm_adaptation import (  # noqa: E402
    fit_transport,
    gamma_key,
    transport_diagnostics,
    transported_features,
)


MEM = {
    "enabled": True,
    "dim": "auto",
    "chunk_size": 4,
    "seq": "image_seq",
    "reset": "never",
}
DATA = {
    "name": "synthetic",
    "num_classes": 4,
    "train_per_class": 8,
    "val_per_class": 4,
    "test_per_class": 4,
    "split_seed": 0,
    "num_tasks": 2,
    "image_size": 32,
    "batch_size": 4,
    "prototype_batch_size": 4,
    "num_workers": 0,
}


class CountedLoader:
    def __init__(self, loader, max_calls):
        self.loader = loader
        self.dataset = loader.dataset
        self.max_calls = max_calls
        self.calls = 0

    def __iter__(self):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError("loader was revisited beyond the current-task protocol")
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def _model():
    backbone, dim = build_backbone({"name": "tinycnn"})
    return TitansClassifier(backbone, dim, 4, MEM)


def _stream_setup():
    source = get_source(DATA)
    stream = build_stream(
        source.splits["train"].labels,
        source.splits["val"].labels,
        source.splits["test"].labels,
        num_classes=source.num_classes,
        num_tasks=2,
        seed=0,
    )
    return source, stream, build_task_loaders(source, stream, DATA)


def test_blend_endpoints_and_one_forward_state_transition():
    torch.manual_seed(5)
    model_a = _model().train()
    model_b = copy.deepcopy(model_a).train()
    x = torch.randn(4, 3, 32, 32)

    bundle = model_a.feature_components(x)
    logits, bundle_b = model_b.forward_with_features(x)

    assert torch.equal(blend_features(bundle.base, bundle.titans, 0.0), bundle.base)
    assert torch.equal(blend_features(bundle.base, bundle.titans, 1.0), bundle.titans)
    assert torch.allclose(logits, model_b.head(bundle_b.titans))
    assert model_a.state_norm() == pytest.approx(model_b.state_norm(), rel=1e-6)


@pytest.mark.parametrize("kind", ["translation", "diagonal", "procrustes", "identity_ridge"])
def test_transport_recovers_synthetic_drift_better_than_identity(kind):
    torch.manual_seed(7)
    x = torch.randn(128, 12)
    q, _ = torch.linalg.qr(torch.randn(12, 12))
    y = torch.nn.functional.normalize(x @ q + 0.05, dim=1)
    transform = fit_transport(x[:96], y[:96], kind=kind, regularization=0.1)
    diag = transport_diagnostics(transform, x[96:], y[96:], beta=1.0)

    assert np.isfinite(list(value for value in diag.values() if isinstance(value, float))).all()
    if kind in ("procrustes", "identity_ridge"):
        assert diag["accepted"]
        assert diag["transport_error"] < diag["identity_error"]


def test_transport_replaces_only_seen_prototypes_and_preserves_counts():
    head = PrototypeHead(3, 4)
    head.update(torch.eye(3), torch.tensor([0, 1, 2]))
    counts = head.proto_count.clone()
    transform = fit_transport(
        torch.eye(3), torch.roll(torch.eye(3), shifts=1, dims=1), kind="procrustes"
    )
    mask = head.seen_mask.clone()
    moved = transported_features(head.prototypes()[mask], transform, beta=1.0)
    head.replace_prototypes(mask, moved)

    assert torch.equal(head.proto_count, counts)
    assert not head.seen_mask[3]
    assert torch.isfinite(head.proto_sum).all()


def test_blend_transport_pipeline_never_reopens_old_prototype_or_val_loader():
    torch.manual_seed(11)
    _, stream, loaders = _stream_setup()
    guarded_prototype = []
    guarded_val = []
    for task_id, task in enumerate(loaders):
        # Task 0 has no old prototypes to transport. Task 1 gets before/after passes.
        p = CountedLoader(task["prototype"], 1 if task_id == 0 else 2)
        v = CountedLoader(task["val"], 1 if task_id == 0 else 2)
        task["prototype"] = p
        task["val"] = v
        guarded_prototype.append(p)
        guarded_val.append(v)

    cfg = {
        "epochs_per_task": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "ncm": {
            "enabled": True,
            "readouts": ["online_current_task"],
            "feature_protocol": "independent_image",
            "blend": {"enabled": True, "gammas": [0.0, 0.5, 1.0]},
            "transport": {
                "enabled": True,
                "type": "identity_ridge",
                "regularization": 1.0,
                "beta": 0.5,
                "safety_gate": True,
            },
        },
    }
    matrix, log = run_continual(
        _model(),
        build_method("titans", {"train": cfg}),
        stream,
        loaders,
        torch.device("cpu"),
        cfg,
        verbose=False,
    )

    assert np.isfinite(matrix).all()
    assert [loader.calls for loader in guarded_prototype] == [1, 2]
    assert [loader.calls for loader in guarded_val] == [1, 2]
    for gamma in (0.0, 0.5, 1.0):
        assert np.isfinite(log[f"ncm_blend_R_{gamma_key(gamma)}"]).all()
    assert all(
        item["old_samples_revisited"] == 0
        for task in log["ncm_diagnostics"].values()
        for item in task["online_blend"].values()
    )


def test_fixed_anchor_distillation_is_finite_and_logged():
    torch.manual_seed(13)
    _, stream, loaders = _stream_setup()
    cfg = {
        "epochs_per_task": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "feature_distillation": {
            "enabled": True,
            "teacher": "frozen_backbone",
            "weight": 0.05,
        },
    }
    _, log = run_continual(
        _model(),
        build_method("titans", {"train": cfg}),
        stream,
        loaders,
        torch.device("cpu"),
        cfg,
        verbose=False,
    )

    assert set(log["feature_distillation"]) == {0, 1}
    assert all(
        value["num_batches"] > 0 and np.isfinite(value["mean_raw_loss"])
        for value in log["feature_distillation"].values()
    )


def test_previous_titans_distillation_starts_after_first_task():
    torch.manual_seed(17)
    _, stream, loaders = _stream_setup()
    cfg = {
        "epochs_per_task": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "feature_distillation": {
            "enabled": True,
            "teacher": "previous_titans",
            "weight": 0.05,
        },
    }
    _, log = run_continual(
        _model(),
        build_method("titans", {"train": cfg}),
        stream,
        loaders,
        torch.device("cpu"),
        cfg,
        verbose=False,
    )

    assert log["feature_distillation"][0]["num_batches"] == 0
    assert log["feature_distillation"][1]["num_batches"] > 0
    assert np.isfinite(log["feature_distillation"][1]["mean_raw_loss"])
