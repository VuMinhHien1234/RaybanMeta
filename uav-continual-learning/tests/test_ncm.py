"""Test baseline NCM (tự skip nếu chưa có torch): kỳ vọng cốt lõi = KHÔNG quên."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from uavcl.data import build_stream, get_source           # noqa: E402
from uavcl.data.loaders import build_task_loaders          # noqa: E402
from uavcl.engine import run_continual                     # noqa: E402
from uavcl.methods import build_method                     # noqa: E402
from uavcl.metrics import average_forgetting               # noqa: E402
from uavcl.models import NCMClassifier, PrototypeHead, build_backbone  # noqa: E402
from uavcl.models.classifier import MASK_FILL              # noqa: E402

DATA_CFG = {
    "name": "synthetic", "num_classes": 4, "train_per_class": 16,
    "val_per_class": 4, "test_per_class": 4, "split_seed": 0,
    "num_tasks": 2, "image_size": 32, "batch_size": 16, "num_workers": 0,
}


def test_ncm_no_forgetting_and_bounded():
    torch.manual_seed(0)
    source = get_source(DATA_CFG)
    stream = build_stream(
        source.splits["train"].labels, source.splits["val"].labels,
        source.splits["test"].labels, num_classes=source.num_classes, num_tasks=2, seed=0,
    )
    loaders = build_task_loaders(source, stream, DATA_CFG)
    backbone, dim = build_backbone({"name": "tinycnn"})
    model = NCMClassifier(backbone, dim, source.num_classes)
    method = build_method("ncm", {})

    # backbone bị đóng băng thật
    assert all(not p.requires_grad for p in model.backbone.parameters())

    R, _ = run_continual(model, method, stream, loaders, torch.device("cpu"),
                         {"epochs_per_task": 1}, verbose=False)
    assert R.shape == (2, 2)
    # prototype class cũ không bị ghi đè -> quên phải ~0 trên dữ liệu giả tách màu
    assert average_forgetting(R) <= 0.05
    assert R[1, 0] >= R[0, 0] - 0.05

    # bộ nhớ thêm bị chặn: C x D (+ C bộ đếm)
    expected = source.num_classes * dim + source.num_classes
    assert method.footprint_floats(model) == expected
    assert np.isfinite(R).all()


def test_prototype_head_matches_manual_class_means_and_masks_unseen():
    head = PrototypeHead(feat_dim=2, num_classes=3)
    feats = torch.tensor([[3.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
    labels = torch.tensor([0, 0, 1])
    head.update(feats, labels)

    expected = torch.tensor([[2.0**-0.5, 2.0**-0.5], [1.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(head.prototypes(), expected, atol=1e-6)
    logits = head.logits(torch.tensor([[1.0, 0.0]]))
    assert logits.shape == (1, 3)
    assert logits[0, 2] == MASK_FILL
    assert head.proto_count.tolist() == [2.0, 1.0, 0.0]


def test_prototype_head_update_is_batch_partition_invariant():
    torch.manual_seed(2)
    feats = torch.randn(12, 5)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    whole = PrototypeHead(5, 3)
    split = PrototypeHead(5, 3)
    whole.update(feats, labels)
    split.update(feats[:5], labels[:5])
    split.update(feats[5:], labels[5:])
    assert torch.allclose(whole.proto_sum, split.proto_sum, atol=1e-6)
    assert torch.equal(whole.proto_count, split.proto_count)


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([[float("nan"), 0.0]]),
        torch.tensor([[float("inf"), 0.0]]),
    ],
)
def test_prototype_head_rejects_nonfinite_features(bad):
    head = PrototypeHead(2, 2)
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        head.update(bad, torch.tensor([0]))


def test_prototype_head_rejects_invalid_labels_and_empty_allowed():
    head = PrototypeHead(2, 2)
    with pytest.raises(ValueError, match="label"):
        head.update(torch.ones(1, 2), torch.tensor([2]))
    head.update(torch.ones(1, 2), torch.tensor([0]))
    with pytest.raises(RuntimeError, match="chưa có prototype"):
        head.logits(torch.ones(1, 2), allowed=[1])


def test_prototype_head_state_dict_round_trip():
    head = PrototypeHead(4, 3)
    head.update(torch.randn(8, 4), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
    restored = PrototypeHead(4, 3)
    restored.load_state_dict(head.state_dict())
    query = torch.randn(5, 4)
    assert torch.equal(head.proto_count, restored.proto_count)
    assert torch.allclose(head.logits(query), restored.logits(query))
