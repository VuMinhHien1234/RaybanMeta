"""Inference checkpoint round-trip for Titans plus online NCM state."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("titans_pytorch")

from uavcl.checkpoint import load_inference_checkpoint, save_inference_checkpoint  # noqa: E402
from uavcl.models import PrototypeHead, TitansClassifier, build_backbone  # noqa: E402


MEM_CFG = {
    "enabled": True,
    "dim": "auto",
    "chunk_size": 4,
    "seq": "image_seq",
    "reset": "never",
}


def _model():
    backbone, dim = build_backbone({"name": "tinycnn"})
    return TitansClassifier(backbone, dim, 3, MEM_CFG)


def test_checkpoint_restores_model_memory_and_online_ncm(tmp_path):
    torch.manual_seed(3)
    model = _model()
    model.ncm_online_head = PrototypeHead(model.head.in_features, model.head.out_features)
    model.train()
    model(torch.randn(4, 3, 32, 32))
    model.eval()
    prototype_images = torch.randn(5, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 0, 1])
    model.ncm_online_head.update(
        model.features(prototype_images, protocol="independent_image"), labels
    )
    query = torch.randn(3, 3, 32, 32)
    expected = model.ncm_online_head.logits(
        model.features(query, protocol="independent_image"), allowed=[0, 1, 2]
    )
    expected_norm = model.state_norm()

    path = tmp_path / "checkpoint.pt"
    save_inference_checkpoint(
        path,
        model,
        config={"train": {"ncm": {"enabled": True}}},
        seen_classes=[0, 1, 2],
        git_commit="test",
    )
    restored = _model()
    payload = load_inference_checkpoint(path, restored)
    restored.eval()
    actual = restored.ncm_online_head.logits(
        restored.features(query, protocol="independent_image"), allowed=[0, 1, 2]
    )

    assert payload["format_version"] == 1
    assert payload["seen_classes"] == [0, 1, 2]
    assert restored.state_norm() == pytest.approx(expected_norm)
    assert torch.equal(
        restored.ncm_online_head.proto_count, model.ncm_online_head.proto_count
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_checkpoint_rejects_unsupported_version(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 99}, path)
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        load_inference_checkpoint(path, _model())


def test_checkpoint_restores_multiple_blend_heads(tmp_path):
    model = _model()
    model.ncm_online_head = PrototypeHead(model.head.in_features, 3)
    model.ncm_blend_heads = torch.nn.ModuleDict(
        {"g0": PrototypeHead(model.head.in_features, 3)}
    )
    model.ncm_transport_heads = torch.nn.ModuleDict(
        {"t0_ridge": PrototypeHead(model.head.in_features, 3)}
    )
    features = torch.randn(6, model.head.in_features)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    model.ncm_online_head.update(features, labels)
    model.ncm_blend_heads["g0"].update(features + 0.1, labels)
    model.ncm_transport_heads["t0_ridge"].update(features - 0.1, labels)

    path = tmp_path / "blend.pt"
    save_inference_checkpoint(
        path,
        model,
        config={
            "train": {
                "ncm": {"blend": {"enabled": True, "gammas": [0.0, 1.0]}}
            }
        },
        seen_classes=[0, 1, 2],
        git_commit="test",
    )
    restored = _model()
    payload = load_inference_checkpoint(path, restored)

    assert "g0" in restored.ncm_blend_heads
    assert torch.equal(
        restored.ncm_blend_heads["g0"].proto_sum,
        model.ncm_blend_heads["g0"].proto_sum,
    )
    assert torch.equal(
        restored.ncm_transport_heads["t0_ridge"].proto_sum,
        model.ncm_transport_heads["t0_ridge"].proto_sum,
    )
    assert payload["ncm_adaptation"]["blend"]["enabled"]
