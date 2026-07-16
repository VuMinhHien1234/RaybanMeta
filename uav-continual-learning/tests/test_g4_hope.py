"""Test G4 — HOPE wiring (cần torch + timm + titans-pytorch; tự skip nếu thiếu).

Kiểm: forward ra logits đúng shape; backbone MỞ BĂNG nhưng nền bị CMS đóng lại;
memory nằm trong tier NHANH nhất; state sống xuyên forward khi reset=never.
"""
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
pytest.importorskip("titans_pytorch")

from uavcl.models import HOPEClassifier                      # noqa: E402
from uavcl.models.cms import build_cms_param_groups          # noqa: E402

MEM = {"enabled": True, "dim": "auto", "chunk_size": 4, "seq": "image_seq", "reset": "never"}
CMS_CFG = {"tiers": [[4, 1], [4, 4], [4, 16]], "etas": [1.0, 0.5, 0.1],
           "order": "late_slow", "attn": "slow"}


def _model(num_classes=6):
    torch.manual_seed(0)
    backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    return HOPEClassifier(backbone, backbone.num_features, num_classes, MEM)


def test_forward_shape_and_state_carries():
    m = _model()
    m.train()
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    assert out.shape == (2, 6)
    assert m._state is not None          # reset=never: state sống sau forward train
    n1 = m.state_norm()
    m(x)
    assert m._state is not None and m.state_norm() >= 0.0
    assert n1 >= 0.0


def test_backbone_unfrozen_but_base_frozen_by_cms():
    m = _model()
    # trước khi CMS can thiệp: backbone mở băng (khác TitansClassifier)
    assert any(p.requires_grad for p in m.backbone.blocks[0].mlp.parameters())
    groups = build_cms_param_groups(m, CMS_CFG)
    # nền pretrained bị CMS đóng băng lại
    assert all(not p.requires_grad for p in m.backbone.patch_embed.parameters())
    # memory + head + post_norm nằm ở tier NHANH nhất
    fast_ids = {id(p) for p in groups[0]["params"]}
    assert {id(p) for p in m.memory.parameters()} <= fast_ids
    assert {id(p) for p in m.head.parameters()} <= fast_ids
    assert {id(p) for p in m.post_norm.parameters()} <= fast_ids
    # attn ở tier chậm nhất (quyết định team)
    assert {id(p) for p in m.backbone.blocks[0].attn.parameters()} <= {id(p) for p in groups[-1]["params"]}


def test_eval_does_not_touch_state():
    m = _model()
    m.train()
    m(torch.randn(2, 3, 224, 224))
    norm_before = m.state_norm()
    m.eval()
    with torch.no_grad():
        o1 = m(torch.randn(2, 3, 224, 224))
        o2 = m(torch.randn(2, 3, 224, 224))
    assert m.state_norm() == pytest.approx(norm_before)  # chấm thi không ghi trí nhớ
    assert torch.isfinite(o1).all() and torch.isfinite(o2).all()


def test_method_hope_registered():
    from uavcl.methods import build_method

    method = build_method("hope", {})
    assert method.name == "hope"
    assert hasattr(method, "begin_task") and hasattr(method, "end_task")
