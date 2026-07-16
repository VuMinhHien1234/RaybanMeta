"""Test G3 — CMSOptimizer (cơ chế chu kỳ + trung bình grad) và tier mapping ViT.

Phần optimizer test bằng SGD(lr=1) để kiểm số CHÍNH XÁC từng bước.
Tự skip nếu thiếu torch; phần mapping cần timm (pretrained=False, không tải mạng).
"""
import pytest

torch = pytest.importorskip("torch")

from uavcl.optim.cms_optimizer import CMSOptimizer          # noqa: E402


def _tiers(p_fast, p_slow, period_slow=3):
    return [
        {"name": "fast", "period": 1, "eta": 1.0, "params": [p_fast], "blocks": [0]},
        {"name": "slow", "period": period_slow, "eta": 1.0, "params": [p_slow], "blocks": [1]},
    ]


@pytest.mark.parametrize("agg,expected_slow", [("sum", -9.0), ("mean", -3.0)])
def test_cms_period_and_grad_aggregation_exact(agg, expected_slow):
    """SGD lr=1: fast bước mỗi step bằng -grad; slow đứng yên tới chu kỳ, rồi bước
    bằng -TỔNG (Eq. 71 nguyên văn) hoặc -TRUNG BÌNH (biến thể) của 3 grad gần nhất."""
    p_fast = torch.nn.Parameter(torch.zeros(2))
    p_slow = torch.nn.Parameter(torch.zeros(2))
    tiers = _tiers(p_fast, p_slow)
    inner = torch.optim.SGD([{"params": [p_fast], "lr": 1.0}, {"params": [p_slow], "lr": 1.0}])
    opt = CMSOptimizer(inner, tiers, grad_agg=agg)

    grads = [1.0, 2.0, 6.0]  # tổng = 9, trung bình = 3
    for k, gval in enumerate(grads, start=1):
        opt.zero_grad()
        p_fast.grad = torch.full_like(p_fast, gval)
        p_slow.grad = torch.full_like(p_slow, gval)
        opt.step()
        if k < 3:
            assert torch.all(p_slow == 0), f"slow không được động đậy trước chu kỳ (step {k})"
    assert torch.allclose(p_fast, torch.full_like(p_fast, -9.0))
    assert torch.allclose(p_slow, torch.full_like(p_slow, expected_slow))


def test_cms_bad_grad_agg_raises():
    p = torch.nn.Parameter(torch.zeros(1))
    inner = torch.optim.SGD([p], lr=1.0)
    with pytest.raises(ValueError):
        CMSOptimizer(inner, _tiers(p, p), grad_agg="median")


def test_cms_accumulator_resets_after_due_step():
    p_fast = torch.nn.Parameter(torch.zeros(1))
    p_slow = torch.nn.Parameter(torch.zeros(1))
    inner = torch.optim.SGD([{"params": [p_fast], "lr": 1.0}, {"params": [p_slow], "lr": 1.0}])
    opt = CMSOptimizer(inner, _tiers(p_fast, p_slow, period_slow=2), grad_agg="sum")
    for gval in [1.0, 1.0, 5.0, 5.0]:  # 2 chu kỳ: sum(1,1)=2, sum(5,5)=10
        opt.zero_grad()
        p_fast.grad = torch.full_like(p_fast, gval)
        p_slow.grad = torch.full_like(p_slow, gval)
        opt.step()
    assert torch.allclose(p_slow, torch.tensor([-12.0]))  # -2 rồi -10 (không rò chu kỳ cũ)


def test_cms_eta_scales_lr():
    p_fast = torch.nn.Parameter(torch.zeros(1))
    p_slow = torch.nn.Parameter(torch.zeros(1))
    tiers = _tiers(p_fast, p_slow, period_slow=1)
    tiers[1]["eta"] = 0.1
    inner = torch.optim.SGD(
        [{"params": [p_fast], "lr": 1.0}, {"params": [p_slow], "lr": 1.0 * tiers[1]["eta"]}]
    )
    opt = CMSOptimizer(inner, tiers)
    opt.zero_grad()
    p_fast.grad = torch.ones_like(p_fast)
    p_slow.grad = torch.ones_like(p_slow)
    opt.step()
    assert torch.allclose(p_fast, torch.tensor([-1.0]))
    assert torch.allclose(p_slow, torch.tensor([-0.1]))   # η làm bước đi khẽ 10x


# ------------------------------------------------------------------ tier mapping
@pytest.mark.parametrize("order", ["late_slow", "early_slow"])
def test_build_groups_on_vit(order):
    timm = pytest.importorskip("timm")
    from uavcl.models import ContinualClassifier
    from uavcl.models.cms import build_cms_param_groups, tier_report

    backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    model = ContinualClassifier(backbone, backbone.num_features, 10)
    cfg = {"tiers": [[4, 1], [4, 4], [4, 16]], "etas": [1.0, 0.5, 0.1],
           "order": order, "attn": "slow"}
    groups = build_cms_param_groups(model, cfg)

    assert [g["name"] for g in groups] == ["fast", "mid", "slow"]
    if order == "late_slow":
        assert groups[0]["blocks"] == [0, 1, 2, 3] and groups[2]["blocks"] == [8, 9, 10, 11]
    else:
        assert groups[0]["blocks"] == [8, 9, 10, 11] and groups[2]["blocks"] == [0, 1, 2, 3]
    # head nằm ở tier nhanh
    head_ids = {id(p) for p in model.head.parameters()}
    assert head_ids <= {id(p) for p in groups[0]["params"]}
    # attn nằm ở tier chậm (quyết định team)
    attn_ids = {id(p) for p in backbone.blocks[0].attn.parameters()}
    assert attn_ids <= {id(p) for p in groups[2]["params"]}
    # nền pretrained bị đóng băng
    assert all(not p.requires_grad for p in backbone.patch_embed.parameters())
    assert "blocks=" in tier_report(groups)


def test_attn_freeze_fallback():
    timm = pytest.importorskip("timm")
    from uavcl.models import ContinualClassifier
    from uavcl.models.cms import build_cms_param_groups

    backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    model = ContinualClassifier(backbone, backbone.num_features, 10)
    build_cms_param_groups(model, {"attn": "freeze"})
    assert all(not p.requires_grad for p in backbone.blocks[0].attn.parameters())


def test_non_vit_backbone_raises():
    from uavcl.models import ContinualClassifier, build_backbone
    from uavcl.models.cms import build_cms_param_groups

    backbone, dim = build_backbone({"name": "tinycnn"})
    model = ContinualClassifier(backbone, dim, 4)
    with pytest.raises(TypeError):
        build_cms_param_groups(model, {})
