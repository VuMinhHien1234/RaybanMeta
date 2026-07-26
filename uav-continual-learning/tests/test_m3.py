"""Test M3 optimizer (cổng bắt buộc trước khi dùng thật — PLAN_G4 §2):
hội tụ trên bài toán lồi + chạy đúng với mọi shape tham số. Tự skip nếu thiếu torch."""
import pytest

torch = pytest.importorskip("torch")

from uavcl.optim import build_optimizer                    # noqa: E402
from uavcl.optim.m3 import M3, newton_schulz               # noqa: E402


def _quadratic_run(opt_factory, steps=400, seed=0, lr_drop_at=200, lr_drop_to=0.01):
    """Bài toán lồi: min ||A x - b||^2, có GIẢM LR giữa chừng.

    Lưu ý bản chất: M3 (như Muon/Lion) cho bước ~hằng số khi lr cố định ->
    dừng ở "sàn dao động" cỡ lr; muốn về sát nghiệm phải decay lr — đây là
    hành vi đúng của lớp optimizer này, không phải bug."""
    torch.manual_seed(seed)
    A = torch.randn(20, 10)
    # b PHẢI nằm trong không gian cột của A (b = A @ x_true) để nghiệm loss=0 tồn tại.
    # (Bản cũ b ngẫu nhiên -> hệ 20 pt/10 ẩn quá xác định, loss tối ưu = 0.2358 != 0
    #  -> test đòi hỏi điều bất khả thi; mọi optimizer kể cả AdamW đều "trượt".)
    b = A @ torch.randn(10)
    x = torch.nn.Parameter(torch.zeros(10))
    opt = opt_factory([x])
    first = last = None
    for i in range(steps):
        if i == lr_drop_at:
            for g in opt.param_groups:
                g["lr"] = lr_drop_to
        opt.zero_grad()
        loss = ((A @ x - b) ** 2).mean()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
        last = float(loss.detach())
    return first, last


def test_m3_converges_on_convex():
    first, last = _quadratic_run(lambda ps: M3(ps, lr=0.05))
    assert last < first * 0.1, f"M3 không hội tụ: {first:.4f} -> {last:.4f}"


def test_m3_delta_mode_converges():
    """Delta Momentum (mặc định dự án) phải hội tụ trên bài toán lồi."""
    first, last = _quadratic_run(lambda ps: M3(ps, lr=0.05, beta_style="delta"))
    assert last < first * 0.1, f"M3(delta) không hội tụ: {first:.4f} -> {last:.4f}"


def test_delta_reduces_to_ema_when_alpha_is_one():
    """α=1, η=1−β: delta ≡ ema — kiểm chứng bằng quỹ đạo tham số y hệt nhau."""
    def run(style, **kw):
        torch.manual_seed(0)
        A, b = torch.randn(8, 4), torch.randn(8)
        x = torch.nn.Parameter(torch.zeros(4))
        opt = M3([x], lr=0.05, beta_style=style, **kw)
        for _ in range(30):
            opt.zero_grad()
            ((A @ x - b) ** 2).mean().backward()
            opt.step()
        return x.detach().clone()

    beta1, beta3 = 0.9, 0.95
    x_ema = run("ema", betas=(beta1, 0.999, beta3))
    x_delta = run("delta", betas=(beta1, 0.999, beta3),
                  delta_alpha=(1.0, 1.0), delta_eta=(1.0 - beta1, 1.0 - beta3))
    assert torch.allclose(x_ema, x_delta, atol=1e-6)


def test_delta_forget_gate_validation():
    with pytest.raises(ValueError):
        M3([torch.nn.Parameter(torch.zeros(2))], delta_alpha=(0.5, 0.9), delta_eta=(0.9, 0.1))


# --------------------------------------------------------------- key_proj_eta (fix 07-18)
# Xấp xỉ rank-1 của P_i (Eq.48-49) — "sai khác #1" trong LOGIC_NESTED_LEARNING.md §5.
def test_key_proj_eta_default_is_zero_no_behavior_change():
    """Mặc định key_proj_eta=0.0 -> quỹ đạo y hệt code trước khi thêm tính năng này."""
    def run(**kw):
        torch.manual_seed(1)
        A, b = torch.randn(8, 4), torch.randn(8)
        x = torch.nn.Parameter(torch.zeros(4))
        opt = M3([x], lr=0.05, **kw)
        for _ in range(20):
            opt.zero_grad()
            ((A @ x - b) ** 2).mean().backward()
            opt.step()
        return x.detach().clone()

    assert torch.allclose(run(), run(key_proj_eta=0.0))


def test_key_proj_eta_removes_aligned_component_each_step():
    """Gradient LẶP LẠI đúng một hướng: m1 luôn nằm dọc đúng hướng đó, nên key_proj_eta=1.0
    (chiếu trực giao toàn phần) phải đưa m1 về ~0 sau MỖI bước — khác hẳn key_proj_eta=0
    (m1 hội tụ về điểm cố định khác 0 của delta-rule thường). Đây chính là 'quên có chọn
    lọc theo hướng x_t' mà bản α vô hướng cũ không làm được."""
    g = torch.tensor([2.0, 0.0, 0.0, 0.0])

    def run(key_proj_eta, steps=10):
        x = torch.nn.Parameter(torch.zeros(4))
        opt = M3([x], lr=0.01, key_proj_eta=key_proj_eta)
        for _ in range(steps):
            opt.zero_grad()
            x.grad = g.clone()
            opt.step()
        return opt.state[x]["m1"].clone()

    m1_off = run(0.0)
    m1_on = run(1.0)
    assert m1_off.norm() > 1e-3   # hành vi cũ: m1 hội tụ về điểm cố định khác 0
    assert m1_on.norm() < 1e-5    # bật hẳn: m1 bị "xoá dọc hướng g" lại từ đầu mỗi bước


def test_key_proj_eta_out_of_range_raises():
    with pytest.raises(ValueError):
        M3([torch.nn.Parameter(torch.zeros(3))], key_proj_eta=1.5)
    with pytest.raises(ValueError):
        M3([torch.nn.Parameter(torch.zeros(3))], key_proj_eta=-0.1)


def test_m3_converges_with_key_proj_eta_on():
    first, last = _quadratic_run(lambda ps: M3(ps, lr=0.05, key_proj_eta=0.5))
    assert last < first * 0.1, f"M3(key_proj_eta=0.5) không hội tụ: {first:.4f} -> {last:.4f}"


def test_m3_paper_mode_stable_on_matrix():
    """Chế độ 'paper' (tích lũy KHÔNG suy giảm — nguyên văn Algorithm 1) chỉ ổn định
    khi Newton–Schulz có tác dụng, tức tham số dạng MA TRẬN (NS chuẩn hoá hướng).
    Trên vector 1D nó phân kỳ — đó là phát hiện đáng ghi vào báo cáo, không phải bug test.
    Ở đây chỉ yêu cầu: chạy trên ma trận -> hữu hạn và có giảm."""
    import math

    torch.manual_seed(0)
    A, B = torch.randn(8, 6), torch.randn(8, 4)
    W = torch.nn.Parameter(torch.zeros(6, 4))
    opt = M3([W], lr=0.02, beta_style="paper")
    first = last = None
    for _ in range(200):
        opt.zero_grad()
        loss = ((A @ W - B) ** 2).mean()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.detach())
        last = float(loss.detach())
    assert math.isfinite(last) and last < first, f"M3(paper) bất ổn: {first:.4f} -> {last:.4f}"


def test_paper_one_step_and_second_step_match_algorithm_1():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = M3([p], lr=0.0, betas=(0.5, 0.25, 0.75), beta_style="paper",
             update_norm="none", frequency=8)
    g1 = torch.tensor([2.0, -4.0])
    g2 = torch.tensor([1.0, 3.0])
    p.grad = g1.clone()
    opt.step()
    state = opt.state[p]
    assert torch.allclose(state["m1"], 0.5 * g1)
    assert torch.allclose(state["v"], 0.25 * g1.square())
    p.grad = g2.clone()
    opt.step()
    assert torch.allclose(state["m1"], 0.5 * (g1 + g2))
    assert torch.allclose(state["v"], 0.25 * (g1.square() + g2.square()))


def test_paper_slow_memory_uses_each_chunk_exactly_once():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = M3([p], lr=0.0, betas=(0.9, 0.999, 0.5), beta_style="paper",
             update_norm="none", frequency=2)
    grads = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
        torch.tensor([5.0, 6.0]),
        torch.tensor([7.0, 8.0]),
    ]
    p.grad = grads[0]; opt.step()
    assert torch.count_nonzero(opt.state[p]["m2"]) == 0
    p.grad = grads[1]; opt.step()
    assert torch.allclose(opt.state[p]["m2"], 0.5 * (grads[0] + grads[1]))
    p.grad = grads[2]; opt.step()
    assert torch.allclose(opt.state[p]["m2"], 0.5 * (grads[0] + grads[1]))
    p.grad = grads[3]; opt.step()
    assert torch.allclose(opt.state[p]["m2"], 0.5 * sum(grads))
    assert torch.count_nonzero(opt.state[p]["chunk_sum"]) == 0


def test_paper_next_chunk_timing_does_not_leak_o2_into_boundary_step():
    def run(timing, steps):
        p = torch.nn.Parameter(torch.zeros(1, 1))
        opt = M3(
            [p], lr=0.1, betas=(0.0, 1.0, 1.0), alpha=1.0,
            beta_style="paper", update_norm="none", frequency=2,
            ns_steps=0, paper_timing=timing,
        )
        for _ in range(steps):
            p.grad = torch.ones_like(p)
            opt.step()
        return float(p)

    # O2 của g1+g2 chỉ được dùng từ step 3 (chunk kế tiếp).
    assert run("next_chunk", 2) == pytest.approx(0.0, abs=1e-8)
    assert run("next_chunk", 3) == pytest.approx(-0.2 / (3.0 ** 0.5 + 1e-8))
    # Mode legacy được giữ để tái lập artifact cũ và dùng O2 ngay tại boundary.
    assert run("legacy_boundary", 2) < -0.1


def test_paper_does_not_bias_correct_v():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = M3([p], lr=0.1, betas=(0.9, 0.5, 0.95), beta_style="paper",
             update_norm="none")
    p.grad = torch.tensor([2.0])
    opt.step()
    expected = -0.1 * (0.9 * 2.0) / ((0.5 * 4.0) ** 0.5 + 1e-8)
    assert float(p) == pytest.approx(expected, rel=1e-6)


def test_paper_strict_does_not_clip_raw_update():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = M3([p], lr=0.01, betas=(1.0, 0.25, 0.95),
             beta_style="paper", update_norm="none")
    p.grad = torch.tensor([1e-6])
    opt.step()
    assert abs(float(p)) > 0.01


def test_delta_clip_baseline_regression_trajectory():
    """Đóng băng P0 tại commit nền 4b9a736 trên chuỗi gradient cố định."""
    p = torch.nn.Parameter(torch.tensor([[0.2, -0.1], [0.3, 0.4]]))
    opt = M3([p], lr=0.003, beta_style="delta", update_norm="clip",
             frequency=2, ns_steps=3, weight_decay=0.01)
    grads = [
        torch.tensor([[0.01, -0.02], [0.03, -0.04]]),
        torch.tensor([[-0.03, 0.01], [0.02, 0.05]]),
        torch.tensor([[0.02, 0.02], [-0.01, 0.03]]),
    ]
    for grad in grads:
        p.grad = grad
        opt.step()
    expected = torch.tensor([
        [0.2040200680, -0.0964208469],
        [0.2951953709, 0.3997883499],
    ])
    assert torch.allclose(p, expected, atol=1e-8, rtol=1e-6)


def test_m3_diagnostics_report_raw_and_post_clip_norms():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    opt = M3([p], lr=0.01, diagnostics=True)
    p.grad = torch.full_like(p, 1e-4)
    opt.record_external_grad_norm(2e-4, 2e-4)
    opt.step()
    diagnostics = opt.diagnostics()
    assert diagnostics["tensor_updates"] == 1
    assert diagnostics["summary"]["raw_step_norm"]["count"] == 1
    assert diagnostics["summary"]["post_step_norm"]["max"] <= 1.0 + 1e-6
    assert diagnostics["summary"]["grad_norm_before_clip"]["count"] == 1


def test_m3_handles_all_param_shapes():
    """1D (bias/norm), 2D (linear), 4D (conv) đều phải bước được, không NaN."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Flatten(), torch.nn.LayerNorm(4 * 8 * 8), torch.nn.Linear(4 * 8 * 8, 5),
    )
    opt = M3(net.parameters(), lr=1e-2, frequency=2)
    for _ in range(5):
        opt.zero_grad()
        out = net(torch.randn(6, 3, 8, 8))
        loss = torch.nn.functional.cross_entropy(out, torch.randint(0, 5, (6,)))
        loss.backward()
        opt.step()
    for p in net.parameters():
        assert torch.isfinite(p).all()


def test_matrix_path_activates_slow_memory_and_reduces_loss():
    """Test hội tụ phải đi qua nhánh ma trận, NS và ít nhất một update M2."""
    torch.manual_seed(7)
    x = torch.randn(96, 12)
    target = x @ torch.randn(12, 4)
    weight = torch.nn.Parameter(torch.zeros(12, 4))
    # Clip-mode giới hạn norm toàn tensor, nên cần LR riêng lớn hơn AdamW-scale.
    opt = M3([weight], lr=5e-2, frequency=4)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = ((x @ weight - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    state = opt.state[weight]
    assert state["step"] == 200
    assert state["m2"].norm() > 0
    assert losses[-1] < losses[0] * 0.1


def test_m3_rejects_nonfinite_gradient():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    opt = M3([p])
    p.grad = torch.full_like(p, float("nan"))
    with pytest.raises(FloatingPointError, match="gradient"):
        opt.step()


@pytest.mark.parametrize("kwargs", [{"frequency": 0}, {"ns_steps": -1}, {"eps": 0.0}])
def test_m3_validates_numerical_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        M3([torch.nn.Parameter(torch.zeros(2, 2))], **kwargs)


def test_matrix_step_size_follows_lr():
    """Bug thật từ VM (‖Δw‖ 1023%/task): chế độ nguyên văn khuếch đại bước đi hàng trăm
    lần lr khi gradient nhỏ. Chế độ legacy 'rms' phải giữ RMS bước ma trận ≈ lr."""
    torch.manual_seed(0)
    g = torch.randn(8, 8) * 1e-4          # gradient rất nhỏ — kịch bản ViT pretrained

    W1 = torch.nn.Parameter(torch.zeros(8, 8))
    opt1 = M3([W1], lr=0.01, update_norm="rms")
    W1.grad = g.clone()
    opt1.step()
    rms_norm = (W1.detach()).pow(2).mean().sqrt()
    assert 0.002 < float(rms_norm) < 0.05, f"bước rms-mode lệch lr: {float(rms_norm):.4f}"

    W2 = torch.nn.Parameter(torch.zeros(8, 8))
    opt2 = M3([W2], lr=0.01, update_norm="none")   # nguyên văn dòng 10
    W2.grad = g.clone()
    opt2.step()
    rms_none = (W2.detach()).pow(2).mean().sqrt()
    # chứng minh hiện tượng khuếch đại (chính là phát hiện ghi vào báo cáo)
    assert float(rms_none) > float(rms_norm) * 20


def test_clip_mode_caps_update_without_forcing_matrix_rms():
    """Default clip follows the reference: ||delta|| <= lr for every tensor."""
    torch.manual_seed(0)
    grad = torch.randn(8, 8) * 1e-4

    clipped = torch.nn.Parameter(torch.zeros(8, 8))
    opt = M3([clipped], lr=0.01)
    clipped.grad = grad.clone()
    opt.step()

    legacy = torch.nn.Parameter(torch.zeros(8, 8))
    opt_legacy = M3([legacy], lr=0.01, update_norm="rms")
    legacy.grad = grad.clone()
    opt_legacy.step()

    assert float(clipped.norm()) <= 0.01 * (1.0 + 1e-5)
    assert float(clipped.norm()) < float(legacy.norm()) / 4


def test_newton_schulz_orthogonalizes():
    """NS của Muon là XẤP XỈ (hệ số tối ưu cho tốc độ, không cho độ chính xác) —
    tiêu chí đúng: gần trực giao HƠN HẲN ma trận thô, không phải trực giao tuyệt đối."""
    torch.manual_seed(0)
    m = torch.randn(16, 16)
    eye = torch.eye(16)
    raw_err = (m @ m.t() - eye).abs().max()
    o = newton_schulz(m, steps=6)
    ns_err = (o @ o.t() - eye).abs().max()
    assert ns_err < 0.7                      # sát trực giao ở mức xấp xỉ
    assert ns_err < raw_err * 0.1            # và tốt hơn ma trận thô >= 10 lần
    v = torch.randn(7)                       # 1D: đi thẳng, không NS
    assert torch.equal(newton_schulz(v), v)


def test_build_optimizer_dispatch():
    p = [torch.nn.Parameter(torch.randn(3, 3))]
    assert isinstance(build_optimizer(p, {"optimizer": "adamw", "lr": 1e-3}), torch.optim.AdamW)
    assert isinstance(build_optimizer(p, {"optimizer": "m3", "lr": 1e-3}), M3)
    with pytest.raises(KeyError):
        build_optimizer(p, {"optimizer": "sgd9000"})
