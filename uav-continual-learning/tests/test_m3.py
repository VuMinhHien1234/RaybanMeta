"""Test M3 optimizer (cổng bắt buộc trước khi dùng thật — PLAN_G4 §2):
hội tụ trên bài toán lồi + chạy đúng với mọi shape tham số. Tự skip nếu thiếu torch."""
import pytest

torch = pytest.importorskip("torch")

from uavcl.optim import build_optimizer                    # noqa: E402
from uavcl.optim.m3 import M3, newton_schulz               # noqa: E402


def _quadratic_run(opt_factory, steps=300, seed=0):
    """Bài toán lồi: min ||A x - b||^2. Trả (loss đầu, loss cuối)."""
    torch.manual_seed(seed)
    A = torch.randn(20, 10)
    b = torch.randn(20)
    x = torch.nn.Parameter(torch.zeros(10))
    opt = opt_factory([x])
    first = last = None
    for _ in range(steps):
        opt.zero_grad()
        loss = ((A @ x - b) ** 2).mean()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
        last = float(loss)
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


def test_m3_paper_mode_also_decreases():
    first, last = _quadratic_run(lambda ps: M3(ps, lr=0.05, beta_style="paper"))
    assert last < first * 0.5, f"M3(paper) không giảm loss: {first:.4f} -> {last:.4f}"


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


def test_newton_schulz_orthogonalizes():
    m = torch.randn(16, 16)
    o = newton_schulz(m, steps=5)
    eye_err = (o @ o.t() - torch.eye(16)).abs().max()
    assert eye_err < 0.35            # gần trực giao là đạt (NS xấp xỉ, không exact)
    v = torch.randn(7)               # 1D: đi thẳng, không NS
    assert torch.equal(newton_schulz(v), v)


def test_build_optimizer_dispatch():
    p = [torch.nn.Parameter(torch.randn(3, 3))]
    assert isinstance(build_optimizer(p, {"optimizer": "adamw", "lr": 1e-3}), torch.optim.AdamW)
    assert isinstance(build_optimizer(p, {"optimizer": "m3", "lr": 1e-3}), M3)
    with pytest.raises(KeyError):
        build_optimizer(p, {"optimizer": "sgd9000"})
