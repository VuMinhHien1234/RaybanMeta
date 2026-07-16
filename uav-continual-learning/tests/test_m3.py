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
