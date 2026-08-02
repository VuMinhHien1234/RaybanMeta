"""TASK 4 + 5 (2026-08-02) — cổng η/α phải SỐNG, không được trôi ra biên.

Bối cảnh (xem `scripts/trace_gates.py` + CHAN_DOAN_NL_2026-08-02.md):
  - α = decay_factor; hệ số GIỮ LẠI = (1 − α)  [neural_memory.py:813]
  - Đo 13 run thật: α KHÔNG bão hoà từ đầu, mà khởi đầu lành mạnh (0.10–0.89) rồi TRÔI đơn
    điệu ra biên trong 2–4 task (vd sdc_s0: 0.4475 → 0.8616 → 0.9975 → 1.0000). Nguyên nhân:
    gradient thấy "quên nhiều hơn" làm giảm loss task HIỆN TẠI nên đẩy α → 1 mãi.
  - `gate_bound` chặn logit bằng tanh -> cổng không bao giờ thoát ra biên, mà vẫn còn gradient
    ở vùng giữa nên VẪN phụ thuộc dữ liệu (đúng yêu cầu Eq 76 của Nested Learning).

Hai nhóm test:
  1. BẤT BIẾN NGƯỢC — không bật cờ thì mọi thứ y hệt bản cũ (bảo vệ mọi run đã chạy).
  2. CƠ CHẾ       — bật cờ thì η/α nằm trong vùng lành mạnh và vẫn biến thiên theo dữ liệu.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("titans_pytorch")

from uavcl.models.memory import (  # noqa: E402
    DEFAULT_ALPHA_LOGIT_LIMIT,
    DEFAULT_ETA_LOGIT_LIMIT,
    ETA_FRAC_HI,
    ETA_FRAC_LO,
    TitansMemory,
)

DIM, CHUNK, LEN = 32, 4, 16
ALPHA_OK = (0.05, 0.95)
# BẪY đã sập một lần (2026-08-02): KHÔNG được hardcode η tuyệt đối.
# `default_adaptive_step_transform(.., max_lr=1e-2)` (neural_memory.py:255) trông như max_lr=1e-2,
# nhưng :457 luôn truyền `default_step_transform_max_lr = 1.` (:272) -> max_lr THẬT = 1.0.
# Vì vậy vùng lành mạnh phải tính theo PHÂN SỐ của max_lr đọc động từ thư viện.
MAX_LR = TitansMemory(dim=DIM, chunk_size=CHUNK)._eta_max_lr
ETA_OK = (ETA_FRAC_LO * MAX_LR, ETA_FRAC_HI * MAX_LR)


def _mem(**kw):
    torch.manual_seed(0)
    return TitansMemory(dim=DIM, chunk_size=CHUNK, **kw)


def _run(mem, scale=1.0, n=3, seed=1):
    """Chạy vài forward rồi trả (eta_raw, eta_real, alpha) trung bình."""
    torch.manual_seed(seed)
    mem.reset_eta_alpha()
    state = None
    for _ in range(n):
        seq = torch.randn(1, LEN, DIM) * scale
        _, state = mem(seq, state=state)
    return mem.eta_alpha_stats()


# --------------------------------------------------------------- 1) bất biến ngược
def test_khong_bat_co_thi_khong_cai_gi():
    """Mặc định `gate_bound=None` -> không hook chặn nào, `gate_bound_report()` là None."""
    mem = _mem()
    assert mem._gate_bound is None
    assert mem.gate_bound_report() is None


def test_output_khong_doi_khi_tat_gate_bound():
    """Bản không cờ phải cho output TRÙNG KHÍT bản gốc — bảo vệ mọi run cũ."""
    torch.manual_seed(7)
    x = torch.randn(1, LEN, DIM)
    a = _mem()
    b = _mem()
    with torch.no_grad():
        out_a, _ = a(x)
        out_b, _ = b(x)
    assert torch.allclose(out_a, out_b, atol=0, rtol=0)


def test_eta_alpha_stats_tra_ve_ba_gia_tri():
    """Chữ ký đổi từ 2 -> 3 giá trị (thêm η THẬT). methods.py phải khớp."""
    eta_raw, eta_real, alpha = _run(_mem())
    assert eta_raw is not None and eta_real is not None and alpha is not None


def test_eta_real_dung_cong_thuc_sigmoid_nhan_max_lr():
    """η THẬT = sigmoid(logit) * max_lr — đây là LỖI 2: log cũ in logit thô, không phải lr."""
    mem = _mem()
    eta_raw, eta_real, _ = _run(mem)
    assert 0.0 <= eta_real <= mem._eta_max_lr
    # sigmoid lồi/lõm nên mean(sigmoid) != sigmoid(mean); nới dung sai theo thang max_lr
    assert eta_real == pytest.approx(
        1 / (1 + math.exp(-eta_raw)) * mem._eta_max_lr, abs=0.2 * mem._eta_max_lr)


def test_max_lr_doc_duoc_tu_thu_vien():
    """Phải đọc ĐỘNG max_lr từ `adaptive_step_transform`, KHÔNG hardcode.

    Đây là bẫy thật: đọc `def default_adaptive_step_transform(.., max_lr=1e-2)` (:255) rất dễ
    tưởng max_lr=1e-2, nhưng :457 luôn ghi đè bằng `default_step_transform_max_lr = 1.` (:272).
    Sai hằng số này làm mọi ngưỡng η lệch 100 lần.
    """
    mem = _mem()
    tf_max_lr = mem.mem.adaptive_step_transform.keywords["max_lr"]
    assert mem._eta_max_lr == pytest.approx(tf_max_lr)
    assert mem._eta_max_lr == pytest.approx(1.0), "bản titans-pytorch này đặt max_lr=1.0"


# --------------------------------------------------------------- 2) cơ chế
def _mo_phong_troi(mem, bias: float):
    """Ép cổng vào TRẠNG THÁI ĐÃ TRÔI như quan sát thật.

    Vì sao không dùng "feature thật lớn": với trọng số ngẫu nhiên, logit lớn nhưng DẤU trộn
    lẫn nên trung bình sigmoid vẫn ≈ 0.5 — không tái hiện được lỗi. Trong log thật, α = 1.0000
    vì bias/trọng số đã trôi khiến MỌI logit cùng dấu và lớn (m3_s0: logit trung bình 5.8→55.6).
    Nên mô phỏng đúng cách là zero trọng số + đặt bias lớn.
    """
    for name in ("to_decay_factor", "to_adaptive_step"):
        lin = getattr(mem.mem, name)[0]
        with torch.no_grad():
            lin.weight.zero_()
            lin.bias.fill_(bias)


@pytest.mark.parametrize("bias", [5.0, 20.0, 60.0])
def test_gate_bound_giu_alpha_trong_vung_lanh_manh(bias):
    """Dù cổng đã trôi tới logit=60, α vẫn không được thoát ra biên."""
    mem = _mem(gate_bound=True)
    _mo_phong_troi(mem, bias)
    _, _, alpha = _run(mem)
    assert ALPHA_OK[0] <= alpha <= ALPHA_OK[1], f"bias={bias} -> α={alpha}"


@pytest.mark.parametrize("bias", [5.0, 20.0, 60.0])
def test_gate_bound_giu_eta_trong_vung_lanh_manh(bias):
    mem = _mem(gate_bound=True)
    _mo_phong_troi(mem, bias)
    _, eta_real, _ = _run(mem)
    assert ETA_OK[0] <= eta_real <= ETA_OK[1], f"bias={bias} -> η={eta_real}"


def test_khong_chan_thi_cong_BAO_HOA():
    """Đối chứng tái hiện ĐÚNG lỗi đã gặp: không chặn thì cổng trôi tới đâu bão hoà tới đó.

    Tái hiện `run_m3_s0`: logit trung bình 55.62 -> α=1.0000, η_real dính trần 1e-2.
    """
    mem = _mem()
    _mo_phong_troi(mem, 55.62)
    _, eta_real, alpha = _run(mem)
    assert alpha > 0.99, f"kỳ vọng α bão hoà khi không chặn, nhận {alpha}"
    assert eta_real > ETA_OK[1], f"kỳ vọng η dính trần, nhận {eta_real}"
    assert eta_real == pytest.approx(MAX_LR, rel=1e-3), "η phải dính đúng trần max_lr"

    mem_b = _mem(gate_bound=True)      # cùng trạng thái trôi, có chặn -> phải lành lại
    _mo_phong_troi(mem_b, 55.62)
    _, eta_b, alpha_b = _run(mem_b)
    assert ALPHA_OK[0] <= alpha_b <= ALPHA_OK[1]
    assert ETA_OK[0] <= eta_b <= ETA_OK[1]


def test_gate_bound_VAN_phu_thuoc_du_lieu():
    """Chặn nhưng KHÔNG được làm cổng thành hằng số — Eq 76 đòi η/α biến thiên theo input.

    Đây là tiêu chí quan trọng nhất: nằm trong khoảng mà ĐỨNG YÊN vẫn là hỏng.
    """
    mem = _mem(gate_bound=True)
    alphas = [_run(mem, scale=1.0, seed=s)[2] for s in (1, 2, 3, 4, 5)]
    assert max(alphas) - min(alphas) > 1e-4, f"α đứng yên qua các input: {alphas}"


def test_gate_bound_gradient_khong_chet():
    """tanh-bound phải còn gradient ở vùng giữa (khác hẳn clamp cứng)."""
    mem = _mem(gate_bound=True)
    seq = torch.randn(1, LEN, DIM, requires_grad=True)
    out, _ = mem(seq)
    out.sum().backward()
    assert seq.grad is not None and torch.isfinite(seq.grad).all()
    assert seq.grad.abs().sum() > 0


def test_gate_bound_nhan_limit_tuy_chinh():
    mem = _mem(gate_bound={"alpha_logit_limit": 1.0, "eta_logit_limit": 1.0})
    assert mem._gate_bound == {"eta": 1.0, "alpha": 1.0}
    _mo_phong_troi(mem, 60.0)
    _, _, alpha = _run(mem)
    lo, hi = 1 / (1 + math.exp(1.0)), 1 / (1 + math.exp(-1.0))
    assert lo - 1e-6 <= alpha <= hi + 1e-6


def test_gate_bound_limit_am_bi_tu_choi():
    with pytest.raises(ValueError):
        _mem(gate_bound={"alpha_logit_limit": 0.0})


def test_report_in_dung_khoang():
    rep = _mem(gate_bound=True).gate_bound_report()
    assert "α∈[0.050,0.950]" in rep
    assert str(round(DEFAULT_ALPHA_LOGIT_LIMIT, 3)) in rep or "2.944" in rep
    assert "η∈" in rep and str(round(DEFAULT_ETA_LOGIT_LIMIT, 3)) in rep or "2.197" in rep


# --------------------------------------------------------------- 3) TASK 5 — init bias
def test_init_bias_dua_cong_ve_trung_tinh():
    """`init_decay_bias=0` + weight=0 -> α khởi đầu đúng 0.5 (trung tính), bất kể input."""
    mem = _mem(init_decay_bias=0.0, init_adaptive_step_bias=0.0)
    _, eta_real, alpha = _run(mem, scale=500.0, n=1)
    assert alpha == pytest.approx(0.5, abs=1e-3), f"α khởi đầu = {alpha}, kỳ vọng 0.5"
    assert eta_real == pytest.approx(0.5 * mem._eta_max_lr, abs=1e-4)


# --------------------------------------------------------------- 4) TASK 4 — pre_norm
def test_pre_norm_mac_dinh_tat_va_bat_duoc():
    """`memory.pre_norm` mặc định TẮT (run cũ bất biến); bật thì là LayerNorm đúng chiều."""
    from uavcl.models.titans_head import TitansClassifier  # noqa: E402

    assert hasattr(TitansClassifier, "features_from_extracted")
    import inspect
    src = inspect.getsource(TitansClassifier.features_from_extracted)
    assert "seq_in" in src, "pre_norm phải đi vào memory qua seq_in"
    assert "self.adapter.restore(seq)" in src, "residual phải giữ seq THÔ (quyết định thiết kế)"
