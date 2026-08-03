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


# =========================================================================================
# G1–G5 (2026-08-03) — vá sau khi phát hiện TRẦN η LỆCH THANG 100 LẦN
#
# Chuyện đã xảy ra: `eta_logit_limit: 2.1972` chọn khi còn tưởng max_lr=1e-2 -> dự định
# η ∈ [1e-3, 9e-3]. max_lr thật là 1.0 -> thực tế η ∈ [0.1, 0.9]. Log 9 task: η bò tới
# 0.896/0.900 = 99,5% trần -> cổng thành HẰNG SỐ, hết phụ thuộc dữ liệu (mất Eq 76).
#
# Vì sao bộ test cũ KHÔNG bắt được: nó đo η theo PHÂN SỐ của max_lr (ETA_FRAC 0.01–0.95).
# Trần lệch cho phân số 0.1–0.9, nằm gọn trong ngưỡng -> xanh. Đóng khung bằng phân số
# khiến sai số thang đo trở nên vô hình. Cùng lỗ hổng với thang λ của SLDA: CÀI ĐÚNG, CHỈNH SAI.
# =========================================================================================


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def test_tam_0_trung_khop_ban_cu():
    """BẤT BIẾN: tâm=0 phải cho ra ĐÚNG công thức cũ `tanh(x/L)*L`, không sai một bit."""
    gb = {"alpha_logit_limit": 2.0, "eta_logit_limit": 2.0}
    a = _mem(gate_bound=gb)
    b = _mem(gate_bound={**gb, "alpha_logit_center": 0.0, "eta_logit_center": 0.0})
    for x, y in zip(_run(a), _run(b)):
        assert x == y, "khai báo center=0.0 tường minh đã làm đổi kết quả"


@pytest.mark.parametrize("tam, nua", [(0.0, 2.1972), (-2.2, 2.2), (-3.0, 2.2), (-5.81, 1.10)])
def test_bound_lech_tam_cho_dung_khoang(tam, nua):
    """Khoảng η báo cáo phải khớp công thức sigmoid(tâm ± nửa)·max_lr."""
    m = _mem(gate_bound={"eta_logit_limit": nua, "eta_logit_center": tam,
                         "alpha_logit_limit": 2.9444})
    lo, hi = m.gate_bound_range("eta")
    assert lo == pytest.approx(_sig(tam - nua) * MAX_LR, rel=1e-9)
    assert hi == pytest.approx(_sig(tam + nua) * MAX_LR, rel=1e-9)


def test_lech_tam_thuc_su_keo_eta_xuong():
    """Không chỉ báo cáo — η ĐO ĐƯỢC phải thấp hơn hẳn và nằm trong trần mới.

    Khẳng định theo trần (đúng chắc chắn) chứ không theo tỉ lệ cố định, để test không phụ
    thuộc vào việc dữ liệu ngẫu nhiên đẩy logit tới đâu.
    """
    cao = _mem(gate_bound={"eta_logit_limit": 2.2, "eta_logit_center": 0.0})
    thap = _mem(gate_bound={"eta_logit_limit": 2.2, "eta_logit_center": -3.0})
    _, e_cao, _ = _run(cao)
    _, e_thap, _ = _run(thap)
    lo_c, hi_c = cao.gate_bound_range("eta")
    lo_t, hi_t = thap.gate_bound_range("eta")
    assert lo_c <= e_cao <= hi_c and lo_t <= e_thap <= hi_t, "η ra ngoài trần đã chặn"
    # KHÔNG đòi hai trần tách rời — chúng CHỒNG LẤN theo thiết kế:
    #   tâm 0.0  -> [0.0998, 0.9002]
    #   tâm −3.0 -> [0.0055, 0.3100]      (chồng nhau ở đoạn 0.0998–0.3100)
    # Điều đúng và đủ để khẳng định là TRẦN thấp hơn hẳn, và η đo được đi theo.
    assert hi_t < hi_c / 2, "trần trên của bản dời tâm phải thấp hơn hẳn"
    assert e_thap < e_cao, f"dời tâm không kéo được η xuống: {e_thap:.4g} vs {e_cao:.4g}"


def test_report_in_khoang_tuyet_doi_sau_khi_doi_tam():
    """Báo cáo KHÔNG được giả định tâm 0 — bản cũ giả định vậy nên in sai khi dời tâm."""
    m = _mem(gate_bound={"eta_logit_limit": 1.10, "eta_logit_center": -5.81,
                         "alpha_logit_limit": 2.9444})
    lo, hi = m.gate_bound_range("eta")
    # Khớp CÔNG THỨC (chặt), rồi mới khớp Ý NGHĨA (lỏng). Bản trước tôi gõ hằng số tính tay
    # 0.008907 — sai ở chữ số thứ 3 (đúng là 0.0089244) và test đỏ vì chính lỗi số học của tôi.
    assert lo == pytest.approx(_sig(-5.81 - 1.10) * MAX_LR, rel=1e-9)
    assert hi == pytest.approx(_sig(-5.81 + 1.10) * MAX_LR, rel=1e-9)
    assert (lo, hi) == pytest.approx((1e-3, 9e-3), rel=0.05), \
        "dời tâm −5.81 nửa 1.10 phải cho η ≈ [1e-3, 9e-3] — đúng dự định ban đầu"
    r = m.gate_bound_report()
    assert "tâm" in r, f"báo cáo phải nói rõ tâm đã dời, nhận: {r}"
    assert "e-0" in r, f"khoảng η phải in dạng khoa học khi rất nhỏ, nhận: {r}"


def test_raise_khi_khong_cai_duoc_hook():
    """⭐ G1 — config yêu cầu chặn mà không tìm thấy cổng nào là LỖI, không phải im lặng.

    Trước đây trả None -> bản vá tự tắt, log không in gì, test vẫn xanh, run vẫn tới cuối.
    Phải để ý sự VẮNG MẶT của một dòng log mới biết — tín hiệu quá yếu để tin.
    """
    class _Rong(torch.nn.Module):     # memory giả, không có to_adaptive_step/to_decay_factor
        def forward(self, x, state=None):
            return x, None

    m = TitansMemory.__new__(TitansMemory)
    torch.nn.Module.__init__(m)
    m.mem = _Rong()
    with pytest.raises(RuntimeError, match="không tìm thấy"):
        m._install_gate_bounds({"eta_logit_limit": 2.0})


def test_khong_bat_thi_van_tra_None_khong_raise():
    """Không khai báo gate_bound thì tuyệt đối không được raise (bất biến ngược)."""
    m = _mem()
    assert m._install_gate_bounds(None) is None
    assert m._install_gate_bounds(False) is None
    assert m.gate_bound_report() is None


def test_gate_bound_range_tra_None_khi_tat():
    m = _mem()
    assert m.gate_bound_range("eta") is None and m.gate_bound_range("alpha") is None


def test_khoang_eta_tuyet_doi_cua_config_that():
    """⭐ TEST CHẶN ĐÚNG LỖI HÔM NAY — đo thang TUYỆT ĐỐI, không phải phân số.

    Với `eta_logit_limit=2.1972, center=0` thì η ∈ [0.1, 0.9] và tâm là 0.5. Một learning-rate
    memory tâm 0,5 là rất lớn: cổng sẽ nhanh chóng dính trần rồi thành hằng số.

    Test này KHÔNG bảo cấu hình đó sai — có thể η cao đúng là thứ Titans cần trên stream trôi.
    Nó chỉ bắt buộc con số phải được NHÌN THẤY và chọn có ý thức, thay vì là tai nạn đơn vị.
    """
    lo, hi = _mem(gate_bound={"eta_logit_limit": DEFAULT_ETA_LOGIT_LIMIT}).gate_bound_range("eta")
    assert (lo, hi) == pytest.approx((0.1, 0.9), rel=1e-3), \
        "trần η mặc định đã đổi — cập nhật comment config và kế hoạch thí nghiệm kèm theo"
    assert _sig(0.0) * MAX_LR == pytest.approx(0.5), \
        "tâm η của trần đối xứng luôn là 0.5·max_lr — muốn nhỏ hơn PHẢI dời tâm"


def test_vi_tri_bao_hoa_tinh_dung_tren_so_that():
    """Tái hiện số thật của run 9 task: η=0.8957 trần [0.1,0.9] -> 99,5%; α=0.0678 -> 2,0%.

    Cả hai đều phải vượt ngưỡng GATE_SAT_FRAC=0.90 (một ở trên, một ở dưới) để sinh cảnh báo.
    """
    from uavcl.methods import GATE_SAT_FRAC

    m = _mem(gate_bound={"eta_logit_limit": DEFAULT_ETA_LOGIT_LIMIT,
                         "alpha_logit_limit": DEFAULT_ALPHA_LOGIT_LIMIT})
    lo_e, hi_e = m.gate_bound_range("eta")
    lo_a, hi_a = m.gate_bound_range("alpha")
    frac_eta = (0.8957 - lo_e) / (hi_e - lo_e)
    frac_alpha = (0.0678 - lo_a) / (hi_a - lo_a)
    assert frac_eta == pytest.approx(0.995, abs=0.01)
    assert frac_alpha == pytest.approx(0.020, abs=0.01)
    assert frac_eta > GATE_SAT_FRAC, "η sát trần mà không bị gắn cờ"
    assert frac_alpha < 1.0 - GATE_SAT_FRAC, "α sát sàn mà không bị gắn cờ"


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
    # G2 (2026-08-03): `_gate_bound` giờ lưu (nửa, tâm) chứ không phải riêng nửa,
    # vì trần phải lệch tâm được. Không khai báo center -> tâm = 0.0 = hành vi cũ.
    assert mem._gate_bound == {"eta": (1.0, 0.0), "alpha": (1.0, 0.0)}
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
