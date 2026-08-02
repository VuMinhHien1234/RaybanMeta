"""B2 (2026-08-02) — ablation `cov_mode` của SLDA: tách đóng góp của hiệp phương sai.

Giả thuyết cần kiểm: +11,5 điểm của SLDA so với NCM đến từ **ma trận hiệp phương sai chung**,
không phải từ khác biệt cài đặt nào khác.

Cách tách sạch nhất: cho Σ_w = I ngay trong đường code SLDA. Khi đó
    score_c(f) = μ_c·f/(1+ε) − ½‖μ_c‖²/(1+ε)
và vì hệ số 1/(1+ε) chung cho mọi lớp:
    argmax_c score_c(f) ≡ argmax_c (μ_c·f − ½‖μ_c‖²) ≡ argmin_c ‖f − μ_c‖²
tức **ĐÚNG BẰNG nearest-class-mean**. Test dưới chứng minh đẳng thức này bằng số.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uavcl.models.slda import COV_MODES, SLDAClassifier  # noqa: E402

D, C, N = 16, 4, 200


class _FakeBackbone(torch.nn.Module):
    """Backbone giả: trả thẳng đầu vào (đã là feature) — tách SLDA khỏi ViT."""

    def forward(self, x):
        return x


def _mk(cov_mode="streaming", **kw):
    torch.manual_seed(0)
    return SLDAClassifier(_FakeBackbone(), D, C, cov_mode=cov_mode, **kw)


def _data(seed=0, n=N, mu_scale=1.0):
    """Sinh feature mà hiệp phương sai THẬT SỰ quan trọng.

    Hai điều kiện, thiếu một cái là ablation vô nghĩa:
      1. Nhiễu **có tương quan** (nhân với ma trận đặc `A`, không phải scale theo chiều).
         Nhiễu chéo thì Mahalanobis chỉ co giãn từng trục, lợi ích nhỏ; nhiễu tương quan
         mới buộc phải XOAY hệ toạ độ — đúng thứ Σ⁻¹ làm mà khoảng cách Euclid không làm được.
      2. Tâm lớp **đủ gần** (`mu_scale=1.0`) để các lớp chồng lấn. Tâm xa quá thì cả hai
         phương pháp đều đúng gần 100% và không phân biệt được (bản test đầu sập vì lỗi này:
         chỉ 0,5% dự đoán khác nhau).

    Với thiết lập này: SLDA ≈ 0.99 · Σ=I ≈ 0.76 — tái hiện đúng khoảng cách quan sát thật.
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, D, generator=g) / (D ** 0.5) * 2.5   # ↳ Σ = AᵀA, KHÔNG chéo
    ys = torch.randint(0, C, (n,), generator=g)
    mu = torch.randn(C, D, generator=g) * mu_scale
    f = mu[ys] + torch.randn(n, D, generator=g) @ A
    return f, ys


def _acc(model, f, ys):
    return (model(f).argmax(dim=1) == ys).float().mean().item()


# ---------------------------------------------------------------- hợp lệ hoá tham số
@pytest.mark.parametrize("mode", COV_MODES)
def test_moi_cov_mode_deu_dung_duoc(mode):
    m = _mk(mode)
    f, ys = _data()
    m.update(f, ys)
    out = m(f)
    assert out.shape == (N, C)
    assert torch.isfinite(out).all()


def test_cov_mode_la_bi_tu_choi():
    with pytest.raises(ValueError):
        _mk("khong-ton-tai")


def test_cov_freeze_after_phai_duong():
    with pytest.raises(ValueError):
        _mk("frozen", cov_freeze_after=0)


# ---------------------------------------------------------------- ARM 3: identity == NCM
def test_identity_TUONG_DUONG_nearest_class_mean():
    """Đây là test quan trọng nhất của B2: Σ=I phải cho đúng dự đoán của NCM."""
    m = _mk("identity")
    f, ys = _data()
    m.update(f, ys)

    pred_slda = m(f).argmax(dim=1)

    mu = (m.feat_sum / m.count.clamp(min=1.0).unsqueeze(1)).float()   # (C, D)
    pred_ncm = torch.cdist(f, mu).argmin(dim=1)                        # nearest class mean

    assert torch.equal(pred_slda, pred_ncm), "Σ=I phải trùng khít NCM"


def test_identity_khong_can_nghich_dao():
    """Σ=I -> Λ = I/(1+ε) tính thẳng, không gọi torch.linalg.inv (rẻ hơn O(D³))."""
    m = _mk("identity")
    f, ys = _data()
    m.update(f, ys)
    goi = {"n": 0}
    thuc = torch.linalg.inv

    def _dem(*a, **k):
        goi["n"] += 1
        return thuc(*a, **k)

    torch.linalg.inv = _dem
    try:
        m._cache_version = -1
        m(f)
    finally:
        torch.linalg.inv = thuc
    assert goi["n"] == 0, "chế độ identity không được nghịch đảo ma trận"


def test_streaming_KHAC_identity_tren_nhieu_tuong_quan():
    """Nếu hai chế độ cho cùng kết quả thì ablation vô nghĩa — phải khác nhau thật."""
    f, ys = _data()
    a, b = _mk("streaming"), _mk("identity")
    a.update(f, ys); b.update(f, ys)
    khac = (a(f).argmax(1) != b(f).argmax(1)).float().mean().item()
    assert khac > 0.05, f"streaming và identity gần trùng nhau ({khac:.4f}) — dữ liệu test quá dễ"


def test_hiep_phuong_sai_LAM_TANG_do_chinh_xac():
    """Bản thu nhỏ của chính thí nghiệm B2: Σ phải làm accuracy TĂNG, không chỉ 'khác'.

    Đây là test có ý nghĩa nhất file — nó phát biểu đúng giả thuyết đang đi kiểm trên
    dữ liệu thật: hiệp phương sai chung là thứ tạo ra khoảng cách giữa SLDA và NCM.
    """
    f, ys = _data()
    a, b = _mk("streaming"), _mk("identity")
    a.update(f, ys); b.update(f, ys)
    acc_sigma, acc_identity = _acc(a, f, ys), _acc(b, f, ys)
    assert acc_sigma > acc_identity + 0.10, (
        f"Σ streaming ({acc_sigma:.3f}) phải vượt Σ=I ({acc_identity:.3f}) rõ rệt "
        "trên nhiễu tương quan")


def test_tam_lop_xa_nhau_thi_hai_che_do_hoa_nhau():
    """Đối chứng ngược: lớp tách bạch thì Σ gần như vô dụng.

    Quan trọng cho việc DIỄN GIẢI kết quả thật — nếu chạy trên dữ liệu mà SLDA không hơn
    NCM, kết luận đúng là 'feature đã đủ tách bạch', không phải 'SLDA hỏng'.
    """
    f, ys = _data(mu_scale=6.0)                  # ↳ tâm lớp rất xa
    a, b = _mk("streaming"), _mk("identity")
    a.update(f, ys); b.update(f, ys)
    assert abs(_acc(a, f, ys) - _acc(b, f, ys)) < 0.05


# ---------------------------------------------------------------- ARM 4: frozen
def test_frozen_dong_bang_sigma_dung_moc():
    m = _mk("frozen", cov_freeze_after=1)
    f1, y1 = _data(seed=1)
    m.update(f1, y1)
    assert m._frozen_sigma.numel() == 0, "chưa hết task 0 thì chưa được đóng băng"
    m.on_task_end()
    assert m._frozen_sigma.numel() == D * D, "phải đóng băng Σ sau task đầu"
    snap = m._frozen_sigma.clone()

    f2, y2 = _data(seed=2)                       # ↳ dữ liệu mới, phân bố khác
    m.update(f2, y2)
    m.on_task_end()
    assert torch.equal(m._frozen_sigma, snap), "Σ đã đóng băng thì không được đổi nữa"


def test_frozen_van_cap_nhat_class_mean():
    """Đóng băng Σ nhưng μ_c PHẢI tiếp tục học — nếu không thì lớp mới sẽ không nhận được."""
    m = _mk("frozen", cov_freeze_after=1)
    f1, y1 = _data(seed=1)
    m.update(f1, y1); m.on_task_end()
    mu_truoc = (m.feat_sum / m.count.clamp(min=1.0).unsqueeze(1)).clone()

    f2, y2 = _data(seed=7)
    m.update(f2, y2)
    mu_sau = m.feat_sum / m.count.clamp(min=1.0).unsqueeze(1)
    assert not torch.allclose(mu_truoc, mu_sau), "μ_c phải tiếp tục cập nhật sau khi Σ đóng băng"


def test_frozen_truoc_moc_hanh_xu_nhu_streaming():
    f, ys = _data()
    a = _mk("frozen", cov_freeze_after=5)        # ↳ mốc xa, chưa tới
    b = _mk("streaming")
    a.update(f, ys); b.update(f, ys)
    assert torch.allclose(a(f), b(f), atol=1e-5)


# ---------------------------------------------------------------- bất biến ngược
def test_mac_dinh_van_la_streaming():
    """Không khai báo gì -> hành vi y hệt bản trước B2 (bảo vệ kết quả 0.8263)."""
    m = _mk()
    assert m.cov_mode == "streaming"
    assert m._frozen_sigma.numel() == 0
    f, ys = _data()
    m.update(f, ys)
    m.on_task_end()                              # ↳ no-op ở chế độ streaming
    assert m._frozen_sigma.numel() == 0


# ---------------------------------------------------------------- B6 — ràng buộc dtype
def test_stats_dtype_la_bi_tu_choi():
    with pytest.raises(ValueError):
        _mk(stats_dtype="float16")


def test_float32_giam_dung_mot_nua_bo_nho():
    """MPS/NPU biên không có float64 -> phải chạy được float32. Bộ nhớ đúng bằng nửa."""
    a, b = _mk(stats_dtype="float64"), _mk(stats_dtype="float32")
    f, ys = _data()
    for m in (a, b):
        m.update(f, ys); m(f)
    ra, rb = a.memory_report(), b.memory_report()
    assert rb["gram_DxD"] * 2 == ra["gram_DxD"]
    assert rb["stats_dtype"] == "float32"


def test_float32_cho_ket_qua_gan_nhu_float64():
    """Sai số tích luỹ của float32 không được đổi dự đoán trên quy mô test.

    Kiểm này quan trọng vì nó quyết định `stats_dtype=float32` có dùng được cho kết quả
    chính hay không. Trên dữ liệu THẬT (nhiều mẫu hơn) vẫn phải đo lại — test này chỉ
    chặn trường hợp sai số lớn tới mức lộ ra ngay ở quy mô nhỏ.
    """
    a, b = _mk(stats_dtype="float64"), _mk(stats_dtype="float32")
    f, ys = _data()
    a.update(f, ys); b.update(f, ys)
    khac = (a(f).argmax(1) != b(f).argmax(1)).float().mean().item()
    assert khac < 0.02, f"float32 lệch {khac:.4f} so với float64 — quá nhiều"


def test_memory_report_dung_byte_that():
    m = _mk()
    f, ys = _data()
    m.update(f, ys)
    m(f)
    r = m.memory_report()
    # gram là float64 -> D*D*8 byte
    assert r["gram_DxD"] == D * D * 8
    assert r["feat_sum_CxD"] == C * D * 8
    assert r["total_bytes"] > r["gram_DxD"]
    assert r["cov_mode"] == "streaming"
