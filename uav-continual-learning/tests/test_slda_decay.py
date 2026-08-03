"""D3 (2026-08-03) — λ (hệ số quên) của SLDA.

Bốn nhóm bảo đảm, xếp theo mức quan trọng:

  1. **λ=1 phải TRÙNG BIT với bản cũ.** Nếu không, mọi kết quả đã chạy trước đây thành vô giá
     trị vì không so được nữa. Đây là ràng buộc bất biến.
  2. **Lớp VẮNG MẶT không bị phân rã.** ⭐ Bẫy dễ mắc nhất: nếu μ_c phân rã theo đồng hồ toàn
     cục thay vì theo-lớp, lớp hiếm gặp sẽ bị xoá sạch — mất luôn "không quên lớp cũ", vốn là
     điểm mạnh lớn nhất của SLDA. Test này chặn đúng chỗ đó.
  3. **Cửa sổ nhớ đúng lý thuyết**: n_c hội tụ về 1/(1−λ).
  4. **Bám được trôi**: khi phân bố dịch đi, λ<1 phải bám theo còn λ=1 thì không.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uavcl.models.slda import SLDAClassifier  # noqa: E402

D, C = 8, 4


class _FakeBackbone(torch.nn.Module):
    def forward(self, x):
        return x


def _mk(**kw):
    torch.manual_seed(0)
    return SLDAClassifier(_FakeBackbone(), D, C, **kw)


def _batch(lop, n=16, dich=0.0, seed=0):
    """n mẫu của một lớp, tâm = lop + dich."""
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(n, D, generator=g) * 0.1 + float(lop) + dich
    y = torch.full((n,), int(lop), dtype=torch.long)
    return f, y


# ---------------------------------------------------------------------------
# 1. BẤT BIẾN — λ=1 phải y hệt bản cũ
# ---------------------------------------------------------------------------

def test_lambda_1_la_mac_dinh():
    m = _mk()
    assert m.decay_mean == 1.0 and m.decay_cov == 1.0


def test_lambda_1_trung_khop_tuyet_doi():
    """λ=1 đi qua NHÁNH KHÁC trong code (không nhân hệ số) — phải cho số y hệt.

    So bằng `equal` chứ không `allclose`: λ=1 không được phép làm phép nhân thừa nào,
    vì float64 × 1.0 tuy đúng về giá trị nhưng ta muốn khẳng định nhánh code bị BỎ QUA.
    """
    a, b = _mk(), _mk(decay_mean=1.0, decay_cov=1.0)
    for lop in range(C):
        f, y = _batch(lop, seed=lop)
        a.update(f, y)
        b.update(f, y)
    assert torch.equal(a.feat_sum, b.feat_sum)
    assert torch.equal(a.count, b.count)
    assert torch.equal(a.gram, b.gram)


def test_lambda_1_count_la_so_dem_that():
    m = _mk()
    for _ in range(5):
        m.update(*_batch(0, n=10))
    assert float(m.count[0]) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 2. ⭐ BẪY — lớp vắng mặt KHÔNG được phân rã
# ---------------------------------------------------------------------------

def test_lop_vang_mat_khong_bi_phan_ra():
    """Lớp 0 học xong rồi biến mất; 200 batch sau chỉ có lớp 1.

    Nếu phân rã theo đồng hồ toàn cục: 0.99^3200 ≈ 1e-14 -> lớp 0 bị xoá sạch.
    Phân rã theo-lớp: lớp 0 phải nguyên vẹn TUYỆT ĐỐI.
    """
    m = _mk(decay_mean=0.99, decay_cov=0.999)
    m.update(*_batch(0, n=16, seed=1))
    s0 = m.feat_sum[0].clone()
    n0 = m.count[0].clone()

    for i in range(200):
        m.update(*_batch(1, n=16, seed=100 + i))

    assert torch.equal(m.feat_sum[0], s0), "s_c của lớp vắng mặt đã bị đụng vào"
    assert torch.equal(m.count[0], n0), "n_c của lớp vắng mặt đã bị đụng vào"
    assert float(m.count[0]) == pytest.approx(16.0)


def test_lop_vang_mat_van_du_doan_dung():
    """Hệ quả thực tế của bẫy trên: sau khi lớp 0 vắng rất lâu, vẫn phải nhận ra nó."""
    m = _mk(decay_mean=0.99, decay_cov=0.999)
    for lop in range(C):
        m.update(*_batch(lop, n=32, seed=lop))
    for i in range(100):
        m.update(*_batch(C - 1, n=16, seed=500 + i))

    f, y = _batch(0, n=32, seed=999)
    assert float((m(f).argmax(1) == y).float().mean()) > 0.9


def test_phan_ra_chi_ap_dung_cho_lop_co_mat():
    """Batch trộn 2 lớp: đúng 2 lớp đó bị nhân hệ số, 2 lớp kia không."""
    m = _mk(decay_mean=0.9)
    for lop in range(C):
        m.update(*_batch(lop, n=10, seed=lop))
    truoc = m.count.clone()

    f = torch.cat([_batch(0, n=4, seed=7)[0], _batch(2, n=6, seed=8)[0]])
    y = torch.cat([torch.zeros(4, dtype=torch.long), torch.full((6,), 2, dtype=torch.long)])
    m.update(f, y)

    # lớp 0: 10·0.9⁴ + 4   |   lớp 2: 10·0.9⁶ + 6   |   lớp 1,3: không đổi
    assert float(m.count[0]) == pytest.approx(10 * 0.9 ** 4 + 4)
    assert float(m.count[2]) == pytest.approx(10 * 0.9 ** 6 + 6)
    assert float(m.count[1]) == pytest.approx(float(truoc[1]))
    assert float(m.count[3]) == pytest.approx(float(truoc[3]))


# ---------------------------------------------------------------------------
# 3. CỬA SỔ NHỚ — n_c hội tụ về 1/(1−λ)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam, cua_so", [(0.9, 10.0), (0.99, 100.0), (0.999, 1000.0)])
def test_n_c_hoi_tu_ve_cua_so_ly_thuyet(lam, cua_so):
    """Chuỗi hình học: Σ λ^k -> 1/(1−λ). Chạy đủ lâu (10× cửa sổ) để hội tụ."""
    m = _mk(decay_mean=lam)
    for i in range(int(cua_so * 10)):
        m.update(*_batch(0, n=1, seed=i))
    assert float(m.count[0]) == pytest.approx(cua_so, rel=0.02)


def test_window_report_khop_so_do():
    m = _mk(decay_mean=0.99, decay_cov=0.999)
    for i in range(1000):
        m.update(*_batch(0, n=1, seed=i))
    r = m.window_report()
    assert r["ky_vong"] == pytest.approx(100.0)
    assert r["n_c_trung_binh"] == pytest.approx(100.0, rel=0.05)
    assert r["decay_mean"] == 0.99 and r["decay_cov"] == 0.999
    assert r["so_lop_da_thay"] == 1


@pytest.mark.parametrize("lam, m", [(0.999, 140), (0.999, 420), (0.99, 420), (0.97, 420)])
def test_ky_vong_huu_han_moi_la_moc_dung(lam, m):
    """⭐ Mốc so sánh phải là (1−λ^m)/(1−λ), KHÔNG phải tiệm cận 1/(1−λ).

    Lỗi 2026-08-03: log in "kỳ vọng 1000" trong khi đo được 131 -> tưởng cài sai. Thực ra
    λ đúng hoàn hảo; chỉ là 1/(1−λ) là giới hạn khi m→∞, mà RESISC45 chỉ cho m=420 lần
    gặp mỗi lớp. So nhầm mốc thì hoặc báo động giả, hoặc bỏ lọt lỗi thật.
    """
    m_obj = _mk(decay_mean=lam)
    for i in range(m):
        m_obj.update(*_batch(0, n=1, seed=i))
    r = m_obj.window_report()
    ky_vong = (1 - lam ** m) / (1 - lam)
    assert r["ky_vong_huu_han"] == pytest.approx(ky_vong, rel=1e-9)
    assert r["n_c_trung_binh"] == pytest.approx(ky_vong, rel=0.01)
    assert r["so_lan_gap_tb"] == pytest.approx(float(m))


def test_ty_le_giu_bat_duoc_lambda_vo_dung():
    """λ_μ=0.9999 với m=420 giữ ~98% trí nhớ -> arm đó gần trùng λ=1, phải phát hiện được."""
    m = _mk(decay_mean=0.9999)
    for i in range(420):
        m.update(*_batch(0, n=1, seed=i))
    assert m.window_report()["ty_le_giu"] > 0.95      # cờ đỏ: λ quá gần 1 so với cỡ dữ liệu

    m2 = _mk(decay_mean=0.99)
    for i in range(420):
        m2.update(*_batch(0, n=1, seed=i))
    assert m2.window_report()["ty_le_giu"] < 0.30      # thực sự có quên


def test_count_raw_khong_bao_gio_phan_ra():
    """count_raw chỉ để chẩn đoán — phải là số đếm thô, không đụng tới λ."""
    m = _mk(decay_mean=0.9, decay_cov=0.9)
    for i in range(50):
        m.update(*_batch(0, n=2, seed=i))
    assert float(m.count_raw[0]) == pytest.approx(100.0)
    assert float(m.count[0]) < 20.0                    # bản có phân rã thì nhỏ hơn hẳn


def test_window_report_lambda_1_la_vo_han():
    assert _mk().window_report()["ky_vong"] == float("inf")


def test_sigma_van_xac_dinh_duong_khi_hai_lambda_lech_nhau():
    """⭐ Lỗi 2026-08-03 — hai đồng hồ khác nhau làm hỏng đẳng thức Σ.

    Σ_w = (G − Σ_c n_c μ_c μ_cᵀ)/N chỉ đúng khi G và (n_c, s_c) **cùng trọng số**. Nếu G
    phân rã toàn cục còn s_c phân rã theo-lớp, phép trừ TRỪ QUÁ TAY: G đã quên đóng góp của
    lớp vắng mặt, mà `between` vẫn trừ đủ. Σ thành **bất định dấu**, `inv(Σ)` ra rác, và mọi
    dự đoán sai — dù s_c, n_c của lớp đó còn nguyên vẹn.

    Bắt bằng trị riêng: Σ đúng thì trị riêng nhỏ nhất phải ≥ 0.
    """
    m = _mk(decay_mean=0.99, decay_cov=0.999)
    for lop in range(C):
        m.update(*_batch(lop, n=32, seed=lop))
    for i in range(100):
        m.update(*_batch(C - 1, n=16, seed=500 + i))

    sig = m._within_class_sigma()
    tri_rieng_min = float(torch.linalg.eigvalsh(sig).min())
    assert tri_rieng_min > -1e-9, \
        f"Σ bất định dấu (λ_min={tri_rieng_min:.3e}) — hai đồng hồ đang bị trộn lẫn"


def test_dem_toan_cuc_va_theo_lop_tach_biet():
    """Cặp đếm toàn cục phải QUÊN lớp vắng mặt; cặp theo-lớp phải GIỮ. Đó là cả điểm của fix."""
    m = _mk(decay_mean=0.99, decay_cov=0.99)
    m.update(*_batch(0, n=16, seed=1))
    for i in range(100):
        m.update(*_batch(1, n=16, seed=100 + i))

    assert float(m.count[0]) == pytest.approx(16.0)        # theo-lớp: nguyên vẹn
    assert float(m.count_g[0]) < 1e-3                      # toàn cục: đã phai gần hết
    assert float(m.count_g[1]) > 1.0                       # lớp đang gặp thì còn


def test_lambda_1_hai_cap_dem_trung_nhau():
    """λ=1 -> hai đồng hồ là một -> cặp toàn cục phải bằng ĐÚNG cặp theo-lớp."""
    m = _mk()
    for lop in range(C):
        m.update(*_batch(lop, n=16, seed=lop))
    assert torch.equal(m.feat_sum_g, m.feat_sum)
    assert torch.equal(m.count_g, m.count)


def test_gram_phan_ra_theo_dong_ho_toan_cuc():
    """Σ dùng chung cho mọi lớp -> tuổi tính theo TỔNG số mẫu, khác hẳn μ_c."""
    m = _mk(decay_cov=0.99)
    f, y = _batch(0, n=1, seed=0)
    m.update(f, y)
    g1 = m.gram.clone()
    m.update(*_batch(1, n=1, seed=1))          # lớp KHÁC, gram vẫn phải phân rã
    ky_vong = g1 * 0.99
    thua = m.gram - ky_vong
    assert torch.allclose(m.gram - thua, ky_vong, atol=1e-12)
    assert float(thua.abs().max()) > 0, "gram phải có phần đóng góp mới"


# ---------------------------------------------------------------------------
# 4. BÁM TRÔI — lý do tồn tại của λ
# ---------------------------------------------------------------------------

def test_lambda_nho_bam_troi_tot_hon_lambda_1():
    """Phân bố dịch dần 0 -> 3. Đo |μ_c − tâm_hiện_tại| ở cuối.

    Đây là toàn bộ luận điểm của D-plan bằng một con số: λ=1 kéo theo cả lịch sử nên tụt lại,
    λ<1 bỏ quá khứ nên bám sát.
    """
    cham = _mk(decay_mean=1.0)      # không quên
    nhanh = _mk(decay_mean=0.98)    # cửa sổ ≈ 50 mẫu

    B, buoc = 8, 60
    for i in range(buoc):
        dich = 3.0 * i / (buoc - 1)
        f, y = _batch(0, n=B, dich=dich, seed=i)
        cham.update(f, y)
        nhanh.update(f, y)

    tam = 3.0                        # tâm ở thời điểm cuối (lop=0 -> 0 + 3.0)
    mu_c = float((cham.feat_sum[0] / cham.count[0]).mean())
    mu_n = float((nhanh.feat_sum[0] / nhanh.count[0]).mean())
    assert abs(mu_n - tam) < abs(mu_c - tam) / 2, \
        f"λ<1 phải bám sát hơn hẳn: nhanh={mu_n:.3f} chậm={mu_c:.3f} tâm={tam}"


def test_lambda_1_tut_lai_ve_trung_binh_lich_su():
    """Đối chứng khẳng định: λ=1 hội tụ về trung bình TOÀN BỘ lịch sử (≈1.5), không phải tâm cuối."""
    m = _mk(decay_mean=1.0)
    B, buoc = 8, 60
    for i in range(buoc):
        m.update(*_batch(0, n=B, dich=3.0 * i / (buoc - 1), seed=i))
    assert float((m.feat_sum[0] / m.count[0]).mean()) == pytest.approx(1.5, abs=0.15)


# ---------------------------------------------------------------------------
# 5. KIỂM ĐẦU VÀO
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.0, -0.1, 1.1, 2.0])
def test_lambda_ngoai_khoang_bi_tu_choi(lam):
    for kw in ({"decay_mean": lam}, {"decay_cov": lam}):
        with pytest.raises(ValueError):
            _mk(**kw)


def test_lambda_hop_le_duoc_chap_nhan():
    m = _mk(decay_mean=0.999, decay_cov=0.9999)
    assert m.decay_mean == 0.999 and m.decay_cov == 0.9999


def test_lambda_khong_pha_float32():
    """stats_dtype='float32' (cho MPS / NPU biên) vẫn phải chạy đúng với λ."""
    m = _mk(decay_mean=0.99, stats_dtype="float32")
    for i in range(300):
        m.update(*_batch(0, n=1, seed=i))
    assert m.count.dtype == torch.float32
    assert float(m.count[0]) == pytest.approx(100.0, rel=0.05)
