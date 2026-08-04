"""M1/M2 (KE_HOACH_SUA 2026-08-04) — test BA TẦNG: TangNhanh, NganHangCheDo, tích hợp SLDA.

Ba test có ⭐ là ba chỗ hỏng khả năng cao nhất (bảng M4 của KE_HOACH_BA_TANG):
gặp lại khớp chế độ CŨ không tạo mới · số chế độ = số điều kiện thật · bẫy tỷ lệ lớp (T2).
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from uavcl.models.ngan_hang_che_do import NganHangCheDo  # noqa: E402
from uavcl.models.slda import SLDAClassifier  # noqa: E402
from uavcl.models.tang_nhanh import TangNhanh  # noqa: E402

D, C = 8, 3


class _FakeBackbone(torch.nn.Module):
    def forward(self, x):
        return x


# Tâm lớp CỐ ĐỊNH cho cả file — KHÔNG sinh lại theo `seed`.
#
# Vì sao phải tách ra hằng số: bài toán là "CÙNG khu vực, KHÁC điều kiện" (BAI_TOAN §2).
# Nếu để `mu` sinh từ chính generator của `seed` thì hai lần gọi `_du_lieu` với seed khác
# nhau sẽ ra hai TẬP LỚP khác nhau — pha 2 thành "khu vực khác" chứ không phải "điều kiện
# khác". Không tầng căn chỉnh nào cứu được chuyện đó (acc rơi về ngẫu nhiên 1/C), nên test
# sẽ đổ lỗi oan cho M1. Đã dính đúng bẫy này một lần (2026-08-04).
MU = torch.randn(C, D, generator=torch.Generator().manual_seed(0)) * 3.0


def _du_lieu(n=300, seed=0, dich=None):
    """Lớp tách bạch (tâm `MU`, nhiễu 0.3) + tuỳ chọn DỊCH toàn cục (mô phỏng điều kiện).

    `seed` CHỈ đổi mẫu + nhiễu; tâm lớp luôn là `MU` để mọi pha cùng một tập lớp.
    """
    g = torch.Generator().manual_seed(seed)
    ys = torch.randint(0, C, (n,), generator=g)
    f = MU[ys] + torch.randn(n, D, generator=g) * 0.3
    if dich is not None:
        f = f + dich
    return f, ys


def _mk_slda(**kw):
    torch.manual_seed(0)
    return SLDAClassifier(_FakeBackbone(), D, C, **kw)


# ================================ TangNhanh ================================
def test_chua_chot_moc_thi_can_chinh_la_identity():
    tn = TangNhanh(D, decay=0.9)
    f = torch.randn(4, D)
    tn.cap_nhat(f)
    assert torch.equal(tn.can_chinh(f), f), "chưa chốt mốc thì không được sửa gì"


def test_m_t_hoi_tu_ve_trung_binh_dong():
    tn = TangNhanh(D, decay=0.9)
    tam = torch.full((D,), 2.5)
    g = torch.Generator().manual_seed(1)
    for _ in range(200):
        tn.cap_nhat(tam + torch.randn(32, D, generator=g) * 0.1)
    assert float((tn.m_t - tam).norm()) < 0.1, "m_t phải hội tụ về trung bình dòng"


def test_can_chinh_keo_nguoc_dung_do_dich():
    """Trôi = dịch toàn cục δ. Sau khi m_t bám kịp, căn chỉnh phải trả feature về gần gốc.

    Dùng dòng nhiễu nhỏ KHÔNG cấu trúc lớp: test này đo riêng phép căn chỉnh, không đo
    nhiễu ước lượng m từ thành phần lớp (cái đó test prequential phía dưới lo)."""
    tn = TangNhanh(D, decay=0.99)                  # ↳ cửa sổ 100 mẫu -> m ước lượng đủ mịn
    g = torch.Generator().manual_seed(2)
    goc = torch.randn(D, generator=g)
    f0 = goc + torch.randn(600, D, generator=g) * 0.1
    for i in range(0, 600, 32):
        tn.cap_nhat(f0[i:i + 32])
    tn.chot_moc()
    delta = torch.full((D,), 4.0)
    f1 = f0 + delta
    for i in range(0, 600, 32):                    # ↳ m_t bám sang điều kiện mới
        tn.cap_nhat(f1[i:i + 32])
    ra = tn.can_chinh(f1[:64])
    assert float((ra - f0[:64]).abs().mean()) < 0.1, "căn chỉnh phải kéo về hệ toạ độ pha 1"


def test_kieu_truc_giu_nguyen_phan_vuong_goc(tmp_path):
    """⭐ Thuốc chống bẫy T2: chỉ thành phần TRÊN trục bị sửa, phần ⊥ (thông tin lớp) giữ nguyên."""
    u = torch.zeros(D); u[0] = 1.0                 # trục điều kiện = chiều 0
    p = tmp_path / "t1.json"
    p.write_text(json.dumps({"truc_dieu_kien": u.tolist()}))
    tn = TangNhanh(D, decay=0.99, kieu="truc", truc_json=str(p))
    g = torch.Generator().manual_seed(3)
    f0 = torch.randn(600, D, generator=g) * 0.1
    for i in range(0, 600, 32):
        tn.cap_nhat(f0[i:i + 32])
    tn.chot_moc()
    dich = torch.zeros(D); dich[0] = 3.0; dich[1] = 2.0   # trôi trên trục + nhiễu ⊥ trục
    f1 = f0 + dich
    for i in range(0, 600, 32):
        tn.cap_nhat(f1[i:i + 32])
    ra = tn.can_chinh(f1[:32])
    assert float((ra[:, 0] - f0[:32, 0]).abs().mean()) < 0.1, "thành phần TRÊN trục phải được sửa"
    assert torch.allclose(ra[:, 1], f1[:32, 1]), "thành phần ⊥ trục KHÔNG được đụng vào"


def test_kieu_truc_thieu_file_bi_chan_som():
    with pytest.raises((ValueError, FileNotFoundError)):
        TangNhanh(D, kieu="truc", truc_json=None)
    with pytest.raises(FileNotFoundError):
        TangNhanh(D, kieu="truc", truc_json="/khong/ton/tai.json")


# ================================ NganHangCheDo ================================
def _bay_mot_chuyen(tn: TangNhanh, nh: NganHangCheDo, dich: torch.Tensor, seed=0):
    """Mô phỏng một chuyến pha 2: 10 batch cùng điều kiện `dich`."""
    g = torch.Generator().manual_seed(seed)
    for b in range(10):
        tn.cap_nhat(dich + torch.randn(32, D, generator=g) * 0.05)
        if b + 1 == nh.cho_khop_sau:
            nh.khop_va_nap(tn)
    nh.ghi_lai(tn)


def _tn_da_chot(seed=3):
    tn = TangNhanh(D, decay=0.9)
    g = torch.Generator().manual_seed(seed)
    for _ in range(20):
        tn.cap_nhat(torch.randn(32, D, generator=g) * 0.05)
    tn.chot_moc()
    return tn


def test_4_dieu_kien_tach_biet_tao_dung_4_che_do():
    """⭐ cửa chặn M2: số chế độ hệ tự tạo = số điều kiện thật."""
    tn, nh = _tn_da_chot(), NganHangCheDo(D, nguong=1.0, k_max=8)
    for k in range(4):
        dich = torch.zeros(D); dich[k] = 5.0       # 4 điều kiện cách nhau rất xa
        _bay_mot_chuyen(tn, nh, dich, seed=10 + k)
    assert len(nh.che_do) == 4, f"phải tạo đúng 4 chế độ, thấy {len(nh.che_do)}"


def test_gap_lai_khop_che_do_cu_khong_tao_moi():
    """⭐ bẫy chính của M2: quay lại điều kiện cũ phải NHẬN RA, không đẻ chế độ mới."""
    tn, nh = _tn_da_chot(), NganHangCheDo(D, nguong=1.0, k_max=8)
    dich_a = torch.zeros(D); dich_a[0] = 5.0
    dich_b = torch.zeros(D); dich_b[1] = 5.0
    _bay_mot_chuyen(tn, nh, dich_a, seed=20)
    _bay_mot_chuyen(tn, nh, dich_b, seed=21)
    _bay_mot_chuyen(tn, nh, dich_a, seed=22)       # ↳ quay lại A
    assert len(nh.che_do) == 2, "gặp lại A không được tạo chế độ thứ 3"
    assert nh.so_lan_nap == 1, "phải có đúng 1 lần 'gặp lại -> nạp'"


def test_nap_snapshot_thich_nghi_tuc_thoi():
    """Cơ chế O3: khớp xong thì m_t CHÍNH LÀ snapshot đã lưu — khỏi học lại."""
    tn, nh = _tn_da_chot(), NganHangCheDo(D, nguong=1.0, cho_khop_sau=1)
    dich = torch.zeros(D); dich[0] = 5.0
    _bay_mot_chuyen(tn, nh, dich, seed=30)
    m_luu = nh.che_do[0]["m"].clone()
    tn.m_t = m_luu + 0.1                           # ↳ đầu chuyến sau: m_t mới bám một phần
    assert nh.khop_va_nap(tn)
    assert torch.equal(tn.m_t, m_luu), "nạp snapshot phải là copy đúng bit"


def test_bay_t2_dich_vuong_goc_truc_khong_de_che_do_moi(tmp_path):
    """⭐ bẫy T2: đổi tỷ lệ lớp dịch m_t theo hướng ⊥ trục — KHÔNG được coi là điều kiện mới."""
    u = torch.zeros(D); u[0] = 1.0
    p = tmp_path / "t1.json"
    p.write_text(json.dumps({"truc_dieu_kien": u.tolist()}))
    tn = TangNhanh(D, decay=0.9, kieu="truc", truc_json=str(p))
    g = torch.Generator().manual_seed(4)
    for _ in range(20):
        tn.cap_nhat(torch.randn(32, D, generator=g) * 0.05)
    tn.chot_moc()
    nh = NganHangCheDo(D, nguong=0.5, k_max=8)
    dich_dk = torch.zeros(D); dich_dk[0] = 2.0     # điều kiện A (trên trục)
    _bay_mot_chuyen(tn, nh, dich_dk, seed=40)
    dich_lop = dich_dk.clone(); dich_lop[3] = 4.0  # cùng điều kiện, tỷ lệ lớp đổi (⊥ trục)
    _bay_mot_chuyen(tn, nh, dich_lop, seed=41)
    assert len(nh.che_do) == 1, "dịch ⊥ trục phải khớp lại chế độ cũ, không tạo mới"


def test_khop_bang_trung_binh_TUOI_khong_bi_nhiem_chuyen_truoc():
    """⭐ Bug bắt bằng mô phỏng đầu-cuối 2026-08-04: tại batch cho_khop_sau, EMA m_t còn
    nhiễm điều kiện chuyến TRƯỚC (λ=0,99, B=32, 5 batch -> còn ~20%; λ cao hơn còn nặng
    hơn) — khớp bằng m_t có thể nạp nhầm CHÍNH chế độ vừa rời khỏi. Khớp phải dùng
    `m_tuoi` (trung bình chỉ của chuyến hiện tại)."""
    tn, nh = _tn_da_chot(), NganHangCheDo(D, nguong=1.0, k_max=8)
    dich_a = torch.zeros(D); dich_a[0] = 5.0
    dich_b = torch.zeros(D); dich_b[0] = -5.0
    _bay_mot_chuyen(tn, nh, dich_a, seed=60)       # bank: [A]
    _bay_mot_chuyen(tn, nh, dich_b, seed=61)       # bank: [A, B]; m_t hiện ≈ B
    m_a_luu = nh.che_do[0]["m"].clone()
    # Quay lại A nhưng m_t VẪN CÒN Ở B (đầu chuyến, EMA chưa rửa) — chỉ m_tuoi biết là A.
    assert nh.khop_va_nap(tn, m_tuoi=dich_a + 0.01), "m_tuoi ≈ A phải khớp được"
    assert torch.equal(tn.m_t, m_a_luu), "phải nạp đúng chế độ A, không phải B"
    nh._khop_hien_tai = None
    # Đối chứng âm: không đưa m_tuoi -> rơi về m_t (≈ A vừa nạp) — hành vi cũ vẫn chạy.
    assert nh.khop_va_nap(tn) is True


def test_k_max_tran_thi_gop_hai_che_do_gan_nhat():
    tn, nh = _tn_da_chot(), NganHangCheDo(D, nguong=0.3, k_max=2)
    for k, do_lon in enumerate((5.0, 5.6, -5.0)):  # ↳ hai cái đầu gần nhau, cái ba xa
        dich = torch.zeros(D); dich[0] = do_lon
        _bay_mot_chuyen(tn, nh, dich, seed=50 + k)
    assert len(nh.che_do) == 2, "tràn k_max phải gộp về đúng k_max"


# ================================ tích hợp SLDA ================================
def test_bat_bien_nguoc_khong_khai_bao_tang_nao():
    """Không khai báo tang_nhanh/ngan_hang -> hành vi TRÙNG BIT bản SLDA cũ."""
    a, b = _mk_slda(), _mk_slda(tang_nhanh=None, ngan_hang=None)
    f, ys = _du_lieu()
    a.update(f, ys); b.update(f, ys)
    assert a.tang_nhanh is None and a.ngan_hang is None
    assert torch.equal(a(f), b(f))


def test_ngan_hang_doi_tang_nhanh():
    with pytest.raises(ValueError):
        _mk_slda(ngan_hang={"enabled": True})


def test_pha1_tang_nhanh_bat_van_trung_voi_khong_bat():
    """Trước khi chốt mốc, căn chỉnh là identity -> kết quả pha 1 không đổi."""
    a = _mk_slda()
    b = _mk_slda(tang_nhanh={"enabled": True, "decay": 0.9})
    f, ys = _du_lieu()
    a.update(f, ys); b.update(f, ys)
    assert torch.allclose(a(f), b(f), atol=1e-5)


def test_hap_thu_khong_nhan_khong_dung_toi_nhan():
    """LUẬT P1: không một nhãn nào lọt vào update ở pha 2 — count_raw phải đứng yên,
    và đưa nhãn SAI vào loader không đổi bất kỳ dự đoán nào."""
    m = _mk_slda(tang_nhanh={"enabled": True, "decay": 0.9})
    f, ys = _du_lieu()
    m.update(f, ys)
    m.chot_moc_pha1()
    raw_truoc = m.count_raw.clone()
    f2, y2 = _du_lieu(128, seed=9, dich=torch.full((D,), 2.0))
    loader_dung = [(f2[i:i + 32], y2[i:i + 32]) for i in range(0, 128, 32)]
    y_sai = torch.randint(0, C, y2.shape)
    loader_sai = [(f2[i:i + 32], y_sai[i:i + 32]) for i in range(0, 128, 32)]
    m.hap_thu_khong_nhan(loader_dung, torch.device("cpu"))
    assert torch.equal(m.count_raw, raw_truoc), "pha 2 mà count_raw nhúc nhích = nhãn lọt vào"
    m2 = _mk_slda(tang_nhanh={"enabled": True, "decay": 0.9})
    m2.update(f, ys); m2.chot_moc_pha1()
    m2.hap_thu_khong_nhan(loader_sai, torch.device("cpu"))
    assert torch.allclose(m.tang_nhanh.m_t, m2.tang_nhanh.m_t), \
        "nhãn đúng hay sai phải cho m_t y hệt — update không được nhìn nhãn"


def test_prequential_tang_nhanh_hoi_phuc_con_dong_bang_thi_khong():
    """Bản thu nhỏ của U0 vs U1: điều kiện dịch mạnh -> U0 sập; U1 hồi phục sau vài batch.

    Hướng dịch lấy theo hình học lớp (2× vector nối hai tâm) chứ không phải một hằng số:
    dịch theo hằng số có thể tình cờ SONG SONG với biên quyết định -> U0 không sập, test
    đo nhầm. Dịch 2× khoảng cách lớp thì đám mây lớp 0 chắc chắn rơi qua vùng lớp khác.
    """
    dich = 2.0 * (MU[1] - MU[0])
    ket_qua = {}
    for ten, tn_cfg in (("U0", None), ("U1", {"enabled": True, "decay": 0.9})):
        m = _mk_slda(tang_nhanh=tn_cfg)
        f, ys = _du_lieu(400, seed=5)
        m.update(f, ys)
        m.chot_moc_pha1()
        f2, y2 = _du_lieu(320, seed=6, dich=dich)
        loader = [(f2[i:i + 32], y2[i:i + 32]) for i in range(0, 320, 32)]
        accs = m.hap_thu_khong_nhan(loader, torch.device("cpu"))
        assert len(accs) == 10, "prequential phải có đúng 1 điểm mỗi batch"
        ket_qua[ten] = accs
    cuoi_u1 = sum(ket_qua["U1"][-3:]) / 3
    cuoi_u0 = sum(ket_qua["U0"][-3:]) / 3
    assert cuoi_u1 > 0.85, f"U1 phải hồi phục (acc cuối {cuoi_u1:.3f})"
    assert cuoi_u1 > cuoi_u0 + 0.2, f"U1 ({cuoi_u1:.3f}) phải vượt xa U0 ({cuoi_u0:.3f})"


def test_memory_report_gom_ca_ba_tang():
    m = _mk_slda(tang_nhanh={"enabled": True, "decay": 0.9},
                 ngan_hang={"enabled": True, "nguong": 0.5})
    f, ys = _du_lieu()
    m.update(f, ys); m(f)
    r = m.memory_report()
    assert "tang_nhanh" in r and r["tang_nhanh"] > 0
    assert "ngan_hang" in r                        # ↳ 0 byte khi chưa có chế độ nào — vẫn phải có mặt
