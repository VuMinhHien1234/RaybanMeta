"""Test cho stream BAY LẶP LẠI + thước đo O1/O3.

Thuần Python, không cần torch — chạy được ở mọi máy.

Bài toán (BAI_TOAN_VA_MUC_TIEU §1): drone bay lại cùng khu vực, điều kiện đã đổi, phải
(a) vẫn nhận đúng, (b) bám trôi không nhãn, (c) **gặp lại điều kiện cũ thì nhận ra ngay**.

Vế (c) — mục tiêu O3 — là thứ AAA/Forgetting KHÔNG đo được, nên phần lớn test dưới đây bảo
vệ đúng chỗ đó.
"""
from __future__ import annotations

import pytest

from uavcl.data.revisit import ChuyenBay, bang_lich_bay, kiem_lich, lich_bay
from uavcl.metrics.revisit import (acc_hien_tai, loi_ich_quay_lai, thoi_gian_hoi_phuc,
                                   tom_tat_revisit)


# ------------------------------------------------------------------ lịch bay

def test_tuan_hoan_thi_dieu_kien_QUAY_LAI():
    """⭐ Lý do tồn tại của cả file: chuyến `chu_ky` phải lặp lại điều kiện chuyến 0.

    Nếu không lặp thì đây chỉ là `drift.py` đổi tên, và O3 không đo được.
    """
    lich = lich_bay(12, che_do="tuan_hoan", chu_ky=4)
    assert lich[4].muc_troi == pytest.approx(lich[0].muc_troi, abs=1e-9)
    assert lich[8].muc_troi == pytest.approx(lich[0].muc_troi, abs=1e-9)
    assert lich[5].muc_troi == pytest.approx(lich[1].muc_troi, abs=1e-9)
    assert lich[4].mode_that == lich[0].mode_that
    assert lich[4].lan_gap_mode == 2 and lich[8].lan_gap_mode == 3


def test_troi_dan_thi_KHONG_lap():
    """Chế độ `troi_dan` phải giữ đúng hành vi D10 — đơn điệu, không quay lại."""
    lich = lich_bay(9, che_do="troi_dan")
    muc = [c.muc_troi for c in lich]
    assert muc == sorted(muc)
    assert muc[0] == pytest.approx(0.0) and muc[-1] == pytest.approx(1.0)


def test_hon_hop_co_ca_mua_VA_troi_dai_han():
    """Thực tế nhất: khí hậu ấm dần CỘNG mùa quay vòng."""
    lich = lich_bay(12, che_do="hon_hop", chu_ky=4, troi_dai_han=0.5)
    m = [c.muc_troi for c in lich]
    assert m[4] > m[0], "phải có xu thế dài hạn đi lên"
    assert m[2] > m[4], "trong một chu kỳ vẫn phải có dao động mùa"


def test_cung_muc_troi_thi_CUNG_che_do():
    """⭐ Bẫy dấu phẩy động: cos(π/2) trả 6,1e-17 chứ không phải 0.

    Chuyến 1 và chuyến 3 đều ở mức 50%, nhưng nếu không làm tròn thì ra
    0.49999999999999994 và 0.5000000000000001 -> rơi vào HAI chế độ khác nhau. Cùng một
    điều kiện vật lý mà bị coi là hai điều kiện thì O3 đo ra số vô nghĩa.
    """
    lich = lich_bay(12, che_do="tuan_hoan", chu_ky=4)
    assert lich[1].muc_troi == lich[3].muc_troi
    assert lich[1].mode_that == lich[3].mode_that


def test_dem_theo_DOT_khong_theo_so_lan_xuat_hien():
    """⭐ Chuyến liên tiếp cùng chế độ là CÙNG MỘT ĐỢT, không phải "quay lại".

    Điều kiện chưa rời đi thì chưa gọi là quay lại được. Không phân biệt chỗ này thì stream
    trôi đơn điệu sẽ báo có quay lại một cách giả tạo (các mức gần nhau rơi cùng ô rời rạc),
    và O3 sẽ đo một hiện tượng không tồn tại.
    """
    lich = lich_bay(9, che_do="troi_dan")
    assert [c.mode_that for c in lich] == [0, 0, 1, 1, 2, 2, 2, 3, 3]
    assert all(c.lan_gap_mode == 1 for c in lich), "trôi đơn điệu KHÔNG có lần quay lại nào"


def test_lich_khong_lap_bi_TU_CHOI():
    """⭐ Cửa chặn: lịch không có chuyến lặp thì O3 không đo được -> phải chết SỚM.

    Chạy 8 tiếng rồi mới phát hiện thước đo chính không tính được là kiểu hỏng tệ nhất.
    """
    with pytest.raises(ValueError, match="quay lại|lặp lại"):
        kiem_lich(lich_bay(9, che_do="troi_dan"))
    kiem_lich(lich_bay(12, che_do="tuan_hoan", chu_ky=4))     # hợp lệ -> không ném


@pytest.mark.parametrize("kw", [{"n_chuyen": 0}, {"chu_ky": 1}, {"che_do": "bay_bong"}])
def test_tham_so_vo_ly_bi_tu_choi(kw):
    with pytest.raises(ValueError):
        lich_bay(**{"n_chuyen": 12, "che_do": "tuan_hoan", "chu_ky": 4, **kw})


def test_bang_lich_bay_neu_ro_so_chuyen_lap():
    s = bang_lich_bay(lich_bay(12, che_do="tuan_hoan", chu_ky=4))
    assert "LẶP LẠI" in s and "/12" in s


# ------------------------------------------------------------------ thước đo

def _R(cheo):
    """Ma trận dưới giả với đường chéo cho trước (chỉ đường chéo được dùng)."""
    n = len(cheo)
    return [[cheo[j] if j <= i else 0.0 for j in range(n)] for i in range(n)]


def test_acc_hien_tai_la_trung_binh_duong_cheo():
    assert acc_hien_tai(_R([0.8, 0.6, 0.7, 0.9])) == pytest.approx(0.75)


def test_loi_ich_quay_lai_duong_khi_co_GHI_NHO():
    """⭐ Thước đo trung tâm. Chế độ 0 gặp lại ở chuyến 2 với accuracy CAO HƠN."""
    r = loi_ich_quay_lai(_R([0.60, 0.50, 0.75, 0.55]), mode_that=[0, 1, 0, 1])
    assert r["loi_ich_quay_lai"] == pytest.approx(((0.75 - 0.60) + (0.55 - 0.50)) / 2)
    assert r["so_lan_quay_lai"] == 2 and r["so_che_do"] == 2


def test_chuyen_lien_tiep_cung_che_do_KHONG_tinh_la_quay_lai():
    """⭐ Cùng bẫy như trên, nhưng ở phía thước đo.

    Bốn chuyến liên tiếp cùng chế độ = một đợt duy nhất. Nếu đếm nhầm thành 3 lần quay lại
    thì mọi stream trôi đơn điệu đều "có ghi nhớ" một cách giả tạo.
    """
    r = loi_ich_quay_lai(_R([0.6, 0.6, 0.6, 0.9]), mode_that=[0, 0, 0, 0])
    assert r["so_lan_quay_lai"] == 0
    assert r["loi_ich_quay_lai"] == pytest.approx(0.0)


def test_loi_ich_bang_0_khi_KHONG_ghi_nho():
    """Hệ thích nghi lại từ đầu mỗi lần -> accuracy y hệt lần đầu -> lợi ích = 0 -> O3 hỏng."""
    r = loi_ich_quay_lai(_R([0.6, 0.5, 0.6, 0.5]), mode_that=[0, 1, 0, 1])
    assert r["loi_ich_quay_lai"] == pytest.approx(0.0)


def test_loi_ich_AM_khi_QUEN_dieu_kien_cu():
    """Ca đáng lo nhất: λ nhỏ quá, quay lại điều kiện cũ thì TỆ HƠN lần đầu.

    Đây đúng là giả thuyết cần kiểm — D10 đo trên stream không lặp nên không thấy được.
    """
    r = loi_ich_quay_lai(_R([0.60, 0.50, 0.45, 0.40]), mode_that=[0, 1, 0, 1])
    assert r["loi_ich_quay_lai"] < 0


def test_loi_ich_tach_theo_LAN_gap():
    """Gặp lần 3 phải tốt hơn lần 2 nếu bộ nhớ điều kiện tích luỹ dần.

    (Sửa 2026-08-04: bản cũ dùng mode_that=[0,0,0] — ba chuyến LIÊN TIẾP cùng chế độ là
    MỘT ĐỢT duy nhất theo đúng luật ⭐ của test_chuyen_lien_tiep_cung_che_do phía trên,
    nên không có "lần 2/3" nào để tách — hai test tự mâu thuẫn. Lịch phải RỜI chế độ rồi
    QUAY LẠI thì mới tính lần; chèn mode đệm 1, 2 mỗi mode chỉ gặp một lần để không
    đóng góp vào lợi ích.)"""
    r = loi_ich_quay_lai(_R([0.60, 0.50, 0.70, 0.50, 0.80]), mode_that=[0, 1, 0, 2, 0])
    assert r["loi_ich_lan_2"] == pytest.approx(0.10)
    assert r["loi_ich_lan_3"] == pytest.approx(0.20)


def test_khong_co_lan_lap_thi_loi_ich_la_0_va_dem_bang_0():
    r = loi_ich_quay_lai(_R([0.6, 0.5, 0.4]), mode_that=[0, 1, 2])
    assert r["so_lan_quay_lai"] == 0 and r["loi_ich_quay_lai"] == 0.0


def test_thoi_gian_hoi_phuc():
    assert thoi_gian_hoi_phuc([0.1, 0.3, 0.5, 0.79, 0.80, 0.80], nguong=0.95) == 3.0
    assert thoi_gian_hoi_phuc([0.80, 0.80, 0.80]) == 0.0
    assert thoi_gian_hoi_phuc([]) == float("inf")


def test_khong_hoi_phuc_thi_bao_inf_chu_khong_lap_liem():
    """Accuracy tụt dần đều -> không bao giờ về ngưỡng. Phải trả inf để còn báo động."""
    assert thoi_gian_hoi_phuc([0.9, 0.7, 0.5, 0.3, 0.1]) == 0.0     # bước 0 đã ≥ ngưỡng
    assert thoi_gian_hoi_phuc([0.1, 0.1, 0.1, 0.9], nguong=10.0) == float("inf")


def test_tom_tat_gom_du_cac_khoa():
    tt = tom_tat_revisit(_R([0.6, 0.5, 0.75, 0.55]), [0, 1, 0, 1])
    for k in ("acc_dieu_kien_hien_tai", "loi_ich_quay_lai", "so_lan_quay_lai",
              "acc_chuyen_dau", "acc_chuyen_cuoi"):
        assert k in tt


# ------------------------------------------------------------------ bất biến ngược

def test_khong_bat_revisit_thi_loaders_di_duong_cu():
    """`stream_type` khác 'revisit' -> `lich` là None -> dùng đúng nhánh drift cũ.

    Bảo vệ 15 run của D10: chúng phải tái lập được y hệt sau khi thêm file này.
    """
    pytest.importorskip("torch")       # ↳ loaders import torch; máy thuần Python thì skip
    from uavcl.data import loaders as L
    import inspect
    src = inspect.getsource(L.build_task_loaders)
    assert 'stream_type", "")).lower() == "revisit"' in src, \
        "nhánh revisit phải có điều kiện rõ ràng, không được bật mặc định"
    assert "lich = None" in src, "mặc định phải là None = đi đường cũ"
