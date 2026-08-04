"""P2 (KE_HOACH_SUA 2026-08-04) — chuyến pha 2 là DÒNG THỜI GIAN: trôi đổi DẦN trong chuyến.

Kèm A3: lịch bay phải đi qua MỘT nguồn duy nhất (`lich_tu_cfg`).
"""
from __future__ import annotations

import pytest

from uavcl.data.revisit import lich_bay, lich_tu_cfg  # noqa: E402


# ================================ A3 — một nguồn lịch bay ================================
def test_lich_tu_cfg_trung_voi_lich_bay_goi_tay():
    cfg = {"che_do": "tuan_hoan", "chu_ky": 4, "severity": 1.0, "n_mode": 3}
    a = lich_tu_cfg(12, cfg)
    b = lich_bay(12, che_do="tuan_hoan", chu_ky=4, severity=1.0, troi_dai_han=0.0, n_mode=3)
    assert [(c.muc_troi, c.mode_that, c.lan_gap_mode) for c in a] == \
           [(c.muc_troi, c.mode_that, c.lan_gap_mode) for c in b]


def test_lich_tu_cfg_dung_mac_dinh_khi_cfg_rong():
    a = lich_tu_cfg(8, {})
    b = lich_bay(8)                                # ↳ mặc định của lich_bay
    assert [(c.muc_troi, c.mode_that) for c in a] == [(c.muc_troi, c.mode_that) for c in b]


# ================================ P2 — trôi trong chuyến ================================
# importorskip đặt TRONG hàm (không phải mức module) để hai test A3 thuần Python phía trên
# vẫn chạy được ở máy chưa có torch.


def _ds(muc=0.5, bien_do=0.4, n=20):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    Image = pytest.importorskip("PIL.Image")
    from uavcl.data.loaders import TaskDatasetDongThoiGian

    class _SplitGia:
        """Split giả: n ảnh xám trơn — đủ cho test dataset, khỏi tải RESISC45."""

        def __init__(self, n_anh):
            self.labels = [i % 2 for i in range(n_anh)]

        def get_image(self, i):
            return Image.new("RGB", (32, 32), color=(128, 128, 128))

    return TaskDatasetDongThoiGian(_SplitGia(n), list(range(n)), image_size=32,
                                   muc_chuyen=muc, bien_do=bien_do, jitter=0.0, seed=7)


def test_muc_di_tuyen_tinh_tu_dau_den_cuoi_chuyen():
    ds = _ds(muc=0.5, bien_do=0.4)
    assert ds.muc_tai(0) == pytest.approx(0.3), "mẫu đầu = muc − bien_do/2"
    assert ds.muc_tai(19) == pytest.approx(0.7), "mẫu cuối = muc + bien_do/2"
    giua = ds.muc_tai(9)
    assert 0.3 < giua < 0.7, "giữa chuyến phải nằm giữa hai đầu"


def test_bien_do_0_la_dieu_kien_phang():
    ds = _ds(muc=0.5, bien_do=0.0)
    assert all(ds.muc_tai(k) == pytest.approx(0.5) for k in (0, 7, 19)), \
        "bien_do=0 phải ≡ hành vi cũ (một mức cho cả chuyến)"


def test_muc_bi_kep_trong_0_1():
    ds = _ds(muc=0.95, bien_do=0.3)
    assert ds.muc_tai(19) == 1.0, "vượt trần phải kẹp về 1.0"
    ds2 = _ds(muc=0.05, bien_do=0.3)
    assert ds2.muc_tai(0) == 0.0, "âm phải kẹp về 0.0"


def test_getitem_ra_tensor_chuan_va_tat_dinh():
    torch = pytest.importorskip("torch")
    ds = _ds()
    x1, y1 = ds[3]
    x2, _ = ds[3]
    assert x1.shape == (3, 32, 32) and isinstance(y1, int)
    assert torch.equal(x1, x2), "cùng chỉ số phải cho đúng một kết quả (seed theo mẫu)"


def test_muc_cang_cao_anh_cang_toi():
    """Trôi thật sự áp lên ảnh: mẫu cuối (trôi cao) phải TỐI hơn mẫu đầu (độ sáng ×(1−0.45m))."""
    ds = _ds(muc=0.5, bien_do=0.8)                 # ↳ đầu 0.1, cuối 0.9
    dau, _ = ds[0]
    cuoi, _ = ds[19]
    # bỏ chuẩn hoá ImageNet ra khỏi so sánh: so trên trung bình thô cùng kênh
    assert float(cuoi.mean()) < float(dau.mean()), "mức trôi cao hơn phải cho ảnh tối hơn"
