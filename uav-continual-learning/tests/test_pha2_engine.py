"""P1 (KE_HOACH_SUA 2026-08-04) — engine phải gác đúng ranh giới pha 1 / pha 2.

Kiểm bằng model + method GIẢ để tách logic gác khỏi SLDA thật:
  - pha2 TẮT  -> fit_task gọi MỌI chuyến (đường cũ bất biến);
  - pha2 BẬT  -> fit_task CHỈ ở chuyến hiệu chỉnh; các chuyến sau đi hap_thu_khong_nhan;
                 chot_moc_pha1 gọi ĐÚNG MỘT lần, ngay cuối pha 1; log có trace.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uavcl.data.stream import TaskSpec  # noqa: E402
from uavcl.engine import run_continual  # noqa: E402

C = 3


class _ModelGia(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.goi_fit: list = []          # ghi qua method
        self.goi_hap_thu: list = []
        self.goi_chot = 0

    def forward(self, x):
        return torch.zeros(x.shape[0], C)

    def hap_thu_khong_nhan(self, loader, device, allowed=None):
        self.goi_hap_thu.append(len(list(loader)))
        return [0.5, 0.6]

    def chot_moc_pha1(self):
        self.goi_chot += 1


class _MethodGia:
    gradient_free = True

    def __init__(self, model):
        self._m = model

    def fit_task(self, model, loader, device):
        self._m.goi_fit.append(len(list(loader)))

    def end_task(self, model, loader, device, allowed):
        pass

    def footprint_floats(self, model):
        return 0


def _dan_dung(n_chuyen=3):
    x = torch.randn(8, 4)
    y = torch.randint(0, C, (8,))
    loader = [(x, y)]
    stream = [TaskSpec(task_id=t, classes=list(range(C)),
                       train_idx=[0], val_idx=[0], test_idx=[0]) for t in range(n_chuyen)]
    loaders = [{"train": loader, "val": loader, "test": loader} for _ in range(n_chuyen)]
    return stream, loaders


def test_pha2_tat_thi_duong_cu_bat_bien():
    model = _ModelGia()
    stream, loaders = _dan_dung()
    run_continual(model, _MethodGia(model), stream, loaders,
                  torch.device("cpu"), {}, verbose=False)
    assert len(model.goi_fit) == 3, "pha2 tắt: fit_task phải chạy MỌI chuyến"
    assert model.goi_hap_thu == [] and model.goi_chot == 0


def test_pha2_bat_gac_dung_ranh_gioi():
    model = _ModelGia()
    stream, loaders = _dan_dung()
    _, log = run_continual(model, _MethodGia(model), stream, loaders, torch.device("cpu"),
                           {"pha2": {"khong_nhan": True, "chuyen_hieu_chinh": 1}},
                           verbose=False)
    assert len(model.goi_fit) == 1, "chỉ chuyến 0 (pha 1) được thấy nhãn"
    assert len(model.goi_hap_thu) == 2, "chuyến 1..2 phải đi đường KHÔNG nhãn"
    assert model.goi_chot == 1, "chốt mốc đúng một lần, cuối pha 1"
    assert sorted(log["trace"].keys()) == [1, 2], "prequential phải được gom vào log"


def test_chuyen_hieu_chinh_2_thi_nhan_song_den_het_chuyen_1():
    model = _ModelGia()
    stream, loaders = _dan_dung(4)
    run_continual(model, _MethodGia(model), stream, loaders, torch.device("cpu"),
                  {"pha2": {"khong_nhan": True, "chuyen_hieu_chinh": 2}}, verbose=False)
    assert len(model.goi_fit) == 2 and len(model.goi_hap_thu) == 2
    assert model.goi_chot == 1
