"""A1 (KE_HOACH_SUA 2026-08-04) — cờ `memory.post_norm` mở khoá ablation D11 Ưu tiên 2.

D11 §2.1: đổi trần η gấp 100 lần -> Δacc = 0,00002. Nghi phạm: LayerNorm ngay sau memory
chuẩn hoá lại thang đo, triệt tiêu độ lớn bước ghi mà η điều khiển. Trước bản vá, lớp này
bật CỨNG — ablation không chạy được bằng config.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("titans_pytorch")

from uavcl.models.titans_head import TitansClassifier  # noqa: E402

D = 64


class _FakeBackbone(torch.nn.Module):
    def forward(self, x):
        return x


def _mk(**memory_cfg):
    torch.manual_seed(0)
    cfg = {"dim": "auto", "chunk_size": 8, "seq": "image_seq", "reset": "image"}
    cfg.update(memory_cfg)
    return TitansClassifier(_FakeBackbone(), D, 5, cfg)


def test_mac_dinh_post_norm_la_layernorm():
    m = _mk()
    assert isinstance(m.post_norm, torch.nn.LayerNorm), \
        "không khai báo key -> giữ đúng hành vi cũ (LayerNorm)"


def test_post_norm_false_la_identity():
    m = _mk(post_norm=False)
    assert isinstance(m.post_norm, torch.nn.Identity)


def test_tat_post_norm_doi_output_that():
    """Hai model chỉ khác cờ này phải cho output KHÁC nhau — nếu trùng thì cờ vô dụng."""
    a, b = _mk(), _mk(post_norm=False)
    b.load_state_dict(a.state_dict(), strict=False)   # ↳ đồng bộ phần tham số chung
    x = torch.randn(2, D)
    with torch.no_grad():
        ya, yb = a(x), b(x)
    assert not torch.allclose(ya, yb), "bật/tắt post_norm phải thay đổi đầu ra"
