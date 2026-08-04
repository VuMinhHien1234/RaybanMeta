"""M2 — TẦNG TRUNG: ngân hàng chế độ điều kiện. Đây là thứ làm mục tiêu O3 khả thi.

Vấn đề nó giải: tầng nhanh bám điều kiện tốt nhưng KHÔNG NHỚ. Gặp lại mùa đông sau 6 tháng,
nó thích nghi lại từ đầu — mất đúng "thời gian hồi phục" như lần đầu. Ngân hàng chế độ lưu
trạng thái tầng nhanh của từng điều kiện đã gặp; khi nhận ra "điều kiện này quen quen" thì
NẠP LẠI trạng thái cũ -> thích nghi tức thời. Lợi ích đo bằng `loi_ich_quay_lai` (O3).

Cơ chế (đúng thiết kế KE_HOACH_BA_TANG, M2):

    mỗi chế độ k:  (m_k, v_k, so_lan_gap)          — chính là snapshot tầng nhanh
    khớp:          k* = argmin d(m_t, m_k)          — d đo TRÊN TRỤC điều kiện nếu có trục,
                                                      ngược lại L2 đầy đủ
    d ≤ nguong  -> GẶP LẠI: nạp (m_k, v_k) vào tầng nhanh, so_lan_gap += 1
    d >  nguong -> điều kiện MỚI: cuối chuyến ghi chế độ mới
    len > k_max -> gộp hai chế độ GẦN NHAU NHẤT (trung bình có trọng số theo so_lan_gap)

Vì sao khớp trên trục quan trọng (bẫy T2): drone bay từ thành phố sang rừng làm tỷ lệ lớp
đổi -> m_t dịch theo hướng gần ⊥ trục điều kiện. Khớp bằng L2 đầy đủ sẽ tưởng đó là điều
kiện mới và nổ số chế độ; khớp trên trục thì thành phần ⊥ bị chiếu bỏ, không đánh lừa được.
=> kieu='truc' là khuyến nghị mặc định một khi T1 xác nhận có trục.

Ba tham số nguy hiểm (chọn sai là tầng này vô dụng hoặc có hại — bảng trong KE_HOACH_BA_TANG):
`nguong` (quá nhỏ: nổ số chế độ · quá lớn: gộp hết làm một), `k_max`, `lam_mode` (EMA khi
cập nhật chế độ đã có). Cửa chặn riêng: test "số chế độ tự tạo = số điều kiện thật".
"""
from __future__ import annotations

import torch

from .tang_nhanh import TangNhanh


class NganHangCheDo:
    def __init__(self, dim: int, nguong: float = 0.5, k_max: int = 8,
                 cho_khop_sau: int = 5, lam_mode: float = 0.5):
        if nguong <= 0:
            raise ValueError(f"ngan_hang.nguong phải > 0 (nhận {nguong})")
        if k_max < 2:
            raise ValueError(f"ngan_hang.k_max phải >= 2 (nhận {k_max})")
        if cho_khop_sau < 1:
            raise ValueError(f"ngan_hang.cho_khop_sau phải >= 1 (nhận {cho_khop_sau})")
        if not (0.0 < lam_mode <= 1.0):
            raise ValueError(f"ngan_hang.lam_mode phải thuộc (0, 1] (nhận {lam_mode})")
        self.dim = int(dim)
        self.nguong = float(nguong)
        # ↳ cho_khop_sau: đầu chuyến, đợi tầng nhanh nhìn đủ N batch cho m_t bớt nhiễu rồi
        #   mới tra ngân hàng. Tra ngay batch 1 thì m_t còn là nhiễu của 32 mẫu đầu.
        self.cho_khop_sau = int(cho_khop_sau)
        self.k_max = int(k_max)
        self.lam_mode = float(lam_mode)
        self.che_do: list[dict] = []             # mỗi phần tử: {m, v, so_lan_gap}
        self._khop_hien_tai: int | None = None   # chế độ đã khớp trong CHUYẾN đang bay
        self.so_lan_nap = 0                      # tổng số lần "gặp lại -> nạp" (chẩn đoán O3)

    # ------------------------------------------------------------------ khoảng cách
    def _d(self, a: torch.Tensor, b: torch.Tensor, truc: torch.Tensor | None) -> float:
        if truc is not None:
            u = truc.to(a.device)
            return abs(float(torch.dot(a - b, u)))   # ↳ |chiếu lên trục| — miễn nhiễm bẫy T2
        return float((a - b).norm())

    def _gan_nhat(self, m: torch.Tensor, truc: torch.Tensor | None):
        if not self.che_do:
            return None, None
        ds = [self._d(m, c["m"], truc) for c in self.che_do]
        k = int(min(range(len(ds)), key=ds.__getitem__))
        return ds[k], k

    # ------------------------------------------------------------------ đầu chuyến
    @torch.no_grad()
    def khop_va_nap(self, tn: TangNhanh, m_tuoi: torch.Tensor | None = None) -> bool:
        """Gọi MỘT lần mỗi chuyến, sau `cho_khop_sau` batch. Trả True nếu GẶP LẠI.

        Gặp lại -> NẠP snapshot cũ vào tầng nhanh: đây chính là "nhận ra ngay, không học
        lại" của O3. Tầng nhanh vẫn tiếp tục EMA sau đó nên lệch nhỏ tự được mài phẳng.

        ⚠️ `m_tuoi` (BẮT BUỘC dùng khi có) — trung bình feature CHỈ CỦA CHUYẾN NÀY.
        Bug bắt được bằng mô phỏng 2026-08-04: tại batch thứ `cho_khop_sau`, EMA `tn.m_t`
        còn nhiễm ~λ^(B·cho_khop_sau) điều kiện của chuyến TRƯỚC (λ=0,99, B=32, 5 batch
        -> còn ~20%; λ=0,995 -> còn 67%). Khớp bằng m_t nhiễm thì ngân hàng có thể nạp
        nhầm CHÍNH chế độ vừa rời khỏi. Khớp phải dùng ước lượng TƯƠI của chuyến hiện tại.
        """
        self._khop_hien_tai = None
        m = m_tuoi if m_tuoi is not None else tn.m_t
        if m is None:
            return False
        d, k = self._gan_nhat(m, tn.truc)
        if d is None or d > self.nguong:
            return False                          # ↳ điều kiện lạ — cuối chuyến sẽ ghi mới
        self._khop_hien_tai = k
        self.che_do[k]["so_lan_gap"] += 1
        self.so_lan_nap += 1
        tn.m_t = self.che_do[k]["m"].clone()      # ↳ NẠP: thích nghi tức thời
        tn.v_t = self.che_do[k]["v"].clone()
        return True

    # ------------------------------------------------------------------ cuối chuyến
    @torch.no_grad()
    def ghi_lai(self, tn: TangNhanh) -> None:
        """Cuối chuyến: cập nhật chế độ đã khớp, hoặc ghi chế độ MỚI nếu chuyến này lạ."""
        if tn.m_t is None:
            return
        if self._khop_hien_tai is not None:       # ↳ chuyến quen: mài chế độ cũ theo EMA
            c = self.che_do[self._khop_hien_tai]
            c["m"] = (1.0 - self.lam_mode) * c["m"] + self.lam_mode * tn.m_t
            c["v"] = (1.0 - self.lam_mode) * c["v"] + self.lam_mode * tn.v_t
        else:                                     # ↳ chuyến lạ: chế độ mới
            self.che_do.append({"m": tn.m_t.clone(), "v": tn.v_t.clone(), "so_lan_gap": 1})
            if len(self.che_do) > self.k_max:
                self._gop_gan_nhat(tn.truc)
        self._khop_hien_tai = None

    @torch.no_grad()
    def _gop_gan_nhat(self, truc: torch.Tensor | None) -> None:
        """K tràn -> gộp HAI chế độ gần nhau nhất (trung bình trọng số theo so_lan_gap)."""
        n = len(self.che_do)
        tot, cap = None, None
        for i in range(n):
            for j in range(i + 1, n):
                d = self._d(self.che_do[i]["m"], self.che_do[j]["m"], truc)
                if tot is None or d < tot:
                    tot, cap = d, (i, j)
        i, j = cap
        a, b = self.che_do[i], self.che_do[j]
        wa = float(a["so_lan_gap"]); wb = float(b["so_lan_gap"]); w = wa + wb
        a["m"] = (wa * a["m"] + wb * b["m"]) / w
        a["v"] = (wa * a["v"] + wb * b["v"]) / w
        a["so_lan_gap"] = int(w)
        del self.che_do[j]

    # ------------------------------------------------------------------ tiện ích
    def extra_bytes(self) -> int:
        return sum(c["m"].numel() * c["m"].element_size()
                   + c["v"].numel() * c["v"].element_size() for c in self.che_do)

    def bao_cao(self) -> dict:
        return {"so_che_do": len(self.che_do), "so_lan_nap": self.so_lan_nap,
                "so_lan_gap": [c["so_lan_gap"] for c in self.che_do],
                "bytes": self.extra_bytes()}

    def export_state(self) -> dict:
        return {"che_do": [{"m": c["m"].cpu(), "v": c["v"].cpu(),
                            "so_lan_gap": c["so_lan_gap"]} for c in self.che_do],
                "so_lan_nap": self.so_lan_nap}

    def import_state(self, st: dict) -> None:
        self.che_do = [{"m": c["m"], "v": c["v"], "so_lan_gap": int(c["so_lan_gap"])}
                       for c in st.get("che_do", [])]
        self.so_lan_nap = int(st.get("so_lan_nap", 0))
