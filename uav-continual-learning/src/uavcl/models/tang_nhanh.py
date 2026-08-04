"""M1 — TẦNG NHANH: thống kê điều kiện hiện tại, KHÔNG nhãn, O(D) bộ nhớ (~3 KB).

Vai trò trong bộ ba tầng (KE_HOACH_SUA / KE_HOACH_BA_TANG 2026-08-04):

    tầng chậm  = SLDA (μ_c, Σ)     — danh tính lớp, học MỘT lần ở pha 1 CÓ nhãn
    TẦNG NHANH = file này           — "điều kiện lúc này", cập nhật MỖI batch, KHÔNG nhãn
    tầng trung = ngan_hang_che_do   — "các điều kiện đã từng gặp", 1 lần/chuyến

Ý tưởng: điều kiện quan sát (nắng, sương, bụi) dịch TOÀN BỘ đám mây feature; thông tin lớp
nằm ở vị trí TƯƠNG ĐỐI giữa các mẫu. Vậy theo dõi trung bình/phương sai chạy của dòng feature
(m_t, v_t) rồi căn mọi feature về hệ toạ độ của pha hiệu chỉnh e₀ (m0, v0) — tầng chậm bên
dưới không hề biết điều kiện đã đổi.

Đây CHÍNH LÀ cổng quên của Titans/NL nhưng bằng công thức đóng: λ cố định nên về mặt cấu
trúc KHÔNG THỂ trôi ra biên như cổng học được đã làm (CHAN_DOAN_NL: bão hoà 9/13 run).

Hai kiểu căn chỉnh — T1 (do_truc_dieu_kien.py) quyết định dùng kiểu nào:

    day_du : f' = (f − m_t) · sqrt(v0 / v_t) + m0        — dịch + co giãn từng chiều
    truc   : f' = f − ((m_t − m0) · û) û                  — CHỈ sửa thành phần trên trục
             điều kiện û; phần vuông góc (mang thông tin lớp) GIỮ NGUYÊN. Đây là thuốc
             chống bẫy T2: đổi tỷ lệ lớp dịch m_t theo hướng ⊥ trục sẽ KHÔNG bị "sửa nhầm".

Vòng đời: pha 1 (có nhãn) chỉ TÍCH LUỸ m_t/v_t; cuối pha 1 gọi `chot_moc()` -> (m0, v0)
đóng băng, căn chỉnh bắt đầu có tác dụng. Trước khi chốt, `can_chinh` là identity — chuyến
hiệu chỉnh tự nó là mốc, không có gì để sửa.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

KIEU = ("day_du", "truc")


class TangNhanh:
    """Không phải nn.Module có chủ đích: không tham số học được, không đi vào optimizer.

    State sống bằng tensor thường; `export_state`/`import_state` phục vụ checkpoint.
    """

    def __init__(self, dim: int, decay: float = 0.99, kieu: str = "day_du",
                 truc_json: str | None = None, eps: float = 1e-6):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"tang_nhanh.decay phải thuộc (0, 1) (nhận {decay}) — "
                             "decay=1 nghĩa là không bám gì, dùng U0 thay vì bật tầng nhanh")
        if kieu not in KIEU:
            raise ValueError(f"tang_nhanh.kieu phải thuộc {KIEU} (nhận {kieu!r})")
        self.dim = int(dim)
        self.decay = float(decay)          # ↳ λ ≈ 0,99 -> cửa sổ hiệu dụng ~1/(1−λ) = 100 mẫu
        self.kieu = str(kieu)
        self.eps = float(eps)
        # Chặn cấu hình vô nghĩa TRƯỚC khi đốt giờ máy: kiểu 'truc' mà không có trục thì
        # không có gì để chiếu — bắt lỗi ở đây chứ không phải sau 1 đêm chạy.
        self.truc: torch.Tensor | None = None
        if self.kieu == "truc":
            if not truc_json:
                raise ValueError("tang_nhanh.kieu='truc' cần tang_nhanh.truc_json "
                                 "(file T1 sinh bởi scripts/do_truc_dieu_kien.py --json)")
            p = Path(truc_json)
            if not p.exists():
                raise FileNotFoundError(f"Không thấy file trục điều kiện: {truc_json} — "
                                        "chạy T1 trước (cửa chặn nhóm B của KE_HOACH_SUA)")
            u = torch.tensor(json.loads(p.read_text())["truc_dieu_kien"], dtype=torch.float32)
            if u.numel() != self.dim:
                raise ValueError(f"Trục điều kiện dài {u.numel()} ≠ dim {self.dim}")
            self.truc = u / (u.norm() + 1e-12)
        # m_t/v_t: trung bình & phương sai CHẠY của dòng feature THÔ (chưa căn chỉnh).
        self.m_t: torch.Tensor | None = None     # (D,)
        self.v_t: torch.Tensor | None = None     # (D,) — CHỈ đường chéo (3 KB, ổn định ước lượng)
        self.m0: torch.Tensor | None = None      # mốc pha 1 — None = chưa chốt
        self.v0: torch.Tensor | None = None
        self.so_mau = 0                          # tổng mẫu đã thấy (chẩn đoán λ, như count_raw)

    # ------------------------------------------------------------------ cập nhật (KHÔNG nhãn)
    @torch.no_grad()
    def cap_nhat(self, feats: torch.Tensor) -> None:
        """Hấp thụ 1 batch feature THÔ (B, D). Không đụng tới nhãn — theo đúng phát biểu P1.

        EMA theo-mẫu, gộp theo batch: r = λ^B; m ← r·m + (1−r)·mean(batch).
        (Xấp xỉ chuẩn của EMA từng-mẫu khi batch cùng phân bố — sai số bậc cao khi điều
        kiện đổi chậm so với một batch, đúng giả định A4 "trôi trơn".)
        """
        f = feats.detach().float()
        if f.dim() != 2 or f.shape[1] != self.dim:
            raise ValueError(f"tang_nhanh nhận feature (B, {self.dim}), thấy {tuple(f.shape)}")
        if not torch.isfinite(f).all():
            raise FloatingPointError("tang_nhanh nhận feature chứa NaN/Inf")
        mb = f.mean(dim=0)
        vb = f.var(dim=0, unbiased=False) if f.shape[0] > 1 else torch.zeros_like(mb)
        if self.m_t is None:                     # ↳ batch đầu tiên = khởi tạo thẳng, khỏi bias-correction
            self.m_t, self.v_t = mb.clone(), vb.clone().clamp(min=self.eps)
        else:
            r = self.decay ** int(f.shape[0])
            self.m_t = r * self.m_t + (1.0 - r) * mb
            if f.shape[0] > 1:                   # ↳ batch 1 mẫu không có phương sai -> giữ nguyên v_t
                self.v_t = (r * self.v_t + (1.0 - r) * vb).clamp(min=self.eps)
        self.so_mau += int(f.shape[0])

    # ------------------------------------------------------------------ mốc pha 1
    @torch.no_grad()
    def chot_moc(self) -> None:
        """Cuối pha hiệu chỉnh: đóng băng (m0, v0). Từ đây `can_chinh` mới có tác dụng."""
        if self.m_t is None:
            raise RuntimeError("chot_moc() gọi khi chưa hấp thụ mẫu nào — pha 1 rỗng?")
        self.m0, self.v0 = self.m_t.clone(), self.v_t.clone()

    @property
    def da_chot(self) -> bool:
        return self.m0 is not None

    # ------------------------------------------------------------------ căn chỉnh
    @torch.no_grad()
    def can_chinh(self, feats: torch.Tensor) -> torch.Tensor:
        """Đưa feature về hệ toạ độ pha 1. Trước khi chốt mốc: identity (không có gì để sửa)."""
        if self.m0 is None or self.m_t is None:
            return feats
        f = feats.float()
        m_t, m0 = self.m_t.to(f.device), self.m0.to(f.device)
        if self.kieu == "truc":
            u = self.truc.to(f.device)
            do_doi = torch.dot(m_t - m0, u)      # ↳ vô hướng: điều kiện đã trôi bao xa TRÊN trục
            return f - do_doi * u                # ↳ phần ⊥ trục (thông tin lớp) không bị đụng
        v_t, v0 = self.v_t.to(f.device), self.v0.to(f.device)
        ty_le = torch.sqrt(v0.clamp(min=self.eps) / v_t.clamp(min=self.eps))
        return (f - m_t) * ty_le + m0

    # ------------------------------------------------------------------ tiện ích
    def extra_bytes(self) -> int:
        """Chi phí bộ nhớ thật — đối chiếu ràng buộc O4 (≤ 10 MB tổng)."""
        n = 0
        for t in (self.m_t, self.v_t, self.m0, self.v0, self.truc):
            if t is not None:
                n += t.numel() * t.element_size()
        return n

    def bao_cao(self) -> dict:
        """Bằng chứng chạy đúng, in ra log (vai trò như window_report của SLDA)."""
        do_troi = 0.0
        if self.da_chot and self.m_t is not None:
            d = self.m_t - self.m0
            do_troi = float(torch.dot(d, self.truc.to(d.device))) if self.truc is not None \
                else float(d.norm())
        return {"kieu": self.kieu, "decay": self.decay, "so_mau": self.so_mau,
                "da_chot": self.da_chot, "do_troi_hien_tai": do_troi,
                "bytes": self.extra_bytes()}

    def export_state(self) -> dict:
        return {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in dict(m_t=self.m_t, v_t=self.v_t, m0=self.m0, v0=self.v0,
                                 so_mau=self.so_mau).items()}

    def import_state(self, st: dict) -> None:
        self.m_t, self.v_t = st.get("m_t"), st.get("v_t")
        self.m0, self.v0 = st.get("m0"), st.get("v0")
        self.so_mau = int(st.get("so_mau", 0))
