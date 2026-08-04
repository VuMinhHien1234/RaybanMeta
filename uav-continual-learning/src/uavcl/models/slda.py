"""SLDA — Streaming Linear Discriminant Analysis trên backbone ĐÓNG BĂNG (task #21).

Vì sao cần (plans/TASKS_UAV_CL.md #21): baseline streaming rẻ + mạnh, chuẩn cho
continual learning trên feature cố định (Hayes & Kanan 2020). Khác NCM ở chỗ ngoài
trung bình mỗi class còn học MA TRẬN HIỆP PHƯƠNG SAI CHUNG của feature -> ranh giới
lớp xét cả "hình dạng" đám mây feature, không chỉ khoảng cách tới tâm.

Toán (streaming, KHÔNG đọc lại data cũ, mỗi mẫu chỉ thấy 1 lần):
    tích luỹ:  s_c  += f        (tổng feature theo class)
               n_c  += 1
               G    += f fᵀ     (tổng outer product toàn cục)
    suy ra:    μ_c  = s_c / n_c
               Σ_w  = (G − Σ_c n_c μ_c μ_cᵀ) / N          (within-class covariance, chính xác)
               Λ    = (Σ_w + ε I)⁻¹                        (shrinkage ε)
    dự đoán:   score_c(f) = (Λ μ_c)ᵀ f − ½ μ_cᵀ Λ μ_c      (LDA discriminant, prior đều)

Giao diện giữ đúng quy ước dự án: forward(x) -> "logits" (B, num_classes), nên
engine/mask_logits/metrics dùng chung không đổi. Prototype/covariance của class cũ
không bị ghi đè khi học class mới -> gần như không quên (như NCM), nhưng biết thêm
tương quan chiều feature. Bộ nhớ: C·D + D² + C float (D=384 -> ~150K float, không phình).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .classifier import mask_logits

COV_MODES = ("streaming", "identity", "frozen")


class SLDAClassifier(nn.Module):
    """`cov_mode` (B2 ablation, 2026-08-02) — tách xem +11,5 điểm đến từ đâu:

    - `streaming` (mặc định): Σ_w cập nhật liên tục từ mọi mẫu đã thấy. Bản gốc.
    - `identity`: ép Σ_w = I. Khi đó Λ = I/(1+ε) và
          argmax_c (μ_c·f − ½‖μ_c‖²)  ≡  argmin_c ‖f − μ_c‖²
      tức **ĐÚNG BẰNG NCM**, chạy qua y hệt đường code này. Đây là đối chứng sạch nhất:
      chênh lệch so với `streaming` CHÍNH LÀ đóng góp của hiệp phương sai, không lẫn
      bất kỳ khác biệt cài đặt nào khác.
    - `frozen`: tính Σ_w một lần sau task thứ `cov_freeze_after` rồi ĐÓNG BĂNG; μ_c vẫn
      cập nhật tiếp. Trả lời: Σ có cần cập nhật liên tục không, hay ước lượng một lần là đủ?

    Ba chế độ dùng chung mọi thứ còn lại -> so 1-biến tuyệt đối.
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int,
                 shrinkage: float = 1e-4, cov_mode: str = "streaming",
                 cov_freeze_after: int = 1, stats_dtype: str = "float64",
                 decay_mean: float = 1.0, decay_cov: float = 1.0,
                 tang_nhanh: dict | None = None, ngan_hang: dict | None = None):
        super().__init__()
        if shrinkage <= 0.0:
            raise ValueError(f"shrinkage phải > 0 (nhận {shrinkage})")
        if cov_mode not in COV_MODES:
            raise ValueError(f"cov_mode phải thuộc {COV_MODES} (nhận {cov_mode!r})")
        if cov_freeze_after < 1:
            raise ValueError("cov_freeze_after phải >= 1 (đếm theo SỐ TASK đã học xong)")
        if stats_dtype not in ("float64", "float32"):
            raise ValueError(f"stats_dtype phải là 'float64' hoặc 'float32' (nhận {stats_dtype!r})")
        if not (0.0 < decay_mean <= 1.0):
            raise ValueError(f"decay_mean phải thuộc (0, 1] (nhận {decay_mean})")
        if not (0.0 < decay_cov <= 1.0):
            raise ValueError(f"decay_cov phải thuộc (0, 1] (nhận {decay_cov})")
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)          # ↳ SLDA định nghĩa trên feature CỐ ĐỊNH.
        self.backbone.eval()
        self.shrinkage = float(shrinkage)
        self.cov_mode = str(cov_mode)
        self.cov_freeze_after = int(cov_freeze_after)
        self.stats_dtype = str(stats_dtype)
        # --- λ: hệ số QUÊN (2026-08-03) --------------------------------------------------
        # Bản gốc cộng dồn VĨNH VIỄN -> μ_c hội tụ về trung bình TOÀN BỘ lịch sử rồi đứng yên.
        # Tốt cho "lớp mới, điều kiện tĩnh"; HỎNG khi điều kiện quan sát trôi dần (mùa, nắng,
        # sương): sau 1 triệu mẫu, một mẫu mới chỉ dịch μ_c đi 1/1.000.001.
        #
        # λ < 1 làm mẫu cũ phân rã theo cấp số nhân -> cửa sổ nhớ hiệu dụng ≈ 1/(1−λ).
        # Đây CHÍNH LÀ cổng quên α của Titans/NL, nhưng **cố định** thay vì học được — và
        # chính vì cố định nên nó KHÔNG THỂ tự trôi ra biên như α đã làm (xem CHAN_DOAN_NL).
        #
        # Hai λ riêng, có lý do:
        #   decay_mean -> μ_c: chỉ D số/lớp, ước lượng dễ, nên bám trôi nhanh được
        #   decay_cov  -> Σ  : D² số (384² = 147.456), cần NHIỀU mẫu, λ nhỏ sẽ vỡ ước lượng
        self.decay_mean = float(decay_mean)
        self.decay_cov = float(decay_cov)
        dt = torch.float64 if stats_dtype == "float64" else torch.float32
        self._tasks_done = 0                 # ↳ tăng ở on_task_end(), dùng cho chế độ frozen.
        self.register_buffer("_frozen_sigma", torch.zeros(0, dtype=dt), persistent=True)  # rỗng = chưa đóng băng
        # buffer: đi theo state_dict/device, KHÔNG bị optimizer đụng vào.
        # float64 (mặc định): cộng dồn hàng chục nghìn mẫu rồi NGHỊCH ĐẢO ma trận — float32
        # lệch theo thứ tự cộng (streaming ≠ batch tới 1e-3); double thì khớp và ổn định số.
        #
        # RÀNG BUỘC TRIỂN KHAI (B6, 2026-08-02): **MPS không hỗ trợ float64**, và nhiều NPU
        # biên cũng vậy. Hai cách xử lý, đều hợp lệ:
        #   (a) stats_dtype="float32" — giảm nửa bộ nhớ (1,32 -> 0,66 MB), đổi lấy sai số
        #       tích luỹ. PHẢI đo trước khi dùng cho kết quả chính.
        #   (b) chạy backbone trên accelerator, giữ thống kê SLDA ở CPU-float64. Phần SLDA
        #       quá nhỏ nên CPU không thành nút cổ chai — đây là thiết kế nên dùng cho drone.
        self.register_buffer("feat_sum", torch.zeros(num_classes, feat_dim, dtype=dt))  # s_c
        self.register_buffer("count", torch.zeros(num_classes, dtype=dt))               # n_c
        self.register_buffer("gram", torch.zeros(feat_dim, feat_dim, dtype=dt))         # G = Σ f fᵀ
        # --- BỘ ĐẾM THỨ HAI, trên ĐỒNG HỒ TOÀN CỤC (sửa lỗi 2026-08-03) ------------------
        # Vì sao phải có: đẳng thức Σ_w = (G − Σ_c n_c μ_c μ_cᵀ)/N chỉ đúng khi G và (n_c, s_c)
        # được **đánh trọng số GIỐNG HỆT NHAU**. Với w_i tuỳ ý thì
        #     G = Σ w_i f f ᵀ,  s_c = Σ_{i∈c} w_i f,  n_c = Σ_{i∈c} w_i
        # mới triệt tiêu đúng. Nhưng thiết kế ở trên cố ý dùng HAI đồng hồ khác nhau
        # (G toàn cục, s_c theo-lớp) -> trọng số lệch -> phép trừ TRỪ QUÁ TAY -> Σ_w mất tính
        # xác định dương -> Λ = inv(Σ) thành rác. Test `test_lop_vang_mat_van_du_doan_dung`
        # bắt đúng lỗi này: lớp 0 vắng 1600 mẫu, G chỉ còn giữ e^-1.6 ≈ 20% đóng góp của nó,
        # mà `between` vẫn trừ đi 100%.
        #
        # Cách sửa: giữ thêm một cặp đếm nhỏ theo đồng hồ TOÀN CỤC, dùng RIÊNG cho đẳng thức Σ.
        # μ_c của bộ phân loại vẫn nằm trên đồng hồ theo-lớp -> lớp hiếm vẫn không bị xoá.
        # Giá phải trả: C·D + C số ≈ 0,14 MB (so với gram 1,18 MB) — rẻ, và là cách duy nhất
        # giữ được ĐỒNG THỜI "Σ đúng" và "không quên lớp cũ".
        self.register_buffer("feat_sum_g", torch.zeros(num_classes, feat_dim, dtype=dt))
        self.register_buffer("count_g", torch.zeros(num_classes, dtype=dt))
        # Số lần gặp THẬT, không bao giờ phân rã — chỉ để CHẨN ĐOÁN, không tham gia tính toán.
        # Vì sao cần: n_c chỉ hội tụ về 1/(1−λ) sau HÀNG NGHÌN lần gặp. Với RESISC45 mỗi lớp
        # chỉ được gặp 420 lần trong cả 9 task, nên so với tiệm cận là so nhầm mốc (đo 131 mà
        # "kỳ vọng" in ra 1000 -> tưởng hỏng). Mốc đúng là (1 − λ^m)/(1 − λ) với m lần gặp.
        self.register_buffer("count_raw", torch.zeros(num_classes, dtype=dt))
        self._cache_w = None       # (D, C) = Λ μ_cᵀ — cache để không nghịch đảo mỗi batch
        self._cache_b = None       # (C,)   = −½ μ_c Λ μ_c
        self._cache_version = -1   # bump theo _version mỗi lần update
        self._version = 0
        # --- BA TẦNG (M1/M2, KE_HOACH_SUA 2026-08-04) — mặc định TẮT -> hành vi cũ bất biến.
        # tầng nhanh: căn feature về hệ toạ độ pha 1 (điều kiện lúc này, KHÔNG nhãn, mỗi batch)
        # tầng trung: ngân hàng chế độ — nhớ các điều kiện đã gặp (1 lần/chuyến, KHÔNG nhãn)
        # Thống kê tầng chậm (μ_c, Σ) từ đây sống trong hệ toạ độ e₀: update() lẫn forward()
        # đều đi qua _can_chinh() trước.
        self.tang_nhanh = None
        self.ngan_hang = None
        tn_cfg = dict(tang_nhanh or {})
        if tn_cfg.get("enabled", False):
            from .tang_nhanh import TangNhanh
            self.tang_nhanh = TangNhanh(
                dim=feat_dim, decay=float(tn_cfg.get("decay", 0.99)),
                kieu=str(tn_cfg.get("kieu", "day_du")),
                truc_json=tn_cfg.get("truc_json"))
        nh_cfg = dict(ngan_hang or {})
        if nh_cfg.get("enabled", False):
            if self.tang_nhanh is None:
                # Chặn cấu hình vô nghĩa: tầng trung LƯU snapshot của tầng nhanh — không có
                # tầng nhanh thì không có gì để lưu/nạp.
                raise ValueError("slda.ngan_hang cần slda.tang_nhanh.enabled=true "
                                 "(tầng trung là bộ nhớ CỦA tầng nhanh)")
            from .ngan_hang_che_do import NganHangCheDo
            self.ngan_hang = NganHangCheDo(
                dim=feat_dim, nguong=float(nh_cfg.get("nguong", 0.5)),
                k_max=int(nh_cfg.get("k_max", 8)),
                cho_khop_sau=int(nh_cfg.get("cho_khop_sau", 5)),
                lam_mode=float(nh_cfg.get("lam_mode", 0.5)))

    def train(self, mode: bool = True):  # noqa: D401
        """Backbone luôn eval (thống kê BatchNorm/LayerNorm không trôi)."""
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def update(self, feats: torch.Tensor, ys: torch.Tensor) -> None:
        """Hấp thụ 1 batch (streaming): cộng dồn s_c, n_c, G. Không giữ lại mẫu.

        BA TẦNG: tầng nhanh nuốt feature THÔ (m_t bám dòng thật), còn thống kê tầng chậm
        nhận feature ĐÃ CĂN CHỈNH — μ_c/Σ sống trong hệ toạ độ e₀. Ở pha 1 căn chỉnh là
        identity (mốc chưa chốt) nên hành vi trùng khít bản cũ.
        """
        if self.tang_nhanh is not None:
            self.tang_nhanh.cap_nhat(feats.detach().float())
            feats = self.tang_nhanh.can_chinh(feats.detach().float())
        f = feats.detach().to(self.feat_sum.dtype)   # ↳ theo stats_dtype (float64 mặc định)
        if not torch.isfinite(f).all():
            raise FloatingPointError("SLDA nhận feature chứa NaN/Inf")
        n_batch = int(ys.numel())

        # --- Σ (gram + cặp đếm toàn cục): phân rã theo ĐỒNG HỒ TOÀN CỤC ------------------
        # Đây là thống kê DÙNG CHUNG cho mọi lớp, nên "tuổi" của nó tính theo tổng số mẫu
        # đã đi qua, không theo lớp nào. Cả BA thứ dưới đây phải phân rã CÙNG hệ số, nếu
        # không thì đẳng thức Σ_w = (G − Σ n_c μ_c μ_cᵀ)/N sai (xem chú thích ở __init__).
        if self.decay_cov < 1.0:
            r = self.decay_cov ** n_batch
            self.gram.mul_(r)
            self.feat_sum_g.mul_(r)
            self.count_g.mul_(r)
        self.gram.add_(f.t() @ f)            # ↳ Σ f fᵀ cộng dồn theo batch (D×D).
        self.feat_sum_g.index_add_(0, ys, f)
        self.count_g.index_add_(0, ys, torch.ones_like(ys, dtype=self.count_g.dtype))

        # --- μ_c (feat_sum, count): phân rã CHỈ Ở LỚP CÓ MẶT trong batch này -------------
        # BẪY quan trọng: nếu phân rã theo đồng hồ toàn cục thì lớp HIẾM GẶP sẽ bị xoá sạch
        # -> mất luôn tính chất "không quên lớp cũ", vốn là điểm mạnh nhất của SLDA.
        # Phân rã theo-lớp giữ được điều đó: μ_c chỉ bám theo những lần c THỰC SỰ xuất hiện.
        if self.decay_mean < 1.0:
            dem = torch.zeros_like(self.count).index_add_(
                0, ys, torch.ones_like(ys, dtype=self.count.dtype))
            he_so = torch.where(dem > 0,
                                torch.as_tensor(self.decay_mean, dtype=self.count.dtype,
                                                device=self.count.device) ** dem,
                                torch.ones_like(self.count))
            self.feat_sum.mul_(he_so.unsqueeze(1))
            self.count.mul_(he_so)

        self.feat_sum.index_add_(0, ys, f)
        self.count.index_add_(0, ys, torch.ones_like(ys, dtype=self.count.dtype))
        self.count_raw.index_add_(0, ys, torch.ones_like(ys, dtype=self.count_raw.dtype))
        self._version += 1

    @torch.no_grad()
    def _within_class_sigma(self) -> torch.Tensor:
        """Σ_w = (G − Σ_c n_c μ_c μ_cᵀ) / N — within-class covariance, chính xác, streaming.

        CHỈ dùng cặp đếm TOÀN CỤC (`feat_sum_g`, `count_g`) — cùng đồng hồ với `gram`.
        Không được dùng `feat_sum`/`count` ở đây: chúng ở đồng hồ theo-lớp, trộn vào là sai
        đẳng thức (xem chú thích dài ở `__init__`). Khi λ=1 hai cặp bằng nhau nên kết quả
        trùng khớp tuyệt đối với bản gốc.
        """
        n = self.count_g
        # clamp rất nhỏ chứ không phải 1.0: có phân rã thì n_c hợp lệ vẫn có thể < 1.
        # Lớp chưa từng thấy có feat_sum_g = 0 nên μ = 0 dù mẫu số bé — không sinh NaN.
        mu = self.feat_sum_g / n.clamp(min=1e-12).unsqueeze(1)                  # (C, D)
        between = self.feat_sum_g.t() @ mu          # ≡ Σ_c n_c μ_c μ_cᵀ, ổn định hơn (D×D)
        sigma = (self.gram - between) / max(float(n.sum()), 1.0)
        return 0.5 * (sigma + sigma.t())                                        # ↳ ép đối xứng (sai số float)

    @torch.no_grad()
    def on_task_end(self) -> None:
        """Gọi sau mỗi task (từ methods.SLDA.end_task). Chỉ có tác dụng ở chế độ `frozen`."""
        self._tasks_done += 1
        if self.cov_mode == "frozen" and self._frozen_sigma.numel() == 0 \
                and self._tasks_done >= self.cov_freeze_after:
            self._frozen_sigma = self._within_class_sigma().clone()
            self._cache_version = -1        # ↳ buộc tính lại cache với Σ đã đóng băng.
            print(f"[slda] ĐÓNG BĂNG Σ sau task {self._tasks_done - 1} "
                  f"(cov_mode=frozen, cov_freeze_after={self.cov_freeze_after})")

    @torch.no_grad()
    def _refresh_cache(self) -> None:
        """Tính Λ, trọng số tuyến tính + bias từ thống kê tích luỹ (lazy, chỉ khi có update mới)."""
        if self._cache_version == self._version:
            return
        d = self.feat_sum.shape[1]
        mu = self.feat_sum / self.count.clamp(min=1.0).unsqueeze(1)            # (C, D)
        eye = torch.eye(d, dtype=mu.dtype, device=mu.device)
        # --- B2: chọn Σ theo cov_mode -------------------------------------------------
        if self.cov_mode == "identity":
            # Σ = I  ->  Λ = I/(1+ε). Không cần nghịch đảo. Tương đương ĐÚNG với NCM.
            lam = eye / (1.0 + self.shrinkage)
        else:
            sigma = (self._frozen_sigma if (self.cov_mode == "frozen"
                                            and self._frozen_sigma.numel() > 0)
                     else self._within_class_sigma())
            lam = torch.linalg.inv(sigma + self.shrinkage * eye)
        w = lam @ mu.t()                                                        # (D, C) = Λ μᵀ
        b = -0.5 * (mu * (mu @ lam)).sum(dim=1)                                 # (C,)
        # class chưa có mẫu: score = 0 mọi nơi -> mask_logits sẽ che, không ảnh hưởng.
        empty = self.count < 1.0
        w[:, empty] = 0.0
        b[empty] = 0.0
        self._cache_w, self._cache_b = w.float(), b.float()                     # ↳ về float32 cho forward.
        self._cache_version = self._version

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x).float()
            if self.tang_nhanh is not None:              # ↳ đưa về hệ toạ độ e₀ trước khi chấm
                feats = self.tang_nhanh.can_chinh(feats)
        self._refresh_cache()
        return feats @ self._cache_w + self._cache_b    # (B, C) discriminant scores — dùng như logits

    # ================== BA TẦNG — pha 2 KHÔNG NHÃN (P1/P2, KE_HOACH_SUA 2026-08-04) =========
    @torch.no_grad()
    def chot_moc_pha1(self) -> None:
        """Cuối pha hiệu chỉnh (chuyến CÓ nhãn cuối cùng): đóng băng mốc (m0, v0) của tầng
        nhanh. Engine gọi đúng một lần. Không có tầng nhanh -> no-op."""
        if self.tang_nhanh is not None:
            self.tang_nhanh.chot_moc()
            print(f"[ba_tang] ĐÃ CHỐT mốc pha 1 (m0/v0) sau {self.tang_nhanh.so_mau} mẫu "
                  f"— căn chỉnh bắt đầu có tác dụng từ chuyến sau")

    @torch.no_grad()
    def hap_thu_khong_nhan(self, loader, device, allowed=None) -> list:
        """Pha 2 — dòng KHÔNG nhãn của bài toán gốc (§2/§3). Trả về chuỗi accuracy
        PREQUENTIAL (test-then-train): mỗi batch DỰ ĐOÁN TRƯỚC bằng trạng thái hiện có,
        RỒI mới cho các tầng không nhãn cập nhật. Đúng nghĩa "trả lời ŷ_t trước khi thấy
        x_{t+1}", và chính chuỗi này nuôi thước đo O2 (thời gian hồi phục).

        LUẬT NHÃN: `y` trong loader CHỈ dùng để CHẤM prequential — cùng nguyên tắc với
        `mode_that`. Không một giá trị y nào chạm vào update (μ_c/Σ/m_t đều không).
        Bằng chứng kiểm được: `count_raw` đứng yên suốt pha 2 (test bắt điều này).
        """
        self.eval()
        accs: list = []
        dem_batch = 0
        m_tuoi_sum = None                         # ↳ trung bình TƯƠI của chuyến này (cho khớp)
        for x, y in loader:
            x = x.to(device)
            raw = self.backbone(x).float()
            # 1) DỰ ĐOÁN TRƯỚC — bằng trạng thái tầng nhanh HIỆN TẠI (chưa thấy batch này)
            f = self.tang_nhanh.can_chinh(raw) if self.tang_nhanh is not None else raw
            self._refresh_cache()
            logits = f @ self._cache_w + self._cache_b
            if allowed is not None:
                logits = mask_logits(logits, allowed)
            pred = logits.argmax(dim=1).cpu()
            accs.append(float((pred == y).float().mean()))
            # 2) RỒI MỚI CẬP NHẬT — chỉ các tầng KHÔNG nhãn, chỉ từ feature thô
            if self.tang_nhanh is not None:
                self.tang_nhanh.cap_nhat(raw)
            dem_batch += 1
            # tầng trung: khớp bằng trung bình TƯƠI của chuyến (KHÔNG dùng m_t — nó còn
            # nhiễm điều kiện chuyến trước, xem chú thích trong khop_va_nap).
            if self.ngan_hang is not None and dem_batch <= self.ngan_hang.cho_khop_sau:
                mb = raw.mean(dim=0)
                m_tuoi_sum = mb if m_tuoi_sum is None else m_tuoi_sum + mb
                if dem_batch == self.ngan_hang.cho_khop_sau:
                    m_tuoi = m_tuoi_sum / float(dem_batch)
                    if self.ngan_hang.khop_va_nap(self.tang_nhanh, m_tuoi=m_tuoi):
                        print(f"[ba_tang] GẶP LẠI chế độ cũ (sau {dem_batch} batch) — "
                              f"nạp snapshot, khỏi học lại (O3)")
        if self.ngan_hang is not None:            # ↳ cuối chuyến: ghi/mài chế độ
            self.ngan_hang.ghi_lai(self.tang_nhanh)
        return accs

    # -------- state ba tầng cho memory_state.pt (run_g1 lưu cuối stream) --------------------
    def export_state(self):
        if self.tang_nhanh is None:
            return None
        st = {"tang_nhanh": self.tang_nhanh.export_state()}
        if self.ngan_hang is not None:
            st["ngan_hang"] = self.ngan_hang.export_state()
        return st

    def import_state(self, st) -> None:
        if st and self.tang_nhanh is not None and "tang_nhanh" in st:
            self.tang_nhanh.import_state(st["tang_nhanh"])
        if st and self.ngan_hang is not None and "ngan_hang" in st:
            self.ngan_hang.import_state(st["ngan_hang"])

    def state_norm(self) -> float:
        if self.tang_nhanh is None or self.tang_nhanh.m_t is None:
            return 0.0
        return float(self.tang_nhanh.m_t.norm())

    def extra_floats(self) -> int:
        return int(self.feat_sum.numel() + self.count.numel() + self.gram.numel()
                   + self.feat_sum_g.numel() + self.count_g.numel())

    def memory_report(self) -> dict:
        """Chi phí bộ nhớ THẬT theo byte (B6) — `extra_floats` không phản ánh dtype.

        Ba bộ đếm là float64 (lý do ở docstring đầu file), cache là float32.
        Chi phí do `gram` chi phối và là **O(D²)**, không phải O(C·D):
        ViT-S D=384 -> 1,1 MB · ViT-L D=1024 -> 8 MB.
        """
        def _b(t):
            return 0 if t is None else t.numel() * t.element_size()
        parts = {
            "gram_DxD": _b(self.gram),
            "feat_sum_CxD": _b(self.feat_sum),
            "count_C": _b(self.count),
            "feat_sum_g_CxD": _b(self.feat_sum_g),   # ↳ cặp đếm đồng hồ toàn cục (cho Σ)
            "count_g_C": _b(self.count_g),
            "frozen_sigma": _b(self._frozen_sigma),
            "cache_w_b": _b(self._cache_w) + _b(self._cache_b),
        }
        # BA TẦNG (O4): cộng chi phí tầng nhanh/trung vào cùng bảng — đối chiếu ràng buộc
        # "tổng bộ nhớ thêm ≤ 10 MB" ngay trong log của mọi run.
        if self.tang_nhanh is not None:
            parts["tang_nhanh"] = self.tang_nhanh.extra_bytes()
        if self.ngan_hang is not None:
            parts["ngan_hang"] = self.ngan_hang.extra_bytes()
        parts["total_bytes"] = sum(parts.values())
        parts["total_MB"] = parts["total_bytes"] / 1e6
        parts["cov_mode"] = self.cov_mode
        parts["stats_dtype"] = self.stats_dtype
        return parts

    @torch.no_grad()
    def window_report(self) -> dict:
        """Cửa sổ nhớ hiệu dụng — bằng chứng trực tiếp λ đang chạy đúng (D2).

        Với λ < 1, `n_c` của một lớp gặp thường xuyên HỘI TỤ về `1/(1−λ)`. So số đo với
        kỳ vọng lý thuyết là cách rẻ nhất để bắt lỗi cài đặt: lệch nhiều = sai công thức
        hoặc phân rã nhầm chỗ.
        """
        seen = self.count[self.count > 0]
        raw = self.count_raw[self.count_raw > 0]
        m = float(raw.mean()) if raw.numel() else 0.0        # số lần gặp trung bình mỗi lớp
        lam = self.decay_mean
        # Mốc ĐÚNG: tổng cấp số nhân CÓ HẠN sau m lần gặp. Tiệm cận 1/(1−λ) chỉ là giới hạn
        # khi m -> ∞; dùng nó làm mốc khi m nhỏ sẽ báo động giả (xem chú thích ở count_raw).
        if lam < 1.0:
            ky_vong_huu_han = (1.0 - lam ** m) / (1.0 - lam) if m > 0 else 0.0
        else:
            ky_vong_huu_han = m
        return {
            "decay_mean": self.decay_mean,
            "decay_cov": self.decay_cov,
            "n_c_trung_binh": float(seen.mean()) if seen.numel() else 0.0,
            "n_c_max": float(seen.max()) if seen.numel() else 0.0,
            "ky_vong": (1.0 / (1.0 - lam)) if lam < 1.0 else float("inf"),   # tiệm cận
            "ky_vong_huu_han": ky_vong_huu_han,                              # mốc để so
            "so_lan_gap_tb": m,
            "ty_le_giu": (ky_vong_huu_han / m) if m > 0 else 0.0,   # so với λ=1 -> ~1 là gần như không quên
            "so_lop_da_thay": int(seen.numel()),
        }
