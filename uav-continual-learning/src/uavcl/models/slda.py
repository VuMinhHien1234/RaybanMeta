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
                 cov_freeze_after: int = 1, stats_dtype: str = "float64"):
        super().__init__()
        if shrinkage <= 0.0:
            raise ValueError(f"shrinkage phải > 0 (nhận {shrinkage})")
        if cov_mode not in COV_MODES:
            raise ValueError(f"cov_mode phải thuộc {COV_MODES} (nhận {cov_mode!r})")
        if cov_freeze_after < 1:
            raise ValueError("cov_freeze_after phải >= 1 (đếm theo SỐ TASK đã học xong)")
        if stats_dtype not in ("float64", "float32"):
            raise ValueError(f"stats_dtype phải là 'float64' hoặc 'float32' (nhận {stats_dtype!r})")
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)          # ↳ SLDA định nghĩa trên feature CỐ ĐỊNH.
        self.backbone.eval()
        self.shrinkage = float(shrinkage)
        self.cov_mode = str(cov_mode)
        self.cov_freeze_after = int(cov_freeze_after)
        self.stats_dtype = str(stats_dtype)
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
        self._cache_w = None       # (D, C) = Λ μ_cᵀ — cache để không nghịch đảo mỗi batch
        self._cache_b = None       # (C,)   = −½ μ_c Λ μ_c
        self._cache_version = -1   # bump theo _version mỗi lần update
        self._version = 0

    def train(self, mode: bool = True):  # noqa: D401
        """Backbone luôn eval (thống kê BatchNorm/LayerNorm không trôi)."""
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def update(self, feats: torch.Tensor, ys: torch.Tensor) -> None:
        """Hấp thụ 1 batch (streaming): cộng dồn s_c, n_c, G. Không giữ lại mẫu."""
        f = feats.detach().to(self.feat_sum.dtype)   # ↳ theo stats_dtype (float64 mặc định)
        if not torch.isfinite(f).all():
            raise FloatingPointError("SLDA nhận feature chứa NaN/Inf")
        self.feat_sum.index_add_(0, ys, f)
        self.count.index_add_(0, ys, torch.ones_like(ys, dtype=self.count.dtype))
        self.gram.add_(f.t() @ f)            # ↳ Σ f fᵀ cộng dồn theo batch (D×D).
        self._version += 1

    @torch.no_grad()
    def _within_class_sigma(self) -> torch.Tensor:
        """Σ_w = (G − Σ_c n_c μ_c μ_cᵀ) / N — within-class covariance, chính xác, streaming."""
        mu = self.feat_sum / self.count.clamp(min=1.0).unsqueeze(1)            # (C, D)
        between = (self.count.unsqueeze(1) * mu).t() @ mu                       # Σ n_c μ μᵀ (D×D)
        sigma = (self.gram - between) / max(float(self.count.sum()), 1.0)
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
        self._refresh_cache()
        return feats @ self._cache_w + self._cache_b    # (B, C) discriminant scores — dùng như logits

    def extra_floats(self) -> int:
        return int(self.feat_sum.numel() + self.count.numel() + self.gram.numel())

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
            "frozen_sigma": _b(self._frozen_sigma),
            "cache_w_b": _b(self._cache_w) + _b(self._cache_b),
        }
        parts["total_bytes"] = sum(parts.values())
        parts["total_MB"] = parts["total_bytes"] / 1e6
        parts["cov_mode"] = self.cov_mode
        parts["stats_dtype"] = self.stats_dtype
        return parts
