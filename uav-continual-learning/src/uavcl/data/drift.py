"""Trôi điều kiện quan sát (domain drift) — mô phỏng drone bay dài ngày.

Vì sao cần: mọi kết quả của dự án tới nay đo ở regime "lớp mới, điều kiện TĨNH". Nhưng câu
hỏi ứng dụng thật là *"drone bay hàng tháng, nắng/mùa/sương đổi dần — model có bám theo
được không?"*. Regime đó chưa ai đo, và nó là chỗ SLDA(λ=1) được dự đoán sẽ hỏng.

Năm phép biến đổi, mỗi phép mô phỏng một hiện tượng vật lý có thật:

    độ sáng      giờ trong ngày, mùa                ×1,00 -> ×0,55
    nhiệt độ màu bình minh ấm -> trưa lạnh          0 -> ±18% lệch kênh R/B
    tương phản   sương mù, khói mù, bụi              ×1,00 -> ×0,65
    mờ Gauss     độ cao bay, ống kính bẩn/ướt        σ 0 -> 1,4 px
    nhiễu        ISO cao khi thiếu sáng              σ 0 -> 0,035

NGUYÊN TẮC QUAN TRỌNG NHẤT — trôi phải TẤT ĐỊNH, không phải nhiễu ngẫu nhiên:

  * **Mức trung bình** dịch theo task một cách xác định  <- đây MỚI là "trôi"
  * Cộng **rung nhẹ** ±jitter quanh mức đó                <- mỗi chuyến bay hơi khác nhau

Nếu để hoàn toàn ngẫu nhiên thì nó chỉ là augmentation thông thường, phân bố không dịch đi,
và λ sẽ hoàn toàn vô dụng — thí nghiệm mất ý nghĩa. Đây là bẫy thiết kế chính của file này.

Áp SAU khi ảnh thành tensor [0,1] và TRƯỚC khi Normalize, để các phép biến đổi mang đúng
ý nghĩa vật lý (độ sáng nhân trên thang [0,1], không phải trên thang đã chuẩn hoá).
"""
from __future__ import annotations

import torch
from torchvision import transforms as T

from .loaders import IMAGENET_MEAN, IMAGENET_STD

# Cường độ TỐI ĐA của từng phép (tại mức trôi 100%, severity=1.0).
# Hiệu chỉnh qua `scripts/calibrate_drift.py` — xem D8.
MAX_DO_SANG = 0.45      # 1 - 0.45 = ×0.55
MAX_AM_MAU = 0.18       # lệch kênh R/B
MAX_TUONG_PHAN = 0.35   # 1 - 0.35 = ×0.65
MAX_MO_SIGMA = 1.4      # px
MAX_NHIEU_SIGMA = 0.035


class ApDungTroi:
    """Áp trôi lên tensor ảnh [0,1] (C,H,W). Tất định theo `muc`, cộng rung ngẫu nhiên."""

    def __init__(self, muc: float, jitter: float = 0.15, seed: int = 0, sinh_ngau: bool = True):
        self.muc = float(max(0.0, min(1.0, muc)))   # ↳ 0 = ảnh gốc, 1 = trôi tối đa
        self.jitter = float(jitter)
        self.sinh_ngau = bool(sinh_ngau)
        self._g = torch.Generator().manual_seed(int(seed))

    def _he_so(self, mac_dinh: float) -> float:
        """Cường độ thực = mức trung bình × (1 ± jitter)."""
        if not self.sinh_ngau or self.jitter <= 0:
            return mac_dinh
        r = float(torch.rand(1, generator=self._g)) * 2.0 - 1.0     # ↳ [-1, 1]
        return mac_dinh * (1.0 + self.jitter * r)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.muc <= 0.0:
            return x                                    # ↳ mức 0 = không đụng gì
        m = self.muc

        # 1) độ sáng — nhân toàn ảnh
        x = x * (1.0 - self._he_so(MAX_DO_SANG * m))

        # 2) nhiệt độ màu — đẩy R lên, B xuống (ấm) hoặc ngược lại; chọn theo mức để TẤT ĐỊNH
        am = self._he_so(MAX_AM_MAU * m)
        if x.shape[0] >= 3:
            x = x.clone()
            x[0] = x[0] * (1.0 + am)                    # ↳ kênh R
            x[2] = x[2] * (1.0 - am)                    # ↳ kênh B

        # 3) tương phản — kéo về giá trị trung bình của ảnh
        tp = 1.0 - self._he_so(MAX_TUONG_PHAN * m)
        x = (x - x.mean()) * tp + x.mean()

        # 4) mờ Gauss — kernel lẻ, tỉ lệ theo sigma
        sig = self._he_so(MAX_MO_SIGMA * m)
        if sig > 0.05:
            k = max(3, int(2 * round(1.5 * sig) + 1))
            x = T.functional.gaussian_blur(x, kernel_size=[k, k], sigma=[sig, sig])

        # 5) nhiễu cảm biến
        ns = self._he_so(MAX_NHIEU_SIGMA * m)
        if ns > 1e-4:
            x = x + torch.randn(x.shape, generator=self._g) * ns

        return x.clamp(0.0, 1.0)


def muc_troi(task_idx: int, num_tasks: int, mode: str = "linear", severity: float = 1.0) -> float:
    """Mức trôi của task thứ `task_idx`, trong [0, severity].

    linear: tăng đều 0 -> 1 qua các task (giống mùa chuyển dần) — MẶC ĐỊNH, thực tế hơn
    step  : nhảy bậc ở giữa stream (0 nửa đầu, 1 nửa sau) — để so, ít thực tế hơn
    """
    if num_tasks <= 1:
        return 0.0
    if mode == "step":
        return float(severity) if task_idx >= num_tasks // 2 else 0.0
    if mode != "linear":
        raise ValueError(f"drift.mode phải là 'linear' hoặc 'step' (nhận {mode!r})")
    return float(severity) * task_idx / (num_tasks - 1)


def build_drift_transform(image_size: int, task_idx: int, num_tasks: int,
                          drift_cfg: dict, train: bool, seed: int = 0):
    """Transform của MỘT task, đã gồm trôi ở đúng mức của task đó.

    Thứ tự có chủ đích:  hình học -> ToTensor -> **TRÔI** -> Normalize
    Trôi phải nằm trên thang [0,1] mới đúng nghĩa vật lý.
    """
    mode = str(drift_cfg.get("mode", "linear"))
    severity = float(drift_cfg.get("severity", 1.0))
    jitter = float(drift_cfg.get("jitter", 0.15))
    m = muc_troi(task_idx, num_tasks, mode, severity)

    # Seed riêng cho từng (task, train/eval) -> tái lập được, và train/eval không trùng nhiễu.
    seed_t = int(seed) * 1000 + task_idx * 2 + (1 if train else 0)

    if train:
        hinh_hoc = [T.RandomResizedCrop(image_size, scale=(0.6, 1.0)), T.RandomHorizontalFlip()]
    else:
        hinh_hoc = [T.Resize((image_size, image_size))]

    return T.Compose(
        hinh_hoc
        + [T.ToTensor()]
        + ([ApDungTroi(m, jitter=jitter, seed=seed_t)] if m > 0 else [])
        + [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def bang_muc_troi(num_tasks: int, drift_cfg: dict) -> str:
    """Chuỗi mô tả mức trôi từng task — in ra lúc dựng loader để có bằng chứng trong log."""
    mode = str(drift_cfg.get("mode", "linear"))
    sev = float(drift_cfg.get("severity", 1.0))
    muc = [muc_troi(t, num_tasks, mode, sev) for t in range(num_tasks)]
    return "  ".join(f"t{t}:{v * 100:.0f}%" for t, v in enumerate(muc))
