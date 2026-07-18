"""NCM — Nearest Class Mean trên backbone ĐÓNG BĂNG (baseline thứ 3 của G1).

Vì sao phải có (bài học lấy từ project Meta-Rayban/CPM): trên đặc trưng của
backbone pretrained đóng băng, mỗi class chỉ cần một *prototype trung bình*;
prototype của class cũ KHÔNG BAO GIỜ bị ghi đè khi học class mới -> gần như
không quên, không cần gradient, bộ nhớ bị chặn O(C·D). Nếu Titans (G2) / CMS
(G3) không thắng nổi baseline "ngây thơ" này thì chưa có gì để báo cáo.

Giao diện giữ đúng quy ước dự án: forward(x) -> "logits" (B, num_classes)
= cosine(feature, prototype), nên engine/mask/metrics dùng chung không đổi.
"""
# ↳ GIẢI THÍCH TỔNG QUAN: NCM là baseline "ngây thơ mà mạnh". Ý tưởng: mỗi class chỉ
#   giữ 1 vector trung bình (prototype) của các ảnh thuộc class đó. Phân loại = xem
#   feature ảnh mới GẦN prototype nào nhất (đo bằng cosine). Vì prototype class cũ
#   không bị đụng khi thêm class mới -> gần như KHÔNG QUÊN. Đây là "vạch chuẩn" mà
#   các phương pháp phức tạp (Titans/CMS/HOPE) phải vượt qua mới có ý nghĩa.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # ↳ Chứa các hàm tiện ích như normalize.


class NCMClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # NCM định nghĩa trên đặc trưng cố định -> luôn đóng băng backbone.
        for p in self.backbone.parameters():
            p.requires_grad_(False)  # ↳ Không học backbone: NCM chỉ dựa trên feature có sẵn.
        self.backbone.eval()
        # buffer (không phải Parameter): không có gradient, đi theo state_dict/device
        # ↳ "buffer" = tensor được lưu/chuyển device cùng model nhưng KHÔNG được optimizer cập nhật.
        self.register_buffer("proto_sum", torch.zeros(num_classes, feat_dim))  # ↳ Tổng feature cộng dồn theo class.
        self.register_buffer("proto_count", torch.zeros(num_classes))          # ↳ Đếm số ảnh mỗi class (để chia trung bình).

    def train(self, mode: bool = True):  # noqa: D401
        """Giữ backbone ở eval kể cả khi .train() (BatchNorm không được cập nhật)."""
        super().train(mode)
        self.backbone.eval()  # ↳ Ép backbone luôn eval để thống kê không trôi.
        return self

    @torch.no_grad()  # ↳ Cập nhật prototype không cần gradient.
    def update_prototypes(self, feats: torch.Tensor, ys: torch.Tensor) -> None:
        """Cộng dồn tổng đặc trưng theo class (trung bình online, bounded)."""
        # ↳ "Học" của NCM chỉ là cộng dồn: với mỗi ảnh, cộng feature vào đúng class của nó.
        feats = F.normalize(feats.detach().float(), dim=1)  # ↳ Chuẩn hoá về vector đơn vị (để cosine chuẩn).
        self.proto_sum.index_add_(0, ys, feats)             # ↳ Cộng feature vào hàng class tương ứng (theo nhãn ys).
        self.proto_count.index_add_(0, ys, torch.ones_like(ys, dtype=torch.float))  # ↳ Tăng bộ đếm class lên 1 mỗi ảnh.

    def prototypes(self) -> torch.Tensor:
        # ↳ Tính prototype = trung bình = tổng / số đếm, rồi chuẩn hoá.
        cnt = self.proto_count.clamp(min=1.0).unsqueeze(1)  # ↳ clamp min=1 để tránh chia 0 (class chưa học).
        return F.normalize(self.proto_sum / cnt, dim=1)  # class chưa học -> vector 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(self.backbone(x).float(), dim=1)  # ↳ Feature ảnh -> vector đơn vị.
        return feats @ self.prototypes().t()  # cosine scores (B, C) — dùng như logits
        # ↳ Nhân ma trận feature (B,D) với prototype^T (D,C) -> điểm cosine (B,C), dùng y như logits.

    def extra_floats(self) -> int:
        """Bộ nhớ thêm ngoài backbone (để so footprint giữa các method)."""
        return int(self.proto_sum.numel() + self.proto_count.numel())  # ↳ Kích thước 2 buffer prototype.
