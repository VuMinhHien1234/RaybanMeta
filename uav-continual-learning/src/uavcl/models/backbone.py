"""Backbone thị giác (owner: N3) — ảnh -> vector đặc trưng (B, feat_dim).

Hai chế độ:
- Tên model timm (vd 'vit_small_patch16_224', 'resnet18'): dùng pretrained,
  num_classes=0 để lấy feature. Đây là "trí nhớ dài hạn" sẽ bị CMS retrofit ở G3.
- 'tinycnn': CNN bé xíu không cần mạng/không pretrained — cho smoke test & CI.
"""
# ↳ GIẢI THÍCH TỔNG QUAN: "backbone" = con mắt của model. Nó nhận ảnh và nén thành
#   1 vector số (feature) mô tả nội dung ảnh. Vector này là đầu vào cho head phân
#   loại (G1), cho bộ nhớ Titans (G2), v.v. B = batch (số ảnh), feat_dim = độ dài vector.
from __future__ import annotations

import torch
import torch.nn as nn  # ↳ nn = "Lego" của PyTorch (các lớp Conv, Linear... để lắp mạng).


class TinyCNN(nn.Module):
    """3 khối conv -> GAP. Chỉ để kiểm tra pipeline chạy, không để lấy số đẹp."""
    # ↳ Mạng CNN tí hon, không cần tải pretrained -> chạy nhanh để test đường ống thông.

    feat_dim = 64  # ↳ Vector đặc trưng ra dài 64 (các phần khác đọc thuộc tính này để biết kích cỡ).

    def __init__(self):
        super().__init__()  # ↳ Bắt buộc gọi khởi tạo của lớp cha nn.Module.
        chans = [3, 16, 32, self.feat_dim]  # ↳ Số kênh qua từng tầng: 3 (RGB) -> 16 -> 32 -> 64.
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):  # ↳ Ghép cặp (vào, ra): (3,16), (16,32), (32,64).
            blocks += [
                nn.Conv2d(cin, cout, 3, padding=1),  # ↳ Tích chập 3x3 học đặc trưng cục bộ, giữ nguyên kích thước.
                nn.BatchNorm2d(cout),                # ↳ Chuẩn hoá theo batch -> train ổn định hơn.
                nn.ReLU(inplace=True),               # ↳ Phi tuyến: cắt phần âm về 0 (inplace: tiết kiệm RAM).
                nn.MaxPool2d(2),                     # ↳ Giảm 1 nửa chiều cao/rộng -> gom thông tin, bớt tính toán.
            ]
        self.features = nn.Sequential(*blocks)       # ↳ Xâu 3 khối conv thành 1 chuỗi chạy tuần tự.
        self.pool = nn.AdaptiveAvgPool2d(1)          # ↳ Gộp toàn bản đồ đặc trưng về 1x1 (Global Average Pooling).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ↳ x: (B, 3, H, W) -> features -> pool về (B, 64, 1, 1) -> flatten -> (B, 64).
        return self.pool(self.features(x)).flatten(1)  # ↳ flatten(1): ép mọi chiều sau chiều batch thành 1 vector.


def build_backbone(backbone_cfg: dict) -> tuple[nn.Module, int]:
    """Trả về (module, feat_dim). module(x) -> (B, feat_dim)."""
    # ↳ "Nhà máy" dựng backbone theo config: chọn TinyCNN hay model timm, có đóng băng không.
    name = str(backbone_cfg.get("name", "vit_small_patch16_224"))  # ↳ Tên model (mặc định ViT-Small 224px).
    if name.lower() == "tinycnn":
        m = TinyCNN()               # ↳ Nhánh test: dùng mạng tí hon.
        feat_dim = m.feat_dim
    else:
        import timm                 # ↳ timm = kho model thị giác pretrained (import khi cần).

        m = timm.create_model(
            name,
            pretrained=bool(backbone_cfg.get("pretrained", True)),  # ↳ Nạp trọng số học sẵn từ ImageNet.
            num_classes=0,  # bỏ head phân loại của timm, chỉ lấy feature  ↳ 0 = không gắn head, trả feature.
        )
        feat_dim = int(m.num_features)  # ↳ timm cho biết độ dài vector đặc trưng của model này.

    if bool(backbone_cfg.get("freeze", False)):  # ↳ Nếu config yêu cầu đóng băng backbone...
        for p in m.parameters():
            p.requires_grad_(False)              # ↳ ...tắt gradient cho mọi tham số (không cập nhật khi train).
        m.eval()                                 # ↳ Đặt chế độ eval (tắt dropout/không cập nhật BatchNorm).
    return m, feat_dim                           # ↳ Trả (mạng, độ dài feature) cho phần lắp ráp model.
