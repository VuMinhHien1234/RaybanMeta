"""Backbone thị giác (owner: N3) — ảnh -> vector đặc trưng (B, feat_dim).

Hai chế độ:
- Tên model timm (vd 'vit_small_patch16_224', 'resnet18'): dùng pretrained,
  num_classes=0 để lấy feature. Đây là "trí nhớ dài hạn" sẽ bị CMS retrofit ở G3.
- 'tinycnn': CNN bé xíu không cần mạng/không pretrained — cho smoke test & CI.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """3 khối conv -> GAP. Chỉ để kiểm tra pipeline chạy, không để lấy số đẹp."""

    feat_dim = 64

    def __init__(self):
        super().__init__()
        chans = [3, 16, 32, self.feat_dim]
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks += [
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)


def build_backbone(backbone_cfg: dict) -> tuple[nn.Module, int]:
    """Trả về (module, feat_dim). module(x) -> (B, feat_dim)."""
    name = str(backbone_cfg.get("name", "vit_small_patch16_224"))
    if name.lower() == "tinycnn":
        m = TinyCNN()
        feat_dim = m.feat_dim
    else:
        import timm

        m = timm.create_model(
            name,
            pretrained=bool(backbone_cfg.get("pretrained", True)),
            num_classes=0,  # bỏ head phân loại của timm, chỉ lấy feature
        )
        feat_dim = int(m.num_features)

    if bool(backbone_cfg.get("freeze", False)):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
    return m, feat_dim
