"""HOPEClassifier (G4, task S1) — ghép Titans (nhanh) + CMS (bền) đúng tinh thần §8 paper.

Dây chuyền:   ảnh → backbone ViT (MỞ BĂNG — nhịp update do CMSOptimizer kiểm soát,
              tầng TRUNG + CHẬM) → SeqAdapter → TitansMemory xuyên task (tầng NHANH
              NHẤT, tự học online trong forward) → post_norm + residual → head.

Khác TitansClassifier (G2) đúng 2 điểm:
1. Backbone KHÔNG đóng băng — gradient chảy về backbone, nhưng CMSOptimizer quyết
   block nào được bước, bao lâu một lần, mạnh nhẹ ra sao (models/cms.py).
2. Vì backbone thay đổi, không gian feature TRÔI dưới chân memory — rủi ro đặc thù
   của HOPE (xem TASKS_G4_SOLO): theo dõi bằng log norm(state) của method `hope`.

Toàn bộ vòng đời state (reset image/task/never, detach mỗi batch, clone khi eval)
kế thừa nguyên từ TitansClassifier — đã test ở G2.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .titans_head import TitansClassifier


class HOPEClassifier(TitansClassifier):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int, memory_cfg: dict):
        super().__init__(backbone, feat_dim, num_classes, memory_cfg)
        # MỞ BĂNG backbone (TitansClassifier vừa đóng) — CMS sẽ đóng lại phần nền
        # (patch_embed/pos_embed/...) và điều tiết phần còn lại theo tier.
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def train(self, mode: bool = True):
        # KHÔNG ép backbone.eval như Titans — backbone đang được train (ViT không có
        # BatchNorm nên không lo thống kê batch trôi).
        return nn.Module.train(self, mode)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Như TitansClassifier nhưng CÓ gradient (backbone được CMS update)."""
        if self.seq_mode == "token_seq":
            ff = getattr(self.backbone, "forward_features", None)
            if ff is None:
                raise ValueError("token_seq cần backbone timm có forward_features (ViT).")
            feats = ff(x)
            n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 0))
            if feats.dim() == 3 and n_prefix > 0:
                feats = feats[:, n_prefix:, :]
            return feats
        return self.backbone(x)
