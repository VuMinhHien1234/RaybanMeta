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
# ↳ GIẢI THÍCH TỔNG QUAN: HOPE = "hợp nhất" 2 ý tưởng. Titans (G2) cho model một bộ
#   NHỚ NHANH; CMS (G3) cho backbone khả năng học ĐA TẦN SỐ (nhanh/chậm). HOPE dùng
#   cả hai cùng lúc -> đúng cấu trúc "nhiều tầng ký ức" của paper Nested Learning §8.
#   Về code, HOPE KẾ THỪA gần hết từ TitansClassifier, chỉ khác đúng 2 điểm ghi ở trên.
# ↳ Lưu ý rủi ro: backbone giờ được học -> feature "trôi" theo thời gian, memory phải
#   theo kịp. Đây chính là bệnh đã đo được (Forgetting 0.956) mà 4 cờ ổn định đang chữa.
from __future__ import annotations

import torch
import torch.nn as nn

from .titans_head import TitansClassifier  # ↳ Kế thừa toàn bộ cơ chế memory/state của G2.


class HOPEClassifier(TitansClassifier):  # ↳ "(TitansClassifier)" = thừa hưởng mọi thứ của lớp cha.
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int, memory_cfg: dict,
                 head: str = "linear"):
        super().__init__(backbone, feat_dim, num_classes, memory_cfg, head=head)  # ↳ Dựng y hệt G2 trước (kể cả đóng băng backbone).
        # MỞ BĂNG backbone (TitansClassifier vừa đóng) — CMS sẽ đóng lại phần nền
        # (patch_embed/pos_embed/...) và điều tiết phần còn lại theo tier.
        for p in self.backbone.parameters():
            p.requires_grad_(True)  # ↳ KHÁC BIỆT #1: bật lại gradient cho backbone để nó được học (qua CMS).

    def train(self, mode: bool = True):
        # KHÔNG ép backbone.eval như Titans — backbone đang được train (ViT không có
        # BatchNorm nên không lo thống kê batch trôi).
        return nn.Module.train(self, mode)  # ↳ Gọi thẳng train gốc, KHÔNG ép backbone về eval như G2.

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Như TitansClassifier nhưng CÓ gradient (backbone được CMS update)."""
        # ↳ KHÁC BIỆT #2: bản G2 bọc @torch.no_grad() (backbone đóng băng); ở đây KHÔNG bọc
        #   để gradient chảy về backbone. Phần logic trích feature thì giống hệt.
        if self.seq_mode == "token_seq":
            ff = getattr(self.backbone, "forward_features", None)
            if ff is None:
                raise ValueError("token_seq cần backbone timm có forward_features (ViT).")
            feats = ff(x)
            n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 0))
            if feats.dim() == 3 and n_prefix > 0:
                feats = feats[:, n_prefix:, :]  # ↳ Bỏ token đặc biệt (cls/reg), giữ patch.
            return feats
        return self.backbone(x)  # ↳ image_seq: feature đã gộp (B, D), có gradient.
