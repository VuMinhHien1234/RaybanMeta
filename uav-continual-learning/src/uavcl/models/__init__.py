"""Models: backbone, classifier, Titans memory (G2), CMS (G3), HOPE (G4).

- backbone.py    (G1) ✓: pretrained vision encoder -> feature vector.
- classifier.py  (G1) ✓: head class-incremental + mask logits.
- ncm.py         (G1) ✓: baseline NCM frozen + prototype.
- memory.py      (G2) ✓: TitansMemory — wrapper NeuralMemory, API (seq, state).
- seq_adapter.py (G2) ✓: ảnh -> chuỗi (image_seq | token_seq).
- titans_head.py (G2) ✓: TitansClassifier — frozen ViT + memory + head, 3 chế độ reset.
- state_utils.py (G2) ✓: detach/clone/norm/save state.
- cms.py         (G3):   khối MLP đa tần số retrofit lên backbone.
- hope.py        (G4):   ghép Titans + CMS.
"""
# ↳ GIẢI THÍCH TỔNG QUAN: __init__ của package `models` — vừa là "mục lục" (docstring
#   liệt kê từng file làm gì theo mốc G1..G4), vừa gom các class model ra ngoài để
#   import gọn: `from uavcl.models import HOPEClassifier`.
from .backbone import TinyCNN, build_backbone
from .classifier import MASK_FILL, ContinualClassifier, CosineHead, build_head, mask_logits
from .hope import HOPEClassifier
from .ncm import NCMClassifier
from .seq_adapter import SeqAdapter
from .titans_head import TitansClassifier

__all__ = [
    # ↳ Danh sách tên công khai của package models.
    "TinyCNN",             # ↳ CNN tí hon cho smoke test.
    "build_backbone",      # ↳ Nhà máy dựng backbone.
    "ContinualClassifier", # ↳ Baseline backbone + head (G1).
    "CosineHead",          # ↳ Đầu cosine chống recency bias (fix nút thắt head).
    "build_head",          # ↳ Nhà máy chọn head linear|cosine.
    "NCMClassifier",       # ↳ Baseline NCM prototype (G1).
    "SeqAdapter",          # ↳ Phiên dịch ảnh <-> chuỗi (G2).
    "TitansClassifier",    # ↳ Model có bộ nhớ Titans (G2).
    "HOPEClassifier",      # ↳ Titans + CMS hợp nhất (G4).
    "mask_logits",         # ↳ Hàm che class không được phép.
    "MASK_FILL",           # ↳ Giá trị dùng để che.
]
