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
from .backbone import TinyCNN, build_backbone
from .classifier import MASK_FILL, ContinualClassifier, mask_logits
from .hope import HOPEClassifier
from .ncm import NCMClassifier
from .seq_adapter import SeqAdapter
from .titans_head import TitansClassifier

__all__ = [
    "TinyCNN",
    "build_backbone",
    "ContinualClassifier",
    "NCMClassifier",
    "SeqAdapter",
    "TitansClassifier",
    "HOPEClassifier",
    "mask_logits",
    "MASK_FILL",
]
