"""Models: backbone, classifier, (sau này) Titans memory, CMS, HOPE.

- backbone.py   (G1, N3):    pretrained vision encoder -> feature vector. ✓
- classifier.py (G1, N3):    head class-incremental + mask logits.         ✓
- memory.py     (G2, N2):    wrapper quanh titans_pytorch.NeuralMemory.
- cms.py        (G3, N2+N3): khối MLP đa tần số retrofit lên backbone.
- hope.py       (G4, N2):    ghép Titans + CMS.
"""
from .backbone import TinyCNN, build_backbone
from .classifier import MASK_FILL, ContinualClassifier, mask_logits
from .ncm import NCMClassifier

__all__ = ["TinyCNN", "build_backbone", "ContinualClassifier", "NCMClassifier", "mask_logits", "MASK_FILL"]
