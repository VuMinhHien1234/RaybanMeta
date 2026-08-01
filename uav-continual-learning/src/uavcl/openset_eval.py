"""#27 — Chấm điểm open-set bằng khoảng cách prototype (plans/TASKS_UAV_CL.md).

UAV gặp class CHƯA HỌC phải biết "không biết". Cách đo (không thêm model nào):
1. Dựng prototype mỗi class ĐÃ HỌC từ feature train (rebuild post-hoc — chỉ để chẩn đoán).
2. Điểm tin cậy của 1 ảnh = max cosine(feature, prototype). Ảnh quen -> gần prototype
   của nó -> điểm CAO; ảnh lạ -> xa mọi prototype -> điểm THẤP.
3. genuine = điểm trên test của class đã học; impostor = test của class GIỮ LẠI
   (build_stream_with_holdout). Đưa vào metrics.openset.open_set_summary -> AUC/EER/TAR@FAR.

Feature lấy qua model.features(x) (sau memory — Titans/HOPE) nếu có, không thì
model.backbone(x). Prototype head (NCM/cosine) kỳ vọng thắng head Linear ở đây:
logit Linear không phải thước đo khoảng cách nên khó tách quen/lạ.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


def _feat_fn(model):
    if hasattr(model, "features"):
        return model.features            # ↳ Titans/HOPE: feature SAU memory.
    if hasattr(model, "backbone"):
        return model.backbone            # ↳ G1/NCM/SLDA: feature backbone.
    raise TypeError("openset_eval cần model có .features() hoặc .backbone")


@torch.no_grad()
def build_prototypes(model, task_loaders: List[Dict], task_ids: Sequence[int],
                     num_classes: int, device) -> torch.Tensor:
    """Prototype = trung bình (chuẩn hoá) feature train theo class, gộp các task đã học."""
    model.eval()
    fn = _feat_fn(model)
    psum, pcnt = None, torch.zeros(num_classes, device=device)
    for tid in task_ids:
        for x, y in task_loaders[tid]["train"]:
            x, y = x.to(device), y.to(device)
            h = F.normalize(fn(x).float(), dim=1)
            if psum is None:
                psum = torch.zeros(num_classes, h.shape[1], device=device)
            psum.index_add_(0, y, h)
            pcnt.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
    if psum is None:
        raise ValueError("không có dữ liệu train nào để dựng prototype")
    return F.normalize(psum / pcnt.clamp(min=1.0).unsqueeze(1), dim=1)  # class chưa học -> vector 0


@torch.no_grad()
def _scores(model, loader, protos: torch.Tensor, device) -> List[float]:
    """Điểm tin cậy mỗi ảnh = max cosine tới prototype (chỉ hàng có norm > 0)."""
    model.eval()
    fn = _feat_fn(model)
    live = protos.norm(dim=1) > 1e-6            # ↳ Bỏ hàng prototype rỗng (class chưa học).
    P = protos[live]
    out: List[float] = []
    for x, _ in loader:
        h = F.normalize(fn(x.to(device)).float(), dim=1)
        out.extend((h @ P.t()).max(dim=1).values.cpu().tolist())
    return out


@torch.no_grad()
def collect_openset_scores(model, task_loaders: List[Dict], unseen_loader, num_classes: int,
                           device) -> tuple[List[float], List[float]]:
    """Trả (genuine, impostor): điểm trên test ĐÃ HỌC (mọi task) và test class GIỮ LẠI."""
    protos = build_prototypes(model, task_loaders, list(range(len(task_loaders))), num_classes, device)
    genuine: List[float] = []
    for tl in task_loaders:
        genuine.extend(_scores(model, tl["test"], protos, device))
    impostor = _scores(model, unseen_loader, protos, device)
    if not genuine or not impostor:
        raise ValueError(f"thiếu mẫu chấm điểm: genuine={len(genuine)}, impostor={len(impostor)}")
    return genuine, impostor
