"""Các "method" học liên tục của G1 (baseline). Giao diện tối giản:

    method.penalty(model)                  -> tensor | None  (cộng vào loss)
    method.end_task(model, loader, ...)    -> gọi SAU KHI học xong mỗi task

G2–G4 (Titans/CMS/HOPE) thay đổi *kiến trúc model*, còn baseline G1 thay đổi
*cách train* — vì vậy chúng nằm ở đây, tách khỏi models/.

- FineTune : không làm gì -> mốc dưới, dự kiến quên nặng nhất.
- EWC      : Elastic Weight Consolidation (Kirkpatrick et al., 2017).
             Sau mỗi task, ước lượng độ quan trọng từng tham số bằng
             Fisher chéo (bình phương gradient), rồi phạt việc kéo các
             tham số quan trọng rời xa giá trị cũ:
                 L = CE + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
- NCM      : gradient-free — backbone đóng băng + prototype trung bình mỗi
             class (xem models/ncm.py). Baseline "đơn giản mà khó thắng":
             kỳ vọng gần như KHÔNG quên. Titans/CMS phải vượt cả nó.

Mỗi method có footprint_floats(model): bộ nhớ THÊM (số float) mà method
tích luỹ qua stream — để so chi phí, không chỉ so accuracy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .models.classifier import mask_logits


class FineTune:
    name = "finetune"
    gradient_free = False  # True -> engine bỏ qua vòng train gradient, gọi fit_task

    def __init__(self, **_):
        pass

    def penalty(self, model) -> Optional[torch.Tensor]:
        return None

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        pass

    def footprint_floats(self, model) -> int:
        return 0


class EWC(FineTune):
    name = "ewc"

    def __init__(self, ewc_lambda: float = 1000.0, max_batches: int = 50, **_):
        self.ewc_lambda = float(ewc_lambda)
        self.max_batches = int(max_batches)
        # mỗi phần tử: {"params": {name: tensor}, "fisher": {name: tensor}}
        self._anchors: List[Dict[str, Dict[str, torch.Tensor]]] = []

    def penalty(self, model) -> Optional[torch.Tensor]:
        if not self._anchors:
            return None
        loss = None
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        for anchor in self._anchors:
            for n, p in params.items():
                term = (anchor["fisher"][n] * (p - anchor["params"][n]) ** 2).sum()
                loss = term if loss is None else loss + term
        return (self.ewc_lambda / 2.0) * loss

    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        """Ước lượng Fisher chéo trên (tối đa max_batches của) task vừa học."""
        was_training = model.training
        model.eval()
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        n_batches = 0
        for x, y in loader:
            if n_batches >= self.max_batches:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=False)
            logits = mask_logits(model(x), allowed)
            F.cross_entropy(logits, y).backward()
            for n, p in params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            n_batches += 1
        model.zero_grad(set_to_none=True)
        if n_batches == 0:
            raise RuntimeError("EWC.end_task: empty loader")
        anchor = {
            "params": {n: p.detach().clone() for n, p in params.items()},
            "fisher": {n: f / n_batches for n, f in fisher.items()},
        }
        self._anchors.append(anchor)
        if was_training:
            model.train()

    def footprint_floats(self, model) -> int:
        # mỗi task lưu (bản sao tham số + Fisher) = 2 x P float -> phình theo số task
        return sum(
            t.numel()
            for anchor in self._anchors
            for group in ("params", "fisher")
            for t in anchor[group].values()
        )


class NCM(FineTune):
    """Gradient-free: chỉ trích đặc trưng và cập nhật prototype (models/ncm.py)."""

    name = "ncm"
    gradient_free = True

    @torch.no_grad()
    def fit_task(self, model, loader, device) -> None:
        if not hasattr(model, "update_prototypes"):
            raise TypeError("Method 'ncm' cần model là NCMClassifier (run_g1 tự chọn đúng).")
        model.eval()
        for x, y in loader:
            feats = model.backbone(x.to(device))
            model.update_prototypes(feats, y.to(device))

    def footprint_floats(self, model) -> int:
        return int(model.extra_floats()) if hasattr(model, "extra_floats") else 0


_METHODS = {"finetune": FineTune, "ewc": EWC, "ncm": NCM}


def build_method(name: str, cfg: dict) -> FineTune:
    name = name.lower()
    if name not in _METHODS:
        raise KeyError(f"Unknown method '{name}'. Available: {sorted(_METHODS)}")
    kwargs = dict(cfg.get(name, {}))  # vd cfg['ewc'] = {ewc_lambda:..., max_batches:...}
    return _METHODS[name](**kwargs)
