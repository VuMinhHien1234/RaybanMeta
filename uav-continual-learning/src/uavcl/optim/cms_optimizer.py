"""CMSOptimizer (G3, task S1) — Continuum Memory System ở mức OPTIMIZER.

Toàn bộ "đa tần số" của CMS bản retrofit nằm ở đây, KHÔNG đụng forward của ViT:

- Tham số được chia thành các TIER, mỗi tier có (chu kỳ p, hệ số học η).
- Gradient của MỌI tier được TÍCH LŨY sau mỗi batch.
- Tier chỉ được optimizer bên trong cập nhật khi `global_step % p == 0`,
  với gradient GỘP của p bước vừa qua; các bước còn lại grad của tier bị đặt
  None -> inner optimizer bỏ qua (kể cả momentum/state của nó cũng không tiến).
- Cách gộp (grad_agg): "sum" = NGUYÊN VĂN Eq. 71 của paper (η⁽ℓ⁾·Σ gradient
  trong cửa sổ C⁽ℓ⁾); "mean" = chia thêm cho p (biến thể của obekt — cùng một
  giá trị lr mang cùng ý nghĩa ở mọi chu kỳ). Hai cách tương đương về toán
  (khác hệ số η đúng p lần) — giữ cả hai để ablate "trung thành vs ổn định".
- η per-tier = scale learning-rate của param-group tương ứng trong inner
  optimizer (η→0: tier gần như bất động = giữ kiến thức pretrained).

Inner optimizer là BẤT KỲ torch optimizer nào (M3 mặc định của dự án, hoặc
AdamW đối chứng) — build qua `uavcl.optim.build_optimizer`.
"""
from __future__ import annotations

from typing import Dict, List

import torch


class CMSOptimizer:
    """Giao diện tối thiểu khớp engine: zero_grad() + step()."""

    def __init__(self, inner, tier_groups: List[Dict], grad_agg: str = "sum"):
        # tier_groups: [{"name", "period", "eta", "params": [Tensor...]}, ...] (fast -> slow)
        if grad_agg not in ("sum", "mean"):
            raise ValueError("grad_agg phải là 'sum' (Eq. 71) hoặc 'mean'")
        self.inner = inner
        self.tiers = tier_groups
        self.grad_agg = grad_agg
        self.global_step = 0
        self._accum = {id(p): torch.zeros_like(p) for g in tier_groups for p in g["params"]}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        self.global_step += 1
        # 1) tích lũy grad batch này cho mọi tier
        for g in self.tiers:
            for p in g["params"]:
                if p.grad is not None:
                    self._accum[id(p)].add_(p.grad)
        # 2) tier đến hạn -> grad = trung bình p bước; chưa đến hạn -> grad None
        due = [g for g in self.tiers if self.global_step % int(g["period"]) == 0]
        due_ids = {id(g) for g in due}
        for g in self.tiers:
            if id(g) in due_ids:
                div = float(g["period"]) if self.grad_agg == "mean" else 1.0
                for p in g["params"]:
                    p.grad = self._accum[id(p)] / div
            else:
                for p in g["params"]:
                    p.grad = None
        # 3) inner optimizer chỉ bước các tier đến hạn (grad None bị bỏ qua)
        self.inner.step()
        # 4) xả bộ tích lũy của tier vừa bước
        for g in due:
            for p in g["params"]:
                self._accum[id(p)].zero_()

    # tiện cho debug/log
    def tier_summary(self) -> str:
        return " | ".join(
            f"{g['name']}(p={g['period']}, eta={g['eta']}, n={sum(p.numel() for p in g['params'])})"
            for g in self.tiers
        )


def build_cms_optimizer(model, train_cfg: dict) -> CMSOptimizer:
    """Gom tier từ model (models/cms.py) -> inner optimizer (m3|adamw) -> CMSOptimizer."""
    from ..models.cms import build_cms_param_groups, tier_report
    from . import build_optimizer

    cms_cfg = dict(train_cfg.get("cms") or {})
    groups = build_cms_param_groups(model, cms_cfg)
    if not hasattr(model, "_cms_groups"):  # in tier_report đúng 1 lần cho cả run
        print(tier_report(groups))
    model._cms_groups = groups  # method `cms` dùng để log ‖Δw‖ per-tier

    base_lr = float(train_cfg.get("lr", 3e-4))
    inner_groups = [
        {"params": g["params"], "lr": base_lr * float(g["eta"])} for g in groups
    ]
    inner = build_optimizer(inner_groups, train_cfg)
    return CMSOptimizer(inner, groups, grad_agg=str(cms_cfg.get("grad_agg", "sum")).lower())
