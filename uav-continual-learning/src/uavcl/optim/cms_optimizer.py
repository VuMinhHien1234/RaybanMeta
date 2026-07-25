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

η-adaptive (task S9, TASKS_G4_SOLO §C — "self-modifying nhẹ"): thay vì η per-tier
là hằng số cố định, cho phép η_tier co giãn theo "surprise" — độ lệch HƯỚNG giữa
gradient gộp của chu kỳ này và hướng update chu kỳ TRƯỚC của chính tier đó
(cosine distance, tự nhiên nằm trong [0, 2]):
    surprise = 1 − cos(grad_agg hiện tại, hướng update trước)
    η_tier   = η_base_của_tier × surprise     (surprise→0: đồng hướng, η→gần 0 — "đã biết
               rồi, ghi khẽ thôi"; surprise→2: ngược hướng hẳn, η→2×η_base — "bất ngờ
               thật, ghi mạnh hơn"). Clip tự nhiên trong [0, η_base×2] nhờ tính chất cosine.
Bật qua `cms.eta_mode: adaptive` trong config — MẶC ĐỊNH `fixed` (giữ nguyên hành
vi cũ, không phá run nào đã chạy/đang chạy).
"""
# ↳ GIẢI THÍCH TỔNG QUAN (ý tưởng cốt lõi CMS ở mức optimizer): thay vì cập nhật MỌI
#   tầng model mỗi bước, ta cho mỗi "tier" một CHU KỲ p: tier nhanh (p=1) cập nhật
#   mỗi bước; tier chậm (p=16) chỉ cập nhật 16 bước một lần, bằng gradient GỘP lại.
#   -> tier chậm gần như đứng yên = giữ kiến thức cũ = chống quên. Tier nhanh linh
#   hoạt = học cái mới. Đây là "wrapper" bọc quanh 1 optimizer thật (M3 hoặc AdamW).
from __future__ import annotations

from typing import Dict, List

import torch


class CMSOptimizer:
    """Giao diện tối thiểu khớp engine: zero_grad() + step()."""
    # ↳ Không kế thừa Optimizer, chỉ cần "trông giống" (có zero_grad + step) để engine gọi được.

    def __init__(self, inner, tier_groups: List[Dict], grad_agg: str = "sum",
                 eta_mode: str = "fixed"):
        # tier_groups: [{"name", "period", "eta", "params": [Tensor...]}, ...] (fast -> slow)
        # ↳ inner = optimizer thật bên trong; tier_groups = các tier (từ models/cms.py).
        if grad_agg not in ("sum", "mean"):
            raise ValueError("grad_agg phải là 'sum' (Eq. 71) hoặc 'mean'")
        if eta_mode not in ("fixed", "adaptive"):
            raise ValueError("eta_mode phải là 'fixed' hoặc 'adaptive'")
        self.inner = inner
        self.tiers = tier_groups
        self.grad_agg = grad_agg
        self.eta_mode = eta_mode
        self.global_step = 0                       # ↳ Đếm tổng số bước để biết tier nào tới hạn.
        self._accum = {id(p): torch.zeros_like(p) for g in tier_groups for p in g["params"]}
        # ↳ Mỗi tham số 1 "bình" tích lũy gradient; khoá = id(p) (địa chỉ tensor).
        # sổ sách cho η-adaptive: base lr gốc mỗi tier (đã có sẵn trong inner.param_groups
        # lúc build_cms_optimizer scale = base_lr*eta) + hướng update chu kỳ trước.
        self._tier_index = {id(g): i for i, g in enumerate(tier_groups)}  # ↳ Tra nhanh tier -> chỉ số.
        self._base_lr = [float(pg["lr"]) for pg in inner.param_groups]    # ↳ Nhớ lr gốc để nhân với surprise.
        self._prev_dir = [None] * len(tier_groups)                        # ↳ Hướng update chu kỳ trước mỗi tier.

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)  # ↳ Xoá gradient (giao cho optimizer bên trong).

    @torch.no_grad()
    def step(self) -> None:
        self.global_step += 1
        # 1) tích lũy grad batch này cho mọi tier
        for g in self.tiers:
            for p in g["params"]:
                if p.grad is not None:
                    self._accum[id(p)].add_(p.grad)   # ↳ Dồn gradient batch này vào bình tích lũy.
        # 2) tier đến hạn -> grad = trung bình p bước; chưa đến hạn -> grad None
        due = [g for g in self.tiers if self.global_step % int(g["period"]) == 0]  # ↳ Tier "tới hạn" bước này.
        due_ids = {id(g) for g in due}
        for g in self.tiers:
            if id(g) in due_ids:
                div = float(g["period"]) if self.grad_agg == "mean" else 1.0  # ↳ mean: chia p; sum: giữ nguyên tổng.
                for p in g["params"]:
                    p.grad = self._accum[id(p)] / div  # ↳ Đặt lại grad = gradient gộp -> inner sẽ bước tier này.
            else:
                for p in g["params"]:
                    p.grad = None                      # ↳ Chưa tới hạn -> grad None -> inner BỎ QUA (đứng yên).
        # 2b) η-adaptive (tắt mặc định): co giãn lr per-tier theo surprise NGAY TRƯỚC khi
        # inner optimizer bước — chỉ tier đến hạn mới có gradient để đo hướng.
        if self.eta_mode == "adaptive":
            self._apply_adaptive_eta(due)
        # 3) inner optimizer chỉ bước các tier đến hạn (grad None bị bỏ qua)
        self.inner.step()                              # ↳ Optimizer thật cập nhật; tier grad=None không đổi.
        # 4) xả bộ tích lũy của tier vừa bước
        for g in due:
            for p in g["params"]:
                self._accum[id(p)].zero_()             # ↳ Đổ bình tích lũy của tier vừa bước về 0.

    def _apply_adaptive_eta(self, due: List[Dict]) -> None:
        """η_tier = η_base × (1 − cos(grad hiện tại, hướng update trước)) — xem docstring
        đầu file. Chỉ chạm tới tier ĐẾN HẠN trong chu kỳ này (due)."""
        # ↳ "surprise" = mức bất ngờ: gradient lần này lệch hướng bao nhiêu so với lần trước.
        #   Đồng hướng (đã quen) -> học nhẹ; ngược hướng (bất ngờ) -> học mạnh.
        for g in due:
            idx = self._tier_index[id(g)]
            grads = [p.grad.flatten() for p in g["params"] if p.grad is not None]  # ↳ Duỗi mọi grad của tier thành 1 vector.
            if not grads:
                continue
            flat = torch.cat(grads)                    # ↳ Nối thành 1 vector dài đại diện hướng gradient tier.
            norm = flat.norm()
            prev = self._prev_dir[idx]                 # ↳ Hướng update chu kỳ trước (None nếu lần đầu).
            if prev is None or norm < 1e-12:
                surprise = 1.0  # chưa có lịch sử / gradient ~0 -> trung tính, = hành vi fixed
            else:
                cos = torch.dot(flat, prev) / (norm * prev.norm() + 1e-12)  # ↳ Cosine giữa 2 hướng.
                surprise = float((1.0 - cos.clamp(-1.0, 1.0)))              # ↳ 1−cos ∈ [0,2].
            surprise = max(0.0, min(2.0, surprise))  # chặn tường minh, phòng lệch số ít do fp
            self._prev_dir[idx] = flat / (norm + 1e-12)  # ↳ Lưu hướng chuẩn hoá cho lần sau so sánh.
            self.inner.param_groups[idx]["lr"] = self._base_lr[idx] * surprise  # ↳ Đặt lr tier = lr gốc × surprise.

    # tiện cho debug/log
    def tier_summary(self) -> str:
        # ↳ Chuỗi tóm tắt các tier để in ra kiểm tra.
        return " | ".join(
            f"{g['name']}(p={g['period']}, eta={g['eta']}, n={sum(p.numel() for p in g['params'])})"
            for g in self.tiers
        )


def build_cms_optimizer(model, train_cfg: dict) -> CMSOptimizer:
    """Gom tier từ model (models/cms.py) -> inner optimizer (m3|adamw) -> CMSOptimizer."""
    # ↳ "Nhà máy" ráp CMSOptimizer: (1) chia tier, (2) dựng optimizer thật, (3) bọc lại.
    from ..models.cms import build_cms_param_groups, tier_report
    from . import build_optimizer

    cms_cfg = dict(train_cfg.get("cms") or {})
    groups = build_cms_param_groups(model, cms_cfg)   # ↳ Chia tham số thành tier + đóng băng nền.
    if not hasattr(model, "_cms_groups"):  # in tier_report đúng 1 lần cho cả run
        print(tier_report(groups))                    # ↳ In bảng "block nào tier nào" để kiểm tra bằng mắt.
    model._cms_groups = groups  # method `cms` dùng để log ‖Δw‖ per-tier  ↳ Gắn vào model cho method dùng.

    base_lr = float(train_cfg.get("lr", 3e-4))
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    m3_cfg = dict(train_cfg.get("m3", {}) or {})
    base_m3_frequency = int(m3_cfg.get("frequency", 16))
    frequency_unit = str(cms_cfg.get("m3_frequency_unit", "global_step")).lower()
    if frequency_unit not in ("global_step", "tier_step"):
        raise ValueError("cms.m3_frequency_unit phải là 'global_step' hoặc 'tier_step'")
    inner_groups = [
        {
            "params": g["params"],
            "lr": base_lr * float(g["eta"]),
            # CMS đã làm tier bước thưa đi. Với global_step, quy đổi f của M3 để hai
            # lịch không vô tình nhân nhau (p=64, f=16 -> 1024 batch mới có M2).
            **(
                {
                    "frequency": max(
                        1,
                        (base_m3_frequency + int(g["period"]) - 1) // int(g["period"]),
                    )
                }
                if optimizer_name == "m3" and frequency_unit == "global_step"
                else {}
            ),
        }
        for g in groups  # ↳ lr mỗi tier = lr gốc × η.
    ]
    inner = build_optimizer(inner_groups, train_cfg)  # ↳ Dựng optimizer thật (M3 hoặc AdamW) trên các nhóm.
    if optimizer_name == "m3":
        freqs = [int(pg["frequency"]) for pg in inner.param_groups]
        print(f"[cms] M3 frequency unit={frequency_unit} | per-tier f={freqs}")
    eta_mode = str(cms_cfg.get("eta_mode", "fixed")).lower()
    if eta_mode == "adaptive":
        print("[cms] eta_mode=adaptive — η per-tier co giãn theo surprise (S9)")
    return CMSOptimizer(inner, groups, grad_agg=str(cms_cfg.get("grad_agg", "sum")).lower(),
                         eta_mode=eta_mode)            # ↳ Trả về wrapper CMS bọc quanh optimizer thật.
