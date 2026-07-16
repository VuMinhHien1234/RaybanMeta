"""Các "method" học liên tục của G1 (baseline). Giao diện (hook) chung:

    method.begin_task(model, device, allowed)          -> gọi TRƯỚC KHI train task
    method.penalty(model)                              -> tensor | None (regularizer tham số, vd EWC)
    method.extra_batch_loss(model, x, logits_full, device)
                                                       -> tensor | None (loss thêm theo batch, vd Replay/LwF)
    method.end_task(model, loader, device, allowed)    -> gọi SAU KHI học xong task
    method.footprint_floats(model)                     -> bộ nhớ THÊM (số float) method tích luỹ

G2–G4 (Titans/CMS/HOPE) thay đổi *kiến trúc model*, còn baseline G1 thay đổi
*cách train* — vì vậy chúng nằm ở đây, tách khỏi models/.

5 baseline (đúng danh sách trong Team_Plan G1: naive/EWC/replay/LwF + NCM bổ sung):
- FineTune : không làm gì -> mốc dưới, dự kiến quên nặng nhất.
- EWC      : Kirkpatrick et al., 2017 — Fisher chéo phạt kéo tham số quan trọng
             rời xa giá trị cũ:  L = CE + (λ/2)·Σ F_i (θ_i − θ*_i)².
- Replay   : giữ một buffer nhỏ ảnh cũ (quota mỗi class); mỗi bước train cộng
             thêm CE trên mini-batch lấy từ buffer -> "ôn bài" chống quên.
             Baseline kinh điển mạnh nhất; chi phí = RAM lưu ảnh.
- LwF      : Learning-without-Forgetting (Li & Hoiem, 2016) — trước mỗi task
             chụp teacher (bản sao model cũ, đóng băng); khi train ép logits
             của class CŨ bám theo teacher (KL, temperature T) -> không cần dữ liệu cũ.
- NCM      : gradient-free — backbone đóng băng + prototype trung bình mỗi
             class (models/ncm.py). "Đơn giản mà khó thắng": gần như KHÔNG quên.
"""
from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .models.classifier import mask_logits


class FineTune:
    name = "finetune"
    gradient_free = False  # True -> engine bỏ qua vòng train gradient, gọi fit_task

    def __init__(self, **_):
        pass

    def begin_task(self, model, device, allowed: Sequence[int]) -> None:
        pass

    def penalty(self, model) -> Optional[torch.Tensor]:
        return None

    def extra_batch_loss(self, model, x, logits_full, device) -> Optional[torch.Tensor]:
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


class Replay(FineTune):
    """Experience replay: buffer ảnh cũ (quota mỗi class) + CE "ôn bài" mỗi bước.

    Ảnh lưu ở CPU dạng float16 để tiết kiệm RAM (224² ~ 0.3MB/ảnh;
    RESISC45 45 class x 20 ảnh ~ 270MB). Chỉnh `buffer_per_class` theo máy.
    """

    name = "replay"

    def __init__(self, buffer_per_class: int = 20, replay_batch: int = 32,
                 weight: float = 1.0, seed: int = 0, **_):
        self.buffer_per_class = int(buffer_per_class)
        self.replay_batch = int(replay_batch)
        self.weight = float(weight)
        self._rng = random.Random(seed)
        self._buf: List[Tuple[torch.Tensor, int]] = []  # (ảnh float16 CPU, nhãn)
        self._seen: List[int] = []                       # class đã có trong buffer

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        """Sau khi học task: lưu tối đa buffer_per_class ảnh mỗi class mới vào buffer."""
        quota = {int(c): self.buffer_per_class for c in allowed}
        for x, y in loader:
            for i in range(len(y)):
                c = int(y[i])
                if quota.get(c, 0) > 0:
                    self._buf.append((x[i].detach().to(torch.float16).cpu(), c))
                    quota[c] -= 1
            if all(v == 0 for v in quota.values()):
                break
        self._seen = sorted(set(self._seen) | {int(c) for c in allowed})

    def extra_batch_loss(self, model, x, logits_full, device) -> Optional[torch.Tensor]:
        # Task đầu tiên: buffer chứa toàn class đang học -> không cần ôn.
        if not self._buf or len(set(c for _, c in self._buf)) <= 0:
            return None
        idx = [self._rng.randrange(len(self._buf)) for _ in range(min(self.replay_batch, len(self._buf)))]
        xs = torch.stack([self._buf[i][0] for i in idx]).float().to(device)
        ys = torch.tensor([self._buf[i][1] for i in idx], dtype=torch.long, device=device)
        logits = mask_logits(model(xs), self._seen)
        return self.weight * F.cross_entropy(logits, ys)

    def footprint_floats(self, model) -> int:
        return sum(x.numel() for x, _ in self._buf)  # (lưu float16 — đếm theo phần tử)


class LwF(FineTune):
    """Learning without Forgetting: distillation từ teacher = model TRƯỚC task hiện tại.

    Không lưu dữ liệu cũ; chỉ ép phân phối logits trên các class CŨ đứng yên:
        L = CE(task mới) + λ · T² · KL( student(x)/T || teacher(x)/T ) trên cột class cũ.
    """

    name = "lwf"

    def __init__(self, lwf_lambda: float = 1.0, temperature: float = 2.0, **_):
        self.lwf_lambda = float(lwf_lambda)
        self.temperature = float(temperature)
        self._teacher = None
        self._old_classes: List[int] = []

    def begin_task(self, model, device, allowed: Sequence[int]) -> None:
        """Trước task mới (trừ task đầu): chụp bản sao model làm teacher, đóng băng."""
        if self._old_classes:
            self._teacher = copy.deepcopy(model).to(device)
            self._teacher.eval()
            for p in self._teacher.parameters():
                p.requires_grad_(False)
        else:
            self._teacher = None

    def extra_batch_loss(self, model, x, logits_full, device) -> Optional[torch.Tensor]:
        if self._teacher is None:
            return None
        with torch.no_grad():
            t_logits = self._teacher(x)
        idx = torch.as_tensor(self._old_classes, dtype=torch.long, device=logits_full.device)
        T = self.temperature
        p_teacher = F.softmax(t_logits[:, idx] / T, dim=1)
        log_p_student = F.log_softmax(logits_full[:, idx] / T, dim=1)
        return self.lwf_lambda * (T * T) * F.kl_div(log_p_student, p_teacher, reduction="batchmean")

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        self._old_classes = sorted(set(self._old_classes) | {int(c) for c in allowed})

    def footprint_floats(self, model) -> int:
        # trong lúc train giữ 1 bản sao model (teacher)
        return sum(p.numel() for p in self._teacher.parameters()) if self._teacher is not None else 0


class TitansCL(FineTune):
    """G2 — train "backbone frozen + TitansMemory + head" như finetune thường,
    nhưng quản lý VÒNG ĐỜI STATE theo chế độ reset của model:

    - begin_task: mode "task" -> xoá state (trí nhớ chỉ sống trong task);
                  mode "never" -> giữ nguyên (trí nhớ xuyên task — đích G2);
                  mode "image" -> model tự không giữ state, không cần làm gì.
    - end_task  : log norm(state) — bằng chứng C2 (ký ức có "phình" không).
    """

    name = "titans"

    def begin_task(self, model, device, allowed: Sequence[int]) -> None:
        if getattr(model, "reset_mode", None) == "task":
            model.reset_state()

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        if hasattr(model, "state_norm"):
            print(f"[titans] reset={model.reset_mode} | norm(state) sau task = {model.state_norm():.4f}")

    def footprint_floats(self, model) -> int:
        return int(model.extra_floats()) if hasattr(model, "extra_floats") else 0


class CMS(FineTune):
    """G3 — train như finetune nhưng optimizer là CMSOptimizer (engine tự chọn khi
    cms.enabled). Method này chỉ lo LOG BẰNG CHỨNG CƠ CHẾ (task S5):

    ‖Δw‖ per-tier per-task — chụp weight đầu task, cuối task đo mức dịch chuyển
    từng tier. Kỳ vọng: tier chậm ≈ 0 (giữ kiến thức), tier nhanh lớn (thích nghi).
    Ngược lại nghĩa là mapping/chu kỳ cài sai — sửa trước khi tin bất kỳ số nào.
    """

    name = "cms"

    def __init__(self, **_):
        self._snap = None

    def begin_task(self, model, device, allowed: Sequence[int]) -> None:
        groups = getattr(model, "_cms_groups", None)
        if groups:
            self._snap = {
                g["name"]: [p.detach().float().cpu().clone() for p in g["params"]]
                for g in groups
            }

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        groups = getattr(model, "_cms_groups", None)
        if not groups or not self._snap:
            return
        for g in groups:
            before = self._snap[g["name"]]
            delta = sum(
                float((p.detach().float().cpu() - b).norm()) ** 2
                for p, b in zip(g["params"], before)
            ) ** 0.5
            base = sum(float(b.norm()) ** 2 for b in before) ** 0.5
            rel = delta / base if base > 0 else 0.0
            print(f"[cms] ‖Δw‖ {g['name']:<5s} (p={g['period']:<3d}): {delta:10.4f}  ({rel:.3%} của ‖w‖)")


class HOPE(CMS):
    """G4 — Titans + CMS chạy chung. Gộp vòng đời của cả hai:
    - state Titans: reset theo memory.reset (như method titans) + log norm(state)
      (canh "feature drift" — backbone trôi dưới chân memory);
    - CMS: log ‖Δw‖ per-tier (kế thừa từ CMS).
    """

    name = "hope"

    def begin_task(self, model, device, allowed: Sequence[int]) -> None:
        if getattr(model, "reset_mode", None) == "task":
            model.reset_state()
        super().begin_task(model, device, allowed)  # snapshot Δw của CMS

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        super().end_task(model, loader, device, allowed)  # in ‖Δw‖ per-tier
        if hasattr(model, "state_norm"):
            print(f"[hope] reset={model.reset_mode} | norm(state) sau task = {model.state_norm():.4f}")

    def footprint_floats(self, model) -> int:
        return int(model.extra_floats()) if hasattr(model, "extra_floats") else 0


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


_METHODS = {"finetune": FineTune, "ewc": EWC, "replay": Replay, "lwf": LwF,
            "ncm": NCM, "titans": TitansCL, "cms": CMS, "hope": HOPE}


def build_method(name: str, cfg: dict) -> FineTune:
    name = name.lower()
    if name not in _METHODS:
        raise KeyError(f"Unknown method '{name}'. Available: {sorted(_METHODS)}")
    kwargs = dict(cfg.get(name, {}))  # vd cfg['ewc'] = {ewc_lambda:..., max_batches:...}
    return _METHODS[name](**kwargs)
