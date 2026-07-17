"""Vòng lặp học liên tục dùng chung cho CẢ dự án (G1 dựng, G2–G4 tái sử dụng).

Giao thức (class-incremental, GEM-style):
  for t in tasks:
      train model trên task t   (logits mask về class của task t; + penalty của method)
      method.end_task(...)
      for j in 0..t:
          R[t, j] = accuracy trên test của task j (logits mask về class ĐÃ THẤY)
Trả về ma trận R -> uavcl.metrics tính average_accuracy / forgetting / BWT.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data.stream import TaskSpec
from .models.classifier import mask_logits


def resolve_device(pref: str = "auto") -> torch.device:
    pref = (pref or "auto").lower()
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, allowed: Sequence[int]) -> float:
    """Accuracy (0..1) với logits mask về `allowed` (thường = các class đã thấy)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = mask_logits(model(x), allowed).argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


def train_one_task(model, method, loader, device, allowed: Sequence[int], train_cfg: dict,
                   opt=None):
    """Train model trên MỘT task. Trả về (loss từng epoch, optimizer đã dùng).

    `opt=None` -> tạo optimizer MỚI cho task này (mặc định — không mang moment cũ sang).
    Truyền `opt` có sẵn -> KÝ ỨC GRADIENT (M1/M2/V của M3) và pha chu kỳ CMS sống
    XUYÊN task — đúng tinh thần NL "optimizer cũng là bộ nhớ dài hạn"
    (bật qua config: train.optimizer_per_task: false).
    """
    epochs = int(train_cfg.get("epochs_per_task", 3))
    # Chọn qua config: train.optimizer = adamw | m3; cms.enabled -> bọc đa tần số (G3).
    from .optim import build_optimizer

    if opt is None:
        if (train_cfg.get("cms") or {}).get("enabled", False):
            from .optim.cms_optimizer import build_cms_optimizer

            opt = build_cms_optimizer(model, train_cfg)
        else:
            opt = build_optimizer(model.parameters(), train_cfg)

    method.begin_task(model, device, allowed)  # vd LwF chụp teacher tại đây
    losses = []
    model.train()
    for ep in range(epochs):
        run, seen = 0.0, 0
        bar = tqdm(loader, desc=f"  epoch {ep + 1}/{epochs}", leave=False)
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            logits_full = model(x)
            logits = mask_logits(logits_full, allowed)
            loss = F.cross_entropy(logits, y)
            pen = method.penalty(model)                                # EWC: phạt tham số
            if pen is not None:
                loss = loss + pen
            extra = method.extra_batch_loss(model, x, logits_full, device)  # Replay/LwF
            if extra is not None:
                loss = loss + extra
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.detach()) * y.numel()
            seen += int(y.numel())
            bar.set_postfix(loss=f"{run / max(seen, 1):.3f}")
        losses.append(run / max(seen, 1))
    return losses, opt


def run_continual(
    model,
    method,
    stream: List[TaskSpec],
    task_loaders: List[Dict],
    device,
    train_cfg: dict,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Chạy cả stream. Trả về (R, log). R[i, j] = acc task j sau khi học task i."""
    T = len(stream)
    R = np.zeros((T, T), dtype=float)
    seen: List[int] = []
    log: dict = {"train_loss": {}, "task_classes": {s.task_id: s.classes for s in stream}}
    # optimizer_per_task=false: ký ức gradient (M3) + pha chu kỳ CMS sống XUYÊN task (NL-đúng hơn)
    persist_opt = not bool(train_cfg.get("optimizer_per_task", True))
    opt_carry = None

    for t, spec in enumerate(stream):
        allowed_train = spec.classes
        if verbose:
            print(f"[task {t}] classes={allowed_train} | train={len(spec.train_idx)}")
        if getattr(method, "gradient_free", False):
            # NCM và các method không train bằng gradient: chỉ "hấp thụ" dữ liệu task
            method.fit_task(model, task_loaders[t]["train"], device)
            log["train_loss"][t] = []
        else:
            losses, opt_used = train_one_task(
                model, method, task_loaders[t]["train"], device, allowed_train, train_cfg,
                opt=opt_carry,
            )
            if persist_opt:
                opt_carry = opt_used
            log["train_loss"][t] = losses
        method.end_task(model, task_loaders[t]["train"], device, allowed_train)

        seen += list(allowed_train)
        allowed_eval = sorted(seen)
        for j in range(t + 1):
            R[t, j] = evaluate(model, task_loaders[j]["test"], device, allowed_eval)
        # (tùy chọn) đo Forward Transfer: đánh giá task KẾ TIẾP trước khi học nó.
        # Lưu ý: với head khởi tạo mới, FWT thường ~ mức đoán mò — có ý nghĩa hơn từ G2+.
        if bool(train_cfg.get("eval_future", False)) and t + 1 < T:
            allowed_next = sorted(set(seen) | set(stream[t + 1].classes))
            R[t, t + 1] = evaluate(model, task_loaders[t + 1]["test"], device, allowed_next)
        if verbose:
            row = "  ".join(f"{R[t, j]:.3f}" for j in range(t + 1))
            print(f"[task {t}] test acc so far: {row}")
    return R, log
