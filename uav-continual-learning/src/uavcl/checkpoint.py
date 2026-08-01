"""#23 — Checkpoint/resume cho run_continual (plans/TASKS_UAV_CL.md).

Vì sao: một run RESISC45 9 task mất 2–6h trên VM CPU; VM preempt/SSH đứt/OOM giữa task 7
là mất trắng. Lưu "ảnh chụp" SAU MỖI TASK; chạy lại với --resume thì nhảy thẳng tới task
kế tiếp, số ra GIỐNG HỆT chạy liền (test: tests/test_checkpoint.py).

Chụp những gì (đủ để tiếp tục, không lưu ảnh/dữ liệu):
- model.state_dict() + state ký ức Titans (model._state — KHÔNG nằm trong state_dict)
- method (pickle nguyên object: EWC anchors, buffer replay/latent, teacher LwF, rng riêng...)
- optimizer mang xuyên task (optimizer_per_task=false) — state m1/m2/V của M3 là KÝ ỨC, phải giữ
- R, log (ma trận kết quả + train_loss + ncm_R đã đo)
- _proto của SDC + RNG toàn cục (torch + python random) để tái lập chính xác

Ghi ATOMIC (ghi .tmp rồi rename) — chết giữa lúc ghi không làm hỏng checkpoint cũ.
Bản khác checkpoint.py của branch NCM_Head (inference-only, dính head riêng của branch);
bản này là TRAIN-resume cho engine nhánh chính, cờ mặc định TẮT -> run cũ bất biến.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch

FORMAT_VERSION = 1


def save_run_checkpoint(path, *, model, opt, method, R, log, task_done: int,
                        num_tasks: int, proto=None) -> None:
    """Lưu toàn bộ trạng thái sau khi XONG task `task_done` (atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opt_state = None
    if opt is not None and hasattr(opt, "state_dict"):
        try:
            opt_state = opt.state_dict()
        except Exception as e:  # noqa: BLE001 — optimizer lạ không chặn được checkpoint
            print(f"[ckpt] WARN: không lưu được optimizer state ({e}) — resume sẽ tạo optimizer mới.")
    payload = {
        "format_version": FORMAT_VERSION,
        "task_done": int(task_done),
        "num_tasks": int(num_tasks),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "memory_state": getattr(model, "_state", None),   # ↳ Titans: tensor state ngoài state_dict.
        "method": method,                                  # ↳ pickle nguyên object (tensor giữ device tag).
        "opt": opt_state,
        "R": R,
        "log": log,
        "proto": proto,                                    # ↳ prototype bền của SDC (nếu có).
        "torch_rng": torch.get_rng_state(),
        "py_rng": random.getstate(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)                                      # ↳ rename atomic: không bao giờ có file dở.


def load_run_checkpoint(path, map_location="cpu") -> dict:
    """Đọc checkpoint. weights_only=False vì có pickle object method (file do CHÍNH MÌNH ghi)."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    ver = payload.get("format_version")
    if ver != FORMAT_VERSION:
        raise ValueError(f"checkpoint format {ver} != {FORMAT_VERSION} — xoá file cũ rồi chạy lại từ đầu.")
    return payload
