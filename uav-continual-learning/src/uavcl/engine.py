"""Vòng lặp học liên tục dùng chung cho CẢ dự án (G1 dựng, G2–G4 tái sử dụng).

Giao thức (class-incremental, GEM-style):
  for t in tasks:
      train model trên task t   (logits mask về class của task t; + penalty của method)
      method.end_task(...)
      for j in 0..t:
          R[t, j] = accuracy trên test của task j (logits mask về class ĐÃ THẤY)
Trả về ma trận R -> uavcl.metrics tính average_accuracy / forgetting / BWT.
"""
# ↳ GIẢI THÍCH TỔNG QUAN (đây là "trái tim" chạy thí nghiệm — mọi G1..G4 đều dùng file này):
#   Vòng lặp: học task 0, chấm điểm; học task 1, chấm lại CẢ task 0 và 1; ... Kết quả
#   gom vào ma trận R. R[i][j] = độ chính xác trên task j SAU KHI học xong task i.
#   Từ R tính được: học tốt không (đường chéo), quên bao nhiêu (cột cũ tụt sau này).
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm  # ↳ Thanh tiến trình hiển thị khi train.

from .data.stream import TaskSpec
from .models.classifier import mask_logits


def resolve_device(pref: str = "auto") -> torch.device:
    # ↳ Chọn thiết bị chạy: "auto" -> tự dò GPU NVIDIA > GPU Apple > CPU.
    pref = (pref or "auto").lower()
    if pref != "auto":
        return torch.device(pref)                  # ↳ Người dùng ép sẵn (vd "cpu") -> tôn trọng.
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()  # ↳ Đánh giá: không cần gradient.
def evaluate(model, loader, device, allowed: Sequence[int]) -> float:
    """Accuracy (0..1) với logits mask về `allowed` (thường = các class đã thấy)."""
    model.eval()                                   # ↳ Chuyển sang chế độ đánh giá (tắt dropout, không ghi ký ức).
    correct = total = 0
    for x, y in loader:                            # ↳ Duyệt từng batch trong tập test.
        x, y = x.to(device), y.to(device)          # ↳ Đưa dữ liệu lên thiết bị.
        pred = mask_logits(model(x), allowed).argmax(dim=1)  # ↳ Dự đoán = class có điểm cao nhất (sau khi che class chưa học).
        correct += int((pred == y).sum())          # ↳ Đếm số dự đoán đúng.
        total += int(y.numel())                    # ↳ Đếm tổng số mẫu.
    return correct / max(total, 1)                 # ↳ Tỉ lệ đúng (max(...,1) tránh chia 0).


# ============================ ĐÒN A — NCM-head shadow eval ============================
# Giả thuyết (KET_LUAN_G1): head Linear train-liên-tục là NÚT THẮT forgetting; NCM (prototype
# class-mean) gần như KHÔNG quên. Test rẻ: KHÔNG train lại — chỉ, sau mỗi task, dựng prototype
# từ FEATURE SAU MEMORY của dữ liệu train đã thấy, rồi phân loại test bằng cosine tới prototype.
# Nếu accuracy NCM-head > head Linear (0.62) và tiến gần/qua NCM gốc (0.69) -> xác nhận head là
# nút thắt, đáng làm bản đầy đủ. Bật bằng train.eval_ncm_head=true (mặc định TẮT -> run cũ bất biến).
@torch.no_grad()
def _memory_prototypes(model, task_loaders, device, seen_task_ids, num_classes: int, feat_dim: int):
    """Prototype = trung bình (đã chuẩn hoá) FEATURE SAU MEMORY theo class, gộp mọi task đã thấy."""
    model.eval()                                   # ↳ eval: model.features đọc BẢN SAO state, không ghi.
    psum = torch.zeros(num_classes, feat_dim, device=device)
    pcnt = torch.zeros(num_classes, device=device)
    for tid in seen_task_ids:
        for x, y in task_loaders[tid]["train"]:
            x, y = x.to(device), y.to(device)
            h = F.normalize(model.features(x).float(), dim=1)   # ↳ feature sau memory, chuẩn hoá.
            psum.index_add_(0, y, h)                            # ↳ cộng dồn theo class.
            pcnt.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
    return F.normalize(psum / pcnt.clamp(min=1.0).unsqueeze(1), dim=1)  # (C, D), class chưa thấy -> vector 0.


@torch.no_grad()
def _evaluate_ncm(model, loader, device, allowed: Sequence[int], prototypes) -> float:
    """Accuracy khi phân loại test bằng cosine(feature-sau-memory, prototype) — mask về `allowed`."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        feats = F.normalize(model.features(x).float(), dim=1)
        logits = feats @ prototypes.t()            # ↳ điểm cosine (B, C) — dùng như logits.
        pred = mask_logits(logits, allowed).argmax(dim=1)
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
    epochs = int(train_cfg.get("epochs_per_task", 3))  # ↳ Số lần lặp qua dữ liệu task này.
    # Chọn qua config: train.optimizer = adamw | m3; cms.enabled -> bọc đa tần số (G3).
    from .optim import build_optimizer

    if opt is None:                                # ↳ Chưa có optimizer -> tạo mới.
        if (train_cfg.get("cms") or {}).get("enabled", False):
            from .optim.cms_optimizer import build_cms_optimizer

            opt = build_cms_optimizer(model, train_cfg)   # ↳ G3/G4: dùng CMSOptimizer đa tần số.
        else:
            opt = build_optimizer(model.parameters(), train_cfg)  # ↳ G1/G2: optimizer thường.

    method.begin_task(model, device, allowed)  # vd LwF chụp teacher tại đây  ↳ Móc "trước task" của method.
    losses = []
    model.train()                                  # ↳ Bật chế độ train.
    for ep in range(epochs):                       # ↳ Lặp qua từng epoch.
        run, seen = 0.0, 0                          # ↳ run = tổng loss có trọng số; seen = số mẫu đã qua.
        bar = tqdm(loader, desc=f"  epoch {ep + 1}/{epochs}", leave=False)
        for x, y in bar:                            # ↳ Duyệt từng batch.
            x, y = x.to(device), y.to(device)
            logits_full = model(x)                  # ↳ Chạy model -> điểm số cho MỌI class.
            logits = mask_logits(logits_full, allowed)  # ↳ Che class không thuộc task hiện tại.
            loss = F.cross_entropy(logits, y)       # ↳ Loss phân loại chính.
            pen = method.penalty(model)                                # EWC: phạt tham số
            if pen is not None:
                loss = loss + pen                   # ↳ Cộng phần phạt (nếu method có, vd EWC).
            extra = method.extra_batch_loss(model, x, logits_full, device)  # Replay/LwF
            if extra is not None:
                loss = loss + extra                 # ↳ Cộng loss phụ theo batch (vd ôn bài Replay / distill LwF).
            opt.zero_grad(set_to_none=True)         # ↳ Xoá gradient cũ.
            loss.backward()                         # ↳ Tính gradient (lan truyền ngược).
            opt.step()                              # ↳ Cập nhật trọng số.
            run += float(loss.detach()) * y.numel() # ↳ Cộng dồn loss (nhân số mẫu để tính trung bình đúng).
            seen += int(y.numel())
            bar.set_postfix(loss=f"{run / max(seen, 1):.3f}")  # ↳ Hiện loss trung bình trên thanh tiến trình.
        losses.append(run / max(seen, 1))           # ↳ Lưu loss trung bình của epoch.
    return losses, opt                              # ↳ Trả loss + optimizer (để có thể dùng lại cho task sau).


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
    # ↳ Hàm chính điều phối toàn bộ thí nghiệm học liên tục.
    T = len(stream)                                 # ↳ Số task.
    R = np.zeros((T, T), dtype=float)               # ↳ Ma trận kết quả TxT, khởi tạo 0.
    seen: List[int] = []                            # ↳ Danh sách class đã học tính đến hiện tại.
    log: dict = {"train_loss": {}, "task_classes": {s.task_id: s.classes for s in stream}}
    # optimizer_per_task=false: ký ức gradient (M3) + pha chu kỳ CMS sống XUYÊN task (NL-đúng hơn)
    persist_opt = not bool(train_cfg.get("optimizer_per_task", True))  # ↳ Có giữ optimizer xuyên task không.
    opt_carry = None                                # ↳ Optimizer mang từ task trước sang (nếu persist).

    for t, spec in enumerate(stream):               # ↳ Học lần lượt từng task t.
        allowed_train = spec.classes                # ↳ Khi train task t, chỉ tính loss trên class của task t.
        if verbose:
            print(f"[task {t}] classes={allowed_train} | train={len(spec.train_idx)}")
        if getattr(method, "gradient_free", False): # ↳ NCM: không train bằng gradient.
            # NCM và các method không train bằng gradient: chỉ "hấp thụ" dữ liệu task
            method.fit_task(model, task_loaders[t]["train"], device)  # ↳ Chỉ cập nhật prototype.
            log["train_loss"][t] = []
        else:
            losses, opt_used = train_one_task(
                model, method, task_loaders[t]["train"], device, allowed_train, train_cfg,
                opt=opt_carry,                      # ↳ Truyền optimizer cũ vào nếu đang giữ xuyên task.
            )
            if persist_opt:
                opt_carry = opt_used                # ↳ Nhớ optimizer để task sau dùng tiếp.
            log["train_loss"][t] = losses
        method.end_task(model, task_loaders[t]["train"], device, allowed_train)  # ↳ Móc "sau task" (EWC tính Fisher, log norm...).

        seen += list(allowed_train)                 # ↳ Cập nhật danh sách class đã học.
        allowed_eval = sorted(seen)                 # ↳ Khi đánh giá, cho phép mọi class ĐÃ học.
        for j in range(t + 1):                      # ↳ Chấm điểm lại toàn bộ task 0..t.
            R[t, j] = evaluate(model, task_loaders[j]["test"], device, allowed_eval)
        # (tùy chọn) đo Forward Transfer: đánh giá task KẾ TIẾP trước khi học nó.
        # Lưu ý: với head khởi tạo mới, FWT thường ~ mức đoán mò — có ý nghĩa hơn từ G2+.
        if bool(train_cfg.get("eval_future", False)) and t + 1 < T:
            allowed_next = sorted(set(seen) | set(stream[t + 1].classes))
            R[t, t + 1] = evaluate(model, task_loaders[t + 1]["test"], device, allowed_next)  # ↳ Điền ô tam giác trên (FWT).
        if verbose:
            row = "  ".join(f"{R[t, j]:.3f}" for j in range(t + 1))
            print(f"[task {t}] test acc so far: {row}")

        # ĐÒN A: NCM-head shadow eval (song song head Linear) — chỉ khi bật cờ + model có features().
        if bool(train_cfg.get("eval_ncm_head", False)) and hasattr(model, "features") and hasattr(model, "head"):
            if "ncm_R" not in log:
                log["ncm_R"] = np.zeros((T, T), dtype=float)
            protos = _memory_prototypes(
                model, task_loaders, device, list(range(t + 1)),
                int(model.head.out_features), int(model.head.in_features),
            )
            for j in range(t + 1):
                log["ncm_R"][t, j] = _evaluate_ncm(model, task_loaders[j]["test"], device, allowed_eval, protos)
            if verbose:
                row = "  ".join(f"{log['ncm_R'][t, j]:.3f}" for j in range(t + 1))
                print(f"[task {t}] NCM-head acc so far: {row}")
    return R, log                                   # ↳ Trả ma trận kết quả + log cho phần tính metric/báo cáo.
