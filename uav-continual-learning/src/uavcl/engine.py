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

import copy  # ↳ deepcopy model làm "mốc đo trôi" cho SDC (giống teacher của LwF).

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


# ---- SDC (Semantic Drift Compensation) cho NCM-head: KHÔNG đọc lại data cũ ----
# Thay vì dựng lại prototype từ MỌI task đã thấy (rebuild), SDC giữ prototype BỀN và:
#   (1) dịch prototype class CŨ theo độ trôi feature — đo CHỈ trên data task hiện tại;
#   (2) thêm prototype cho class MỚI (một lần).
# -> chi phí: 1 lượt data task hiện tại; lưu trữ: C×D + 1 model cũ tạm thời ở ranh giới.
@torch.no_grad()
def _sdc_shift(proto, old_model, new_model, loader, device, seen_before, sigma: float = 0.5):
    """Dịch prototype class CŨ theo trôi feature, CHỈ dùng data task hiện tại (in-place).

    δ = feature(model mới) − feature(model cũ) trên mẫu task hiện tại; prototype class c
    dịch theo trung bình δ có trọng số Gaussian theo khoảng cách mẫu → prototype c."""
    if not seen_before:
        return proto                                   # ↳ Chưa có class cũ -> khỏi dịch.
    old_model.eval(); new_model.eval()
    fo_list, d_list = [], []
    for x, _ in loader:                                # ↳ CHỈ duyệt data task hiện tại (không đụng task cũ).
        x = x.to(device)
        fo = F.normalize(old_model.features(x).float(), dim=1)   # ↳ feature theo model CŨ.
        fn = F.normalize(new_model.features(x).float(), dim=1)   # ↳ feature theo model MỚI.
        fo_list.append(fo); d_list.append(fn - fo)               # ↳ δ = độ trôi.
    fo = torch.cat(fo_list); dl = torch.cat(d_list)             # (M, D)
    two_s2 = 2.0 * sigma * sigma
    for c in seen_before:                              # ↳ Chỉ dịch prototype class CŨ.
        p = proto[c]
        w = torch.exp(-((fo - p) ** 2).sum(dim=1) / two_s2)     # ↳ mẫu gần prototype cũ -> trọng số lớn.
        denom = w.sum()
        if float(denom) < 1e-8:
            continue                                   # ↳ không mẫu nào gần -> để prototype nguyên.
        drift = (w.unsqueeze(1) * dl).sum(dim=0) / denom        # ↳ trôi ước lượng cho prototype này.
        proto[c] = F.normalize(p + drift, dim=0)               # ↳ dịch rồi chuẩn hoá lại.
    return proto


@torch.no_grad()
def _add_new_protos(proto, model, loader, device, new_classes):
    """Tính prototype cho các class MỚI (một lần, từ data task hiện tại) rồi ghi vào `proto`."""
    model.eval()
    C, D = int(proto.shape[0]), int(proto.shape[1])
    psum = torch.zeros(C, D, device=device)
    pcnt = torch.zeros(C, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        h = F.normalize(model.features(x).float(), dim=1)       # ↳ feature sau memory.
        psum.index_add_(0, y, h)                                # ↳ cộng dồn theo class.
        pcnt.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
    for c in new_classes:
        c = int(c)
        if float(pcnt[c]) > 0:
            proto[c] = F.normalize(psum[c] / pcnt[c], dim=0)    # ↳ trung bình -> prototype class mới.
    return proto


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
            # port từ branch NCM_Head: recipe "M3 improved" cần clip tổng norm gradient.
            # train.grad_clip_norm không đặt (None) -> bỏ qua, hành vi cũ bất biến.
            grad_clip = train_cfg.get("grad_clip_norm")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=float(grad_clip),
                    error_if_nonfinite=True,
                )
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

    # NCM-head shadow eval: 'rebuild' (cũ — dựng lại từ MỌI task) | 'sdc' (mới — bù trôi, không đọc lại data cũ)
    want_ncm = bool(train_cfg.get("eval_ncm_head", False))            # ↳ Có chạy NCM-head shadow không.
    ncm_mode = str(train_cfg.get("ncm_head_mode", "rebuild")).lower()  # ↳ Chế độ dựng prototype (mặc định = cũ).
    sdc_sigma = float(train_cfg.get("sdc_sigma", 0.5))               # ↳ Độ rộng kernel Gaussian của SDC.
    _proto = None                                                    # ↳ Prototype BỀN qua các task (chỉ cho SDC).

    # #23 checkpoint/resume (mặc định TẮT — không có checkpoint_path thì hành vi cũ y nguyên).
    ckpt_path = train_cfg.get("checkpoint_path") or None             # ↳ Đường dẫn file checkpoint (str|None).
    stop_after = train_cfg.get("stop_after_task", None)              # ↳ Dừng sớm sau task N (chạy theo ca / test).
    stop_after = int(stop_after) if stop_after is not None else None
    start_task = 0
    if ckpt_path and bool(train_cfg.get("resume", False)):
        import os
        if os.path.exists(ckpt_path):
            from .checkpoint import load_run_checkpoint
            pay = load_run_checkpoint(ckpt_path, map_location=device)
            if int(pay["num_tasks"]) != T:
                raise ValueError(f"checkpoint có {pay['num_tasks']} task, stream hiện tại {T} — không khớp.")
            model.load_state_dict(pay["model"])
            if pay.get("memory_state") is not None:
                model._state = pay["memory_state"]                    # ↳ Ký ức Titans (ngoài state_dict).
            method = pay["method"]                                    # ↳ Buffer/anchor/teacher sống lại nguyên vẹn.
            R, log = pay["R"], pay["log"]
            _proto = pay.get("proto")
            if persist_opt and pay.get("opt") is not None:            # ↳ Dựng lại optimizer rồi nạp state (M3 m1/m2/V).
                from .optim import build_optimizer
                if (train_cfg.get("cms") or {}).get("enabled", False):
                    from .optim.cms_optimizer import build_cms_optimizer
                    opt_carry = build_cms_optimizer(model, train_cfg)
                else:
                    opt_carry = build_optimizer(model.parameters(), train_cfg)
                try:
                    opt_carry.load_state_dict(pay["opt"])
                except Exception as e:  # noqa: BLE001
                    print(f"[ckpt] WARN: không nạp được optimizer state ({e}) — dùng optimizer mới.")
            torch.set_rng_state(pay["torch_rng"].cpu())
            import random as _random
            _random.setstate(pay["py_rng"])
            start_task = int(pay["task_done"]) + 1
            seen = [c for s in stream[:start_task] for c in s.classes]  # ↳ Khôi phục danh sách class đã học.
            if verbose:
                print(f"[ckpt] RESUME từ '{ckpt_path}': xong tới task {pay['task_done']} -> chạy tiếp task {start_task}.")
        elif verbose:
            print(f"[ckpt] resume=true nhưng chưa có '{ckpt_path}' -> chạy từ đầu.")

    for t, spec in enumerate(stream):               # ↳ Học lần lượt từng task t.
        if t < start_task:
            continue                                # ↳ Resume: các task đã xong trong checkpoint thì bỏ qua.
        allowed_train = spec.classes                # ↳ Khi train task t, chỉ tính loss trên class của task t.
        if verbose:
            print(f"[task {t}] classes={allowed_train} | train={len(spec.train_idx)}")
        # SDC: chụp model TRƯỚC khi train task này (mốc đo trôi feature) — chỉ khi cần, t>0.
        old_model_sdc = None
        if want_ncm and ncm_mode == "sdc" and t > 0 and hasattr(model, "features"):
            old_model_sdc = copy.deepcopy(model).eval()   # ↳ Bản sao đóng băng = "model cũ".
            for p in old_model_sdc.parameters():
                p.requires_grad_(False)
        # ---- P1 (KE_HOACH_SUA 2026-08-04) + P4 (2026-08-09): PHA 2 KHÔNG NHÃN -----------
        # Bài toán gốc (BAI_TOAN §2/§3): nhãn CHỈ có ở pha hiệu chỉnh. Trước bản vá
        # này, fit_task nhận (x, y) CÓ NHÃN ở MỌI chuyến — giải một bài dễ hơn bài thật.
        # Bật `pha2.khong_nhan`: từ chuyến `chuyen_hieu_chinh` trở đi KHÔNG dùng nhãn;
        # model nào có `hap_thu_khong_nhan` thì tự cập nhật các tầng không nhãn (và trả
        # chuỗi prequential nuôi O2); không có -> đóng băng (mốc U0).
        #
        # P4 (2026-08-09) — NHẤC CỬA PHA 2 RA NGOÀI nhánh `gradient_free`. Vì sao: trước
        # đây cửa này nằm TRONG nhánh đó, mà TitansCL/HOPE có gradient_free=False, nên
        # chúng rơi thẳng vào train_one_task và NHẬN NHÃN ở cả 12 chuyến — trong khi
        # U0/U1/U2 chỉ có nhãn ở chuyến 0. Phép so khi đó vô hiệu vì hai bên giải hai
        # bài toán khác nhau. Nay pha 2 áp cho MỌI method.
        # Mặc định TẮT -> mọi run cũ (khong_nhan=false HOẶC gradient_free=true) bất biến.
        _p2 = dict(train_cfg.get("pha2") or {})
        _khong_nhan = bool(_p2.get("khong_nhan", False))
        _chuyen_hc = int(_p2.get("chuyen_hieu_chinh", 1))
        _la_pha2 = _khong_nhan and t >= _chuyen_hc

        if _la_pha2:
            if hasattr(model, "hap_thu_khong_nhan"):
                _allowed_preq = sorted(set(seen) | set(allowed_train))
                log.setdefault("trace", {})[t] = model.hap_thu_khong_nhan(
                    task_loaders[t]["train"], device, allowed=_allowed_preq)
            elif verbose:
                print(f"[pha2] chuyến {t}: KHÔNG nhãn, model không có tầng không nhãn "
                      "-> đóng băng hoàn toàn (mốc U0)")
            log["train_loss"][t] = []
        elif getattr(method, "gradient_free", False):  # ↳ NCM/SLDA pha 1: hấp thụ CÓ nhãn.
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

        if _khong_nhan and t == _chuyen_hc - 1 and hasattr(model, "chot_moc_pha1"):
            model.chot_moc_pha1()       # ↳ cuối pha hiệu chỉnh: đóng băng mốc (m0, v0)
        method.end_task(model, task_loaders[t]["train"], device, allowed_train)  # ↳ Móc "sau task" (EWC tính Fisher, log norm...).

        seen += list(allowed_train)                 # ↳ Cập nhật danh sách class đã học.
        allowed_eval = sorted(seen)                 # ↳ Khi đánh giá, cho phép mọi class ĐÃ học.
        # (2026-08-09) Với stream revisit, MỌI thước đo (O1 `acc_hien_tai`, O3
        # `loi_ich_quay_lai`) chỉ đọc ĐƯỜNG CHÉO R[t][t]; O2/O3-preq lấy từ log["trace"].
        # Mà `pha2.test_chung=true` khiến mỗi lượt eval = TOÀN BỘ tập test, nên tam giác
        # dưới tốn 66/78 lượt (12 chuyến) mà không ai đọc -> 6h/run. Cờ này bỏ chúng đi.
        # Mặc định TẮT -> run cũ bất biến. Khi BẬT, average_accuracy/AAA/forgetting/BWT
        # mất nghĩa (tam giác dưới = 0) — run_g1 ghi null cho chúng.
        _chi_cheo = bool(train_cfg.get("eval_chi_duong_cheo", False))
        for j in ([t] if _chi_cheo else range(t + 1)):   # ↳ Chấm điểm lại task 0..t (hoặc chỉ t).
            R[t, j] = evaluate(model, task_loaders[j]["test"], device, allowed_eval)
        # (tùy chọn) đo Forward Transfer: đánh giá task KẾ TIẾP trước khi học nó.
        # Lưu ý: với head khởi tạo mới, FWT thường ~ mức đoán mò — có ý nghĩa hơn từ G2+.
        if bool(train_cfg.get("eval_future", False)) and t + 1 < T:
            allowed_next = sorted(set(seen) | set(stream[t + 1].classes))
            R[t, t + 1] = evaluate(model, task_loaders[t + 1]["test"], device, allowed_next)  # ↳ Điền ô tam giác trên (FWT).
        if verbose:
            row = "  ".join(f"{R[t, j]:.3f}" for j in ([t] if _chi_cheo else range(t + 1)))
            print(f"[task {t}] test acc so far: {row}"
                  + ("  (chỉ đường chéo)" if _chi_cheo else ""))

        # ĐÒN A: NCM-head shadow eval (song song head Linear) — chỉ khi bật cờ + model có features().
        if want_ncm and hasattr(model, "features") and hasattr(model, "head"):
            if "ncm_R" not in log:
                log["ncm_R"] = np.zeros((T, T), dtype=float)
            num_classes, feat_dim = int(model.head.out_features), int(model.head.in_features)
            if ncm_mode == "sdc":
                # SDC: KHÔNG đọc lại data cũ — dịch prototype cũ theo trôi + thêm prototype class mới.
                if _proto is None:
                    _proto = torch.zeros(num_classes, feat_dim, device=device)  # ↳ prototype bền, khởi tạo 1 lần.
                if old_model_sdc is not None:                    # ↳ t>0: bù trôi prototype class CŨ.
                    _sdc_shift(_proto, old_model_sdc, model, task_loaders[t]["train"], device,
                               seen_before=sorted(set(seen) - set(spec.classes)), sigma=sdc_sigma)
                _add_new_protos(_proto, model, task_loaders[t]["train"], device, new_classes=list(spec.classes))
                protos = _proto
            else:  # 'rebuild' (mặc định): giữ NGUYÊN hành vi cũ — dựng lại từ mọi task đã thấy.
                protos = _memory_prototypes(
                    model, task_loaders, device, list(range(t + 1)), num_classes, feat_dim,
                )
            for j in range(t + 1):
                log["ncm_R"][t, j] = _evaluate_ncm(model, task_loaders[j]["test"], device, allowed_eval, protos)
            if verbose:
                row = "  ".join(f"{log['ncm_R'][t, j]:.3f}" for j in range(t + 1))
                tag = "NCM-head(sdc)" if ncm_mode == "sdc" else "NCM-head"
                print(f"[task {t}] {tag} acc so far: {row}")
        del old_model_sdc                            # ↳ Giải phóng bản sao model cũ ngay sau khi dùng.

        # #23: lưu checkpoint SAU MỖI TASK (atomic) + tuỳ chọn dừng sớm theo ca.
        if ckpt_path:
            from .checkpoint import save_run_checkpoint
            save_run_checkpoint(ckpt_path, model=model, opt=opt_carry, method=method,
                                R=R, log=log, task_done=t, num_tasks=T, proto=_proto)
            if verbose:
                print(f"[ckpt] đã lưu sau task {t} -> {ckpt_path}")
        if stop_after is not None and t >= stop_after:
            if verbose:
                print(f"[engine] dừng sớm sau task {t} (train.stop_after_task={stop_after}).")
            break
    return R, log                                   # ↳ Trả ma trận kết quả + log cho phần tính metric/báo cáo.
