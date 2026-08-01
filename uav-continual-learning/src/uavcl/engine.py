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

import time
import warnings
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm  # ↳ Thanh tiến trình hiển thị khi train.

from .data.stream import TaskSpec
from .models.classifier import mask_logits
from .models.ncm import PrototypeHead
from .models.ncm_adaptation import (
    fit_transport,
    gamma_key,
    transport_diagnostics,
    transported_features,
)
from .models.titans_head import blend_features


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


def _readout_features(model, x: torch.Tensor, protocol: str) -> torch.Tensor:
    """Call the model feature API while preserving compatibility with older models."""
    if not hasattr(model, "features"):
        raise TypeError("Feature readout cần model có method features(x)")
    try:
        return model.features(x, protocol=protocol)
    except TypeError:
        if protocol != "stream_batch_legacy":
            raise
        return model.features(x)


def _readout_blended_features(
    model, x: torch.Tensor, protocol: str, gamma: float
) -> torch.Tensor:
    if not hasattr(model, "feature_components"):
        if float(gamma) != 1.0:
            raise TypeError("Anchored blend cần model có feature_components()")
        return _readout_features(model, x, protocol)
    bundle = model.feature_components(x, protocol=protocol)
    return blend_features(bundle.base, bundle.titans, gamma)


@torch.no_grad()  # ↳ Đánh giá: không cần gradient.
def evaluate(
    model,
    loader,
    device,
    allowed: Sequence[int],
    feature_protocol: str = "stream_batch_legacy",
) -> float:
    """Accuracy (0..1) với logits mask về `allowed` (thường = các class đã thấy)."""
    model.eval()                                   # ↳ Chuyển sang chế độ đánh giá (tắt dropout, không ghi ký ức).
    correct = total = 0
    for x, y in loader:                            # ↳ Duyệt từng batch trong tập test.
        x, y = x.to(device), y.to(device)          # ↳ Đưa dữ liệu lên thiết bị.
        if (
            feature_protocol != "stream_batch_legacy"
            and hasattr(model, "features")
            and hasattr(model, "head")
        ):
            logits = model.head(_readout_features(model, x, feature_protocol))
        else:
            logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("evaluation logits chứa NaN/Inf")
        pred = mask_logits(logits, allowed).argmax(dim=1)  # ↳ Dự đoán = class có điểm cao nhất (sau khi che class chưa học).
        correct += int((pred == y).sum())          # ↳ Đếm số dự đoán đúng.
        total += int(y.numel())                    # ↳ Đếm tổng số mẫu.
    return correct / max(total, 1)                 # ↳ Tỉ lệ đúng (max(...,1) tránh chia 0).


# ============================ NCM feature readouts ============================
_NCM_READOUTS = ("online_current_task", "posthoc_full_seen_train")


def _resolve_ncm_config(train_cfg: dict) -> dict | None:
    raw = train_cfg.get("ncm")
    if raw is None:
        if not bool(train_cfg.get("eval_ncm_head", False)):
            return None
        warnings.warn(
            "train.eval_ncm_head đã cũ; dùng train.ncm.readouts=[posthoc_full_seen_train]",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            "readouts": ("posthoc_full_seen_train",),
            "feature_protocol": "stream_batch_legacy",
            "prototype_loader": "train",
            "legacy_output": True,
            "fail_on_nonfinite": True,
            "gammas": (1.0,),
            "transport": {"enabled": False},
        }
    if not isinstance(raw, dict):
        raise TypeError("train.ncm phải là một mapping")
    if not bool(raw.get("enabled", True)):
        return None
    readouts = raw.get("readouts", ["online_current_task", "posthoc_full_seen_train"])
    if isinstance(readouts, str):
        readouts = [readouts]
    readouts = tuple(str(value).lower() for value in readouts)
    unknown = sorted(set(readouts) - set(_NCM_READOUTS))
    if unknown:
        raise ValueError(f"NCM readout không hợp lệ: {unknown}; chọn {_NCM_READOUTS}")
    if not readouts:
        raise ValueError("train.ncm.readouts không được rỗng khi NCM enabled")
    protocol = str(raw.get("feature_protocol", "independent_image")).lower()
    if protocol not in ("stream_batch_legacy", "independent_image"):
        raise ValueError(f"NCM feature_protocol không hợp lệ: {protocol}")
    transform = str(raw.get("prototype_transform", "eval")).lower()
    if transform not in ("eval", "train_legacy"):
        raise ValueError("prototype_transform phải là eval hoặc train_legacy")
    blend_raw = raw.get("blend", {}) or {}
    if not isinstance(blend_raw, dict):
        raise TypeError("train.ncm.blend phải là một mapping")
    blend_enabled = bool(blend_raw.get("enabled", False))
    gammas = blend_raw.get("gammas", [1.0]) if blend_enabled else [1.0]
    if isinstance(gammas, (int, float)):
        gammas = [gammas]
    gammas = tuple(sorted({float(value) for value in gammas}))
    if not gammas or any(not 0.0 <= value <= 1.0 for value in gammas):
        raise ValueError("train.ncm.blend.gammas phải là danh sách không rỗng trong [0,1]")

    transport_raw = raw.get("transport", {}) or {}
    if not isinstance(transport_raw, dict):
        raise TypeError("train.ncm.transport phải là một mapping")
    transport = {
        "enabled": bool(transport_raw.get("enabled", False)),
        "type": str(transport_raw.get("type", "identity_ridge")).lower(),
        "regularization": float(transport_raw.get("regularization", 10.0)),
        "beta": float(transport_raw.get("beta", 0.5)),
        "safety_gate": bool(transport_raw.get("safety_gate", True)),
        "min_improvement": float(transport_raw.get("min_improvement", 0.0)),
        "max_condition": float(transport_raw.get("max_condition", 1e6)),
        "screen_gamma": float(transport_raw.get("screen_gamma", gammas[-1])),
    }
    candidate_raw = transport_raw.get("candidates", []) or []
    if not isinstance(candidate_raw, list):
        raise TypeError("train.ncm.transport.candidates phải là một list")
    candidates = []
    for index, item in enumerate(candidate_raw):
        if not isinstance(item, dict):
            raise TypeError("mỗi transport candidate phải là một mapping")
        candidates.append(
            {
                "name": str(item.get("name", f"candidate_{index}")),
                "type": str(item.get("type", "identity_ridge")).lower(),
                "regularization": float(item.get("regularization", 10.0)),
                "beta": float(item.get("beta", 0.5)),
            }
        )
    transport["candidates"] = candidates
    if transport["enabled"] and "online_current_task" not in readouts:
        raise ValueError("prototype transport cần readout online_current_task")
    if not 0.0 <= transport["beta"] <= 1.0:
        raise ValueError("train.ncm.transport.beta phải nằm trong [0,1]")
    if candidates and transport["screen_gamma"] not in gammas:
        raise ValueError("transport.screen_gamma phải xuất hiện trong blend.gammas")
    if any(not 0.0 <= item["beta"] <= 1.0 for item in candidates):
        raise ValueError("beta của mọi transport candidate phải nằm trong [0,1]")

    return {
        "readouts": readouts,
        "feature_protocol": protocol,
        "prototype_loader": "prototype" if transform == "eval" else "train",
        "legacy_output": False,
        "fail_on_nonfinite": bool(raw.get("fail_on_nonfinite", True)),
        "gammas": gammas,
        "transport": transport,
    }


@torch.no_grad()
def _collect_blended_features(
    model,
    loader,
    device,
    *,
    feature_protocol: str,
    gammas: Sequence[float],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Read one current-task loader pass and keep paired blend features on CPU."""
    model.eval()
    chunks = {gamma_key(gamma): [] for gamma in gammas}
    labels = []
    for x, y in loader:
        x = x.to(device)
        if not hasattr(model, "feature_components"):
            raise TypeError("blend/transport cần model có feature_components()")
        bundle = model.feature_components(x, protocol=feature_protocol)
        for gamma in gammas:
            chunks[gamma_key(gamma)].append(
                blend_features(bundle.base, bundle.titans, gamma).detach().float().cpu()
            )
        labels.append(y.detach().long().cpu())
    if not labels:
        raise RuntimeError("loader feature hiện tại rỗng")
    return (
        {key: torch.cat(values, dim=0) for key, values in chunks.items()},
        torch.cat(labels, dim=0),
    )


@torch.no_grad()
def _fit_prototype_head(
    model,
    head: PrototypeHead,
    loaders,
    device,
    task_ids: Sequence[int],
    *,
    loader_key: str,
    feature_protocol: str,
) -> int:
    model.eval()
    samples = 0
    for task_id in task_ids:
        if loader_key not in loaders[task_id]:
            raise KeyError(f"Task loader {task_id} thiếu key '{loader_key}'")
        for x, y in loaders[task_id][loader_key]:
            x, y = x.to(device), y.to(device)
            head.update(_readout_features(model, x, feature_protocol), y)
            samples += int(y.numel())
    return samples


@torch.no_grad()
def _evaluate_ncm(
    model,
    loader,
    device,
    allowed: Sequence[int],
    head: PrototypeHead,
    *,
    feature_protocol: str,
    gamma: float = 1.0,
) -> float:
    """Accuracy of a stateful prototype head on post-memory features."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        feats = _readout_blended_features(model, x, feature_protocol, gamma)
        pred = head.logits(feats, allowed=allowed).argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


@torch.no_grad()
def _evaluate_blend_heads(
    model,
    loader,
    device,
    allowed: Sequence[int],
    heads: dict[str, PrototypeHead],
    gammas: Sequence[float],
    *,
    feature_protocol: str,
) -> dict[str, float]:
    """Evaluate every gamma while encoding each image through Titans only once."""
    model.eval()
    correct = {gamma_key(gamma): 0 for gamma in gammas}
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        bundle = model.feature_components(x, protocol=feature_protocol)
        for gamma in gammas:
            key = gamma_key(gamma)
            feats = blend_features(bundle.base, bundle.titans, gamma)
            pred = heads[key].logits(feats, allowed=allowed).argmax(dim=1)
            correct[key] += int((pred == y).sum())
        total += int(y.numel())
    return {key: value / max(total, 1) for key, value in correct.items()}


def _transport_candidate_key(index: int, candidate: dict) -> str:
    safe_name = "".join(
        char if char.isalnum() else "_" for char in str(candidate["name"]).lower()
    ).strip("_")
    return f"t{index}_{safe_name or 'candidate'}"


@torch.no_grad()
def _evaluate_transport_candidates(
    model,
    loader,
    device,
    allowed: Sequence[int],
    heads: dict[str, PrototypeHead],
    *,
    gamma: float,
    feature_protocol: str,
) -> dict[str, float]:
    model.eval()
    correct = {key: 0 for key in heads}
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        bundle = model.feature_components(x, protocol=feature_protocol)
        feats = blend_features(bundle.base, bundle.titans, gamma)
        for key, head in heads.items():
            pred = head.logits(feats, allowed=allowed).argmax(dim=1)
            correct[key] += int((pred == y).sum())
        total += int(y.numel())
    return {key: value / max(total, 1) for key, value in correct.items()}


def _ncm_head_diagnostics(head: PrototypeHead, samples_encoded: int, runtime_sec: float) -> dict:
    prototypes = head.prototypes()
    return {
        "samples_encoded": int(samples_encoded),
        "runtime_sec": round(float(runtime_sec), 6),
        "prototype_counts": [float(value) for value in head.proto_count.detach().cpu()],
        "prototype_norms": [float(value) for value in prototypes.norm(dim=1).detach().cpu()],
        "seen_mask": [bool(value) for value in head.seen_mask.detach().cpu()],
        "state_finite": bool(
            torch.isfinite(head.proto_sum).all() and torch.isfinite(head.proto_count).all()
        ),
        "extra_floats": head.extra_floats(),
    }


def train_one_task(
    model,
    method,
    loader,
    device,
    allowed: Sequence[int],
    train_cfg: dict,
    opt=None,
    *,
    task_started: bool = False,
):
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
    if not task_started:
        method.begin_task(model, device, allowed)
    losses = []
    model.train()                                  # ↳ Bật chế độ train.
    for ep in range(epochs):                       # ↳ Lặp qua từng epoch.
        run, seen = 0.0, 0                          # ↳ run = tổng loss có trọng số; seen = số mẫu đã qua.
        bar = tqdm(loader, desc=f"  epoch {ep + 1}/{epochs}", leave=False)
        for batch_idx, (x, y) in enumerate(bar):    # ↳ Duyệt từng batch.
            x, y = x.to(device), y.to(device)
            feature_bundle = None
            if getattr(method, "requires_feature_bundle", False):
                if not hasattr(model, "forward_with_features"):
                    raise TypeError(
                        "method cần feature distillation nhưng model thiếu forward_with_features()"
                    )
                logits_full, feature_bundle = model.forward_with_features(x)
            else:
                logits_full = model(x)              # ↳ Chạy model -> điểm số cho MỌI class.
            if not torch.isfinite(logits_full).all():
                raise FloatingPointError(
                    f"logits chứa NaN/Inf ở epoch={ep + 1}, batch={batch_idx + 1}. "
                    "Run bị dừng để không tạo accuracy giả."
                )
            logits = mask_logits(logits_full, allowed)  # ↳ Che class không thuộc task hiện tại.
            loss = F.cross_entropy(logits, y)       # ↳ Loss phân loại chính.
            pen = method.penalty(model)                                # EWC: phạt tham số
            if pen is not None:
                loss = loss + pen                   # ↳ Cộng phần phạt (nếu method có, vd EWC).
            extra = method.extra_batch_loss(
                model, x, logits_full, device, feature_bundle=feature_bundle
            )
            if extra is not None:
                loss = loss + extra                 # ↳ Cộng loss phụ theo batch (vd ôn bài Replay / distill LwF).
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"loss chứa NaN/Inf ở epoch={ep + 1}, batch={batch_idx + 1}."
                )
            opt.zero_grad(set_to_none=True)         # ↳ Xoá gradient cũ.
            loss.backward()                         # ↳ Tính gradient (lan truyền ngược).
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
    log: dict = {
        "train_loss": {},
        "task_classes": {s.task_id: s.classes for s in stream},
        "state_norm": {},
        "state_finite": {},
        "accuracy_after_task": {},
    }
    # optimizer_per_task=false: ký ức gradient (M3) + pha chu kỳ CMS sống XUYÊN task (NL-đúng hơn)
    persist_opt = not bool(train_cfg.get("optimizer_per_task", True))  # ↳ Có giữ optimizer xuyên task không.
    opt_carry = None                                # ↳ Optimizer mang từ task trước sang (nếu persist).
    previous_state_norm = None
    eval_feature_protocol = str(
        train_cfg.get("eval_feature_protocol", "stream_batch_legacy")
    ).lower()
    if eval_feature_protocol not in ("stream_batch_legacy", "independent_image"):
        raise ValueError(f"train.eval_feature_protocol không hợp lệ: {eval_feature_protocol}")

    ncm_cfg = _resolve_ncm_config(train_cfg)
    online_ncm = None
    online_heads: dict[str, PrototypeHead] = {}
    transport_candidate_heads: dict[str, PrototypeHead] = {}
    if ncm_cfg is not None:
        if not hasattr(model, "features") or not hasattr(model, "head"):
            raise TypeError("NCM readout cần model có features() và Linear head")
        num_classes = int(model.head.out_features)
        feat_dim = int(model.head.in_features)
        log["ncm_protocol"] = {
            "readouts": list(ncm_cfg["readouts"]),
            "feature_protocol": ncm_cfg["feature_protocol"],
            "prototype_loader": ncm_cfg["prototype_loader"],
            "blend_gammas": list(ncm_cfg["gammas"]),
            "transport": dict(ncm_cfg["transport"]),
            "revisit_policy": {
                "online_current_task": False,
                "posthoc_full_seen_train": True,
            },
        }
        log["ncm_diagnostics"] = {}
        if "online_current_task" in ncm_cfg["readouts"]:
            model.ncm_blend_heads = nn.ModuleDict()
            for gamma in ncm_cfg["gammas"]:
                key = gamma_key(gamma)
                head = PrototypeHead(feat_dim, num_classes).to(device)
                if float(gamma) == 1.0:
                    online_ncm = head
                    model.ncm_online_head = head
                    log["ncm_online_R"] = np.zeros((T, T), dtype=float)
                else:
                    model.ncm_blend_heads[key] = head
                online_heads[key] = head
                log[f"ncm_blend_R_{key}"] = np.zeros((T, T), dtype=float)
            # Preserve the historical alias even when a screening config omits gamma=1.
            if online_ncm is None:
                online_ncm = online_heads[gamma_key(ncm_cfg["gammas"][-1])]
            candidates = ncm_cfg["transport"].get("candidates", [])
            if candidates:
                model.ncm_transport_heads = nn.ModuleDict()
                for index, candidate in enumerate(candidates):
                    key = _transport_candidate_key(index, candidate)
                    head = PrototypeHead(feat_dim, num_classes).to(device)
                    model.ncm_transport_heads[key] = head
                    transport_candidate_heads[key] = head
                    log[f"ncm_transport_R_{key}"] = np.zeros((T, T), dtype=float)
        if "posthoc_full_seen_train" in ncm_cfg["readouts"]:
            log["ncm_posthoc_R"] = np.zeros((T, T), dtype=float)
            if ncm_cfg["legacy_output"]:
                log["ncm_R"] = log["ncm_posthoc_R"]

    progress_path = train_cfg.get("_progress_checkpoint_path")
    start_task = 0
    if progress_path:
        from pathlib import Path

        progress_path = Path(progress_path)
        if progress_path.exists():
            from .checkpoint import load_progress_checkpoint
            from .optim import build_optimizer

            if persist_opt:
                if (train_cfg.get("cms") or {}).get("enabled", False):
                    from .optim.cms_optimizer import build_cms_optimizer

                    opt_carry = build_cms_optimizer(model, train_cfg)
                else:
                    opt_carry = build_optimizer(model.parameters(), train_cfg)
            resumed = load_progress_checkpoint(
                progress_path, model, optimizer=opt_carry, map_location=device
            )
            R = np.asarray(resumed["result_matrix"], dtype=float)
            log = resumed["log"]
            seen = [int(value) for value in resumed["seen_classes"]]
            previous_state_norm = resumed.get("previous_state_norm")
            start_task = int(resumed["next_task"])
            if hasattr(method, "_completed_tasks"):
                method._completed_tasks = start_task
            if verbose:
                print(
                    f"[RESUME] task-boundary checkpoint -> tiếp tục từ task {start_task}/{T}"
                )

    last_optimizer = opt_carry
    for t in range(start_task, T):
        spec = stream[t]                            # ↳ Học lần lượt từng task t.
        allowed_train = spec.classes                # ↳ Khi train task t, chỉ tính loss trên class của task t.
        if verbose:
            print(f"[task {t}] classes={allowed_train} | train={len(spec.train_idx)}")
        method.begin_task(model, device, allowed_train)
        transport_before = transport_before_val = None
        if (
            ncm_cfg is not None
            and online_heads
            and ncm_cfg["transport"].get("enabled", False)
            and t > 0
        ):
            transport_before, _ = _collect_blended_features(
                model,
                task_loaders[t][ncm_cfg["prototype_loader"]],
                device,
                feature_protocol=ncm_cfg["feature_protocol"],
                gammas=ncm_cfg["gammas"],
            )
            transport_before_val, _ = _collect_blended_features(
                model,
                task_loaders[t]["val"],
                device,
                feature_protocol=ncm_cfg["feature_protocol"],
                gammas=ncm_cfg["gammas"],
            )

        if getattr(method, "gradient_free", False): # ↳ NCM: không train bằng gradient.
            # NCM và các method không train bằng gradient: chỉ "hấp thụ" dữ liệu task
            method.fit_task(
                model, task_loaders[t].get("prototype", task_loaders[t]["train"]), device
            )
            log["train_loss"][t] = []
        else:
            losses, opt_used = train_one_task(
                model, method, task_loaders[t]["train"], device, allowed_train, train_cfg,
                opt=opt_carry,                      # ↳ Truyền optimizer cũ vào nếu đang giữ xuyên task.
                task_started=True,
            )
            if persist_opt:
                opt_carry = opt_used                # ↳ Nhớ optimizer để task sau dùng tiếp.
            last_optimizer = opt_used
            log["train_loss"][t] = losses
        method.end_task(model, task_loaders[t]["train"], device, allowed_train)  # ↳ Móc "sau task" (EWC tính Fisher, log norm...).
        if hasattr(method, "distillation_diagnostics"):
            distill_diag = method.distillation_diagnostics()
            if distill_diag is not None:
                log.setdefault("feature_distillation", {})[t] = distill_diag
        if hasattr(model, "state_norm"):
            current_state_norm = float(model.state_norm())
            log["state_norm"][t] = current_state_norm
            _validate_state_health(current_state_norm, previous_state_norm, train_cfg, t)
            previous_state_norm = current_state_norm
        if hasattr(model, "state_isfinite"):
            log["state_finite"][t] = bool(model.state_isfinite())

        seen += list(allowed_train)                 # ↳ Cập nhật danh sách class đã học.
        allowed_eval = sorted(seen)                 # ↳ Khi đánh giá, cho phép mọi class ĐÃ học.
        for j in range(t + 1):                      # ↳ Chấm điểm lại toàn bộ task 0..t.
            R[t, j] = evaluate(
                model,
                task_loaders[j]["test"],
                device,
                allowed_eval,
                feature_protocol=eval_feature_protocol,
            )
        # (tùy chọn) đo Forward Transfer: đánh giá task KẾ TIẾP trước khi học nó.
        # Lưu ý: với head khởi tạo mới, FWT thường ~ mức đoán mò — có ý nghĩa hơn từ G2+.
        if bool(train_cfg.get("eval_future", False)) and t + 1 < T:
            allowed_next = sorted(set(seen) | set(stream[t + 1].classes))
            R[t, t + 1] = evaluate(
                model,
                task_loaders[t + 1]["test"],
                device,
                allowed_next,
                feature_protocol=eval_feature_protocol,
            )  # ↳ Điền ô tam giác trên (FWT).
        log["accuracy_after_task"][t] = R[t].tolist()
        if verbose:
            row = "  ".join(f"{R[t, j]:.3f}" for j in range(t + 1))
            print(f"[task {t}] test acc so far: {row}")

        if ncm_cfg is not None:
            task_diag = {}
            protocol = ncm_cfg["feature_protocol"]
            loader_key = ncm_cfg["prototype_loader"]

            if online_heads:
                started = time.perf_counter()
                after_features, current_labels = _collect_blended_features(
                    model,
                    task_loaders[t][loader_key],
                    device,
                    feature_protocol=protocol,
                    gammas=ncm_cfg["gammas"],
                )
                after_val = None
                after_val_labels = None
                if transport_before_val is not None:
                    after_val, after_val_labels = _collect_blended_features(
                        model,
                        task_loaders[t]["val"],
                        device,
                        feature_protocol=protocol,
                        gammas=ncm_cfg["gammas"],
                    )
                else:
                    after_val, after_val_labels = _collect_blended_features(
                        model,
                        task_loaders[t]["val"],
                        device,
                        feature_protocol=protocol,
                        gammas=ncm_cfg["gammas"],
                    )

                blend_diag = {}
                for gamma in ncm_cfg["gammas"]:
                    key = gamma_key(gamma)
                    head = online_heads[key]
                    transport_diag = {
                        "enabled": bool(ncm_cfg["transport"].get("enabled", False)),
                        "accepted": False,
                        "reason": "first_task_or_disabled",
                    }
                    if transport_before is not None and after_val is not None:
                        cfg_t = ncm_cfg["transport"]
                        transform = fit_transport(
                            transport_before[key].to(device),
                            after_features[key].to(device),
                            kind=cfg_t["type"],
                            regularization=cfg_t["regularization"],
                        )
                        transport_diag = transport_diagnostics(
                            transform,
                            transport_before_val[key].to(device),
                            after_val[key].to(device),
                            beta=cfg_t["beta"],
                            min_improvement=cfg_t["min_improvement"],
                        )
                        if transport_diag["condition_number"] > cfg_t["max_condition"]:
                            transport_diag["accepted"] = False
                            transport_diag["reason"] = "condition_number_too_large"
                        if not cfg_t["safety_gate"]:
                            transport_diag["accepted"] = True
                            transport_diag["reason"] = "safety_gate_disabled"
                        if transport_diag["accepted"]:
                            old_mask = head.seen_mask.clone()
                            old_prototypes = head.prototypes()[old_mask]
                            moved = transported_features(
                                old_prototypes, transform, cfg_t["beta"]
                            )
                            head.replace_prototypes(old_mask, moved)

                    head.update(after_features[key].to(device), current_labels.to(device))
                    val_logits = head.logits(
                        after_val[key].to(device), allowed=allowed_eval
                    )
                    current_val_accuracy = float(
                        (
                            val_logits.argmax(dim=1)
                            == after_val_labels.to(device)
                        ).float().mean()
                    )
                    matrix_key = f"ncm_blend_R_{key}"
                    blend_diag[key] = {
                        "gamma": float(gamma),
                        **_ncm_head_diagnostics(
                            head,
                            int(current_labels.numel()),
                            time.perf_counter() - started,
                        ),
                        "revisit_old_train": False,
                        "old_samples_revisited": 0,
                        "current_task_validation_accuracy": current_val_accuracy,
                        "transport": transport_diag,
                    }
                candidate_diag = {}
                screen_gamma = float(ncm_cfg["transport"].get("screen_gamma", 1.0))
                screen_key = gamma_key(screen_gamma)
                for index, candidate in enumerate(
                    ncm_cfg["transport"].get("candidates", [])
                ):
                    candidate_key = _transport_candidate_key(index, candidate)
                    head = transport_candidate_heads[candidate_key]
                    cfg_t = ncm_cfg["transport"]
                    transport_diag = {
                        "enabled": True,
                        "accepted": False,
                        "reason": "first_task",
                    }
                    if transport_before is not None:
                        transform = fit_transport(
                            transport_before[screen_key].to(device),
                            after_features[screen_key].to(device),
                            kind=candidate["type"],
                            regularization=candidate["regularization"],
                        )
                        transport_diag = transport_diagnostics(
                            transform,
                            transport_before_val[screen_key].to(device),
                            after_val[screen_key].to(device),
                            beta=candidate["beta"],
                            min_improvement=cfg_t["min_improvement"],
                        )
                        if transport_diag["condition_number"] > cfg_t["max_condition"]:
                            transport_diag["accepted"] = False
                            transport_diag["reason"] = "condition_number_too_large"
                        if not cfg_t["safety_gate"]:
                            transport_diag["accepted"] = True
                            transport_diag["reason"] = "safety_gate_disabled"
                        if transport_diag["accepted"]:
                            old_mask = head.seen_mask.clone()
                            moved = transported_features(
                                head.prototypes()[old_mask], transform, candidate["beta"]
                            )
                            head.replace_prototypes(old_mask, moved)
                    head.update(
                        after_features[screen_key].to(device), current_labels.to(device)
                    )
                    val_logits = head.logits(
                        after_val[screen_key].to(device), allowed=allowed_eval
                    )
                    candidate_diag[candidate_key] = {
                        **candidate,
                        "gamma": screen_gamma,
                        "current_task_validation_accuracy": float(
                            (
                                val_logits.argmax(dim=1)
                                == after_val_labels.to(device)
                            ).float().mean()
                        ),
                        "revisit_old_train": False,
                        "old_samples_revisited": 0,
                        "transport": transport_diag,
                    }
                for j in range(t + 1):
                    scores = _evaluate_blend_heads(
                        model,
                        task_loaders[j]["test"],
                        device,
                        allowed_eval,
                        online_heads,
                        ncm_cfg["gammas"],
                        feature_protocol=protocol,
                    )
                    for key, score in scores.items():
                        log[f"ncm_blend_R_{key}"][t, j] = score
                    if transport_candidate_heads:
                        candidate_scores = _evaluate_transport_candidates(
                            model,
                            task_loaders[j]["test"],
                            device,
                            allowed_eval,
                            transport_candidate_heads,
                            gamma=screen_gamma,
                            feature_protocol=protocol,
                        )
                        for key, score in candidate_scores.items():
                            log[f"ncm_transport_R_{key}"][t, j] = score
                if "ncm_online_R" in log:
                    log["ncm_online_R"][t] = log[
                        f"ncm_blend_R_{gamma_key(1.0)}"
                    ][t]
                if verbose:
                    for gamma in ncm_cfg["gammas"]:
                        matrix_key = f"ncm_blend_R_{gamma_key(gamma)}"
                        row = "  ".join(
                            f"{log[matrix_key][t, j]:.3f}" for j in range(t + 1)
                        )
                        print(f"[task {t}] NCM-blend gamma={gamma:g}: {row}")
                task_diag["online_blend"] = blend_diag
                if candidate_diag:
                    task_diag["transport_candidates"] = candidate_diag
                if gamma_key(1.0) in blend_diag:
                    task_diag["online_current_task"] = blend_diag[gamma_key(1.0)]

            if "posthoc_full_seen_train" in ncm_cfg["readouts"]:
                posthoc = PrototypeHead(
                    int(model.head.in_features), int(model.head.out_features)
                ).to(device)
                started = time.perf_counter()
                encoded = _fit_prototype_head(
                    model,
                    posthoc,
                    task_loaders,
                    device,
                    list(range(t + 1)),
                    loader_key=loader_key,
                    feature_protocol=protocol,
                )
                for j in range(t + 1):
                    log["ncm_posthoc_R"][t, j] = _evaluate_ncm(
                        model,
                        task_loaders[j]["test"],
                        device,
                        allowed_eval,
                        posthoc,
                        feature_protocol=protocol,
                    )
                current_samples = len(task_loaders[t][loader_key].dataset)
                task_diag["posthoc_full_seen_train"] = {
                    **_ncm_head_diagnostics(
                        posthoc, encoded, time.perf_counter() - started
                    ),
                    "revisit_old_train": t > 0,
                    "old_samples_revisited": int(max(encoded - current_samples, 0)),
                }
                if online_ncm is not None:
                    common = online_ncm.seen_mask & posthoc.seen_mask
                    alignment = (
                        online_ncm.prototypes()[common] * posthoc.prototypes()[common]
                    ).sum(dim=1)
                    task_diag["online_posthoc_alignment"] = {
                        "cosine_by_seen_class": [
                            float(value) for value in alignment.detach().cpu()
                        ],
                        "mean_cosine": (
                            float(alignment.mean().detach().cpu())
                            if alignment.numel()
                            else None
                        ),
                    }
                if verbose:
                    row = "  ".join(
                        f"{log['ncm_posthoc_R'][t, j]:.3f}" for j in range(t + 1)
                    )
                    print(f"[task {t}] NCM-posthoc acc so far: {row}")

            log["ncm_diagnostics"][t] = task_diag

        if progress_path:
            from .checkpoint import save_progress_checkpoint

            save_progress_checkpoint(
                progress_path,
                model,
                optimizer=last_optimizer,
                next_task=t + 1,
                result_matrix=R,
                log=log,
                seen_classes=seen,
                previous_state_norm=previous_state_norm,
            )
            if verbose:
                print(f"[checkpoint] đã lưu tiến độ sau task {t}")
    return R, log                                   # ↳ Trả ma trận kết quả + log cho phần tính metric/báo cáo.


def _validate_state_health(
    current: float, previous: float | None, train_cfg: dict, task_id: int
) -> None:
    """Optional campaign guardrails for long-running memory experiments."""
    if not np.isfinite(current):
        raise FloatingPointError(f"norm(state) không hữu hạn sau task {task_id}: {current}")

    max_norm = train_cfg.get("max_state_norm")
    if max_norm is not None and current >= float(max_norm):
        raise FloatingPointError(
            f"norm(state)={current:.4f} vượt ngưỡng {float(max_norm):.4f} sau task {task_id}"
        )

    max_growth = train_cfg.get("max_state_norm_growth")
    if (
        max_growth is not None
        and previous is not None
        and previous > 0.0
        and current / previous > float(max_growth)
    ):
        raise FloatingPointError(
            f"norm(state) tăng {current / previous:.2f}x sau task {task_id}, "
            f"vượt ngưỡng {float(max_growth):.2f}x"
        )
