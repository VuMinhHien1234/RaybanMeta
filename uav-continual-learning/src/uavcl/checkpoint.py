"""Versioned inference checkpoints for classifier, Titans state, and online NCM."""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from .models.ncm import PrototypeHead
from .models.state_utils import state_to_cpu, state_to_device

CHECKPOINT_FORMAT_VERSION = 1
PROGRESS_FORMAT_VERSION = 1


def _cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def save_inference_checkpoint(
    path: str | Path,
    model,
    *,
    config: dict,
    seen_classes: list[int],
    git_commit: str | None,
) -> None:
    """Atomically save all state needed for inference without old training images."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    online = getattr(model, "ncm_online_head", None)
    online_spec = None
    if online is not None:
        online_spec = {
            "feat_dim": int(online.feat_dim),
            "num_classes": int(online.num_classes),
        }
    blend_heads = getattr(model, "ncm_blend_heads", None)
    blend_specs = {}
    if blend_heads is not None:
        blend_specs = {
            str(key): {
                "feat_dim": int(head.feat_dim),
                "num_classes": int(head.num_classes),
            }
            for key, head in blend_heads.items()
        }
    transport_heads = getattr(model, "ncm_transport_heads", None)
    transport_specs = {}
    if transport_heads is not None:
        transport_specs = {
            str(key): {
                "feat_dim": int(head.feat_dim),
                "num_classes": int(head.num_classes),
            }
            for key, head in transport_heads.items()
        }
    memory_state = model.export_state() if hasattr(model, "export_state") else None
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": _cpu_state_dict(model),
        "memory_state": None if memory_state is None else state_to_cpu(memory_state),
        "online_ncm": online_spec,
        "ncm_blend_heads": blend_specs,
        "ncm_transport_heads": transport_specs,
        "ncm_adaptation": {
            "blend": (config.get("train", {}).get("ncm", {}) or {}).get("blend"),
            "transport": (config.get("train", {}).get("ncm", {}) or {}).get("transport"),
            "feature_distillation": config.get("train", {}).get("feature_distillation"),
        },
        "seen_classes": [int(value) for value in seen_classes],
        "config": config,
        "git_commit": git_commit,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_inference_checkpoint(
    path: str | Path,
    model,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """Load a trusted project checkpoint into an already constructed model."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint phải là một mapping")
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format_version={version!r} không được hỗ trợ; "
            f"cần version {CHECKPOINT_FORMAT_VERSION}"
        )
    required = {"model_state_dict", "seen_classes", "config"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Checkpoint thiếu key bắt buộc: {missing}")

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(map_location)
    online_spec = payload.get("online_ncm")
    if online_spec is not None and not hasattr(model, "ncm_online_head"):
        model.ncm_online_head = PrototypeHead(
            int(online_spec["feat_dim"]), int(online_spec["num_classes"])
        ).to(model_device)
    blend_specs = payload.get("ncm_blend_heads", {}) or {}
    if blend_specs and not hasattr(model, "ncm_blend_heads"):
        model.ncm_blend_heads = torch.nn.ModuleDict()
    for key, spec in blend_specs.items():
        if key not in model.ncm_blend_heads:
            model.ncm_blend_heads[key] = PrototypeHead(
                int(spec["feat_dim"]), int(spec["num_classes"])
            ).to(model_device)
    transport_specs = payload.get("ncm_transport_heads", {}) or {}
    if transport_specs and not hasattr(model, "ncm_transport_heads"):
        model.ncm_transport_heads = torch.nn.ModuleDict()
    for key, spec in transport_specs.items():
        if key not in model.ncm_transport_heads:
            model.ncm_transport_heads[key] = PrototypeHead(
                int(spec["feat_dim"]), int(spec["num_classes"])
            ).to(model_device)
    model.load_state_dict(payload["model_state_dict"], strict=strict)

    if hasattr(model, "import_state") and payload.get("memory_state") is not None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device(map_location)
        model.import_state(state_to_device(payload["memory_state"], device))
    return payload


def save_progress_checkpoint(
    path: str | Path,
    model,
    *,
    optimizer,
    next_task: int,
    result_matrix,
    log: dict,
    seen_classes: list[int],
    previous_state_norm: float | None,
) -> None:
    """Atomically save a task-boundary checkpoint for preemption-safe resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    memory_state = model.export_state() if hasattr(model, "export_state") else None
    payload = {
        "format_version": PROGRESS_FORMAT_VERSION,
        "model_state_dict": _cpu_state_dict(model),
        "memory_state": None if memory_state is None else state_to_cpu(memory_state),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "next_task": int(next_task),
        "result_matrix": np.asarray(result_matrix),
        "log": log,
        "seen_classes": [int(value) for value in seen_classes],
        "previous_state_norm": previous_state_norm,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_progress_checkpoint(
    path: str | Path,
    model,
    *,
    optimizer=None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Restore a trusted task-boundary checkpoint and all RNG streams."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if payload.get("format_version") != PROGRESS_FORMAT_VERSION:
        raise ValueError(
            f"Progress checkpoint version {payload.get('format_version')!r} không được hỗ trợ"
        )
    required = {
        "model_state_dict",
        "next_task",
        "result_matrix",
        "log",
        "seen_classes",
        "rng_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Progress checkpoint thiếu key: {missing}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if hasattr(model, "import_state") and payload.get("memory_state") is not None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device(map_location)
        model.import_state(state_to_device(payload["memory_state"], device))
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    rng = payload["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        # map_location=cuda also moves RNG byte tensors, but CUDA generators
        # require their serialized state to be supplied as CPU ByteTensor.
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    return payload
