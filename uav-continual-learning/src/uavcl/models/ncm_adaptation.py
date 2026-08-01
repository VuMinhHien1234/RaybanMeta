"""No-replay feature blending and prototype transport for online NCM."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def gamma_key(gamma: float) -> str:
    """Stable ModuleDict/artifact key for a blend coefficient."""
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0 or not math.isfinite(gamma):
        raise ValueError("blend gamma phải hữu hạn và nằm trong [0, 1]")
    return f"g{gamma:.6f}".rstrip("0").rstrip(".").replace(".", "p")


@dataclass(frozen=True)
class TransportMap:
    kind: str
    matrix: torch.Tensor
    shift: torch.Tensor
    regularization: float

    def apply(self, features: torch.Tensor) -> torch.Tensor:
        values = features.float() @ self.matrix + self.shift
        if not torch.isfinite(values).all():
            raise FloatingPointError("prototype transport tạo NaN/Inf")
        return values


def _validate_pair(before: torch.Tensor, after: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    before = before.detach().float()
    after = after.detach().float()
    if before.ndim != 2 or before.shape != after.shape or before.shape[0] < 2:
        raise ValueError(
            "transport cần X_before/X_after cùng shape (N,D), N >= 2; "
            f"nhận {tuple(before.shape)} và {tuple(after.shape)}"
        )
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise FloatingPointError("feature dùng fit transport chứa NaN/Inf")
    return F.normalize(before, dim=1), F.normalize(after, dim=1)


def fit_transport(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    kind: str = "identity_ridge",
    regularization: float = 10.0,
) -> TransportMap:
    """Fit a bounded affine map from paired current-task representations."""
    x, y = _validate_pair(before, after)
    kind = str(kind).lower()
    reg = float(regularization)
    if not math.isfinite(reg) or reg < 0.0:
        raise ValueError("transport regularization phải hữu hạn và không âm")
    dim = x.shape[1]
    eye = torch.eye(dim, device=x.device, dtype=x.dtype)
    zero = torch.zeros(dim, device=x.device, dtype=x.dtype)

    if kind == "identity":
        matrix, shift = eye, zero
    elif kind == "translation":
        matrix, shift = eye, y.mean(dim=0) - x.mean(dim=0)
    elif kind == "diagonal":
        x_mean, y_mean = x.mean(dim=0), y.mean(dim=0)
        xc, yc = x - x_mean, y - y_mean
        scale = ((xc * yc).sum(dim=0) + reg) / ((xc * xc).sum(dim=0) + reg)
        matrix = torch.diag(scale)
        shift = y_mean - x_mean * scale
    elif kind == "procrustes":
        x_mean, y_mean = x.mean(dim=0), y.mean(dim=0)
        u, _, vh = torch.linalg.svd((x - x_mean).T @ (y - y_mean), full_matrices=False)
        matrix = u @ vh
        shift = y_mean - x_mean @ matrix
    elif kind == "identity_ridge":
        x_mean, y_mean = x.mean(dim=0), y.mean(dim=0)
        xc, yc = x - x_mean, y - y_mean
        lhs = xc.T @ xc + reg * eye
        rhs = xc.T @ yc + reg * eye
        matrix = torch.linalg.solve(lhs, rhs)
        shift = y_mean - x_mean @ matrix
    else:
        raise ValueError(
            "transport type không hợp lệ; chọn identity, translation, diagonal, "
            "procrustes hoặc identity_ridge"
        )
    if not torch.isfinite(matrix).all() or not torch.isfinite(shift).all():
        raise FloatingPointError("transport map chứa NaN/Inf")
    return TransportMap(kind=kind, matrix=matrix, shift=shift, regularization=reg)


def transported_features(
    features: torch.Tensor, transform: TransportMap, beta: float
) -> torch.Tensor:
    beta = float(beta)
    if not 0.0 <= beta <= 1.0 or not math.isfinite(beta):
        raise ValueError("transport beta phải hữu hạn và nằm trong [0, 1]")
    source = F.normalize(features.detach().float(), dim=1)
    mapped = F.normalize(transform.apply(source), dim=1)
    return F.normalize((1.0 - beta) * source + beta * mapped, dim=1)


@torch.no_grad()
def transport_diagnostics(
    transform: TransportMap,
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    beta: float,
    min_improvement: float = 0.0,
) -> dict:
    """Evaluate a transform on current-task held-out pairs only."""
    x, y = _validate_pair(before, after)
    prediction = transported_features(x, transform, beta)
    identity_error = float((1.0 - (x * y).sum(dim=1)).mean())
    transport_error = float((1.0 - (prediction * y).sum(dim=1)).mean())
    improvement = identity_error - transport_error
    singular_values = torch.linalg.svdvals(transform.matrix)
    smallest = float(singular_values.min())
    condition = float(singular_values.max()) / max(smallest, 1e-12)
    finite = all(
        math.isfinite(value)
        for value in (identity_error, transport_error, improvement, condition)
    )
    accepted = bool(finite and improvement > float(min_improvement))
    return {
        "kind": transform.kind,
        "regularization": transform.regularization,
        "beta": float(beta),
        "identity_error": identity_error,
        "transport_error": transport_error,
        "improvement": improvement,
        "condition_number": condition,
        "matrix_norm": float(transform.matrix.norm()),
        "accepted": accepted,
        "reason": "improved_heldout_alignment" if accepted else "no_heldout_improvement",
    }
