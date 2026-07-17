"""M3 — Multi-scale Momentum Muon, cài theo Algorithm 1 (§7.2) của NL.pdf.

Ý tưởng (theo paper): optimizer cũng là BỘ NHỚ — momentum là ký ức của gradient.
M3 = Adam + Muon + CMS áp vào chính optimizer, với HAI tầng ký ức gradient:

  mỗi bước t:   g_t = gradient
                M1 = update(M1, g_t)              # ký ức NHANH (dòng 7)
                V  = update(V, g_t²)              # moment bậc 2 kiểu Adam (dòng 8)
                O1 = NewtonSchulz_T(M1)           # trực giao hoá kiểu Muon (dòng 9)
  mỗi f bước:   M2 = update(M2, Σ g trong chunk)  # ký ức CHẬM (dòng 3, Eq. 75)
                O2 = NewtonSchulz_T(M2)           # (dòng 4) — dùng lại trong cả chunk
  cập nhật:     Θ ← Θ − η · (O1 + α·O2) / (√V + ε)   # (dòng 10)

Ba chế độ cập nhật ký ức (beta_style):
- "delta" (MẶC ĐỊNH của dự án — Delta Momentum, Eq. 48–49 §4.3): cập nhật theo delta-rule
      M ← α·M + η·(g − M)  =  (α−η)·M + η·g
  Chỉ ghi phần SAI LỆCH (g − M) — gradient mới lệch bao nhiêu so với ký ức thì ghi bấy
  nhiêu, nên update PHỤ THUỘC TRẠNG THÁI hiện tại (điểm paper chê Hebbian không có);
  α = cổng quên, η = tốc độ ghi — HAI NÚM TÁCH RIÊNG (α<1: chủ động quên gradient cũ).
  Trung thực mà nói: đây là Eq. 49 với key hằng — bản key-ma-trận đầy đủ cần momentum
  kích thước (dim×dim) cho TỪNG tham số, bất khả thi cho optimizer trên trọng số model.
  Khi α=1, η=1−β: trùng khớp "ema" (EMA là trường hợp riêng của delta).
- "ema": M ← β·M + (1−β)·g — momentum Hebbian quen thuộc kiểu Adam/Muon (đối chứng).
- "paper": nguyên văn Algorithm 1: M ← M + β·g, V ← V + β·g² (tích lũy kiểu AdaGrad) —
  cho ablation "trung thành pseudocode".

Newton–Schulz chỉ áp cho tham số dạng ma trận (ndim ≥ 2, reshape về 2D);
bias/norm 1 chiều đi thẳng (chuẩn Muon). Weight decay kiểu decoupled (như AdamW).
"""
from __future__ import annotations

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)  # hệ số quintic chuẩn của Muon (Jordan et al. 2024)


def newton_schulz(m: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Trực giao hoá gần đúng ma trận m (giữ 'hướng', bỏ 'độ lớn lệch trục')."""
    if m.ndim < 2 or steps <= 0 or m.numel() == 0:
        return m.clone()
    a, b, c = _NS_COEFFS
    orig_shape = m.shape
    x = m.reshape(m.shape[0], -1).float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.t()
    x = x / (x.norm() + eps)
    for _ in range(steps):
        A = x @ x.t()
        x = a * x + (b * A + c * (A @ A)) @ x
    if transposed:
        x = x.t()
    return x.reshape(orig_shape).to(m.dtype)


class M3(torch.optim.Optimizer):
    """torch.optim.Optimizer — dùng thay AdamW: M3(model.parameters(), lr=...)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999, 0.95),  # (β1 nhanh, β2 moment bậc 2, β3 chậm)
        alpha: float = 0.5,                 # trọng số ký ức chậm khi cộng (dòng 10)
        frequency: int = 16,                # f — bao nhiêu bước mới update ký ức chậm
        ns_steps: int = 5,                  # T — số vòng Newton–Schulz
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        beta_style: str = "delta",          # "delta" | "ema" | "paper"
        delta_alpha: tuple = (0.999, 0.9999),  # cổng quên α (ký ức nhanh, chậm)
        delta_eta: tuple = (0.1, 0.05),        # tốc độ ghi η (ký ức nhanh, chậm)
        update_norm: str = "rms",           # "rms" (ổn định, ngữ nghĩa lr kiểu Muon) | "none" (nguyên văn dòng 10)
    ):
        # VÌ SAO CÓ update_norm (bằng chứng đo được trên EuroSAT/ViT):
        # Dòng 10 Algorithm 1 chia (O1+αO2) cho sqrt(V): tử số đã bị Newton–Schulz chuẩn
        # hoá về ~O(1) (vứt độ lớn gradient), mẫu số lại ~độ lớn gradient (rất nhỏ với
        # backbone pretrained) -> bước đi khuếch đại hàng trăm lần lr -> ‖Δw‖ ~1000%/task,
        # accuracy sập. "rms": chuẩn hoá RMS của bước ma trận về 1 rồi nhân lr — mỗi bước
        # dịch đúng cỡ lr (như Muon). "none": nguyên văn paper, giữ cho ablation.
        if beta_style not in ("delta", "ema", "paper"):
            raise ValueError("beta_style phải là 'delta', 'ema' hoặc 'paper'")
        if update_norm not in ("rms", "none"):
            raise ValueError("update_norm phải là 'rms' hoặc 'none'")
        for a, e in zip(delta_alpha, delta_eta):
            if not (0.0 < e <= a <= 1.0):
                raise ValueError(f"cần 0 < η <= α <= 1 (nhận α={a}, η={e})")
        defaults = dict(lr=lr, betas=tuple(betas), alpha=alpha, frequency=int(frequency),
                        ns_steps=int(ns_steps), eps=eps, weight_decay=weight_decay,
                        beta_style=beta_style, delta_alpha=tuple(delta_alpha),
                        delta_eta=tuple(delta_eta), update_norm=update_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            b1, b2, b3 = group["betas"]
            lr, alpha, eps = group["lr"], group["alpha"], group["eps"]
            f, T, wd = group["frequency"], group["ns_steps"], group["weight_decay"]
            style = group["beta_style"]
            da1, da2 = group["delta_alpha"]
            de1, de2 = group["delta_eta"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m1"] = torch.zeros_like(p)          # ký ức nhanh
                    st["m2"] = torch.zeros_like(p)          # ký ức chậm
                    st["v"] = torch.zeros_like(p)           # moment bậc 2
                    st["chunk_sum"] = torch.zeros_like(p)   # Σ g chờ đủ chunk (Eq. 75)
                    st["o2"] = torch.zeros_like(p)          # NS(ký ức chậm), dùng lại trong chunk
                st["step"] += 1

                # decoupled weight decay (như AdamW)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # dòng 7–8: cập nhật ký ức nhanh + moment bậc 2
                if style == "delta":
                    # Delta Momentum (Eq. 48–49): M ← α·M + η·(g − M) = (α−η)·M + η·g
                    st["m1"].mul_(da1 - de1).add_(g, alpha=de1)
                    st["v"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                elif style == "ema":
                    st["m1"].mul_(b1).add_(g, alpha=1.0 - b1)
                    st["v"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                else:  # "paper" — nguyên văn Algorithm 1
                    st["m1"].add_(g, alpha=b1)
                    st["v"].addcmul_(g, g, value=b2)

                # dòng 3–4 + Eq. 75: mỗi f bước, gộp chunk gradient vào ký ức chậm
                st["chunk_sum"].add_(g)
                if st["step"] % f == 0:
                    if style == "delta":
                        # delta-rule trên trung bình chunk: M2 ← α·M2 + η·(ḡ − M2)
                        st["m2"].mul_(da2 - de2).add_(st["chunk_sum"], alpha=de2 / f)
                    elif style == "ema":
                        st["m2"].mul_(b3).add_(st["chunk_sum"], alpha=(1.0 - b3) / f)
                    else:
                        st["m2"].add_(st["chunk_sum"], alpha=b3)
                    st["o2"] = newton_schulz(st["m2"], steps=T)
                    st["chunk_sum"].zero_()

                # dòng 9–10: trực giao hoá ký ức nhanh, cộng 2 tầng, chia sqrt(V).
                # QUY ƯỚC MUON: NS + ký ức chậm chỉ áp cho tham số MA TRẬN (ndim>=2);
                # bias/norm 1D rơi về update kiểu Adam thuần (áp nguyên M3 lên vector
                # gây limit-cycle — đã quan sát được trên bài toán lồi 1D).
                if p.ndim >= 2:
                    update = newton_schulz(st["m1"], steps=T) + alpha * st["o2"]
                else:
                    update = st["m1"]
                if style == "paper":
                    denom = st["v"].sqrt().add(eps)
                else:
                    # ema/delta: hiệu chỉnh bias cho V kiểu Adam (bước đầu V còn "non")
                    bias_corr = 1.0 - b2 ** st["step"]
                    denom = (st["v"] / bias_corr).sqrt().add(eps)
                step_dir = update / denom
                if p.ndim >= 2 and group["update_norm"] == "rms" and style != "paper":
                    # giữ ngữ nghĩa lr: mỗi bước dịch chuyển RMS ≈ lr (chống khuếch đại NS/√V).
                    # KHÔNG áp cho "paper": chế độ đó là nguyên văn tuyệt đối — sqrt(V) tích lũy
                    # của nó tự đóng vai bước-giảm-dần kiểu AdaGrad.
                    step_dir = step_dir / step_dir.pow(2).mean().sqrt().add(1e-12)
                p.add_(step_dir, alpha=-lr)
        return loss
