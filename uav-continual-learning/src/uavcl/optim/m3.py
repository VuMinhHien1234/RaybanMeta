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
- "delta" (MẶC ĐỊNH của dự án — xấp xỉ delta-rule thực dụng): cập nhật
      M ← α·M + η·(g − M)  =  (α−η)·M + η·g
  Chỉ ghi phần SAI LỆCH (g − M) — gradient mới lệch bao nhiêu so với ký ức thì ghi bấy
  nhiêu; α = cổng quên, η = tốc độ ghi — HAI NÚM TÁCH RIÊNG.
  Tên cấu hình "delta" được giữ để tương thích các run cũ, nhưng đây KHÔNG phải
  Delta Momentum Eq. 48–49 đầy đủ. Công thức đầy đủ còn có hệ số quên phụ thuộc
  gᵀg và preconditioner P_i; repo tham khảo cài nó thành optimizer riêng, không
  ghép trực tiếp vào M3.
  Khi α=1, η=1−β: trùng khớp "ema" (EMA là trường hợp riêng của delta).
- "ema": M ← β·M + (1−β)·g — momentum Hebbian quen thuộc kiểu Adam/Muon (đối chứng).
- "paper": nguyên văn Algorithm 1: M ← M + β·g, V ← V + β·g² (tích lũy kiểu AdaGrad) —
  cho ablation "trung thành pseudocode".

Newton–Schulz chỉ áp cho tham số dạng ma trận (ndim ≥ 2, reshape về 2D);
bias/norm 1 chiều đi thẳng (chuẩn Muon). Weight decay kiểu decoupled (như AdamW).

key_proj_eta (thử nghiệm riêng của dự án):
Đây là phép quên theo hướng rank-1 khả thi O(dim), lấy cảm hứng từ Delta Rule nhưng
không phải P_i của Eq. 48–49. Nó trừ phần hình chiếu của m1 lên hướng gradient:
    ĝ = g / ‖g‖ ;  m1 ← m1 − key_proj_eta · (m1 · ĝ) · ĝ    (áp SAU delta-rule chuẩn ở trên)
key_proj_eta=0 (MẶC ĐỊNH) = tắt hẳn, giống hệt code trước khi sửa (không phá test/run cũ).
key_proj_eta=1 = xoá sạch phần m1 đang nằm dọc hướng gradient hiện tại mỗi bước (an toàn
về số vì đây là một phép chiếu trực giao đúng nghĩa, không thể "quá đà" kiểu khuếch đại).
CHƯA thử tự sinh giá trị mục tiêu v̂ kiểu self-modifying Titans (§8.1): đã dò
titans_pytorch 0.5.5 — to_values chỉ là linear projection cố định của input, không có
hook nào để value tự sinh từ trạng thái memory. Muốn làm đúng bản đó phải viết lại/fork
NeuralMemory, không phải chỉnh cấu hình — để dành, ghi trong LOGIC_NESTED_LEARNING.md.
"""
# ↳ GIẢI THÍCH TỔNG QUAN (đọc cái này trước, docstring dài ở trên là chi tiết học thuật):
#   - "optimizer" là thứ quyết định cập nhật trọng số model thế nào sau mỗi bước.
#   - M3 thay cho AdamW. Ý tưởng chính của paper: chính optimizer cũng là 1 BỘ NHỚ.
#     "Momentum" (m1, m2) = trung bình gradient các bước trước = ký ức về hướng đi.
#   - M3 giữ 2 ký ức: m1 (nhanh, cập nhật mỗi bước) và m2 (chậm, mỗi f bước) — y hệt
#     tinh thần CMS "đa tần số" nhưng áp lên optimizer thay vì lên model.
#   - "Newton-Schulz" = phép làm "gọn hướng" của ma trận cập nhật (đặc trưng Muon):
#     giữ HƯỚNG, bỏ ĐỘ LỚN lệch trục -> bước đi cân đối hơn.
from __future__ import annotations

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)  # hệ số quintic chuẩn của Muon (Jordan et al. 2024)
# ↳ 3 hằng số của công thức lặp Newton-Schulz bậc 5 (không cần nhớ, là hằng số chuẩn).


def newton_schulz(m: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Trực giao hoá gần đúng ma trận m (giữ 'hướng', bỏ 'độ lớn lệch trục')."""
    # ↳ Biến ma trận m thành ma trận "gần trực giao" (các cột vuông góc, độ dài ~1).
    #   Trực quan: chuẩn hoá hướng cập nhật để không tầng nào bị kéo dài quá mức.
    if m.ndim < 2 or steps <= 0 or m.numel() == 0:
        return m.clone()                          # ↳ Vector 1D / rỗng -> không xử lý, trả bản sao.
    a, b, c = _NS_COEFFS
    orig_shape = m.shape
    x = m.reshape(m.shape[0], -1).float()          # ↳ Ép về ma trận 2D (gộp các chiều còn lại).
    transposed = x.shape[0] > x.shape[1]           # ↳ Đảm bảo "cao <= rộng" để phép lặp ổn định.
    if transposed:
        x = x.t()
    x = x / (x.norm() + eps)                        # ↳ Chuẩn hoá về norm ~1 trước khi lặp.
    for _ in range(steps):                          # ↳ Lặp `steps` lần cho x tiến dần tới trực giao.
        A = x @ x.t()
        x = a * x + (b * A + c * (A @ A)) @ x       # ↳ Công thức đa thức bậc 5 của Newton-Schulz.
    if transposed:
        x = x.t()                                   # ↳ Trả lại chiều gốc nếu đã transpose.
    return x.reshape(orig_shape).to(m.dtype)        # ↳ Về đúng shape và kiểu dữ liệu ban đầu.


class M3(torch.optim.Optimizer):
    """torch.optim.Optimizer — dùng thay AdamW: M3(model.parameters(), lr=...)."""
    # ↳ Kế thừa Optimizer của PyTorch nên dùng y hệt AdamW: tạo, rồi zero_grad()/step().

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
        update_norm: str = "clip",          # "clip" (reference) | "rms" (legacy) | "none" (Algorithm 1)
        key_proj_eta: float = 0.0,          # xấp xỉ rank-1 của P_i — 0 = tắt (mặc định, xem docstring)
    ):
        # VÌ SAO CÓ update_norm (bằng chứng đo được trên EuroSAT/ViT):
        # Dòng 10 Algorithm 1 chia (O1+αO2) cho sqrt(V): tử số đã bị Newton–Schulz chuẩn
        # hoá về ~O(1) (vứt độ lớn gradient), mẫu số lại ~độ lớn gradient (rất nhỏ với
        # backbone pretrained) -> bước đi khuếch đại hàng trăm lần lr -> ‖Δw‖ ~1000%/task,
        # accuracy sập. "clip" chỉ co update khi Frobenius norm > 1, khớp implementation
        # tham khảo và không khuếch đại update nhỏ. "rms" là hành vi legacy đã dùng trong
        # các run 07-23: ép RMS = 1 nên bước của tensor lớn tăng theo sqrt(numel).
        # "none" giữ nguyên dòng 10 của Algorithm 1 để làm ablation.
        if beta_style not in ("delta", "ema", "paper"):          # ↳ Kiểm tra tham số hợp lệ (fail sớm).
            raise ValueError("beta_style phải là 'delta', 'ema' hoặc 'paper'")
        if update_norm not in ("clip", "rms", "none"):
            raise ValueError("update_norm phải là 'clip', 'rms' hoặc 'none'")
        if not (0.0 <= key_proj_eta <= 1.0):
            raise ValueError(f"key_proj_eta cần trong [0, 1] (nhận {key_proj_eta})")
        if lr < 0.0:
            raise ValueError(f"lr phải >= 0 (nhận {lr})")
        if int(frequency) < 1:
            raise ValueError(f"frequency phải >= 1 (nhận {frequency})")
        if int(ns_steps) < 0:
            raise ValueError(f"ns_steps phải >= 0 (nhận {ns_steps})")
        if eps <= 0.0:
            raise ValueError(f"eps phải > 0 (nhận {eps})")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay phải >= 0 (nhận {weight_decay})")
        for a, e in zip(delta_alpha, delta_eta):
            if not (0.0 < e <= a <= 1.0):                        # ↳ Ràng buộc toán: 0 < η <= α <= 1.
                raise ValueError(f"cần 0 < η <= α <= 1 (nhận α={a}, η={e})")
        defaults = dict(lr=lr, betas=tuple(betas), alpha=alpha, frequency=int(frequency),
                        ns_steps=int(ns_steps), eps=eps, weight_decay=weight_decay,
                        beta_style=beta_style, delta_alpha=tuple(delta_alpha),
                        delta_eta=tuple(delta_eta), update_norm=update_norm,
                        key_proj_eta=float(key_proj_eta))        # ↳ Gói mọi siêu tham số vào "defaults".
        super().__init__(params, defaults)                       # ↳ Optimizer cha lo việc nhóm tham số.

    @torch.no_grad()  # ↳ Cập nhật trọng số KHÔNG cần tính đạo hàm -> tắt autograd cho nhanh/nhẹ.
    def step(self, closure=None):
        # ↳ Hàm được gọi mỗi bước train (sau loss.backward()) để cập nhật trọng số.
        loss = None
        if closure is not None:                    # ↳ closure: cách gọi lại forward+backward (ít dùng ở đây).
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:            # ↳ Duyệt từng nhóm tham số (CMS chia nhiều nhóm).
            b1, b2, b3 = group["betas"]            # ↳ 3 hệ số quán tính: nhanh, bậc 2, chậm.
            lr, alpha, eps = group["lr"], group["alpha"], group["eps"]
            f, T, wd = group["frequency"], group["ns_steps"], group["weight_decay"]
            style = group["beta_style"]
            da1, da2 = group["delta_alpha"]        # ↳ Cổng quên α cho ký ức nhanh/chậm.
            de1, de2 = group["delta_eta"]          # ↳ Tốc độ ghi η cho ký ức nhanh/chậm.
            key_proj_eta = group["key_proj_eta"]

            for p in group["params"]:              # ↳ Duyệt từng tham số (tensor trọng số) trong nhóm.
                if p.grad is None:
                    continue                       # ↳ Không có gradient (vd CMS chưa tới hạn) -> bỏ qua.
                g = p.grad                         # ↳ g = gradient hiện tại của tham số này.
                if g.is_sparse:
                    raise RuntimeError("M3 chưa hỗ trợ sparse gradient")
                if not torch.isfinite(g).all():
                    raise FloatingPointError("M3 nhận gradient chứa NaN/Inf")
                st = self.state[p]                 # ↳ "st" = sổ ghi nhớ RIÊNG cho tham số p (momentum...).
                if not st:                         # ↳ Lần đầu gặp p -> khởi tạo các ký ức về 0.
                    st["step"] = 0
                    st["m1"] = torch.zeros_like(p)          # ký ức nhanh
                    st["m2"] = torch.zeros_like(p)          # ký ức chậm
                    st["v"] = torch.zeros_like(p)           # moment bậc 2
                    st["chunk_sum"] = torch.zeros_like(p)   # Σ g chờ đủ chunk (Eq. 75)
                    st["o2"] = torch.zeros_like(p)          # NS(ký ức chậm), dùng lại trong chunk
                st["step"] += 1                    # ↳ Đếm số bước đã xử lý tham số này.

                # decoupled weight decay (như AdamW)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)          # ↳ Co nhẹ trọng số về 0 (điều hoà), tách khỏi gradient.

                # dòng 7–8: cập nhật ký ức nhanh + moment bậc 2
                if style == "delta":
                    # Xấp xỉ delta-rule của dự án: M <- alpha*M + eta*(g-M).
                    # Không đồng nhất với Delta Momentum Eq. 48-49 đầy đủ.
                    st["m1"].mul_(da1 - de1).add_(g, alpha=de1)  # ↳ Cập nhật ký ức nhanh theo delta-rule.
                    if key_proj_eta > 0.0 and g.numel() > 1:
                        # xấp xỉ rank-1 của P_i (xem docstring đầu file): quên THÊM đúng phần
                        # m1 đang nằm dọc hướng gradient hiện tại — phép chiếu trực giao, O(dim).
                        g_hat = g / (g.norm() + eps)            # ↳ Vector đơn vị của gradient (chỉ hướng).
                        proj = torch.sum(st["m1"] * g_hat)      # ↳ Độ dài hình chiếu của m1 lên hướng đó.
                        st["m1"].sub_(g_hat, alpha=float(proj) * key_proj_eta)  # ↳ Trừ bớt phần chiếu -> "quên có hướng".
                    st["v"].mul_(b2).addcmul_(g, g, value=1.0 - b2)  # ↳ Ký ức bình phương gradient (như Adam).
                elif style == "ema":
                    st["m1"].mul_(b1).add_(g, alpha=1.0 - b1)   # ↳ Momentum trung bình trượt kiểu Adam.
                    st["v"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                else:  # "paper" — nguyên văn Algorithm 1
                    st["m1"].add_(g, alpha=b1)                   # ↳ Cộng dồn thẳng (kiểu AdaGrad, cho ablation).
                    st["v"].addcmul_(g, g, value=b2)

                # dòng 3–4 + Eq. 75: mỗi f bước, gộp chunk gradient vào ký ức chậm
                st["chunk_sum"].add_(g)            # ↳ Cộng gradient vào "sọt" chờ đủ f bước.
                if st["step"] % f == 0:            # ↳ Cứ f bước một lần thì cập nhật ký ức CHẬM.
                    if style == "delta":
                        # delta-rule trên trung bình chunk: M2 ← α·M2 + η·(ḡ − M2)
                        st["m2"].mul_(da2 - de2).add_(st["chunk_sum"], alpha=de2 / f)
                    elif style == "ema":
                        st["m2"].mul_(b3).add_(st["chunk_sum"], alpha=(1.0 - b3) / f)
                    else:
                        st["m2"].add_(st["chunk_sum"], alpha=b3)
                    st["o2"] = newton_schulz(st["m2"], steps=T)  # ↳ Làm gọn hướng ký ức chậm, dùng lại cả chunk.
                    st["chunk_sum"].zero_()                       # ↳ Đổ sọt về 0 cho chunk kế tiếp.

                # dòng 9–10: trực giao hoá ký ức nhanh, cộng 2 tầng, chia sqrt(V).
                # QUY ƯỚC MUON: NS + ký ức chậm chỉ áp cho tham số MA TRẬN (ndim>=2);
                # bias/norm 1D rơi về update kiểu Adam thuần (áp nguyên M3 lên vector
                # gây limit-cycle — đã quan sát được trên bài toán lồi 1D).
                if p.ndim >= 2:
                    update = newton_schulz(st["m1"], steps=T) + alpha * st["o2"]  # ↳ Ma trận: gộp ký ức nhanh (đã trực giao) + chậm.
                else:
                    update = st["m1"]                # ↳ Vector 1D (bias/norm): dùng thẳng ký ức nhanh.
                if style == "paper":
                    denom = st["v"].sqrt().add(eps)  # ↳ Mẫu số kiểu AdaGrad (không hiệu chỉnh bias).
                else:
                    # ema/delta: hiệu chỉnh bias cho V kiểu Adam (bước đầu V còn "non")
                    bias_corr = 1.0 - b2 ** st["step"]        # ↳ Bù cho việc V khởi tạo bằng 0 (những bước đầu).
                    denom = (st["v"] / bias_corr).sqrt().add(eps)
                step_dir = update / denom            # ↳ Hướng bước = tử số / sqrt(V) (chuẩn hoá theo độ lớn gradient).
                norm_mode = group["update_norm"]
                if norm_mode == "clip":
                    # Implementation tham khảo giới hạn norm toàn update ở 1. Khác với
                    # legacy RMS, phép này chỉ co update lớn và không khuếch đại update nhỏ.
                    step_norm = step_dir.norm()
                    step_dir = step_dir / step_norm.clamp_min(1.0)
                elif p.ndim >= 2 and norm_mode == "rms":
                    # Chế độ legacy để tái lập run cũ: ép RMS bước ma trận = 1.
                    step_dir = step_dir / step_dir.pow(2).mean().sqrt().add(1e-12)  # ↳ "Cầu chì": ép RMS bước = 1.
                if not torch.isfinite(step_dir).all():
                    raise FloatingPointError("M3 tạo hướng cập nhật chứa NaN/Inf")
                p.add_(step_dir, alpha=-lr)          # ↳ CẬP NHẬT THẬT: trọng số ← trọng số − lr·hướng_bước.
        return loss
