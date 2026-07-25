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
# ↳ GIẢI THÍCH TỔNG QUAN: "method" = CHIẾN LƯỢC chống quên. Engine gọi các "hook"
#   (móc) chung ở những thời điểm cố định (trước/sau/trong task). Mỗi method cài các
#   hook khác nhau. FineTune là lớp GỐC "không làm gì"; các method khác kế thừa và
#   chỉ ghi đè đúng hook mình cần -> đọc code chỉ cần chú ý phần được ghi đè.
from __future__ import annotations

import copy    # ↳ Để deepcopy model làm "teacher" (LwF).
import random  # ↳ Bốc ngẫu nhiên mẫu từ buffer (Replay).
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .models.classifier import mask_logits


class FineTune:
    # ↳ Baseline "ngây thơ": học task mới, KHÔNG làm gì để giữ task cũ -> quên nhiều nhất.
    #   Cũng là lớp CHA định nghĩa đủ 5 hook (đa số trả None/pass) cho method khác kế thừa.
    name = "finetune"
    gradient_free = False  # True -> engine bỏ qua vòng train gradient, gọi fit_task

    def __init__(self, **_):
        pass  # ↳ **_ nuốt mọi tham số cấu hình thừa (để build_method truyền vào không lỗi).

    def begin_task(self, model, device, allowed: Sequence[int]) -> None:
        pass  # ↳ Hook trước task: FineTune không cần chuẩn bị gì.

    def penalty(self, model) -> Optional[torch.Tensor]:
        return None  # ↳ Không có phần phạt tham số.

    def extra_batch_loss(self, model, x, logits_full, device) -> Optional[torch.Tensor]:
        return None  # ↳ Không có loss phụ theo batch.

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        pass  # ↳ Hook sau task: không làm gì.

    def footprint_floats(self, model) -> int:
        return 0  # ↳ Không tốn bộ nhớ thêm.


class EWC(FineTune):
    # ↳ EWC: sau mỗi task, ước lượng tham số nào QUAN TRỌNG (Fisher) rồi PHẠT nếu task
    #   sau kéo chúng đi xa -> giữ kiến thức cũ mà không cần lưu dữ liệu cũ.
    name = "ewc"

    def __init__(self, ewc_lambda: float = 1000.0, max_batches: int = 50, **_):
        self.ewc_lambda = float(ewc_lambda)   # ↳ Độ mạnh của phần phạt (λ càng lớn càng "bảo thủ").
        self.max_batches = int(max_batches)   # ↳ Số batch tối đa dùng để ước lượng Fisher (cho nhanh).
        # mỗi phần tử: {"params": {name: tensor}, "fisher": {name: tensor}}
        self._anchors: List[Dict[str, Dict[str, torch.Tensor]]] = []  # ↳ "Mỏ neo": mỗi task cũ lưu 1 bộ (giá trị + độ quan trọng).

    def penalty(self, model) -> Optional[torch.Tensor]:
        # ↳ Tính phần phạt = Σ độ_quan_trọng × (giá_trị_hiện_tại − giá_trị_cũ)².
        if not self._anchors:
            return None  # ↳ Task đầu tiên chưa có mỏ neo nào.
        loss = None
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        for anchor in self._anchors:               # ↳ Cộng phạt cho từng task cũ đã neo.
            for n, p in params.items():
                term = (anchor["fisher"][n] * (p - anchor["params"][n]) ** 2).sum()  # ↳ F·(θ−θ*)².
                loss = term if loss is None else loss + term
        return (self.ewc_lambda / 2.0) * loss      # ↳ Nhân λ/2 theo công thức EWC.

    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        """Ước lượng Fisher chéo trên (tối đa max_batches của) task vừa học."""
        # ↳ Fisher ≈ trung bình bình phương gradient -> tham số nào gradient lớn = nhạy = quan trọng.
        was_training = model.training
        model.eval()
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}  # ↳ Khởi tạo Fisher = 0.
        n_batches = 0
        for x, y in loader:
            if n_batches >= self.max_batches:
                break                              # ↳ Đủ số batch cần -> dừng cho nhanh.
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=False)
            logits = mask_logits(model(x), allowed)
            F.cross_entropy(logits, y).backward()  # ↳ Tính gradient để lấy độ nhạy.
            for n, p in params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2  # ↳ Cộng bình phương gradient.
            n_batches += 1
        model.zero_grad(set_to_none=True)
        if n_batches == 0:
            raise RuntimeError("EWC.end_task: empty loader")
        anchor = {
            "params": {n: p.detach().clone() for n, p in params.items()},   # ↳ Chụp GIÁ TRỊ tham số hiện tại (θ*).
            "fisher": {n: f / n_batches for n, f in fisher.items()},        # ↳ Trung bình Fisher.
        }
        self._anchors.append(anchor)               # ↳ Lưu mỏ neo cho task này.
        if was_training:
            model.train()                          # ↳ Trả model về trạng thái train như trước.

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
    # ↳ Ý tưởng đơn giản mà mạnh: giữ lại 1 ít ảnh cũ, mỗi bước train "ôn" thêm chúng.

    name = "replay"

    def __init__(self, buffer_per_class: int = 20, replay_batch: int = 32,
                 weight: float = 1.0, seed: int = 0, **_):
        self.buffer_per_class = int(buffer_per_class)  # ↳ Số ảnh giữ lại mỗi class.
        self.replay_batch = int(replay_batch)          # ↳ Mỗi bước ôn bao nhiêu ảnh cũ.
        self.weight = float(weight)                    # ↳ Trọng số của loss ôn bài.
        self._rng = random.Random(seed)                # ↳ Bộ ngẫu nhiên cố định để bốc mẫu.
        self._buf: List[Tuple[torch.Tensor, int]] = []  # (ảnh float16 CPU, nhãn)  ↳ Kho ảnh cũ.
        self._seen: List[int] = []                       # class đã có trong buffer

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        """Sau khi học task: lưu tối đa buffer_per_class ảnh mỗi class mới vào buffer."""
        quota = {int(c): self.buffer_per_class for c in allowed}  # ↳ Hạn mức còn lại mỗi class.
        for x, y in loader:
            for i in range(len(y)):
                c = int(y[i])
                if quota.get(c, 0) > 0:            # ↳ Class này còn chỗ -> lưu thêm 1 ảnh.
                    self._buf.append((x[i].detach().to(torch.float16).cpu(), c))  # ↳ Lưu float16/CPU cho nhẹ RAM.
                    quota[c] -= 1
            if all(v == 0 for v in quota.values()):
                break                              # ↳ Mọi class đủ hạn mức -> dừng.
        self._seen = sorted(set(self._seen) | {int(c) for c in allowed})  # ↳ Cập nhật danh sách class có trong kho.

    def extra_batch_loss(self, model, x, logits_full, device) -> Optional[torch.Tensor]:
        # ↳ Loss "ôn bài": bốc ngẫu nhiên ảnh cũ, tính thêm cross-entropy trên chúng.
        if not self._buf:  # task đầu: buffer còn rỗng -> chưa có gì để ôn
            return None
        idx = [self._rng.randrange(len(self._buf)) for _ in range(min(self.replay_batch, len(self._buf)))]  # ↳ Chọn ngẫu nhiên.
        xs = torch.stack([self._buf[i][0] for i in idx]).float().to(device)  # ↳ Gom ảnh cũ thành batch.
        ys = torch.tensor([self._buf[i][1] for i in idx], dtype=torch.long, device=device)
        logits = mask_logits(model(xs), self._seen)  # ↳ Che về các class đã có trong kho.
        return self.weight * F.cross_entropy(logits, ys)  # ↳ Trả loss ôn bài (đã nhân trọng số).

    def footprint_floats(self, model) -> int:
        return sum(x.numel() for x, _ in self._buf)  # (lưu float16 — đếm theo phần tử)


class LwF(FineTune):
    """Learning without Forgetting: distillation từ teacher = model TRƯỚC task hiện tại.

    Không lưu dữ liệu cũ; chỉ ép phân phối logits trên các class CŨ đứng yên:
        L = CE(task mới) + λ · T² · KL( student(x)/T || teacher(x)/T ) trên cột class cũ.
    """
    # ↳ Ý tưởng: giữ 1 bản sao model cũ (teacher). Khi học task mới, ép model mới cho ra
    #   phân phối trên CLASS CŨ giống teacher -> không quên, mà không cần ảnh cũ.

    name = "lwf"

    def __init__(self, lwf_lambda: float = 1.0, temperature: float = 2.0, **_):
        self.lwf_lambda = float(lwf_lambda)     # ↳ Độ mạnh của phần "bám theo teacher".
        self.temperature = float(temperature)  # ↳ "Nhiệt độ" làm mềm phân phối khi distill.
        self._teacher = None                    # ↳ Bản sao model cũ (chưa có ở task đầu).
        self._old_classes: List[int] = []       # ↳ Các class cũ cần giữ.

    def begin_task(self, model, device, allowed: Sequence[int]) -> None:
        """Trước task mới (trừ task đầu): chụp bản sao model làm teacher, đóng băng."""
        if self._old_classes:                   # ↳ Có class cũ -> mới cần teacher.
            self._teacher = copy.deepcopy(model).to(device)  # ↳ Sao chép nguyên model hiện tại.
            self._teacher.eval()
            for p in self._teacher.parameters():
                p.requires_grad_(False)          # ↳ Đóng băng teacher (chỉ để tham chiếu).
        else:
            self._teacher = None                 # ↳ Task đầu: chưa có gì để giữ.

    def extra_batch_loss(self, model, x, logits_full, device) -> Optional[torch.Tensor]:
        # ↳ Loss distillation: so phân phối class cũ của model hiện tại với teacher.
        if self._teacher is None:
            return None
        with torch.no_grad():
            t_logits = self._teacher(x)          # ↳ Điểm số của teacher (không gradient).
        idx = torch.as_tensor(self._old_classes, dtype=torch.long, device=logits_full.device)  # ↳ Cột class cũ.
        T = self.temperature
        p_teacher = F.softmax(t_logits[:, idx] / T, dim=1)          # ↳ Phân phối mềm của teacher.
        log_p_student = F.log_softmax(logits_full[:, idx] / T, dim=1)  # ↳ log-phân phối của model hiện tại.
        return self.lwf_lambda * (T * T) * F.kl_div(log_p_student, p_teacher, reduction="batchmean")  # ↳ KL divergence (nhân T² chuẩn distillation).

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        self._old_classes = sorted(set(self._old_classes) | {int(c) for c in allowed})  # ↳ Thêm class vừa học vào "class cũ".

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
    # ↳ Method của G2: cách train giống FineTune, chỉ thêm việc QUẢN LÝ KÝ ỨC (state) của
    #   Titans + IN norm(state) để theo dõi bộ nhớ có ổn định không.

    name = "titans"

    def begin_task(self, model, device, allowed: Sequence[int]) -> None:
        if getattr(model, "reset_mode", None) == "task":
            model.reset_state()  # ↳ Chế độ "task": xoá ký ức khi bắt đầu task mới.
        if hasattr(model, "reset_eta_alpha"):
            model.reset_eta_alpha()  # ↳ Bắt đầu đo η_t/α_t cho task này (vá lỗ hổng Task 2).

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        if hasattr(model, "state_norm"):
            print(f"[titans] reset={model.reset_mode} | norm(state) sau task = {model.state_norm():.4f}")
            # ↳ In độ lớn ký ức sau mỗi task -> con số theo dõi "phình/nổ" bộ nhớ.
        # TASK 4 + hướng 1: log độ 'sống' các nhánh self-modifying q/k/v (β, ‖W_state‖). ‖W_state‖ ~ 0
        # = nhánh chưa kích hoạt (≈ selfref); tăng dần = projection đang tự sinh theo M_{t-1}.
        if hasattr(model, "self_mod_stats"):
            st = model.self_mod_stats()
            if st:
                parts = "  ".join(f"{tag}:β={b:.2f} |W|={w:.2f}" for tag, (b, w) in st.items())
                print(f"[titans]   self-mod {parts}")
        # η_t (tốc độ ghi, Eq 76) + α_t (cổng quên) — vá lỗ hổng Task 2: η lớn dần / α~1 = hướng NỔ norm.
        if hasattr(model, "eta_alpha_stats"):
            eta, alpha = model.eta_alpha_stats()
            if eta is not None or alpha is not None:
                es = f"{eta:.4f}" if eta is not None else "n/a"
                as_ = f"{alpha:.4f}" if alpha is not None else "n/a"
                print(f"[titans]   eta_t(avg)={es}  alpha_t/forget-gate(avg)={as_}")

    def footprint_floats(self, model) -> int:
        return int(model.extra_floats()) if hasattr(model, "extra_floats") else 0


class CMS(FineTune):
    """G3 — train như finetune nhưng optimizer là CMSOptimizer (engine tự chọn khi
    cms.enabled). Method này chỉ lo LOG BẰNG CHỨNG CƠ CHẾ (task S5):

    ‖Δw‖ per-tier per-task — chụp weight đầu task, cuối task đo mức dịch chuyển
    từng tier. Kỳ vọng: tier chậm ≈ 0 (giữ kiến thức), tier nhanh lớn (thích nghi).
    Ngược lại nghĩa là mapping/chu kỳ cài sai — sửa trước khi tin bất kỳ số nào.
    """
    # ↳ Method của G3: bản thân việc "đa tần số" do CMSOptimizer làm. Method này chỉ ĐO
    #   xem mỗi tier đã dịch chuyển bao nhiêu (‖Δw‖) để kiểm chứng cơ chế chạy đúng.

    name = "cms"

    def __init__(self, **_):
        self._snap = None  # ↳ Ảnh chụp trọng số đầu task (để so cuối task).

    def begin_task(self, model, device, allowed: Sequence[int]) -> None:
        groups = getattr(model, "_cms_groups", None)  # ↳ Các tier do CMSOptimizer gắn vào model.
        if groups:
            self._snap = {
                g["name"]: [p.detach().float().cpu().clone() for p in g["params"]]  # ↳ Chụp trọng số từng tier.
                for g in groups
            }

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        groups = getattr(model, "_cms_groups", None)
        if not groups or not self._snap:
            return
        for g in groups:
            before = self._snap[g["name"]]         # ↳ Trọng số đầu task.
            delta = sum(
                float((p.detach().float().cpu() - b).norm()) ** 2   # ↳ Bình phương độ dịch chuyển mỗi tensor.
                for p, b in zip(g["params"], before)
            ) ** 0.5                                # ↳ Căn tổng = ‖Δw‖ của tier.
            base = sum(float(b.norm()) ** 2 for b in before) ** 0.5  # ↳ ‖w‖ ban đầu để tính tỉ lệ %.
            rel = delta / base if base > 0 else 0.0
            print(f"[cms] ‖Δw‖ {g['name']:<5s} (p={g['period']:<3d}): {delta:10.4f}  ({rel:.3%} của ‖w‖)")
            # ↳ In dịch chuyển tuyệt đối + tương đối: tier chậm nên ~0%, tier nhanh nên lớn.


class HOPE(CMS):
    """G4 — Titans + CMS chạy chung. Gộp vòng đời của cả hai:
    - state Titans: reset theo memory.reset (như method titans) + log norm(state)
      (canh "feature drift" — backbone trôi dưới chân memory);
    - CMS: log ‖Δw‖ per-tier (kế thừa từ CMS).
    """
    # ↳ Method của G4 = CMS (kế thừa nguyên phần log ‖Δw‖) + thêm quản lý/log ký ức Titans.

    name = "hope"

    def begin_task(self, model, device, allowed: Sequence[int]) -> None:
        if getattr(model, "reset_mode", None) == "task":
            model.reset_state()                      # ↳ Xử lý ký ức Titans (như method titans)...
        super().begin_task(model, device, allowed)  # snapshot Δw của CMS  ↳ ...rồi chụp trọng số cho CMS.

    @torch.no_grad()
    def end_task(self, model, loader, device, allowed: Sequence[int]) -> None:
        super().end_task(model, loader, device, allowed)  # in ‖Δw‖ per-tier
        if hasattr(model, "state_isfinite") and not model.state_isfinite():
            raise FloatingPointError(
                "Titans memory state chứa NaN/Inf sau task. Dừng run để không ghi metrics "
                "sai; thử giảm train.lr, bật train.grad_clip_norm, hoặc tắt từng stability flag."
            )
        if hasattr(model, "state_norm"):
            print(f"[hope] reset={model.reset_mode} | norm(state) sau task = {model.state_norm():.4f}")
            # ↳ Con số norm(state) này chính là thứ đã lộ ra bệnh (262->302->120... Forgetting 0.956).

    def footprint_floats(self, model) -> int:
        return int(model.extra_floats()) if hasattr(model, "extra_floats") else 0


class NCM(FineTune):
    """Gradient-free: chỉ trích đặc trưng và cập nhật prototype (models/ncm.py)."""
    # ↳ Method của baseline NCM: KHÔNG train bằng gradient. Engine thấy gradient_free=True
    #   sẽ gọi fit_task thay cho vòng train thường.

    name = "ncm"
    gradient_free = True  # ↳ Cờ báo engine: bỏ vòng train gradient, gọi fit_task.

    @torch.no_grad()
    def fit_task(self, model, loader, device) -> None:
        # ↳ "Học" của NCM = duyệt dữ liệu, cộng feature vào prototype của từng class.
        if not hasattr(model, "update_prototypes"):
            raise TypeError("Method 'ncm' cần model là NCMClassifier (run_g1 tự chọn đúng).")
        model.eval()
        for x, y in loader:
            feats = model.backbone(x.to(device))       # ↳ Trích feature (backbone đóng băng).
            model.update_prototypes(feats, y.to(device))  # ↳ Cộng dồn vào prototype.

    def footprint_floats(self, model) -> int:
        return int(model.extra_floats()) if hasattr(model, "extra_floats") else 0


_METHODS = {"finetune": FineTune, "ewc": EWC, "replay": Replay, "lwf": LwF,
            "ncm": NCM, "titans": TitansCL, "cms": CMS, "hope": HOPE}
# ↳ "Sổ đăng ký" ánh xạ tên method (trong config) -> class tương ứng.


def build_method(name: str, cfg: dict) -> FineTune:
    # ↳ Nhà máy: từ tên -> tạo đối tượng method, truyền vào tham số riêng của nó từ config.
    name = name.lower()
    if name not in _METHODS:
        raise KeyError(f"Unknown method '{name}'. Available: {sorted(_METHODS)}")
    kwargs = dict(cfg.get(name, {}))  # vd cfg['ewc'] = {ewc_lambda:..., max_batches:...}
    return _METHODS[name](**kwargs)
