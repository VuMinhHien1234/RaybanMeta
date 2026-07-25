"""TitansClassifier (G2, task A4) — ghép: frozen backbone -> adapter -> TitansMemory -> head.

Dây chuyền:  ảnh -> [mắt: ViT đóng băng] -> feature -> [phiên dịch: SeqAdapter]
             -> chuỗi -> [não: TitansMemory + state] -> [head Linear] -> logits (B, C)

Ra logits đúng giao thức của dự án -> cắm thẳng vào engine G1, KHÔNG sửa engine/metrics.

Ba chế độ reset (bậc thang A/B/C — đổi bằng 1 dòng yaml `memory.reset`):
- "image": không giữ state giữa các lần forward — trí nhớ chỉ sống trong 1 lần nhìn (sanity).
- "task" : state nối qua các batch TRONG task; reset khi sang task mới
           (reset do method `titans` gọi trong begin_task — engine không đổi).
- "never": state sống XUYÊN task — đích của G2.

Hai luật an toàn cài sẵn:
1. Truncated BPTT: state được detach sau MỖI batch train — ký ức (giá trị) vẫn nối dài,
   nhưng gradient không chảy ngược quá 1 batch. (Nếu không detach, backward lần 2 vào
   graph cũ sẽ crash + RAM nổ. NeuralMemory vẫn tự học online trong forward nên trí nhớ
   xuyên batch KHÔNG mất tác dụng.)
2. Chấm thi không ghi trí nhớ: khi eval, forward xuất phát từ BẢN SAO state (clone) và
   không lưu state mới -> eval bao nhiêu lần cũng ra đúng một kết quả, stream không bị bẩn.
"""
# ↳ GIẢI THÍCH TỔNG QUAN: Đây là model của G2. Nó nối 4 mảnh đã làm ở các file khác:
#   backbone (mắt, đóng băng) -> SeqAdapter (phiên dịch ra chuỗi) -> TitansMemory (não
#   có ký ức) -> head (Linear cho ra điểm số từng class). Điểm tinh tế nằm ở 2 "luật
#   an toàn" và 3 chế độ reset ký ức — đọc kỹ docstring trên trước khi đọc code.
from __future__ import annotations

import torch
import torch.nn as nn

from .memory import TitansMemory
from .seq_adapter import SeqAdapter
from .state_utils import (
    clone_state,
    count_floats,
    detach_state,
    state_isfinite,
    state_norm,
    state_to_cpu,
)

RESET_MODES = ("image", "task", "never")  # ↳ 3 chế độ giữ ký ức hợp lệ (xem docstring).

# G4 fix (đọc số HOPE --quick 07-17: norm(state) nhảy 262->302->120->201->179, Forgetting
# 0.956): titans-pytorch ĐÃ CÓ 4 cờ ổn định đúng tinh thần "tự điều chỉnh theo ngữ cảnh"
# của NL.pdf §8, nhưng project chưa bật (mem_kwargs rỗng trước đây). Bật có kiểm soát qua
# yaml — KHÔNG đổi mặc định của titans-pytorch khi key vắng mặt trong config (an toàn cho
# mọi config G2 cũ chưa cập nhật).
#   memory.gated_transition:        cổng học được, quyết định update mới ghi đè bao nhiêu %
#                                    (chính là cơ chế "self-modifying nhẹ" cho forget gate).
#   memory.spectral_norm_surprises: chuẩn hoá Newton-Schulz cho "surprise" trước khi ghi —
#                                    cùng họ với update_norm=rms đã cứu M3 khỏi nổ bước.
#   memory.qk_rmsnorm:               chuẩn hoá query/key — đọc/ghi bền hơn khi feature bên
#                                    dưới TRÔI (đúng bệnh feature-drift của HOPE).
#   memory.max_grad_norm:            clip gradient nội bộ khi tính surprise (float | None).
_STABILITY_KEYS = ("gated_transition", "spectral_norm_surprises", "qk_rmsnorm")  # ↳ 3 cờ dạng bật/tắt.


def _stability_kwargs(memory_cfg: dict) -> dict:
    # ↳ Đọc 4 cờ ổn định từ config -> gom thành dict để truyền xuống TitansMemory.
    #   Key nào KHÔNG có trong config thì bỏ qua -> giữ đúng mặc định thư viện (an toàn ngược).
    kw = {}
    for k in _STABILITY_KEYS:
        if k in memory_cfg:
            kw[k] = bool(memory_cfg[k])              # ↳ 3 cờ boolean.
    if memory_cfg.get("max_grad_norm") is not None:
        kw["max_grad_norm"] = float(memory_cfg["max_grad_norm"])  # ↳ Ngưỡng clip (số thực).
    return kw


class TitansClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int, memory_cfg: dict):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():  # G2: backbone LUÔN đóng băng (quyết định đã chốt)
            p.requires_grad_(False)           # ↳ Tắt gradient: mắt không học lại, chỉ có não (memory) học.
        self.backbone.eval()

        self.seq_mode = str(memory_cfg.get("seq", "image_seq")).lower()   # ↳ image_seq | token_seq.
        self.reset_mode = str(memory_cfg.get("reset", "image")).lower()   # ↳ image | task | never.
        if self.reset_mode not in RESET_MODES:
            raise ValueError(f"memory.reset '{self.reset_mode}' không hợp lệ, chọn {RESET_MODES}")

        dim = memory_cfg.get("dim", "auto")
        dim = int(feat_dim) if dim in (None, "auto") else int(dim)  # ↳ "auto" = lấy đúng độ dài feature backbone.
        if dim != int(feat_dim):
            raise ValueError(f"memory.dim={dim} phải bằng feat_dim={feat_dim} (chưa hỗ trợ lớp chiếu)")
            # ↳ Chưa hỗ trợ lớp chiếu -> bắt buộc bộ nhớ cùng chiều với feature.

        self.adapter = SeqAdapter(self.seq_mode)     # ↳ Bộ phiên dịch ảnh <-> chuỗi.
        stab_kwargs = _stability_kwargs(memory_cfg)  # ↳ Gom 4 cờ ổn định (fix 07-18).
        self.memory = TitansMemory(
            dim=dim, chunk_size=int(memory_cfg.get("chunk_size", 64)), **stab_kwargs  # ↳ Não có ký ức.
        )
        self.post_norm = nn.LayerNorm(dim)  # luật C2: ổn định số sau memory  ↳ Chuẩn hoá đầu ra memory.
        self.head = nn.Linear(dim, num_classes)      # ↳ Lớp tuyến tính -> điểm số (logit) cho từng class.
        self._state = None  # "cục ký ức" hiện tại của stream  ↳ Bắt đầu chưa có ký ức.

    # ---- giữ backbone ở eval kể cả khi model.train() (như NCMClassifier) ----
    def train(self, mode: bool = True):
        # ↳ Ghi đè .train(): khi bật chế độ train cho cả model, vẫn ép backbone ở eval
        #   (vì backbone đóng băng, không muốn nó cập nhật thống kê BatchNorm).
        super().train(mode)
        self.backbone.eval()
        return self

    # ------------------------------------------------------------- features
    @torch.no_grad()  # ↳ Backbone đóng băng -> không cần gradient khi trích feature (tiết kiệm RAM/tốc độ).
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        # ↳ Ảnh -> feature. Với token_seq thì lấy feature theo từng patch (ViT), bỏ token đặc biệt.
        if self.seq_mode == "token_seq":
            ff = getattr(self.backbone, "forward_features", None)  # ↳ Hàm của ViT cho ra feature theo patch.
            if ff is None:
                raise ValueError("token_seq cần backbone timm có forward_features (ViT).")
            feats = ff(x)  # ViT: (B, prefix+P, D)
            n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 0))  # ↳ Số token đầu đặc biệt (cls/reg).
            if feats.dim() == 3 and n_prefix > 0:
                feats = feats[:, n_prefix:, :]  # ↳ Cắt bỏ token đặc biệt, chỉ giữ patch thật.
            return feats
        return self.backbone(x)  # (B, D) pooled  ↳ image_seq: lấy thẳng feature đã gộp.

    # -------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._extract(x)          # ↳ Bước 1: ảnh -> feature.
        seq = self.adapter(feats)  # (1, L, D)  ↳ Bước 2: feature -> chuỗi cho memory.

        if self.training and self.reset_mode in ("task", "never"):
            # nối ký ức: xuất phát từ state hiện tại, LƯU state mới (đã detach — luật 1)
            out, new_state = self.memory(seq, state=self._state)  # ↳ Đọc + tự ghi ký ức, có nối tiếp state cũ.
            if not torch.isfinite(out).all() or not state_isfinite(new_state):
                raise FloatingPointError(
                    "Titans tạo NaN/Inf trong output hoặc memory state ngay tại forward."
                )
            if new_state is not None:
                self._state = detach_state(new_state)  # ↳ LUẬT 1: cắt gradient nhưng giữ giá trị ký ức.
        else:
            # eval, hoặc mode "image": xuất phát từ BẢN SAO (luật 2) / từ trắng; không lưu
            init = None
            if (not self.training) and self.reset_mode in ("task", "never") and self._state is not None:
                init = clone_state(self._state)  # ↳ LUẬT 2: eval trên BẢN SAO ký ức, không làm bẩn state thật.
            out, _ = self.memory(seq, state=init)  # ↳ Bỏ qua state mới (dấu "_") -> không ghi khi eval.
            if not torch.isfinite(out).all():
                raise FloatingPointError("Titans tạo NaN/Inf trong output ngay tại forward.")

        h = self.post_norm(self.adapter.restore(out) + self.adapter.restore(seq))  # residual
        # ↳ Bước 3: gấp chuỗi về (B,D); cộng residual (đầu vào + đầu ra memory) rồi chuẩn hoá -> ổn định, đỡ mất tín hiệu gốc.
        return self.head(h)  # ↳ Bước 4: (B,D) -> (B, num_classes) logits.

    # ------------------------------------------------------- state lifecycle
    def reset_state(self) -> None:
        self._state = None  # ↳ Xoá ký ức (gọi khi sang task mới ở chế độ "task").

    def state_norm(self) -> float:
        return state_norm(self._state)  # ↳ Độ lớn ký ức hiện tại -> log để phát hiện phình/nổ.

    def state_isfinite(self) -> bool:
        return state_isfinite(self._state)

    def export_state(self):
        """State (CPU) để torch.save cùng checkpoint — S10."""
        return None if self._state is None else state_to_cpu(self._state)  # ↳ Kéo về CPU trước khi lưu file.

    def import_state(self, state) -> None:
        self._state = state  # ↳ Nạp lại ký ức từ checkpoint.

    def extra_floats(self) -> int:
        """Chi phí thêm: tham số memory + head + state hiện tại (so với backbone trần)."""
        n = self.memory.extra_floats() + sum(p.numel() for p in self.head.parameters())  # ↳ Tham số memory + head.
        return int(n + count_floats(self._state))  # ↳ Cộng luôn kích thước ký ức đang giữ.
