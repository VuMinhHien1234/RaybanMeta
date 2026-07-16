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
from __future__ import annotations

import torch
import torch.nn as nn

from .memory import TitansMemory
from .seq_adapter import SeqAdapter
from .state_utils import clone_state, count_floats, detach_state, state_norm, state_to_cpu

RESET_MODES = ("image", "task", "never")


class TitansClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int, memory_cfg: dict):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():  # G2: backbone LUÔN đóng băng (quyết định đã chốt)
            p.requires_grad_(False)
        self.backbone.eval()

        self.seq_mode = str(memory_cfg.get("seq", "image_seq")).lower()
        self.reset_mode = str(memory_cfg.get("reset", "image")).lower()
        if self.reset_mode not in RESET_MODES:
            raise ValueError(f"memory.reset '{self.reset_mode}' không hợp lệ, chọn {RESET_MODES}")

        dim = memory_cfg.get("dim", "auto")
        dim = int(feat_dim) if dim in (None, "auto") else int(dim)
        if dim != int(feat_dim):
            raise ValueError(f"memory.dim={dim} phải bằng feat_dim={feat_dim} (chưa hỗ trợ lớp chiếu)")

        self.adapter = SeqAdapter(self.seq_mode)
        self.memory = TitansMemory(dim=dim, chunk_size=int(memory_cfg.get("chunk_size", 64)))
        self.post_norm = nn.LayerNorm(dim)  # luật C2: ổn định số sau memory
        self.head = nn.Linear(dim, num_classes)
        self._state = None  # "cục ký ức" hiện tại của stream

    # ---- giữ backbone ở eval kể cả khi model.train() (như NCMClassifier) ----
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    # ------------------------------------------------------------- features
    @torch.no_grad()
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_mode == "token_seq":
            ff = getattr(self.backbone, "forward_features", None)
            if ff is None:
                raise ValueError("token_seq cần backbone timm có forward_features (ViT).")
            feats = ff(x)  # ViT: (B, prefix+P, D)
            n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 0))
            if feats.dim() == 3 and n_prefix > 0:
                feats = feats[:, n_prefix:, :]
            return feats
        return self.backbone(x)  # (B, D) pooled

    # -------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._extract(x)
        seq = self.adapter(feats)  # (1, L, D)

        if self.training and self.reset_mode in ("task", "never"):
            # nối ký ức: xuất phát từ state hiện tại, LƯU state mới (đã detach — luật 1)
            out, new_state = self.memory(seq, state=self._state)
            if new_state is not None:
                self._state = detach_state(new_state)
        else:
            # eval, hoặc mode "image": xuất phát từ BẢN SAO (luật 2) / từ trắng; không lưu
            init = None
            if (not self.training) and self.reset_mode in ("task", "never") and self._state is not None:
                init = clone_state(self._state)
            out, _ = self.memory(seq, state=init)

        h = self.post_norm(self.adapter.restore(out) + self.adapter.restore(seq))  # residual
        return self.head(h)

    # ------------------------------------------------------- state lifecycle
    def reset_state(self) -> None:
        self._state = None

    def state_norm(self) -> float:
        return state_norm(self._state)

    def export_state(self):
        """State (CPU) để torch.save cùng checkpoint — S10."""
        return None if self._state is None else state_to_cpu(self._state)

    def import_state(self, state) -> None:
        self._state = state

    def extra_floats(self) -> int:
        """Chi phí thêm: tham số memory + head + state hiện tại (so với backbone trần)."""
        n = self.memory.extra_floats() + sum(p.numel() for p in self.head.parameters())
        return int(n + count_floats(self._state))
