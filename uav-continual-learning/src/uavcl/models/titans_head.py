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

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import TitansMemory
from .seq_adapter import SeqAdapter
from .state_utils import (
    clone_state,
    count_floats,
    detach_state,
    repeat_state_batch,
    state_isfinite,
    state_norm,
    state_to_cpu,
)

RESET_MODES = ("image", "task", "never")  # ↳ 3 chế độ giữ ký ức hợp lệ (xem docstring).
FEATURE_PROTOCOLS = ("stream_batch_legacy", "independent_image")


@dataclass(frozen=True)
class FeatureBundle:
    """Stable backbone and plastic post-memory features from one forward."""

    base: torch.Tensor
    titans: torch.Tensor


def blend_features(base: torch.Tensor, titans: torch.Tensor, gamma: float) -> torch.Tensor:
    """Interpolate stable ViT and plastic Titans directions for cosine NCM."""
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("blend gamma phải nằm trong [0, 1]")
    if base.shape != titans.shape:
        raise ValueError(
            f"base/titans feature phải cùng shape, nhận {tuple(base.shape)} và "
            f"{tuple(titans.shape)}"
        )
    if gamma == 0.0:
        return base
    if gamma == 1.0:
        return titans
    return F.normalize(
        (1.0 - gamma) * F.normalize(base, dim=-1)
        + gamma * F.normalize(titans, dim=-1),
        dim=-1,
    )

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
# TASK 2 (η/α data-dependent, NL.pdf Eq 76): titans-pytorch ĐÃ tự tính η_t (to_adaptive_step)
# và α_t forget (to_decay_factor) TỪ INPUT theo mặc định -> phần Eq 76 coi như có sẵn. Cờ thêm
# duy nhất là per_parameter_lr_modulation: cho "mạng ngoài" điều tiết lr theo TỪNG ma trận memory
# (một tầng self-modifying nhẹ nữa). Mặc định tắt -> bật qua memory.per_parameter_lr_modulation.
_STABILITY_KEYS = ("gated_transition", "spectral_norm_surprises", "qk_rmsnorm",
                   "per_parameter_lr_modulation")  # ↳ các cờ dạng bật/tắt truyền thẳng xuống NeuralMemory.


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
        # ---- BẬC 1 "Deep Self-Referential Titans" (NL.pdf §8.1): làm bộ nhớ SÂU hơn ----
        # Paper nhấn chữ "Deep": memory MLP càng sâu / càng nhiều đầu -> biểu đạt càng mạnh,
        # đúng tinh thần "nhiều tầng" của Nested Learning. Mặc định depth=2, heads=1 = GIỮ
        # NGUYÊN hành vi cũ (không thêm kwarg nào khi config vắng key -> mọi run cũ bất biến).
        depth = int(memory_cfg.get("depth", 2))      # ↳ số lớp MLP của memory model (2 = default titans-pytorch).
        if depth != 2:
            # giữ expansion_factor=4.0 như default_model_kwargs của titans-pytorch, chỉ đổi depth.
            stab_kwargs["default_model_kwargs"] = dict(depth=depth, expansion_factor=4.0)
        heads = int(memory_cfg.get("heads", 1))      # ↳ số đầu memory song song (1 = default).
        if heads > 1:
            if dim % heads != 0:                     # ↳ giữ dim_inner = dim: mỗi đầu dim_head = dim/heads.
                raise ValueError(f"memory.heads={heads} phải chia hết memory dim={dim}")
            stab_kwargs["heads"] = heads
            stab_kwargs["dim_head"] = dim // heads
        self_ref = bool(memory_cfg.get("self_referential", False))  # ↳ TASK 3 (NL §8.1 Eq 79): mặc định tắt.
        self_mod = bool(memory_cfg.get("self_modifying", False))    # ↳ TASK 4 (NL §8.1 cuối): value tự sinh theo M_{t-1}.
        self_mod_rp = bool(memory_cfg.get("self_modifying_readpath", False))  # ↳ hướng 1 (k/q); mặc định tắt (v2 tốt hơn).
        self.memory = TitansMemory(
            dim=dim, chunk_size=int(memory_cfg.get("chunk_size", 64)),
            self_referential=self_ref, self_modifying=self_mod,
            self_modifying_readpath=self_mod_rp, **stab_kwargs  # ↳ self-mod bao trùm self-ref.
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

    # ------------------------------------------------------------- features
    def _feature_components_stream_batch(self, x: torch.Tensor) -> FeatureBundle:
        """Feature SAU memory (vector h trước head) — (B, D). Dùng cho head Linear và cho
        NCM-head (đòn A): đọc chính feature đã-được-memory-làm-giàu bằng prototype thay vì Linear.
        Giữ NGUYÊN luật vòng đời state: train thì nối+ghi (detach); eval thì đọc BẢN SAO, không ghi."""
        feats = self._extract(x)          # ↳ Bước 1: ảnh -> feature.
        seq = self.adapter(feats)  # (1, L, D)  ↳ Bước 2: feature -> chuỗi cho memory.

        if self.training and self.reset_mode in ("task", "never"):
            # nối ký ức: xuất phát từ state hiện tại, LƯU state mới (đã detach — luật 1)
            out, new_state = self.memory(seq, state=self._state)  # ↳ Đọc + tự ghi ký ức, có nối tiếp state cũ.
            if new_state is not None:
                self._state = detach_state(new_state)  # ↳ LUẬT 1: cắt gradient nhưng giữ giá trị ký ức.
        else:
            # eval, hoặc mode "image": xuất phát từ BẢN SAO (luật 2) / từ trắng; không lưu
            init = None
            if (not self.training) and self.reset_mode in ("task", "never") and self._state is not None:
                init = clone_state(self._state)  # ↳ LUẬT 2: eval trên BẢN SAO ký ức, không làm bẩn state thật.
            out, _ = self.memory(seq, state=init)  # ↳ Bỏ qua state mới (dấu "_") -> không ghi khi eval.

        residual = self.adapter.restore(seq)
        titans = self.post_norm(self.adapter.restore(out) + residual)
        return FeatureBundle(base=residual, titans=titans)
        # ↳ Gấp chuỗi về (B,D); cộng residual (đầu vào + đầu ra memory) rồi chuẩn hoá -> ổn định, đỡ mất tín hiệu gốc.

    def _features_stream_batch(self, x: torch.Tensor) -> torch.Tensor:
        return self._feature_components_stream_batch(x).titans

    def feature_components(
        self, x: torch.Tensor, protocol: str = "stream_batch_legacy"
    ) -> FeatureBundle:
        """Return base and post-memory features from one state transition."""
        protocol = str(protocol).lower()
        if protocol not in FEATURE_PROTOCOLS:
            raise ValueError(f"feature protocol '{protocol}' không hợp lệ, chọn {FEATURE_PROTOCOLS}")
        if protocol == "stream_batch_legacy":
            return self._feature_components_stream_batch(x)
        if self.training:
            raise RuntimeError("independent_image chỉ dùng khi model ở eval mode")
        if x.shape[0] == 0:
            raise ValueError("independent_image không nhận batch rỗng")

        feats = self._extract(x)
        if self.seq_mode == "image_seq":
            seq = feats.unsqueeze(1)
        else:
            if feats.ndim != 3:
                raise ValueError(
                    f"token_seq independent cần (B,P,D), nhận {tuple(feats.shape)}"
                )
            seq = feats

        init = None
        if self.reset_mode in ("task", "never") and self._state is not None:
            init = repeat_state_batch(self._state, x.shape[0])
        out, _ = self.memory(seq, state=init)
        if self.seq_mode == "image_seq":
            memory_features = out.squeeze(1)
            residual = seq.squeeze(1)
        else:
            memory_features = out.mean(dim=1)
            residual = seq.mean(dim=1)
        return FeatureBundle(
            base=residual,
            titans=self.post_norm(memory_features + residual),
        )

    def features(
        self,
        x: torch.Tensor,
        protocol: str = "stream_batch_legacy",
        blend_gamma: float = 1.0,
    ) -> torch.Tensor:
        """Extract post-memory features under an explicit image-evaluation protocol.

        ``stream_batch_legacy`` preserves historical behavior where the images in one
        DataLoader batch form a temporal sequence. ``independent_image`` evaluates
        every image from the same memory-state snapshot, so other images and batch
        boundaries cannot alter its feature.
        """
        bundle = self.feature_components(x, protocol=protocol)
        return blend_features(bundle.base, bundle.titans, blend_gamma)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, FeatureBundle]:
        """Training forward exposing features without updating memory twice."""
        bundle = self.feature_components(x, protocol="stream_batch_legacy")
        return self.head(bundle.titans), bundle

    # -------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))  # ↳ (B,D) feature sau memory -> (B, num_classes) logits.

    # ------------------------------------------------------- state lifecycle
    def reset_state(self) -> None:
        self._state = None  # ↳ Xoá ký ức (gọi khi sang task mới ở chế độ "task").

    def state_norm(self) -> float:
        return state_norm(self._state)  # ↳ Độ lớn ký ức hiện tại -> log để phát hiện phình/nổ.

    def state_isfinite(self) -> bool:
        return state_isfinite(self._state)

    def reset_eta_alpha(self) -> None:
        """Reset bộ đếm η_t/α_t (gọi ở begin_task) — vá lỗ hổng Task 2."""
        if hasattr(self.memory, "reset_eta_alpha"):
            self.memory.reset_eta_alpha()

    def eta_alpha_stats(self):
        """(η_t, α_t) trung bình trong task — η lớn / α~1 = ghi hung/không quên = hướng NỔ."""
        return self.memory.eta_alpha_stats() if hasattr(self.memory, "eta_alpha_stats") else (None, None)

    def self_mod_stats(self):
        """{q|k|v: (β, ‖W_state‖)} của các nhánh self-modifying (TASK 4 + hướng 1), else None — để log."""
        mem = getattr(self.memory, "mem", None)             # ↳ NeuralMemory bên trong TitansMemory.
        if mem is None:
            return None
        out = {}
        for tag, attr in (("q", "to_queries"), ("k", "to_keys"), ("v", "to_values")):
            proj = getattr(mem, attr, None)
            if proj is not None and hasattr(proj, "branch_strength"):
                out[tag] = proj.branch_strength()           # (β, ‖W_state‖)
        return out or None

    def export_state(self):
        """State (CPU) để torch.save cùng checkpoint — S10."""
        return None if self._state is None else state_to_cpu(self._state)  # ↳ Kéo về CPU trước khi lưu file.

    def import_state(self, state) -> None:
        self._state = state  # ↳ Nạp lại ký ức từ checkpoint.

    def extra_floats(self) -> int:
        """Chi phí thêm: tham số memory + head + state hiện tại (so với backbone trần)."""
        n = self.memory.extra_floats() + sum(p.numel() for p in self.head.parameters())  # ↳ Tham số memory + head.
        return int(n + count_floats(self._state))  # ↳ Cộng luôn kích thước ký ức đang giữ.
