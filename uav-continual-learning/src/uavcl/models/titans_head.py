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

from .classifier import build_head
from .memory import TitansMemory
from .seq_adapter import SeqAdapter
from .state_utils import clone_state, count_floats, detach_state, state_norm, state_to_cpu

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
# TASK 2 (η/α data-dependent, NL.pdf Eq 76): titans-pytorch ĐÃ tự tính η_t (to_adaptive_step)
# và α_t forget (to_decay_factor) TỪ INPUT theo mặc định -> phần Eq 76 coi như có sẵn. Cờ thêm
# duy nhất là per_parameter_lr_modulation: cho "mạng ngoài" điều tiết lr theo TỪNG ma trận memory
# (một tầng self-modifying nhẹ nữa). Mặc định tắt -> bật qua memory.per_parameter_lr_modulation.
_STABILITY_KEYS = ("gated_transition", "spectral_norm_surprises", "qk_rmsnorm",
                   "per_parameter_lr_modulation")  # ↳ các cờ dạng bật/tắt truyền thẳng xuống NeuralMemory.

# TASK 5 (2026-08-02) — khởi tạo cổng TRUNG TÍNH. Thư viện đã hỗ trợ sẵn (neural_memory.py:526-540):
# đặt bias hằng số + zero trọng số Linear -> cổng khởi đầu là hằng số ở giữa dải, rồi mới học dần
# thành phụ thuộc dữ liệu. Giá trị 0.0 -> sigmoid(0)=0.5 -> α khởi đầu 0.5, η khởi đầu 0.5*max_lr.
# LƯU Ý: cách này KHÔNG tự ngăn trôi về sau (trọng số vẫn lớn dần) — nó bổ trợ cho `gate_bound`.
_INIT_BIAS_KEYS = ("init_adaptive_step_bias", "init_decay_bias", "init_momentum_bias")


def _stability_kwargs(memory_cfg: dict) -> dict:
    # ↳ Đọc các cờ ổn định từ config -> gom thành dict để truyền xuống TitansMemory.
    #   Key nào KHÔNG có trong config thì bỏ qua -> giữ đúng mặc định thư viện (an toàn ngược).
    kw = {}
    for k in _STABILITY_KEYS:
        if k in memory_cfg:
            kw[k] = bool(memory_cfg[k])              # ↳ 3 cờ boolean.
    for k in _INIT_BIAS_KEYS:                        # ↳ TASK 5: 3 bias khởi tạo (số thực).
        if memory_cfg.get(k) is not None:
            kw[k] = float(memory_cfg[k])
    if memory_cfg.get("max_grad_norm") is not None:
        kw["max_grad_norm"] = float(memory_cfg["max_grad_norm"])  # ↳ Ngưỡng clip (số thực).
    return kw


class TitansClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int, memory_cfg: dict,
                 head: str = "linear"):
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
            self_modifying_readpath=self_mod_rp,
            gate_bound=memory_cfg.get("gate_bound"),  # ↳ TASK 4: chặn trôi cổng η/α (mặc định None = tắt).
            **stab_kwargs  # ↳ self-mod bao trùm self-ref.
        )
        # G1 (2026-08-03) — in CẢ KHI TẮT. Bản cũ chỉ in khi bật, nên muốn biết bản vá có chạy
        # không thì phải để ý sự VẮNG MẶT của một dòng log — tín hiệu quá yếu. Giờ log luôn có
        # đúng một dòng khẳng định, đọc là biết ngay.
        _gb = self.memory.gate_bound_report() if hasattr(self.memory, "gate_bound_report") else None
        print(f"[titans] gate_bound {'BẬT — ' + _gb if _gb else 'TẮT (config không khai báo)'}")
        # A1 (2026-08-04, KE_HOACH_SUA) — post_norm giờ TẮT ĐƯỢC qua config. Vì sao: D11 §2.1
        # phát hiện đổi trần η gấp 100 lần mà Δacc = 0,00002; nghi phạm số 1 là chính LayerNorm
        # này — nó chuẩn hoá lại thang đo NGAY SAU memory nên có thể triệt tiêu độ lớn bước ghi
        # mà η điều khiển. Ablation Ưu tiên 2: chạy với `memory.post_norm: false`, 3 seed.
        # Mặc định BẬT (đúng hành vi cũ) -> mọi run trước đó bất biến. Identity giữ nguyên
        # đường gọi ở features_from_extracted, không phải sửa chỗ nào khác.
        _pn = bool(memory_cfg.get("post_norm", True))
        self.post_norm = nn.LayerNorm(dim) if _pn else nn.Identity()  # luật C2 (khi bật): ổn định số sau memory
        print(f"[titans] post_norm {'BẬT (mặc định — LayerNorm sau memory)' if _pn else 'TẮT — ablation D11 Ưu tiên 2'}")
        # TASK 4 (bổ trợ) — chuẩn hoá TRƯỚC memory. Hai cổng η/α là nn.Linear áp THẲNG lên feature
        # ViT thô (neural_memory.py:451, :514); không có chuẩn hoá nào ở phía trước -> logit dễ lớn.
        # `qk_rmsnorm` KHÔNG cứu được vì nó chỉ chạm q/k trong đường đọc/ghi, không chạm 2 cổng này.
        # Mặc định TẮT -> run cũ bất biến. Đây là thuốc BỔ TRỢ: nó sửa thang đo lúc khởi tạo, còn
        # đà TRÔI do gradient thì phải dùng `gate_bound`.
        self.pre_norm = nn.LayerNorm(dim) if bool(memory_cfg.get("pre_norm", False)) else None
        self.head = build_head(head, dim, num_classes)  # ↳ linear (mặc định) | cosine (chống recency bias).
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
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature SAU memory (vector h trước head) — (B, D). Dùng cho head Linear và cho
        NCM-head (đòn A): đọc chính feature đã-được-memory-làm-giàu bằng prototype thay vì Linear.
        Giữ NGUYÊN luật vòng đời state: train thì nối+ghi (detach); eval thì đọc BẢN SAO, không ghi."""
        return self.features_from_extracted(self._extract(x))  # ↳ Bước 1: ảnh -> feature; phần sau tách riêng.

    def features_from_extracted(self, feats: torch.Tensor) -> torch.Tensor:
        """Đường SAU backbone: feature đã trích -> adapter -> memory -> post_norm (B, D).

        Tách riêng (task #22, plans/TASKS_UAV_CL.md) để latent replay đưa feature CŨ đã lưu
        đi lại qua memory+head mà không cần ảnh gốc. Luật vòng đời state giữ nguyên như features()."""
        seq = self.adapter(feats)  # (1, L, D)  ↳ Bước 2: feature -> chuỗi cho memory.
        # TASK 4: chỉ ĐẦU VÀO memory được chuẩn hoá; residual phía dưới vẫn dùng `seq` THÔ
        # (thay đổi tối thiểu — đường tín hiệu gốc không bị đụng, post_norm đã lo chênh thang đo).
        seq_in = self.pre_norm(seq) if self.pre_norm is not None else seq

        if self.training and self.reset_mode in ("task", "never"):
            # nối ký ức: xuất phát từ state hiện tại, LƯU state mới (đã detach — luật 1)
            out, new_state = self.memory(seq_in, state=self._state)  # ↳ Đọc + tự ghi ký ức, có nối tiếp state cũ.
            if new_state is not None:
                self._state = detach_state(new_state)  # ↳ LUẬT 1: cắt gradient nhưng giữ giá trị ký ức.
        else:
            # eval, hoặc mode "image": xuất phát từ BẢN SAO (luật 2) / từ trắng; không lưu
            init = None
            if (not self.training) and self.reset_mode in ("task", "never") and self._state is not None:
                init = clone_state(self._state)  # ↳ LUẬT 2: eval trên BẢN SAO ký ức, không làm bẩn state thật.
            out, _ = self.memory(seq_in, state=init)  # ↳ Bỏ qua state mới (dấu "_") -> không ghi khi eval.

        return self.post_norm(self.adapter.restore(out) + self.adapter.restore(seq))  # residual + chuẩn hoá
        # ↳ Gấp chuỗi về (B,D); cộng residual (đầu vào + đầu ra memory) rồi chuẩn hoá -> ổn định, đỡ mất tín hiệu gốc.

    # -------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))  # ↳ (B,D) feature sau memory -> (B, num_classes) logits.

    def forward_from_feats(self, feats: torch.Tensor) -> torch.Tensor:
        """Logits từ feature-sau-backbone đã lưu (latent replay #22): bỏ qua backbone."""
        return self.head(self.features_from_extracted(feats))

    # ------------------------------------------------------- state lifecycle
    def reset_state(self) -> None:
        self._state = None  # ↳ Xoá ký ức (gọi khi sang task mới ở chế độ "task").

    def state_norm(self) -> float:
        return state_norm(self._state)  # ↳ Độ lớn ký ức hiện tại -> log để phát hiện phình/nổ.

    def reset_eta_alpha(self) -> None:
        """Reset bộ đếm η_t/α_t (gọi ở begin_task) — vá lỗ hổng Task 2."""
        if hasattr(self.memory, "reset_eta_alpha"):
            self.memory.reset_eta_alpha()

    def eta_alpha_stats(self):
        """(η logit thô, η THẬT, α_t) trung bình trong task.

        η THẬT = sigmoid(logit)*max_lr (xem memory.py). α = decay_factor; GIỮ LẠI = (1−α):
        α→0 = không quên gì = hướng NỔ norm; α→1 = xoá sạch = memory không tích luỹ.
        """
        return self.memory.eta_alpha_stats() if hasattr(self.memory, "eta_alpha_stats") else (None, None, None)

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
