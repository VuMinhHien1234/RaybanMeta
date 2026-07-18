"""CMS tier mapping (G3, task S3) — chia backbone ViT thành các tầng tần số.

Quyết định team (2026-07-14):
- tiers mặc định [[4,1],[4,4],[4,16]] (fast -> slow) cho ViT 12 block;
- attention + LayerNorm -> TẦNG CHẬM NHẤT (`attn: slow`); đường lùi `attn: freeze`;
- patch_embed / pos_embed / cls_token / norm cuối: ĐÓNG BĂNG (nền pretrained);
- head phân loại: tầng NHANH nhất.

`order` quyết định block nào bền — câu hỏi nghiên cứu phải ablate:
- "late_slow" : block ĐẦU nhanh, block CUỐI chậm (giữ ngữ nghĩa).
- "early_slow": block ĐẦU chậm (giữ đặc trưng thấp), block CUỐI nhanh.

Yêu cầu backbone: kiểu ViT của timm, phơi `backbone.blocks[i].mlp/.attn/.norm1/.norm2`
(vit_tiny/small/base... đều được). Không phải ViT -> báo lỗi rõ.
"""
# ↳ GIẢI THÍCH TỔNG QUAN (ý tưởng CMS = "đa tần số"): não người có phần nhớ nhanh
#   (dễ thay đổi) và phần nhớ chậm (bền). CMS bắt chước: chia các tầng của backbone
#   thành nhóm "nhanh/vừa/chậm". Nhóm NHANH cập nhật mỗi bước; nhóm CHẬM chỉ cập nhật
#   thi thoảng và nhẹ -> kiến thức cũ được giữ (chống quên).
#   File này KHÔNG train gì cả — nó chỉ PHÂN NHÓM tham số (tier) + đóng băng phần nền.
#   Việc "ai bước lúc nào" do CMSOptimizer (optim/cms_optimizer.py) lo.
#   Ký hiệu tier = [n_block, period]: n_block block, cứ `period` bước mới cập nhật 1 lần.
from __future__ import annotations

from typing import Dict, List

ORDERS = ("late_slow", "early_slow")  # ↳ Hai cách gán block nào "chậm" (xem docstring).
ATTN_MODES = ("slow", "freeze")       # ↳ Attention đưa vào tier chậm nhất, hay đóng băng hẳn.


def _blocks_of(model):
    # ↳ Lấy danh sách block Transformer của backbone; kiểm tra đúng là ViF (có .blocks[i].mlp).
    backbone = getattr(model, "backbone", model)  # ↳ model có thể là HOPE (bọc backbone) hoặc backbone trần.
    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) == 0 or not hasattr(blocks[0], "mlp"):
        raise TypeError(
            "CMS cần backbone kiểu ViT của timm (có .blocks[i].mlp) — "
            f"backbone hiện tại: {type(backbone).__name__}. "
            "Đổi backbone.name sang vit_tiny/vit_small... trong config."
        )
    return backbone, blocks


def build_cms_param_groups(model, cms_cfg: dict) -> List[Dict]:
    """Trả về tier groups (fast -> slow) và ĐÓNG BĂNG phần nền. Gọi 1 lần/run."""
    # ↳ Đây là hàm chính: đọc config CMS -> chia tham số thành các tier + đóng băng nền.
    tiers = [list(t) for t in cms_cfg.get("tiers", [[4, 1], [4, 4], [4, 16]])]  # ↳ Mặc định 3 tier cho ViT 12 block.
    etas = list(cms_cfg.get("etas", [1.0, 0.5, 0.1]))  # ↳ Hệ số tốc độ học của từng tier (nhanh học mạnh, chậm học nhẹ).
    order = str(cms_cfg.get("order", "late_slow")).lower()
    attn = str(cms_cfg.get("attn", "slow")).lower()
    if order not in ORDERS:
        raise ValueError(f"cms.order '{order}' không hợp lệ, chọn {ORDERS}")
    if attn not in ATTN_MODES:
        raise ValueError(f"cms.attn '{attn}' không hợp lệ, chọn {ATTN_MODES}")
    if len(etas) != len(tiers):
        raise ValueError(f"cms.etas ({len(etas)}) phải cùng độ dài cms.tiers ({len(tiers)})")

    backbone, blocks = _blocks_of(model)
    n_blocks = len(blocks)
    if sum(t[0] for t in tiers) != n_blocks:  # ↳ Tổng số block khai báo trong tiers phải khớp số block thật.
        raise ValueError(
            f"Tổng block trong cms.tiers = {sum(t[0] for t in tiers)} "
            f"phải bằng số block của backbone = {n_blocks}"
        )

    # --- đóng băng phần nền (embedding vào + norm cuối) ---
    # ↳ Phần "móng nhà" (chia ảnh thành patch, vị trí, norm cuối) giữ nguyên từ pretrained -> đóng băng.
    frozen_names = []
    for name in ("patch_embed", "pos_embed", "cls_token", "norm", "reg_token"):
        obj = getattr(backbone, name, None)
        if obj is None:
            continue                              # ↳ Backbone không có phần này -> bỏ qua.
        params = obj.parameters() if hasattr(obj, "parameters") else [obj]  # ↳ Có thể là module hoặc tensor rời.
        for p in params:
            p.requires_grad_(False)               # ↳ Tắt gradient -> không bao giờ update.
        frozen_names.append(name)

    # --- chia block vào tier (fast -> slow) ---
    idxs = list(range(n_blocks))                  # ↳ [0,1,...,11] theo thứ tự block.
    if order == "early_slow":  # block đầu phải rơi vào tier CHẬM -> duyệt ngược
        idxs = idxs[::-1]                          # ↳ Đảo danh sách để block đầu vào tier chậm.
    names = ["fast", "mid", "slow"] if len(tiers) == 3 else [f"tier{i}" for i in range(len(tiers))]
    groups: List[Dict] = []
    cursor = 0                                    # ↳ Con trỏ chạy dọc idxs.
    for (n, period), eta, name in zip(tiers, etas, names):
        block_ids = sorted(idxs[cursor : cursor + n])  # ↳ Lấy n block cho tier này.
        cursor += n
        params = []
        for i in block_ids:
            params += [p for p in blocks[i].mlp.parameters()]  # ↳ Chỉ đưa phần MLP của block vào tier.
        groups.append(
            {"name": name, "period": int(period), "eta": float(eta),
             "params": params, "blocks": block_ids}  # ↳ 1 tier = tên + chu kỳ + hệ số + tham số + id block.
        )

    # --- attention + norm của MỌI block: vào tier chậm nhất, hoặc đóng băng ---
    # ↳ Attention giữ "cách nhìn quan hệ giữa các vùng ảnh" -> muốn bền -> cho vào tier chậm nhất.
    attn_params = []
    for b in blocks:
        for sub in ("attn", "norm1", "norm2", "ls1", "ls2"):  # ↳ Các thành phần attention + layernorm + layerscale.
            m = getattr(b, sub, None)
            if m is not None and hasattr(m, "parameters"):
                attn_params += list(m.parameters())
    if attn == "freeze":
        for p in attn_params:
            p.requires_grad_(False)               # ↳ Đường lùi: đóng băng hẳn attention.
    else:  # "slow" — quyết định team
        groups[-1]["params"] += attn_params       # ↳ groups[-1] = tier chậm nhất.

    # --- head + (G4) memory Titans + post_norm: tier NHANH nhất ---
    # Memory là tầng tần số cao nhất của HOPE — đúng thứ bậc lý thuyết của paper.
    # ↳ head (quyết định class) + memory Titans + post_norm cần linh hoạt nhất -> tier nhanh (groups[0]).
    for attr in ("head", "memory", "post_norm"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "parameters"):
            groups[0]["params"] += [p for p in m.parameters() if p.requires_grad]  # ↳ Chỉ lấy tham số còn học.

    groups[0]["frozen_base"] = frozen_names  # để tier_report in ra  ↳ Nhét thông tin phụ để in báo cáo.
    groups[0]["attn_mode"] = attn
    groups[0]["order"] = order
    return groups  # ↳ Trả list tier (fast->slow) cho CMSOptimizer dùng.


def tier_report(groups: List[Dict]) -> str:
    """Bảng kiểm bằng mắt: block nào ở tier nào — chạy lần đầu PHẢI đọc (task S3)."""
    # ↳ Chỉ để IN cho người kiểm tra phân nhóm có đúng ý không. Không ảnh hưởng train.
    lines = [
        f"[cms] order={groups[0].get('order')} | attn={groups[0].get('attn_mode')} | "
        f"frozen base: {groups[0].get('frozen_base')}"
    ]
    for g in groups:
        n_params = sum(p.numel() for p in g["params"])  # ↳ Đếm tham số mỗi tier.
        lines.append(
            f"[cms]  {g['name']:<5s} period={g['period']:<3d} eta={g['eta']:<4.2f} "
            f"blocks={g['blocks']} | {n_params:,} params"
        )
    return "\n".join(lines)
