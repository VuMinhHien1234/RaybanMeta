#!/usr/bin/env python3
"""T1 + T2 — CỬA CHẶN: điều kiện quan sát có phải một TRỤC CHUNG không?

Câu hỏi quyết định cả kiến trúc, trả lời trong 15 phút, trước khi viết dòng code nào:

  Khi điều kiện đổi (nắng -> chiều muộn, khô -> sương), feature của 45 lớp có dịch
  CÙNG MỘT HƯỚNG không?

  CÓ  -> "điều kiện" là một trục chung, ước lượng được KHÔNG CẦN NHÃN chỉ bằng vài số.
         Căn chỉnh tuyến tính đủ dùng -> bộ nhớ nhiều tầng bằng công thức đóng (rẻ 67 lần).
  KHÔNG -> không tồn tại "điều kiện toàn cục" để căn chỉnh. Phải dùng bộ nhớ phi tuyến
         học được (Titans), dù đắt.

T2 kiểm cái bẫy nguy hiểm nhất của hướng này (giả định A5 trong BAI_TOAN_VA_MUC_TIEU):
tầng nhanh định theo dõi trung bình feature toàn cục `m_t`. Nhưng `m_t` dịch vì HAI lý do:
điều kiện đổi, VÀ drone bay từ thành phố sang rừng làm tỷ lệ lớp đổi. Lẫn hai thứ đó thì
hệ sẽ "sửa" luôn thông tin lớp và làm hỏng phân loại.

Dùng:
    python scripts/do_truc_dieu_kien.py --config configs/drift_slda_arm3_dexuat.yaml
    python scripts/do_truc_dieu_kien.py --config ... --n-moi-lop 40 --json t1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavcl.data.drift import ApDungTroi                                # noqa: E402
from uavcl.data.loaders import IMAGENET_MEAN, IMAGENET_STD             # noqa: E402
from uavcl.data.sources import get_source                              # noqa: E402
from uavcl.models.backbone import build_backbone                       # noqa: E402
from uavcl.utils.config import load_config                             # noqa: E402

# Ngưỡng ĐẠT — xem bảng quyết định ở cuối file.
COS_TOT, COS_KEM = 0.80, 0.30
PC1_TOT = 0.70


def _tf(image_size, muc, seed=1234):
    from torchvision import transforms as T
    return T.Compose(
        [T.Resize((image_size, image_size)), T.ToTensor()]
        + ([ApDungTroi(muc, jitter=0.0, seed=seed)] if muc > 0 else [])   # jitter=0: so sánh sạch
        + [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


@torch.no_grad()
def _mu_theo_lop(backbone, split, idx_theo_lop, image_size, muc, device, bs=64):
    """Trả về (C, D): trung bình feature của từng lớp, dưới MỘT mức trôi cố định."""
    tf = _tf(image_size, muc)
    mus = []
    for idxs in idx_theo_lop:
        F = []
        for i in range(0, len(idxs), bs):
            x = torch.stack([tf(split.get_image(j)) for j in idxs[i:i + bs]]).to(device)
            F.append(backbone(x).float().cpu())
        mus.append(torch.cat(F).mean(0))
    return torch.stack(mus).numpy()


def _phan_tich_truc(V):
    """V: (C, D) các vector dịch. Trả về (cos trung bình từng đôi, tỉ lệ PC1, trục đơn vị)."""
    N = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    C = len(N)
    cos = N @ N.T
    cos_tb = float((cos.sum() - C) / (C * (C - 1)))          # bỏ đường chéo
    # PCA không trừ trung bình: ta muốn biết "một hướng chung" giải thích được bao nhiêu,
    # chứ không phải biến thiên quanh trung bình.
    _, S, Vt = np.linalg.svd(V, full_matrices=False)
    ti_le_pc1 = float(S[0] ** 2 / (S ** 2).sum())
    truc = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    if float(V.mean(0) @ truc) < 0:
        truc = -truc                                          # hướng theo chiều trôi tăng
    return cos_tb, ti_le_pc1, truc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-moi-lop", type=int, default=40, help="số ảnh mỗi lớp để ước lượng μ_c")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    d = cfg["data"]
    image_size = int(d.get("image_size", 224))
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
           else "cpu")

    source = get_source(d)
    split = source.splits["train"]
    C = source.num_classes
    backbone, _ = build_backbone(cfg["backbone"])
    backbone = backbone.to(dev).eval()

    # cùng số ảnh mỗi lớp -> μ_c không bị lệch vì lớp nhiều/ít mẫu
    g = np.random.default_rng(0)
    idx_theo_lop = []
    for c in range(C):
        ids = [i for i, y in enumerate(split.labels) if int(y) == c]
        idx_theo_lop.append(list(g.permutation(ids)[:a.n_moi_lop]))

    print(f"\nthiết bị={dev} · {C} lớp × {a.n_moi_lop} ảnh · D={backbone(torch.zeros(1,3,image_size,image_size).to(dev)).shape[1]}")
    print("\nđang trích feature ở trôi 0% ...", flush=True)
    mu0 = _mu_theo_lop(backbone, split, idx_theo_lop, image_size, 0.0, dev)
    print("đang trích feature ở trôi 100% ...", flush=True)
    mu1 = _mu_theo_lop(backbone, split, idx_theo_lop, image_size, 1.0, dev)

    V = mu1 - mu0                                             # (C, D) vector dịch từng lớp
    cos_tb, pc1, truc = _phan_tich_truc(V)
    do_lon = np.linalg.norm(V, axis=1)

    # ================================ T1 ================================
    print("\n" + "=" * 68)
    print("T1 — TRÔI CÓ PHẢI MỘT TRỤC CHUNG KHÔNG?")
    print("=" * 68)
    print(f"  cosine trung bình giữa các vector dịch : {cos_tb:.4f}")
    print(f"  PC1 giải thích được                    : {pc1 * 100:.1f}% phương sai")
    print(f"  độ dài vector dịch                     : {do_lon.mean():.3f} ± {do_lon.std():.3f}")
    print(f"  khoảng cách giữa các lớp (tham chiếu)  : {np.linalg.norm(mu0 - mu0.mean(0), axis=1).mean():.3f}")

    if cos_tb >= COS_TOT and pc1 >= PC1_TOT:
        ket_luan_t1 = "TRUC_CHUNG"
        print(f"\n  ✅ ĐIỀU KIỆN = MỘT TRỤC CHUNG (cos ≥ {COS_TOT}, PC1 ≥ {PC1_TOT:.0%})")
        print("     -> căn chỉnh tuyến tính không nhãn LÀ ĐỦ. Bộ nhớ ba tầng công thức đóng.")
        print("     -> Titans là THỪA cho phần thích nghi điều kiện.")
    elif cos_tb < COS_KEM:
        ket_luan_t1 = "KHONG_CO_TRUC"
        print(f"\n  ❌ KHÔNG CÓ TRỤC CHUNG (cos < {COS_KEM})")
        print("     -> mỗi lớp phản ứng với điều kiện một kiểu; không có 'điều kiện toàn cục'.")
        print("     -> căn chỉnh tuyến tính BẤT KHẢ THI. Phải dùng bộ nhớ phi tuyến (Titans).")
    else:
        ket_luan_t1 = "MOT_PHAN"
        print(f"\n  ⚠️  CÓ TRỤC CHUNG NHƯNG KHÔNG TRỌN ({COS_KEM} ≤ cos < {COS_TOT})")
        print("     -> tầng nhanh xử được phần chung; phần riêng theo lớp phải giao cho tầng trung.")

    # ================================ T2 ================================
    # Bẫy: đổi TỶ LỆ LỚP (bay thành phố -> rừng) có giả dạng đổi ĐIỀU KIỆN không?
    # Mô phỏng bằng cách lấy trung bình có trọng số lệch trên μ_c, KHÔNG đụng tới mức trôi.
    print("\n" + "=" * 68)
    print("T2 — ĐỔI TỶ LỆ LỚP CÓ GIẢ DẠNG ĐỔI ĐIỀU KIỆN KHÔNG?")
    print("=" * 68)
    rng = np.random.default_rng(7)
    dich_dk = float(np.abs(V.mean(0) @ truc))                 # dịch do ĐIỀU KIỆN, chiếu lên trục
    chieu, tong = [], []
    for _ in range(20):
        w = rng.dirichlet(np.full(C, 0.15))                   # tỷ lệ lớp rất lệch
        dm = w @ mu0 - mu0.mean(0)                            # dịch m_t do TỶ LỆ LỚP, cùng điều kiện
        chieu.append(abs(float(dm @ truc)))
        tong.append(float(np.linalg.norm(dm)))
    chieu, tong = np.array(chieu), np.array(tong)
    ty_le = chieu.mean() / max(dich_dk, 1e-12)

    print(f"  dịch do ĐIỀU KIỆN (chiếu lên trục)     : {dich_dk:.4f}")
    print(f"  dịch do TỶ LỆ LỚP (tổng độ dài)        : {tong.mean():.4f} ± {tong.std():.4f}")
    print(f"  dịch do TỶ LỆ LỚP (chiếu lên trục)     : {chieu.mean():.4f} ± {chieu.std():.4f}")
    print(f"  tỷ lệ nhiễu/tín hiệu trên trục          : {ty_le:.3f}")
    print(f"  phần tỷ lệ lớp bị lọc bỏ khi chiếu      : {(1 - chieu.mean() / max(tong.mean(),1e-12)) * 100:.1f}%")

    if ty_le < 0.3:
        ket_luan_t2 = "CHIEU_LA_DU"
        print(f"\n  ✅ CHIẾU LÊN TRỤC ĐIỀU KIỆN LÀ ĐỦ để tách (nhiễu chỉ {ty_le:.0%} tín hiệu)")
        print("     -> tầng nhanh chỉ cần theo dõi 1 số vô hướng: hình chiếu của m_t lên trục.")
    else:
        ket_luan_t2 = "CAN_CAN_BANG_LOP"
        print(f"\n  ⚠️  TỶ LỆ LỚP GÂY NHIỄU MẠNH ({ty_le:.0%} tín hiệu)")
        print("     -> phải cân bằng lớp bằng nhãn giả, hoặc dùng thống kê bậc hai (ít nhạy hơn).")

    print("\n" + "=" * 68)
    print(f"  KẾT LUẬN: T1={ket_luan_t1}  T2={ket_luan_t2}")
    print("=" * 68 + "\n")

    if a.json:
        Path(a.json).write_text(json.dumps({
            "cos_trung_binh": cos_tb, "pc1": pc1, "ket_luan_t1": ket_luan_t1,
            "dich_dieu_kien": dich_dk, "dich_ty_le_lop_chieu": float(chieu.mean()),
            "ty_le_nhieu_tin_hieu": float(ty_le), "ket_luan_t2": ket_luan_t2,
            "truc_dieu_kien": truc.tolist(),
        }, indent=2), encoding="utf-8")
        print(f"  đã lưu {a.json} — trục điều kiện dùng lại cho tầng nhanh (M1)\n")


if __name__ == "__main__":
    main()
