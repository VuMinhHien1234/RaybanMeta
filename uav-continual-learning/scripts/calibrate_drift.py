#!/usr/bin/env python3
"""D8/D9 — HIỆU CHỈNH cường độ trôi TRƯỚC khi đốt giờ máy.

Vì sao bắt buộc: nếu trôi quá nhẹ thì mọi arm λ đều như nhau (không có gì để đo); nếu quá
nặng thì feature ViT vỡ hẳn và mọi phương pháp đều chết (cũng không phân biệt được). Chỉ
vùng giữa mới cho thí nghiệm có ý nghĩa.

Bài học rút ra từ smoke 3-task lần trước: **kiểm setup 15 phút, tránh mất 8 tiếng.**

Cách đo: chạy NCM trên feature ĐÓNG BĂNG (không train gì) ở từng mức trôi, xem accuracy tụt
bao nhiêu. NCM rẻ và là cận dưới — phương pháp nào cũng ít nhất bằng nó.

Dùng:
    python scripts/calibrate_drift.py --config configs/drift_slda_arm3_dexuat.yaml
    python scripts/calibrate_drift.py --config ... --severity 0.6      # thử mức khác
    python scripts/calibrate_drift.py --config ... --dump-samples troi.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavcl.data.drift import ApDungTroi, muc_troi                      # noqa: E402
from uavcl.data.loaders import IMAGENET_MEAN, IMAGENET_STD             # noqa: E402
from uavcl.data.sources import get_source                              # noqa: E402
from uavcl.models.backbone import build_backbone                       # noqa: E402
from uavcl.utils.config import load_config                             # noqa: E402

# Ngưỡng ĐẠT: ở mức trôi 100%, NCM phải tụt vào vùng này.
DAT_LO, DAT_HI = 0.45, 0.60


def _tf(image_size, m, jitter, seed):
    from torchvision import transforms as T
    return T.Compose(
        [T.Resize((image_size, image_size)), T.ToTensor()]
        + ([ApDungTroi(m, jitter=jitter, seed=seed)] if m > 0 else [])
        + [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


@torch.no_grad()
def _ncm_acc(backbone, fit_split, test_split, idx_fit, idx_test,
             image_size, m, jitter, device, num_classes, bs=64):
    """Dựng prototype từ idx_fit rồi đo trên idx_test — CÙNG mức trôi m."""
    tf = _tf(image_size, m, jitter, seed=1234)

    def feats(split, idxs):
        F, Y = [], []
        for i in range(0, len(idxs), bs):
            lo = idxs[i:i + bs]
            x = torch.stack([tf(split.get_image(j)) for j in lo]).to(device)
            F.append(backbone(x).float().cpu())
            Y.extend(int(split.labels[j]) for j in lo)
        return torch.cat(F), torch.tensor(Y)

    Ff, Yf = feats(fit_split, idx_fit)
    mu = torch.zeros(num_classes, Ff.shape[1])
    mu.index_add_(0, Yf, Ff)
    dem = torch.zeros(num_classes).index_add_(0, Yf, torch.ones(len(Yf)))
    mu = mu / dem.clamp(min=1).unsqueeze(1)

    Ft, Yt = feats(test_split, idx_test)
    d = torch.cdist(Ft, mu)
    d[:, dem < 1] = float("inf")        # ↳ lớp không có mẫu dựng prototype -> không được chọn
    return float((d.argmin(1) == Yt).float().mean())


def _dump(split, idx, image_size, num_tasks, cfg_drift, ra_file):
    """D9 — lưu lưới ảnh cùng một cảnh ở mọi mức trôi, để NHÌN BẰNG MẮT."""
    from torchvision.utils import save_image
    anh = []
    for t in range(num_tasks):
        m = muc_troi(t, num_tasks, cfg_drift.get("mode", "linear"),
                     float(cfg_drift.get("severity", 1.0)))
        tf = _tf(image_size, m, 0.0, seed=7)          # jitter=0 -> so sánh sạch
        x = tf(split.get_image(idx))
        x = x * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        anh.append(x.clamp(0, 1))
    save_image(torch.stack(anh), ra_file, nrow=num_tasks)
    print(f"\n  đã lưu {ra_file} — {num_tasks} mức trôi, trái sang phải 0% -> 100%")
    print("  NHÌN BẰNG MẮT: ảnh cuối phải trông như 'cùng cảnh, chiều muộn nhiều sương',")
    print("  KHÔNG phải như ảnh hỏng/nhiễu trắng. Trông hỏng = severity quá cao.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--severity", type=float, default=None, help="ghi đè severity để thử")
    ap.add_argument("--n-fit", type=int, default=2000, help="số mẫu dựng prototype")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--dump-samples", default=None, help="ghi lưới ảnh mẫu ra file PNG")
    a = ap.parse_args()

    cfg = load_config(a.config)
    d = cfg["data"]
    drift_cfg = dict(d.get("drift") or {})
    if a.severity is not None:
        drift_cfg["severity"] = a.severity
    num_tasks = int(d.get("num_tasks", 9))
    image_size = int(d.get("image_size", 224))
    jitter = float(drift_cfg.get("jitter", 0.15))

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
           else "cpu")
    print(f"\nthiết bị={dev} · severity={drift_cfg.get('severity', 1.0)} · "
          f"mode={drift_cfg.get('mode', 'linear')} · jitter={jitter}")

    source = get_source(d)
    tr, te = source.splits["train"], source.splits["test"]
    backbone, _ = build_backbone(cfg["backbone"])
    backbone = backbone.to(dev).eval()

    g = torch.Generator().manual_seed(0)
    idx_fit = torch.randperm(len(tr.labels), generator=g)[:a.n_fit].tolist()
    idx_test = torch.randperm(len(te.labels), generator=g)[:a.n_test].tolist()

    if a.dump_samples:
        _dump(te, idx_test[0], image_size, num_tasks, drift_cfg, a.dump_samples)

    print(f"\nNCM trên feature đóng băng ({a.n_fit} mẫu dựng prototype, {a.n_test} mẫu test)\n")
    print(f"  {'task':>5} {'mức trôi':>10} {'NCM acc':>10}   {'tụt so với t0':>14}")
    print("  " + "-" * 48)
    accs = []
    for t in range(num_tasks):
        m = muc_troi(t, num_tasks, drift_cfg.get("mode", "linear"),
                     float(drift_cfg.get("severity", 1.0)))
        acc = _ncm_acc(backbone, tr, te, idx_fit, idx_test,
                       image_size, m, jitter, dev, source.num_classes)
        accs.append(acc)
        tut = "" if t == 0 else f"{(acc - accs[0]) * 100:+13.2f}đ"
        print(f"  {t:>5} {m * 100:9.0f}% {acc:10.4f}   {tut:>14}")

    cuoi = accs[-1]
    print()
    print("=" * 60)
    if cuoi > DAT_HI:
        print(f"  ❌ TRÔI QUÁ NHẸ — NCM ở mức 100% vẫn {cuoi:.4f} (> {DAT_HI})")
        print(f"     Mọi arm λ sẽ như nhau, thí nghiệm vô nghĩa.")
        print(f"     → tăng severity, thử: --severity {float(drift_cfg.get('severity', 1.0)) * 1.5:.1f}")
    elif cuoi < DAT_LO - 0.20:
        print(f"  ❌ TRÔI QUÁ NẶNG — NCM ở mức 100% chỉ còn {cuoi:.4f}")
        print(f"     Feature ViT đã vỡ, mọi phương pháp đều chết, không phân biệt được.")
        print(f"     → giảm severity, thử: --severity {float(drift_cfg.get('severity', 1.0)) * 0.6:.1f}")
    elif cuoi < DAT_LO:
        print(f"  ⚠️  HƠI NẶNG — {cuoi:.4f}, hơi dưới vùng đề nghị [{DAT_LO}, {DAT_HI}]")
        print(f"     Vẫn chạy được nhưng nên cân nhắc giảm severity chút.")
    else:
        print(f"  ✅ ĐẠT — NCM tụt từ {accs[0]:.4f} xuống {cuoi:.4f} "
              f"(−{(accs[0] - cuoi) * 100:.1f} điểm), nằm trong vùng đề nghị.")
        print(f"     Có chỗ cho λ thể hiện, mà feature chưa vỡ. CHẠY ĐƯỢC campaign.")
    print("=" * 60)
    print(f"\n  Nhớ: severity đang dùng = {drift_cfg.get('severity', 1.0)} — "
          "ghi giá trị này vào 5 config arm trước khi chạy.\n")


if __name__ == "__main__":
    main()
