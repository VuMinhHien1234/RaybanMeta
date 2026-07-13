#!/usr/bin/env python3
"""G0 smoke: verify a (pretrained-capable) vision backbone extracts features.

Run from project root:  python scripts/smoke_backbone.py            # random weights (no download)
                        python scripts/smoke_backbone.py --pretrained # download weights (needs net)
Confirms the vision pipeline (image -> feature sequence) works before G1.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_small_patch16_224")
    ap.add_argument("--pretrained", action="store_true",
                    help="download pretrained weights (needs network)")
    args = ap.parse_args()

    try:
        import torch
        import timm
    except ImportError as e:
        print(f"[SKIP] missing dependency: {e.name}. Install per README (torch + timm).")
        return 1

    from uavcl.utils import get_device

    device = get_device()
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=0).to(device).eval()
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        feats = model.forward_features(x)                  # patch feature map / tokens
        emb = model.forward_head(feats, pre_logits=True)   # pooled embedding

    print(f"device: {device}; model: {args.model}; pretrained={args.pretrained}")
    print(f"[OK] features {tuple(feats.shape)} -> embedding {tuple(emb.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
