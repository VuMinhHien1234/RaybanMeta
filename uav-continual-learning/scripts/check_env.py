#!/usr/bin/env python3
"""G0 environment doctor. Run from project root:  python scripts/check_env.py"""
import importlib
import pathlib
import platform
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def check(name: str, attr: str = "__version__") -> bool:
    try:
        m = importlib.import_module(name)
        print(f"  [OK]   {name:16s} {getattr(m, attr, '?')}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [MISS] {name:16s} ({e.__class__.__name__})")
        return False


def main() -> int:
    print("== System ==")
    print(f"  python  {platform.python_version()}  ({platform.system()} {platform.machine()})")

    print("== Core deps ==")
    for p in ["numpy", "yaml", "sklearn", "pandas", "tqdm", "matplotlib"]:
        check(p)

    print("== ML deps ==")
    has_torch = check("torch")
    check("torchvision")
    check("timm")
    check("titans_pytorch", attr="__name__")
    check("datasets")  # HuggingFace — cần cho RESISC45 (G1)

    print("== Device ==")
    if has_torch:
        import torch
        mps = getattr(torch.backends, "mps", None)
        dev = "cuda" if torch.cuda.is_available() else (
            "mps" if (mps is not None and mps.is_available()) else "cpu")
        print(f"  torch {torch.__version__} -> device: {dev}")
    else:
        print("  torch not installed (see README bước 3)")

    print("== Package ==")
    ok_pkg = check("uavcl")

    print("\nDone. If the rows above are [OK], you are ready for G1.")
    return 0 if ok_pkg else 1


if __name__ == "__main__":
    raise SystemExit(main())
