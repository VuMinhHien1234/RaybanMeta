from .seed import seed_everything


def get_device() -> str:
    """Return the best available device string: 'cuda' | 'mps' | 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


__all__ = ["seed_everything", "get_device"]
