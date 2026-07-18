# ↳ GIẢI THÍCH TỔNG QUAN: File này gom các tiện ích chung của package `utils`.
#   Ngoài việc "kéo" seed_everything ra ngoài cho dễ import, nó định nghĩa
#   get_device() — chọn phần cứng tính toán tốt nhất đang có.
from .seed import seed_everything  # ↳ Re-export: cho phép `from uavcl.utils import seed_everything`.


def get_device() -> str:
    """Return the best available device string: 'cuda' | 'mps' | 'cpu'."""
    # ↳ Trả về tên thiết bị để đẩy model/dữ liệu lên đó: GPU NVIDIA > GPU Apple > CPU.
    try:
        import torch                      # ↳ Cần torch để hỏi phần cứng; chưa cài -> mặc định CPU.
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():         # ↳ Ưu tiên 1: GPU NVIDIA (nhanh nhất cho train).
        return "cuda"
    mps = getattr(torch.backends, "mps", None)  # ↳ Ưu tiên 2: MPS = GPU trên máy Mac Apple Silicon.
    if mps is not None and mps.is_available():   # ↳ Kiểm tra torch có hỗ trợ mps VÀ máy này bật được.
        return "mps"
    return "cpu"                          # ↳ Không có GPU -> chạy CPU (chậm hơn nhưng vẫn chạy).


__all__ = ["seed_everything", "get_device"]  # ↳ Danh sách tên "công khai" khi ai đó `from uavcl.utils import *`.
