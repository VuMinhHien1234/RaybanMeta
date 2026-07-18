# ↳ GIẢI THÍCH TỔNG QUAN: Máy học có nhiều bước ngẫu nhiên (khởi tạo trọng số,
#   xáo dữ liệu...). "Seed" = gieo cùng 1 hạt giống ngẫu nhiên để mỗi lần chạy ra
#   KẾT QUẢ GIỐNG NHAU -> tái lập được thí nghiệm, so sánh mới công bằng.
import os      # ↳ Để đặt biến môi trường PYTHONHASHSEED.
import random  # ↳ Bộ sinh số ngẫu nhiên chuẩn của Python.

import numpy as np  # ↳ numpy có bộ ngẫu nhiên riêng, phải seed riêng.


def seed_everything(seed: int = 0) -> int:
    """Seed python, numpy and (if installed) torch for reproducible runs."""
    # ↳ Gieo hạt cho TẤT CẢ nguồn ngẫu nhiên có thể ảnh hưởng kết quả.
    random.seed(seed)                          # ↳ Seed bộ random của Python.
    np.random.seed(seed)                       # ↳ Seed bộ random của numpy.
    os.environ["PYTHONHASHSEED"] = str(seed)   # ↳ Cố định cách băm (hash) -> thứ tự set/dict ổn định.
    try:
        import torch                           # ↳ torch có thể chưa cài (vd chạy test data thuần Python) -> bọc try.
        torch.manual_seed(seed)                # ↳ Seed cho CPU của PyTorch (khởi tạo trọng số...).
        if torch.cuda.is_available():          # ↳ Nếu có GPU NVIDIA...
            torch.cuda.manual_seed_all(seed)   # ↳ ...seed cho mọi GPU nữa.
    except ImportError:
        pass                                   # ↳ Không có torch thì bỏ qua, phần trên vẫn seed xong.
    return seed                                # ↳ Trả lại seed để log ra cho biết đã dùng hạt nào.
