"""Data & continual-learning stream (owner: N1).

Cách dùng chuẩn (xem scripts/run_g1.py):

    from uavcl.data import get_source, build_stream, build_task_loaders

    source = get_source(cfg["data"])                     # resisc45 | eurosat | synthetic
    stream = build_stream(                                # chia class-incremental
        source.splits["train"].labels,
        source.splits["val"].labels,
        source.splits["test"].labels,
        num_classes=source.num_classes,
        num_tasks=cfg["data"]["num_tasks"],
        seed=cfg["seed"],
    )
    loaders = build_task_loaders(source, stream, cfg["data"])   # cần torch

`stream.py` + `sources.py` (trừ eurosat) không import torch — test được ở mọi máy.
"""
# ↳ GIẢI THÍCH TỔNG QUAN: File __init__ của package `data` gom 3 module con
#   (sources, stream, loaders) và "phơi" các hàm hay dùng ra ngoài để import gọn.
#   Docstring trên chính là công thức 3 bước: lấy dữ liệu -> chia thành chuỗi task
#   -> dựng DataLoader. Ghi nhớ đúng thứ tự này là hiểu cả tầng dữ liệu.
from .sources import DataSource, get_source
from .stream import TaskSpec, build_stream, describe_stream, indices_by_task, split_classes, stratified_split

__all__ = [
    # ↳ Danh sách tên public — cũng là "mục lục" những gì tầng data cung cấp.
    "DataSource",       # ↳ Hộp chuẩn chứa dataset.
    "get_source",       # ↳ Lấy dataset theo tên.
    "TaskSpec",         # ↳ Mô tả 1 task.
    "build_stream",     # ↳ Tạo chuỗi task học liên tục.
    "describe_stream",  # ↳ In tóm tắt chuỗi task.
    "indices_by_task",  # ↳ Gom mẫu theo task.
    "split_classes",    # ↳ Chia class thành nhóm.
    "stratified_split", # ↳ Chia tỉ lệ giữ cân bằng class.
]
