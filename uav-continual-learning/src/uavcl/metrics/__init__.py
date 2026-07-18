# ↳ GIẢI THÍCH TỔNG QUAN: __init__ của package `metrics` — gom 2 nhóm chỉ số ra ngoài:
#   continual (đo học liên tục: accuracy/forgetting/BWT/FWT) và openset (đo nhận biết
#   mẫu lạ). Nhờ vậy chỉ cần `from uavcl.metrics import average_forgetting, ...`.
from .continual import average_accuracy, backward_transfer, average_forgetting, forward_transfer
from .openset import auc, eer, open_set_summary, roc_points, tar_at_far

__all__ = [
    # ↳ Danh sách tên công khai (cũng là mục lục các chỉ số của dự án).
    "average_accuracy",    # ↳ Độ chính xác trung bình cuối cùng.
    "backward_transfer",   # ↳ Học mới ảnh hưởng task cũ (âm = quên).
    "average_forgetting",  # ↳ Mức quên trung bình (số chính cần giảm).
    "forward_transfer",    # ↳ Kiến thức cũ giúp task mới chưa học.
    "roc_points",          # ↳ Điểm (FAR, TAR) theo ngưỡng.
    "auc",                 # ↳ Diện tích dưới ROC.
    "eer",                 # ↳ Mức lỗi cân bằng.
    "tar_at_far",          # ↳ TAR khi ép FAR tối đa.
    "open_set_summary",    # ↳ Gom mọi chỉ số open-set.
]
