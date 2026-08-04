# ↳ GIẢI THÍCH TỔNG QUAN: __init__ của package `metrics` — gom 2 nhóm chỉ số ra ngoài:
#   continual (đo học liên tục: accuracy/forgetting/BWT/FWT) và openset (đo nhận biết
#   mẫu lạ). Nhờ vậy chỉ cần `from uavcl.metrics import average_forgetting, ...`.
from .continual import (average_accuracy, average_anytime_accuracy, backward_transfer,
                        average_forgetting, forward_transfer)
from .openset import auc, eer, open_set_summary, roc_points, tar_at_far
# Bài toán BAY LẶP LẠI: 3 chỉ số trên KHÔNG đo được O3 ("gặp lại điều kiện cũ thì nhận ra
# ngay, không học lại"). Xem metrics/revisit.py — `loi_ich_quay_lai` là chỉ số trung tâm.
from .revisit import (acc_hien_tai, in_bao_cao, loi_ich_quay_lai, thoi_gian_hoi_phuc,
                      tom_tat_revisit)

__all__ = [
    # ↳ Danh sách tên công khai (cũng là mục lục các chỉ số của dự án).
    "average_accuracy",    # ↳ Độ chính xác trung bình cuối cùng.
    "average_anytime_accuracy",  # ↳ #24: trung bình acc TẠI MỌI MỐC (regime streaming/UAV).
    "backward_transfer",   # ↳ Học mới ảnh hưởng task cũ (âm = quên).
    "average_forgetting",  # ↳ Mức quên trung bình (số chính cần giảm).
    "forward_transfer",    # ↳ Kiến thức cũ giúp task mới chưa học.
    "roc_points",          # ↳ Điểm (FAR, TAR) theo ngưỡng.
    "auc",                 # ↳ Diện tích dưới ROC.
    "eer",                 # ↳ Mức lỗi cân bằng.
    "tar_at_far",          # ↳ TAR khi ép FAR tối đa.
    "open_set_summary",    # ↳ Gom mọi chỉ số open-set.
    "acc_hien_tai",        # ↳ O1: đúng bao nhiêu trên chính điều kiện ĐANG bay (đường chéo).
    "loi_ich_quay_lai",    # ↳ ⭐ O3: Acc(gặp lại điều kiện X) − Acc(lần đầu gặp X).
    "thoi_gian_hoi_phuc",  # ↳ O2: bao nhiêu bước để bám kịp sau khi điều kiện đổi.
    "tom_tat_revisit",     # ↳ Gom cả ba, gọi từ run_g1.
    "in_bao_cao",          # ↳ In gọn ra log, kèm cảnh báo khi O3 thất bại.
]
