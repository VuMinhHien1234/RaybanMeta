"""Thao tác trên "state" của bộ nhớ Titans (G2).

State do titans-pytorch trả về là cấu trúc lồng nhau (namedtuple/dict/tuple)
chứa tensor. Các hàm ở đây đi đệ quy qua cấu trúc đó, áp dụng phép biến đổi
lên TỪNG tensor, giữ nguyên hình dạng cấu trúc — nhờ vậy code không phụ
thuộc vào version cụ thể của thư viện.

- detach_state : cắt gradient (truncated BPTT) nhưng GIỮ giá trị ký ức.
- clone_state  : bản sao độc lập — dùng khi eval ("chấm thi không ghi trí nhớ").
- state_to_cpu : chuyển về CPU để torch.save (checkpoint S10).
- state_norm   : độ lớn tổng của state — log mỗi task để phát hiện "phình" (C2).
- count_floats : đếm phần tử — báo cáo chi phí bộ nhớ.
"""
# ↳ GIẢI THÍCH TỔNG QUAN: "state" = cục ký ức của Titans. Nó KHÔNG phải 1 tensor
#   phẳng mà là 1 "cây" lồng nhau (tuple trong dict trong list...) chứa nhiều tensor.
#   Muốn xử lý cả cây (detach/clone/đo norm) mà không cần biết hình dạng chính xác,
#   ta viết hàm ĐỆ QUY đi khắp cây rồi áp phép biến đổi lên từng tensor lá.
from __future__ import annotations

import math

import torch


def _is_tensordict_like(obj) -> bool:
    """titans-pytorch gói weights/updates trong tensordict.TensorDict — KHÔNG phải dict
    thường nên phải nhận diện riêng (bỏ sót nó = state còn dính graph -> double-backward)."""
    # ↳ TensorDict trông giống dict nhưng là kiểu riêng của thư viện. Nếu không nhận ra
    #   nó, ta sẽ quên detach bên trong -> backward lần 2 vào graph cũ -> crash.
    if torch.is_tensor(obj):
        return False                              # ↳ Tensor thường thì không phải tensordict.
    mod = (type(obj).__module__ or "").split(".")[0]  # ↳ Lấy tên gói gốc của kiểu đối tượng.
    if mod == "tensordict":
        return True                               # ↳ Thuộc gói `tensordict` -> đúng là nó.
    return hasattr(obj, "apply") and hasattr(obj, "batch_size") and hasattr(obj, "keys")
    # ↳ Dự phòng: nhận diện theo "dáng" (có đủ 3 phương thức đặc trưng) phòng khi đổi version.


def _tree_map(fn, obj):
    """Áp fn lên mọi tensor trong cấu trúc lồng nhau, giữ nguyên khung."""
    # ↳ Hàm lõi: đi khắp cây, gặp tensor thì gọi fn(tensor), gặp container thì đệ quy vào trong.
    if torch.is_tensor(obj):
        return fn(obj)                            # ↳ Lá là tensor -> áp phép biến đổi.
    if _is_tensordict_like(obj):
        return obj.apply(lambda t: fn(t) if torch.is_tensor(t) else t)  # ↳ TensorDict có sẵn .apply.
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return type(obj)(*(_tree_map(fn, o) for o in obj))  # ↳ namedtuple: dựng lại đúng kiểu, đệ quy từng phần tử.
    if isinstance(obj, tuple):
        return tuple(_tree_map(fn, o) for o in obj)  # ↳ tuple thường.
    if isinstance(obj, list):
        return [_tree_map(fn, o) for o in obj]       # ↳ list.
    if isinstance(obj, dict):
        return {k: _tree_map(fn, v) for k, v in obj.items()}  # ↳ dict: giữ key, đệ quy value.
    return obj  # int/float/None/str... giữ nguyên     # ↳ Không phải tensor/container -> để nguyên.


def _tree_tensors(obj):
    """Liệt kê mọi tensor trong cấu trúc (generator)."""
    # ↳ Giống _tree_map nhưng chỉ "nhặt" tensor ra để đếm/đo, không biến đổi. `yield` = trả dần.
    if torch.is_tensor(obj):
        yield obj
    elif _is_tensordict_like(obj):
        for v in obj.values():
            yield from _tree_tensors(v)           # ↳ yield from: nối các tensor từ nhánh con.
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            yield from _tree_tensors(o)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tree_tensors(v)


def detach_state(state):
    # ↳ "Cắt dây gradient": giữ nguyên GIÁ TRỊ ký ức nhưng ngắt liên kết với đồ thị tính đạo hàm.
    #   Đây là truncated BPTT — chống crash "backward lần 2" và tránh RAM phình vô hạn.
    return _tree_map(lambda t: t.detach(), state)


def clone_state(state):
    # ↳ Tạo BẢN SAO độc lập của ký ức. Dùng khi eval: chấm thi trên bản sao, không đụng bản gốc.
    return _tree_map(lambda t: t.detach().clone(), state)


def state_to_cpu(state):
    # ↳ Kéo mọi tensor về CPU để lưu file (torch.save) — checkpoint không phụ thuộc GPU.
    return _tree_map(lambda t: t.detach().cpu(), state)


def state_norm(state) -> float:
    """sqrt(tổng bình phương norm các tensor float) — 0.0 nếu state rỗng/None."""
    # ↳ Đo "độ lớn" cục ký ức. Theo dõi số này qua các task: nếu nó phình/nổ tức là
    #   bộ nhớ mất ổn định (chính là bệnh Forgetting 0.956 đã gặp ở HOPE).
    if state is None:
        return 0.0                                # ↳ Chưa có ký ức -> norm = 0.
    total = 0.0
    for t in _tree_tensors(state):                # ↳ Duyệt mọi tensor trong cây.
        if t.is_floating_point() and t.numel() > 0:  # ↳ Chỉ tính tensor số thực, có phần tử.
            total += float(t.detach().float().norm()) ** 2  # ↳ Cộng bình phương norm từng tensor.
    return math.sqrt(total)                       # ↳ Căn bậc hai của tổng = norm toàn cục (như Pythagoras).


def count_floats(state) -> int:
    # ↳ Đếm tổng số phần tử trong ký ức -> ước lượng chi phí bộ nhớ để so giữa các method.
    if state is None:
        return 0
    return sum(t.numel() for t in _tree_tensors(state))  # ↳ numel = number of elements của mỗi tensor.
