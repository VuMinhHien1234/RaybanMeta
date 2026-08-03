"""Class-incremental task stream — pure Python (no torch), so N1 can unit-test
the split logic anywhere.

Idea: chia các class của dataset thành T nhóm; task t = "học các class mới của
nhóm t". Model xem task 1 -> 2 -> ... tuần tự, không được xem lại dữ liệu cũ
(trừ khi method có replay). Đây là kịch bản UAV gặp môi trường/đối tượng mới
dần theo thời gian.
"""
# ↳ GIẢI THÍCH TỔNG QUAN FILE NÀY (rất quan trọng để hiểu "continual learning"):
#   - Dataset có sẵn N class. Ta KHÔNG cho model học hết 1 lúc, mà chia thành T
#     "task" (đợt). Đợt 1 dạy vài class, đợt 2 dạy vài class KHÁC, v.v.
#   - Model học lần lượt và KHÔNG được xem lại dữ liệu đợt trước -> dễ "quên"
#     (catastrophic forgetting). Cả dự án xoay quanh việc chống quên này.
#   - File này thuần Python (không import torch) nên chạy/test được ở mọi máy,
#     kể cả máy chưa cài PyTorch.
from __future__ import annotations

import random  # ↳ Dùng để xáo thứ tự class (có seed để tái lập).
from dataclasses import dataclass, field       # ↳ dataclass = tạo class chứa dữ liệu gọn, không cần viết __init__.
from typing import Dict, List, Sequence        # ↳ Chỉ để chú thích kiểu (gợi ý cho người đọc/IDE).


def split_classes(
    num_classes: int,     # ↳ Tổng số class của dataset (vd EuroSAT = 10).
    num_tasks: int,       # ↳ Muốn chia thành bao nhiêu đợt học.
    seed: int = 0,        # ↳ Hạt ngẫu nhiên để cố định cách xáo class.
    shuffle: bool = True, # ↳ Có xáo thứ tự class trước khi chia không.
) -> List[List[int]]:
    """Chia `num_classes` class thành `num_tasks` nhóm (gần) đều nhau.

    shuffle=True: xáo thứ tự class theo seed (thứ tự task ảnh hưởng kết quả CL,
    nên phải cố định bằng seed để cả team tái lập được).
    """
    if not (1 <= num_tasks <= num_classes):    # ↳ Chặn vô lý: không thể có nhiều task hơn số class.
        raise ValueError(f"num_tasks={num_tasks} must be in [1, {num_classes}]")
    order = list(range(num_classes))           # ↳ Bắt đầu bằng thứ tự [0,1,2,...,N-1].
    if shuffle:
        random.Random(seed).shuffle(order)     # ↳ Xáo tại chỗ bằng bộ random RIÊNG gieo bằng seed (tái lập được).
    base, extra = divmod(num_classes, num_tasks)  # ↳ base = số class mỗi nhóm; extra = phần dư chia không hết.
    groups: List[List[int]] = []
    i = 0                                      # ↳ Con trỏ chạy dọc danh sách `order`.
    for t in range(num_tasks):
        n = base + (1 if t < extra else 0)     # ↳ `extra` nhóm đầu nhận thêm 1 class để chia đều nhất có thể.
        groups.append(sorted(order[i : i + n]))  # ↳ Lấy n class kế tiếp, sort lại cho gọn/ổn định.
        i += n                                 # ↳ Dời con trỏ qua đoạn vừa lấy.
    return groups                              # ↳ Trả về danh sách nhóm class, vd [[3,7],[0,9],...].


def indices_by_task(labels: Sequence[int], task_classes: List[List[int]]) -> List[List[int]]:
    """Với 1 split (vd train), trả về danh sách chỉ số mẫu thuộc từng task."""
    # ↳ Mục tiêu: biết mỗi TASK gồm những MẪU nào (theo chỉ số trong split).
    cls2task = {c: t for t, cs in enumerate(task_classes) for c in cs}
    # ↳ Bảng tra ngược "class -> task": vd class 7 nằm ở task 0 thì cls2task[7]=0.
    buckets: List[List[int]] = [[] for _ in task_classes]  # ↳ Mỗi task 1 "giỏ" rỗng để bỏ chỉ số vào.
    for idx, y in enumerate(labels):           # ↳ Duyệt từng mẫu: idx = vị trí, y = nhãn class.
        t = cls2task.get(int(y))               # ↳ Xem nhãn này thuộc task nào (None nếu class không dùng).
        if t is not None:
            buckets[t].append(idx)             # ↳ Bỏ chỉ số mẫu vào đúng giỏ task của nó.
    return buckets                             # ↳ Trả [ [chỉ số của task0], [task1], ... ].


def stratified_split(
    indices: Sequence[int],   # ↳ Danh sách chỉ số cần chia đôi.
    labels: Sequence[int],    # ↳ Nhãn tương ứng (để giữ tỉ lệ class).
    fraction: float,          # ↳ Tỉ lệ cho phần NHỎ (vd 0.10 = 10%).
    seed: int = 0,
) -> tuple[List[int], List[int]]:
    """Tách `indices` thành (phần_lớn, phần_nhỏ) giữ tỉ lệ class (per-class).

    Dùng để tạo val/test khi dataset không có split sẵn. Deterministic theo seed.
    """
    # ↳ "Stratified" = chia mà mỗi class đều bị cắt đúng tỉ lệ, tránh lệch class.
    by_class: Dict[int, List[int]] = {}        # ↳ Gom chỉ số theo từng class.
    for i in indices:
        by_class.setdefault(int(labels[i]), []).append(i)  # ↳ class chưa có key -> tạo list rỗng rồi thêm.
    rng = random.Random(seed)                  # ↳ Bộ random cố định theo seed.
    big: List[int] = []                        # ↳ Phần lớn (vd train).
    small: List[int] = []                      # ↳ Phần nhỏ (vd val/test).
    for c in sorted(by_class):                 # ↳ Duyệt từng class theo thứ tự cố định (sorted -> tái lập).
        idxs = by_class[c][:]                  # ↳ Copy list chỉ số của class c ([:] để không sửa bản gốc).
        rng.shuffle(idxs)                      # ↳ Xáo trong nội bộ class.
        k = int(round(len(idxs) * fraction))   # ↳ Số mẫu đưa vào phần nhỏ = làm tròn(số mẫu * tỉ lệ).
        if len(idxs) > 1:
            k = min(max(k, 1), len(idxs) - 1)  # ↳ Đảm bảo mỗi bên có ít nhất 1 mẫu (không để class biến mất).
        else:
            k = 0                              # ↳ Class chỉ có 1 mẫu -> dồn hết vào phần lớn.
        small += idxs[:k]                      # ↳ k mẫu đầu -> phần nhỏ.
        big += idxs[k:]                        # ↳ Còn lại -> phần lớn.
    return sorted(big), sorted(small)          # ↳ Sort cho kết quả ổn định giữa các lần chạy.


@dataclass
class TaskSpec:
    """Một task trong stream. Các chỉ số là *cục bộ theo split*
    (train_idx trỏ vào split train, test_idx trỏ vào split test...)."""
    # ↳ Đây là "phiếu mô tả" 1 task: gồm những class nào và những mẫu (chỉ số) nào
    #   cho train/val/test. field(default_factory=list) = mặc định là list rỗng.
    task_id: int                                          # ↳ Số thứ tự task (0,1,2,...).
    classes: List[int] = field(default_factory=list)      # ↳ Các class MỚI mà task này dạy.
    train_idx: List[int] = field(default_factory=list)    # ↳ Chỉ số mẫu train của task.
    val_idx: List[int] = field(default_factory=list)      # ↳ Chỉ số mẫu val của task.
    test_idx: List[int] = field(default_factory=list)     # ↳ Chỉ số mẫu test của task.


def build_stream(
    train_labels: Sequence[int],   # ↳ Nhãn của toàn bộ split train.
    val_labels: Sequence[int],     # ↳ Nhãn của toàn bộ split val.
    test_labels: Sequence[int],    # ↳ Nhãn của toàn bộ split test.
    num_classes: int,
    num_tasks: int,
    seed: int = 0,
    shuffle_classes: bool = True,
) -> List[TaskSpec]:
    """Ghép mọi thứ: chia class -> gom chỉ số của 3 split theo từng task."""
    # ↳ Đây là hàm "tổng đạo diễn" tạo ra cả chuỗi task học liên tục.
    groups = split_classes(num_classes, num_tasks, seed=seed, shuffle=shuffle_classes)  # ↳ Bước 1: chia class thành T nhóm.
    tr = indices_by_task(train_labels, groups)  # ↳ Bước 2a: gom mẫu train theo task.
    va = indices_by_task(val_labels, groups)    # ↳ Bước 2b: gom mẫu val theo task.
    te = indices_by_task(test_labels, groups)   # ↳ Bước 2c: gom mẫu test theo task.
    stream = [
        TaskSpec(task_id=t, classes=groups[t], train_idx=tr[t], val_idx=va[t], test_idx=te[t])
        for t in range(num_tasks)               # ↳ Bước 3: gói mỗi task thành 1 TaskSpec.
    ]
    for spec in stream:
        if not spec.train_idx or not spec.test_idx:  # ↳ Kiểm tra an toàn: task nào thiếu train/test là cấu hình sai.
            raise ValueError(f"Task {spec.task_id} has empty train/test — check labels/num_tasks")
    return stream                               # ↳ Trả về danh sách TaskSpec = "kịch bản" học liên tục.


def chia_deu_theo_lop(
    labels: Sequence[int], num_tasks: int, seed: int = 0
) -> List[List[int]]:
    """Chia MỌI mẫu thành `num_tasks` phần, mỗi phần có đủ mọi lớp với số lượng cân bằng.

    Khác `indices_by_task` (chia theo LỚP): ở đây chia theo MẪU, mọi task đều thấy đủ 45 lớp.
    Dùng cho stream domain-incremental — task khác nhau ở ĐIỀU KIỆN, không ở tập lớp.
    """
    theo_lop: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        theo_lop.setdefault(int(y), []).append(i)
    ra: List[List[int]] = [[] for _ in range(num_tasks)]
    rng = random.Random(seed)
    for c in sorted(theo_lop):                     # ↳ sorted -> thứ tự cố định, tái lập được
        idxs = theo_lop[c][:]
        rng.shuffle(idxs)
        for k, i in enumerate(idxs):               # ↳ rải vòng tròn -> mọi task đều có lớp c
            ra[k % num_tasks].append(i)
    for phan in ra:
        phan.sort()
    return ra


def build_domain_stream(
    train_labels: Sequence[int],
    val_labels: Sequence[int],
    test_labels: Sequence[int],
    num_classes: int,
    num_tasks: int,
    seed: int = 0,
) -> List[TaskSpec]:
    """Stream DOMAIN-incremental: mọi task có ĐỦ mọi lớp, chỉ khác ĐIỀU KIỆN quan sát.

    Vì sao phải tách khỏi `build_stream` (class-incremental): nếu để lẫn "lớp mới" và
    "điều kiện trôi" trong cùng một stream thì khi accuracy tụt, KHÔNG tách được nguyên
    nhân là *quên lớp* hay *trôi điều kiện* — kết quả không diễn giải được.

    Việc áp trôi nằm ở `data/drift.py` + `loaders.py`; hàm này chỉ lo chia mẫu.

    Ngữ nghĩa ma trận accuracy đổi theo:
        R[i][j] = accuracy ở ĐIỀU KIỆN j, sau khi đã học tới ĐIỀU KIỆN i
        - đường chéo  R[i][i] : hoạt động trong điều kiện HIỆN TẠI  <- chỉ số chính
        - dưới chéo   R[i][j] : quay lại điều kiện CŨ có còn chạy không
    Nên "forgetting" ở đây = *mất khả năng hoạt động ở điều kiện cũ*, không phải quên lớp.
    """
    if num_tasks < 1:
        raise ValueError(f"num_tasks phải >= 1 (nhận {num_tasks})")
    moi_lop = sorted(set(int(y) for y in train_labels))
    tr = chia_deu_theo_lop(train_labels, num_tasks, seed=seed)
    va = chia_deu_theo_lop(val_labels, num_tasks, seed=seed + 1)
    te = chia_deu_theo_lop(test_labels, num_tasks, seed=seed + 2)
    stream = [
        TaskSpec(task_id=t, classes=moi_lop,       # ↳ MỌI task đều "dạy" đủ mọi lớp
                 train_idx=tr[t], val_idx=va[t], test_idx=te[t])
        for t in range(num_tasks)
    ]
    for spec in stream:
        if not spec.train_idx or not spec.test_idx:
            raise ValueError(f"Task {spec.task_id} rỗng — num_tasks quá lớn so với số mẫu?")
    return stream


def build_stream_with_holdout(
    train_labels: Sequence[int],
    val_labels: Sequence[int],
    test_labels: Sequence[int],
    num_classes: int,
    num_tasks: int,
    seed: int = 0,
    shuffle_classes: bool = True,
    holdout: int = 5,
) -> tuple[List[TaskSpec], List[int]]:
    """#27 open-set — như build_stream nhưng GIỮ LẠI `holdout` class KHÔNG BAO GIỜ train.

    Các class giữ lại làm "mẫu LẠ" (unseen) để đo open-set: model phải biết nói "không
    biết" khi gặp chúng. Trả (stream trên các class còn lại, danh sách class giữ lại).
    Cùng seed -> cùng phép xáo -> tái lập được."""
    if not (1 <= holdout < num_classes):
        raise ValueError(f"holdout cần trong [1, {num_classes - 1}] (nhận {holdout})")
    order = list(range(num_classes))
    if shuffle_classes:
        random.Random(seed).shuffle(order)         # ↳ Cùng RNG/seed với split_classes -> nhất quán.
    held = sorted(order[-holdout:])                # ↳ Cắt ĐUÔI sau xáo làm class lạ.
    kept = order[:-holdout]
    if len(kept) < num_tasks:
        raise ValueError(f"còn {len(kept)} class cho {num_tasks} task — giảm holdout/num_tasks")
    base, extra = divmod(len(kept), num_tasks)     # ↳ Chia phần còn lại thành num_tasks nhóm đều.
    groups, i = [], 0
    for t in range(num_tasks):
        k = base + (1 if t < extra else 0)
        groups.append(sorted(kept[i:i + k]))
        i += k
    tr = indices_by_task(train_labels, groups)
    va = indices_by_task(val_labels, groups)
    te = indices_by_task(test_labels, groups)
    stream = [
        TaskSpec(task_id=t, classes=groups[t], train_idx=tr[t], val_idx=va[t], test_idx=te[t])
        for t in range(num_tasks)
    ]
    for spec in stream:
        if not spec.train_idx or not spec.test_idx:
            raise ValueError(f"Task {spec.task_id} has empty train/test — check labels/num_tasks")
    return stream, held


def describe_stream(stream: List[TaskSpec], class_names: Sequence[str] | None = None) -> str:
    # ↳ Chỉ để IN RA cho người xem: mỗi task có class gì, bao nhiêu mẫu. Không ảnh hưởng train.
    lines = []
    for s in stream:
        names = (
            ", ".join(class_names[c] for c in s.classes) if class_names else str(s.classes)
        )  # ↳ Nếu biết tên class thì hiện tên; không thì hiện số class.
        lines.append(
            f"task {s.task_id}: {len(s.classes)} classes [{names}] | "
            f"train {len(s.train_idx)} / val {len(s.val_idx)} / test {len(s.test_idx)}"
        )  # ↳ Ghép 1 dòng tóm tắt cho mỗi task.
    return "\n".join(lines)                      # ↳ Nối các dòng bằng xuống dòng.
