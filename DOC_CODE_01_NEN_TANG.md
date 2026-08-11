# Đọc code — Bài 1: Nền tảng

> **Bộ 3 bài, đọc theo thứ tự.** Bài 1 (file này) → Bài 2 (engine + methods) → Bài 3 (models + phần mới).
> Mỗi hàm được viết theo cùng một khuôn: **nhận gì → làm gì → trả gì → ai gọi nó → bẫy**.

---

## Trả lời câu hỏi "bắt đầu từ file nào, hàm nào"

**File đầu tiên:** `src/uavcl/data/stream.py`
**Hàm đầu tiên:** `split_classes()` — 20 dòng, không import gì ngoài `random`.

**Vì sao đúng chỗ đó:**

1. Không cần torch → chạy được ngay, không phải cài gì.
2. Nó tạo ra khái niệm **"task"** — đơn vị cơ bản mà toàn bộ 5.000 dòng còn lại xoay quanh.
3. Nó ngắn và tự chứa: đọc 20 dòng là hiểu trọn vẹn một thứ, không phải nhảy file.
4. Có test tính tay đối chiếu ngay: `tests/test_stream.py`.

**Sai lầm hay gặp:** mở `engine.py` hoặc `methods.py` trước. Hai file đó là trái tim, nhưng chúng gọi tới 15 file khác — đọc trước sẽ phải nhảy liên tục và không đọng lại gì.

**Thứ tự 12 hàm đầu tiên** (làm theo đúng thứ tự này):

| # | Hàm | File | Dòng |
|---|---|---|---:|
| 1 | `split_classes` | `data/stream.py` | ~20 |
| 2 | `indices_by_task` | `data/stream.py` | ~12 |
| 3 | `TaskSpec` | `data/stream.py` | ~8 |
| 4 | `build_stream` | `data/stream.py` | ~20 |
| 5 | `average_accuracy` | `metrics/continual.py` | 3 |
| 6 | `average_forgetting` | `metrics/continual.py` | ~12 |
| 7 | `backward_transfer` | `metrics/continual.py` | ~8 |
| 8 | `average_anytime_accuracy` | `metrics/continual.py` | ~5 |
| 9 | `mask_logits` | `models/classifier.py` | 5 |
| 10 | `ContinualClassifier.forward` | `models/classifier.py` | 2 |
| 11 | `build_backbone` | `models/backbone.py` | ~20 |
| 12 | `build_task_loaders` | `data/loaders.py` | ~30 |

Xong 12 hàm này (~150 dòng code thật) là bạn hiểu **60% khung sườn** dự án. Phần còn lại là biến thể.

---

# PHẦN A — `data/stream.py` (246 dòng)

**Nhiệm vụ file:** biến "một dataset có N lớp" thành "một chuỗi T task học nối tiếp".
**Đặc điểm:** thuần Python, **không import torch** — chủ ý, để test logic chia ở mọi máy.

---

## A1. `split_classes()` ⭐ HÀM ĐẦU TIÊN NÊN ĐỌC

```python
def split_classes(num_classes: int, num_tasks: int,
                  seed: int = 0, shuffle: bool = True) -> List[List[int]]
```

**Nhận:** tổng số lớp (45 với RESISC45), số task muốn chia (9), seed, có xáo không.
**Trả:** danh sách `num_tasks` nhóm lớp, ví dụ `[[3,7,12,20,41], [0,9,15,22,38], ...]`.

**Làm gì, từng bước:**

```python
if not (1 <= num_tasks <= num_classes):     # (1) chặn vô lý
    raise ValueError(...)
order = list(range(num_classes))            # (2) [0,1,2,...,44]
if shuffle:
    random.Random(seed).shuffle(order)      # (3) xáo bằng RNG RIÊNG gieo bằng seed
base, extra = divmod(num_classes, num_tasks)  # (4) 45//9=5, dư 0
groups, i = [], 0
for t in range(num_tasks):
    n = base + (1 if t < extra else 0)      # (5) `extra` nhóm đầu nhận thêm 1
    groups.append(sorted(order[i:i+n]))     # (6) cắt n lớp kế tiếp, sort cho ổn định
    i += n
return groups
```

**Ba chi tiết đáng dừng lại:**

- **Dòng (3) — `random.Random(seed)` chứ không phải `random.shuffle`.** Tạo một bộ sinh **riêng**, không đụng tới bộ toàn cục. Nghĩa là gọi hàm này không làm lệch mọi thứ ngẫu nhiên khác trong chương trình. Đây là thói quen tốt, nên bắt chước.

- **Vì sao phải xáo?** Docstring nói: *"thứ tự task ảnh hưởng kết quả CL"*. Học "máy bay → sân bay → đường băng" khác hẳn "máy bay → rừng → biển". Xáo bằng seed cố định để (a) không bị thiên vị thứ tự alphabet của dataset, (b) cả team chạy ra cùng số.

- **Dòng (4)(5) — chia dư.** 45/9 chia hết nên `extra=0`. Nhưng với 10 lớp / 3 task thì `base=3, extra=1` → nhóm đầu 4 lớp, hai nhóm sau 3 lớp. Không có nhóm nào rỗng.

**Ai gọi:** `build_stream()` và `build_stream_with_holdout()`.

**Thử ngay:**
```python
from uavcl.data.stream import split_classes
print(split_classes(10, 3, seed=0))   # chạy lại 2 lần phải ra y hệt
print(split_classes(10, 3, seed=1))   # đổi seed -> nhóm khác
```

---

## A2. `indices_by_task()`

```python
def indices_by_task(labels: Sequence[int],
                    task_classes: List[List[int]]) -> List[List[int]]
```

**Nhận:** danh sách nhãn của cả một split (ví dụ 18.900 nhãn của split train), và kết quả của `split_classes`.
**Trả:** với mỗi task, danh sách **chỉ số** mẫu thuộc task đó.

```python
cls2task = {c: t for t, cs in enumerate(task_classes) for c in cs}   # bảng tra ngược
buckets = [[] for _ in task_classes]
for idx, y in enumerate(labels):
    t = cls2task.get(int(y))
    if t is not None:          # None = lớp này không dùng (bị holdout)
        buckets[t].append(idx)
return buckets
```

**Điểm cần hiểu:** hàm trả về **chỉ số**, không phải ảnh. Ảnh vẫn nằm nguyên trong `DataSource`. Cả dự án làm việc bằng chỉ số cho tới tận `DataLoader` — nhờ vậy chia task không tốn RAM và không copy dữ liệu.

**`cls2task.get(y)` trả `None`** khi lớp `y` không nằm trong nhóm nào. Đây là chỗ `build_stream_with_holdout` lợi dụng: lớp bị giữ lại đơn giản là không xuất hiện trong `task_classes` nên tự động bị bỏ qua.

---

## A3. `TaskSpec` — cấu trúc dữ liệu trung tâm

```python
@dataclass
class TaskSpec:
    task_id: int
    classes: List[int]     = field(default_factory=list)
    train_idx: List[int]   = field(default_factory=list)
    val_idx: List[int]     = field(default_factory=list)
    test_idx: List[int]    = field(default_factory=list)
```

**Đây là "phiếu mô tả" một task**, và là thứ engine nhận vào. 5 trường, không có phương thức nào.

**Lưu ý quan trọng ghi trong docstring:** các chỉ số là **cục bộ theo split** — `train_idx` trỏ vào split train, `test_idx` trỏ vào split test. Cùng con số `5` trong `train_idx` và `test_idx` là **hai ảnh khác nhau**.

**`field(default_factory=list)`** thay vì `= []`: nếu viết `= []` thì mọi `TaskSpec` sẽ dùng **chung một list** (bẫy kinh điển của Python — mutable default argument).

---

## A4. `build_stream()` ⭐ tổng đạo diễn

```python
def build_stream(train_labels, val_labels, test_labels,
                 num_classes, num_tasks, seed=0, shuffle_classes=True) -> List[TaskSpec]
```

Chỉ 3 bước, ghép hai hàm trên:

```python
groups = split_classes(num_classes, num_tasks, seed=seed, shuffle=shuffle_classes)  # B1
tr = indices_by_task(train_labels, groups)     # B2a
va = indices_by_task(val_labels, groups)       # B2b
te = indices_by_task(test_labels, groups)      # B2c
stream = [TaskSpec(task_id=t, classes=groups[t],
                   train_idx=tr[t], val_idx=va[t], test_idx=te[t])
          for t in range(num_tasks)]           # B3
for spec in stream:                            # B4 — cửa chặn
    if not spec.train_idx or not spec.test_idx:
        raise ValueError(f"Task {spec.task_id} has empty train/test — ...")
return stream
```

**Bước B4 đáng học theo.** Task rỗng nghĩa là cấu hình sai (num_tasks quá lớn, hoặc nhãn lệch). Ném lỗi **ngay** với thông báo chỉ đúng chỗ, thay vì để chương trình chạy 2 tiếng rồi crash ở chỗ khác. Kiểu "cửa chặn sớm" này lặp lại khắp dự án.

**Đây là hàm bạn gọi 90% thời gian.** Ba hàm dựng stream còn lại là biến thể cho trường hợp đặc biệt.

---

## A5. `chia_deu_theo_lop()` — chia theo MẪU thay vì theo LỚP

```python
def chia_deu_theo_lop(labels, num_tasks, seed=0) -> List[List[int]]
```

Khác `indices_by_task` ở một điểm cốt lõi:

| Hàm | Chia theo | Kết quả |
|---|---|---|
| `indices_by_task` | **lớp** | task 0 chỉ có lớp {3,7,12,20,41} |
| `chia_deu_theo_lop` | **mẫu** | task 0 có **đủ 45 lớp**, mỗi lớp 1/T số mẫu |

```python
theo_lop = {}                          # gom chỉ số theo lớp
for i, y in enumerate(labels):
    theo_lop.setdefault(int(y), []).append(i)
ra = [[] for _ in range(num_tasks)]
rng = random.Random(seed)
for c in sorted(theo_lop):             # sorted -> thứ tự cố định, tái lập được
    idxs = theo_lop[c][:]
    rng.shuffle(idxs)
    for k, i in enumerate(idxs):
        ra[k % num_tasks].append(i)    # ⭐ rải VÒNG TRÒN -> mọi task đều có lớp c
```

**`k % num_tasks`** là toàn bộ mẹo: rải vòng tròn đảm bảo mỗi task nhận số mẫu của lớp `c` chênh nhau tối đa 1.

**Dùng cho:** domain-incremental và revisit — nơi task mới nghĩa là **điều kiện mới**, không phải lớp mới.

---

## A6. `build_domain_stream()`

```python
def build_domain_stream(train_labels, val_labels, test_labels,
                        num_classes, num_tasks, seed=0) -> List[TaskSpec]
```

Giống `build_stream` nhưng dùng `chia_deu_theo_lop`, và **mọi `TaskSpec.classes` đều là toàn bộ lớp**:

```python
moi_lop = sorted(set(int(y) for y in train_labels))
tr = chia_deu_theo_lop(train_labels, num_tasks, seed=seed)
va = chia_deu_theo_lop(val_labels,   num_tasks, seed=seed + 1)   # seed lệch -> 3 split chia khác nhau
te = chia_deu_theo_lop(test_labels,  num_tasks, seed=seed + 2)
stream = [TaskSpec(task_id=t, classes=moi_lop, ...) for t in range(num_tasks)]
```

**Đọc kỹ docstring hàm này** — nó giải thích ngữ nghĩa ma trận R **đổi hoàn toàn**:

```
R[i][j] = accuracy ở ĐIỀU KIỆN j, sau khi đã học tới ĐIỀU KIỆN i
  đường chéo R[i][i] : hoạt động trong điều kiện HIỆN TẠI   <- chỉ số chính
  dưới chéo  R[i][j] : quay lại điều kiện CŨ có còn chạy không
```

"Forgetting" ở đây = **mất khả năng hoạt động ở điều kiện cũ**, không phải quên lớp. Cùng một ma trận R, cùng một hàm metric, nhưng **ý nghĩa khác hẳn**. Đây là lý do `metrics/revisit.py` phải ra đời (Bài 3).

**Vì sao tách khỏi `build_stream`:** docstring nói thẳng — nếu trộn "lớp mới" và "điều kiện trôi" vào cùng một stream thì khi accuracy tụt, **không tách được nguyên nhân**.

---

## A7. `build_stream_with_holdout()` — cho open-set

```python
def build_stream_with_holdout(..., holdout: int = 5) -> tuple[List[TaskSpec], List[int]]
```

Giữ lại `holdout` lớp **không bao giờ train**, làm "mẫu lạ" để đo xem model có biết nói "tôi không biết" không.

```python
order = list(range(num_classes))
if shuffle_classes:
    random.Random(seed).shuffle(order)   # ⭐ CÙNG RNG/seed với split_classes -> nhất quán
held = sorted(order[-holdout:])          # cắt ĐUÔI sau xáo làm lớp lạ
kept = order[:-holdout]
# ... rồi chia `kept` thành num_tasks nhóm y như split_classes
return stream, held
```

**Chi tiết tinh tế:** dùng **đúng** `random.Random(seed).shuffle(order)` như trong `split_classes`. Cùng seed → cùng phép xáo → lớp giữ lại tái lập được, và stream trên phần còn lại khớp với run không-holdout ở phần đầu.

---

## A8. `stratified_split()` — dụng cụ phụ

```python
def stratified_split(indices, labels, fraction, seed=0) -> tuple[List[int], List[int]]
```

Cắt `indices` thành (phần lớn, phần nhỏ) **giữ tỉ lệ từng lớp**. Dùng khi dataset không có split sẵn (EuroSAT).

```python
for c in sorted(by_class):
    idxs = by_class[c][:]              # [:] = copy, không sửa bản gốc
    rng.shuffle(idxs)
    k = int(round(len(idxs) * fraction))
    if len(idxs) > 1:
        k = min(max(k, 1), len(idxs) - 1)   # ⭐ mỗi bên ÍT NHẤT 1 mẫu
    else:
        k = 0                               # lớp chỉ 1 mẫu -> dồn hết vào phần lớn
    small += idxs[:k]; big += idxs[k:]
return sorted(big), sorted(small)
```

**Dòng có ⭐** là chỗ dễ sai nhất: không kẹp thì lớp hiếm (5 mẫu, fraction=0.1 → `round(0.5)=0`) sẽ **biến mất khỏi tập val** → metric trên lớp đó vô nghĩa.

---

## A9. `describe_stream()`

Chỉ để **in ra cho người xem**, không ảnh hưởng train. Trả chuỗi kiểu:

```
task 0: 5 classes [airplane, airport, ...] | train 1050 / val 175 / test 350
```

**Nên luôn in ra ở đầu mỗi run** (`run_g1.py` có làm). Nhìn một cái là biết chia task đúng chưa, trước khi đốt vài tiếng máy.

---

# PHẦN B — `metrics/continual.py` (83 dòng)

**Nhiệm vụ:** biến ma trận R thành các con số so sánh được.
**Thuần numpy.** Đọc ngay sau `stream.py` vì hai file này khép kín một vòng: một cái tạo task, một cái chấm điểm task.

**Nhắc lại quy ước:** `R[i, j]` = accuracy trên task `j` **sau khi** học xong task `i`. Hàng = giai đoạn train, cột = task được chấm. Chỉ **đường chéo và tam giác dưới** có nghĩa; tam giác trên bị bỏ qua (trừ khi bật `eval_future`).

Ví dụ để bám theo, 3 task:

```
        task0   task1   task2
after0  0.90     —       —
after1  0.70    0.85     —
after2  0.60    0.75    0.88
```

---

## B1. `average_accuracy(R)` — 3 dòng

```python
return float(np.mean(R[-1, :]))
```

Trung bình **hàng cuối**. Ví dụ: `(0.60 + 0.75 + 0.88)/3 = 0.743`.

**Nghĩa:** học xong hết, trung bình làm được bao nhiêu. **Cao = tốt.**

---

## B2. `average_forgetting(R)` ⭐ SỐ CHÍNH CỦA DỰ ÁN

```python
forgets = []
for j in range(T - 1):
    prev_best = np.max(R[j:T-1, j])      # đỉnh từng đạt trên task j, TRƯỚC giai đoạn cuối
    forgets.append(prev_best - R[-1, j])
return float(np.mean(forgets))
```

Ví dụ:
- task 0: đỉnh = `max(R[0,0], R[1,0]) = max(0.90, 0.70) = 0.90`; cuối = 0.60 → quên **0.30**
- task 1: đỉnh = `max(R[1,1]) = 0.85`; cuối = 0.75 → quên **0.10**
- trung bình = **0.20**

**Ba chi tiết dễ đọc lướt qua:**

1. **`range(T-1)`** — task cuối cùng không tính. Nó vừa học xong, chưa có cơ hội quên.
2. **`R[j:T-1, j]`** — lấy từ hàng `j` (lúc vừa học task j) tới hàng `T-2`, **không gồm hàng cuối**. Vì hàng cuối chính là thứ đem trừ.
3. **Thấp = tốt.** Ngược chiều với `average_accuracy`. Đọc bảng kết quả phải nhớ chiều.

Bảng G1: finetune quên 0.61, replay quên 0.079. Chênh **7,7 lần** — đó là toàn bộ giá trị của việc chống quên.

---

## B3. `backward_transfer(R)` — BWT

```python
diffs = [R[-1, j] - R[j, j] for j in range(T - 1)]
return float(np.mean(diffs))
```

Với mỗi task cũ `j`: (điểm cuối) − (điểm ngay khi vừa học xong nó).

- task 0: `0.60 − 0.90 = −0.30`
- task 1: `0.75 − 0.85 = −0.10`
- BWT = **−0.20**

**Âm = học cái mới làm cái cũ tệ đi** (bình thường). **Dương = học cái mới làm cái cũ TỐT LÊN** — hiếm và đáng chú ý. Replay trên EuroSAT đạt BWT **+0.225**: ôn bài trên buffer khiến task cũ tốt hơn cả lúc mới học xong.

**Khác `average_forgetting` chỗ nào?** BWT so với **lúc vừa học xong**, forgetting so với **đỉnh từng đạt**. Nếu accuracy task j còn tăng sau đó rồi mới tụt, hai số sẽ khác nhau.

---

## B4. `average_anytime_accuracy(R)` — AAA

```python
stage_means = [float(np.mean(R[t, :t+1])) for t in range(T)]
return float(np.mean(stage_means))
```

- sau task 0: `mean(0.90) = 0.900`
- sau task 1: `mean(0.70, 0.85) = 0.775`
- sau task 2: `mean(0.60, 0.75, 0.88) = 0.743`
- AAA = **0.806**

**Vì sao cần:** `average_accuracy` chỉ nhìn hàng cuối. UAV dùng model **liên tục trong lúc học**, không đợi hết stream. Model sập ở giữa hành trình rồi hồi lại cuối kỳ vẫn bị AAA phạt, dù acc cuối đẹp.

**`R[t, :t+1]`** — chỉ lấy phần **đã thấy** ở mốc `t`. Không lấy cả hàng (các ô chưa đo là 0, sẽ kéo trung bình xuống sai).

---

## B5. `forward_transfer(R, chance=0.0)` — FWT

```python
vals = [R[j-1, j] - chance for j in range(1, T)]
return float(np.mean(vals))
```

`R[j-1, j]` = accuracy trên task `j` **ngay trước khi** học nó — ô **tam giác trên**.

**Hai cảnh báo trong docstring:**
1. Ô này chỉ được điền khi bật `train.eval_future: true`. Không bật thì `R[j-1,j] = 0` và FWT vô nghĩa. `run_g1.py` xử lý bằng cách trả `None` nếu cờ tắt.
2. Với head khởi tạo mới, FWT thường ≈ mức đoán mò. Metric này chỉ có ý nghĩa từ **G2+** khi model mang bộ nhớ xuyên task.

---

## B6. `metrics/__init__.py` — mục lục

Không có logic, nhưng **đáng đọc như một mục lục**: `__all__` liệt kê 15 tên, chia 3 nhóm — continual (5), open-set (5), revisit (5, gồm 3 chỉ số O1/O2/O3 + 2 hàm gom/in báo cáo). Nhìn `__all__` là biết dự án đo những gì.

---

# PHẦN C — `models/classifier.py` (80 dòng)

**Nhiệm vụ:** model đơn giản nhất (baseline G1) + **quy ước mask** dùng chung cả dự án.

---

## C1. `mask_logits()` ⭐ 5 DÒNG QUAN TRỌNG NHẤT DỰ ÁN

```python
MASK_FILL = -1.0e4     # đủ nhỏ cho fp16/fp32 mà không sinh NaN như -inf

def mask_logits(logits: torch.Tensor, allowed: Sequence[int]) -> torch.Tensor:
    out = logits.new_full(logits.shape, MASK_FILL)
    idx = torch.as_tensor(list(allowed), dtype=torch.long, device=logits.device)
    out[:, idx] = logits[:, idx]
    return out
```

**Nhận:** logits `(B, C)` đủ mọi lớp, và danh sách lớp được phép.
**Trả:** tensor cùng shape, cột không được phép bị đè bằng `-1e4`.

**Vì sao đây là hàm quan trọng nhất:** nó là cách dự án cài đặt "chỉ xét một số lớp" mà **không cần đổi kiến trúc model**. Head luôn có đủ 45 cột từ task 0. Muốn giới hạn thì che, không phải cắt.

Quy ước dùng (viết trong docstring đầu file, giữ nguyên từ G1 tới G4):

| Thời điểm | `allowed` = |
|---|---|
| **TRAIN** task t | đúng các lớp của task t |
| **EVAL** sau task t | các lớp **đã thấy** (task 0..t) |

**Hai chi tiết kỹ thuật:**

- **`-1e4` chứ không phải `-inf`.** `-inf` đi qua softmax trong một số đường có thể sinh `NaN` (0 × inf). `-1e4` sau softmax ≈ 0 nhưng luôn là số hữu hạn.
- **`logits.new_full(...)`** tự lấy đúng dtype và device của `logits`. Viết `torch.full(...)` là sẽ lỗi device khi chạy GPU.

**Ai gọi:** `engine.train_one_task`, `engine.evaluate`, `engine._evaluate_ncm`, và các method (`Replay`, `LwF`, `LatentReplay`).

---

## C2. `ContinualClassifier` — 3 phương thức

```python
class ContinualClassifier(nn.Module):
    def __init__(self, backbone, feat_dim, num_classes, head="linear"):
        self.backbone = backbone
        self.head = build_head(head, feat_dim, num_classes)

    def forward(self, x):                 # ảnh -> feature -> logits (B, C)
        return self.head(self.backbone(x))

    def forward_from_feats(self, feats):  # feature ĐÃ CÓ -> logits, bỏ qua backbone
        return self.head(feats)
```

**`forward` là 1 dòng.** Đây chính là Hợp đồng 2: mọi model chỉ cần `forward(x) -> (B, C)`.

**`forward_from_feats` sinh ra vì `LatentReplay`** (#22): backbone đóng băng → feature cũ không "ôi" → lưu vector 384-d thay vì ảnh 224² (rẻ hơn ~400 lần). Nhưng để dùng lại feature đã lưu thì phải có đường **bỏ qua backbone**. Đó là hàm này.

> Ví dụ điển hình của việc **một yêu cầu ở tầng method đẻ ra một phương thức ở tầng model**. Khi đọc gặp một phương thức lạ, hỏi "ai gọi nó" thường ra ngay lý do tồn tại.

---

## C3. `CosineHead` — fix "recency bias"

```python
class CosineHead(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=16.0):
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)
        self.scale  = nn.Parameter(torch.tensor(float(scale)))
        self.out_features = int(num_classes)   # để mask_logits đọc như nn.Linear
        self.in_features  = int(feat_dim)

    def forward(self, x):
        xn = F.normalize(x, dim=1)              # feature -> vector đơn vị
        wn = F.normalize(self.weight, dim=1)    # trọng số lớp -> vector đơn vị
        return self.scale * (xn @ wn.t())       # cosine × scale -> logits
```

**Vấn đề nó giải:** với `nn.Linear`, độ lớn trọng số lớp mới **phình to** lấn lớp cũ → model thiên vị lớp vừa học. Đó là "recency bias", một nguyên nhân forgetting.

**Cách giải:** chuẩn hoá cả hai vế → logit chỉ phụ thuộc **hướng**, bất biến **độ lớn**. `scale` học được giữ logit đủ "sắc" cho cross-entropy.

**Bằng chứng ghi trong docstring** (Titans RESISC45 seed0):
```
head Linear : acc 0.583 / forgetting 0.274
NCM-head    : acc 0.758 / forgetting 0.078
```
Chênh 17,5 điểm chỉ do **cách đọc feature**, không đổi feature. `CosineHead` biến phát hiện đó thành fix cố định: "prototype" giờ chính là vector trọng số, được gradient cập nhật — không cần dựng prototype, không cần đọc lại data cũ.

**Chi tiết quan trọng:** khai báo `out_features` / `in_features` để **giao diện y hệt `nn.Linear`**. Nhờ vậy `mask_logits`, NCM-head shadow eval, metrics dùng chung không phải sửa gì. Bật bằng một dòng yaml: `head: cosine`.

---

## C4. `build_head()`

```python
def build_head(kind, feat_dim, num_classes, scale=16.0) -> nn.Module:
    kind = (kind or "linear").lower()
    if kind == "linear":  return nn.Linear(feat_dim, num_classes)
    if kind == "cosine":  return CosineHead(feat_dim, num_classes, scale=scale)
    raise ValueError(f"head '{kind}' không hợp lệ, chọn 'linear' | 'cosine'")
```

Mẫu **factory** xuất hiện 5 lần trong dự án (`build_head`, `build_backbone`, `build_method`, `get_source`, `build_optimizer`). Đều cùng khuôn: đọc một chuỗi từ config → trả object → tên lạ thì ném lỗi **kèm danh sách hợp lệ**.

---

# PHẦN D — `models/backbone.py` (63 dòng)

## D1. `build_backbone()`

```python
def build_backbone(backbone_cfg: dict) -> tuple[nn.Module, int]:
    name = str(backbone_cfg.get("name", "vit_small_patch16_224"))
    if name.lower() == "tinycnn":
        m = TinyCNN(); feat_dim = m.feat_dim
    else:
        import timm
        m = timm.create_model(name,
                              pretrained=bool(backbone_cfg.get("pretrained", True)),
                              num_classes=0)      # ⭐ bỏ head của timm, chỉ lấy feature
        feat_dim = int(m.num_features)
    if bool(backbone_cfg.get("freeze", False)):
        for p in m.parameters(): p.requires_grad_(False)
        m.eval()
    return m, feat_dim
```

**Trả về tuple `(module, feat_dim)`** — không phải chỉ module. Vì phần lắp ráp cần biết độ dài feature để dựng head. ViT-S → 384.

**`num_classes=0`** là mẹo của timm: không gắn head phân loại, `forward` trả thẳng feature. Nếu quên tham số này, model sẽ trả logits 1000 lớp ImageNet và mọi thứ sau đó sai.

**`import timm` đặt trong hàm**, không ở đầu file. Nhờ vậy nhánh `tinycnn` chạy được trên máy chưa cài timm. Kỹ thuật này lặp lại nhiều nơi (`from PIL import Image` trong `get_image`, `from datasets import load_dataset` trong `_load_resisc45`).

**Nhánh `freeze`:** tắt gradient **và** gọi `m.eval()`. Chỉ tắt gradient là chưa đủ — BatchNorm vẫn cập nhật thống kê chạy trong `train()` mode dù không có gradient.

---

## D2. `TinyCNN`

3 khối `Conv → BatchNorm → ReLU → MaxPool`, rồi Global Average Pooling → vector 64-d.

```python
def forward(self, x):
    return self.pool(self.features(x)).flatten(1)   # (B,3,H,W) -> (B,64)
```

Không pretrained, không cần mạng. Chỉ để **kiểm pipeline chạy thông**, không để lấy số đẹp. Dùng trong `configs/g1_smoke.yaml` và các test.

---

# PHẦN E — `data/sources.py` (194 dòng)

**Nhiệm vụ:** 3 dataset khác nhau → **cùng một hình dạng**.

## E1. `DataSource` + 3 lớp Split

```python
@dataclass
class DataSource:
    name: str
    num_classes: int
    class_names: List[str]
    splits: Dict[str, object]      # 'train' | 'val' | 'test' -> *Split
```

Mỗi `*Split` có đúng 3 thứ: `__len__()`, `get_image(i)`, `.labels`. Ba cài đặt khác nhau ở **chỗ ảnh nằm**:

| Lớp | Ảnh ở đâu | `get_image(i)` làm gì |
|---|---|---|
| `ImagePathSplit` | đĩa | `Image.open(self.paths[i]).convert("RGB")` |
| `HFSplit` | parquet HuggingFace | `self.ds[i]["image"].convert("RGB")` |
| `ArraySplit` | RAM (numpy uint8) | `Image.fromarray(self.arrays[i])` |

Cả ba đều trả **ảnh PIL RGB** → đi qua cùng một transform. Đây là chỗ "chuẩn hoá giao diện" xảy ra.

**Ghi chú cuối docstring đầu file:** *"Mọi class ở đây đều picklable (không dùng closure) để DataLoader num_workers>0 hoạt động trên cả macOS (spawn)."* macOS dùng `spawn` chứ không `fork`, nên object phải pickle được để copy sang tiến trình con. Dùng closure ở đây là hỏng, và lỗi sẽ khó hiểu.

## E2. `get_source()` — điểm vào duy nhất

```python
_REGISTRY = {"resisc45": _load_resisc45, "eurosat": _load_eurosat, "synthetic": _load_synthetic}

def get_source(data_cfg: dict) -> DataSource:
    name = str(data_cfg.get("name", "")).lower()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](data_cfg)
```

Thêm dataset mới = viết `_load_xxx` + thêm **một dòng** vào `_REGISTRY`. Không đụng file khác.

## E3. Ba hàm tải

- **`_load_resisc45`** — 45 lớp, 31.500 ảnh. Có sẵn 3 split → bọc thẳng. Số lớp và tên lớp lấy từ **metadata** (`ds["train"].features["label"]`), không hardcode.
- **`_load_eurosat`** — 10 lớp, không có split chính thức → tự chia bằng `stratified_split`: cắt 10% làm test, rồi trong 90% còn lại cắt 1/9 (≈10% tổng) làm val. Hai seed lệch nhau (`seed`, `seed+1`).
- **`_load_synthetic`** — mỗi lớp một màu trung bình riêng + nhiễu Gauss (σ=25). "Salt" khác nhau cho train/val/test để 3 tập không trùng. Model học được ngay → dùng smoke test, **không dùng để đánh giá hay/dở**.

---

# PHẦN F — `data/loaders.py` (230 dòng)

**Đây là chỗ đầu tiên cần torch.** File dài nhất trong `data/` vì gánh cả phần drift và pha 2.

## F1. `build_transforms(image_size, train)`

```python
if train:
    return T.Compose([T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                      T.RandomHorizontalFlip(),
                      T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
return T.Compose([T.Resize((image_size, image_size)),
                  T.ToTensor(),
                  T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
```

**Train có augment ngẫu nhiên, eval không.** Nếu eval cũng augment thì mỗi lần chấm ra số khác nhau → không so sánh được.

`IMAGENET_MEAN/STD` phải khớp backbone pretrained của timm. Sai chuẩn hoá là mất vài điểm accuracy **không rõ lý do** — bẫy im lặng, rất khó tìm.

## F2. `TaskDataset` — 3 phương thức

```python
def __getitem__(self, k):
    i = self.indices[k]                            # chỉ số cục bộ -> chỉ số thật trong split
    x = self.transform(self.split.get_image(i))
    y = int(self.split.labels[i])                  # ⭐ nhãn GỐC, không remap
    return x, y
```

**Dòng có ⭐ là Hợp đồng 1.** Docstring nói rõ: *"Nhãn giữ nguyên id toàn cục (không remap) — head có đủ cột cho mọi class, việc giới hạn class nào được dùng do mask logits ở engine quyết định."*

Nếu remap nhãn về `0..4` cho mỗi task thì `mask_logits` vô dụng, và eval trên tập lớp đã thấy sẽ không làm được.

## F3. `build_task_loaders()` ⭐ hàm bạn thật sự gọi

```python
def build_task_loaders(source, stream, data_cfg) -> List[Dict[str, DataLoader]]
```

**Trả:** danh sách, phần tử `t` là `{"train": DataLoader, "val": ..., "test": ...}` của task `t`. Chính là thứ `engine.run_continual` nhận vào.

Cấu trúc hàm: đọc config → 3 khối cờ tuỳ chọn → 2 hàm nội bộ → vòng lặp cuối.

**Hai hàm nội bộ đáng đọc:**

```python
def tf_cua_task(t, split_name, train):
    """Transform của task t. Không bật drift -> trả đúng transform cũ."""
    if not drift_bat or split_name not in ap_cho:
        return tf_train if train else tf_eval          # ← đường cũ, bất biến
    if lich is not None:                                # revisit: mức trôi theo LỊCH BAY
        cfg_t = dict(drift_cfg, mode="linear", severity=lich[t].muc_troi)
        return build_drift_transform(image_size, 1, 2, cfg_t, train=train, seed=...)
    return build_drift_transform(image_size, t, len(stream), drift_cfg, train=train, seed=...)
```

Mẹo ở nhánh revisit: gọi `build_drift_transform(task_idx=1, num_tasks=2, severity=muc)` để **ép** đúng mức trôi mong muốn, thay vì viết lại logic dựng transform lần thứ hai. Comment giải thích rõ — đáng học.

```python
def dl(split_name, idx, train, t) -> DataLoader:
    if dong_tg and train and t >= chuyen_hc and split_name in ap_cho:
        ds = TaskDatasetDongThoiGian(...)               # pha 2: dòng thời gian
        return DataLoader(ds, batch_size=..., shuffle=not p2_thu_tu, ...)
    ds = TaskDataset(source.splits[split_name], idx, tf_cua_task(t, split_name, train))
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=train,                    # ⭐ chỉ xáo khi train
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=False)                  # giữ lô cuối, không bỏ mẫu
```

**`shuffle=train`** — một biểu thức, hai ý: train thì xáo, val/test thì giữ thứ tự cố định để chấm điểm tái lập được.

**Nguyên tắc lặp lại lần nữa:** mọi khối drift/pha-2 đều bắt đầu bằng `if not bật: return đường_cũ`. Bật cờ mới thì run cũ **bất biến**.

## F4. `TaskDatasetDongThoiGian` — pha 2

Khác `TaskDataset` ở **ba điểm**, đều bám phát biểu bài toán gốc:

1. **mức trôi tính theo vị trí mẫu trong chuyến** — yêu cầu (c) "nắng gắt dần suốt 10 phút"
2. **transform kiểu eval** (resize, không crop/flip ngẫu nhiên) — dữ liệu triển khai không phải dữ liệu augment
3. **đi kèm `shuffle=False`** — vị trí `k` = thời gian; xáo trộn là **xoá thời gian**

```python
def muc_tai(self, k: int) -> float:
    n = len(self.indices)
    tien_do = k / max(n - 1, 1)                       # 0 (đầu chuyến) -> 1 (cuối chuyến)
    m = self.muc_chuyen + self.bien_do * (tien_do - 0.5)
    return float(min(1.0, max(0.0, m)))               # kẹp về [0,1]

def __getitem__(self, k):
    i = self.indices[k]
    x = self._pre(self.split.get_image(i))            # PIL -> tensor [0,1]
    m = self.muc_tai(k)
    if m > 0.0:
        x = self._ApDungTroi(m, jitter=self.jitter, seed=self.seed + k)(x)
    return self._norm(x), int(self.split.labels[i])
```

**`seed=self.seed + k`** — seed theo **từng mẫu**, nên mỗi khung hình tái lập được độc lập.

**Thứ tự phép biến đổi** giữ đúng nguyên tắc của `drift.py`: hình học → `ToTensor` → **TRÔI** (trên thang `[0,1]`) → `Normalize`.

## F5. `build_eval_loader()`

DataLoader eval trên **danh sách chỉ số tuỳ ý** của một split, không aug không xáo. Dùng cho open-set: gom mẫu test của các lớp giữ lại thành một loader.

---

# PHẦN G — `utils/` (86 dòng, đọc trong 10 phút)

## G1. `seed_everything(seed=0)` — 23 dòng

```python
random.seed(seed)                        # random của Python
np.random.seed(seed)                     # random của numpy
os.environ["PYTHONHASHSEED"] = str(seed) # cố định hash -> thứ tự set/dict ổn định
try:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
except ImportError:
    pass
return seed
```

**Phải seed cả bốn chỗ.** Sót một chỗ là hai lần chạy ra số khác nhau, và bạn sẽ mất nửa ngày đi tìm.

**`PYTHONHASHSEED`** ít người biết: Python băm chuỗi ngẫu nhiên theo mỗi lần khởi động, làm thứ tự duyệt `set` thay đổi. Nếu code có `for x in some_set` thì thứ tự đó ảnh hưởng kết quả.

**Bọc `try/except ImportError`** để file này chạy được cả khi chưa cài torch.

## G2. `load_config` / `save_config`

```python
cfg = yaml.safe_load(f)                  # safe_load: không chạy code lạ trong file
if not isinstance(cfg, dict): raise ValueError(...)

yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
# sort_keys=False -> giữ thứ tự key; allow_unicode=True -> ghi thẳng tiếng Việt
```

`save_config` được gọi cuối mỗi run để lưu config **đã dùng** (sau khi override) vào thư mục kết quả → tái lập được chính xác run đó.

## G3. `apply_overrides(cfg, pairs)` — `--set a.b=c`

```python
for pair in pairs or []:
    key, _, raw = pair.partition("=")
    if not _: raise ValueError(f"Override '{pair}' must look like key.sub=value")
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})     # lặn xuống, tạo nhánh nếu chưa có
    node[parts[-1]] = _coerce(raw.strip())
return cfg
```

Cho phép `--set train.lr=1e-4 data.num_tasks=5` mà không mở file yaml ra sửa. Cực tiện khi chạy quét tham số.

## G4. `_coerce(v)` — bẫy `1e-4`

```python
out = yaml.safe_load(v)                   # mượn parser yaml đoán kiểu
if out is None and v.strip() not in ("null", "~", ""):
    return v
if isinstance(out, str) and any(ch.isdigit() for ch in out):
    try:    return float(out)             # ⭐ '1e-4' -> 0.0001
    except ValueError: return out
return out
```

**Bẫy thật:** YAML **không** coi `1e-4` là số (chuẩn YAML đòi `1.0e-4`), nên `safe_load("1e-4")` trả về **chuỗi** `"1e-4"`. Không có dòng ⭐ thì `--set train.lr=1e-4` sẽ đặt learning rate thành một chuỗi, và torch sẽ báo lỗi ở chỗ khác hoàn toàn.

## G5. `get_device()` trong `utils/__init__.py`

```python
if torch.cuda.is_available():  return "cuda"      # GPU NVIDIA
mps = getattr(torch.backends, "mps", None)
if mps is not None and mps.is_available(): return "mps"   # GPU Apple Silicon
return "cpu"
```

`getattr(..., None)` vì `torch.backends.mps` không tồn tại ở torch cũ. `engine.resolve_device()` làm y hệt nhưng trả `torch.device` thay vì chuỗi.

---

# Checklist Bài 1

- [ ] Chạy `pytest tests/test_stream.py tests/test_metrics.py -q` — xanh
- [ ] Gọi tay `split_classes(10, 3, seed=0)` và `seed=1`, thấy khác nhau
- [ ] Tự tính `average_forgetting` trên ma trận 3×3 ví dụ ở Phần B, ra **0.20**
- [ ] Giải thích được vì sao `mask_logits` dùng `-1e4` chứ không phải `-inf`
- [ ] Giải thích được vì sao nhãn **không** được remap về `0..4` mỗi task
- [ ] Chạy `python scripts/run_g1.py --config configs/g1_smoke.yaml`, mở `acc_matrix.csv`

Xong checklist → sang **Bài 2: engine + methods**.
