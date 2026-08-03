# Kế hoạch: SLDA + λ và bộ dữ liệu có TRÔI

Ngày: 2026-08-03 · Nhánh đề xuất: `feat/slda-drift`

> **Trạng thái: ĐỀ XUẤT — chưa code.** Tài liệu này để chốt thiết kế trước.

## Câu hỏi thí nghiệm

> Khi điều kiện quan sát **trôi dần** (nắng, mùa, sương mù), SLDA hiện tại có bám theo được
> không? Nếu không, hệ số quên `λ` có sửa được không, và λ tối ưu phụ thuộc tốc độ trôi ra sao?

Và câu hỏi phụ, quan trọng cho đề tài:

> Ở regime trôi này, **Titans (đã vá cổng) có thắng SLDA không?** Đây là regime mà lập luận
> của Nested Learning mạnh nhất, và chưa ai đo.

---

# NHÓM 0 — Bốn quyết định thiết kế cần chốt TRƯỚC

Đây là phần quan trọng nhất. Chọn sai thì chạy xong mới biết thí nghiệm vô nghĩa —
đúng lỗi đã mắc với smoke 3 task lần trước.

## Q1. Stream kiểu gì? — **domain-incremental**, không phải class-incremental

Hiện tại `stream.py:build_stream` chia **45 lớp thành 9 nhóm**, mỗi task học lớp MỚI.
Đó là class-incremental.

Cho thí nghiệm trôi, cần ngược lại: **mọi task đều có ĐỦ 45 lớp, chỉ khác điều kiện quan sát.**

| | Class-incremental (hiện tại) | **Domain-incremental (cần thêm)** |
|---|---|---|
| Task 0 | lớp 0–4, ảnh gốc | **45 lớp**, mức trôi 0% |
| Task 4 | lớp 20–24, ảnh gốc | **45 lớp**, mức trôi 50% |
| Task 8 | lớp 40–44, ảnh gốc | **45 lớp**, mức trôi 100% |

**Vì sao bắt buộc tách:** nếu để lẫn cả hai (lớp mới + trôi), không tách được "tụt vì quên lớp"
khỏi "tụt vì trôi điều kiện". Kết quả sẽ không diễn giải được.

→ Cần hàm mới `build_domain_stream()` trong `stream.py`. Không sửa `build_stream` cũ.

## Q2. Tập test có trôi theo không? — **CÓ**, và đây là chỗ tinh tế nhất

Hai lựa chọn, ý nghĩa hoàn toàn khác nhau:

| | Test giữ ảnh gốc | **Test trôi theo task** |
|---|---|---|
| Đo cái gì | Model tụt bao nhiêu so với điều kiện lý tưởng | **Model hoạt động thế nào trong điều kiện HIỆN TẠI** |
| Giống thực tế | Không — drone không bao giờ thấy ảnh "gốc" | **Có** — drone bay trong điều kiện hôm nay |

Chọn **test trôi theo**. Khi đó ma trận accuracy vẫn dùng được và mang ý nghĩa mới:

```
R[i][j] = accuracy trên test ở ĐIỀU KIỆN j, sau khi đã học tới ĐIỀU KIỆN i
```

| Vùng ma trận | Nghĩa |
|---|---|
| **Đường chéo** `R[i][i]` | Hoạt động trong điều kiện hiện tại — **chỉ số chính** |
| **Dưới chéo** `R[i][j], j<i` | Quay lại điều kiện cũ có còn chạy không (mùa hè năm sau) |
| Trên chéo | Điều kiện tương lai — bỏ qua (`eval_future=false`) |

Và **"forgetting" ở đây đổi nghĩa**: không phải quên lớp, mà là **mất khả năng hoạt động ở
điều kiện cũ**. Cần ghi rõ trong báo cáo, kẻo người đọc hiểu nhầm.

## Q3. Trôi bằng phép biến đổi gì? — 5 phép, mô phỏng điều kiện thật

| Phép | Mô phỏng | Dải (0% → 100%) |
|---|---|---|
| Độ sáng | giờ trong ngày, mùa | ×1,0 → ×0,55 |
| Nhiệt độ màu | bình minh ấm → trưa lạnh | 0 → ±18% lệch kênh R/B |
| Tương phản | sương mù, khói mù | ×1,0 → ×0,65 |
| Mờ Gauss | độ cao, ống kính bẩn | σ 0 → 1,4 px |
| Nhiễu | ISO cao khi thiếu sáng | σ 0 → 0,035 |

**Quan trọng — trôi phải TẤT ĐỊNH, không phải nhiễu ngẫu nhiên:**

- **Mức trung bình** dịch theo task một cách xác định — đây mới là "trôi"
- Cộng **rung nhẹ ngẫu nhiên** ±15% quanh mức đó — mô phỏng mỗi chuyến bay hơi khác

Nếu để hoàn toàn ngẫu nhiên thì nó chỉ là augmentation, không phải drift, và λ sẽ vô dụng.

## Q4. λ áp thế nào? — khác nhau cho từng đại lượng

| Đại lượng | Phân rã khi nào | Vì sao |
|---|---|---|
| `s_c`, `n_c` (bậc 1, D số/lớp) | **chỉ khi lớp c xuất hiện** | Lớp hiếm không bị xoá → giữ được tính "không quên lớp" |
| `G` (bậc 2, D² số, dùng chung) | **theo đồng hồ toàn cục** | Là thống kê chung của toàn stream |

Nếu để `s_c` phân rã theo đồng hồ toàn cục, lớp hiếm gặp sẽ bị xoá sạch — **mất luôn điểm
mạnh nhất của SLDA**. Đây là bẫy dễ mắc nhất khi cài.

**Hai λ riêng biệt:**

```yaml
slda:
  decay_mean: 0.999    # cho μ_c — bám trôi nhanh (ít tham số, ước lượng dễ)
  decay_cov:  0.9999   # cho Σ   — ổn định hơn (D² tham số, cần nhiều mẫu)
```

λ nhỏ cho `Σ` là nguy hiểm: `Σ` có 384² = 147.456 số, cần đủ mẫu hiệu dụng mới ước lượng nổi.
λ = 0,99 chỉ cho ~100 mẫu — chắc chắn hỏng.

Tách hai λ có thể là **đóng góp riêng** của đề tài, chưa thấy ai làm với SLDA. Và nó khớp
thẳng với ý CMS của NL: **hai tầng ở hai tần số**, khác ở chỗ tần số do người đặt.

---

# NHÓM 1 — Code `λ` cho SLDA

## TASK D1 — Thêm `decay_mean` / `decay_cov` vào `SLDAClassifier`

**File:** `src/uavcl/models/slda.py`

```python
def __init__(..., decay_mean: float = 1.0, decay_cov: float = 1.0):
    # 1.0 = không quên = hành vi hiện tại, BẤT BIẾN NGƯỢC
    if not (0 < decay_mean <= 1.0): raise ValueError(...)
    if not (0 < decay_cov  <= 1.0): raise ValueError(...)
```

Trong `update()`:

```python
# G phân rã theo đồng hồ toàn cục (thống kê chung)
if self.decay_cov < 1.0:
    self.gram.mul_(self.decay_cov ** len(ys))     # ↳ luỹ thừa theo số mẫu trong batch
self.gram.add_(f.t() @ f)

# s_c, n_c phân rã CHỈ ở các lớp có mặt trong batch này
if self.decay_mean < 1.0:
    co_mat = torch.zeros_like(self.count).index_add_(0, ys, torch.ones_like(ys, dtype=...))
    he_so  = torch.where(co_mat > 0, self.decay_mean ** co_mat, torch.ones_like(co_mat))
    self.feat_sum.mul_(he_so.unsqueeze(1))
    self.count.mul_(he_so)
self.feat_sum.index_add_(0, ys, f)
self.count.index_add_(0, ys, ...)
```

**Rủi ro:** THẤP. Mặc định `1.0` → `mul_(1.0)` → không đổi gì.
**Thời gian:** 45 phút.

## TASK D2 — Log cửa sổ nhớ hiệu dụng

**File:** `src/uavcl/methods.py`, `SLDA.end_task`

In `n_c` trung bình sau mỗi task. Với λ < 1 nó **hội tụ về `1/(1−λ)`** — đó là bằng chứng
trực tiếp cơ chế đang chạy, giống vai trò của `trace_gates.py` với Titans.

```
[slda] cua so nho: n_c trung binh = 987 (ky vong 1/(1-0.999) = 1000)
```

Lệch nhiều so với kỳ vọng = cài sai. **Đây là cửa chặn tự động.**

**Thời gian:** 20 phút.

## TASK D3 — Test

**File mới:** `tests/test_slda_decay.py`

| Test | Kiểm gì |
|---|---|
| `decay=1.0` cho kết quả **trùng khít** bản cũ | bất biến ngược |
| `n_c` hội tụ về `1/(1−λ)` | công thức cửa sổ đúng |
| Lớp **không xuất hiện** thì `s_c`, `n_c` **không đổi** | ⭐ bẫy Q4 |
| μ_c bám theo trung bình trôi (mô phỏng như bảng đã tính) | cơ chế đúng |
| `decay ≤ 0` hoặc `> 1` bị từ chối | validate |

Test thứ ba quan trọng nhất — nó chặn đúng cái bẫy dễ mắc nhất.

**Thời gian:** 45 phút.

---

# NHÓM 2 — Code bộ dữ liệu có TRÔI

## TASK D4 — Module phép biến đổi trôi

**File mới:** `src/uavcl/data/drift.py`

```python
def build_drift_transform(image_size, task_idx, num_tasks, drift_cfg, train, seed):
    """Trả transform với cường độ trôi = f(task_idx / (num_tasks-1))."""
```

- `mode: linear` — cường độ tăng tuyến tính theo task
- `mode: step` — nhảy bậc (dùng để so, ít thực tế hơn)
- Rung ngẫu nhiên ±15% quanh mức trung bình, **gieo bằng `seed`** → tái lập được

**Thời gian:** 1 giờ.

## TASK D5 — Stream domain-incremental

**File:** `src/uavcl/data/stream.py` — thêm hàm mới, **không sửa `build_stream`**

```python
def build_domain_stream(labels_train, labels_val, labels_test, num_tasks, seed):
    """Mọi task có ĐỦ mọi lớp; chia mẫu đều cho các task.
    Task khác nhau ở ĐIỀU KIỆN (do transform), không ở tập lớp."""
```

Chia ngẫu nhiên có phân tầng theo lớp để mỗi task có đủ mọi lớp với số lượng cân bằng.

**Thời gian:** 45 phút.

## TASK D6 — Nối vào loader

**File:** `src/uavcl/data/loaders.py:66` `build_task_loaders`

Hiện tại dựng `tf_train`/`tf_eval` **một lần cho mọi task** (dòng 73–74). Phải đổi thành
**per-task** khi bật drift.

```python
for t, spec in enumerate(stream):
    if drift_bat:
        tf_tr = build_drift_transform(..., task_idx=t, train=True, ...)
        tf_ev = build_drift_transform(..., task_idx=t, train=False, ...)
    else:
        tf_tr, tf_ev = tf_train, tf_eval        # ↳ đường cũ, bất biến
```

**Rủi ro:** TRUNG BÌNH — đụng vào đường dữ liệu dùng chung. Phải giữ nhánh cũ nguyên vẹn.
**Thời gian:** 45 phút.

## TASK D7 — Config

**File mới:** `configs/drift_resisc45_base.yaml`

```yaml
data:
  num_tasks: 9
  stream_type: domain           # ← mới; mặc định 'class' = hành vi cũ
  drift:
    enabled: true
    mode: linear
    severity: 1.0               # hệ số nhân, dùng để hiệu chỉnh ở D8
    jitter: 0.15
    apply_to: [train, val, test]
slda:
  decay_mean: 1.0               # arm gốc; các arm khác override
  decay_cov: 1.0
```

**Thời gian:** 20 phút.

---

# NHÓM 3 — Hiệu chỉnh & kiểm chứng ⭐ ĐỪNG BỎ QUA

Đây là phần rút ra từ sai lầm lần trước: **kiểm setup trước khi đốt giờ**.

## TASK D8 — Hiệu chỉnh cường độ trôi (CỬA CHẶN)

**File mới:** `scripts/calibrate_drift.py`

Chạy **NCM đóng băng** (rẻ, không train) trên test set ở từng mức trôi, in bảng:

```
muc troi   0%    25%    50%    75%   100%
NCM acc   0.71   0.67   0.62   0.57   0.52
```

| Kết quả | Nghĩa | Xử lý |
|---|---|---|
| 100% vẫn ≈ 0,70 | **Trôi quá nhẹ** — không có gì để đo | tăng `severity` |
| 100% tụt còn ~0,50–0,55 | ✅ **Vừa đúng** | chạy tiếp |
| 100% tụt dưới 0,25 | **Trôi quá nặng** — feature vỡ, mọi phương pháp đều chết | giảm `severity` |

**Chưa vào được vùng xanh thì chưa chạy campaign.** Mất 15 phút, tránh mất 8 tiếng.

**Thời gian:** 45 phút viết + 15 phút chạy.

## TASK D9 — Xuất ảnh mẫu để nhìn bằng mắt

**File:** thêm cờ `--dump-samples` vào `calibrate_drift.py`

Lưu 1 ảnh của cùng một lớp ở 9 mức trôi, xếp cạnh nhau thành lưới.

**Vì sao cần:** con số accuracy không cho biết ảnh trông có **hợp lý** không. Ảnh trôi 100%
phải trông như *"cùng cảnh đó, chụp lúc chiều muộn nhiều sương"* — chứ không phải như ảnh hỏng.

Ảnh này cũng **đưa thẳng vào báo cáo** được.

**Thời gian:** 20 phút.

---

# NHÓM 4 — Campaign

## TASK D10 — Quét λ, 5 arm × 3 seed

| Arm | `decay_mean` | `decay_cov` | Ý nghĩa |
|---|---|---|---|
| 1 | 1,0 | 1,0 | **SLDA hiện tại** — kỳ vọng tụt dần |
| 2 | 0,9999 | 0,9999 | quên rất chậm |
| 3 | **0,999** | **0,9999** | ⭐ **cấu hình đề xuất** — hai λ khác nhau |
| 4 | 0,99 | 0,999 | quên nhanh |
| 5 | 0,99 | 0,99 | ⚠️ đối chứng: λ nhỏ cho Σ **phải hỏng** |

Arm 5 để chứng minh **Σ cần nhiều mẫu hơn μ** — nếu nó không hỏng thì lập luận tách hai λ sai.

**Thời gian:** ~6 giờ máy.

## TASK D11 — Titans trên cùng stream trôi

Chạy config `_gates.yaml` (đã vá cổng) với `stream_type: domain` + drift, seed 0.

**Đây là lần đầu Nested Learning được đo ở regime của chính nó.** Dù kết quả ra sao cũng
là nội dung cho chương 4.

**Thời gian:** ~4 giờ máy.

---

# Thứ tự & tổng chi phí

```
Q1–Q4 chốt thiết kế  (đọc + duyệt)
   ↓
D1 → D2 → D3        (λ, ~2 giờ, rủi ro thấp)
   ↓
D4 → D5 → D6 → D7   (dataset trôi, ~3 giờ, rủi ro trung bình ở D6)
   ↓
D8 → D9  ⭐ CỬA CHẶN (~1,5 giờ — hiệu chỉnh trước khi đốt giờ máy)
   ↓
D10 (6h máy)  →  D11 (4h máy)
```

| | Công sức | Máy |
|---|---|---|
| Code + test | ~5 giờ | — |
| Hiệu chỉnh | ~1,5 giờ | — |
| Campaign | — | ~10 giờ |

---

# Bảng kết quả cần điền & cách đọc

| Arm | Acc đường chéo (điều kiện hiện tại) | Acc điều kiện CŨ | Chú thích |
|---|---|---|---|
| λ = 1,0 | | | |
| λ_μ 0,999 / λ_Σ 0,9999 | | | |
| Titans (đã vá) | | | |

## Bốn kịch bản

| Quan sát | Kết luận |
|---|---|
| λ=1 tụt dần theo task, λ<1 giữ được | ✅ **λ là lời giải cho trôi.** Đóng góp rõ ràng |
| Mọi λ đều tụt như nhau | Trôi vượt khả năng feature đóng băng → phải mở băng backbone |
| λ<1 giữ điều kiện mới nhưng **mất điều kiện cũ** | Đánh đổi thật — cần λ theo tốc độ trôi, hoặc hai tầng nhớ |
| **Titans thắng** | 🔥 **Cứu được hướng NL** — bằng chứng luận điểm NL nằm ở domain shift |

Kịch bản cuối là kịch bản duy nhất trong toàn bộ dự án có thể lật ngược kết luận hiện tại.
Đó là lý do **D11 đáng chạy dù D10 ra sao**.

---

# Rủi ro đã lường trước

| Rủi ro | Cách chặn |
|---|---|
| Trôi quá nhẹ/nặng → thí nghiệm vô nghĩa | **D8 là cửa chặn**, hiệu chỉnh trước |
| λ áp sai (phân rã lớp vắng mặt) | Test riêng ở **D3** |
| `Σ` với λ nhỏ ước lượng vỡ | Tách hai λ + **arm 5 làm đối chứng** |
| Sửa `loaders.py` phá run cũ | `drift.enabled=false` → đi đúng nhánh cũ, không đụng |
| Ma trận accuracy bị hiểu nhầm | Ghi rõ ngữ nghĩa mới trong **Q2** vào báo cáo |

---

# Cần bạn duyệt trước khi mình code

1. **Q1** — stream domain-incremental riêng, không trộn với class-incremental. Đồng ý?
2. **Q2** — test set trôi theo task (thay vì giữ ảnh gốc). Đồng ý?
3. **Q4** — tách hai λ cho μ và Σ. Đồng ý, hay muốn một λ chung cho đơn giản?
4. Có muốn làm luôn **D11** (Titans trên stream trôi) không, hay để sau?

Duyệt xong mình code một mạch D1 → D9, rồi bạn chạy D8 để hiệu chỉnh trước khi vào campaign.
