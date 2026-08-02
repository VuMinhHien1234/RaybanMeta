# B1 — Xác minh SLDA `0.8263`

Ngày: 2026-08-02 · Artifact: `result_test/res_base/artifacts_slda_s{0,1,2}`
Mã nguồn: `src/uavcl/models/slda.py` · `src/uavcl/methods.py:386-405` · `src/uavcl/engine.py:273-276`

## KẾT LUẬN: ĐẠT cả 4 điểm. Kết quả `0.8263` dùng được.

---

## Kiểm 1 — Test set đúng 45 lớp, cùng split với các phương pháp khác ✅

| Kiểm | Kết quả |
|---|---|
| `num_tasks` | 9 |
| `acc_matrix` | 9 hàng × 9 cột = 9 task × 5 lớp = **45 lớp** |
| `split_protocol` | **`combined31500_v2`** — trùng với campaign Titans |
| `expected_total_samples` | 31 500 |
| Tam giác trên (task chưa học) | **toàn 0** ở cả 3 seed → không đánh giá task chưa học |

Split trùng là điểm quan trọng nhất — nghĩa là `0.8263` **so trực tiếp được** với
Frozen NCM `0.7114`, γ 0.25 `0.7206`, oracle `0.7459`. (Khác hẳn `artifacts_m3_s0`
dùng split cũ, không so được.)

## Kiểm 2 — Mỗi ảnh chỉ thấy đúng một lần ✅

```python
# methods.py:393
gradient_free = True        # -> engine bỏ vòng train gradient

# engine.py:273-276
if getattr(method, "gradient_free", False):
    method.fit_task(model, task_loaders[t]["train"], device)   # gọi MỘT lần

# methods.py:396-402
def fit_task(self, model, loader, device):
    for x, y in loader:                     # duyệt loader ĐÚNG một vòng
        feats = model.backbone(x.to(device))
        model.update(feats, y.to(device))
```

`epochs_per_task: 3` trong config **không được áp dụng** cho SLDA — engine rẽ nhánh
`gradient_free` trước khi tới vòng epoch. Mỗi ảnh đi qua đúng một lần.

Và `task_loaders[t]["train"]` chỉ là loader của **task hiện tại** → không chạm data cũ.

> **Ghi chú:** kể cả nếu có lặp k epoch thì SLDA vẫn bất biến về mặt toán:
> `μ_c = (k·Σf)/(k·n_c)` và `Σ = (k·G − k·between)/(k·N)` — hệ số k triệt tiêu ở cả hai.
> Nên không có cách nào "ăn gian" bằng số vòng lặp.

## Kiểm 3 — Hiệp phương sai cập nhật streaming, không quét lại data cũ ✅

`slda.py` chỉ giữ **ba bộ đếm cộng dồn**, không lưu mẫu nào:

```python
self.feat_sum.index_add_(0, ys, f)   # s_c += f
self.count.index_add_(0, ys, 1)      # n_c += 1
self.gram.add_(f.t() @ f)            # G   += f fᵀ
```

Σ suy ra từ ba đại lượng đó bằng công thức đóng:

```
Σ_w = (G − Σ_c n_c μ_c μ_cᵀ) / N        # within-class covariance, CHÍNH XÁC
Λ   = (Σ_w + εI)⁻¹
```

Không có vòng lặp nào đọc lại `task_loaders[t']` với `t' < t`. **Điều kiện no-replay
được thoả nghiêm ngặt.**

Chi tiết đáng khen trong code: dùng `float64` cho ba bộ đếm (`slda.py:40-41`) đúng vì
cộng dồn hàng chục nghìn mẫu rồi nghịch đảo ma trận — `float32` sẽ lệch theo thứ tự cộng.

## Kiểm 4 — Accuracy matrix nhất quán với metrics ✅

| Seed | mean(hàng cuối acc_matrix) | `average_accuracy` | Lệch |
|---|---:|---:|---:|
| 0 | 0.827411 | 0.827419 | 8e-6 |
| 1 | 0.825567 | 0.825561 | 6e-6 |
| 2 | 0.825967 | 0.825973 | 6e-6 |

Lệch cỡ `1e-5` là do CSV làm tròn 4 chữ số, không phải sai lệch thật.

**Mean 3 seed = `0.8263 ± 0.0010`** — khớp con số đã báo cáo `0.8263 ± 0.0009`.

## Xác nhận thêm — SLDA thật sự không dùng Titans

`config.yaml` của run: `memory.enabled: False`. Và bảng norm trong `compare_all` không
có dòng `run_slda_s*.log` nào — vì không có Titans state để đo.

---

## Hai điểm cần sửa (nhỏ, không ảnh hưởng kết quả)

### 1. `config.yaml` lưu sai `method`

File config lưu kèm artifact ghi `train.method: titans`, trong khi `metrics.json` ghi
`method: slda`. Nguyên nhân: cờ CLI `--method slda` ghi đè lúc chạy nhưng không được
ghi ngược vào config trước khi lưu.

Không sai kết quả, nhưng **gây nhầm khi tái lập**. Nên sửa `run_g1.py` ghi cờ CLI vào
config trước khi dump.

### 2. Con số bộ nhớ mình báo trước đây thiếu — phải đính chính

Trước mình nói "590 KB". Đó là tính `384² × 4 byte` (float32). **Nhưng code dùng `float64`:**

| Bộ đệm | Kích thước | float64 |
|---|---|---:|
| `gram` (D×D) | 384 × 384 | **1 125 KB** |
| `feat_sum` (C×D) | 45 × 384 | 135 KB |
| `count` (C) | 45 | 0.4 KB |
| cache `w`,`b` (float32) | 384×45 + 45 | 68 KB |
| **Tổng** | | **≈ 1,32 MB** |

Ép về float32 khi lưu thì còn ~659 KB, nhưng nên giữ float64 lúc tính (lý do ở `slda.py:40`).

**Lưu ý mở rộng:** chi phí do `gram` chi phối và là **O(D²)**, không phải O(C·D). Với
ViT-S (D=384) là 1,1 MB — chấp nhận được. Với ViT-L (D=1024) sẽ là **8 MB**. Đây là
giới hạn thật của SLDA và nên nêu trong phần chi phí triển khai (B6).

Dù vậy, so với replay ảnh thì vẫn thắng tuyệt đối: bộ nhớ SLDA **cố định**, không tăng
theo thời gian bay.

---

## Việc tiếp theo trong hướng B

`0.8263` đã chốt. Bước kế là **B2 — ablation cơ chế**, thí nghiệm quan trọng nhất của
hướng này: cùng feature frozen, chỉ đổi bộ đọc.

| Arm | Bộ đọc | Kỳ vọng |
|---|---|---|
| 1 | NCM (chỉ trung bình lớp) | 0.7114 |
| 2 | **SLDA (Σ chung, streaming)** | **0.8263** |
| 3 | SLDA với Σ = I | ≈ arm 1 nếu giả thuyết đúng |
| 4 | SLDA với Σ tính 1 lần ở task 0 rồi đóng băng | tách "Σ có cần streaming không" |

Arm 3 và 4 đều làm được bằng cách thêm cờ vào `SLDAClassifier`, không cần train lại
backbone — mỗi arm chỉ tốn một lượt quét dữ liệu.

Kết quả biến câu "SLDA thắng" thành **"hiệp phương sai chung là thứ tạo ra +11,5 điểm,
và nó cần được cập nhật liên tục"** — một phát biểu có cơ chế, đủ sức làm nội dung
chính của một chương.
