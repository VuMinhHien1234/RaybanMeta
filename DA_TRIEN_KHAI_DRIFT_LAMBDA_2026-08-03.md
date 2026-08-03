# Đã triển khai xong D1–D9 — λ + dataset trôi

Ngày 2026-08-03 · theo `KE_HOACH_DRIFT_LAMBDA_2026-08-03.md`

---

## 1. Tóm tắt: cái này để làm gì

Mọi kết quả của dự án tới nay đo ở một regime duy nhất: **lớp mới xuất hiện dần, điều kiện
quan sát TĨNH**. Ở đó SLDA thắng đậm (0,8266 so với NCM 0,6924, +13,8 điểm).

Nhưng câu hỏi ứng dụng thật của UAV là câu khác:

> Drone bay hàng tháng. Nắng, mùa, sương, bụi ống kính đổi dần. Model có bám theo được không?

Regime đó **chưa ai trong dự án đo**. Và đó chính là chỗ SLDA hiện tại được **dự đoán sẽ hỏng**:
nó cộng dồn vĩnh viễn, nên sau 1 triệu mẫu, một mẫu mới chỉ dịch μ_c đi 1/1.000.001. Nó
"không quên" — nhưng cũng có nghĩa là **không thích nghi**.

λ (hệ số quên) là thứ vá chỗ đó. Và đây là điểm đáng chú ý về mặt học thuật:

> **λ chính là cổng quên α của Titans/Nested Learning — nhưng CỐ ĐỊNH thay vì học được.**
> Và chính vì cố định, nó **không thể tự trôi ra biên** như α đã làm (xem `CHAN_DOAN_NL`:
> 9/13 run có α trôi tới biên trong 2–4 task).

Nên thí nghiệm này không phải "bỏ Nested Learning" — nó là **thí nghiệm đối chứng trực tiếp
cho luận điểm trung tâm của NL**: cổng quên *học được* có thật sự hơn cổng quên *đặt tay* không?

---

## 2. Đã code xong những gì

| Mã | Việc | File | Trạng thái |
|---|---|---|---|
| D1 | `decay_mean` / `decay_cov` + logic phân rã | `src/uavcl/models/slda.py` | ✅ |
| D2 | `window_report()` + log ở `end_task` | `slda.py`, `methods.py` | ✅ |
| D3 | 16 test cho λ | `tests/test_slda_decay.py` | ✅ (chưa chạy) |
| D4 | Dataset trôi — 5 phép biến đổi vật lý | `src/uavcl/data/drift.py` (mới) | ✅ |
| D5 | Domain-incremental stream | `src/uavcl/data/stream.py` | ✅ |
| D6 | Transform riêng theo task | `src/uavcl/data/loaders.py` | ✅ |
| D7 | 6 config arm | `configs/drift_*.yaml` | ✅ |
| D8 | Script hiệu chỉnh cường độ trôi | `scripts/calibrate_drift.py` (mới) | ✅ |
| D9 | `--dump-samples` — lưới ảnh để nhìn bằng mắt | cùng file trên | ✅ |

`py_compile` xanh trên cả 10 file.

---

## 3. Hai quyết định thiết kế quan trọng nhất

### 3.1 μ_c phân rã **theo-lớp**, Σ phân rã **theo đồng hồ toàn cục**

Đây là bẫy dễ mắc nhất của cả kế hoạch, nên nói kỹ.

Cách làm sai (và trực giác nhất): mỗi batch thì nhân **toàn bộ** `feat_sum` với λ.

Hậu quả: lớp *Sân bay* xuất hiện ở task 0 rồi biến mất. Sau 200 batch chỉ có lớp khác,
thống kê của *Sân bay* bị nhân `0,99^3200 ≈ 10⁻¹⁴` — **bị xoá sạch**. Ta vừa phá đúng cái
tính chất làm SLDA mạnh: không quên lớp cũ.

Cách đúng, đã cài:

```python
# Σ (gram) — thống kê DÙNG CHUNG cho mọi lớp, tuổi tính theo TỔNG mẫu đã đi qua
if self.decay_cov < 1.0:
    self.gram.mul_(self.decay_cov ** n_batch)

# μ_c — chỉ phân rã ở lớp CÓ MẶT trong batch này
if self.decay_mean < 1.0:
    dem = torch.zeros_like(self.count).index_add_(0, ys, torch.ones_like(...))
    he_so = torch.where(dem > 0, decay_mean ** dem, 1.0)   # lớp vắng -> nhân 1.0
    self.feat_sum.mul_(he_so.unsqueeze(1))
    self.count.mul_(he_so)
```

μ_c chỉ bám theo những lần lớp c **thực sự xuất hiện**. Lớp hiếm gặp giữ nguyên trí nhớ.

Test `test_lop_vang_mat_khong_bi_phan_ra` chặn đúng chỗ này, dùng `torch.equal` (bằng tuyệt
đối, không phải xấp xỉ).

### 3.2 Hai λ riêng, không phải một

| | λ_μ (`decay_mean`) | λ_Σ (`decay_cov`) |
|---|---|---|
| ước lượng gì | D số / lớp (384) | D² số (147.456) |
| cần bao nhiêu mẫu | ít | rất nhiều |
| nên đặt | nhỏ hơn — bám nhanh | lớn hơn — giữ ổn định |

Dùng chung một λ thì hoặc μ bám chậm, hoặc Σ vỡ ước lượng. Arm 5 (`λ_μ = λ_Σ = 0,99`) tồn tại
chính là để **chứng minh bằng số** rằng dùng chung là dở — nó là đối chứng âm.

---

## 4. Dataset trôi — trôi TẤT ĐỊNH, không phải augmentation

Năm phép, mỗi phép mô phỏng một hiện tượng vật lý có thật:

| phép | mô phỏng | 0% → 100% |
|---|---|---|
| độ sáng | giờ trong ngày, mùa | ×1,00 → ×0,55 |
| nhiệt độ màu | bình minh ấm → trưa lạnh | 0 → ±18% lệch R/B |
| tương phản | sương mù, khói, bụi | ×1,00 → ×0,65 |
| mờ Gauss | độ cao bay, ống kính bẩn | σ 0 → 1,4 px |
| nhiễu | ISO cao khi thiếu sáng | σ 0 → 0,035 |

**Nguyên tắc sống còn của file này:**

- **Mức trung bình dịch theo task một cách xác định** ← đây mới là "trôi"
- Cộng **rung nhẹ** ±15% quanh mức đó ← mỗi chuyến bay hơi khác nhau

Nếu để hoàn toàn ngẫu nhiên thì nó chỉ là augmentation thường, phân bố **không dịch đi**, và
λ hoàn toàn vô dụng — thí nghiệm mất sạch ý nghĩa. Đây là bẫy thiết kế chính, đã tránh.

Thứ tự áp: `hình học → ToTensor → TRÔI → Normalize`. Trôi phải nằm trên thang [0,1] mới đúng
nghĩa vật lý (nhân độ sáng trên thang đã chuẩn hoá là vô nghĩa).

**Stream cũng đổi kiểu**: từ class-incremental (5 lớp mới mỗi task) sang **domain-incremental**
— mọi task có **đủ 45 lớp**, chỉ khác điều kiện quan sát. Vì nếu vừa thêm lớp mới vừa trôi thì
không tách được nguyên nhân.

Mặc định `drift.enabled: false` → mọi config cũ chạy **y hệt trước**, không đụng gì.

---

## 5. ⛔ CỬA CHẶN — chạy 2 lệnh này trước khi đốt giờ VM

Bài học từ lần trước: smoke test 3-task cho kết quả **ngược dấu** so với regime thật 9-task,
và tôi suýt kết luận sai. Kiểm setup 20 phút rẻ hơn nhiều so với mất 8 tiếng máy.

### Bước 1 — pytest (2 phút)

```bash
cd ~/Desktop/Raybanmeta/uav-continual-learning
source .venv/bin/activate
python -m pytest tests/test_slda_decay.py tests/test_slda_cov_mode.py tests/test_slda.py -q
```

Phải **xanh hết**. Nếu `test_lop_vang_mat_khong_bi_phan_ra` đỏ → dừng lại, đừng chạy gì thêm.

### Bước 2 — hiệu chỉnh cường độ trôi (15 phút)

```bash
python scripts/calibrate_drift.py \
    --config configs/drift_slda_arm3_dexuat.yaml \
    --dump-samples troi_mau.png
```

Script chạy NCM trên feature đóng băng ở từng mức trôi rồi in bảng. **Đọc dòng kết luận cuối:**

| kết luận | nghĩa là | làm gì |
|---|---|---|
| ❌ TRÔI QUÁ NHẸ (> 0,60) | mọi arm λ sẽ như nhau, vô nghĩa | `--severity 1.5` rồi chạy lại |
| ✅ ĐẠT (0,45–0,60) | có chỗ cho λ thể hiện, feature chưa vỡ | chạy campaign |
| ⚠️ HƠI NẶNG (0,25–0,45) | vẫn được, cân nhắc giảm | tuỳ |
| ❌ QUÁ NẶNG (< 0,25) | ViT vỡ hẳn, ai cũng chết | `--severity 0.6` rồi chạy lại |

**Rồi mở `troi_mau.png` nhìn bằng mắt.** 9 ảnh cùng một cảnh, trái sang phải 0% → 100%.
Ảnh cuối phải trông như *"cùng cảnh đó, chiều muộn nhiều sương"* — **không phải** như ảnh hỏng
hay nhiễu trắng. Trông hỏng = severity quá cao, dù con số có nằm trong ngưỡng.

Số nào đạt thì **ghi severity đó vào cả 6 config** trước khi chạy.

---

## 6. Sau khi qua cửa chặn

5 arm, biến duy nhất là λ:

| arm | λ_μ | λ_Σ | cửa sổ μ | giả thuyết |
|---|---|---|---|---|
| 1 | 1,0 | 1,0 | ∞ | SLDA hiện tại — **dự đoán hỏng ở đây** |
| 2 | 0,9999 | 0,9999 | 10.000 | quên rất chậm |
| 3 | 0,999 | 0,9999 | 1.000 | ⭐ đề xuất |
| 4 | 0,99 | 0,999 | 100 | quên nhanh |
| 5 | 0,99 | 0,99 | 100 | đối chứng âm — Σ quên quá nhanh nên vỡ |

Kết quả mong đợi: **hình chữ U ngược** — arm 1 kém (không bám trôi), arm 5 kém (Σ vỡ), đỉnh ở
arm 3 hoặc 4. Nếu đường thẳng tuột thì trôi quá nhẹ, quay lại bước hiệu chỉnh.

Chỉ số quan trọng nhất ở đây là **AAA** (Average Anytime Accuracy), không phải accuracy cuối —
vì drone cần đúng *mọi lúc*, không phải chỉ đúng ở cuối chuyến bay.

Rồi D11: chạy Titans (cổng đã vá) trên **đúng stream đó**. Đó mới là so sánh thật giữa
*cổng quên học được* và *cổng quên đặt tay*.

---

## 7. Còn nợ, chưa giải quyết

1. **`0,6933` so với `0,7114`** — hai biến thể NCM chênh 1,9 điểm, chưa rõ khác nhau chỗ nào.
   Phải chốt trước khi viết báo cáo, không thì bảng baseline không đáng tin.
2. **Metrics JSON chưa commit vào git** — kết quả campaign đang chỉ nằm ở file rời.
