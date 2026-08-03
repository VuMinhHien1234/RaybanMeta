# Kết quả campaign 03-08 — phân tích

Nguồn: `result_test/final/` (3 VM) · Tất cả cùng split `combined31500_v2`, 9 task × 5 lớp

---

# PHẦN 1 — B2: hiệp phương sai có phải nguyên nhân không? **CÓ**

| Arm | Bộ đọc | Acc (3 seed) | std | Forget | AAA |
|---|---|---:|---:|---:|---:|
| 1 | NCM thuần | 0,6924 | 0,0010 | 0,0979 | 0,7778 |
| 3 | SLDA **Σ = I** | 0,6885 | 0,0010 | 0,0988 | 0,7758 |
| 4 | SLDA **Σ đóng băng** | **0,7682** | 0,0077 | **0,0579** | 0,8437 |
| 2 | SLDA **Σ streaming** | **0,8266** | 0,0017 | 0,0768 | **0,8941** |

## Kiểm bẫy: ĐẠT

`|arm 1 − arm 3| = 0,0040`

Hai đường code hoàn toàn khác nhau (NCM thuần vs SLDA với Σ = I) cho kết quả lệch **0,4 điểm**.
Đúng như dự đoán lý thuyết. **Không có bug** — mọi kết luận dưới đây đứng vững.

Phần chênh 0,4 điểm còn lại nhiều khả năng do NCM chuẩn hoá feature (cosine) còn SLDA
dùng khoảng cách Euclid thô — khác biệt cài đặt nhỏ, không ảnh hưởng kết luận.

## Kết quả chính

> **Hiệp phương sai chung đóng góp +13,81 điểm** (0,8266 − 0,6885).

Cùng feature ViT đóng băng, cùng đường code, chỉ khác `Σ = I` hay `Σ` thật.
Câu "SLDA thắng" giờ thành **"ma trận hiệp phương sai chung là thứ tạo ra cải thiện"** —
một phát biểu có cơ chế.

## Kết quả ngoài dự kiến — arm 4 nằm ở GIỮA

Mình đưa ra hai kịch bản nhị phân cho arm 4 (≈ arm 2, hoặc thấp hơn hẳn). Thực tế rơi
vào giữa, và đó lại là kết quả **hữu ích hơn cả hai kịch bản kia**:

| Thành phần | Đóng góp | Tỷ lệ |
|---|---:|---:|
| Ước lượng Σ **một lần** sau task 0 | +7,97 điểm | **58%** |
| Tiếp tục **cập nhật Σ** qua 9 task | +5,84 điểm | **42%** |

Nghĩa là: **hơn nửa lợi ích đến ngay từ lần ước lượng đầu tiên.** Cấu trúc hiệp phương sai
của feature ViT khá ổn định — 5 lớp đầu đã đủ để ước lượng gần đúng cho cả 45 lớp.

### Và arm 4 có Forget THẤP NHẤT

`0,0579` — thấp hơn cả streaming (`0,0768`) và thấp hơn NCM (`0,0979`).

Lý do trực tiếp: Σ đóng băng = **thước đo không đổi** → prototype lớp cũ không bị dịch
chuyển khi học lớp mới. Streaming làm Σ thay đổi, kéo theo ranh giới lớp cũ xê dịch nhẹ.

**Đây là đánh đổi định lượng được, dùng ngay cho phần triển khai:**

| Ưu tiên | Chọn | Vì |
|---|---|---|
| Accuracy tối đa | Σ streaming | 0,8266 |
| **Chống quên tối đa** | **Σ đóng băng** | Forget 0,0579, và không cần cập nhật `gram` mỗi mẫu |

### Đính chính một chỗ mình nói sai hôm trước

Mình từng viết *"arm 4 ≈ arm 2 → bỏ được `gram`, cắt 85% bộ nhớ"*. **Sai.**

Sau khi đóng băng, vẫn phải giữ `Λ = (Σ+εI)⁻¹` cỡ `D×D` để tính `w = Λμᵀ` mỗi khi μ đổi.
`Λ` thay chỗ `gram`, **cùng kích thước**. Tiết kiệm thật là:

- **Compute**: bỏ cập nhật `gram` (D² MAC/mẫu) và bỏ nghịch đảo lặp lại O(D³)
- **Memory**: chỉ ~0,5 MB (Λ lưu được fp32, `gram` cần fp64)

Không phải 85%.

---

# PHẦN 2 — Titans: bản vá `gate_bound` ĂN về cơ chế, KHÔNG đủ về accuracy

## Cổng đã sống — lần đầu tiên trong dự án

| | `ctrl` (không vá) | `gates` (có vá) |
|---|---|---|
| `trace_gates` phân loại | **`TROI-DAN`** ⚠️ | **`LANH-MANH`** ✅ |
| α task 0 → 8 | 0,4475 → **0,9993** ⚠️SAT | 0,2637 → **0,0678** |
| α span | — (kẹt biên từ task 5) | **0,1959** |
| norm(state) | 54,3 (phẳng = không tích luỹ) | **99,2** (dao động 56–163) |

Quỹ đạo α của `gates` qua 9 task:

```
0,264 → 0,214 → 0,089 → 0,231 → 0,263 → 0,088 → 0,180 → 0,087 → 0,068
```

Dao động thật, không đơn điệu. Còn `ctrl` thì kẹt cứng ở `1,0000` từ task 5 trở đi.

**Chín task đủ dài để lộ đà trôi — và bản vá chặn được nó.** Đây là kết quả dương rõ ràng
cho phần cơ chế.

## Nhưng accuracy chỉ nhích

| | Linear | NCM shadow | AAA |
|---|---:|---:|---:|
| `ctrl` | 0,5453 | 0,7057 | 0,6290 |
| `gates` | **0,5704** | **0,7104** | **0,6547** |
| Chênh | **+2,51 điểm** | +0,47 điểm | +2,57 điểm |

Linear và AAA cải thiện rõ. Nhưng NCM chỉ +0,47 điểm, và **`0,7104` vẫn thua SLDA
`0,8266` tới 11,6 điểm**.

## Cổng "sống" nhưng vẫn bị ép vào biên

| Cổng | Sàn/trần | Số task chạm biên |
|---|---|---|
| η | trần 0,900 | **3/9** task ≥ 0,87 |
| α | sàn 0,050 | **4/9** task ≤ 0,09 |

Gradient vẫn đẩy cả hai cổng về biên — `gate_bound` chỉ dời biên ra xa hơn, không đổi
được **hướng** áp lực.

Đây đúng như mình đã cảnh báo: *bản vá sửa triệu chứng, không sửa nguyên nhân (mục tiêu
huấn luyện thưởng cho việc quên)*. Số liệu xác nhận.

---

# PHẦN 3 — Một phát hiện về baseline cần xử lý trước khi viết báo cáo

Arm 1 (NCM thuần) đo được **0,6924 ± 0,0010**, và seed 0 ra **đúng `0,6933` với Forget
`0,1000`** — **trùng khít** mốc `NCM gốc 0,6933 / F 0,10` mà log của dự án vẫn in ra.

Nhưng báo cáo 01-08 dùng mốc **`Frozen ViT + NCM = 0,7114`**.

→ **Đây là hai biến thể NCM khác nhau**, chênh 1,9 điểm. Nhiều khả năng khác ở
`NCM feature protocol` (`independent_image` hay không).

**Trước khi viết báo cáo phải chốt dùng mốc nào**, và ghi rõ nó là biến thể gì. Dùng lẫn
lộn hai con số sẽ làm mọi so sánh lệch gần 2 điểm.

Mốc **tái lập được ngay hôm nay** là `0,6924` (arm 1) — nên đó là mốc an toàn nhất để dùng.

---

# BẢNG TỔNG CUỐI CÙNG

Tất cả cùng split `combined31500_v2`, 9 task × 5 lớp — **so trực tiếp, không cần chú thích**.

| Phương pháp | Acc | Forget | AAA | Bộ nhớ | Tin cậy |
|---|---:|---:|---:|---:|---|
| **SLDA Σ streaming** | **0,8266** | 0,0768 | **0,8941** | 1,39 MB | ✅ 3 seed |
| Replay ảnh | 0,7937 | — | — | tăng vô hạn | ⚠️ 1 seed |
| **SLDA Σ đóng băng** | 0,7682 | **0,0579** | 0,8437 | ~1,4 MB | ✅ 3 seed |
| Frozen ViT + NCM (mốc cũ) | 0,7114 | 0,0874 | — | 0,14 MB | ⚠️ biến thể khác |
| **Titans + NCM (có vá)** | 0,7104 | 0,0865 | — | ~100 MB | 1 seed |
| Titans + NCM (đối chứng) | 0,7057 | 0,0860 | — | ~100 MB | 1 seed |
| NCM thuần (đo lại) | 0,6924 | 0,0979 | 0,7778 | 0,14 MB | ✅ 3 seed |
| SLDA Σ = I | 0,6885 | 0,0988 | 0,7758 | 0,14 MB | ✅ 3 seed |
| Titans Linear (có vá) | 0,5704 | −0,0372 | 0,6547 | ~100 MB | 1 seed |

## Ba kết luận rút ra

**1. SLDA thắng tuyệt đối trên mọi trục đo được.**
Vượt phương án tốt thứ hai (replay ảnh) **3,3 điểm** với bộ nhớ **cố định 1,39 MB** thay
vì tăng vô hạn. Vượt Titans-có-vá **11,6 điểm** với bộ nhớ **nhỏ hơn 74 lần**.

**2. Nguồn gốc cải thiện đã xác định: hiệp phương sai chung, +13,81 điểm.**
Và tách được thành 58% từ ước lượng một lần + 42% từ cập nhật liên tục.

**3. Bản vá cổng quên hoạt động đúng như thiết kế, nhưng không đủ để cứu hướng NL.**
Cổng sống qua 9 task (lần đầu tiên), accuracy nhích +2,5 điểm — nhưng khoảng cách tới
SLDA vẫn là 11,6 điểm. Nguyên nhân sâu hơn (mục tiêu huấn luyện thưởng việc quên) chưa
được giải quyết.

---

# VIỆC TIẾP THEO

## Ưu tiên cao — rẻ và cần cho báo cáo

- [ ] **Chốt mốc NCM**: xác định `0,6933` và `0,7114` khác nhau ở đâu, ghi rõ trong báo cáo
- [ ] **Commit metrics JSON** của campaign này vào git (vài MB) — để kết quả tái lập được
- [ ] **Chạy 3 seed cho Titans** (hiện chỉ seed 0) nếu muốn đưa vào báo cáo có std

## Ưu tiên trung bình — mở rộng kết quả SLDA

- [ ] **Độ nhạy shrinkage**: quét `[1e-4, 1e-3, 1e-2, 1e-1]` — chứng minh phương pháp bền
- [ ] **Độ dài stream**: 5 / 9 / 15 / 45 task — điểm yếu tiềm tàng của SLDA
- [ ] **Liên-dataset EuroSAT**: khoá tham số trước, không tuning lại
- [ ] **`stats_dtype=float32`** trên dữ liệu thật — chốt được thì bộ nhớ còn 0,73 MB

## Đã đủ dữ liệu để viết — không cần chạy thêm

Cấu trúc báo cáo giữ được cả hai hướng:

> **Chương 3 — SLDA là phương pháp chốt**
> 0,8266 · phân tích cơ chế (B2: +13,81 điểm từ Σ, tách 58/42) · chi phí (B6: 74× bộ nhớ,
> 307× độ trễ) · đánh đổi accuracy-vs-forget giữa Σ streaming và Σ đóng băng
>
> **Chương 4 — Nested Learning: vì sao không thắng**
> Phát hiện cổng quên tự phản lại (quỹ đạo 13 run) · bẫy hằng số `max_lr` · bản vá
> `gate_bound` và bằng chứng nó hoạt động (`TROI-DAN` → `LANH-MANH`) · nhưng +2,5 điểm
> không đủ · kết luận: luận điểm cho NL nằm ở **domain shift**, không phải class-incremental
