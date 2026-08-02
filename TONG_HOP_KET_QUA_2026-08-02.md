# Tổng hợp & tổ chức lại kết quả — 2026-08-02

Nguồn: `compare_all.py result_test` (campaign 3 VM: cosine / SDC / SLDA+latent replay).
Tài liệu này gom nhóm lại 60 dòng thô thành các nhóm có nghĩa và loại bỏ dòng rác.

---

## 0. Kết quả nổi bật trong một câu

**SLDA đạt `0.8263` — cao hơn mọi thứ đã đo từ trước tới nay, kể cả oracle và replay ảnh — và nó chạy với `memory.enabled=false`, tức KHÔNG dùng Titans.**

| So với | Chênh lệch |
|---|---:|
| Frozen ViT + NCM (`0.7114`) | **+11.49 điểm** |
| Anchored Blend gamma 0.25 (`0.7206`) | **+10.57 điểm** |
| Titans + NCM post-hoc, oracle (`0.7459`) | **+8.04 điểm** |
| Replay ảnh (`0.7937`) | **+3.26 điểm** |

Toàn bộ khoảng cách mà nhánh Titans + NCM đang tranh giành là **3.45 điểm**.
Đổi classifier vừa đem lại **11.49 điểm**. Nút thắt thật không nằm ở feature.

---

## 1. Nhóm A — Campaign mới (kết quả đáng tin, cùng split, cùng config)

### A1. SLDA — THẮNG, bền 3/3 seed, không replay

| Seed | Acc | Forget | AAA |
|---:|---:|---:|---:|
| 0 | 0.8274 | 0.0778 | 0.8884 |
| 1 | 0.8256 | 0.0654 | 0.8857 |
| 2 | 0.8260 | 0.0852 | 0.9075 |
| **Mean** | **0.8263 ± 0.0009** | **0.0761** | **0.8939** |

- Vượt mốc thắng (`Acc ≥ 0.72 & Forget ≤ 0.10`) ở **cả 3 seed**, ở **head chính** — không phải NCM shadow.
- `std = 0.0009` — cực kỳ ổn định qua class order.
- `AAA = 0.8939` là con số quan trọng nhất cho chế độ streaming/UAV: **trung bình accuracy tại MỌI mốc thời gian**, không chỉ lúc cuối.
- **Không có dòng `run_slda_s*.log` trong bảng norm** — đúng như mong đợi, vì `memory.enabled=false` nên không có Titans state. Xác nhận SLDA không dùng Titans.

### A2. Cosine head — tốt, nhưng con số đẹp là oracle

| Seed | Linear Acc | Fgt | AAA | NCM Acc | NCM Fgt |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6122 | 0.2444 | 0.7581 | 0.7388 | 0.0709 |
| 1 | 0.6109 | 0.2406 | 0.7165 | 0.7318 | 0.0676 |
| 2 | 0.6850 | 0.1604 | 0.8211 | 0.7418 | 0.0658 |
| **Mean** | **0.6360 ± 0.0424** | 0.2151 | 0.7652 | **0.7375 ± 0.0051** | 0.0681 |

**Cảnh báo đọc số:** theo chú thích của chính `compare_all.py`, NCM shadow của run cosine
chạy ở chế độ **rebuild** — tức **có quét lại toàn bộ data cũ**. Vậy `0.7375` là **oracle**,
không phải kết quả triển khai được. Con số deployable của cosine là cột Linear: `0.6360`.

Điểm tích cực có thật: cosine ổn định hoá training. Seed 1 với Titans+M3 thuần từng nổ
(`norm ≈ 1.75e6`), nhưng `run_cosine_s1` có `norm = 54.5` — bình thường.

### A3. Latent replay — ổn định, nhưng chỉ ngang baseline

| Seed | Linear Acc | Fgt | AAA | NCM Acc | NCM Fgt |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6522 | -0.0341 | 0.7019 | 0.7174 | 0.0728 |
| 1 | 0.6671 | -0.0404 | 0.7204 | 0.7234 | 0.0682 |
| 2 | 0.6595 | -0.0143 | 0.7248 | 0.7159 | 0.0727 |
| **Mean** | **0.6596 ± 0.0075** | **-0.0296** | 0.7157 | **0.7189 ± 0.0040** | 0.0712 |

- Forgetting **âm** (`-0.0296`) = backward transfer dương, học task mới còn giúp task cũ tốt lên. Đây là hành vi lành mạnh.
- Nhưng NCM `0.7189` chỉ ngang gamma 0.25 (`0.7206`), mà lại **có buffer replay**. Không đáng.

### A4. SDC — THẤT BẠI, và thất bại có hệ thống

| Seed | NCM Acc | NCM Fgt | norm(state) | Trạng thái |
|---:|---:|---:|---:|---|
| 0 | 0.6965 | 0.1186 | 54.4 | Dưới baseline |
| 1 | **0.2308** | **0.7149** | **575.4** | **SẬP** |
| 2 | 0.6912 | 0.1264 | 53.8 | Dưới baseline |
| Mean (2 seed sống) | 0.6939 ± 0.0037 | 0.1225 | | |
| Mean (cả 3 seed) | 0.5395 ± 0.2674 | 0.3200 | | |

Đối chiếu control cùng VM — `artifacts_rebuild_s0`: NCM `0.7098`, Fgt `0.0876`.
Cùng seed 0, cùng mọi thứ, chỉ khác cách cập nhật prototype:

> **SDC `0.6965` < rebuild `0.7098`** → SDC ước lượng độ dịch chuyển **kém hơn** việc
> đo lại thật. Chênh 1.3 điểm là cái giá của việc không đọc data cũ.

Và cả hai đều **thua Frozen NCM `0.7114`**. Theo bảng quyết định đã thống nhất:
**SDC rơi vào nhánh "chưa dùng được"**.

---

## 2. Nhóm B — Mốc so sánh (từ báo cáo trước, split `combined31500_v2`)

| Mốc | Acc | Fgt | Có đọc data cũ? |
|---|---:|---:|---|
| Frozen ViT + NCM | 0.7114 | 0.0874 | Không |
| Titans + NCM online cũ | 0.6190 | 0.2671 | Không |
| Anchored Blend gamma 0.25 | 0.7206 | 0.0935 | Không |
| Titans + NCM post-hoc | 0.7459 | 0.0656 | **CÓ** (oracle) |
| Replay ảnh | 0.7937 | — | **CÓ** |

---

## 3. Nhóm C — Loại bỏ khỏi mọi phân tích

Các dòng sau **không được đưa vào so sánh**:

| Nhóm | Acc quan sát | Lý do loại |
|---|---|---|
| `hope` + m3 (5 dòng) | 0.208 – 0.222, Fgt 0.94–0.97 | Hỏng hoàn toàn |
| `cms` + m3 (4 dòng) | 0.217 – 0.388, Fgt 0.57–0.67 | Hỏng |
| `artifacts_fix0718*` titans+m3 | 0.194 – 0.394 | Thí nghiệm cũ, cấu hình đã bị thay thế |
| `artifacts_smoke` (2 dòng) | 0.3125 | Smoke test, không phải kết quả |
| `artifacts_gcp` ewc/finetune/lwf | 0.33 – 0.43 | Baseline yếu, split cũ |
| `artifacts_gcp` ncm `0.7486` | 0.7486 | **Split CŨ** — không so được với 0.7114. Dễ nhầm nhất trong bảng |
| Các run nổ norm > 1e4 | — | `run_v2_seed1`, `run_ncmhead_seed1`, `run_m3_s1` (norm ≈ 1.7e6) |

---

## 4. Vấn đề ổn định — mẫu lặp lại quanh seed 1

| Run | norm | alpha |
|---|---:|---:|
| `run_m3_s1` | 1 757 552 ⚠️ | 0.0000 |
| `run_ncmhead_seed1` | 1 753 927 ⚠️ | — |
| `run_v2_seed1` | 1 644 728 ⚠️ | — |
| `run_sdc_s1` | 575 (cao bất thường) | 1.0000 |
| `run_cosine_s1` | 54.5 (bình thường) | 1.0000 |
| `run_latreplay_s1` | 88.5 (bình thường) | 1.0000 |

**Seed 1 nổ lặp lại qua nhiều config khác nhau** — đây không phải xui rủi ngẫu nhiên mà là
một tương tác bất ổn định có thể tái lập. Cosine head và latent replay dập được nó;
SDC thì không (norm 575 = gấp 10 lần bình thường, kèm accuracy sập còn 0.23).

Ngoài ra `alpha = 1.0000` (forget-gate không quên gì) xuất hiện ở **toàn bộ 3 run SDC** —
đúng chỉ dấu "hướng nổ" ghi trong chú thích của script.

---

## 5. Vấn đề của chính bảng compare — cần sửa script

1. **Đếm trùng.** `res1/res2/res3` chứa folder trùng tên nên nhiều run bị liệt kê 2–4 lần
   (`hope 0.2084` ×2, `titans 0.2103` ×3, `titans 0.2503` ×3...). Bảng 60 dòng thực chất
   chỉ khoảng 40 run.
2. **Cắt tên folder ở 26 ký tự.** 12 dòng đều hiện `artifacts_titans_resisc45_` — không
   phân biệt được config nào. Cần nới cột hoặc in đường dẫn đầy đủ.
3. **Trộn split cũ và mới.** `artifacts_gcp ncm 0.7486` nằm cạnh các run split mới, rất dễ
   bị đọc nhầm thành baseline. Cần thêm cột `split`.
4. **Không đánh dấu oracle.** NCM shadow chế độ `rebuild` có đọc data cũ nhưng hiển thị
   giống hệt chế độ `sdc` (không đọc). Cần cột `replay: yes/no`.

---

## 6. Kết luận

1. **SDC thất bại** — dưới baseline ở 2 seed, sập ở seed còn lại. Không cứu được bằng tinh chỉnh.
2. **Cosine head có giá trị nhưng không phải ở chỗ đang nghĩ** — giá trị thật của nó là
   **ổn định hoá training** (dập được seed 1 nổ), không phải accuracy.
3. **SLDA làm thay đổi câu hỏi nghiên cứu.** Nó thắng bằng cách bỏ hẳn Titans và chỉ đổi
   classifier: từ NCM (chỉ dùng trung bình lớp) sang LDA streaming (dùng thêm ma trận
   hiệp phương sai dùng chung). Điều này gợi ý nút thắt suốt thời gian qua là
   **classifier, không phải feature**.
4. Hệ quả: **phải kiểm chứng SLDA thật kỹ trước khi tin.** Một phương pháp vượt cả oracle
   8 điểm và vượt replay ảnh 3.3 điểm luôn cần bị nghi ngờ trước.

## 7. Việc cần làm — theo thứ tự

### Bước 1 — Kiểm chứng SLDA (bắt buộc, làm trước mọi thứ)

Kiểm 4 điểm, bất kỳ điểm nào sai là kết quả vô hiệu:

- [ ] Test set cuối có đúng đủ **45 lớp** và giống hệt test set của các run khác không?
- [ ] Stream có đúng **9 task × 5 lớp**, và SLDA chỉ thấy mỗi ảnh **một lần** không?
- [ ] Ma trận hiệp phương sai được cập nhật **streaming**, không quét lại data cũ?
- [ ] Accuracy matrix có nhất quán với `Acc = 0.8263` và `Fgt = 0.0761` không?

### Bước 2 — Thí nghiệm quyết định (một lệnh, trả lời dứt điểm câu hỏi Titans)

Chạy **SLDA với `memory.enabled=true`** (bật Titans), 3 seed:

- Nếu **> 0.8263** → feature Titans có giá trị thật, đề tài đứng vững, hướng mới là SLDA + Titans.
- Nếu **≈ 0.8263** → Titans không đóng góp gì. Cần định vị lại đề tài một cách trung thực.
- Nếu **< 0.8263** → Titans đang làm hỏng feature. Kết quả tiêu cực nhưng vẫn là phát hiện có giá trị.

Đây là thí nghiệm rẻ nhất và quan trọng nhất còn lại.

### Bước 3 — Dọn dẹp

- [ ] Sửa `compare_all.py`: dedup, nới cột tên, thêm cột `split` và `replay yes/no`.
- [ ] Commit `metrics*.json` + `summary.csv` của campaign này vào repo (nhẹ, vài MB).
- [ ] Khép lại SDC bằng một ghi chú kết quả âm — có giá trị, đừng xoá.
- [ ] Điều tra seed 1: tại sao nổ lặp lại, và tại sao cosine dập được.

### Bước 4 — Nếu SLDA đứng vững

- [ ] Xác nhận liên-dataset trên EuroSAT, khoá tham số trước.
- [ ] Đo latency / memory / energy. SLDA giữ ma trận `d×d` (ViT-S: 384×384 ≈ 590 KB) — nhỏ, nhưng cần số thật.
- [ ] Cân nhắc thay Frozen NCM bằng SLDA trong cấu hình triển khai.
