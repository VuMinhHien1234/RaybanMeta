# Trạng thái dự án & hai hướng đi

Ngày: 2026-08-02 · Nhánh code: `fix/nl-gates-alive`

---

# PHẦN 1 — ĐÃ ĐẠT ĐƯỢC GÌ

## 1.1. Kết quả số, kèm mức độ tin cậy

Cột **Tin cậy** là điểm quan trọng nhất của bảng này. Ngày 02-08 phát hiện hai cổng
η/α của Titans bị bão hoà trong hầu hết run — nên **mọi số có dùng Titans memory đều
bị nhiễm**, còn các số không dùng Titans thì không ảnh hưởng.

| Phương pháp | Acc | Forget | Seeds | Dùng Titans? | **Tin cậy** |
|---|---:|---:|---:|---|---|
| **SLDA** | **0.8263 ± 0.0009** | 0.076 | 3 | Không | ✅ **Sạch** (chưa xác minh) |
| Replay ảnh | 0.7937 | — | 1 | Không | ✅ Sạch |
| **Frozen ViT + NCM** | **0.7114** | 0.087 | 3 | Không | ✅ **Sạch** |
| Titans + NCM post-hoc (oracle) | 0.7459 | 0.066 | 3 | Có | ⚠️ Nhiễm |
| Anchored Blend γ=0.25 | 0.7206 ± 0.0015 | 0.094 | 3+3 | Có (25%) | ⚠️ Nhiễm một phần |
| Cosine + rebuild (oracle) | 0.7375 ± 0.0051 | 0.068 | 3 | Có | ⚠️ Nhiễm |
| Latent replay (NCM) | 0.7189 ± 0.0040 | 0.071 | 3 | Có | ⚠️ Nhiễm |
| SDC | 0.6939 (2/3 seed sống) | 0.123 | 3 | Có | ⚠️ Nhiễm |
| Titans + NCM online | 0.6190 | 0.267 | 3 | Có | ⚠️ Nhiễm |

**Đọc bảng:** ba dòng sạch duy nhất là SLDA, replay ảnh, và Frozen NCM. Và chúng nằm
đúng ở top bảng.

**AAA (accuracy trung bình tại mọi mốc — chỉ số quan trọng nhất cho UAV):**
SLDA `0.8939` · latent replay `0.7157` · cosine `0.7652`

## 1.2. Phát hiện khoa học

### A. Cơ chế "học cách quên" của Titans/NL tự phản lại chính nó

Đo quỹ đạo η/α theo từng task trên 13 run:

- α (cổng quên) **khởi đầu lành mạnh** (0.10–0.89) rồi **trôi đơn điệu ra biên trong 2–4 task**
- 9/13 run phân loại `TROI-DAN`, 0 run bão hoà từ đầu
- Nguyên nhân: **gradient trên loss task hiện tại luôn thưởng cho việc quên** — quên lớp cũ
  làm giảm nhiễu khi học lớp mới, nên gradient đẩy α → biên và không có gì đẩy ngược

Bằng chứng định lượng: tương quan α ↔ norm(state) **đơn điệu hoàn hảo**, và **cả hai
chiều hỏng** đều quan sát được:

| α | Giữ lại | norm | Hệ quả |
|---:|---:|---:|---|
| 0.0000 | 100% | **1 757 552** | Nổ |
| 0.1246 | 88% | 122.5 | Phình |
| 0.4814 | 52% | 59.8 | Lành mạnh |
| 1.0000 | 0% | ~54 (phẳng) | Memory không tích luỹ |

Đây là **phát biểu tổng quát, không phụ thuộc dataset**, và chưa thấy công bố nào nêu.

### B. Bẫy hằng số trong `titans-pytorch`

`default_adaptive_step_transform(.., max_lr=1e-2)` trông như `max_lr = 1e-2`, nhưng dòng
`:457` luôn ghi đè bằng `default_step_transform_max_lr = 1.` (`:272`). **max_lr thật = 1.0.**
Nghĩa là memory ghi với bước cỡ đầy đủ, không phải 0.01. Bất kỳ ai đọc code này đều dễ sập.

### C. 2/3 trụ cột Nested Learning chưa từng thực sự chạy

| Đóng góp NL | Thực tế |
|---|---|
| 1. Expressive optimizer (M3) | ✅ Chạy |
| 2. Self-modifying, Eq 76 η/α phụ thuộc dữ liệu | ❌ Hai cổng bão hoà → cơ chế chết |
| 3. Continuum Memory System (đa tần số) | ❌ Không có khối `cms:` trong bất kỳ config g2_* nào |

### D. "Head Linear là nút thắt" — kết luận cũ yếu đi khi cổng lành

| | Linear | NCM | Khoảng cách |
|---|---:|---:|---:|
| Cổng hỏng | 0.7007 | 0.7300 | **2.93 điểm** |
| Cổng lành | 0.7300 | 0.7383 | **0.83 điểm** |

Một phần cái gọi là "nút thắt head Linear" thực ra là hệ quả của cổng hỏng.

### E. η bão hoà ở trần không phải "học nhanh" mà là ghi loạn

Khi ghì η từ 1.000 xuống 0.101, accuracy trên task **mới nhất** tăng từ 0.483 lên **0.554**
(+7.1 điểm) — tốt hơn ở cả trục học mới lẫn trục giữ cũ.

## 1.3. Hạ tầng đã xây

| Công cụ | Chức năng |
|---|---|
| `scripts/trace_gates.py` | Trích quỹ đạo η/α theo từng task, tự phân loại `TROI-DAN`/`DUNG-YEN`/`LANH-MANH` |
| `scripts/compare_all.py` (nâng cấp) | Đọc cả 2 định dạng log, quy đổi η thật, cờ ⚠️SAT tự động |
| `memory.gate_bound` | Chặn trôi cổng bằng tanh-bound, gradient không chết |
| `memory.pre_norm` | Chuẩn hoá đầu vào memory |
| `memory.init_*_bias` | Khởi tạo cổng trung tính |
| Cảnh báo ⚠️ trong `methods.py` | Tự báo khi cổng bão hoà, ngay trong log train |
| `tests/test_gate_bound.py` | 16 test khoá lại các bẫy đã gặp |
| 2 config mới | `_gates.yaml` (không CMS) · `_cms.yaml` (có CMS) |

Mọi cờ mới **mặc định tắt** → mọi run cũ vẫn tái lập được.

## 1.4. Chưa đạt được / đã bị vô hiệu

- ❌ **Chưa có phương pháp dùng Titans nào vượt Frozen NCM** trong điều kiện online/no-replay
- ❌ Kết luận "SDC thất bại" — đo khi cổng hỏng, phải xem lại
- ❌ Kết luận "gamma 0.25 thắng" — nhiễm một phần
- ❌ SLDA `0.8263` chưa qua xác minh
- ❌ 2,7 GB artifact báo cáo 01-08 không có trong git, không ai tái lập được
- ❌ Công việc rải trên 8 nhánh, `gamma 0.25` chưa port sang nhánh chính

---

# PHẦN 2 — HƯỚNG A: THEO ĐUỔI NESTED LEARNING

## Câu hỏi trung tâm

> Khi các cơ chế NL **thực sự chạy**, chúng có đóng góp gì cho continual learning
> không replay trên ảnh viễn thám không?

## Lộ trình

### A1. Xác nhận bản vá ở regime thật ⏱ 8h máy · CHẶN

Hai run 9 task × 5 lớp, khác đúng một khối config.

```bash
# arm 1 — có vá
--config configs/g2_titans_resisc45_selfmod_m3_gates.yaml
# arm 2 — đối chứng
--config configs/g2_titans_resisc45_selfmod_m3_improved.yaml
# chung: --set seed=0 train.lr=0.005 train.eval_ncm_head=true train.eval_future=false train.checkpoint=true
```

**Tiêu chí đạt:** `trace_gates.py` xếp arm 1 là `LANH-MANH` (không `DUNG-YEN`), α không kẹt
ở biên qua đủ 9 task, và Linear/NCM ≥ đối chứng.

**Không đạt → dừng hướng A**, chuyển sang B.

### A2. Ablation tách biến ⏱ 12h máy

Config `_gates.yaml` hiện bật 3 thứ cùng lúc. Phải tách:

| Arm | gate_bound | pre_norm | init_bias |
|---|---|---|---|
| base | ✗ | ✗ | ✗ |
| G | ✓ | ✗ | ✗ |
| G+P | ✓ | ✓ | ✗ |
| G+P+I | ✓ | ✓ | ✓ |

**Tiêu chí:** biết được thành phần nào tạo ra cải thiện. Không có bước này thì không
viết được vào báo cáo.

### A3. Bền qua seed ⏱ 24h máy (3 VM song song ≈ 8h)

3 seed cho arm thắng ở A2 + đối chứng. **Tiêu chí:** 3/3 seed vượt đối chứng,
`std < 0.02`, không seed nào nổ norm.

### A4. Chạy lại các thí nghiệm đã bị nhiễm ⏱ 40h máy

SDC, cosine head, latent replay — cả ba kết luận cũ đều đo khi cổng hỏng.
**Đây là khoản chi phí lớn nhất và không tránh được** nếu muốn giữ chúng trong báo cáo.

Nếu không đủ thời gian: bỏ hẳn ba nhánh này khỏi báo cáo và ghi rõ lý do.

### A5. Bật CMS (trụ cột NL thứ 3) ⏱ 8h máy

Config `_cms.yaml` đã sẵn. **Xác minh bắt buộc trong log:** `‖Δw‖` per-tier phải cho
thấy tier chậm ≈ 0, tier nhanh lớn. Ngược lại → mapping tier sai, dừng.

### A6. Thí nghiệm quyết định: SLDA + Titans ⏱ 8h máy

`--method slda --set memory.enabled=true`, 3 seed.

| Kết quả | Kết luận |
|---|---|
| > 0.8263 | Feature Titans có giá trị thật → **hướng A thắng** |
| ≈ 0.8263 | Titans không đóng góp → hướng A kết thúc bằng kết quả âm |
| < 0.8263 | Titans làm hỏng feature → kết quả âm mạnh hơn |

### A7. Liên-dataset EuroSAT ⏱ 8h máy

Khoá mọi tham số trước, không tuning lại.

## Tổng chi phí hướng A

**~100 giờ máy** (≈ 35h nếu chạy 3 VM song song) + khoảng 2 tuần công.

## Rủi ro

1. **Cao nhất:** A1 không đạt → mất công vá mà không đổi được kết luận
2. `gate_bound` sửa **triệu chứng**, không sửa **nguyên nhân** (mục tiêu huấn luyện thưởng
   cho việc quên). Gradient vẫn ép α về sát biên, chỉ là biên mới ở 0.05 thay vì 0
3. Backbone đóng băng khiến phần lớn cơ chế NL không có chỗ tác động
4. Ngay cả khi thắng, khoản lợi có thể vẫn dưới SLDA `0.8263`

## Kịch bản kết thúc

| | Nội dung báo cáo |
|---|---|
| **Tốt** | "Chỉ ra và sửa được lỗi làm vô hiệu cơ chế NL; sau khi sửa, Titans đạt X, vượt baseline" |
| **Trung bình** | "Sau khi sửa, Titans cải thiện nhưng vẫn dưới SLDA — chi phí không tương xứng" |
| **Xấu nhưng vẫn dùng được** | "Cơ chế học-cách-quên tự phản lại trong CL; chỉ ra bằng quỹ đạo 13 run + bản vá; kết quả âm có kiểm soát" |

---

# PHẦN 3 — HƯỚNG B: XOAY QUANH SLDA

## Câu hỏi trung tâm

> Một bộ nhớ bậc 2 dạng công thức đóng có đủ cho continual learning trên thiết bị biên
> không, và nó thắng nhờ cái gì?

## Lộ trình

### B1. Xác minh tính đúng đắn ⏱ 2h người · CHẶN

Bốn điểm, sai một điểm là kết quả vô hiệu:

- [ ] Test set cuối đúng đủ **45 lớp**, giống hệt các phương pháp khác
- [ ] Stream đúng 9 task × 5 lớp, SLDA chỉ thấy mỗi ảnh **một lần**
- [ ] Ma trận hiệp phương sai cập nhật **streaming**, không quét lại data cũ
- [ ] Accuracy matrix nhất quán với `Acc = 0.8263`, `Fgt = 0.0761`

Kiểm bằng `src/uavcl/models/slda.py` + `metrics.json` + `acc_matrix.csv`.

### B2. Ablation: cải thiện đến từ đâu? ⏱ 4h máy

Cùng feature frozen ViT, chỉ đổi bộ đọc:

| Bộ đọc | Dùng gì | Kỳ vọng |
|---|---|---|
| NCM | trung bình lớp | 0.7114 |
| **SLDA (Σ chung)** | trung bình + hiệp phương sai | **0.8263** |
| SLDA với Σ = I | trung bình, bỏ hiệp phương sai | ≈ NCM nếu giả thuyết đúng |
| NCM + whitening cố định | Σ tính 1 lần ở task 0 | tách "Σ có cần streaming không" |

**Đây là thí nghiệm quan trọng nhất của hướng B.** Nó biến "SLDA thắng" thành
"**hiệp phương sai chung là thứ tạo ra +11.5 điểm**" — một phát biểu có cơ chế.

### B3. Độ nhạy tham số ⏱ 4h máy

SLDA có hệ số shrinkage cho Σ. Quét `[1e-4, 1e-3, 1e-2, 1e-1]`, 3 seed.
**Tiêu chí:** accuracy ổn định trong dải rộng → phương pháp bền, không cần tuning.

### B4. Độ dài stream ⏱ 12h máy

Chạy 5 / 9 / 15 / 45 task trên cùng 45 lớp.

**Vì sao quan trọng:** đây là điểm yếu tiềm tàng của SLDA — Σ ước lượng từ dữ liệu
streaming, task càng nhỏ thì mỗi lần cập nhật càng ít mẫu. 45 task × 1 lớp là ca khó nhất.
Nếu SLDA vẫn giữ được thì kết luận rất mạnh.

### B5. Liên-dataset EuroSAT ⏱ 6h máy

Khoá tham số trước. **Tiêu chí:** vẫn vượt Frozen NCM ≥ 5 điểm.

### B6. Chi phí triển khai ⏱ 4h người

Đo thật, không ước lượng:

| Chỉ số | Cách đo | Vì sao cần |
|---|---|---|
| Bộ nhớ | `d×d` float32 = 384² × 4B ≈ **590 KB** | So với replay buffer tăng vô hạn |
| Latency/ảnh | `time.perf_counter` quanh predict | Drone có ngân sách thời gian thực |
| Thời gian cập nhật | quanh `fit`/`update` | Có kịp cập nhật giữa hai khung hình? |
| Năng lượng | `powermetrics` (macOS) hoặc RAPL | Ngân sách pin của UAV |

**Đây là phần biến kết quả học thuật thành lập luận triển khai được**, và hiện chưa ai làm.

### B7. So với baseline CL kinh điển ⏱ 8h máy

EWC · LwF · iCaRL · Replay — trên **cùng split mới** `combined31500_v2` (các số cũ đều
ở split cũ, không so được).

### B8. Open-set ⏱ 4h máy

Đã có `metrics/openset.py`. Đo AUC của SLDA vs NCM. Drone gặp vật thể lạ là chuyện thường —
biết nói "tôi không biết" có giá trị thực tế cao.

### B9. SLDA + Titans (đưa hướng A vào làm một tầng) ⏱ 8h máy

Vẫn nên chạy, nhưng ở vai trò **một mục ablation của hướng B**, không phải câu hỏi trung tâm.
Trả lời được "feature thích nghi có cộng thêm gì lên trên bộ đọc bậc 2 không".

## Tổng chi phí hướng B

**~50 giờ máy** (≈ 18h nếu 3 VM song song) + khoảng 1 tuần công.
**Bằng một nửa hướng A.**

## Rủi ro

1. **Cao nhất:** B1 phát hiện bug → mất kết quả chính. Nên làm trước tiên
2. Đề tài phải định vị lại: NL từ đối tượng nghiên cứu thành **baseline đối chứng**
3. SLDA là phương pháp đã có từ 2020, không phải đóng góp mới — **đóng góp phải nằm ở
   việc áp dụng cho UAV/viễn thám + phân tích cơ chế (B2) + chi phí triển khai (B6)**

## Kịch bản kết thúc

Hướng B gần như **chắc chắn có kết quả viết được**, vì con số chính đã nằm trong tay,
chỉ cần xác minh và phân tích. Rủi ro duy nhất là B1.

---

# PHẦN 4 — SO SÁNH & KHUYẾN NGHỊ

| | Hướng A (NL) | Hướng B (SLDA) |
|---|---|---|
| Chi phí máy | ~100h | **~50h** |
| Công sức | ~2 tuần | **~1 tuần** |
| Kết quả đã có trong tay | Không (đều nhiễm) | **Có (0.8263, 3 seed)** |
| Khả năng ra kết quả dương | Trung bình | **Cao** |
| Trung thành đề tài gốc | **Cao** | Thấp — phải định vị lại |
| Đóng góp mới | Cao nếu thắng | Trung bình (phương pháp cũ, ứng dụng mới) |
| Rủi ro lớn nhất | A1 không đạt → mất trắng | B1 phát hiện bug |

## Khuyến nghị: làm cả hai, nhưng KHÔNG song song

**Bước 0 (2 giờ, chung cho cả hai hướng, làm ngay):**

1. B1 — xác minh SLDA. Rẻ nhất, rủi ro cao nhất.
2. Commit toàn bộ `metrics*.json` vào git.
3. Viết một trang ghi nhận các số cũ đo khi cổng hỏng, kèm bảng ⚠️SAT làm bằng chứng.

**Bước 1 (8 giờ máy): A1.** Đây là điểm rẽ.

- **A1 đạt** → còn nhiều thời gian thì đi hướng A; ít thời gian thì lấy A1 làm một chương
  của báo cáo hướng B
- **A1 không đạt** → dồn toàn bộ vào hướng B, và phát hiện cổng quên trở thành chương
  "vì sao phương pháp phức tạp không thắng"

**Cách viết báo cáo giữ được cả hai:**

> Chương 1–2: bài toán + baseline
> Chương 3: **SLDA — phương pháp chốt** (0.8263, phân tích cơ chế B2, chi phí B6)
> Chương 4: **Nested Learning — vì sao không thắng** (phát hiện cổng quên, bản vá, A1)
> Chương 5: kết luận + hướng mở

Cấu trúc này **không lãng phí bất cứ thứ gì đã làm**, và không phụ thuộc vào việc A1
ra kết quả nào.

---

## Việc cần làm ngay hôm nay

| # | Việc | Thời gian |
|---|---|---|
| 1 | B1 — xác minh SLDA (4 điểm kiểm) | 2h |
| 2 | Commit metrics JSON vào git | 30ph |
| 3 | `pytest -q` xác nhận bản vá | 5ph |
| 4 | Khởi động A1 (2 run 9 task) | bấm chạy rồi để đó |

Xong 4 việc này là có đủ dữ kiện để chọn hướng.
