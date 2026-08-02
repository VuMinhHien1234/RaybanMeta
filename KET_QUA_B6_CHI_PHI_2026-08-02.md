# B6 — Chi phí triển khai: SLDA vs Titans (số đo thật)

Ngày: 2026-08-02 · Máy: MacBook Pro (MPS + CPU) · `scripts/bench_cost.py`
Dữ liệu thô: `bench_vits.json`, `bench_vitl.json`

Backbone bị loại khỏi phép đo có chủ đích — nó **chung cho mọi phương pháp**. Mọi con số
dưới đây là **chi phí TĂNG THÊM so với ViT trần**.

---

## Bảng chính

### ViT-S (D = 384) — cấu hình đang dùng

| Phương pháp | Thiết bị | Bộ nhớ | Suy luận | Cập nhật |
|---|---|---:|---:|---:|
| SLDA streaming (fp64) | CPU | 1,388 MB | 0,0003 ms | 0,0083 ms |
| SLDA identity (fp64) | CPU | 1,388 MB | 0,0003 ms | 0,0069 ms |
| SLDA frozen (fp64) | CPU | 1,388 MB | 0,0003 ms | 0,0068 ms |
| **SLDA streaming (fp32)** | MPS | **0,728 MB** | 0,0071 ms | 0,0229 ms |
| **Titans (self-mod, depth 3)** | MPS | **102,682 MB** ² | **2,1782 ms** | — ¹ |

### ViT-L (D = 1024) — nếu đổi backbone lớn hơn

| Phương pháp | Thiết bị | Bộ nhớ | Suy luận |
|---|---|---:|---:|
| SLDA streaming (fp64) | CPU | 8,942 MB | 0,0004 ms |
| SLDA streaming (fp32) | MPS | 4,563 MB | 0,0081 ms |
| **Titans (self-mod, depth 3)** | MPS | **729,948 MB** ² | **28,3667 ms** |

¹ Titans làm test-time learning **ngay trong forward** — ghi bộ nhớ không tách rời khỏi
suy luận được. Đây tự nó đã là một đặc điểm triển khai đáng lưu ý (xem mục 4).

² Gồm **tham số + state**. Bản đo đầu chỉ đếm tham số vì hàm duyệt state không nhận ra
NamedTuple của `titans-pytorch` (đã sửa). Tách ra:

| | Tham số | State | Tổng |
|---|---:|---:|---:|
| ViT-S | 31,90 MB | **70,79 MB** | 102,68 MB |
| ViT-L | 226,61 MB | **503,34 MB** | 729,95 MB |

**State lớn gấp 2,2 lần chính tham số** — đây mới là phần drone phải mang theo và ghi lại
liên tục. Lưu ý: kích thước state phụ thuộc độ dài chuỗi/batch; file `memory_state.pt` của
run thật trên đĩa là **55 MB**. Nên khoảng thật là **55–103 MB**, tức **40–74×** so với SLDA.

---

## 1. Bộ nhớ: chênh 74–82 lần

| | SLDA fp64 | SLDA fp32 | Titans | Tỷ lệ (so fp64) |
|---|---:|---:|---:|---:|
| ViT-S | 1,388 MB | 0,728 MB | 102,682 MB | **74,0×** |
| ViT-L | 8,942 MB | 4,563 MB | 729,948 MB | **81,6×** |

So với bản SLDA float32, tỷ lệ lên tới **141×** ở ViT-S.

Con số này khớp bậc độ lớn với quan sát độc lập: file `memory_state.pt` của run thật trên
đĩa là **55 MB**, trong khi toàn bộ trạng thái SLDA gói lại chỉ hơn 1 MB.

**`float32` cắt được 48–49%** (1,388 → 0,728 MB · 8,942 → 4,563 MB) và cho phép chạy trên
MPS/NPU biên. Cần đo accuracy trước khi dùng chính thức — đã có cờ `slda.stats_dtype`.

## 2. Độ trễ: chênh 307–3494 lần (đo CÙNG thiết bị)

Để tránh so chéo thiết bị, dùng hai dòng cùng chạy trên MPS:

| | Titans | SLDA fp32 | Tỷ lệ |
|---|---:|---:|---:|
| ViT-S | 2,1823 ms | 0,0071 ms | **307×** |
| ViT-L | 28,3007 ms | 0,0081 ms | **3494×** |

### Đặt vào ngân sách thời gian thực của drone

Camera 30 khung/giây → **33,3 ms cho mỗi khung**, và trong đó backbone đã ăn phần lớn.

| Backbone | Titans chiếm | Còn lại cho backbone + mọi thứ khác |
|---|---:|---|
| ViT-S | **6,6%** | Chấp nhận được |
| ViT-L | **85,0%** | **Không khả thi** |

SLDA chiếm **0,02%** ở ViT-S và **0,024%** ở ViT-L — thực tế là bằng không.

## 3. Độ mở rộng theo chiều feature

D tăng 2,67 lần (384 → 1024), tức `D²` tăng 7,1 lần:

| Chỉ số | 384 | 1024 | Tỷ lệ | So với D² |
|---|---:|---:|---:|---|
| Bộ nhớ SLDA | 1,388 | 8,942 | 6,4× | ✅ đúng O(D²) |
| Bộ nhớ Titans | 102,682 | 729,948 | 7,1× | ✅ đúng O(D²) |
| Độ trễ SLDA | 0,0071 | 0,0081 | **1,1×** | 🟢 gần như phẳng |
| Độ trễ Titans | 2,1823 | 28,3007 | **13,0×** | 🔴 **tệ hơn O(D²)** |

Hai dòng cuối là điểm đáng chú ý nhất:

- **SLDA gần như không đắt thêm** khi đổi backbone lớn — vì suy luận chỉ là một phép nhân
  `f @ W` với `W` cỡ `D×C`, quá nhỏ so với chính backbone
- **Titans đắt lên nhanh hơn cả O(D²)** (13× so với 7,1× kỳ vọng) — nhiều khả năng bị chặn
  bởi băng thông bộ nhớ chứ không phải phép tính

## 4. Ba khác biệt định tính, không nằm trong bảng số

### a. SLDA **không thể** phát tán, Titans thì có

Σ là ma trận hiệp phương sai, **nửa xác định dương theo cấu trúc**; cộng `εI` đảm bảo luôn
nghịch đảo được. Không tồn tại kịch bản nào SLDA nổ.

Titans thì đã nổ — `norm(state) = 1 757 552` ở `m3_s1`.

Với drone bay hàng tháng không người giám sát, **"không thể hỏng" đáng giá hơn "tốt hơn vài
điểm phần trăm"**.

### b. Titans sửa trọng số ngay trong forward

Hệ quả trực tiếp cho triển khai:

- Cùng một ảnh đưa vào hai lần cho **hai kết quả khác nhau** — khó kiểm định
- Trạng thái **~100 MB** phải ghi lại liên tục
- Có thể trôi dần trong thực địa mà không ai phát hiện

SLDA tất định: cùng thống kê tích luỹ thì cùng đầu ra.

### c. Ràng buộc `float64` — phát hiện khi đo

**MPS không hỗ trợ float64**, và nhiều NPU biên cũng vậy. SLDA hiện dùng fp64 cho ba bộ đếm
(có lý do: cộng dồn hàng chục nghìn mẫu rồi nghịch đảo ma trận). Hai cách xử lý:

| Cách | Bộ nhớ | Chạy trên NPU? | Ghi chú |
|---|---:|---|---|
| (a) `stats_dtype=float32` | 0,728 MB | ✅ | Phải đo accuracy trước |
| (b) backbone trên accelerator, thống kê SLDA ở **CPU-fp64** | 1,388 MB | ✅ | **Khuyến nghị** |

**Cách (b) không phải giải pháp chữa cháy — nó nhanh hơn.** Số đo cho thấy SLDA trên CPU
(0,0003 ms) **nhanh hơn 23 lần** so với chính nó trên MPS (0,0071 ms), vì ma trận `384×45`
quá nhỏ, chi phí khởi động kernel GPU lớn hơn cả phép tính.

---

## Kết luận cho phần "phù hợp UAV"

| Trục | Người thắng | Biên độ |
|---|---|---|
| Bộ nhớ | SLDA | **74–82×** |
| Độ trễ | SLDA | 307–3494× |
| Mở rộng theo D | SLDA | 1,1× so với 13,0× |
| Bảo đảm không phát tán | SLDA | có / không |
| Tính tất định | SLDA | có / không |
| **Khả năng thích nghi feature** | **Titans** | SLDA không làm được |

Dòng cuối là **giới hạn thật của SLDA** và cũng là chỗ duy nhất Nested Learning có lý do
tồn tại: khi drone bay vào **điều kiện thị giác mới** (ban đêm, hồng ngoại, sương mù, mùa
khác), feature pretrain hỏng và SLDA bó tay.

> Luận điểm cho Nested Learning **không nằm ở class-incremental, mà ở domain shift.**

Trong bài toán hiện tại (cùng dataset, lớp mới), feature đóng băng là đủ — nên phương pháp
phức tạp không có đất diễn, đúng như số liệu cho thấy.

---

## Cần làm tiếp trong B6

- [ ] Đo **năng lượng** (`powermetrics` trên macOS hoặc RAPL trên Linux) — chưa có
- [ ] Đo trên **phần cứng biên thật** (Jetson/Raspberry Pi) thay vì MacBook
- [ ] Đo accuracy của `stats_dtype=float32` trên dữ liệu thật để chốt cách (a) hay (b)
- [x] ~~Sửa `_state_bytes`~~ — đã sửa và chạy lại. Số Titans giờ gồm cả state (tăng từ
      31,9 lên 102,7 MB ở ViT-S)
- [ ] Đo state ở nhiều độ dài chuỗi khác nhau để chốt con số cho báo cáo (hiện 55–103 MB)
