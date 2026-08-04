# Bài toán và mục tiêu

Xác định lại ngày 2026-08-04, sau khi có kết quả D10/D11 · thay cho mọi phát biểu trước đó

---

# 1. Kịch bản

Một UAV bay tuần tra một khu vực cố định.

**Lần đầu**, khu vực được bay qua và dữ liệu được **gán nhãn** (thủ công hoặc bán tự động) —
đây là pha hiệu chỉnh dưới đất, làm một lần, không giới hạn tính toán.

**Những lần sau**, drone bay lại **cùng khu vực đó**, nhưng điều kiện quan sát đã khác: giờ
trong ngày, mùa, mây, sương, bụi bám ống kính, độ cao bay. Không có nhãn. Không có người.

Hệ phải làm được ba việc, và ba việc này có **tần số thay đổi khác nhau hàng triệu lần**:

| | Việc | Đổi khi nào |
|---|---|---|
| a | Vẫn nhận đúng *"đây là sân bay, kia là ruộng"* | gần như không bao giờ |
| b | Nhận ra *"điều kiện hôm nay giống chuyến bay tháng trước"* | mỗi chuyến bay |
| c | Bám theo *"nắng đang gắt dần suốt 10 phút qua"* | mỗi khung hình |

---

# 2. Phát biểu hình thức

**Pha 1 — hiệu chỉnh, CÓ nhãn, offline**

Cho tập `D₀ = {(xᵢ, yᵢ)}`, `yᵢ ∈ {1..C}`, thu dưới điều kiện `e₀`.
Không giới hạn thời gian và tính toán. Chạy một lần.

**Pha 2 — triển khai, KHÔNG nhãn, trực tuyến**

Nhận dòng `x₁, x₂, …` dưới các điều kiện `e₁, e₂, …` trong đó:

- `e_t` **biến thiên trơn** (không nhảy đột ngột) — mặt trời lặn dần, sương dâng dần
- `e_t` **có thể quay lại** giá trị đã gặp — mùa và giờ đều tuần hoàn
- `e_t` **không quan sát được** — hệ không được cho biết "hôm nay là mùa đông"
- Tập lớp **không đổi**: `y ∈ {1..C}`, đúng những lớp đã gán nhãn ở pha 1

Với mỗi `x_t` phải trả lời `ŷ_t` **ngay lập tức**, trước khi thấy `x_{t+1}`.
Không được lưu lại ảnh thô. Không được đọc lại dữ liệu cũ.

---

# 3. Cho trước và phải học

| | Có sẵn | Phải tự học ở pha 2 |
|---|---|---|
| Nhãn | chỉ pha 1 | không có |
| Danh tính lớp | học ở pha 1 | **giữ nguyên**, không học lại |
| Điều kiện `e_t` | không bao giờ biết | **suy ra từ dữ liệu không nhãn** |
| Ánh xạ điều kiện → hiệu chỉnh | không | **học và ghi nhớ** |

Điểm mấu chốt: **không cần nhãn để bám *điều kiện*, chỉ cần nhãn để học *lớp*.**
Hai thứ đó tách rời được, và chính chỗ tách đó là nơi bài toán trở nên giải được.

---

# 4. Ràng buộc phần cứng

Drone, không phải máy chủ. Số đo thật của dự án (B6, ViT-S/16, D=384):

| | Bộ nhớ | Độ trễ suy luận |
|---|---:|---:|
| Ngân sách một khung hình 30 fps | — | **33,3 ms** |
| SLDA (fp64) | 1,53 MB | 0,0071 ms |
| Titans (self-mod, depth 3) | **102,68 MB** | 2,1782 ms |
| Titans ở ViT-L | 729,95 MB | **28,37 ms** = 85% ngân sách |

Ràng buộc đặt ra: **tổng bộ nhớ thêm ≤ 10 MB, độ trễ thêm ≤ 5% ngân sách khung hình.**
Không được lưu ảnh thô (dung lượng + riêng tư).

---

# 5. Mục tiêu

| | Mục tiêu | Đo bằng |
|---|---|---|
| **O1** | **Giữ danh tính.** Điều kiện đổi thì vẫn nhận đúng lớp đã học | Acc ở mọi điều kiện |
| **O2** | **Bám trôi không nhãn.** Điều kiện đổi dần thì tự chỉnh theo | Thời gian hồi phục |
| **O3** | **Ghi nhớ điều kiện.** Gặp lại điều kiện cũ thì nhận ra ngay, **không học lại** | ⭐ Lợi ích khi quay lại |
| **O4** | **Chạy được trên drone** | MB và ms |

**O3 là mục tiêu trung tâm** và là thứ phân biệt bài toán này với "test-time adaptation" thông
thường. Thích nghi trực tuyến thì nhiều người làm; **ghi nhớ để lần sau khỏi thích nghi lại**
mới là phần chưa được giải quyết, và là phần đúng nghĩa "học liên tục".

---

# 6. Thước đo

Thước đo chuẩn của continual learning (`AAA`, `Average Accuracy`, `Forgetting`) **không đo được
O3**. Bài học từ D10: λ hoạt động tốt nhưng `AAA` che mất hoàn toàn, phải đổi thước đo mới thấy
(+3,91 điểm ở mức trôi 100%, 3,10σ — vô hình trong `AAA`).

Ba thước đo cần định nghĩa:

| Thước đo | Công thức | Đo mục tiêu |
|---|---|---|
| **Acc điều kiện hiện tại** | trung bình đường chéo ma trận accuracy | O1 |
| **Thời gian hồi phục** | số mẫu để accuracy về 95% mức ổn định sau khi `e` đổi | O2 |
| ⭐ **Lợi ích khi quay lại** | `Acc(gặp lại điều kiện X) − Acc(lần đầu gặp X)` | **O3** |

Lợi ích khi quay lại ≈ 0 nghĩa là hệ **không ghi nhớ được điều kiện** — nó thích nghi lại từ
đầu mỗi lần, và O3 thất bại dù O1/O2 có tốt đến đâu.

---

# 7. Vì sao đây là bài toán Nested Learning

Ba yêu cầu ở mục 1 có tần số cập nhật khác nhau **hàng triệu lần**. Không phải chọn lựa thẩm
mỹ — đó là **cấu trúc của chính bài toán**. Và bộ nhớ nhiều tầng theo tần số chính là
**Continuum Memory System**, đóng góp thứ ba của bài Nested Learning.

| Tầng | Giữ gì | Tần số | Nhãn |
|---|---|---|---|
| Chậm | danh tính lớp | 1 lần / hiệu chỉnh | có |
| Trung | các điều kiện đã từng gặp | 1 lần / chuyến bay | không |
| Nhanh | điều kiện lúc này | 1 lần / khung hình | không |

**Quan trọng — "theo Nested Learning" KHÔNG đồng nghĩa "dùng Titans".** NL là khung lý thuyết;
Titans là một hiện thực hoá. Một hệ ba tầng cài bằng công thức đóng cũng là hệ NL, và có hai
tính chất mà bản gradient không có:

1. **Cổng không thể trôi ra biên.** Đã đo: cổng học được bão hoà ở 9/13 run
   (class-incremental) và 0–4/9 task (drift). Cổng cố định thì **về mặt cấu trúc** không thể.
2. **Rẻ hơn 67 lần** về bộ nhớ, 307 lần về độ trễ.

Đổi lại, bản đóng chỉ biểu diễn được biến đổi **affine**. Nếu điều kiện tác động **phi tuyến**
lên không gian feature thì phải quay lại Titans — và **thí nghiệm T1 sẽ trả lời chính xác câu
đó trong 15 phút** trước khi viết dòng code nào.

---

# 8. Giả định (nêu rõ để còn xét lại)

| # | Giả định | Nếu sai thì sao |
|---|---|---|
| A1 | **Tập lớp đóng** — không có lớp mới ở pha 2 | Cần thêm phát hiện open-set. Dự án đã có `openset_eval.py`, mở rộng được |
| A2 | **Backbone đóng băng** (ViT-S/16 pretrained) | Nếu điều kiện làm vỡ feature thì căn chỉnh không cứu được — cần thích nghi backbone |
| A3 | Pha 1 có đủ nhãn và phủ hết các lớp | Thiếu lớp nào thì lớp đó không bao giờ nhận được |
| A4 | Điều kiện biến thiên **trơn**, không nhảy bậc | Nhảy đột ngột thì tầng nhanh trễ; tầng trung phải gánh |
| A5 | Tỷ lệ lớp trong dòng dữ liệu **không lệch quá mạnh** | Bay thành phố → rừng làm thống kê toàn cục dịch, giả dạng đổi điều kiện. **T2 kiểm chính chỗ này** |

A5 là giả định nguy hiểm nhất và đã có thí nghiệm riêng để kiểm.

---

# 9. Ngoài phạm vi

Ghi rõ để không bị trách là thiếu, và để khỏi sa đà:

- **Lớp mới ở pha 2** (open-set / novelty) — giả định A1
- **Thích nghi backbone** — chỉ chỉnh phần đọc, không chỉnh mắt
- **Bay nhiều drone, chia sẻ bộ nhớ** — một drone, một bộ nhớ
- **Định vị / SLAM** — bài toán này chỉ phân loại, không ước lượng vị trí
- **Nén mô hình, lượng tử hoá** — trực giao với hướng nghiên cứu

---

# 10. Trạng thái hiện tại

| | Đã có | Còn thiếu |
|---|---|---|
| Dữ liệu | RESISC45 45 lớp · trôi 5 phép vật lý đã hiệu chỉnh (NCM 0,6900 → 0,5310) | **Stream có QUAY LẠI** — hiện chỉ trôi đơn điệu |
| Tầng chậm | ✅ SLDA `μ_c, Σ`, đo được 0,7817 | — |
| Tầng nhanh | ❌ | `m_t, C_t` + căn chỉnh |
| Tầng trung | ❌ | Ngân hàng chế độ điều kiện |
| Thước đo | ✅ Acc, AAA, Forget | **Lợi ích khi quay lại**, Thời gian hồi phục |
| Chẩn đoán | — | T1 (trục điều kiện), T2 (bẫy tỷ lệ lớp) |

**Cảnh báo quan trọng về kết quả D10:** con số *λ=0,99 hơn λ=1 là 3,91 điểm ở mức trôi 100%*
đo trên stream **trôi đơn điệu, không lặp**. Ở đó quên điều kiện cũ gần như miễn phí vì không
bao giờ gặp lại. Trong bài toán **có quay lại**, quên đi là phải học lại — **con số này không
chuyển sang được và có thể đảo dấu.**

Kế hoạch triển khai: `KE_HOACH_BA_TANG_2026-08-04.md`
