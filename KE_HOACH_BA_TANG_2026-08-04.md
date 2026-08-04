# Kế hoạch — Bộ nhớ BA TẦNG cho UAV bay lặp lại

Lập 2026-08-04 · hướng đi sâu, thay cho việc chạy thêm campaign Titans

---

## Bài toán, phát biểu lại cho chính xác

> Drone bay qua một khu vực **đã được gán nhãn** → ghi nhớ. Những lần sau **quay lại** khu vực
> đó, dù điều kiện quan sát đã thay đổi ít nhiều, vẫn phải nhận ra đúng, **và** tự điều chỉnh
> theo điều kiện đang đổi dần.

Ba yêu cầu tách bạch, và chúng có tần số cập nhật **khác nhau hàng triệu lần**:

| Yêu cầu | Đổi khi nào | Cần nhãn |
|---|---|---|
| *"đây là sân bay, kia là ruộng"* | gần như không bao giờ | có — một lần |
| *"điều kiện này giống chuyến bay tháng trước"* | mỗi chuyến bay | không |
| *"nắng đang gắt dần trong 10 phút qua"* | mỗi khung hình | không |

Chính vì tần số khác nhau như vậy mà **bộ nhớ nhiều tầng** không phải lựa chọn thẩm mỹ — nó là
hệ quả bắt buộc của bài toán. Đây cũng đúng là **Continuum Memory System**, đóng góp thứ ba của
bài Nested Learning, nhưng cài bằng công thức đóng thay vì gradient.

---

## ⛔ Điều phải thừa nhận trước: bộ dữ liệu hiện tại KHÔNG khớp bài toán

Stream D10 trôi **đơn điệu** 0% → 100%, không lặp lại bao giờ. Bài toán thật thì **quay lại**.

| | D10 đã chạy | Bài toán thật |
|---|---|---|
| Điều kiện cũ có gặp lại | không bao giờ | có — mùa, giờ đều tuần hoàn |
| Quên điều kiện cũ tốn gì | gần như không | **đắt** — lần sau phải học lại |
| Kết luận về λ | λ=0,99 hơn λ=1 **3,9 điểm** ở trôi 100% | **có thể ngược lại** |

Con số +3,9 điểm của D10 **không chuyển sang được**. Nó đúng vì điều kiện cũ không quay lại;
quên đi là lãi. Trong bài toán của bạn, quên mùa đông vào tháng 6 nghĩa là tháng 12 học lại
từ đầu.

Nên nhóm R dưới đây (dựng stream có quay lại) là **bắt buộc**, không phải tuỳ chọn.

---

# NHÓM T — Chẩn đoán trước khi code (2 giờ, ⛔ cửa chặn)

## T1 — Trôi có phải BIẾN ĐỔI CHUNG không? ⭐ quyết định cả kiến trúc

**Câu hỏi.** Khi điều kiện đổi, feature của 45 lớp có dịch **cùng một hướng** không?

**Cách đo.** Với mỗi lớp c, tính `v_c = μ_c(trôi 100%) − μ_c(trôi 0%)`. Rồi đo cosine giữa các
`v_c` từng đôi một, và tỉ lệ phương sai mà thành phần chính thứ nhất giải thích được.

**Vì sao quyết định mọi thứ:**

| Kết quả | Kiến trúc kéo theo |
|---|---|
| `cos > 0,8`, PC1 giải thích > 70% | ✅ Điều kiện = **một trục chung** → tầng nhanh/trung chỉ cần lưu vài số. Rẻ. **Titans là thừa** |
| `cos ≈ 0,3–0,8` | ⚠️ Có trục chung nhưng còn phần riêng theo lớp → tầng trung phải lưu nhiều hơn |
| `cos < 0,3` | ❌ Không có "điều kiện toàn cục" → căn chỉnh tuyến tính bất khả thi, **phải quay lại Titans** |

**Sản phẩm phụ quan trọng:** vector riêng đầu tiên của `{v_c}` chính là **trục điều kiện**.
Ta sẽ dùng nó ở T2 và M1.

- File: `scripts/do_truc_dieu_kien.py`
- Thời gian: 1h code + 15 phút chạy trên Mac
- Rủi ro: thấp

## T2 — Bẫy: đổi TỶ LỆ LỚP có giả dạng đổi điều kiện không?

**Vì sao phải hỏi.** Tầng nhanh dự định theo dõi `m_t` = trung bình feature toàn cục. Nhưng `m_t`
dịch vì **hai** nguyên nhân: điều kiện đổi, **và** drone bay từ thành phố sang rừng làm tỷ lệ
lớp đổi. Trộn hai thứ đó thì hệ sẽ "sửa" luôn cả thông tin lớp và làm hỏng phân loại.

**Đây là bẫy chết người nhất của cả hướng này.**

**Cách đo.** Giữ **nguyên** mức trôi, chỉ đổi tỷ lệ lớp (ví dụ 80% ruộng so với 80% đô thị). Đo
`m_t` dịch bao nhiêu, và **hình chiếu của nó lên trục điều kiện T1** là bao nhiêu.

| Kết quả | Kéo theo |
|---|---|
| Dịch do tỷ lệ lớp gần **vuông góc** với trục điều kiện | ✅ Chỉ cần **chiếu lên trục điều kiện** là tách được. Rẻ |
| Chiếu lên trục điều kiện đáng kể | ⚠️ Phải cân bằng lớp bằng nhãn giả, hoặc dùng thống kê bậc hai |

- File: thêm cờ `--doi-ty-le-lop` vào script T1
- Thời gian: 1h
- Rủi ro: thấp (chỉ đo)

---

# NHÓM R — Stream CÓ QUAY LẠI (4 giờ, rủi ro trung bình)

## R1 — `src/uavcl/data/revisit.py`

Sinh lịch bay: `n_chuyen` chuyến, mỗi chuyến đi qua **toàn bộ** địa điểm, mỗi chuyến có một
**điều kiện** riêng.

```python
def lich_bay(n_chuyen, che_do, seed):
    """Trả về list mức trôi cho từng chuyến bay."""
    # 'tuan_hoan' : sin — mùa quay vòng, ĐIỀU KIỆN CŨ QUAY LẠI   ← kịch bản chính
    # 'troi_dan'  : tăng đều — như D10, để so sánh
    # 'hon_hop'   : trôi dài hạn + dao động mùa chồng lên
```

Ba chế độ có chủ đích: `tuan_hoan` là bài toán thật, `troi_dan` là mốc so với D10,
`hon_hop` là thực tế nhất (khí hậu ấm dần **cộng** mùa quay vòng).

## R2 — Đánh dấu "chuyến bay này lặp lại điều kiện nào"

Mỗi chuyến ghi kèm nhãn `mode_that` (chỉ dùng để **chấm điểm**, model không được thấy). Nhờ đó
mới đo được: khi gặp lại một điều kiện đã từng gặp, hệ có nhận ra không.

## R3 — Nối vào `stream.py` / `loaders.py`

Thêm `stream_type: revisit`. Mặc định tắt → mọi config cũ bất biến.

## R4 — Config

`revisit_slda_*.yaml` — 8 chuyến bay × 45 lớp, `che_do: tuan_hoan`, chu kỳ 4 chuyến
(chuyến 5 lặp lại điều kiện chuyến 1).

- Thời gian: 4h · Rủi ro: **trung bình** — R3 đụng vào code đang chạy được

---

# NHÓM Đ — ⭐ ĐỊNH NGHĨA THƯỚC ĐO (1 giờ, nhưng quan trọng nhất)

Bài học lớn nhất từ D10: **thước đo sai thì che mất kết quả đúng.** λ hoạt động tốt, nhưng
`AAA` trung bình trên mọi task đã che mất, phải đo theo điều kiện hiện tại mới thấy.

Ở đây cũng vậy — `AAA` **không đo được** thứ ta quan tâm. Cần ba thước đo mới:

| Thước đo | Định nghĩa | Đo cái gì |
|---|---|---|
| **Acc lần đầu** | accuracy chuyến bay đầu tiên gặp điều kiện X | khả năng tổng quát hoá |
| ⭐ **Lợi ích khi quay lại** | `Acc(gặp lại X) − Acc(lần đầu gặp X)` | **tầng trung có tác dụng không** |
| **Thời gian hồi phục** | số mẫu để accuracy về 95% mức ổn định sau khi điều kiện đổi | tầng nhanh bám nhanh không |

**Lợi ích khi quay lại là con số của cả dự án.** Nếu tầng trung hoạt động, chuyến bay thứ 5
(lặp điều kiện chuyến 1) phải **tốt hơn hẳn** chuyến 1 — vì hệ đã có sẵn phép căn chỉnh cho
điều kiện đó, không phải học lại.

Nếu con số này ≈ 0 thì tầng trung vô dụng, và nên bỏ nó đi cho gọn.

- File: `src/uavcl/metrics/revisit.py` + nối vào `run_g1.py`
- Thời gian: 1h · Rủi ro: thấp

---

# NHÓM M — Ba tầng (10 giờ)

## M1 — Tầng NHANH: thống kê điều kiện hiện tại (3h, rủi ro thấp)

```python
class TangNhanh:
    """m_t, C_t toàn cục — KHÔNG cần nhãn. Bám điều kiện đang đổi."""
    # cập nhật mỗi batch:  m_t ← λ·m_t + (1−λ)·mean(f)
    # căn chỉnh:           f' = (f − m_t)·(s₀/s_t) + m₀
    # λ ≈ 0.99 -> cửa sổ ~100 mẫu ~ vài chục giây bay
```

Nếu T2 báo tỷ lệ lớp gây nhiễu, chỉ căn chỉnh **thành phần nằm trên trục điều kiện** của T1,
giữ nguyên phần vuông góc (phần mang thông tin lớp).

Chỉ giữ đường chéo của `C_t`, không giữ ma trận đầy đủ: **3 KB** thay vì 1,2 MB, và ổn định hơn
nhiều về mặt ước lượng.

## M2 — Tầng TRUNG: ngân hàng chế độ điều kiện (5h, ⚠️ rủi ro CAO NHẤT)

```python
class NganHangCheDo:
    """K chế độ điều kiện đã từng gặp. Đây là thứ làm việc QUAY LẠI trở nên rẻ."""
    # mỗi chế độ:  (m_k, s_k, so_lan_gap)
    # khớp:        k* = argmin ||m_t − m_k||  trên TRỤC ĐIỀU KIỆN
    # nếu ||m_t − m_k*|| > nguong  ->  tạo chế độ MỚI
    # nếu K > K_max                ->  gộp hai chế độ gần nhau nhất
```

Ba tham số phải chọn, và **chọn sai thì tầng này thành vô dụng hoặc có hại**:

| Tham số | Quá nhỏ | Quá lớn |
|---|---|---|
| `nguong` tạo chế độ mới | nổ số chế độ, mỗi cái vài mẫu → nhiễu | mọi điều kiện gộp làm một → mất tác dụng |
| `K_max` | gộp quá tay, mất phân biệt | tốn bộ nhớ |
| λ trong mỗi chế độ | chế độ không ổn định | chế độ không cập nhật được |

Đây là chỗ dễ sai nhất của cả kế hoạch. **Phải có cửa chặn riêng**: chạy với `mode_that` đã
biết, xem số chế độ hệ tự tạo có khớp số điều kiện thật không.

## M3 — Hợp nhất (2h, rủi ro trung bình)

Nối ba tầng vào `SLDAClassifier`. **Mọi tầng mặc định TẮT** → `slda.py` hiện tại bất biến,
mọi kết quả D10 vẫn tái lập được.

## M4 — Test (3h)

| Test | Chặn cái gì |
|---|---|
| Tắt hết tầng → trùng bit với SLDA hiện tại | bất biến ngược |
| Tầng nhanh trên dữ liệu KHÔNG trôi → không đổi gì | không tự làm hỏng khi không có trôi |
| Đưa 4 điều kiện tách biệt → tạo đúng 4 chế độ | ⭐ M2 khớp đúng |
| Gặp lại điều kiện cũ → khớp lại chế độ CŨ, không tạo mới | ⭐ **bẫy chính của M2** |
| Đổi tỷ lệ lớp mà giữ điều kiện → **không** tạo chế độ mới | ⭐ bẫy T2 |
| `m_t` hội tụ đúng `λ` lý thuyết | λ cài đúng |

Ba test có ⭐ là ba chỗ hỏng có khả năng cao nhất.

---

# NHÓM E — Thí nghiệm (1 đêm máy)

| Arm | Tầng chậm | Tầng nhanh | Tầng trung | Trả lời |
|---|---|---|---|---|
| 1 | ✅ | ❌ | ❌ | mốc — SLDA λ=1 hiện tại |
| 2 | ✅ | ❌ | ❌ | SLDA λ=0,99 — kiểm xem +3,9 điểm của D10 còn không khi có quay lại |
| 3 | ✅ | ✅ | ❌ | **căn chỉnh không nhãn đáng giá bao nhiêu** |
| 4 | ✅ | ✅ | ✅ | **⭐ ngân hàng chế độ đáng giá bao nhiêu** |
| 5 | ✅ | ✅ | ✅ | `che_do: troi_dan` — nối lại với D10 để so |

Arm 3→4 là phép so **một biến** cho câu hỏi trung tâm: *ghi nhớ các điều kiện đã gặp có làm
việc quay lại rẻ hơn không?*

---

# Thứ tự và cửa chặn

```
T1 → T2                      2h   ⛔ CỬA CHẶN — cos < 0,3 thì DỪNG, đổi hướng
   ↓
R1 → R2 → R3 → R4            4h   stream có quay lại
   ↓
Đ (thước đo)                 1h   ⭐ làm TRƯỚC khi cài tầng, để biết đang tối ưu cái gì
   ↓
M1 → M4(một phần)            4h   tầng nhanh, rủi ro thấp
   ↓
E arm 1-3                    1 đêm  ⛔ CỬA CHẶN — tầng nhanh không ăn thì M2 vô nghĩa
   ↓
M2 → M3 → M4                 8h   tầng trung, rủi ro cao
   ↓
E arm 4-5                    1 đêm
```

Tổng: **~19 giờ code + 2 đêm máy.**

Hai cửa chặn có chủ đích: T1 chặn trước khi viết dòng nào, và E(arm 1–3) chặn trước khi bỏ 8
giờ vào phần rủi ro nhất. Nếu tầng nhanh — thứ đơn giản và chắc chắn nhất — mà không cải thiện
được gì, thì tầng trung phức tạp hơn cũng không cứu được.

---

# Cái gì có thể giết hướng này

Liệt kê trước để không tự lừa mình sau:

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Trôi không phải affine | T1 cho `cos < 0,3` | Dừng, quay lại Titans — nó biểu diễn được phi tuyến |
| Tỷ lệ lớp giả dạng điều kiện | T2 cho hình chiếu lớn | Chiếu lên trục điều kiện; không đủ thì cần nhãn giả |
| Ngân hàng chế độ nổ hoặc gộp sạch | số chế độ ≠ số điều kiện thật | Quét `nguong`; vẫn hỏng thì bỏ M2, giữ hai tầng |
| Lợi ích khi quay lại ≈ 0 | thước đo Đ ra 0 | Tầng trung vô dụng → bỏ, và đó vẫn là kết quả đáng viết |
| Feature ViT vỡ ở điều kiện cực đoan | accuracy sụp dù căn chỉnh đúng | Ngoài tầm SLDA — cần thích nghi backbone, tức Titans |

Hai dòng cuối đáng chú ý: **cả hai đều dẫn ngược về Titans**. Nghĩa là hướng ba tầng không loại
bỏ Titans — nó xác định chính xác *khi nào* mới cần tới Titans. Đó là một đóng góp sạch sẽ hơn
so với việc chỉ tuyên bố một bên thắng.

---

# Còn nợ từ trước, chưa xử

1. `0,6933` so với `0,7114` — hai biến thể NCM chênh 1,9 điểm, phải chốt trước khi viết báo cáo.
2. Ablation `post_norm` — η có phải cần gạt không (4h máy, 0 code). Cần nếu muốn viết phần Titans.
3. D11 chạy lại với `train.eval_ncm_head: true` — phép so Titans/SLDA hiện **không hợp lệ**.
