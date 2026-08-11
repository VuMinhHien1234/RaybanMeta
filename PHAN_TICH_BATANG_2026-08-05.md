# Phân tích kết quả ba tầng — đợt chạy 04→05/08/2026

Nguồn: `result_test/batang/` (gói giữa chừng lúc 15:34 ngày 05/08).
Máy `uavcl-slda` **vẫn đang chạy**: mới xong 3/9 run của track chính.

| Run | Xong lúc | Mất |
|---|---|---|
| U0 seed 0 | 19:55 04/08 | 6h17 |
| U1 seed 0 | 02:21 05/08 | 6h26 |
| U0 seed 1 | 08:30 05/08 | 6h09 |
| U1 seed 1 | đang chạy | — |

**~6h20/run** — không phải 1,1–1,4 h như ước lượng ban đầu (tôi đoán ~20 ảnh/s, thực tế
~6,6 ảnh/s). 9 run track chính = **57 giờ**, còn lại ~35 giờ nữa mới tới lượt U2.

---

## 1. Cái đã chắc chắn

### T1/T2 — trục điều kiện

```
cosine trung bình giữa các vector dịch : 0.7497      -> MOT_PHAN (ngưỡng TRUC_CHUNG là 0.8)
PC1 giải thích được                    : 77.4%
độ dài vector dịch                     : 36.415 ± 3.746
khoảng cách giữa các lớp (tham chiếu)  : 21.351
tỷ lệ nhiễu tỷ lệ-lớp / tín hiệu       : 0.041       -> T2 = CHIEU_LA_DU
```

### O4 — chi phí: ĐẠT thoải mái

0,0307 MB (trần 10) · 0,0688 ms/khung (trần 1,67). Không có gì phải lo về mặt triển khai biên.

### O1 — tầng nhanh có tác dụng, nhưng nhỏ

Đường chéo `R[t][t]` tách theo chế độ, seed 0 (so cặp cùng seed):

| chế độ | mức trôi | U0 đóng băng | U1 +tầng nhanh | Δ | thiệt hại do trôi | **M1 vá được** |
|---|---|---:|---:|---:|---:|---:|
| m0 | 0% | 0,7748 | 0,7759 | +0,0011 | — | — |
| m1 | 50% | 0,6759 | 0,6936 | +0,0177 | 0,0989 | **17,9%** |
| m2 | 100% | 0,3209 | 0,3733 | +0,0524 | 0,4539 | **11,5%** |
| **TB** | | **0,6119** | **0,6341** | **+0,0222** | | |

Mô phỏng `mo_phong_ba_tang.py` hứa 0,596 → 0,998 (+0,40). Thực tế **+0,022**, kém 18 lần.

---

## 2. Vì sao M1 chỉ vá được ~12%

T1 đã nói trước, chỉ là lúc đó chưa quy ra hệ quả. Với `cos = 0,7497`:

```
||V_c||           = 36,4     độ dịch của mỗi lớp khi trôi 0% -> 100%
||V̄||  ≈ 36,4·√0,75 = 31,5     phần CHUNG (T1 đo trực tiếp: 31,77 ✓)
phần DƯ ≈ 36,4·√0,25 = 18,2     phần RIÊNG của từng lớp
khoảng cách giữa các lớp        = 21,35
```

Tầng nhanh trừ đi được phần chung 31,5. Nhưng **phần dư riêng từng lớp còn 18,2, bằng 85%
khoảng cách giữa các lớp**. Sau khi căn chỉnh, các lớp vẫn bị đẩy lệch gần bằng khoảng cách
tới lớp hàng xóm — nên vẫn nhầm.

Mô phỏng đặt trôi = phép dịch toàn cục thuần tuý, tức `cos = 1,0`, phần dư = 0. Đó là lý do
nó cho +0,40 còn máy thật cho +0,02. **Đây không phải bug — đây là kết quả: giả định "điều
kiện = một phép tịnh tiến chung" chỉ đúng 75% trên feature ViT-S thật.**

---

## 3. ⛔ Vấn đề chí tử: U2 KHÔNG THỂ đo được chính nó

Đây là lý do tôi khuyên **dừng campaign trước khi tới U2**.

### 3a. Hệ hoàn toàn TĨNH ở pha 2

Ba lần gặp cùng một chế độ cho accuracy gần như y hệt:

```
U1 mode 2 (100% trôi) : 0,3760 · 0,3727 · 0,3711     σ ≈ 0,002
U1 mode 0 (0%  trôi)  : 0,7748 · 0,7776 · 0,7754
```

Pha 2 không nhãn nên μ_c/Σ đóng băng hoàn toàn; thứ duy nhất động là `m_t`. Mà `m_t` hội tụ
về cùng một điểm mỗi lần gặp lại cùng điều kiện. Nên `loi_ich_quay_lai` = +0,002 ≈ 0 —
**đúng như thiết kế**, U1 không có bộ nhớ điều kiện.

Suy ra: O3 chỉ có thể đến từ **tốc độ hội tụ**, đo ở vài batch đầu chuyến. Đó chính là lý do
`o3_preq10` tồn tại. Nhưng…

### 3b. Cửa sổ đo và cơ chế lệch pha nhau

`tang_nhanh.decay = 0,99`, batch 32 → `r = 0,99³² = 0,725`, hằng số thời gian **3,6 batch**:

| batch | 1 | 2 | 3 | 4 | **5** | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| m_t đã tự hội tụ | 27% | 47% | 62% | 72% | **80%** | 86% | 92% | **96%** |

`ngan_hang.cho_khop_sau = 5` → ngân hàng chỉ tra **sau batch 5**, lúc M1 đã tự đi được 80%
quãng đường. Đến batch 10 thì M1 tự làm xong 96%. **Ngân hàng không còn gì để cứu.**

### 3c. Trần lý thuyết nhỏ hơn nhiễu 5 lần

Lấy khoảng cách acc giữa "căn đúng" và "không căn" ở mức trôi nặng nhất (0,3733 − 0,3209 =
0,0524) làm trần, giả sử ngân hàng khớp HOÀN HẢO ngay khi được phép:

```
TRẦN o3_preq10 = Σ(k=6..10) r^k × 0,0524 / 10 = +0,0022
```

Trong khi nhiễu quan sát được của chính chỉ số đó là **±0,008 … 0,012**. Tín hiệu nhỏ hơn
nhiễu **5 lần**.

### 3d. Bằng chứng thực nghiệm rằng preq10 đang đo nhiễu

| run | o3_preq10 |
|---|---:|
| U0 seed 0 — **không thích nghi gì cả** | **+0,0117** |
| U0 seed 1 — không thích nghi gì cả | +0,0078 |
| U1 seed 0 — có tầng nhanh | **−0,0074** |

Arm đóng băng hoàn toàn lại có "lợi ích quay lại" DƯƠNG và LỚN HƠN arm có thích nghi. Không
có cơ chế nào giải thích được điều đó ngoài nhiễu. Chỉ số đang ở dưới ngưỡng phân giải.

**Kết luận: chạy U2 thêm 19 giờ sẽ ra một con số không phân biệt được với 0 — và tệ hơn, rất
có thể là số ÂM, rồi bị đọc nhầm thành "ngân hàng chế độ không hoạt động" trong khi sự thật
là "phép đo không nhìn thấy nó".**

---

## 4. Khuyến nghị

### Dừng campaign hiện tại

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'pkill -f chay_dem_ba_tang; pkill -f run_g1.py'
```

Tiết kiệm ~35 giờ máy. U0/U1 seed 0 đã đủ để rút ra mọi kết luận ở mục 1–3; seed 1/2 chỉ
thêm σ cho một kết luận đã rõ.

### Ba sửa TRƯỚC khi chạy lại

| # | Sửa | Vì sao |
|---|---|---|
| 1 | `ngan_hang.cho_khop_sau: 5 → 1` | tra ngân hàng ngay batch 1, đúng lúc M1 còn lệch 72% |
| 2 | `tang_nhanh.decay: 0,99 → 0,999` | τ = 32 batch thay vì 3,6 → sau 10 batch M1 mới tự đi 27%, **có transient thật để mà cứu** |
| 3 | `tang_nhanh.kieu: day_du → truc` | T2 đo được: chiếu lên trục lọc bỏ 80,7% nhiễu tỷ lệ lớp (6,79 → 1,31). T1 = MOT_PHAN chỉ vì cos 0,7497 thiếu 0,05 so với ngưỡng 0,8 — quy tắc trong driver quá cứng |

Trần o3_preq10 sau khi sửa:

| cấu hình | τ | M1 tự hội tụ sau 10 batch | trần preq10 | so với nhiễu |
|---|---|---|---|---|
| hiện tại (0,99 / 5) | 3,6 | 96% | +0,0022 | 0,2× ❌ |
| 0,99 / 1 | 3,6 | 96% | +0,0095 | 0,9× ❌ |
| 0,995 / 1 | 6,7 | 80% | +0,0196 | 2,0× ⚠️ |
| **0,999 / 1** | **31,7** | **27%** | **+0,0391** | **3,9× ✅** |

Mỗi chuyến có ~197 batch nên `decay = 0,999` (cửa sổ ~1000 mẫu) vẫn hội tụ thừa trong chuyến.

### Chạy lại rẻ hơn nhiều

Không cần 12 chuyến để trả lời câu hỏi O3 — **5 chuyến là đủ** (`c0..c4`, có 1 lần gặp lại
+ 1 chế độ mới, đúng như smoke). 3 arm × 3 seed × 5 chuyến ≈ 2,6 h/run → **~24 giờ**, chia
3 máy khác region thì **~8 giờ**.

Và chạy U1/U2 trước, U0 sau — U0 chỉ là mốc, đã có số rồi.

---

## 5. Còn dùng được gì từ đợt này

- **O4 ĐẠT** — số chính thức, khỏi chạy lại: 0,0307 MB · 0,0688 ms/khung.
- **T1/T2** — `artifacts_t1/t1_t2.json`, tái sử dụng được cho mọi lần chạy sau.
- **Bảng O1 per-mode** (mục 1) — đây là kết quả thật, vào thẳng báo cáo.
- **Phát hiện `cos = 0,75` ⇒ phần dư 18,2 ≈ 85% khoảng cách lớp** — theo tôi đây mới là
  đóng góp khoa học đáng kể nhất của đợt chạy: nó định lượng chính xác **vì sao căn chỉnh
  tuyến tính không nhãn là không đủ**, và biện minh cho việc cần bộ nhớ phi tuyến. Đúng cái
  mà nhánh Titans đang cố trả lời.

## 6. Hạn chế của chính phân tích này

- U1 mới có **1 seed**. So cặp trong-seed rất chặt (σ giữa 3 lần lặp cùng chế độ ≈ 0,002),
  nhưng chênh lệch giữa seed thì lớn (U0: 0,6119 vs 0,5773) — nên con số +0,022 nên đọc là
  "cùng seed 0", chưa phải trung bình quần thể.
- Trần preq10 ở mục 3c giả định acc biến thiên tuyến tính theo mức căn chỉnh. Thô. Nhưng kể
  cả nới gấp 3 lần thì +0,0066 vẫn dưới nhiễu — kết luận không đổi.
- Chưa kiểm giả thuyết "`kieu: truc` sẽ tốt hơn". T2 ủng hộ mạnh, nhưng `truc` chỉ sửa thành
  phần trên trục nên O1 có thể **thấp hơn** `day_du`. Đáng chạy 1 run 5 chuyến (~2,6 h) để
  biết trước khi cam kết cả campaign.
