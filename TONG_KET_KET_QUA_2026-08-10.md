# Tổng kết toàn bộ kết quả đã chạy

Cập nhật 2026-08-10 · Tổng hợp từ `result_test/` · Backbone chung: **ViT-S/16 pretrained, đóng băng** (trừ HOPE)

---

# 0. Bốn chế độ thí nghiệm — đọc bảng nào cho việc gì

Dự án đã chạy trên **bốn chế độ khác nhau**. Số ở chế độ này **không so được** với chế độ kia.

| Chế độ | Task mới nghĩa là | Nhãn | Số task | Dùng để |
|---|---|---|---:|---|
| **A. class-incremental** | lớp mới | mọi task | 9 | so baseline chuẩn ngành |
| **B. domain-incremental** | điều kiện mới, **trôi đơn điệu** | mọi task | 9 | quét λ |
| **C. revisit** ⭐ | điều kiện mới, **có QUAY LẠI** | **chỉ chuyến 0** | 12 | **bài toán UAV thật** |
| D. smoke/bench | — | — | 2–3 | kiểm cơ chế, đo chi phí |

**Chỉ chế độ C là bài toán đích.** A và B là đường dẫn tới nó.

---

# 1. ⭐ BẢNG CHÍNH — Chế độ C: bài toán UAV thật

12 chuyến bay · 3 điều kiện tuần hoàn (chu kỳ 4) · 9/12 chuyến lặp lại · nhãn **chỉ ở chuyến 0** · chấm trên toàn bộ 6.300 ảnh test · **3 seed**

| | O1 (acc điều kiện đang bay) | O3 (lợi ích quay lại) | Bộ nhớ | Độ trễ | O4 |
|---|---:|---:|---:|---:|:---:|
| **U0** đóng băng hoàn toàn | 0.5962 ± 0.0175 | −0.0009 ± 0.0186 | 1,50 MB | 0,007 ms | ✅ |
| **U1** + tầng nhanh (M1) | **0.6282 ± 0.0066** | −0.0047 ± 0.0026 | 1,51 MB | 0,054 ms | ✅ |
| **U2** + ngân hàng chế độ (M2) | 0.6282 ± 0.0066 | −0.0047 ± 0.0026 | 1,53 MB | 0,069 ms | ✅ |
| **U3** HOPE (NL đầy đủ) | 0.5753 ± 0.0152 | −0.0089 ± 0.0062 | **102,75 MB** | **2,178 ms** | ❌ |

## Ba phép so ghép cặp theo seed

```
U1 − U0   O1 = +0.0319 ± 0.0109   t = +5.07   p < 0.05   ✅  tầng nhanh CÓ tác dụng
U2 − U1   O1 = −0.0000 ± 0.0001   t = −0.30   n.s.       ❌  ngân hàng VÔ DỤNG
U3 − U1   O1 = −0.0529 ± 0.0175   t = −5.23   p < 0.05   ❌  HOPE THUA
```

Từng seed:
```
U1 − U0 :  +0.0222  +0.0437  +0.0299    3/3 dương
U3 − U1 :  −0.0502  −0.0368  −0.0715    3/3 âm
```

## Đọc theo mức trôi — chỗ lộ ra nhiều nhất

Trung bình đường chéo, gộp 3 seed:

| điều kiện | mức trôi | U0 | U1 | U3 HOPE | U1 − U3 |
|---|---:|---:|---:|---:|---:|
| m0 | 0% | 0.7700 | 0.7699 | 0.7436 | **0.0263** |
| m1 | 50% | 0.6622 | 0.6831 | 0.6355 | 0.0476 |
| m2 | **100%** | **0.2905** | **0.3767** | **0.2867** | **0.0900** |

Ba điều đọc được:

1. **Trôi 100% làm sụp 45 điểm** (0.770 → 0.291). Đây là điểm yếu lớn nhất của cả hệ.
2. **Tầng nhanh cứu +8,6 điểm ở m2** (0.291 → 0.377) — hiệu quả nhất đúng chỗ khó nhất.
3. **HOPE thua ngay ở trôi 0%** (−2,63đ) — nơi ký ức **không liên quan**. Đó là chênh **bộ đọc**, không phải kiến trúc. Nửa còn lại (−2,66đ) mới là kiến trúc.

---

# 2. Chế độ A — class-incremental RESISC45 (9 task, có nhãn mọi task)

| Method | Acc | Forgetting | Bộ nhớ thêm | Ghi chú |
|---|---:|---:|---:|---|
| finetune | 0.4144 | 0.6133 | 0 | mốc dưới |
| ewc | 0.4259 | 0.5999 | 390M float ⚠️ | +0,01 so finetune, tốn 3h |
| lwf | 0.5900 | 0.2777 | 21,7M | |
| **ncm** | 0.6924 ± 0.0010 | 0.0979 | **17K** | ⭐ mốc "không được thua" |
| replay | 0.7937 | 0.0787 | 135M ⚠️ | mạnh nhất G1, lưu ảnh thô |
| latent_replay | 0.6596 ± 0.0075 | **−0.0296** | 0,3M | rẻ hơn replay 400× |
| **slda** | **0.8266 ± 0.0017** | 0.0774 | 1,39 MB | ⭐ **cao nhất toàn dự án** |
| titans (G2) | 0.4247 ± 0.1985 | 0.3798 | 102,68 MB | σ rất lớn — có seed nổ |
| hope (G4, cấu hình n1) | 0.2688 ± 0.0472 | −0.0004 | 102,75 MB | cấu hình `post_norm=false` |

## Ablation `cov_mode` — Σ đóng góp bao nhiêu

| | Acc | So với |
|---|---:|---|
| SLDA `identity` (ép Σ = I ≡ NCM) | 0.6885 ± 0.0010 | — |
| SLDA `frozen` (Σ đóng băng sau task 1) | 0.7682 ± 0.0077 | +8,0đ |
| **SLDA `streaming`** | **0.8266 ± 0.0017** | **+13,8đ** |
| NCM (đối chứng độc lập) | 0.6924 ± 0.0010 | ≈ identity ✓ |

> **Ma trận hiệp phương sai đóng góp +13,8 điểm.** Và `identity ≈ NCM` xác nhận phép so là sạch.

---

# 3. Chế độ B — domain drift, quét λ (9 task, có nhãn)

| Arm | λ_μ | λ_Σ | Acc |
|---|---:|---:|---:|
| **arm1** | 1.0 | 1.0 | **0.7817 ± 0.0049** |
| arm2 | 0.999 | 0.9999 | 0.7815 ± 0.0030 |
| arm3 | 0.99 | 0.9999 | 0.7692 ± 0.0039 |
| arm4 | 0.97 | 0.9999 | 0.7033 ± 0.0023 |
| arm5 | 0.99 | **0.99** | 0.5029 ± 0.0180 |

Đọc thẳng: **λ=1 thắng, càng quên càng tệ.** Nhưng `KET_QUA_D10_D11` chỉ ra đó là **kết luận sai vì thước đo sai** — nhìn theo mức trôi thì λ=0,99 hơn λ=1 **+3,91 điểm ở trôi 100%** (3,10σ), bị `AAA` che mất.

⚠️ **Quan trọng:** quét này chạy trên stream **trôi đơn điệu, không quay lại**. `KET_QUA_D10_D11 §10` cảnh báo con số +3,91 *"không chuyển sang được và có thể đảo dấu"* trên stream có quay lại. **Chưa quét λ trên chế độ C.**

## Titans trên stream trôi (D11)

| | Acc | Đường chéo |
|---|---:|---:|
| `drift_titans_gates` (η ∈ [0,1 · 0,9]) | 0.4937 ± 0.0077 | 0.2759 |
| `drift_titans_eta_thap` (η ∈ [0,012 · 0,5]) | 0.4937 ± 0.0074 | 0.2765 |
| SLDA arm3 (cùng stream) | 0.7692 | 0.7759 |

**Đổi trần η gấp 100 lần → Δacc = 0,0000157.** Ba seed, một biến. **η không phải cần gạt.**

---

# 4. Bộ đọc — phát hiện xuyên suốt

Trên **cùng feature**, chỉ đổi cách đọc:

| Cấu hình | head Linear (triển khai được) | NCM-head (⚠️ **oracle**) |
|---|---:|---:|
| Titans + head cosine | 0.6360 ± 0.0424 | 0.7375 |
| Titans + head linear (SDC) | 0.4804 ± 0.1454 | 0.5395 |
| HOPE n1 (post_norm=false) | 0.2688 ± 0.0472 | 0.6874 |

⚠️ NCM-head chạy chế độ `rebuild` — **quét lại toàn bộ data cũ**, không triển khai được. Chỉ dùng chẩn đoán.

Hai điều:
1. **Đổi head Linear → cosine: +15,6 điểm** và dập được seed nổ (`norm(state)` 1,75e6 → 54,5).
2. Ngay cả bộ đọc **oracle** của HOPE (0.6874) vẫn **thua NCM trần** (0.6924) — làm gì đó phức tạp thua không làm gì.

---

# 5. Chi phí triển khai (B6, đo thật)

| Phương pháp | Bộ nhớ | Suy luận | O4 (10 MB · 1,67 ms) |
|---|---:|---:|:---:|
| SLDA streaming (fp32) | **0,73 MB** | 0,0071 ms | ✅ |
| SLDA (fp64) | 1,39 MB | — | ✅ |
| Ba tầng (M1+M2 thêm) | +0,031 MB | +0,069 ms | ✅ **dư 325×** |
| **Titans self-mod depth 3** | **102,68 MB** | **2,178 ms** | ❌ **vượt 10,3×** |
| ↳ ViT-L (D=1024) | 729,95 MB | 28,37 ms | ❌ 85% khung hình |

Tách: tham số 31,90 MB + **state 70,79 MB**. State lớn gấp **2,2 lần** tham số — đó mới là phần drone phải mang.

Xác nhận **O(D²)**: D ×2,67 → bộ nhớ ×7,1 ✓

---

# 6. Chẩn đoán cơ chế — vì sao HOPE thua

Đo từ `run_U3_hope_s0.log`:

## α bò lên trần → ký ức không tích luỹ

```
α qua 12 chuyến:  0.843 → 0.866 → 0.879 → 0.887 → 0.895 → 0.900
                → 0.904 → 0.907 → 0.910 → 0.912 → 0.914 → 0.915
      (% khoảng):  88% ─────────────────────────────────────→ 96%
⚠️ 11 cảnh báo BÃO HOÀ

norm(state):  54.36  54.97  54.42  54.07  53.87  53.87
              55.06  54.77  54.62  55.24  54.88  54.74     ← PHẲNG
```

α = 0.915 → giữ lại **8,5%/chunk** → sau 3 chunk (96 ảnh ≈ 3 giây bay) còn **0,06%**.

Cổng η thì **khoẻ**: 0.571 → 0.587, ổn định 59–61% khoảng, không bão hoà. **1/2 cổng sống.**

## Khoảng cổng dẫn cho bài toán KHÁC

`α ∈ [0.05, 0.95]` dẫn từ class-incremental. Với 1 chuyến = 49 chunk:

| α | giữ/chunk | còn sau 1 chuyến | còn sau 1 chu kỳ |
|---:|---:|---:|---:|
| 0.9151 (đo được) | 0.085 | 3×10⁻⁵³ | 10⁻²¹⁰ |
| 0.05 (**sàn cho phép**) | 0.95 | **8,1%** | 0,004% |
| 0.0035 (bán rã = 1 chu kỳ) | 0.9965 | 84% | 50% |

> **Ngay ở sàn cho phép, chỉ 8% ký ức sống qua một chuyến.** Cấu hình cổng **về mặt cấu trúc** không cho phép nhớ xuyên chuyến.

## CMS chỉ tác động ở chuyến 0

`‖Δw‖` không đổi từ chuyến 1 — vì pha 2 không có gradient. CMS thiết kế cho **dòng dài** bước gradient; bài toán chỉ có **một** đợt. Không hỏng, **không có việc làm**.

Ở chuyến 0 nó chạy đúng: `fast 4.077 (2,38%) > mid 0.428 (0,28%) > slow 0.057 (0,018%)` — chênh 71×.

---

# 7. Ba kết luận có bằng chứng

**① Căn chỉnh điều kiện không nhãn có giá trị đo được.**
`U1 − U0 = +0.0319 ± 0.0109`, t=5,07, p<0,05, **3/3 seed dương**. Chi phí: **6 KB · 0,047 ms**.

**② Ghi nhớ điều kiện KHÔNG đáng giá khi ước lượng lại rẻ.**
`U2 − U1 = 0.0000`. Cơ chế: λ=0,99 batch 32 → `r = 0.725/batch` → tầng nhanh hội tụ trong **~4 batch (4 giây bay)**. Snapshot lưu trữ (trung bình EMA nhiều lần gặp) **mờ hơn** ước lượng tươi. Ứng viên nạp-sớm còn **hại 1,2 điểm**.
Ngân hàng chạy đúng cơ chế: 3–4 chế độ (thật = 3), nạp lại 7–8/9 chuyến lặp.

**③ Hiện thực hoá NL bằng gradient không sống được ở biên.**
Chi phí **102,75 MB** vs trần 10 MB (**10,3×**). O1 thua **5,3 điểm** (t=−5,23, p<0,05). Ba nguyên nhân định lượng ở §6.
⚠️ **Nhưng một nửa khoảng cách là bộ đọc, không phải kiến trúc** — quy kết toàn bộ cho NL là sai.

---

# 8. Cái CHƯA chạy trên bài toán đích (chế độ C)

| | Cần gì | Chi phí |
|---|---|---|
| Quét **λ** trên revisit | chỉ config | ~5h |
| **Oracle có nhãn** (`revisit_arm1_lam1.yaml` — config có sẵn) | chỉ config | ~3h |
| **Pha 1 dài hơn** (`chuyen_hieu_chinh > 1`) — kiểm Σ có đói dữ liệu không | chỉ config | ~2h |
| Sửa **khoảng cổng α** cho HOPE rồi chạy lại | chỉ config | ~4h |
| **NCM** trên revisit — kiểm Σ có phản tác dụng khi trôi nặng không | chỉ config | ~2h |

**Cả năm đều là config, không cần code.** Và cả năm đều nhắm vào các điểm yếu đã lộ ra ở §1 và §6.

---

# 9. Bảng tóm — một trang

```
BÀI TOÁN UAV (revisit, 12 chuyến, pha 2 không nhãn, 3 seed)

                        O1          Δ ghép cặp        MB       ms     O4
  U0 đóng băng       0.5962        —                1.50    0.007    ✅
  U1 +tầng nhanh     0.6282        +0.0319 ***      1.51    0.054    ✅   ⭐ tốt nhất
  U2 +ngân hàng      0.6282        +0.0000          1.53    0.069    ✅
  U3 HOPE (NL)       0.5753        −0.0529 ***    102.75    2.178    ❌

  *** p < 0.05, 3 seed ghép cặp

THAM CHIẾU (chế độ khác, KHÔNG so trực tiếp được)
  SLDA class-incremental        0.8266     ← cao nhất dự án
  replay class-incremental      0.7937     ← lưu ảnh thô
  NCM class-incremental         0.6924     ← mốc "không được thua"
  Titans/HOPE class-inc         0.27–0.64  ← luôn thua NCM
```

---

*Nguồn: `result_test/{batang,u3_hope,final,drift,d11,res_cos,res_sdc,res_base}`. Mọi con số ± là độ lệch chuẩn qua seed.*
