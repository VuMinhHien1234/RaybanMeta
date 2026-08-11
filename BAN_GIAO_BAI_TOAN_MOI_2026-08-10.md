# Bàn giao — Bài toán UAV mới, kết quả đã có, và cách chạy phương án của bạn

Ngày 2026-08-10 · Nhánh code: **`feat/cms-ba-tang`** · Repo: `github.com/VuMinhHien1234/RaybanMeta`

> **Mục đích tài liệu:** để bạn (a) hiểu **bài toán mới** khác bài toán cũ ở đâu, (b) biết **mốc phải vượt**, (c) **chạy phương án của mình** trên đúng bài toán và so sánh công bằng.
>
> Đọc tối thiểu: **Phần A** (bài toán) → **Phần D** (mốc) → **Phần F** (cách chạy).

---

# PHẦN A — BÀI TOÁN MỚI

## A1. Kịch bản

> Một UAV tuần tra **một khu vực cố định**.
>
> **Lần đầu** — bay qua, dữ liệu được **gán nhãn** thủ công. Đây là *pha hiệu chỉnh* dưới đất, làm **một lần**, không giới hạn thời gian và tính toán.
>
> **Những lần sau** — bay lại **cùng khu vực đó**, nhưng điều kiện quan sát đã khác: giờ trong ngày, mùa, mây, sương, bụi bám ống kính, độ cao bay. **Không có nhãn. Không có người.**

## A2. Ba việc phải làm — ba tần số lệch nhau ~10⁴ lần

| | Việc | Đổi khi nào |
|---|---|---|
| a | Nhận đúng *"đây là sân bay, kia là ruộng"* | gần như không bao giờ |
| b | Nhận ra *"điều kiện hôm nay giống chuyến tháng trước"* | mỗi chuyến bay |
| c | Bám theo *"nắng đang gắt dần suốt 10 phút"* | mỗi khung hình |

Đây là **cấu trúc của bài toán**, không phải lựa chọn thiết kế.

## A3. Phát biểu hình thức

**Pha 1 — hiệu chỉnh, CÓ nhãn, ngoại tuyến**
Cho `D₀ = {(xᵢ, yᵢ)}`, `yᵢ ∈ {1..C}`, thu dưới điều kiện `e₀`. Không giới hạn tính toán. Chạy một lần.

**Pha 2 — triển khai, KHÔNG nhãn, trực tuyến**
Nhận dòng `x₁, x₂, …` dưới điều kiện `e₁, e₂, …` trong đó:
- `e_t` **biến thiên trơn** — mặt trời lặn dần, sương dâng dần
- `e_t` **có thể quay lại** giá trị đã gặp — mùa và giờ đều tuần hoàn
- `e_t` **không quan sát được** — hệ không được cho biết *"hôm nay là mùa đông"*
- Tập lớp **không đổi**: `y ∈ {1..C}`

Với mỗi `x_t` phải trả `ŷ_t` **ngay**, trước khi thấy `x_{t+1}`.

## A4. Stream cụ thể trong code

```
RESISC45 — 31.500 ảnh, 45 lớp · ViT-S/16 pretrained, ĐÓNG BĂNG
12 chuyến bay × ~1.580 ảnh train · chấm trên TOÀN BỘ 6.300 ảnh test

LỊCH BAY (in ra ở đầu mỗi run):
  c0 : 0%/m0       c4 : 0%/m0(lần 2)     c8 : 0%/m0(lần 3)
  c1 : 50%/m1      c5 : 50%/m1(lần 3)    c9 : 50%/m1(lần 5)
  c2 : 100%/m2     c6 : 100%/m2(lần 2)   c10: 100%/m2(lần 3)
  c3 : 50%/m1(2)   c7 : 50%/m1(lần 4)    c11: 50%/m1(lần 6)

  3 điều kiện thật (m0/m1/m2) · chu kỳ 4 chuyến · 9/12 chuyến LẶP LẠI
```

**Trôi** = 5 phép biến đổi vật lý, cường độ tăng theo `severity`:

| Phép | Mô phỏng | 0% → 100% |
|---|---|---|
| độ sáng | giờ trong ngày, mùa | ×1,00 → ×0,55 |
| nhiệt độ màu | bình minh ấm → trưa lạnh | 0 → ±18% lệch R/B |
| tương phản | sương mù, khói, bụi | ×1,00 → ×0,65 |
| mờ Gauss | độ cao bay, ống kính bẩn | σ 0 → 1,4 px |
| nhiễu | ISO cao khi thiếu sáng | σ 0 → 0,035 |

Thêm: điều kiện **đổi dần ±12,5% ngay trong một chuyến** (`trong_chuyen.bien_do: 0.25`).

## A5. ⚖️ LUẬT — được gì, cấm gì

| | ✅ Được | ❌ Cấm |
|---|---|---|
| **Chuyến 0** (pha 1) | nhãn thật · gradient · train bao lâu tuỳ | — |
| **Chuyến 1–11** (pha 2) | cập nhật thống kê / ký ức **không nhãn** · trả lời tức thì | nhãn mới · gradient · lưu ảnh thô · đọc lại dữ liệu cũ |
| **Phần cứng** | — | > **10 MB** bộ nhớ thêm · > **1,67 ms**/khung (5% của 33,3 ms @30fps) |

`y` ở pha 2 **chỉ dùng để chấm điểm**, không được chạm vào update. Test `count_raw` đứng yên kiểm đúng điều này — **giữ nguyên test đó khi thêm method mới**.

## A6. Thước đo

| | Mục tiêu | Công thức | Đọc ở đâu |
|---|---|---|---|
| **O1** | Giữ danh tính | trung bình **đường chéo** `R[i][i]` | `metrics_revisit.json` → `acc_dieu_kien_hien_tai` |
| **O2** | Bám trôi không nhãn | số batch hồi về 95% mức ổn định | → `o2_hoi_phuc_tb_batch` |
| **O3** ⭐ | **Ghi nhớ điều kiện** | `Acc(gặp lại X) − Acc(lần đầu X)` | → `o3_preq10_loi_ich` |
| **O4** | Chạy trên drone | MB · ms | `scripts/bench_cost.py` |

⚠️ **KHÔNG dùng** `Average Accuracy`, `Forgetting`, `AAA`, `BWT`. Chúng trung bình trên **mọi chuyến đã bay** — tức thưởng cho việc nhớ lịch sử. *Drone thì bay bây giờ.* Bật `train.eval_chi_duong_cheo: true` thì `run_g1.py` ghi `null` cho chúng.

---

# PHẦN B — KHÁC GÌ BÀI TOÁN CŨ

## B1. Bảng so sánh

| | **Bài toán CŨ** (t7 → 03/08) | **Bài toán MỚI** (04/08 →) |
|---|---|---|
| Kịch bản | UAV gặp **địa hình mới** dần | UAV **bay lại cùng chỗ**, điều kiện đổi |
| Cái gì đổi giữa các task | **tập lớp** | **điều kiện quan sát** |
| Tập lớp | 5 → 10 → … → 45 | **45, cố định** |
| Số task / chuyến | 9 | **12** |
| Nhãn | **mọi task** | **chỉ chuyến 0** |
| Điều kiện cũ quay lại | không có khái niệm | ✅ **9/12 chuyến** |
| Trôi trong một task | không | ✅ ±12,5% |
| Vấn đề trung tâm | quên **lớp** | **trôi điều kiện** + **ghi nhớ điều kiện** |
| Ràng buộc phần cứng | không nêu | ✅ **≤10 MB · ≤1,67 ms** |
| Thước đo | Acc · Forgetting · BWT · AAA | **O1 · O2 · O3 · O4** |
| Nghĩa `R[i][j]` | acc **lớp** nhóm j sau khi học i | acc ở **điều kiện** j sau khi qua i |
| "Forgetting" nghĩa là | mất khả năng nhận **lớp cũ** | mất khả năng hoạt động ở **điều kiện cũ** |
| Config | `g1_*.yaml`, `g2_*`, `g3_*`, `g4_*` | **`revisit_*.yaml`** |

## B2. Vì sao đổi — ba lý do, đều từ dữ liệu

**① Kịch bản cũ không tồn tại với UAV tuần tra.**
Drone bay một khu vực cố định. Khu vực đó không tự mọc thêm loại địa hình mới. Cái thật sự đổi là nắng, mùa, sương.

**② Thước đo cũ che mất kết quả đúng.**
D10 quét λ: theo `AAA` thì *"λ=1 thắng, càng quên càng tệ"*. Nhưng nhìn theo mức trôi, λ=0,99 hơn λ=1 **+3,91 điểm ở trôi 100%** (3,10σ). `AAA` trung bình trên mọi task đã thấy — sau task 8 thì **8/9 bộ test là điều kiện quá khứ**.

**③ Stream trôi đơn điệu làm "quên" thành miễn phí.**
`data/revisit.py` ghi rõ:
> *"Stream trôi của D10 tăng **đơn điệu** 0% → 100%, điều kiện cũ **không bao giờ quay lại**. Ở đó quên điều kiện cũ gần như **miễn phí**… Nhưng bài toán thật thì drone **bay lại cùng chỗ**, và mùa/giờ đều **tuần hoàn**. Con số +3,91 **không chuyển sang được và có thể đảo dấu**."*

## B3. ⚠️ Cảnh báo bắt buộc

**Số của hai bài toán KHÔNG so trực tiếp được.**

```
SLDA ở bài toán CŨ  :  0.8266    (nhãn 9 task, không trôi)
SLDA ở bài toán MỚI :  0.5962    (nhãn 1 chuyến, trôi tới 100%)
```

Không phải SLDA tệ đi — bài toán khó hơn hẳn. Đừng đặt hai cột này cạnh nhau trong báo cáo mà không ghi rõ chế độ.

---

# PHẦN C — KẾT QUẢ TRÊN BÀI TOÁN CŨ

*Chế độ class-incremental, 9 task × 5 lớp, có nhãn mọi task, không trôi.*

## C1. Năm baseline G1

**RESISC45** (ViT-S, 9 task)

| Method | Acc ↑ | Forgetting ↓ | Bộ nhớ thêm | Thời gian |
|---|---:|---:|---:|---:|
| finetune | 0.4144 | 0.6133 | 0 | 2.187 s |
| ewc | 0.4259 | 0.5999 | 390M float (~1,5 GB) ⚠️ | 10.681 s ⚠️ |
| lwf | 0.5900 | 0.2777 | 21,7M | 2.249 s |
| **ncm** | **0.6933** | 0.1000 | **17K** | 837 s |
| **replay** | **0.7937** | **0.0787** | 135M ⚠️ (lưu ảnh thô) | 3.164 s |

**EuroSAT** (bộ nhẹ, 10 lớp)

| Method | Acc ↑ | Forgetting ↓ | BWT |
|---|---:|---:|---:|
| finetune | 0.4213 | 0.2566 | −0.2566 |
| ewc | 0.4371 | 0.1902 | −0.1902 |
| lwf | 0.3739 | **0.6089** ⚠️ | −0.6089 |
| **ncm** | **0.7486** | 0.1113 | −0.1113 |
| replay | 0.6191 | **0.0027** | **+0.2253** |

> **Kết luận G1:** NCM rẻ hơn replay **8.000 lần** mà chỉ kém 10 điểm. Mọi phương pháp phức tạp phải **thắng NCM** mới có ý nghĩa.

## C2. SLDA và ablation `cov_mode` (3 seed)

| | Acc | Ghi chú |
|---|---:|---|
| SLDA `identity` (ép Σ = I ≡ NCM) | 0.6885 ± 0.0010 | đối chứng |
| SLDA `frozen` (Σ đóng băng sau task 1) | 0.7682 ± 0.0077 | +8,0đ |
| **SLDA `streaming`** | **0.8266 ± 0.0017** | **+13,8đ · cao nhất toàn dự án** |
| NCM (đối chứng độc lập) | 0.6924 ± 0.0010 | ≈ identity ✓ |

**Ma trận hiệp phương sai đóng góp +13,8 điểm.** `identity ≈ NCM` xác nhận phép so sạch.

## C3. Titans / CMS / HOPE (G2–G4)

| Cấu hình | Linear (dùng được) | NCM-head (⚠️ oracle) | n |
|---|---:|---:|---:|
| Titans + M3 (seed 0) | 0.6079 | 0.7655 | 1 |
| Titans + AdamW (seed 0/1) | 0.5908 / 0.5630 | 0.7497 / 0.7371 | 2 |
| **Titans + head cosine** | **0.6360 ± 0.0424** | 0.7375 ± 0.0051 | 3 |
| Titans + linear + SDC | 0.4804 ± 0.1454 | 0.5395 ± 0.2674 | 3 |
| Latent replay | 0.6596 ± 0.0075 | 0.7189 ± 0.0039 | 3 |
| HOPE (`post_norm=off`) | 0.2688 ± 0.0472 | 0.6874 ± 0.0028 | 3 |
| Titans gộp mọi biến thể | 0.4247 ± 0.1985 | — | 24 |

**Không cấu hình nào thắng NCM (0.6924) bằng bộ đọc dùng được.**

## C4. Domain drift — quét λ (9 task, có nhãn, trôi **đơn điệu**)

| Arm | λ_μ | λ_Σ | Acc |
|---|---:|---:|---:|
| arm1 | 1.0 | 1.0 | **0.7817 ± 0.0049** |
| arm2 | 0.999 | 0.9999 | 0.7815 ± 0.0030 |
| arm3 | 0.99 | 0.9999 | 0.7692 ± 0.0039 |
| arm4 | 0.97 | 0.9999 | 0.7033 ± 0.0023 |
| arm5 | 0.99 | **0.99** | 0.5029 ± 0.0180 |

⚠️ **Chưa quét λ trên bài toán mới.** D10 cảnh báo kết quả này *"có thể đảo dấu"* khi điều kiện quay lại.

**Titans trên cùng stream (D11):** `gates` 0.4937 ± 0.0077 · `eta_thap` 0.4937 ± 0.0074.
Đổi trần η **gấp 100 lần → Δacc = 0,0000157**. Ba seed. **η không phải cần gạt.**

---

# PHẦN D — KẾT QUẢ TRÊN BÀI TOÁN MỚI ⭐ MỐC PHẢI VƯỢT

*Chế độ revisit · 12 chuyến · nhãn chỉ ở chuyến 0 · 3 seed · `test_chung=true`*

## D1. Bảng chính

| | **O1** (acc điều kiện đang bay) | **O3** (lợi ích quay lại) | Bộ nhớ | Độ trễ | O4 |
|---|---:|---:|---:|---:|:---:|
| **U0** đóng băng hoàn toàn | 0.5962 ± 0.0175 | −0.0009 ± 0.0186 | 1,50 MB | 0,007 ms | ✅ |
| **U1** + tầng nhanh (M1) | **0.6282 ± 0.0066** ⭐ | −0.0047 ± 0.0026 | 1,51 MB | 0,054 ms | ✅ |
| **U2** + ngân hàng chế độ (M2) | 0.6282 ± 0.0066 | −0.0047 ± 0.0026 | 1,53 MB | 0,069 ms | ✅ |
| **U3** HOPE (Titans+CMS+M3) | 0.5753 ± 0.0152 | −0.0089 ± 0.0062 | **102,75 MB** | **2,178 ms** | ❌ |

## D2. Phép so ghép cặp theo seed

```
U1 − U0   O1 = +0.0319 ± 0.0109   t = +5.07   p < 0.05   ✅  3/3 seed dương
U2 − U1   O1 = −0.0000 ± 0.0001   t = −0.30   n.s.       ❌  ngân hàng VÔ DỤNG
U3 − U1   O1 = −0.0529 ± 0.0175   t = −5.23   p < 0.05   ❌  3/3 seed âm
```

## D3. Bóc theo mức trôi — chỗ lộ ra nhiều nhất

| điều kiện | trôi | U0 | U1 | U3 HOPE |
|---|---:|---:|---:|---:|
| m0 | 0% | 0.7700 | 0.7699 | 0.7436 |
| m1 | 50% | 0.6622 | 0.6831 | 0.6355 |
| m2 | **100%** | **0.2905** | **0.3767** | 0.2867 |

**Ba điều bất kỳ phương án mới nào cũng phải để ý:**

1. 🔴 **Trôi 100% làm sụp 45 điểm** (0.770 → 0.291). Đây là điểm yếu lớn nhất, chưa ai giải. m2 chiếm **3/12 chuyến**.
2. Tầng nhanh cứu **+8,6 điểm ở m2** — hiệu quả nhất đúng chỗ khó nhất.
3. HOPE thua **ngay ở trôi 0%** (−2,63đ), nơi ký ức **không liên quan** → đó là chênh **bộ đọc**, không phải kiến trúc.

## D4. Cơ chế — vì sao HOPE thua (đo từ log)

```
α (cổng quên) qua 12 chuyến:  0.843 → … → 0.915   (88% → 96% khoảng)
                              ⚠️ 11 cảnh báo BÃO HOÀ
norm(state):  54.36 … 54.74   ← PHẲNG, ký ức KHÔNG tích luỹ
```
α = 0.915 → giữ lại **8,5%/chunk** → sau 96 ảnh (~3 giây bay) còn **0,06%**.

Và **khoảng cổng `α ∈ [0.05, 0.95]` dẫn cho bài toán CŨ**: ngay ở sàn, chỉ **8,1%** ký ức sống qua một chuyến (49 chunk). Về cấu trúc **không thể** nhớ xuyên chuyến.

**CMS chỉ tác động ở chuyến 0** — pha 2 không gradient. `‖Δw‖` không đổi từ chuyến 1.

---

# PHẦN E — BA PHÁT HIỆN XUYÊN SUỐT

## E1. ⭐ Bộ đọc quan trọng hơn kiến trúc

Trên **cùng feature**, chỉ đổi cách đọc:

| | Linear | NCM-head (oracle) | Δ |
|---|---:|---:|---:|
| HOPE `post_norm=off` | 0.2688 | 0.6874 | **+0.419** |
| Titans + linear | 0.5396 | 0.7098 | +0.170 |
| Titans + cosine | 0.6360 | 0.7375 | +0.101 |

Và σ qua seed: Linear 0.042–0.047 vs NCM-head **0.003–0.005** — **ổn định gấp 8–17 lần**.

Trần NCM-head luôn **0,69–0,77** dù đổi optimizer/head/self-mod/CMS. **Toàn bộ bộ máy Titans làm feature tốt hơn rất ít.**

⚠️ NCM-head chạy `rebuild` = **quét lại toàn bộ data cũ** → **oracle, không triển khai được**.
SDC (bản triển khai được) **thất bại**: sập 1/3 seed (0.2308, forgetting +0.715).
**`head: cosine` là cách duy nhất thành công** chuyển lợi ích đó vào head dùng được: **+15,6 điểm**, dập được seed nổ (`norm(state)` 1,75e6 → 54,5).

## E2. Chi phí — ranh giới cứng (B6, đo thật)

| | Bộ nhớ | Suy luận | O4 |
|---|---:|---:|:---:|
| SLDA streaming (fp32) | **0,73 MB** | 0,0071 ms | ✅ |
| Ba tầng (M1+M2 thêm) | +0,031 MB | +0,069 ms | ✅ dư 325× |
| **Titans self-mod depth 3** | **102,68 MB** | **2,178 ms** | ❌ **vượt 10,3×** |
| ↳ ViT-L (D=1024) | 729,95 MB | 28,37 ms | ❌ 85% khung hình |

Tách: tham số 31,90 MB + **state 70,79 MB**. Xác nhận **O(D²)** (D ×2,67 → ×7,1).

## E3. Ghi nhớ điều kiện không đáng giá — khi ước lượng lại rẻ

λ=0,99, batch 32 → `r = 0.99³² = 0.725/batch` → tầng nhanh hội tụ trong **~4 batch ≈ 4 giây bay**.
Snapshot lưu trữ (trung bình EMA nhiều lần gặp) **mờ hơn** ước lượng tươi → nạp lại **hại 1,2 điểm**.
Ngân hàng chạy đúng cơ chế (3–4 chế độ, thật = 3; nạp lại 7–8/9 chuyến lặp) — nhưng **không có gì để tiết kiệm**.

---

# PHẦN F — 🔧 CHẠY PHƯƠNG ÁN CỦA BẠN

## F1. Chuẩn bị

```bash
git clone https://github.com/VuMinhHien1234/RaybanMeta.git
cd RaybanMeta && git checkout feat/cms-ba-tang
cd uav-continual-learning
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python -m pytest -q          # PHẢI XANH (228 passed)
```

⚠️ **Đúng nhánh `feat/cms-ba-tang`.** `main` lệch 20 commit và **không có** code bài toán mới (`data/revisit.py`, `metrics/revisit.py`, `tang_nhanh.py`).

## F2. Thêm method mới — ba bước

**Bước 1** — viết class trong `src/uavcl/methods.py`, kế thừa `FineTune`, ghi đè hook cần:

```python
class PhuongAnCuaBan(FineTune):
    name = "phuong_an_cua_ban"
    gradient_free = True        # True -> engine gọi model.hap_thu_khong_nhan ở pha 2
                                # False -> engine train bằng gradient (chỉ hợp lệ ở chuyến 0)
```

**Bước 2** — đăng ký:
```python
_METHODS = {..., "phuong_an_cua_ban": PhuongAnCuaBan}
```

**Bước 3** — model phải có `hap_thu_khong_nhan(loader, device, allowed)` cho pha 2:
```python
@torch.no_grad()
def hap_thu_khong_nhan(self, loader, device, allowed=None) -> list:
    """Prequential: DỰ ĐOÁN trước bằng trạng thái hiện có, RỒI mới cập nhật.
    `y` CHỈ dùng để chấm. Trả về list accuracy từng batch -> nuôi O2/O3."""
    accs = []
    for x, y in loader:
        x = x.to(device)
        logits = mask_logits(self(x), allowed)       # 1) dự đoán
        accs.append(float((logits.argmax(1).cpu() == y).float().mean()))
        self.cap_nhat_khong_nhan(x)                  # 2) rồi mới cập nhật, KHÔNG dùng y
    return accs
```

Thiếu hàm này → log in `[pha2] ... đóng băng hoàn toàn (mốc U0)` và bạn chỉ đo lại U0.

## F3. Config — copy rồi sửa 1–2 dòng

```bash
cp configs/revisit_U1_tangnhanh.yaml configs/revisit_UX_cuaban.yaml
```

**Chỉ đổi**:
```yaml
train:
  method: phuong_an_cua_ban      # ← tên vừa đăng ký
  eval_chi_duong_cheo: true      # ← BẬT: 6h/run -> ~55 phút
log:
  dir: ./artifacts_revisit_UX
```

🚫 **KHÔNG đổi** khối `data:` — đó là định nghĩa bài toán. Đổi `num_tasks`, `drift`, `pha2` là **so sánh mất hiệu lực**.

## F4. Chạy — dò trước, 3 seed sau

```bash
# 1. THỬ 2 chuyến (~10 phút) — kiểm cơ chế
.venv/bin/python scripts/run_g1.py --config configs/revisit_UX_cuaban.yaml \
  --set seed=0 train.stop_after_task=1 log.dir=./artifacts_thu_UX 2>&1 | tee thu_UX.log

# 2. Đủ 3 seed (~3h)
for S in 0 1 2; do
  .venv/bin/python scripts/run_g1.py --config configs/revisit_UX_cuaban.yaml \
    --set seed=$S log.dir=./artifacts_revisit_UX_s$S > run_UX_s$S.log 2>&1
done
```

## F5. ⛔ Bốn cửa chặn — đọc TRƯỚC khi nhìn accuracy

```bash
grep -c "đóng băng hoàn toàn" thu_UX.log     # phải = 0
grep "số chuyến LẶP LẠI" thu_UX.log          # phải = 9/12
grep "\[drift\] BẬT" thu_UX.log              # phải có
grep "count_raw" run_UX_s0.log               # nếu method dùng thống kê: phải đứng yên ở pha 2
```

| Cửa | Hỏng nghĩa là |
|---|---|
| có `đóng băng hoàn toàn` | thiếu `hap_thu_khong_nhan` → đang đo lại U0 |
| lịch bay ≠ 9/12 | đổi nhầm `drift` → O3 không so được |
| không có `[drift] BẬT` | chạy không trôi → bài toán khác hẳn |
| `count_raw` tăng ở pha 2 | 🔴 **RÒ RỈ NHÃN** — kết quả vô hiệu |

## F6. Đọc kết quả

```bash
python3 -c "
import json,glob,statistics as st
O1=[];O3=[]
for d in sorted(glob.glob('artifacts_revisit_UX_s*')):
    m=json.load(open(glob.glob(d+'/results/*/metrics_revisit.json')[0]))
    O1.append(m['acc_dieu_kien_hien_tai']); O3.append(m.get('o3_preq10_loi_ich',0))
print('O1 = %.4f ± %.4f' % (st.mean(O1), st.stdev(O1)))
print('O3 = %+.4f ± %.4f' % (st.mean(O3), st.stdev(O3)))
"
.venv/bin/python scripts/bench_cost.py 2>&1 | tail -20      # O4
```

**Luật báo cáo:**
- Luôn **3 seed + σ**. Một seed không đọc được (U3 σ = 0.015, khoảng cách thật 0.053).
- So **ghép cặp theo seed**, không so trung bình rời.
- Ghi kèm **MB và ms**. Vượt 10 MB / 1,67 ms là **không đạt O4**, dù accuracy đẹp.
- Bóc theo **m0 / m1 / m2** — trung bình che mất chỗ sụp ở trôi 100%.

## F7. Mốc phải vượt

```
U0  0.5962 ± 0.0175    (không làm gì)      ← thua cái này thì phương án có hại
U1  0.6282 ± 0.0066    (tầng nhanh)        ← MỐC CHÍNH, phải vượt
    1,51 MB · 0,054 ms                     ← và phải rẻ tương đương
```

---

# PHẦN G — CHƯA AI LÀM (đều chỉ cần config)

| | Kiểm gì | Giờ máy |
|---|---|---|
| Quét **λ** trên revisit | D10 cảnh báo +3,91đ *"có thể đảo dấu"* khi quay lại | ~5h |
| **Oracle có nhãn** (`revisit_arm1_lam1.yaml`) | trần trên — biết 0,63 là tốt hay tệ | ~3h |
| **Pha 1 dài hơn** (`data.pha2.chuyen_hieu_chinh=3`) | Σ 384×384 ước lượng từ **1.597 mẫu** — đói dữ liệu? | ~2h |
| **NCM** trên revisit | Σ có phản tác dụng khi trôi nặng không | ~2h |
| **Sửa khoảng cổng α** cho HOPE (`center=-3.453, limit=3.453`) | HOPE có được thi đấu đúng luật không | ~4h |
| **Vá 45 điểm sụp ở trôi 100%** | ⭐ khoảng trống lớn nhất, chưa ai chạm | — |

---

# PHỤ LỤC — Bản đồ file

| | |
|---|---|
| Phát biểu bài toán | `BAI_TOAN_VA_MUC_TIEU_2026-08-04.md` |
| Kết quả G1 | `uav-continual-learning/docs/KET_LUAN_G1.md` |
| Kết quả D10/D11 | `KET_QUA_D10_D11_2026-08-04.md` |
| Chẩn đoán cổng NL | `CHAN_DOAN_NL_2026-08-02.md` |
| Chi phí triển khai | `KET_QUA_B6_CHI_PHI_2026-08-02.md` |
| Hướng dẫn đọc code | `HUONG_DAN_DOC_CODE_FULL.md` + `DOC_CODE_01/02/03_*.md` |
| Quy trình chạy VM | `uav-continual-learning/HUONG_DAN_CHAY.md` |
| Kết quả thô | `result_test/{batang, u3_hope, final, drift, d11, res_cos, res_sdc, res_base}` |

**Code chính:** `src/uavcl/engine.py` (vòng lặp) · `methods.py` (10 chiến lược) · `models/slda.py` · `models/tang_nhanh.py` · `data/revisit.py` (lịch bay) · `metrics/revisit.py` (O1/O2/O3)

---

*Mọi con số ± là độ lệch chuẩn qua 3 seed. Câu hỏi về bài toán: đọc `BAI_TOAN_VA_MUC_TIEU_2026-08-04.md` trước.*
