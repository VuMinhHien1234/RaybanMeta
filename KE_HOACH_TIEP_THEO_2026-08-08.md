# Kế hoạch tiếp theo

Lập 2026-08-08, sau khi đọc kết quả **U0/U1/U2 (ba tầng)** và **n1_hope (Titans/CMS)**.

---

# 1. Tình trạng: cái gì đã chắc, cái gì chưa đo được

## 1.1. Đã chắc chắn

| | Kết quả | Bằng chứng |
|---|---|---|
| ✅ **Tầng nhanh (M1) hoạt động** | O1: **U1 − U0 = +0.0319 ± 0.0109** | 3/3 seed dương (`+0.0222 · +0.0437 · +0.0299`); σ của U1 (0.0066) nhỏ hơn U0 (0.0175) |
| ✅ **Cổng Titans đã sống** | 0/27 task bão hoà | so với 8/13 run có `α = 1.0000` trước vá |
| ✅ **State Titans đã tích luỹ** | `norm(state)` 62 → 292 qua 9 task | trước vá phẳng ~54 |
| ✅ **O4 — ba tầng đạt rất thoải mái** | 0.0307 MB · 0.0688 ms | trần 10 MB · 1.67 ms |
| ❌ **Titans/HOPE thua NCM trần** | oracle 0.6874 < NCM 0.6933 | và bản triển khai được (Linear) chỉ 0.2688 |

## 1.2. **Chưa đo được** — không phải "âm tính"

Đây là phần quan trọng nhất phải hiểu đúng:

| Câu hỏi | Vì sao chưa đo được |
|---|---|
| **O3 — ngân hàng chế độ (M2) có giá trị không?** | `cho_khop_sau = 12` > `K = 10` → ngân hàng nạp ở batch 12, cửa sổ đo là batch 0–9. **Hành động hoàn toàn ngoài vùng đo.** Kết quả `U2 − U1 = +0.0000 ± 0.0000` là **tất yếu về mặt cơ học**, không mang thông tin |
| **CMS đa tần số có tác dụng không?** | `cms:` đặt dưới `train:` thay vì top-level → `run_g1.py:189` không thấy → model là Titans-đóng-băng thay vì HOPE → tier mid/slow **rỗng** → `‖Δw‖ = 0.0000` chính xác. **Lần thứ 3 CMS không chạy** |
| **`post_norm` có phải thủ phạm triệt tiêu η?** | Lần chạy đổi **4 biến cùng lúc** (post_norm, gate_bound, pre_norm, CMSOptimizer). Không tách được |

**Ba câu hỏi trên là toàn bộ nội dung kế hoạch này.**

---

# 2. P0 — Ba bản vá code (làm TRƯỚC, ~1 giờ, 0 giờ máy)

Không vá thì mọi run sau đều lặp lại đúng lỗi cũ.

## P0.1 ⭐ `eval_chi_duong_cheo` — 6h/run → ~55 phút/run

**Vì sao:** với `num_tasks: 12` + `pha2.test_chung: true`, engine chấm `1+2+…+12 = 78` lượt × 6300 ảnh = **491.400 forward pass**. Nhưng cả ba thước đo của bài toán (`acc_hien_tai`, `loi_ich_quay_lai`, O2) **chỉ đọc 12 ô đường chéo**. 66/78 lượt bị vứt đi.

**Sửa** — `src/uavcl/engine.py`, dòng 310:

```python
# CŨ
        for j in range(t + 1):                      # ↳ Chấm điểm lại toàn bộ task 0..t.
            R[t, j] = evaluate(model, task_loaders[j]["test"], device, allowed_eval)

# MỚI
        # Với stream revisit, mọi thước đo (O1/O2/O3) chỉ đọc ĐƯỜNG CHÉO R[t][t].
        # test_chung=true khiến mỗi lượt eval = toàn bộ 6300 ảnh -> tam giác dưới
        # tốn 66/78 lượt mà không ai đọc. Cờ này bỏ chúng đi. Mặc định TẮT.
        _chi_cheo = bool(train_cfg.get("eval_chi_duong_cheo", False))
        for j in ([t] if _chi_cheo else range(t + 1)):
            R[t, j] = evaluate(model, task_loaders[j]["test"], device, allowed_eval)
```

Và dòng 318 (in log) cho khỏi in số 0 gây hiểu nhầm:

```python
            row = "  ".join(f"{R[t, j]:.3f}" for j in ([t] if _chi_cheo else range(t + 1)))
```

**Và** trong `scripts/run_g1.py`, cho `metrics.json` ghi `null` thay vì số rác:

```python
        "average_accuracy": None if cfg["train"].get("eval_chi_duong_cheo") else average_accuracy(R),
        "average_anytime_accuracy": None if cfg["train"].get("eval_chi_duong_cheo") else average_anytime_accuracy(R),
        "average_forgetting": None if cfg["train"].get("eval_chi_duong_cheo") else average_forgetting(R),
        "backward_transfer": None if cfg["train"].get("eval_chi_duong_cheo") else backward_transfer(R),
```

> **So được với U0/U1 đã chạy không?** Có. `R[t][t]` tính **y hệt** dù có bỏ tam giác dưới hay không, và `o3_preq10` lấy từ `log["trace"]` chứ không từ R. Ba thước đo O1/O2/O3 **bất biến**.

## P0.2 ⭐ Cửa chặn `cho_khop_sau < K`

**Vì sao:** đây là lỗi vừa đốt 12 giờ máy cho hai run không mang thông tin. Phải để máy tự chặn, không để người nhớ.

**Sửa** — `scripts/run_g1.py`, ngay sau khối đọc `_p2cfg` (dòng ~147):

```python
        # CỬA CHẶN (2026-08-08): O3-preq10 đo trung bình K=10 batch ĐẦU mỗi chuyến.
        # Ngân hàng chế độ khớp & nạp ở batch `cho_khop_sau`. Nếu cho_khop_sau >= K thì
        # nó hành động NGOÀI cửa sổ đo -> O3(U2) không thể khác O3(U1), và kết quả
        # +0.0000 ± 0.0000 trông như "ngân hàng vô dụng" trong khi nó chưa từng được đo.
        _nh = (cfg.get("slda") or {}).get("ngan_hang") or {}
        if bool(_p2cfg.get("khong_nhan")) and bool(_nh.get("enabled")):
            _ck = int(_nh.get("cho_khop_sau", 5))
            if _ck >= 10:
                raise ValueError(
                    f"slda.ngan_hang.cho_khop_sau={_ck} >= K=10 của O3-preq10 — ngân hàng "
                    "sẽ nạp NGOÀI cửa sổ đo, O3 không thể khác U1. Đặt cho_khop_sau <= 8.")
```

## P0.3 Sửa vị trí `cms:` trong config

**Vì sao:** `run_g1.py:189` đọc **top-level** `cfg["cms"]` để chọn model; `engine.py:156` đọc `train_cfg["cms"]` để chọn optimizer. Config `g2_titans_resisc45_selfmod_m3_cms.yaml` chỉ có cái thứ hai → CMSOptimizer chạy trên model **backbone đóng băng** → tier mid/slow rỗng.

**Sửa** — trong `configs/g2_titans_resisc45_selfmod_m3_cms.yaml`, chuyển khối `cms:` từ dưới `train:` lên **top-level** (ngang hàng với `data:`, `memory:`, `train:`), y như `g3_cms_*.yaml` và `g4_hope_*.yaml`:

```yaml
memory:
  ...

cms:                      # ← TOP-LEVEL (run_g1 đọc chỗ này để chọn HOPEClassifier)
  enabled: true
  m3_frequency_unit: global_step
  grad_agg: sum
  eta_mode: fixed

train:
  method: hope            # đổi luôn cho khớp model
  ...
```

⚠️ **Hệ quả phải biết trước:** sửa xong thì model thành `HOPEClassifier` → backbone **mở băng** → `trainable_params` nhảy từ 7,99M lên >21M → thời gian chạy tăng đáng kể, và đây mới là **HOPE thật lần đầu tiên**.

## Kiểm sau khi vá

```bash
cd ~/RaybanMeta/uav-continual-learning
.venv/bin/python -m pytest -q          # PHẢI XANH 228 passed
git add -A && git commit -m "fix: eval chi duong cheo + cua chan cho_khop_sau + cms top-level"
git push
```

---

# 3. Ưu tiên chạy

## 🔴 P1 — Đo O3 cho thật (câu hỏi TRUNG TÂM)

Đây là thứ duy nhất chưa từng được đo tử tế, và là lý do tồn tại của cả dự án.

### P1a. Một run dò (~55 phút sau khi vá P0.1)

Ghép **ngưỡng của A** (4 chế độ — tốt nhất) với **thời điểm của B** (nằm trong cửa sổ đo):

```bash
.venv/bin/python scripts/run_g1.py --config configs/revisit_U2_nganhang.yaml \
  --set seed=0 log.dir=./artifacts_U2_D train.eval_chi_duong_cheo=true \
        slda.ngan_hang.nguong=12.0 slda.ngan_hang.cho_khop_sau=5 \
  > run_U2_D.log 2>&1

grep -E "tầng trung|PREQ10|GẶP LẠI" run_U2_D.log | tail -5
```

**Tiêu chí qua cửa — cả BA phải đạt:**

| | Ngưỡng |
|---|---|
| `số chế độ` | **3–5** (số thật = 3) |
| `nạp lại` | **≥ 6** trên 9 chuyến lặp |
| `o3_preq10` **khác** U1 (−0.0074) | bất kỳ, miễn **không** trùng khít |

Điều kiện thứ ba là quan trọng nhất — nó chứng minh ngân hàng **có được đo**.

**Nếu hỏng:** thử `nguong=15.0 cho_khop_sau=5`, rồi `nguong=12.0 cho_khop_sau=3`.

### P1b. Ba seed (~3 giờ)

```bash
for S in 0 1 2; do
  .venv/bin/python scripts/run_g1.py --config configs/revisit_U2_nganhang.yaml \
    --set seed=$S log.dir=./artifacts_revisit_U2_v2_s$S train.eval_chi_duong_cheo=true \
          slda.ngan_hang.nguong=12.0 slda.ngan_hang.cho_khop_sau=5 \
    > run_U2_v2_s$S.log 2>&1
done
```

**Số công bố:** `o3_preq10(U2_v2) − o3_preq10(U1)`, ghép cặp theo seed, 3 seed, kèm σ.

**Luật đọc:**

| Kết quả | Kết luận |
|---|---|
| > 0 và > 2σ | ⭐ **Ghi nhớ điều kiện có giá trị** — đây là đóng góp chính của luận văn |
| ≈ 0, σ nhỏ | Ngân hàng **không** giúp. Vẫn là kết quả: *"trong regime này, bám trôi đủ, không cần nhớ"* |
| < 0 | Ngân hàng **có hại** — nạp snapshot cũ làm hỏng ước lượng hiện tại. Cũng là kết quả, cần giải thích |

Cả ba trường hợp đều báo cáo được. Chỉ có `+0.0000 ± 0.0000` như hiện tại là **không** báo cáo được.

## 🟡 P2 — Trần trên có nhãn (~1 giờ sau vá, 3 seed)

```bash
for S in 0 1 2; do
  .venv/bin/python scripts/run_g1.py --config configs/revisit_arm1_lam1.yaml \
    --set seed=$S log.dir=./artifacts_revisit_arm1_s$S train.eval_chi_duong_cheo=true \
    > run_arm1_s$S.log 2>&1
done
```

**Vì sao cần:** hiện U2 ≈ 0.628 mà **không biết 0.628 là tốt hay tệ**. Oracle có nhãn cho biết pha-2-không-nhãn đang phải trả giá bao nhiêu.

Bảng cuối cùng sẽ đọc được:
```
trần có nhãn      0.78     ← nếu được phép dùng nhãn ở mọi chuyến
U2 (ba tầng)      0.63     ← không nhãn, có nhớ
U1 (tầng nhanh)   0.63     ← không nhãn, không nhớ
U0 (đóng băng)    0.60     ← không làm gì
```

## 🟡 P3 — Ablation `post_norm`, ĐÚNG MỘT BIẾN (~4,5h/seed)

Chạy lại **y hệt** cấu hình n1 nhưng `post_norm=true`, và **tắt `eval_ncm_head`** (bộ đọc NCM rebuild là phần tốn giờ nhất, mà câu hỏi ở đây là về head Linear):

```bash
.venv/bin/python scripts/run_g1.py --config configs/g2_titans_resisc45_selfmod_m3_cms.yaml \
  --set seed=0 memory.post_norm=true train.method=hope log.dir=./artifacts_n2_postnorm_s0 \
  > run_n2_postnorm_s0.log 2>&1
```

**Giả thuyết đang kiểm:** `post_norm` không phải "lớp thừa triệt tiêu η" như D11 nghi, mà là **cái neo giữ thang đo cho head Linear**.

Ba dấu hiệu ủng hộ, đo được từ n1:
- `norm(state)` tăng **4,7 lần** (62 → 292) khi tắt post_norm
- Linear sập 0.6079 → 0.2688, σ lớn (±0.047)
- NCM-head (chuẩn hoá feature) **không hề hấn**: 0.6874, σ chỉ 0.0028

**Đọc:** `post_norm=true` cho Linear quay về vùng ~0.6 → giả thuyết đúng, và câu D11 Ưu tiên 2 khép lại.

## 🟢 P4 — CMS thật (HOPE) lần đầu (~6h/seed, làm sau P0.3)

```bash
.venv/bin/python scripts/run_g1.py --config configs/g2_titans_resisc45_selfmod_m3_cms.yaml \
  --set seed=0 train.method=hope log.dir=./artifacts_n3_hope_that_s0 \
  > run_n3_hope_that_s0.log 2>&1
```

**Cửa chặn cơ chế — đọc TRƯỚC accuracy:**

```bash
grep "trainable_params" artifacts_n3_hope_that_s0/results/*/metrics.json   # phải > 21.000.000
grep "‖Δw‖" run_n3_hope_that_s0.log | head -6
```

| | Phải thấy |
|---|---|
| `trainable_params` | **> 21M** (n1 chỉ 7,99M = backbone đóng băng) |
| `‖Δw‖ slow` | **≠ 0.0000** nhưng nhỏ (< 0.1% ‖w‖) |
| `‖Δw‖ fast` | lớn hơn `mid` lớn hơn `slow` |

`slow` vẫn ra `0.0000` chính xác → P0.3 chưa ăn, dừng sửa tiếp.

## ⚪ P5 — `TitansKhongNhan` (~2h code, chỉ làm nếu P1 xong và còn thời gian)

Cần để đo Titans **trên chính bài toán UAV** (revisit + pha 2 không nhãn), cùng seed với U0/U1/U2.

Hiện `engine.py` chỉ gác pha-2-không-nhãn trong nhánh `gradient_free`, mà `TitansCL.gradient_free = False` → chạy revisit ngay bây giờ thì Titans **có nhãn ở cả 12 chuyến** = so sai bài toán.

**Đừng làm trước P1** — không có mốc U1/U2 thì con số Titans không diễn giải được.

---

# 4. Lịch chạy hai máy

| | `uavcl-slda` | `uavcl-titans` |
|---|---|---|
| **Ngay** | P0 (vá code, trên Mac) → push → pull trên cả 2 máy | |
| **Buổi 1** | **P1a** dò (~1h) → nếu qua → **P1b** 3 seed (~3h) | **P3** post_norm 1 biến, seed 0 (~4,5h) |
| **Buổi 2** | **P2** oracle 3 seed (~3h) | **P4** HOPE thật, seed 0 (~6h) |
| **Buổi 3** | (dự phòng: chạy lại P1 nếu ngưỡng chưa chuẩn) | P3/P4 thêm seed nếu buổi 1–2 có tín hiệu |
| **Sau cùng** | P5 nếu còn thời gian | |

**Đường găng là P1.** P2–P5 đều là bổ trợ. Nếu chỉ còn thời gian cho một thứ, làm P1.

---

# 5. Bảng kết quả đích — điền vào là xong phần thực nghiệm

```
BÀI TOÁN UAV (revisit, pha 2 KHÔNG nhãn, 12 chuyến, 3 seed)

                     O1 (acc)         O3 (lợi ích quay lại)      MB      ms
  trần có nhãn       ____ ± ____      —                          1.5     0.007
  U0 đóng băng       0.596 ± 0.018    −0.001 ± 0.019             1.5     0.007
  U1 + tầng nhanh    0.628 ± 0.007    −0.005 ± 0.003             1.5     0.028
  U2 + ngân hàng     ____ ± ____      ____ ± ____   ⭐           1.5     0.069
  (Titans/HOPE)      —                —                        102.7     2.178

  U1 − U0  =  +0.0319 ± 0.0109   (O1)   ✅ đã có
  U2 − U1  =  ____ ± ____        (O3)   ⭐ THIẾU — P1
```

Ba ô trống là toàn bộ việc còn lại của phần thực nghiệm.

---

# 6. Rủi ro và phương án dự phòng

| Rủi ro | Dấu hiệu | Phương án B |
|---|---|---|
| Không ngưỡng nào cho `nạp lại ≥ 6` **và** nằm trong cửa sổ đo | P1a hỏng cả 3 lần thử | Giảm `cho_khop_sau` xuống 3, hoặc tăng `K` bằng cách sửa `run_g1.py:345` thành `K = 20` (dùng cùng K cho **cả** U1 và U2 để so công bằng) |
| U2 − U1 ≈ 0 với σ nhỏ, ngân hàng đo được | P1b ra `+0.001 ± 0.002` | **Đây là kết quả hợp lệ.** Báo cáo: *"trong regime này, bám trôi không nhãn là đủ; nhớ điều kiện không thêm giá trị đo được"* + phân tích vì sao (λ=0,99 hội tụ trong ~4 batch nên chi phí học lại vốn đã rất thấp) |
| Hết giờ máy | | Bỏ P4, P5. Giữ P1 + P2. Phần Titans báo cáo bằng số **đã có** (0.6874 oracle < 0.6933 NCM trần, 102,68 MB > 10 MB) |
| `post_norm=true` cũng không cứu Linear | P3 ra ~0.3 | Nguyên nhân nằm ở CMSOptimizer bọc rỗng hoặc `gate_bound`. Chạy thêm 1 biến: bỏ `train.cms` |

---

# 7. Điều KHÔNG làm — để khỏi sa đà

| | Vì sao bỏ |
|---|---|
| Quét trần η của Titans | D11: đổi 100× → Δacc **0.0000157**, 3 seed. Đã kết luận |
| `beta_style=paper` / `update_norm=none` của M3 | 1 đêm, gần chắc chắn tệ hơn. Chỉ chạy **1 seed** nếu cần dòng *"đã cài đúng Algorithm 1"* trong báo cáo |
| Thêm ablation `cov_mode` của SLDA | đã có 3 seed × 3 chế độ |
| Titans trên EuroSAT | dataset phụ, ngoài câu chuyện UAV |
| Tinh chỉnh thêm M3 | chưa thắng nổi AdamW (0.424 vs 0.577), và **nhánh chính SLDA không dùng optimizer** |

---

# 8. Ba câu chuyện sẽ viết trong báo cáo

Dù P1 ra kết quả nào, ba luận điểm sau **đã có đủ số**:

**① Hiện thực hoá NL bằng gradient không sống được ở biên.**
Cài đủ cả 3 đóng góp của Nested Learning. Đo trên ràng buộc phần cứng UAV thật: vi phạm ngân sách bộ nhớ **10,3 lần** (102,68 vs 10 MB). Cổng học được bão hoà **8/13 run** trước khi vá. Sau khi vá cổng sống và state tích luỹ (62→292), nhưng bộ đọc tốt nhất (oracle) vẫn **thua ViT đóng băng + NCM** (0.6874 < 0.6933).

**② Hiện thực hoá cùng cấu trúc bằng công thức đóng thì sống được.**
Ba tầng theo đúng khung CMS (đóng góp #3 của NL, hợp lệ theo Eq 9/21/33 của chính bài báo): **0,031 MB · 0,069 ms** — rẻ hơn **3.300 lần** về bộ nhớ, **32 lần** về độ trễ. λ cố định **về mặt cấu trúc không thể** trôi ra biên như cổng học được.

**③ Bám trôi không nhãn có giá trị đo được.**
`U1 − U0 = +0.0319 ± 0.0109` trên O1, 3/3 seed dương. Ước lượng điều kiện từ thống kê feature **không cần một nhãn nào**.

Ô còn trống duy nhất: **ghi nhớ điều kiện (M2) có thêm giá trị không** — chính là P1.

---

*Cập nhật file này sau mỗi mốc. Ưu tiên P0 → P1; mọi thứ khác là bổ trợ.*
