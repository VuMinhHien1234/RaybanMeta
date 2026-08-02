# Chẩn đoán: các cơ chế Nested Learning có thật sự đang chạy không?

Ngày: 2026-08-02 · Nhánh: `memory_titan_task4_v2`
Đối tượng kiểm tra: (1) `alpha` kẹt ở 1.0, (2) tần số CMS bị nhân sai.

**Kết luận ngắn: 2/3 trụ cột của Nested Learning chưa từng thực sự chạy trong bất kỳ
run nào của chiến dịch gần đây.**

---

## LỖI 1 — Hướng dẫn đọc `alpha` bị ĐẢO NGƯỢC

### Bằng chứng từ mã nguồn thư viện

`.venv/.../titans_pytorch/neural_memory.py`:

```python
# dòng 634 — thư viện TỰ áp sigmoid
decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

# dòng 813 — learned forgetting, Eq (13)
update = self.assoc_scan(1. - decay_factor, update, prev = last_update, remove_prev = False)
```

Hệ số truyền vào `assoc_scan` là **`1 − decay_factor`**, tức đây là **hệ số GIỮ LẠI**.

| `decay_factor` (= α) | Hệ số giữ lại `1−α` | Nghĩa thực tế |
|---:|---:|---|
| 1.0 | 0.0 | **Xoá sạch** tích luỹ mỗi chunk |
| 0.0 | 1.0 | **Giữ 100%** — không quên gì cả |

### Nhãn hiện tại trong dự án

`src/uavcl/models/memory.py:65` và chú thích cuối `compare_all.py`:

> `alpha(forget-gate)~1 = không quên = hướng nổ`

**Tên biến đúng** (nó đúng là forget-gate), **nhưng hướng đọc ngược 180°**.
Phải là: `alpha ~ 0 = không quên = hướng nổ`.

### Bằng chứng thực nghiệm — chính dữ liệu của bạn xác nhận

Sắp xếp 13 run theo `norm(state)`:

| Run | α | Giữ lại `1−α` | norm(state) |
|---|---:|---:|---:|
| `m3_s1` | **0.0000** | **1.0000** | **1 757 552** ⚠️ |
| `sdc_s1` | 1.0000 | 0.0000 | 575.4 |
| `cosine_s2` | 0.1246 | 0.8754 | 122.5 |
| `latreplay_s1` | 1.0000 | 0.0000 | 88.5 |
| `cosine_s0` | 0.2441 | 0.7559 | 63.9 |
| `latreplay_s0` | 0.4814 | 0.5186 | 59.8 |
| `m3_s0` | 1.0000 | 0.0000 | 54.6 |
| `cosine_s1` | 1.0000 | 0.0000 | 54.5 |
| `sdc_s0` | 1.0000 | 0.0000 | 54.4 |
| `latreplay_s2` | 1.0000 | 0.0000 | 54.3 |
| `rebuild_s0` | 1.0000 | 0.0000 | 53.8 |
| `sdc_s2` | 1.0000 | 0.0000 | 53.8 |
| `adamw_s0` | 0.8335 | 0.1665 | 53.6 |

**Run duy nhất có α = 0.0000 (giữ lại 100%) chính là run nổ norm 1,75 triệu.**
Đây là bằng chứng quyết định cho chiều đọc đúng — và nó nằm sẵn trong log của bạn.

### Hệ quả nghiêm trọng

**8/13 run có `α = 1.0000` — cổng quên bão hoà ở mức xoá tối đa.**

Bao gồm **cả 3 run SDC**, `rebuild_s0`, `cosine_s1`, `latreplay_s1/s2`, `m3_s0`.

Ở các run đó `norm(state)` đứng yên ~54 suốt cả 9 task — **state không hề tích luỹ**.
Titans đang chạy như một phép biến đổi gần-như-không-trạng-thái. Ký ức liên-task,
tức toàn bộ lý do tồn tại của Titans, đã không hoạt động.

---

## LỖI 2 — Đầu dò `eta` đo sai vị trí, con số trong log vô nghĩa

### Bằng chứng

```python
# neural_memory.py:629-630
adaptive_lr = self.to_adaptive_step(seq)          # <- probe hook Ở ĐÂY (logit thô)
adaptive_lr = self.adaptive_step_transform(adaptive_lr)   # <- transform THẬT ở đây

# neural_memory.py:255-256
def default_adaptive_step_transform(adaptive_step, max_lr = 1e-2):
    return adaptive_step.sigmoid() * max_lr

# neural_memory.py:272 + :457  <-- BẪY: default 1e-2 ở trên KHÔNG BAO GIỜ được dùng
default_step_transform_max_lr = 1.,
adaptive_step_transform = partial(default_adaptive_step_transform,
                                  max_lr = default_step_transform_max_lr)
```

Probe (`memory.py:78-79`) gắn vào `to_adaptive_step` — **trước** `adaptive_step_transform`.
Nên `eta = 55.6201` trong log **không phải learning rate**, mà là logit thô.

> **ĐÍNH CHÍNH 2026-08-02 (bắt được bởi `tests/test_gate_bound.py`).** Bản đầu tài liệu này
> ghi `max_lr = 1e-2` — SAI. Dòng `:457` luôn ghi đè bằng `default_step_transform_max_lr = 1.`
> (`:272`), nên **max_lr THẬT = 1.0**. Mọi con số η ở bản trước lệch **100 lần**.

η thật = `sigmoid(55.62) × 1.0` = **1.0 = đúng bằng trần `max_lr`**.

**Đính chính này làm chẩn đoán NẶNG hơn, không nhẹ đi:** memory không chạy với
learning-rate 0.01 mà với **1.0 — bước ghi cỡ đầy đủ**. Ở `m3_s1`, η = 1.0 (ghi hết cỡ)
đi kèm α = 0 (giữ lại 100%) chính là công thức nổ: ghi tối đa, không quên gì, tích luỹ vô hạn.

### Quy đổi lại toàn bộ

| Run | η log (thô) | η THẬT (×1.0) | Trạng thái |
|---|---:|---:|---|
| `m3_s1` | 59.674 | **1.0000** | Dính trần — ghi hết cỡ |
| `m3_s0` | 55.620 | **1.0000** | Dính trần |
| `sdc_s0` | 18.415 | **1.0000** | Dính trần |
| `cosine_s1` | 17.926 | **1.0000** | Dính trần |
| `latreplay_s2` | 14.734 | **1.0000** | Dính trần |
| `sdc_s1` | −5.960 | 0.0026 | Bão hoà đáy |
| `cosine_s0` | −5.229 | 0.0053 | Bão hoà đáy |
| `adamw_s0` | −0.810 | 0.3079 | **Lành mạnh** |
| `latreplay_s0` | −0.101 | 0.4748 | **Lành mạnh** |

**9/13 run có η bão hoà.** Chỉ 2 run có η nằm ở vùng giữa còn phản ứng theo dữ liệu.

---

## Nguyên nhân gốc chung của Lỗi 1 & 2

Cả hai cổng đều là `nn.Linear(dim, heads)` áp thẳng lên feature:

```python
self.to_adaptive_step = Sequential(nn.Linear(dim, heads), ...)   # dòng 451
self.to_decay_factor  = Sequential(nn.Linear(dim, heads), ...)   # dòng 514
```

Trong `titans_head.py` có `post_norm = nn.LayerNorm(dim)` — nhưng đó là **SAU** memory.
**Không có chuẩn hoá nào TRƯỚC khi vào memory.** Feature ViT-S thô đi thẳng vào hai
`Linear` này → logit cỡ ±50 → **cả hai sigmoid đều bão hoà**.

`qk_rmsnorm: true` chỉ chuẩn hoá query/key trong đường đọc/ghi, **không chạm** tới
hai cổng này.

### Đây chính xác là chỗ Nested Learning bị vô hiệu hoá

NL Eq 76 nói: η_t và α_t **phụ thuộc dữ liệu** (`x_t·W_η`, `x_t·W_α`). Khi cả hai
sigmoid bão hoà, đạo hàm ≈ 0 và giá trị đầu ra là hằng số — **sự phụ thuộc dữ liệu
đã chết**. Cơ chế cốt lõi của paper không chạy.

Đáng chú ý: thư viện **có sẵn** `init_adaptive_step_bias` / `init_decay_bias`
(dòng 526–540) đúng để tránh chuyện này. Dự án chưa dùng.

---

## LỖI 3 — CMS chưa từng được bật

Bản sửa tần số per-tier trong `cms_optimizer.py:147-169` (`m3_frequency_unit`)
**đã có trong code và viết đúng**:

```python
"frequency": max(1, (base_m3_frequency + int(g["period"]) - 1) // int(g["period"]))
```

Nhưng nó chỉ chạy khi `cms.enabled`:

```python
# engine.py:156
if (train_cfg.get("cms") or {}).get("enabled", False):
    opt = build_cms_optimizer(model, train_cfg)
```

Kiểm config đang chạy:

```
$ grep -c "cms" configs/g2_titans_resisc45_selfmod_m3_improved.yaml
0
```

**Không có mục `cms:` nào.** Chỉ `g3_cms_*.yaml` và `g4_hope_*.yaml` mới có.

→ Bản sửa đúng nhưng **chưa từng được kích hoạt**. Continuum Memory System —
đóng góp thứ 3 của NL, cơ chế đa tần số — **hoàn toàn vắng mặt** trong mọi kết quả
của chiến dịch vừa rồi.

---

## Bảng tổng: 3 trụ cột NL, tình trạng thực tế

| Đóng góp NL | Cài đặt | Thực tế khi chạy |
|---|---|---|
| 1. Expressive optimizer (M3 deep-memory) | Có | ✅ **Đang chạy** |
| 2. Self-modifying module (Eq 76 η/α data-dependent) | Có | ❌ **Cả hai cổng bão hoà → cơ chế chết** |
| 3. Continuum Memory System (đa tần số) | Có | ❌ **Không bật trong config** |

**Đây là câu trả lời cho câu hỏi "tại sao nhiều tầng lại thua SLDA".**
Không phải NL sai. Là các tầng đang tê liệt, và tầng duy nhất thực sự tích luỹ
thống kê theo thời gian trong toàn hệ thống lại là… SLDA.

---

## Kế hoạch sửa — theo đúng tinh thần NL

### Sửa 1 — Đảo lại hướng đọc (5 phút, không đụng kết quả)

`src/uavcl/models/memory.py:65` và chú thích `compare_all.py`:

```
# CŨ (SAI):  α_t ~ 1 = không quên = hướng nổ
# MỚI:       α_t = decay_factor; giữ lại = (1-α). α~0 = không quên = hướng NỔ.
#            α~1 = xoá sạch mỗi chunk = memory không tích luỹ.
```

Thêm cảnh báo tự động khi `α > 0.99` hoặc `α < 0.01` → **cổng bão hoà, số không tin được**.

### Sửa 2 — Log η THẬT thay vì logit thô

Đổi hook từ `to_adaptive_step` sang sau `adaptive_step_transform`, hoặc áp
`sigmoid(x) * max_lr` ngay trong hook. Log cả hai (thô + thật) để không mất
khả năng so với run cũ.

### Sửa 3 — Gỡ bão hoà: chuẩn hoá đầu vào memory ⭐ QUAN TRỌNG NHẤT

Thêm `pre_norm = nn.LayerNorm(dim)` **trước** khi seq vào memory
(`titans_head.py`, cạnh `post_norm` sẵn có). Đặt sau cờ config
`memory.pre_norm: true` (mặc định `false` → run cũ bất biến).

Tiêu chí xong: η thật nằm trong `[1e-4, 9e-3]` và α nằm trong `[0.05, 0.95]`,
**biến thiên theo task** — tức Eq 76 sống lại.

Phương án dự phòng nếu vẫn bão hoà: dùng `init_decay_bias` / `init_adaptive_step_bias`
mà thư viện đã hỗ trợ, cho hai cổng khởi đầu trung tính.

### Sửa 4 — Bật CMS trong config Titans

Tạo `configs/g2_titans_resisc45_selfmod_m3_cms.yaml` = config hiện tại + thêm:

```yaml
train:
  cms:
    enabled: true
    m3_frequency_unit: global_step   # bản sửa đã có sẵn
    grad_agg: sum                    # nguyên văn Eq 71
    eta_mode: fixed
```

Xác minh khi chạy: dòng `[cms] M3 frequency unit=global_step | per-tier f=[...]`
phải in ra, và `‖Δw‖` per-tier phải cho thấy **tier chậm ≈ 0, tier nhanh lớn**.
Nếu ngược → mapping tier sai, dừng lại sửa trước.

---

## Thứ tự chạy sau khi sửa

1. `pytest -q` phải xanh.
2. **Smoke 1 task**, chỉ đọc η/α — chưa quan tâm accuracy. Cổng phải nằm trong
   vùng lành mạnh. **Chưa đạt thì không chạy tiếp** — mọi số sau đó đều vô nghĩa.
3. Chạy lại Titans 3 seed với cổng đã sống → so với chính nó trước khi sửa.
   Đây mới là **lần đầu tiên NL thực sự được kiểm chứng** trong dự án này.
4. Rồi mới tới thí nghiệm quyết định **SLDA + `memory.enabled=true`**.

Bước 4 chạy trước bước 2–3 sẽ cho kết quả không kết luận được: nếu Titans không
thêm gì, ta sẽ không phân biệt được "Titans vô dụng" với "Titans đang tê liệt".
