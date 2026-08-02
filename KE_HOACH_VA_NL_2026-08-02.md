# Kế hoạch vá — khôi phục cơ chế Nested Learning

Ngày: 2026-08-02 · Nhánh đề xuất: `fix/nl-gates-alive`

> **Trạng thái: ĐỀ XUẤT — chưa thực hiện.** Tài liệu này để bạn đọc và quyết định.
> Mỗi task chứa sẵn **bằng chứng lỗi** + **cách sửa**, đọc độc lập được, không cần file khác.

---

## Tóm tắt: 3 trụ cột Nested Learning, tình trạng thực tế

| Đóng góp NL | Cài đặt | Thực tế khi chạy | Task sửa |
|---|---|---|---|
| 1. Expressive optimizer (M3 deep-memory) | Có | ✅ **Đang chạy** | — |
| 2. Self-modifying, Eq 76 η/α phụ thuộc dữ liệu | Có | ❌ **Hai cổng bão hoà → cơ chế chết** | 0–5 |
| 3. Continuum Memory System (đa tần số) | Có | ❌ **Không bật trong config** | 6 |

**Đây là câu trả lời cho "tại sao nhiều tầng lại thua SLDA".** Không phải NL sai — là các
tầng đang tê liệt. Và trong toàn hệ thống, thành phần duy nhất thực sự tích luỹ thống kê
theo thời gian lại chính là SLDA.

## Nguyên tắc xuyên suốt

1. **Mọi thay đổi hành vi phải nằm sau một cờ config mặc định TẮT.**
   Không có cờ → chạy y hệt hôm nay → mọi run cũ vẫn tái lập được, và so được 1-biến.
2. **Sửa chẩn đoán trước, sửa cơ chế sau.** Chưa nhìn đúng thì chưa được sửa.
3. **Không tin bất kỳ accuracy nào cho tới khi hai cổng η/α ra khỏi vùng bão hoà.**

Tổng công sức: **~4 giờ code + 1 lần chạy smoke**, chưa kể chạy lại campaign.

---

# NHÓM 0 — Miễn phí, làm được ngay, không sửa gì

## TASK 0 — Trích quỹ đạo α/η theo task từ log ĐÃ CÓ

### Lỗi liên quan — LỖI 1: hướng đọc `alpha` bị ngược 180°

Bằng chứng trong `.venv/.../titans_pytorch/neural_memory.py`:

```python
# dòng 634 — thư viện TỰ áp sigmoid
decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

# dòng 813 — learned forgetting, Eq (13)
update = self.assoc_scan(1. - decay_factor, update, prev = last_update, remove_prev = False)
```

Hệ số truyền vào `assoc_scan` là **`1 − decay_factor`** → đó là hệ số **GIỮ LẠI**.
Vậy **α = 1 → giữ lại 0 → xoá sạch**, không phải "không quên".

Dữ liệu của chính dự án chứng minh chiều đọc đúng:

| Run | α | Giữ lại `1−α` | norm(state) |
|---|---:|---:|---:|
| `m3_s1` | **0.0000** | **100%** | **1 757 552** ⚠️ |
| `cosine_s2` | 0.1246 | 88% | 122.5 |
| `cosine_s0` | 0.2441 | 76% | 63.9 |
| `latreplay_s0` | 0.4814 | 52% | 59.8 |
| `adamw_s0` | 0.8335 | 17% | 53.6 |
| 8 run còn lại | 1.0000 | **0%** | ~54 (phẳng) |

Run duy nhất có α = 0 (giữ lại 100%) chính là run nổ 1,75 triệu. **Quan hệ đơn điệu hoàn hảo.**

**Hệ quả:** 8/13 run có α = 1.0000 — gồm **cả 3 run SDC**. Ở đó `norm(state)` đứng yên ~54
suốt 9 task → **state không hề tích luỹ**. Titans chạy như phép biến đổi gần-như-không-trạng-thái.
Ký ức liên-task, lý do tồn tại duy nhất của Titans, đã không hoạt động.

### Vì sao task này đáng làm đầu tiên

`methods.py:263` in η/α **sau MỖI task**, `memory.py:83` reset ở đầu mỗi task.
Nghĩa là log hiện có **đã chứa 9 giá trị α cho mỗi run** — nhưng `compare_all.py:113`
chỉ lấy giá trị **cuối cùng** (`eas[-1]`).

**Toàn bộ dữ liệu cần để xác nhận chẩn đoán đã nằm sẵn trên đĩa.** Không chạy lại gì.

### Việc

- **File mới:** `scripts/trace_gates.py` (đọc-thôi, ~40 dòng, chỉ thư viện chuẩn)
- Đọc mọi `run_*.log`, `grep` toàn bộ dòng
  `eta_t(avg)=... alpha_t/forget-gate(avg)=...`, in dãy 9 giá trị theo task cho từng run.

### Câu hỏi cần trả lời

- α bão hoà ở 1.0 **ngay từ task 0**, hay **trôi dần** lên 1.0 qua các task?
- η bão hoà từ đầu hay muộn?

### Vì sao câu trả lời quyết định hướng Task 4

- Bão hoà **ngay từ đầu** → lỗi thang đo feature → `pre_norm` (Task 4) là đúng thuốc.
- **Trôi dần** → vòng phản hồi trong lúc train → cần thêm chặn (weight decay trên hai cổng,
  hoặc clamp logit).

**Rủi ro:** không, chỉ đọc file · **Thời gian:** 30 phút
**Làm task này trước khi viết dòng code sửa nào.**

---

# NHÓM 1 — Sửa chẩn đoán (KHÔNG đổi hành vi train)

Ba task này không chạm vào một phép tính nào của model. Chỉ sửa chữ và cách log.
Rủi ro bằng 0. Nếu chỉ duyệt một nhóm, duyệt nhóm này.

## TASK 1 — Đảo lại hướng đọc α

### Lỗi đang sửa — LỖI 1 (bằng chứng đầy đủ ở TASK 0)

Chú thích hiện tại ở **hai chỗ** đều ghi ngược:

- `src/uavcl/models/memory.py:65` → `α_t = cổng quên = sigmoid(to_decay_factor) ∈ (0,1)`
- `scripts/compare_all.py:118` → `alpha(forget-gate)~1 = không quên = hướng nổ`

**Tên biến đúng** (nó đúng là forget-gate), **nhưng hướng đọc ngược 180°**.

### Cách sửa

**File 1** — `src/uavcl/models/memory.py:65`:

```python
# CŨ:
self._alpha_sum = 0.0; self._alpha_cnt = 0  # α_t = cổng quên = sigmoid(to_decay_factor) ∈ (0,1)

# MỚI:
# α_t = decay_factor = sigmoid(to_decay_factor) ∈ (0,1). Hệ số GIỮ LẠI = (1 − α_t)
# — xem neural_memory.py:813 `assoc_scan(1. - decay_factor, ...)`.
#   α → 1: xoá sạch tích luỹ mỗi chunk (memory KHÔNG tích luỹ, norm phẳng ~54)
#   α → 0: giữ 100%, không quên gì  → HƯỚNG NỔ norm (bằng chứng: m3_s1 α=0 → norm 1.75e6)
self._alpha_sum = 0.0; self._alpha_cnt = 0
```

**File 2** — `scripts/compare_all.py:118`:

```python
# CŨ:  "alpha(forget-gate)~1 = không quên = hướng nổ"
# MỚI: "alpha=decay, giữ lại=(1−alpha) · alpha~0 = không quên = hướng NỔ ·
#       alpha~1 = xoá sạch mỗi chunk, memory KHÔNG tích luỹ"
```

**Rủi ro:** không có, chỉ chú thích và chuỗi in · **Thời gian:** 5 phút

## TASK 2 — Log η THẬT thay vì logit thô

### Lỗi đang sửa — LỖI 2: đầu dò η gắn sai chỗ

Trong `neural_memory.py`:

```python
# dòng 629-630
adaptive_lr = self.to_adaptive_step(seq)                   # <- probe hook Ở ĐÂY (logit thô)
adaptive_lr = self.adaptive_step_transform(adaptive_lr)    # <- transform THẬT ở đây

# dòng 255-256
def default_adaptive_step_transform(adaptive_step, max_lr = 1e-2):
    return adaptive_step.sigmoid() * max_lr
```

Probe (`memory.py:78-79`) gắn vào `to_adaptive_step`, tức **trước** transform.
Nên `eta = 55.6201` trong log **không phải learning rate**, mà là logit thô.
η thật = `sigmoid(55.62) × 0.01` = **0.01 = đúng bằng trần `max_lr`**.

Quy đổi lại toàn bộ:

| Run | η log (thô) | η THẬT | Trạng thái |
|---|---:|---:|---|
| `m3_s1` | 59.674 | 1.00e-02 | Bão hoà trần |
| `sdc_s0` | 18.415 | 1.00e-02 | Bão hoà trần |
| `latreplay_s2` | 14.734 | 1.00e-02 | Bão hoà trần |
| `cosine_s2` | 13.143 | 1.00e-02 | Bão hoà trần |
| `sdc_s1` | −5.960 | 2.57e-05 | Bão hoà đáy |
| `cosine_s0` | −5.229 | 5.33e-05 | Bão hoà đáy |
| `rebuild_s0` | −4.137 | 1.57e-04 | Gần đáy |
| `latreplay_s0` | −0.101 | 4.75e-03 | **Lành mạnh** |
| `adamw_s0` | −0.810 | 3.08e-03 | **Lành mạnh** |

**9/13 run có η bão hoà.** Chỉ `latreplay_s0` và `adamw_s0` còn ở vùng giữa,
tức còn phản ứng theo dữ liệu.

### Cách sửa

`src/uavcl/models/memory.py:70-73` — giữ nguyên tổng logit thô (để so run cũ),
**cộng thêm** bộ đếm η thật:

```python
def _eta_hook(_m, _inp, out):
    t = out[0] if isinstance(out, tuple) else out
    if torch.is_tensor(t) and t.numel() > 0:
        d = t.detach().float()
        self._eta_sum      += float(d.mean())                              # logit thô (giữ để so run cũ)
        self._eta_real_sum += float(d.sigmoid().mean() * self._eta_max_lr) # η THẬT
        self._eta_cnt += 1
```

`self._eta_max_lr` đọc từ `mem.adaptive_step_transform` nếu lấy được, không thì mặc định `1e-2`.

Đổi dòng log `methods.py:263` thành:
`eta_raw=... eta_real=... alpha=... keep=(1-alpha)`

**Rủi ro:** không đổi hành vi train. **Nhưng đổi định dạng dòng log** → phải cập nhật
regex `compare_all.py:99` đọc được cả định dạng cũ lẫn mới (dùng optional group).
**Thời gian:** 20 phút

## TASK 3 — Tự động cảnh báo khi cổng bão hoà

### Lỗi đang sửa — cả LỖI 1 và LỖI 2 lẽ ra đã bị bắt từ tháng trước

Cả hai lỗi tồn tại được lâu vì η/α chỉ là **hai con số in ra**, không ai biết ngưỡng nào
là bất thường. Task này biến chúng thành cảnh báo không thể bỏ qua.

### Cách sửa

`src/uavcl/methods.py`, khối `end_task` của class `Titans`, sau khi in η/α:

```python
if alpha is not None and (alpha > 0.99 or alpha < 0.01):
    print(f"[titans]   ⚠️ CỔNG QUÊN BÃO HOÀ (α={alpha:.4f}) — Eq 76 không còn phụ thuộc dữ liệu")
if eta_real is not None and not (1e-4 <= eta_real <= 9e-3):
    print(f"[titans]   ⚠️ η BÃO HOÀ (η={eta_real:.2e}) — ngoài vùng lành mạnh [1e-4, 9e-3]")
```

**Rủi ro:** không, chỉ thêm dòng in · **Thời gian:** 15 phút

---

# NHÓM 2 — Sửa cơ chế (CÓ đổi hành vi, nằm sau cờ config)

## TASK 4 — `pre_norm`: chuẩn hoá đầu vào memory ⭐ TASK QUAN TRỌNG NHẤT

### Lỗi đang sửa — NGUYÊN NHÂN GỐC của cả LỖI 1 và LỖI 2

Hai cổng đều là `nn.Linear(dim, heads)` áp thẳng lên feature ViT thô:

```python
self.to_adaptive_step = Sequential(nn.Linear(dim, heads), ...)   # neural_memory.py:451
self.to_decay_factor  = Sequential(nn.Linear(dim, heads), ...)   # neural_memory.py:514
```

Trong `titans_head.py:114` chỉ có `post_norm = nn.LayerNorm(dim)` — chuẩn hoá **SAU** memory.
**Không có chuẩn hoá nào TRƯỚC.** Feature thô → logit cỡ ±50 → **cả hai sigmoid bão hoà**.

`qk_rmsnorm: true` **không cứu được**: nó chỉ chuẩn hoá query/key trong đường đọc/ghi,
không chạm tới hai cổng này.

**Đây chính xác là chỗ Nested Learning bị vô hiệu hoá.** NL Eq 76 nói η_t và α_t
**phụ thuộc dữ liệu** (`x_t·W_η`, `x_t·W_α`). Sigmoid bão hoà → đạo hàm ≈ 0, đầu ra là
hằng số → **sự phụ thuộc dữ liệu đã chết**. Cơ chế cốt lõi của paper không chạy.

### Cách sửa

**File:** `src/uavcl/models/titans_head.py`

**Thay đổi 1** — khai báo, cạnh `post_norm` (dòng ~114):

```python
self.post_norm = nn.LayerNorm(dim)
# MỚI: chuẩn hoá TRƯỚC memory để to_adaptive_step/to_decay_factor không bão hoà (Eq 76).
self.pre_norm = nn.LayerNorm(dim) if bool(memory_cfg.get("pre_norm", False)) else None
```

**Thay đổi 2** — trong `features_from_extracted` (dòng ~153):

```python
seq = self.adapter(feats)                       # (1, L, D)
seq_in = self.pre_norm(seq) if self.pre_norm is not None else seq
# ... mọi lời gọi self.memory(...) dùng seq_in thay cho seq
# residual ở dòng cuối GIỮ NGUYÊN seq thô:
return self.post_norm(self.adapter.restore(out) + self.adapter.restore(seq))
```

### Quyết định thiết kế cần bạn duyệt

Residual nên dùng `seq` thô hay `seq_in` đã chuẩn hoá?

- **Đề xuất: giữ `seq` thô.** Thay đổi tối thiểu, đường tín hiệu gốc không bị đụng,
  `post_norm` vốn đã xử lý chênh lệch thang đo. Chỉ *đầu vào memory* được chuẩn hoá.
- Phương án khác (dùng `seq_in` cho cả residual) nhất quán hơn về thang đo nhưng thay đổi
  nhiều hơn — để làm ablation về sau.

### Config & test

- Thêm `memory.pre_norm: true` vào một config **MỚI**, không sửa config cũ.
- **Test mới:** `tests/test_pre_norm_gates.py`
  - `pre_norm=False` → output trùng khít bản hiện tại (bảo vệ bất biến ngược)
  - `pre_norm=True` → sau 1 forward, α ∈ [0.05, 0.95] và η thật ∈ [1e-4, 9e-3]

**Rủi ro:** **TRUNG BÌNH.** Thêm tham số học được (2×`dim` = 768 float — không đáng kể)
và đổi phân phối đầu vào memory. Mặc định tắt nên không run nào bị ảnh hưởng.
**Lùi lại:** bỏ cờ khỏi config, không cần revert code · **Thời gian:** 45 phút

## TASK 5 — Khởi tạo cổng trung tính (bổ trợ, chỉ làm nếu Task 4 chưa đủ)

### Lỗi đang sửa — cùng nguyên nhân gốc, dùng cơ chế có sẵn của thư viện

Thư viện **đã hỗ trợ sẵn** đúng việc này (`neural_memory.py:526-540`):

```python
if exists(init_adaptive_step_bias):
    linear = self.to_adaptive_step[0]
    nn.init.zeros_(linear.weight)
    nn.init.constant_(linear.bias, init_adaptive_step_bias)

if exists(init_decay_bias):
    linear = self.to_decay_factor[0]
    nn.init.zeros_(linear.weight)
    nn.init.constant_(linear.bias, init_decay_bias)
```

**Dự án chưa dùng bao giờ.** Hai tham số này tồn tại đúng để phòng bão hoà.

### Cách sửa

Thêm hai key vào `_STABILITY_KEYS` (`titans_head.py:54`) dưới dạng **số thực**
(hiện `_STABILITY_KEYS` chỉ xử lý bool — cần nhánh riêng cho float),
truyền xuống khi config có mặt.

**Giá trị đề xuất:**

- `init_decay_bias: 0.0` → α khởi đầu = 0.5, trung tính
- `init_adaptive_step_bias: 0.0` → η khởi đầu = 0.5 × max_lr = 5e-3, giữa vùng lành mạnh

**Lưu ý:** trọng số init về 0 nghĩa là hai cổng khởi đầu **hằng số**, rồi mới học dần thành
phụ thuộc dữ liệu. Đây là cách của chính tác giả thư viện.

**Cảnh báo:** cách này **không tự nó ngăn bão hoà về sau** — nếu feature vẫn lớn, trọng số
sẽ lớn dần và bão hoà lại. Task 4 là thuốc chính, task này là bổ trợ.

**Rủi ro:** THẤP, chỉ có tác dụng khi config khai báo · **Thời gian:** 20 phút
**Điều kiện làm:** chỉ khi smoke của Task 4 vẫn thấy bão hoà.

---

# NHÓM 3 — Bật CMS (không sửa code)

## TASK 6 — Config Titans + CMS

### Lỗi đang sửa — LỖI 3: CMS chưa từng được bật

Bản sửa tần số per-tier ở `cms_optimizer.py:147-169` **đã có trong code và viết đúng**:

```python
"frequency": max(1, (base_m3_frequency + int(g["period"]) - 1) // int(g["period"]))
```

(Đây là bản sửa cho bẫy `p=64 × f=16 → 1024 batch mới có một lần cập nhật M2`.)

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
đóng góp thứ 3 của NL — **hoàn toàn vắng mặt** trong mọi kết quả gần đây.

**Không cần sửa code — chỉ cần một config.**

### Cách sửa

**File mới:** `configs/g2_titans_resisc45_selfmod_m3_cms.yaml`
= bản sao config hiện tại + thêm:

```yaml
train:
  cms:
    enabled: true
    m3_frequency_unit: global_step   # bản sửa per-tier đã có sẵn
    grad_agg: sum                    # nguyên văn Eq 71 của paper
    eta_mode: fixed                  # để adaptive cho ablation sau
```

### Xác minh bắt buộc khi chạy — phải thấy trong log

1. Dòng `[cms] M3 frequency unit=global_step | per-tier f=[...]`
2. `tier_report` in ra bảng phân tier
3. `‖Δw‖` per-tier: **tier chậm ≈ 0, tier nhanh lớn**

**Nếu (3) ngược lại → mapping tier sai → DỪNG, sửa trước khi tin số nào.**

**Rủi ro:** THẤP về code (không sửa gì), **CAO về kết quả** — đây là lần đầu CMS chạy
cùng Titans, chưa ai biết nó cư xử ra sao · **Thời gian:** 20 phút viết config

**Ghi chú:** chạy **sau** khi Task 4 xong. Bật CMS trong lúc cổng còn bão hoà chỉ tạo
thêm một biến không kiểm soát được.

---

# NHÓM 4 — Kiểm chứng (bắt buộc, không được bỏ)

## TASK 7 — `pytest -q` phải xanh

- Gồm test mới của Task 4 + toàn bộ 21 file test hiện có.
- **Đặc biệt kiểm:** `test_g2_titans.py`, `test_self_modifying_memory.py`, `test_g3_cms.py`
  — ba file dễ vỡ nhất bởi các thay đổi trên.
- **Chặn:** đỏ → dừng, không chạy gì tiếp · **Thời gian:** 15 phút

## TASK 8 — Smoke 1 task, CHỈ đọc cổng, KHÔNG quan tâm accuracy ⭐ CỬA CHẶN

**Chạy:** 1 task, 1 epoch, `memory.pre_norm=true`, `log.dir=./artifacts_smoke_gates`

**Tiêu chí ĐẠT — cả bốn phải thoả:**

| Chỉ số | Ngưỡng |
|---|---|
| α mỗi task | trong `[0.05, 0.95]` |
| η thật mỗi task | trong `[1e-4, 9e-3]` |
| α **biến thiên** giữa các task | `max − min ≥ 0.05` |
| `norm(state)` | `< 1e4` |

**Ba tiêu chí đầu quan trọng hơn tiêu chí thứ tư.** Cổng nằm trong khoảng nhưng **đứng yên**
vẫn là hỏng — Eq 76 đòi hỏi η/α **phụ thuộc dữ liệu**, tức phải thay đổi theo task.

**Chưa đạt → quay lại Task 5, hoặc xem lại kết luận Task 0.**
**Thời gian:** ~30 phút

## TASK 9 — Chạy lại Titans 3 seed, so 1-biến

- **Chỉ chạy sau khi Task 8 ĐẠT.**
- So `pre_norm=true` với `pre_norm=false` trên **cùng seed, cùng mọi thứ khác**.
- **Đây mới là lần đầu tiên Nested Learning thực sự được kiểm chứng trong dự án này.**
- Mốc phải vượt: Titans+NCM online cũ `0.6190` · Frozen NCM `0.7114` · SLDA `0.8263`
- **Thời gian:** theo thời gian chạy VM

---

# Thứ tự thực hiện

```
TASK 0  (miễn phí, quyết định hướng Task 4)
   ↓
TASK 1, 2, 3  (rủi ro 0 — có thể duyệt riêng nhóm này)
   ↓
TASK 4  (pre_norm, sau cờ config)
   ↓
TASK 7  (pytest)  →  TASK 8  (smoke, CỬA CHẶN)
   ↓                     ↓ chưa đạt
TASK 9  (3 seed)      TASK 5  (init bias) → quay lại TASK 8
   ↓
TASK 6  (bật CMS)  →  chạy lại
   ↓
Thí nghiệm quyết định: SLDA + memory.enabled=true
```

**Lý do SLDA + Titans phải nằm CUỐI:** chạy nó bây giờ, dù kết quả ra sao, bạn cũng không
phân biệt được **"Titans vô dụng"** với **"Titans đang tê liệt"**. Chỉ sau khi hai cổng
sống lại thì kết quả đó mới kết luận được điều gì.

---

# Bảng quyết định — duyệt từng nhóm riêng được

| Nhóm | Task | Lỗi sửa | Đổi hành vi? | Rủi ro | Công sức | Duyệt? |
|---|---|---|---|---|---|---|
| 0 | 0 | Lỗi 1 (xác nhận) | Không | Không | 30 ph | ☐ |
| 1 | 1 | Lỗi 1 | Không | Không | 5 ph | ☐ |
| 1 | 2 | Lỗi 2 | Không | Không | 20 ph | ☐ |
| 1 | 3 | Lỗi 1 + 2 (phòng tái phát) | Không | Không | 15 ph | ☐ |
| 2 | 4 | Nguyên nhân gốc | Có, sau cờ | Trung bình | 45 ph | ☐ |
| 2 | 5 | Nguyên nhân gốc (bổ trợ) | Có, sau cờ | Thấp | 20 ph | ☐ |
| 3 | 6 | Lỗi 3 | Chỉ config | Thấp code / cao kết quả | 20 ph | ☐ |
| 4 | 7, 8, 9 | — | Không | Không | 45 ph + chạy | ☐ |

**Khuyến nghị tối thiểu: duyệt Nhóm 0 + Nhóm 1.** Không đổi một phép tính nào, nhưng đủ để
xác nhận chẩn đoán bằng dữ liệu đã có và ngăn lỗi đọc ngược lặp lại.
