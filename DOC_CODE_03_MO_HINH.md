# Đọc code — Bài 3: Mô hình, optimizer, và bài toán UAV thật

> Đọc sau **Bài 1** (nền tảng) và **Bài 2** (engine + methods).

**Thứ tự đọc trong bài này** — chia 3 nhóm, đọc hết nhóm này rồi sang nhóm khác:

| Nhóm | File | Dòng | Vì sao thứ tự này |
|---|---|---:|---|
| **I. Gradient-free** | `ncm.py` → `slda.py` | 63 → 408 | NCM 63 dòng là bản rút gọn của SLDA. Hiểu NCM rồi đọc SLDA rất nhanh |
| **II. Titans + CMS** | `seq_adapter.py` → `memory.py` → `titans_head.py` → `cms.py` → `cms_optimizer.py` → `hope.py` → `m3.py` | 1.019 | theo đúng đường đi của dữ liệu |
| **III. Bài toán UAV thật** ⭐ | `drift.py` → `revisit.py` → `metrics/revisit.py` → `tang_nhanh.py` → `ngan_hang_che_do.py` → `slda.hap_thu_khong_nhan` | 696 | phần đang chạy, và có nhiều bài học phương pháp nhất |

**Nếu chỉ có thời gian cho một nhóm, chọn nhóm III.**

---

# NHÓM I — Gradient-free

## I.1. `models/ncm.py` (63 dòng) — đọc file này đầu tiên

**Nearest Class Mean.** RESISC45: acc 0.693, forgetting 0.100, **17K float**. Đứng nhì bảng G1 với 63 dòng code và 0 tham số train.

### `__init__`

```python
def __init__(self, backbone, feat_dim, num_classes):
    self.backbone = backbone
    for p in self.backbone.parameters():
        p.requires_grad_(False)          # NCM định nghĩa trên đặc trưng CỐ ĐỊNH
    self.backbone.eval()
    self.register_buffer("proto_sum",   torch.zeros(num_classes, feat_dim))
    self.register_buffer("proto_count", torch.zeros(num_classes))
```

**`register_buffer` chứ không phải `nn.Parameter`.** Khác biệt quyết định:

| | `nn.Parameter` | `register_buffer` |
|---|---|---|
| optimizer cập nhật | ✓ | ✗ |
| có trong `state_dict()` | ✓ | ✓ |
| `.to(device)` mang theo | ✓ | ✓ |

Prototype cần được lưu và chuyển device, nhưng **tuyệt đối không** được optimizer đụng vào. Buffer là đúng công cụ.

### `train(mode)` — ghi đè có chủ đích

```python
def train(self, mode: bool = True):
    super().train(mode)
    self.backbone.eval()      # ⭐ ép backbone luôn eval
    return self
```

Nếu không ghi đè, `model.train()` sẽ bật BatchNorm cập nhật thống kê chạy → feature **trôi** dù không có gradient. Prototype tính ở task 0 sẽ không còn khớp với feature ở task 5. Ba dòng này giữ toàn bộ tính đúng đắn của NCM.

### `update_prototypes()` — toàn bộ "học" của NCM

```python
@torch.no_grad()
def update_prototypes(self, feats, ys):
    feats = F.normalize(feats.detach().float(), dim=1)
    self.proto_sum.index_add_(0, ys, feats)
    self.proto_count.index_add_(0, ys, torch.ones_like(ys, dtype=torch.float))
```

**3 dòng.** Không optimizer, không loss, không backward.

**`index_add_(0, ys, feats)`** — với mỗi hàng `i` của `feats`, cộng vào hàng `ys[i]` của `proto_sum`. Thay cho vòng lặp Python, nhanh hơn nhiều lần và chạy trên GPU.

**`F.normalize` trước khi cộng** — mỗi ảnh đóng góp một vector **đơn vị**. Không chuẩn hoá thì ảnh có feature norm lớn sẽ lấn át.

### `prototypes()` và `forward()`

```python
def prototypes(self):
    cnt = self.proto_count.clamp(min=1.0).unsqueeze(1)   # clamp tránh chia 0
    return F.normalize(self.proto_sum / cnt, dim=1)      # lớp chưa học -> vector 0

def forward(self, x):
    feats = F.normalize(self.backbone(x).float(), dim=1)
    return feats @ self.prototypes().t()      # cosine (B, C) — DÙNG NHƯ LOGITS
```

**Dòng cuối là chỗ NCM cắm vừa vào dự án.** Vì cả hai vế đã chuẩn hoá, tích vô hướng **chính là** cosine. Shape `(B, C)` giống logits → `mask_logits`, `evaluate`, `metrics` dùng chung **không sửa gì**.

**Vì sao NCM gần như không quên:** prototype lớp cũ **không bao giờ bị ghi đè** khi học lớp mới — mỗi lớp có hàng riêng trong `proto_sum`. Không có cơ chế nào để lớp mới làm hỏng lớp cũ.

> Vai trò trong dự án (docstring): *"Nếu Titans (G2) / CMS (G3) không thắng nổi baseline ngây thơ này thì chưa có gì để báo cáo."* **Mốc dưới không được phép thua.**

---

## I.2. `models/slda.py` (408 dòng) — model chủ lực hiện tại ⭐

**Streaming LDA** (Hayes & Kanan 2020). Như NCM nhưng học thêm **hiệp phương sai chung** → hơn NCM **+11,5 điểm**.

### Toán (ghi ở docstring đầu file)

```
tích luỹ:  s_c += f          (tổng feature theo lớp)
           n_c += 1
           G   += f fᵀ       (tổng outer product TOÀN CỤC)

suy ra:    μ_c = s_c / n_c
           Σ_w = (G − Σ_c n_c μ_c μ_cᵀ) / N        (within-class covariance)
           Λ   = (Σ_w + ε I)⁻¹                      (shrinkage ε)

dự đoán:   score_c(f) = (Λ μ_c)ᵀ f − ½ μ_cᵀ Λ μ_c
```

**Khác NCM chỗ nào về mặt hình học:** NCM đo khoảng cách tới tâm lớp. SLDA đo khoảng cách **sau khi kéo giãn không gian** theo `Λ` — chiều nào feature dao động nhiều thì bị nén lại, chiều nào ổn định thì được khuếch đại. Ranh giới lớp xét cả "hình dạng" đám mây feature.

### Các buffer — và vì sao có **hai** bộ đếm

```python
self.register_buffer("feat_sum", torch.zeros(num_classes, feat_dim, dtype=dt))  # s_c
self.register_buffer("count",    torch.zeros(num_classes, dtype=dt))            # n_c
self.register_buffer("gram",     torch.zeros(feat_dim, feat_dim, dtype=dt))     # G
# --- BỘ ĐẾM THỨ HAI, trên ĐỒNG HỒ TOÀN CỤC ---
self.register_buffer("feat_sum_g", torch.zeros(num_classes, feat_dim, dtype=dt))
self.register_buffer("count_g",    torch.zeros(num_classes, dtype=dt))
self.register_buffer("count_raw",  torch.zeros(num_classes, dtype=dt))  # chỉ CHẨN ĐOÁN
```

**Đây là chỗ khó nhất file, và đáng đọc kỹ nhất.** Comment giải thích:

> Đẳng thức `Σ_w = (G − Σ_c n_c μ_c μ_cᵀ)/N` chỉ đúng khi `G` và `(n_c, s_c)` được **đánh trọng số giống hệt nhau**. Nhưng thiết kế cố ý dùng **hai đồng hồ khác nhau** (G toàn cục, s_c theo-lớp) → trọng số lệch → phép trừ **trừ quá tay** → `Σ_w` mất tính xác định dương → `Λ = inv(Σ)` thành rác.

**Vì sao lại cố ý dùng hai đồng hồ?** Vì λ phải phân rã khác nhau ở hai chỗ:

| Thống kê | Phân rã theo | Lý do |
|---|---|---|
| `gram`, `feat_sum_g`, `count_g` | **đồng hồ toàn cục** (mọi batch) | Σ dùng chung cho mọi lớp, "tuổi" tính theo tổng mẫu |
| `feat_sum`, `count` (μ_c) | **theo-lớp** (chỉ khi lớp c xuất hiện) | phân rã toàn cục sẽ **xoá sạch lớp hiếm** → mất điểm mạnh nhất của SLDA |

Giải pháp: giữ **thêm** một cặp đếm nhỏ theo đồng hồ toàn cục, dùng **riêng** cho đẳng thức Σ. Giá: `C·D + C` ≈ 0,14 MB (so với gram 1,18 MB).

> **Test bắt đúng lỗi này:** `test_lop_vang_mat_van_du_doan_dung` — lớp 0 vắng 1600 mẫu, `G` chỉ còn giữ `e^-1.6 ≈ 20%` đóng góp của nó, mà `between` vẫn trừ đi 100%.

**`stats_dtype="float64"` mặc định** — cộng dồn hàng chục nghìn mẫu rồi **nghịch đảo ma trận**; float32 lệch theo thứ tự cộng (streaming ≠ batch tới 1e-3). Docstring cũng ghi ràng buộc triển khai: **MPS không hỗ trợ float64** → hai cách xử lý, đều hợp lệ, đều ghi rõ.

### `update()` — hấp thụ một batch

```python
@torch.no_grad()
def update(self, feats, ys):
    if self.tang_nhanh is not None:                      # BA TẦNG
        self.tang_nhanh.cap_nhat(feats.detach().float()) #   tầng nhanh nuốt feature THÔ
        feats = self.tang_nhanh.can_chinh(feats.detach().float())  # tầng chậm nhận feature ĐÃ CĂN
    f = feats.detach().to(self.feat_sum.dtype)
    if not torch.isfinite(f).all():
        raise FloatingPointError("SLDA nhận feature chứa NaN/Inf")

    # --- Σ: phân rã theo ĐỒNG HỒ TOÀN CỤC ---
    if self.decay_cov < 1.0:
        r = self.decay_cov ** n_batch
        self.gram.mul_(r); self.feat_sum_g.mul_(r); self.count_g.mul_(r)   # ⭐ CẢ BA cùng hệ số
    self.gram.add_(f.t() @ f)
    self.feat_sum_g.index_add_(0, ys, f)
    self.count_g.index_add_(0, ys, torch.ones_like(ys, dtype=self.count_g.dtype))

    # --- μ_c: phân rã CHỈ Ở LỚP CÓ MẶT trong batch ---
    if self.decay_mean < 1.0:
        dem = torch.zeros_like(self.count).index_add_(0, ys, torch.ones_like(...))
        he_so = torch.where(dem > 0, decay_mean ** dem, torch.ones_like(self.count))
        self.feat_sum.mul_(he_so.unsqueeze(1)); self.count.mul_(he_so)
    self.feat_sum.index_add_(0, ys, f)
    self.count.index_add_(0, ys, ...)
    self.count_raw.index_add_(0, ys, ...)      # không bao giờ phân rã — chỉ chẩn đoán
    self._version += 1
```

**`torch.where(dem > 0, λ**dem, 1)`** — chỉ lớp **có mặt** trong batch mới bị phân rã. Đây là dòng thực thi nguyên tắc "μ_c chỉ bám theo những lần c thực sự xuất hiện".

**`f.t() @ f`** — tính `Σ f fᵀ` cho cả batch bằng một phép nhân ma trận, thay vì vòng lặp.

### `_within_class_sigma()`

```python
n = self.count_g                                # ⭐ CHỈ dùng cặp đếm TOÀN CỤC
mu = self.feat_sum_g / n.clamp(min=1e-12).unsqueeze(1)
between = self.feat_sum_g.t() @ mu              # ≡ Σ_c n_c μ_c μ_cᵀ, ổn định hơn
sigma = (self.gram - between) / max(float(n.sum()), 1.0)
return 0.5 * (sigma + sigma.t())                # ép đối xứng (bù sai số float)
```

**`clamp(min=1e-12)` chứ không phải `min=1.0`** — comment giải thích: có phân rã thì `n_c` hợp lệ vẫn có thể `< 1`. Kẹp về 1.0 sẽ làm sai μ.

**`0.5*(σ + σᵀ)`** — về toán σ phải đối xứng, nhưng sai số dấu phẩy động phá vỡ điều đó, và `torch.linalg.inv` trên ma trận không đối xứng cho kết quả kém ổn định. Một dòng, rẻ, và tránh được một lớp lỗi khó tìm.

### `_refresh_cache()` — lazy, và `cov_mode`

```python
if self._cache_version == self._version:
    return                                        # chưa có update mới -> khỏi tính lại
mu = self.feat_sum / self.count.clamp(min=1.0).unsqueeze(1)
eye = torch.eye(d, ...)
if self.cov_mode == "identity":
    lam = eye / (1.0 + self.shrinkage)            # Σ = I -> ĐÚNG BẰNG NCM
else:
    sigma = (self._frozen_sigma if cov_mode=="frozen" and có sẵn
             else self._within_class_sigma())
    lam = torch.linalg.inv(sigma + self.shrinkage * eye)
w = lam @ mu.t()                                  # (D, C)
b = -0.5 * (mu * (mu @ lam)).sum(dim=1)           # (C,)
empty = self.count < 1.0
w[:, empty] = 0.0; b[empty] = 0.0                 # lớp chưa có mẫu -> score 0
```

**Cache theo `_version`** — nghịch đảo ma trận 384×384 mỗi batch thì quá đắt. Chỉ tính lại khi có `update()` mới. `forward()` gọi `_refresh_cache()` mỗi lần nhưng thường trả về ngay.

**Ba `cov_mode` — thiết kế ablation đẹp:**

| mode | Σ | Tương đương | Trả lời câu hỏi |
|---|---|---|---|
| `streaming` | cập nhật liên tục | — | bản gốc |
| `identity` | ép `= I` | **đúng bằng NCM** | +11,5 điểm đến từ hiệp phương sai? |
| `frozen` | tính 1 lần rồi đóng băng | — | Σ có cần cập nhật liên tục không? |

Docstring nói rõ vì sao `identity` là đối chứng sạch: nó chạy **qua y hệt đường code này**, nên chênh lệch so với `streaming` **chính là** đóng góp của hiệp phương sai, không lẫn bất kỳ khác biệt cài đặt nào khác.

Chứng minh tương đương ghi thẳng trong docstring:
```
Σ = I → Λ = I/(1+ε), và argmax_c (μ_c·f − ½‖μ_c‖²) ≡ argmin_c ‖f − μ_c‖²
```

### `window_report()` — bằng chứng λ chạy đúng

```python
m = float(raw.mean())                     # số lần gặp trung bình mỗi lớp (count_raw)
if lam < 1.0:
    ky_vong_huu_han = (1.0 - lam**m) / (1.0 - lam)     # ⭐ mốc ĐÚNG
else:
    ky_vong_huu_han = m
return {..., "ky_vong": 1/(1-lam),                    # tiệm cận (chỉ tham khảo)
             "ky_vong_huu_han": ky_vong_huu_han,      # mốc để SO
             "ty_le_giu": ky_vong_huu_han / m}
```

**Bài học đáng ghi nhớ nhất trong file:** `n_c` chỉ hội tụ về tiệm cận `1/(1−λ)` sau **hàng nghìn** lần gặp. RESISC45 chỉ cho **420 lần/lớp** trong cả 9 task → so với tiệm cận là **so nhầm mốc** (đo 131 mà "kỳ vọng" in ra 1000 → tưởng hỏng). Mốc đúng là tổng cấp số nhân **hữu hạn** `(1 − λ^m)/(1 − λ)`.

`methods.SLDA.end_task` dùng số này để cảnh báo:
```python
lech = abs(w["n_c_trung_binh"] - w["ky_vong_huu_han"]) / max(w["ky_vong_huu_han"], 1e-9)
if lech > 0.05:
    print("⚠️ n_c lệch ... — nghi cài sai chỗ phân rã")
if w["ty_le_giu"] > 0.90:
    print("⚠️ giữ >90% trí nhớ — λ_μ quá gần 1 so với cỡ dữ liệu, arm này sẽ gần trùng λ=1")
```

**Cảnh báo thứ hai đặc biệt hay:** nó báo trước rằng một arm thí nghiệm sẽ **không cho thông tin gì**, trước khi bạn đốt 8 tiếng chạy nó.

---

# NHÓM II — Titans, CMS, HOPE

## II.1. `models/seq_adapter.py` (61 dòng) — đọc đầu tiên trong nhóm

Titans chỉ ăn chuỗi `(1, L, D)`. Adapter quyết định **"thời gian" nghĩa là gì**.

```python
def forward(self, feats):
    if self.mode == "image_seq":
        if feats.dim() == 3:                    # (B,P,D) -> gộp token
            feats = feats.mean(dim=1)
        self._last_bp = (feats.shape[0], 1)
        return feats.unsqueeze(0)               # (1, B, D) — B bước thời gian
    B, P, D = feats.shape                       # token_seq
    self._last_bp = (B, P)
    return feats.reshape(1, B*P, D)             # chuỗi dài B*P

def restore(self, seq_out):
    B, P = self._last_bp
    out = seq_out.squeeze(0)                    # (L, D)
    if P == 1: return out                       # image_seq: L == B
    return out.reshape(B, P, -1).mean(dim=1)    # token_seq: gấp lại + trung bình patch
```

| mode | 1 bước thời gian = | Ý nghĩa |
|---|---|---|
| `image_seq` (mặc định) | **1 ảnh** | chuỗi = "từng lần UAV nhìn thấy cảnh" |
| `token_seq` | **1 patch** | chi tiết trong ảnh, nhưng ý nghĩa "thời gian stream" yếu hơn |

**`self._last_bp` là state ẩn.** `forward` ghi, `restore` đọc. Bắt buộc gọi theo cặp, đúng thứ tự. Đây là kiểu thiết kế cần cẩn thận — nhưng ở đây chấp nhận được vì hai hàm luôn được gọi liền nhau trong `features_from_extracted`.

## II.2. `models/memory.py` (281 dòng) — bọc thư viện + chẩn đoán cổng

**Nhiệm vụ 1 — bọc `titans_pytorch.NeuralMemory` sau MỘT chữ ký cố định:**

```python
out, state = memory(seq, state=None)      # (1,L,D) -> (1,L,D)
```

Vì sao: cả dự án chỉ phụ thuộc chữ ký này. Thư viện đổi version → sửa **một** file.

**Nhiệm vụ 2 — chẩn đoán hai cổng.** Đây mới là phần lớn code, và là phần đáng đọc.

`NeuralMemory` có hai cổng học được:

| Cổng | Là gì | Bão hoà thì sao |
|---|---|---|
| **η** (eta) | tốc độ **ghi** — learning rate của memory | ngoài dải lành mạnh = Eq 76 hết phụ thuộc dữ liệu |
| **α** (alpha) | tốc độ **quên** (`decay_factor`), giữ lại = `1−α` | α→1: xoá sạch mỗi chunk, memory không tích luỹ<br>α→0: không quên gì → **NỔ** norm (đã đo: 1.75e6) |

Bốn phương thức xử lý:

```python
def _detect_eta_max_lr(self):
    """Đọc max_lr THẬT từ `adaptive_step_transform` (functools.partial) — fallback 1e-2."""
def _install_gate_bounds(self, cfg):
    """Chặn logit của HAI cổng để chúng không trôi ra biên (Eq 76 phải sống)."""
def _install_eta_alpha_probes(self):
    """Gắn forward-hook lên to_adaptive_step (η) và to_decay_factor (α) để ghi trung bình."""
def eta_alpha_stats(self):
    """(η logit thô, η THẬT, α) trung bình kể từ lần reset."""
```

**`_detect_eta_max_lr` là một bài học đắt.** Comment trong `methods.py`:

> `max_lr` thật của thư viện là **1.0** (`neural_memory.py:272`), **không phải 1e-2** như default của hàm transform gợi ý.

Vì hardcode nhầm thang 100×, ngưỡng cảnh báo bão hoà sai, và lỗi **sống sót trọn một ngày**. Bản vá: đọc giá trị thật từ `functools.partial` thay vì giả định.

**`_install_eta_alpha_probes` đăng ký SAU `_install_gate_bounds`** — vì PyTorch gọi hook theo thứ tự đăng ký, và hook trước **có thể thay đổi output**. Đăng ký sai thứ tự thì probe đo giá trị **trước khi** bị chặn, tức đo sai.

**`gate_bound_range(ten)`** — trả khoảng `(lo, hi)` **thật** sau khi chặn:

> Tính từ (nửa, tâm) chứ **không** giả định tâm 0 — bản cũ giả định vậy nên khi dời tâm sẽ in sai.

## II.3. `models/titans_head.py` (253 dòng) — ghép lại

```
ảnh → [ViT đóng băng] → feature → [SeqAdapter] → chuỗi
    → [TitansMemory + state] → post_norm + residual → [head] → logits
```

### Ba chế độ reset — bậc thang A/B/C

```python
self.reset_mode = str(memory_cfg.get("reset", "image")).lower()   # image | task | never
```

| Chế độ | Ký ức sống | Vai trò |
|---|---|---|
| `image` (A) | mỗi batch reset | sanity check — không có trí nhớ |
| `task` (B) | trong một task | bậc trung gian |
| `never` (C) | xuyên toàn bộ stream | ⭐ **đích thật của G2** |

Đổi bằng **một dòng yaml**. Ba bậc chạy được độc lập, và `run_dir_name` thêm hậu tố `_r{reset}` để không đè kết quả nhau.

### `features_from_extracted()` — hai luật vòng đời state ⭐

```python
seq = self.adapter(feats)                                  # (1, L, D)
if self.training and self.reset_mode in ("task", "never"):
    out, new_state = self.memory(seq_in, state=self._state)   # nối tiếp state cũ, CÓ ghi
    self._state = detach_state(new_state)                     # ⭐ LUẬT 1
else:
    init = None
    if (not self.training) and self.reset_mode in ("task","never") and self._state is not None:
        init = clone_state(self._state)                       # ⭐ LUẬT 2
    out, _ = self.memory(seq_in, state=init)                  # bỏ state mới -> KHÔNG ghi
return self.post_norm(self.adapter.restore(out) + self.adapter.restore(seq))   # residual
```

**LUẬT 1 — `detach_state` sau mỗi batch.** Giữ **giá trị** ký ức nhưng **cắt** đồ thị đạo hàm. Không detach thì graph nối dài vô hạn qua mọi batch → RAM nổ và lỗi `double-backward`. Đây là truncated BPTT.

**LUẬT 2 — eval trên `clone_state`.** Chấm điểm mà vô tình sửa ký ức là kết quả vô nghĩa: ma trận R sẽ phụ thuộc **thứ tự chấm**. Clone rồi vứt bản mới đi (`out, _ = ...`).

**Residual `+ adapter.restore(seq)`** — nếu memory cho ra rác, model vẫn còn đường feature gốc. Cứu cánh khi memory chưa ổn định.

### `state_utils.py` (106 dòng) — dụng cụ đi kèm

State do thư viện trả về là cấu trúc **lồng nhau** (namedtuple / dict / tuple / `TensorDict`). Các hàm đi **đệ quy** áp phép biến đổi lên từng tensor, giữ nguyên khung:

```python
def _tree_map(fn, obj):      # áp fn lên MỌI tensor, giữ nguyên cấu trúc
def detach_state(state)      # cắt gradient, giữ giá trị
def clone_state(state)       # bản sao độc lập
def state_to_cpu(state)      # kéo về CPU trước khi torch.save
def state_norm(state)        # sqrt(Σ ‖tensor‖²) — 0.0 nếu rỗng
def count_floats(state)      # đếm float để tính footprint
```

**Bẫy đã vá, ghi trong docstring của `_is_tensordict_like`:**

> `titans-pytorch` gói weights/updates trong `tensordict.TensorDict` — **không phải dict thường** nên phải nhận diện riêng (bỏ sót nó = state còn dính graph → double-backward).

Nhờ cách viết đệ quy này, code **không phụ thuộc version cụ thể** của thư viện.

## II.4. `models/cms.py` (137 dòng) — chia ViT thành tầng tần số

**Không đụng forward của ViT.** Chỉ nhóm tham số lại.

```python
def build_cms_param_groups(model, cms_cfg) -> List[Dict]:
    tiers = cms_cfg.get("tiers", [[4,1], [4,4], [4,16]])   # [số block, chu kỳ]
    etas  = cms_cfg.get("etas",  [1.0, 0.5, 0.1])          # hệ số lr từng tier
```

**Quyết định của team, ghi ngay docstring:**

| Thành phần | Đi đâu | Vì sao |
|---|---|---|
| `patch_embed`, `pos_embed`, `cls_token`, `norm` cuối | **ĐÓNG BĂNG** | nền pretrained, không nên đụng |
| MLP của block | chia 3 tier theo `tiers` | phần "kiến thức" chia theo tần số |
| `attn` + LayerNorm mọi block | tier **CHẬM NHẤT** | giữ "cách nhìn quan hệ giữa vùng ảnh" → muốn bền |
| `head`, `memory`, `post_norm` | tier **NHANH NHẤT** | cần linh hoạt nhất |

**Cửa chặn:**
```python
if sum(t[0] for t in tiers) != n_blocks:
    raise ValueError(f"Tổng block trong cms.tiers = {...} phải bằng số block của backbone = {n_blocks}")
```
Sai số block là mọi thứ sau đó lệch âm thầm.

**`order`:**
```python
idxs = list(range(n_blocks))
if order == "early_slow":
    idxs = idxs[::-1]       # block ĐẦU rơi vào tier CHẬM
```
Ablation: block gần input nên nhanh hay chậm? Cả hai đều có lý lẽ, nên để cấu hình được.

**`tier_report(groups)`** — docstring viết hoa: *"chạy lần đầu **PHẢI đọc**"*. In ra block nào ở tier nào, bao nhiêu tham số. Kiểm tra bằng mắt trước khi tin số.

## II.5. `optim/cms_optimizer.py` (178 dòng) — CMS ở mức optimizer

**Không kế thừa `torch.optim.Optimizer`**, chỉ cần "trông giống": có `zero_grad()` + `step()`.

### `step()` — 4 giai đoạn

```python
@torch.no_grad()
def step(self):
    self.global_step += 1
    # 1) tích luỹ grad batch này cho MỌI tier
    for g in self.tiers:
        for p in g["params"]:
            if p.grad is not None:
                self._accum[id(p)].add_(p.grad)
    # 2) tier đến hạn -> grad = gộp; chưa đến hạn -> grad None
    due = [g for g in self.tiers if self.global_step % int(g["period"]) == 0]
    for g in self.tiers:
        if id(g) in due_ids:
            div = float(g["period"]) if self.grad_agg == "mean" else 1.0
            for p in g["params"]:
                p.grad = self._accum[id(p)] / div
        else:
            for p in g["params"]:
                p.grad = None                 # ⭐ inner optimizer BỎ QUA hoàn toàn
    if self.eta_mode == "adaptive":
        self._apply_adaptive_eta(due)
    # 3) inner optimizer bước — tier grad=None không đổi
    self.inner.step()
    # 4) xả bình tích luỹ của tier vừa bước
    for g in due:
        for p in g["params"]:
            self._accum[id(p)].zero_()
```

**`p.grad = None` là toàn bộ mẹo.** Optimizer PyTorch bỏ qua tham số có `grad is None` — kể cả **momentum/state của nó cũng không tiến**. Nghĩa là tier chậm thật sự **đứng yên**, không phải chỉ bước nhỏ.

**`grad_agg`:**
- `"sum"` = nguyên văn Eq. 71 của paper: `η⁽ℓ⁾·Σ gradient` trong cửa sổ
- `"mean"` = chia thêm cho `p` — cùng giá trị lr mang cùng ý nghĩa ở mọi chu kỳ

Tương đương về toán (khác hệ số η đúng `p` lần), giữ cả hai để ablate **"trung thành với paper vs ổn định thực dụng"**.

### `_apply_adaptive_eta()` — "self-modifying nhẹ"

```python
flat = torch.cat([p.grad.flatten() for p in g["params"] if p.grad is not None])
prev = self._prev_dir[idx]
if prev is None or flat.norm() < 1e-12:
    surprise = 1.0                     # chưa có lịch sử -> trung tính = hành vi fixed
else:
    cos = torch.dot(flat, prev) / (flat.norm() * prev.norm() + 1e-12)
    surprise = float(1.0 - cos.clamp(-1.0, 1.0))         # ∈ [0, 2]
self._prev_dir[idx] = flat / (flat.norm() + 1e-12)
self.inner.param_groups[idx]["lr"] = self._base_lr[idx] * surprise
```

**`surprise = 1 − cos(grad hiện tại, hướng update trước)`:**
- `surprise → 0`: gradient đồng hướng lần trước = *"đã biết rồi, ghi khẽ thôi"*
- `surprise → 2`: ngược hướng hẳn = *"bất ngờ thật, ghi mạnh hơn"*

Clip tự nhiên trong `[0, 2·η_base]` nhờ tính chất cosine — không cần kẹp tay.

### `build_cms_optimizer()` — chi tiết dễ bỏ sót

```python
"frequency": max(1, (base_m3_frequency + int(g["period"]) - 1) // int(g["period"]))
```

Quy đổi `frequency` của M3 theo chu kỳ tier. **Vì sao:** CMS đã làm tier bước thưa đi; nếu để nguyên `f=16` thì với `p=64` phải mất `64×16 = 1024` batch mới có một lần cập nhật ký ức chậm M2. **Hai lịch nhân nhau** — không ai chủ ý muốn thế.

## II.6. `models/hope.py` (57 dòng) — G4

Kế thừa `TitansClassifier`, khác đúng **2 điểm**:

```python
def __init__(self, ...):
    super().__init__(...)                     # dựng y hệt G2 (kể cả đóng băng backbone)
    for p in self.backbone.parameters():
        p.requires_grad_(True)                # KHÁC BIỆT #1: MỞ BĂNG

def train(self, mode=True):
    return nn.Module.train(self, mode)        # KHÔNG ép backbone.eval như Titans

def _extract(self, x):
    """Như TitansClassifier nhưng CÓ gradient."""
    ...                                       # KHÁC BIỆT #2: bỏ @torch.no_grad()
```

**Rủi ro đặc thù, ghi rõ trong docstring:** backbone thay đổi → không gian feature **trôi dưới chân memory**. Theo dõi bằng `norm(state)` mà method `hope` in ra sau mỗi task. Đây chính là bệnh đã đo được (**Forgetting 0.956**, norm 262→302→120…).

**Ghi chú kỹ thuật:** không ép `backbone.eval()` vì ViT **không có BatchNorm** nên không lo thống kê batch trôi. Với ResNet thì lập luận này sai — đáng nhớ nếu đổi backbone.

## II.7. `optim/m3.py` (232 dòng) — optimizer hai tầng ký ức

**Ý tưởng:** optimizer cũng là **bộ nhớ** — momentum là ký ức của gradient. M3 = Adam + Muon + CMS áp vào chính optimizer.

```
g_t = gradient
M1 = update(M1, g_t)        # ký ức NHANH        (dòng 7)
V  = update(V, g_t²)        # moment bậc 2 Adam  (dòng 8)
O1 = NewtonSchulz_T(M1)     # trực giao hoá kiểu Muon
M2 = update(M2, O1)         # ký ức CHẬM (mỗi f bước một lần, Eq. 75)
```

### `newton_schulz(m, steps=5)`

Trực giao hoá **gần đúng** ma trận: giữ "hướng", bỏ "độ lớn lệch trục". Iteration thay vì SVD — nhanh hơn nhiều bậc.

### `step()` — bốn chi tiết đáng đọc

**(a) Quy ước Muon: chỉ áp cho ma trận**
```python
if p.ndim >= 2:
    update = newton_schulz(st["m1"], steps=T) + alpha * st["o2"]
else:
    update = st["m1"]      # bias/norm 1D: Adam thuần
```
Comment: *"áp nguyên M3 lên vector gây limit-cycle — đã quan sát được trên bài toán lồi 1D"*.

**(b) `update_norm` — vì sao có, kèm bằng chứng đo được**

> Dòng 10 Algorithm 1 chia `(O1+αO2)` cho `sqrt(V)`: tử số **đã bị Newton–Schulz chuẩn hoá về ~O(1)** (vứt độ lớn gradient), mẫu số lại **~độ lớn gradient** (rất nhỏ với backbone pretrained) → bước đi khuếch đại **hàng trăm lần** lr → `‖Δw‖ ~1000%/task`, accuracy sập.

Ba lựa chọn, đều giữ lại:

| mode | Làm gì | Dùng khi |
|---|---|---|
| `clip` (mặc định) | chỉ co update khi norm > 1 | khớp implementation tham khảo |
| `rms` | ép RMS = 1 | tái lập run cũ 07-23 |
| `none` | nguyên văn Algorithm 1 | ablation trung thành với paper |

**Ba mode cùng tồn tại** thay vì "sửa cho đúng rồi xoá cái cũ" — nhờ vậy run cũ vẫn tái lập được, và so sánh "paper vs thực dụng" vẫn làm được.

**(c) `beta_style` — cũng ba lựa chọn**

| style | Cập nhật M1 | Ghi chú |
|---|---|---|
| `delta` | `M ← (α−η)·M + η·g` | xấp xỉ delta-rule của dự án. **Không** đồng nhất Eq. 48-49 đầy đủ — docstring nói thẳng |
| `ema` | `M ← β₁·M + (1−β₁)·g` | momentum Adam thường |
| `paper` | `M ← M + β₁·g` | nguyên văn Algorithm 1 |

**(d) `key_proj_eta` — phép quên có hướng, thử nghiệm**
```python
g_hat = g / (g.norm() + eps)
proj  = torch.sum(st["m1"] * g_hat)
st["m1"].sub_(g_hat, alpha=float(proj) * key_proj_eta)
```
Quên **thêm** đúng phần `m1` đang nằm dọc hướng gradient hiện tại — phép chiếu trực giao, `O(dim)`. Mặc định `0.0` = tắt.

**Validation dày đặc ở `__init__`** — 8 phép kiểm tra, kể cả ràng buộc toán `0 < η ≤ α ≤ 1`. Fail sớm với thông báo rõ.

---

# NHÓM III — Bài toán UAV thật ⭐

Phần từ 03/08 trở đi. Nhiều bài học phương pháp nhất trong cả dự án.

## III.1. `data/drift.py` (139 dòng) — mô phỏng trôi

Năm phép biến đổi, mỗi phép mô phỏng một hiện tượng vật lý **có thật**:

| Phép | Hiện tượng | Dải (mức 0 → 100%) |
|---|---|---|
| độ sáng | giờ trong ngày, mùa | ×1,00 → ×0,55 |
| nhiệt độ màu | bình minh ấm → trưa lạnh | 0 → ±18% lệch kênh R/B |
| tương phản | sương mù, khói, bụi | ×1,00 → ×0,65 |
| mờ Gauss | độ cao bay, ống kính bẩn/ướt | σ 0 → 1,4 px |
| nhiễu | ISO cao khi thiếu sáng | σ 0 → 0,035 |

### ⭐ Nguyên tắc quan trọng nhất file

> **Trôi phải TẤT ĐỊNH, không phải nhiễu ngẫu nhiên:**
> - **Mức trung bình** dịch theo task một cách xác định ← đây MỚI là "trôi"
> - Cộng **rung nhẹ** ±jitter quanh mức đó ← mỗi chuyến bay hơi khác nhau
>
> Nếu để hoàn toàn ngẫu nhiên thì nó chỉ là augmentation thông thường, **phân bố không dịch đi**, và λ sẽ hoàn toàn vô dụng — thí nghiệm mất ý nghĩa. Đây là **bẫy thiết kế chính** của file này.

Cài đặt:
```python
def _he_so(self, mac_dinh):
    """Cường độ thực = mức trung bình × (1 ± jitter)."""
    r = float(torch.rand(1, generator=self._g)) * 2.0 - 1.0     # [-1, 1]
    return mac_dinh * (1.0 + self.jitter * r)
```

`mac_dinh` là **tất định** theo `muc`; `jitter` chỉ rung quanh nó. Generator có seed riêng → tái lập.

### `muc_troi(task_idx, num_tasks, mode, severity)`

```python
if mode == "step":   return severity if task_idx >= num_tasks // 2 else 0.0
return severity * task_idx / (num_tasks - 1)          # linear (mặc định)
```

### `build_drift_transform()` — thứ tự có chủ đích

```python
return T.Compose(hinh_hoc + [T.ToTensor()]
                 + ([ApDungTroi(m, jitter=jitter, seed=seed_t)] if m > 0 else [])
                 + [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
```

```
hình học → ToTensor → TRÔI → Normalize
```

**Trôi phải nằm trên thang `[0,1]`** mới đúng nghĩa vật lý: "giảm độ sáng 45%" là nhân trên thang `[0,1]`, không phải trên thang đã trừ mean chia std.

**Seed riêng cho từng (task, train/eval):**
```python
seed_t = int(seed) * 1000 + task_idx * 2 + (1 if train else 0)
```
Tái lập được, và train/eval không trùng nhiễu.

## III.2. `data/revisit.py` (134 dòng) — file làm rõ nhất tư duy dự án

### Vì sao có file này

> Stream trôi của D10 tăng **đơn điệu** 0% → 100%, điều kiện cũ **không bao giờ quay lại**. Ở đó quên điều kiện cũ gần như **miễn phí** — đó chính là lý do arm λ=0,99 thắng λ=1 tới 3,91 điểm ở mức trôi 100%.
>
> Nhưng bài toán thật thì drone **bay lại cùng chỗ**, và mùa/giờ đều **tuần hoàn**. Quên điều kiện mùa đông vào tháng 6 nghĩa là tháng 12 phải học lại từ đầu. Con số +3,91 điểm **không chuyển sang được và có thể đảo dấu**.

Khác biệt cốt lõi:

```
drift.py    : mức trôi = hàm ĐƠN ĐIỆU của chỉ số task    -> không lặp
revisit.py  : mức trôi = hàm TUẦN HOÀN của chỉ số chuyến -> CÓ lặp
```

### `ChuyenBay`

```python
@dataclass
class ChuyenBay:
    chi_so: int          # chuyến thứ mấy
    muc_troi: float      # mức trôi [0,1] — model KHÔNG thấy
    mode_that: int       # nhãn chế độ điều kiện — CHỈ để chấm điểm
    lan_gap_mode: int    # lần thứ mấy gặp chế độ này (1 = lần đầu)
```

**`mode_that` là "ground truth" mà model không được biết.** Cùng nguyên tắc với nhãn `y` ở pha 2: dùng để chấm, không dùng để học.

### `lich_bay()` — ba chi tiết cực đáng đọc

```python
dao_dong = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / chu_ky))    # sin: 0 -> 1 -> 0
m = severity * dao_dong
m = round(float(m), 6)                                            # ⭐ CHI TIẾT 1
mode = int(round(m * (n_mode - 1)))                               # ⭐ CHI TIẾT 2
if mode != mode_truoc:                                            # ⭐ CHI TIẾT 3
    dem_dot[mode] = dem_dot.get(mode, 0) + 1
mode_truoc = mode
```

**CHI TIẾT 1 — `round(m, 6)`, và comment giải thích:**

> **Bắt buộc làm tròn:** `cos(π/2)` trả `6,1e-17` chứ không phải 0, nên chuyến 1 và chuyến 3 cùng ở mức 50% lại ra `0.49999999999999994` và `0.5000000000000001` → rơi vào **hai chế độ khác nhau**. Cùng một điều kiện vật lý mà bị coi là hai điều kiện thì **O3 vô nghĩa**.

Một dòng `round()`, và không có nó thì toàn bộ thí nghiệm cho ra số 0 mà không ai biết vì sao.

**CHI TIẾT 2 — `round` chứ không phải `floor`** khi rời rạc hoá: để mức `1.0` không rơi ra ngoài dải, và hai chuyến mức gần nhau về cùng một chế độ.

**CHI TIẾT 3 — đếm theo ĐỢT, không theo lần xuất hiện:**

> Với `troi_dan`, các chuyến liên tiếp có mức gần nhau sẽ rơi cùng một ô rời rạc — nhưng đó **không phải** "quay lại", điều kiện có rời đi đâu mà quay lại. Quay lại thật = **đã rời khỏi chế độ rồi trở về**. Không phân biệt chỗ này thì `troi_dan` sẽ báo có quay lại một cách **giả tạo**, và thước đo O3 sẽ đo một hiện tượng **không tồn tại**.

### `lich_tu_cfg()` — một nguồn sự thật

```python
def lich_tu_cfg(n_chuyen, drift_cfg):
    return lich_bay(n_chuyen,
                    che_do=str(drift_cfg.get("che_do", "tuan_hoan")),
                    chu_ky=int(drift_cfg.get("chu_ky", 4)), ...)
```

> Trước đây `run_g1.py` và `loaders.py` mỗi nơi tự đọc `drift_cfg` và tự điền mặc định. Hiện khớp nhau, nhưng sửa mặc định một nơi là hai lịch lệch **ngầm** — model train trên lịch này, chấm điểm trên lịch kia, và **không có lỗi nào được ném ra**.

**"Không có lỗi nào được ném ra"** là phần đáng sợ nhất. Mọi call site bắt buộc đi qua hàm này.

### `kiem_lich()` — cửa chặn

```python
lap = sum(1 for c in lich if c.lan_gap_mode > 1)
if lap == 0:
    raise ValueError("Lịch bay KHÔNG có chuyến nào lặp lại chế độ điều kiện cũ -> "
                     "không đo được 'Lợi ích khi quay lại' (mục tiêu O3). "
                     "Tăng n_chuyen, giảm chu_ky, hoặc đổi che_do sang 'tuan_hoan'.")
```

Lỗi **kèm cách sửa**. Ném **trước khi** đốt giờ máy.

## III.3. `metrics/revisit.py` (116 dòng) — thước đo trung tâm ⭐

### Vì sao phải viết thước đo mới

> `AAA` / `Average Accuracy` / `Forgetting` **không đo được O3**. Chúng chỉ nói mô hình đúng bao nhiêu, không nói nó có **ghi nhớ điều kiện** hay đang thích nghi lại từ đầu mỗi lần.
>
> Bài học trực tiếp từ D10: λ hoạt động tốt (+3,91 điểm, 3,10σ) nhưng `AAA` trung bình trên mọi task đã **che mất hoàn toàn** — nhìn `AAA` thì kết luận là "λ vô dụng".
>
> **Thước đo sai thì che mất kết quả đúng. Nên định nghĩa thước đo TRƯỚC khi cài cơ chế.**

Nếu bạn chỉ nhớ một câu từ cả ba bài, nhớ câu in đậm đó.

### `acc_hien_tai(R)` — O1

```python
return sum(float(R[i][i]) for i in range(n)) / n
```

Trung bình **đường chéo**: đúng bao nhiêu trên chính điều kiện **đang bay**.

> `AAA` trung bình trên mọi chuyến đã bay, mà sau chuyến thứ 9 thì **8/9 bộ test là điều kiện QUÁ KHỨ** — thưởng cho việc nhớ lịch sử, không thưởng cho việc bay đúng hôm nay. **Drone thì bay bây giờ.**

### `loi_ich_quay_lai(R, mode_that)` — O3 ⭐⭐

```
Lợi ích khi quay lại = Acc(gặp lại chế độ X) − Acc(lần ĐẦU gặp chế độ X)
```

```python
lan_dau, chenh, theo_lan, dem_dot = {}, [], {}, {}
mode_truoc = None
for i, m in enumerate(mode_that):
    m = int(m); acc = float(R[i][i])              # ⭐ đo trên ĐƯỜNG CHÉO
    if m == mode_truoc:
        continue                                   # ⭐ chuyến liên tiếp cùng chế độ = cùng ĐỢT
    mode_truoc = m
    dem_dot[m] = dem_dot.get(m, 0) + 1
    if m not in lan_dau:
        lan_dau[m] = acc                           # lần đầu gặp -> ghi mốc
        continue
    d = acc - lan_dau[m]
    chenh.append(d)
    theo_lan.setdefault(dem_dot[m], []).append(d)
return {"loi_ich_quay_lai": mean(chenh), "so_lan_quay_lai": len(chenh),
        "so_che_do": len(lan_dau), **{f"loi_ich_lan_{k}": mean(v) for k,v in theo_lan.items()}}
```

**Đọc kết quả:**
- **> 0**: hệ nhận ra điều kiện cũ và tận dụng được ✓
- **≈ 0**: hệ thích nghi lại từ đầu mỗi lần → **KHÔNG ghi nhớ được điều kiện, O3 thất bại** dù O1/O2 có tốt đến đâu. Docstring nói thẳng: *"lúc đó nên bỏ tầng trung cho gọn"*

**`loi_ich_lan_k`** tách theo lần gặp thứ mấy — xem lợi ích có tăng dần theo số lần gặp không.

### `thoi_gian_hoi_phuc(acc_theo_buoc, nguong=0.95)` — O2

```python
k = max(1, len(v) // 5)
on_dinh = sum(v[-k:]) / k          # mức ổn định = trung bình 20% bước cuối
muc = nguong * on_dinh
for i, x in enumerate(v):
    if x >= muc: return float(i)
return float("inf")                 # ⭐ KHÔNG lấp liếm
```

> Trả về `inf` nếu không bao giờ hồi tới ngưỡng — trường hợp đó có nghĩa là hệ **không** bám kịp trôi, phải **báo động** chứ không lấp liếm bằng một con số lớn.

**Cảnh báo quan trọng** (in ở `run_g1.py`):
```
⚠️ O2 tính so với mức ổn định CỦA CHÍNH arm — arm đứng im ở mức thấp
   cũng 'hồi phục' nhanh. Luôn đọc O2 KÈM acc.
```

Một chỉ số có thể **bị đánh lừa bởi một mô hình tệ**. Ghi cảnh báo ngay trong output là cách xử lý đúng.

### `in_bao_cao()` — log tự chẩn đoán

```python
if tt.get("so_lan_quay_lai", 0) == 0:
    dong.append("⚠️ KHÔNG có chuyến nào lặp lại chế độ cũ — lịch bay sai, O3 không đo được")
elif li < 0.005:
    dong.append("⚠️ Lợi ích ≈ 0 — hệ đang thích nghi LẠI TỪ ĐẦU mỗi lần, KHÔNG ghi nhớ điều kiện")
```

**Log tự nói ra kết luận**, không bắt người đọc suy luận từ con số.

## III.4. `models/tang_nhanh.py` (158 dòng) — M1, tầng nhanh

### Vị trí trong bộ ba

```
tầng chậm  = SLDA (μ_c, Σ)    — danh tính lớp, học MỘT lần ở pha 1 CÓ nhãn
TẦNG NHANH = file này          — "điều kiện lúc này", mỗi batch, KHÔNG nhãn
tầng trung = ngan_hang_che_do  — "các điều kiện đã gặp", 1 lần/chuyến
```

Ánh xạ thẳng vào ba tần số của bài toán (a/b/c ở §1 tài liệu tổng quan).

### Ý tưởng

> Điều kiện quan sát (nắng, sương, bụi) dịch **toàn bộ** đám mây feature; thông tin lớp nằm ở vị trí **tương đối** giữa các mẫu. Vậy theo dõi trung bình/phương sai chạy `(m_t, v_t)` rồi căn mọi feature về hệ toạ độ của pha hiệu chỉnh `e₀` `(m0, v0)` — **tầng chậm bên dưới không hề biết điều kiện đã đổi**.

Và:

> Đây **chính là** cổng quên của Titans/NL nhưng bằng **công thức đóng**: λ cố định nên về mặt cấu trúc **không thể trôi ra biên** như cổng học được đã làm (CHAN_DOAN_NL: bão hoà 9/13 run).

**Đây là một luận điểm thiết kế mạnh:** thay cổng học được (có thể hỏng) bằng hằng số (không thể hỏng), đổi lấy mất tính thích nghi.

### `__init__` — không phải `nn.Module`, có chủ đích

```python
class TangNhanh:
    """Không phải nn.Module có chủ đích: không tham số học được, không đi vào optimizer."""
```

Validation dày, kèm giải thích:
```python
if not (0.0 < decay < 1.0):
    raise ValueError(f"tang_nhanh.decay phải thuộc (0, 1) (nhận {decay}) — "
                     "decay=1 nghĩa là không bám gì, dùng U0 thay vì bật tầng nhanh")
```
Không chỉ báo sai, mà **chỉ đúng thứ nên dùng thay thế**.

### `cap_nhat(feats)` — EMA, KHÔNG nhãn

```python
mb = f.mean(dim=0)
vb = f.var(dim=0, unbiased=False) if f.shape[0] > 1 else torch.zeros_like(mb)
if self.m_t is None:                       # batch đầu = khởi tạo thẳng, khỏi bias-correction
    self.m_t, self.v_t = mb.clone(), vb.clone().clamp(min=self.eps)
else:
    r = self.decay ** int(f.shape[0])      # ⭐ r = λ^B
    self.m_t = r * self.m_t + (1.0 - r) * mb
    if f.shape[0] > 1:
        self.v_t = (r * self.v_t + (1.0 - r) * vb).clamp(min=self.eps)
self.so_mau += int(f.shape[0])
```

**`r = λ^B` chứ không phải `λ`.** EMA theo-**mẫu** nhưng cập nhật theo-**batch**: `B` mẫu đi qua thì hệ số phân rã phải là `λ^B`. Docstring gọi là *"xấp xỉ chuẩn của EMA từng-mẫu khi batch cùng phân bố"*, và ghi rõ giả định: **A4 "trôi trơn"**.

**`v_t` chỉ giữ đường chéo** — `D` số thay vì `D²`. Docstring: *"3 KB, ổn định ước lượng"*. Ước lượng ma trận hiệp phương sai đầy đủ từ vài trăm mẫu là không đáng tin.

### `chot_moc()` và `can_chinh()` — vòng đời hai pha

```python
def chot_moc(self):
    """Cuối pha hiệu chỉnh: đóng băng (m0, v0). Từ đây `can_chinh` mới có tác dụng."""
    if self.m_t is None:
        raise RuntimeError("chot_moc() gọi khi chưa hấp thụ mẫu nào — pha 1 rỗng?")
    self.m0, self.v0 = self.m_t.clone(), self.v_t.clone()

def can_chinh(self, feats):
    if self.m0 is None or self.m_t is None:
        return feats                          # ⭐ chưa chốt mốc -> IDENTITY
    if self.kieu == "truc":
        u = self.truc.to(f.device)
        do_doi = torch.dot(m_t - m0, u)       # điều kiện trôi bao xa TRÊN trục
        return f - do_doi * u                 # phần ⊥ trục (thông tin lớp) KHÔNG bị đụng
    ty_le = torch.sqrt(v0.clamp(min=eps) / v_t.clamp(min=eps))
    return (f - m_t) * ty_le + m0             # day_du: dịch + co giãn từng chiều
```

**Trước khi chốt mốc, `can_chinh` là identity.** Nhờ vậy pha 1 chạy **trùng khít** bản cũ → mọi run cũ bất biến. Đây là cách bật tính năng mới mà không phá gì.

**Hai kiểu căn chỉnh — T1 quyết định dùng kiểu nào:**

| kiểu | Công thức | Khi nào |
|---|---|---|
| `day_du` | `f' = (f − m_t)·√(v0/v_t) + m0` | mặc định, dịch + co giãn từng chiều |
| `truc` | `f' = f − ((m_t − m0)·û)·û` | **có trục điều kiện** (T1 xác nhận) |

**Vì sao `truc` tốt hơn khi có trục:** nó **chỉ** sửa thành phần trên trục điều kiện; phần vuông góc (mang thông tin lớp) **giữ nguyên**. Đây là thuốc chống bẫy T2: đổi tỷ lệ lớp dịch `m_t` theo hướng ⊥ trục sẽ **không** bị "sửa nhầm".

Cửa chặn:
```python
if self.kieu == "truc":
    if not truc_json:
        raise ValueError("tang_nhanh.kieu='truc' cần tang_nhanh.truc_json "
                         "(file T1 sinh bởi scripts/do_truc_dieu_kien.py --json)")
    if not p.exists():
        raise FileNotFoundError(f"Không thấy file trục điều kiện: {truc_json} — "
                                "chạy T1 trước (cửa chặn nhóm B)")
```

### `bao_cao()` — bằng chứng chạy đúng

```python
do_troi = float(torch.dot(d, self.truc)) if self.truc is not None else float(d.norm())
return {"kieu":..., "decay":..., "so_mau":..., "da_chot":...,
        "do_troi_hien_tai": do_troi, "bytes": self.extra_bytes()}
```

`do_troi_hien_tai` là con số dùng để **hiệu chỉnh `nguong` của ngân hàng chế độ** — xem chú thích trong `configs/revisit_U2_nganhang.yaml`.

## III.5. `models/ngan_hang_che_do.py` (149 dòng) — M2, tầng trung

### Vấn đề nó giải

> Tầng nhanh bám điều kiện tốt nhưng **KHÔNG NHỚ**. Gặp lại mùa đông sau 6 tháng, nó thích nghi lại từ đầu — mất đúng "thời gian hồi phục" như lần đầu.

Ngân hàng lưu **snapshot trạng thái tầng nhanh** của từng điều kiện đã gặp.

### Cơ chế (4 dòng tóm tắt cả file)

```
mỗi chế độ k:  (m_k, v_k, so_lan_gap)     — chính là snapshot tầng nhanh
khớp:          k* = argmin d(m_t, m_k)     — d đo TRÊN TRỤC nếu có, ngược lại L2
d ≤ nguong  -> GẶP LẠI: nạp (m_k, v_k) vào tầng nhanh, so_lan_gap += 1
d >  nguong -> điều kiện MỚI: cuối chuyến ghi chế độ mới
len > k_max -> gộp hai chế độ GẦN NHAU NHẤT (trung bình trọng số theo so_lan_gap)
```

### `_d()` — vì sao khớp trên trục quan trọng (bẫy T2)

```python
def _d(self, a, b, truc):
    if truc is not None:
        return abs(float(torch.dot(a - b, truc)))   # |chiếu lên trục|
    return float((a - b).norm())                     # L2 đầy đủ
```

> Drone bay từ thành phố sang rừng làm **tỷ lệ lớp** đổi → `m_t` dịch theo hướng gần ⊥ trục điều kiện. Khớp bằng L2 đầy đủ sẽ tưởng đó là **điều kiện mới** và **nổ số chế độ**; khớp trên trục thì thành phần ⊥ bị chiếu bỏ, không đánh lừa được.

### `khop_va_nap()` — đầu chuyến, và một bug đã bắt được

```python
@torch.no_grad()
def khop_va_nap(self, tn, m_tuoi=None) -> bool:
    self._khop_hien_tai = None
    m = m_tuoi if m_tuoi is not None else tn.m_t          # ⭐
    d, k = self._gan_nhat(m, tn.truc)
    if d is None or d > self.nguong:
        return False                                       # điều kiện lạ
    self._khop_hien_tai = k
    self.che_do[k]["so_lan_gap"] += 1
    self.so_lan_nap += 1
    tn.m_t = self.che_do[k]["m"].clone()                   # ⭐ NẠP: thích nghi tức thời
    tn.v_t = self.che_do[k]["v"].clone()
    return True
```

**Dòng `m = m_tuoi if ... else tn.m_t` sinh ra từ một bug bắt được bằng mô phỏng:**

> Tại batch thứ `cho_khop_sau`, EMA `tn.m_t` còn nhiễm `~λ^(B·cho_khop_sau)` điều kiện của chuyến **TRƯỚC** (λ=0,99, B=32, 5 batch → còn ~20%; λ=0,995 → còn **67%**). Khớp bằng `m_t` nhiễm thì ngân hàng có thể **nạp nhầm chính chế độ vừa rời khỏi**. Khớp phải dùng ước lượng **TƯƠI** của chuyến hiện tại.

**`cho_khop_sau`** (mặc định 5): đầu chuyến đợi tầng nhanh nhìn đủ N batch cho `m_t` bớt nhiễu rồi mới tra ngân hàng. Tra ngay batch 1 thì `m_t` còn là nhiễu của 32 mẫu đầu.

### `ghi_lai()` — cuối chuyến

```python
if self._khop_hien_tai is not None:               # chuyến quen: MÀI chế độ cũ theo EMA
    c["m"] = (1 - lam_mode)*c["m"] + lam_mode*tn.m_t
    c["v"] = (1 - lam_mode)*c["v"] + lam_mode*tn.v_t
else:                                              # chuyến lạ: chế độ MỚI
    self.che_do.append({"m": tn.m_t.clone(), "v": tn.v_t.clone(), "so_lan_gap": 1})
    if len(self.che_do) > self.k_max:
        self._gop_gan_nhat(tn.truc)
```

### `_gop_gan_nhat()` — khi K tràn

Tìm cặp gần nhau nhất (O(k²), k ≤ 8 nên rẻ), gộp bằng **trung bình có trọng số theo `so_lan_gap`** — chế độ gặp nhiều lần có tiếng nói lớn hơn.

### Ba tham số nguy hiểm

| Tham số | Nhỏ quá | Lớn quá |
|---|---|---|
| `nguong` | nổ số chế độ (mỗi chuyến một chế độ) | gộp hết làm một |
| `k_max` | gộp liên tục, mất phân giải | tốn bộ nhớ |
| `lam_mode` | chế độ không cập nhật theo trôi | chế độ bị chuyến mới nhất chi phối |

**Cửa chặn riêng** (in ở `run_g1.py`):
```python
if bc["so_che_do"] > 2 * len(set(mode_that)):
    print("⚠️ số chế độ NỔ so với điều kiện thật — giảm nguong hoặc xem lại T1")
```

## III.6. `SLDAClassifier.hap_thu_khong_nhan()` — pha 2 ⭐

**Hàm cài đặt trực tiếp phát biểu bài toán gốc.** Đọc cùng nhánh pha 2 của `engine.py` (Bài 2, §A4.7).

```python
@torch.no_grad()
def hap_thu_khong_nhan(self, loader, device, allowed=None) -> list:
    self.eval()
    accs, dem_batch, m_tuoi_sum = [], 0, None
    for x, y in loader:
        raw = self.backbone(x.to(device)).float()

        # 1) DỰ ĐOÁN TRƯỚC — bằng trạng thái HIỆN TẠI (chưa thấy batch này)
        f = self.tang_nhanh.can_chinh(raw) if self.tang_nhanh is not None else raw
        self._refresh_cache()
        logits = f @ self._cache_w + self._cache_b
        if allowed is not None:
            logits = mask_logits(logits, allowed)
        accs.append(float((logits.argmax(dim=1).cpu() == y).float().mean()))

        # 2) RỒI MỚI CẬP NHẬT — chỉ tầng KHÔNG nhãn, chỉ từ feature THÔ
        if self.tang_nhanh is not None:
            self.tang_nhanh.cap_nhat(raw)
        dem_batch += 1

        # tầng trung: khớp bằng trung bình TƯƠI của chuyến này
        if self.ngan_hang is not None and dem_batch <= self.ngan_hang.cho_khop_sau:
            mb = raw.mean(dim=0)
            m_tuoi_sum = mb if m_tuoi_sum is None else m_tuoi_sum + mb
            if dem_batch == self.ngan_hang.cho_khop_sau:
                m_tuoi = m_tuoi_sum / float(dem_batch)
                if self.ngan_hang.khop_va_nap(self.tang_nhanh, m_tuoi=m_tuoi):
                    print("[ba_tang] GẶP LẠI chế độ cũ — nạp snapshot, khỏi học lại (O3)")
    if self.ngan_hang is not None:
        self.ngan_hang.ghi_lai(self.tang_nhanh)     # cuối chuyến: ghi/mài chế độ
    return accs
```

**Thứ tự "1) dự đoán TRƯỚC, 2) RỒI mới cập nhật" là toàn bộ ý nghĩa của hàm.** Đây là đánh giá **prequential** (test-then-train) — đúng nghĩa *"phải trả lời ŷ_t ngay, trước khi thấy x_{t+1}"* trong phát biểu bài toán.

**Luật nhãn, ghi rõ trong docstring:**

> `y` trong loader **CHỈ** dùng để chấm prequential — cùng nguyên tắc với `mode_that`. **Không một giá trị y nào chạm vào update** (μ_c/Σ/m_t đều không). Bằng chứng kiểm được: `count_raw` **đứng yên** suốt pha 2 (test bắt điều này).

**"Bằng chứng kiểm được"** là điểm đáng học nhất: thay vì hứa "code đúng", họ chỉ ra một **đại lượng quan sát được** (`count_raw` phải đứng yên) mà test có thể kiểm tự động.

**Giá trị trả về `accs`** chính là `log["trace"][t]` trong engine → nuôi O2 (thời gian hồi phục) và O3-PREQ10 (số công bố chính).

---

# Bảng tổng — ba track thí nghiệm hiện tại

Mỗi track thêm đúng **một** tầng → so sánh một biến:

| Track | Config | Có gì | Đo được |
|---|---|---|---|
| **U0** | `revisit_U0_dongbang.yaml` | SLDA đóng băng hoàn toàn ở pha 2 | mốc dưới |
| **U1** | `revisit_U1_tangnhanh.yaml` | + tầng nhanh (M1) | bám trôi có lợi không |
| **U2** | `revisit_U2_nganhang.yaml` | + ngân hàng chế độ (M2) | ⭐ **nhớ điều kiện có lợi không** |

**Con số công bố = O3-PREQ10(U2) − O3-PREQ10(U1)**, 3 seed, kèm σ.

Phép so **một biến** nên tự khử nhiễu kiểu *"học thêm dữ liệu thì tự khắc tốt lên"*, và tự khử carryover EMA.

Kiểm bằng:
```bash
python scripts/tom_tat_ba_tang.py --cua-chan      # exit 3 nếu U1 không hơn U0
```

---

# Checklist Bài 3

- [ ] Giải thích được vì sao NCM dùng `register_buffer` chứ không `nn.Parameter`
- [ ] Giải thích được vì sao SLDA cần **hai** cặp bộ đếm (`count` và `count_g`)
- [ ] Chứng minh được `cov_mode="identity"` ≡ NCM
- [ ] Kể được hai luật vòng đời state của Titans (`detach` khi train, `clone` khi eval) và hậu quả nếu thiếu
- [ ] Giải thích được vì sao `p.grad = None` làm tier CMS **thật sự** đứng yên
- [ ] Giải thích được vì sao `round(m, 6)` trong `lich_bay` là **bắt buộc**
- [ ] Giải thích được vì sao O3 phải đo trên **10 batch đầu** chứ không phải đường chéo R
- [ ] Giải thích được vì sao `khop_va_nap` phải dùng `m_tuoi` chứ không phải `tn.m_t`
- [ ] Chạy `pytest tests/test_slda.py tests/test_ba_tang.py tests/test_revisit.py -q` — xanh
- [ ] Chạy `python scripts/mo_phong_ba_tang.py` (thuần numpy, không cần GPU) và đọc output

---

# Ba bài học phương pháp rút ra từ cả dự án

Đây là thứ đáng mang đi nơi khác, không chỉ dùng cho repo này:

**1. Thước đo sai thì che mất kết quả đúng — định nghĩa thước đo TRƯỚC khi cài cơ chế.**
D10: λ hoạt động (+3,91 điểm, 3,10σ) nhưng AAA che mất hoàn toàn. Nếu dừng ở đó thì kết luận là "λ vô dụng" — sai, và không ai biết là sai.

**2. Đo đúng chỗ, đúng lúc.**
O3 trên đường chéo R **bị mù** với ngân hàng chế độ, vì chấm cuối chuyến lúc tầng nhanh đã tự hội tụ. Lợi ích nằm ở **đầu** chuyến. Cùng một cơ chế, cùng một dữ liệu, đo lệch chỗ là ra 0.

**3. Log phải in kèm mốc so sánh, không chỉ in giá trị.**
`"η=0.896"` không có mốc → lỗi thang 100× sống sót trọn một ngày. `"η=0.896 (99% khoảng [0.01, 0.9])"` thì máy tự nói ra vấn đề.

Và nguyên tắc kỹ thuật xuyên suốt: **tính năng mới mặc định TẮT** để mọi run cũ bất biến, và **cửa chặn sớm** với thông báo kèm cách sửa, ném ra **trước khi** đốt giờ máy.
