# Kế hoạch — chế độ TỰ LUYỆN giữa các chuyến bay

Lập 2026-08-10, sau khi định nghĩa lại bài toán: **cho phép cập nhật mô hình giữa hai chuyến, không dùng nhãn mới.**

Khung nghiên cứu: **B — chẩn đoán.** Nested Learning là *đối tượng nghiên cứu*, không phải bên thua.

---

# PHẦN 0 — Nguyên tắc, và CẮT gì

## 0.1. Vì sao đổi bài toán

Chẩn đoán từ U3: **CMS và M3 không có việc để làm** vì bài toán cũ chỉ có **một** đợt gradient (chuyến 0). Cho phép cập nhật giữa chuyến biến 1 đợt thành **12** — đúng chế độ mà CMS được thiết kế.

Đây là làm bài toán **thực tế hơn**, không phải dễ hơn: drone hạ cánh, sạc pin, có thời gian tính toán. *"Không bao giờ cập nhật nữa"* mới là giả định phi thực tế.

## 0.2. ✂️ CẮT — không làm nữa

| Việc | Vì sao cắt |
|---|---|
| **M0 tầng nền hiệu chỉnh phi tuyến** | Thuộc khung A (tối đa hoá điểm số). Lệch trọng tâm NL. Giữ làm *future work* |
| **λ-sweep tìm ranh giới O3** | O3 đã có câu trả lời (−0,0006 ± 0,0010) kèm cơ chế. Reframe đổi hẳn câu hỏi |
| **R1/R3 chẩn đoán α và CMS trên cấu hình CŨ** | α đo trực tiếp được rồi (bò 88%→96%); CMS bất lực là **lập luận cấu trúc**, không cần run. Và cả hai đổi nghĩa dưới chế độ mới |
| **Ablation `beta_style=paper` của M3** | Ưu tiên thấp. Chỉ chạy **1 seed** nếu báo cáo cần dòng *"đã cài đúng Algorithm 1"* |
| **Sửa chú thích config lặt vặt** | `default.yaml` ghi `muon`, `m3.eps` bị nuốt… — không ảnh hưởng kết quả. Dồn vào một lần cuối |
| **Ngân hàng chế độ (M2)** | Đã đo: ≈ 0, có dấu hiệu hại. **Đóng lại.** Giữ nguyên số đã có làm kết quả âm |

## 0.3. ✅ GIỮ — nhưng ưu tiên thấp

| Việc | Khi nào |
|---|---|
| Oracle có nhãn (`revisit_arm1`) | Sau T3 — cho biết trần trên |
| Sửa định vị văn liệu (TTA thay vì CL) | Lúc viết báo cáo |
| `CMSOptimizer.state_dict()` | Chỉ khi cần `--resume` |

---

# T0 — VIẾT LẠI PHÁT BIỂU BÀI TOÁN ⛔ LÀM TRƯỚC MỌI THỨ

**30 phút, không code, không giờ máy.**

> Đừng chạy trước khi phát biểu rõ. Đó là lỗi đã mắc **hai lần**: D10 (thước đo AAA che mất λ) và U2 (`cho_khop_sau` nằm ngoài cửa sổ đo). Cả hai đốt >12h máy cho kết quả không đọc được.

## T0.1. Sửa `BAI_TOAN_VA_MUC_TIEU` §2 — Pha 2

Thay đoạn hiện tại bằng:

```
Pha 2 — triển khai, KHÔNG nhãn mới, hai chế độ xen kẽ:

  TRONG CHUYẾN (trực tuyến, ngân sách chặt)
    • nhận x_t, trả ŷ_t NGAY, trước khi thấy x_{t+1}
    • KHÔNG gradient, KHÔNG lưu ảnh thô
    • được lưu ĐẶC TRƯNG NÉN (384-d fp16 ≈ 0,75 KB/ảnh)

  GIỮA HAI CHUYẾN (ngoại tuyến, drone hạ cánh + sạc)
    • được cập nhật mô hình bằng đặc trưng đã lưu
    • KHÔNG có nhãn mới của con người
    • nguồn giám sát hợp lệ: (a) mục tiêu tự giám sát trên chuyến vừa bay
                              (b) đặc trưng pha 1 CÓ nhãn, đã lưu từ đầu
```

## T0.2. Sửa §4 — Ràng buộc

Thêm dòng ngân sách buffer:

```
| Buffer đặc trưng | ≤ 2 MB | = 45 lớp × 20 mẫu × 384-d fp16 ≈ 0,69 MB |
```

Tổng trần vẫn **10 MB · 1,67 ms**.

## T0.3. Sửa §9 — Ngoài phạm vi

Bỏ *"thích nghi backbone"* khỏi danh sách ngoài phạm vi (giờ nằm trong). Thêm:
- *"Nhãn mới từ con người ở pha 2"* — vẫn ngoài phạm vi
- *"Lưu ảnh thô"* — vẫn ngoài phạm vi

## T0.4. Ghi rõ LUẬT — bảng một trang

| | Được phép | Cấm |
|---|---|---|
| Trong chuyến | suy luận · cập nhật thống kê/ký ức không nhãn · lưu đặc trưng | gradient · lưu ảnh thô · nhãn |
| Giữa hai chuyến | gradient · replay đặc trưng pha 1 · mục tiêu tự giám sát | nhãn mới · ảnh thô · đọc lại dữ liệu pha 1 dạng ảnh |

**Cửa chặn T0:** không sang T1 khi chưa có bảng luật này viết ra. Nó là thứ phản biện đọc đầu tiên.

---

# T1 — HẠ TẦNG TỰ LUYỆN (code, ~3h)

## T1.1. Buffer đặc trưng pha 1 — `src/uavcl/models/bo_dem_latent.py` (~50 dòng)

```python
class BoDemLatent:
    """Đặc trưng pha 1 CÓ nhãn, quota mỗi lớp. Neo danh tính lớp khi tự luyện.

    45 lớp × 20 mẫu × 384-d fp16 = 0,69 MB — trong trần 2 MB của §4.
    Backbone đóng băng ở pha 1 nên đặc trưng KHÔNG 'ôi' (bài học #22 latent replay).
    """
    def __init__(self, num_classes, quota=20, seed=0): ...
    def nap(self, feats, ys):        # gọi cuối chuyến 0
    def lay(self, n, device):        # trả (z, y) ngẫu nhiên
    def extra_bytes(self):           # đối chiếu O4
```

## T1.2. SLDA tự luyện — `slda.py`, thêm `hap_thu_va_hoc()` (~40 dòng)

SLDA **không có tham số gradient** (μ_c, Σ là buffer). Bản tương đương của "học từ kinh nghiệm" là **cập nhật thống kê bằng nhãn giả**:

```python
@torch.no_grad()
def hap_thu_va_hoc(self, loader, device, allowed=None, nguong_tin=0.3):
    """Pha 2 ba tầng: bay (prequential) -> hạ cánh (cập nhật μ_c, Σ bằng NHÃN GIẢ).

    nguong_tin: chỉ nhận mẫu có BIÊN top1−top2 > nguong_tin. Chống tích luỹ lỗi —
    rủi ro chính của tự luyện. nguong_tin=inf -> không nhận gì (= U1, đối chứng).
    """
    accs, kho = [], []
    for x, y in loader:                       # --- BAY ---
        raw = self.backbone(x.to(device)).float()
        f = self.tang_nhanh.can_chinh(raw) if self.tang_nhanh else raw
        self._refresh_cache(); logits = f @ self._cache_w + self._cache_b
        if allowed is not None: logits = mask_logits(logits, allowed)
        accs.append(float((logits.argmax(1).cpu() == y).float().mean()))
        if self.tang_nhanh: self.tang_nhanh.cap_nhat(raw)
        top2 = logits.topk(2, dim=1)
        bien = top2.values[:, 0] - top2.values[:, 1]
        kho.append((f[bien > nguong_tin], top2.indices[bien > nguong_tin, 0]))
    for f, yg in kho:                         # --- HẠ CÁNH ---
        if len(f): self.update(f, yg)         # nhãn GIẢ, không phải y
    return accs
```

⚠️ **Không dùng `y` ở vế học** — chỉ dùng để chấm prequential. Test phải kiểm điều này.

## T1.3. HOPE tự luyện — `titans_head.py`, thêm `hap_thu_va_hoc()` (~60 dòng)

```python
def hap_thu_va_hoc(self, loader, device, allowed=None, buffer=None, opt=None,
                   epochs=2, trong_so_neo=1.0):
    """Pha 2 HOPE: bay (suy luận + thu latent) -> hạ cánh (gradient, KHÔNG nhãn mới).

        L = entropy_min(latent chuyến này)  +  w · CE(latent pha 1, nhãn pha 1)
            └── bám trôi, không nhãn ──┘      └── neo lớp, nhãn HỢP LỆ ──┘

    `opt` là CMSOptimizer giữ xuyên chuyến -> CMS chạy suốt 12 chuyến, đúng Eq 71.
    """
    accs, lat = [], []
    with torch.no_grad():                     # --- BAY ---
        for x, y in loader:
            x = x.to(device)
            self.eval(); logits = mask_logits(self(x), allowed)
            accs.append(float((logits.argmax(1).cpu() == y).float().mean()))
            self.train(); lat.append(self._extract(x).half().cpu())   # memory tự ghi
    self.train()                              # --- HẠ CÁNH ---
    for _ in range(epochs):
        for z in lat:
            p = F.softmax(mask_logits(self.forward_from_feats(z.float().to(device)), allowed), 1)
            loss = -(p * p.clamp_min(1e-8).log()).sum(1).mean()
            if buffer is not None:
                zb, yb = buffer.lay(32, device)
                loss = loss + trong_so_neo * F.cross_entropy(
                    mask_logits(self.forward_from_feats(zb), allowed), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    self.eval()
    return accs
```

## T1.4. Engine — cho phép truyền `opt` + `buffer` vào pha 2 (~15 dòng)

```python
if _la_pha2:
    if hasattr(model, "hap_thu_va_hoc"):          # ← ưu tiên bản CÓ HỌC
        log.setdefault("trace", {})[t] = model.hap_thu_va_hoc(
            task_loaders[t]["train"], device, allowed=_allowed_preq,
            buffer=_buffer, opt=opt_carry, **(train_cfg.get("tu_luyen") or {}))
    elif hasattr(model, "hap_thu_khong_nhan"):    # ← bản cũ, không học
        ...
```

Và nạp buffer cuối chuyến 0:
```python
if _khong_nhan and t == _chuyen_hc - 1 and _buffer is not None:
    _buffer.nap(...)      # đặc trưng pha 1 CÓ nhãn
```

## T1.5. Method mới — `methods.py` (~15 dòng)

```python
class SldaTuLuyen(SLDA):   name = "slda_tu_luyen";  gradient_free = True
class HopeTuLuyen(HOPE):   name = "hope_tu_luyen";  gradient_free = True
_METHODS.update({"slda_tu_luyen": SldaTuLuyen, "hope_tu_luyen": HopeTuLuyen})
```

## T1.6. Test — `tests/test_tu_luyen.py` (~80 dòng)

| # | Test | Bắt lỗi gì |
|---|---|---|
| 1 | `count_raw` **đứng yên** khi `nguong_tin=inf` | rò rỉ nhãn thật vào update |
| 2 | Buffer ≤ 2 MB với 45 lớp × 20 | vỡ ngân sách O4 |
| 3 | `hap_thu_va_hoc` trả list dài = số batch | hỏng chuỗi prequential |
| 4 | ⭐ **`y` không chạm vào update** — chạy 2 lần với `y` xáo trộn, `μ_c` phải giống hệt | **rò rỉ nhãn** — lỗi nguy hiểm nhất |
| 5 | `opt.step()` được gọi đúng `epochs × n_batch` lần | CMS không nhận đủ bước |

**Cửa chặn T1:** `pytest -q` xanh + test #4 phải có.

---

# T2 — U5: BA TẦNG + TỰ LUYỆN

**Chạy trước U4.** Nó tách câu hỏi *"tự luyện có đáng không"* khỏi câu hỏi *"NL có đáng không"*.

## T2.1. Config `revisit_U5_tuluyen.yaml`

= `revisit_U1_tangnhanh.yaml` + đổi 2 dòng:

```yaml
train:
  method: slda_tu_luyen         # ← thay slda
  eval_chi_duong_cheo: true
  tu_luyen:
    nguong_tin: 0.3             # biên top1−top2; hiệu chỉnh ở T2.2
```

## T2.2. Dò ngưỡng tin cậy — 3 run × ~1h

`nguong_tin` là tham số nguy hiểm nhất của tự luyện: **thấp → nhận nhãn giả sai → tích luỹ lỗi · cao → không nhận gì → thành U1.**

```bash
for NG in 0.1 0.3 1.0; do
  .venv/bin/python scripts/run_g1.py --config configs/revisit_U5_tuluyen.yaml \
    --set seed=0 log.dir=./artifacts_U5_ng$NG train.tu_luyen.nguong_tin=$NG \
    > run_U5_ng$NG.log 2>&1
done
```

**Cửa chặn — đọc đường chéo 12 chuyến:**

| Dấu hiệu | Nghĩa | Làm gì |
|---|---|---|
| Đường chéo **tăng dần** | ✅ tự luyện có tác dụng | chọn ngưỡng đó |
| Đường chéo **phẳng** ≈ U1 | ngưỡng quá cao, không nhận gì | giảm ngưỡng |
| Đường chéo **tụt** sau chuyến 4–5 | 🔴 **TÍCH LUỸ LỖI** | tăng ngưỡng, hoặc tăng `trong_so_neo` |

## T2.3. Ba seed với ngưỡng thắng — ~3h

**Số công bố:** `O1(U5) − O1(U1)`, ghép cặp theo seed, 3 seed, kèm σ và t.

---

# T3 — U4: HOPE + TỰ LUYỆN ⭐ CÂU HỎI TRUNG TÂM

Chỉ chạy **sau khi** T2 xong — cần biết tự luyện tự nó đáng bao nhiêu.

## T3.1. Config `revisit_U4_hope_tuluyen.yaml`

= `revisit_U3_hope.yaml` + đổi:

```yaml
train:
  method: hope_tu_luyen
  optimizer_per_task: false     # ⭐ CMSOptimizer sống xuyên 12 chuyến
  eval_ncm_head: true           # tách bộ đọc — thay cho R1/R2 đã cắt
  tu_luyen:
    epochs: 2
    trong_so_neo: 1.0
memory:
  gate_bound:
    alpha_logit_center: -3.453  # dải α [0.001, 0.5], dẫn từ chu kỳ bay
    alpha_logit_limit:   3.453  # thay [0.05,0.95] bê từ class-incremental
```

⚠️ Sửa `α` **cùng lúc** với tự luyện là **hai biến**. Chấp nhận, vì cả hai đều là "sửa cho đúng bài toán mới". Nhưng phải ghi rõ trong báo cáo, và T3.2 sẽ tách.

## T3.2. Ba run dò — ~10h, một đêm

| Run | Đổi gì | Tách được |
|---|---|---|
| **D1** | tự luyện + α mới | bản đầy đủ |
| **D2** | tự luyện + α **cũ** `[0.05, 0.95]` | đóng góp riêng của **α** |
| **D3** | tự luyện + α mới + `cms.tiers=[[12,1]]` | đóng góp riêng của **CMS đa tần số** |

```bash
nohup bash -c '
.venv/bin/python scripts/run_g1.py --config configs/revisit_U4_hope_tuluyen.yaml \
  --set seed=0 log.dir=./artifacts_D1 > run_D1.log 2>&1
.venv/bin/python scripts/run_g1.py --config configs/revisit_U4_hope_tuluyen.yaml \
  --set seed=0 log.dir=./artifacts_D2 \
        memory.gate_bound.alpha_logit_center=0.0 memory.gate_bound.alpha_logit_limit=2.9444 \
  > run_D2.log 2>&1
.venv/bin/python scripts/run_g1.py --config configs/revisit_U4_hope_tuluyen.yaml \
  --set seed=0 log.dir=./artifacts_D3 cms.tiers=[[12,1]] cms.etas=[1.0] \
  > run_D3.log 2>&1
' > d_chandoan.log 2>&1 &
```

**Đọc — bốn cửa cơ chế trước, số sau:**

```bash
tr '\r' '\n' < run_D1.log | grep -E "α=|norm\(state\)|‖Δw‖" | cut -c1-95
grep -A4 "ĐÒN A" run_D1.log          # Linear vs NCM-head trên CÙNG feature
```

| Cửa | Phải thấy | Ý nghĩa |
|---|---|---|
| α | trong dải, **biến thiên**, không bão hoà | Eq 76 sống |
| `norm(state)` | **tăng** qua 12 chuyến | ký ức tích luỹ — lần đầu |
| `‖Δw‖` | **đổi ở MỌI chuyến**, `fast > mid > slow` | ⭐ **CMS chạy suốt** — điều bất khả ở bài toán cũ |
| NCM-head vs Linear | chênh bao nhiêu | tách bộ đọc khỏi kiến trúc |

**Kết luận rút từ D1/D2/D3:**

```
D1 − D2  =  đóng góp của việc dẫn lại thang α
D1 − D3  =  đóng góp của CMS đa tần số   ← chưa ai đo được con số này
```

## T3.3. Ba seed cấu hình thắng — ~10h

**Số công bố:** `O1(U4) − O1(U5)` — so **một biến**: kiến trúc NL vs công thức đóng, **cùng chế độ tự luyện**.

---

# T4 — Trần trên có nhãn (~3h, ưu tiên thấp)

```bash
for S in 0 1 2; do
  .venv/bin/python scripts/run_g1.py --config configs/revisit_arm1_lam1.yaml \
    --set seed=$S log.dir=./artifacts_arm1_s$S train.eval_chi_duong_cheo=true \
    > run_arm1_s$S.log 2>&1
done
```

Cho biết pha-2-không-nhãn đang trả giá bao nhiêu. Không có nó thì không biết 0,63 là tốt hay tệ.

# T5 — Bench O4 lại (~15 phút)

```bash
.venv/bin/python scripts/bench_ba_tang.py 2>&1 | tee bench_tuluyen.log
```

Cộng thêm buffer 0,69 MB. Ba tầng: ~2,2/10 MB ✅. HOPE: ~103 MB ❌ — reframe **không** cứu O4.

# T6 — Dọn nợ kỹ thuật (~1h, làm cuối)

| | Sửa |
|---|---|
| `run_g1.py` | cảnh báo `"giảm nguong"` → `"tăng nguong"` (ngược logic) |
| `optim/__init__.py` | truyền `eps` vào M3 (đang bị nuốt âm thầm) |
| `configs/*.yaml` × 5 | `"Delta Momentum (Eq.48-49)"` → `"EMA hai núm — KHÔNG phải Eq.48-49"` |
| `configs/default.yaml` | bỏ `muon` (không tồn tại) |
| `configs/revisit_U2_*.yaml` | chú thích `nguong` lỗi thời |

---

# BẢNG KẾT QUẢ ĐÍCH

```
BÀI TOÁN UAV — revisit, 12 chuyến, pha 2 không nhãn mới, 3 seed

                              O1          cập nhật pha 2      MB       ms
  U0  đóng băng            0.596 ± 0.018   không              1.50    0.007
  U1  + tầng nhanh         0.628 ± 0.007   thống kê           1.51    0.054
  U2  + ngân hàng          0.628 ± 0.007   (M2 ≈ 0, đóng)     1.53    0.069
  U3  HOPE                 0.575 ± 0.015   không            102.75    2.178
  ────────── SAU KHI ĐỔI BÀI TOÁN ──────────
  U5  ba tầng + tự luyện   _____           nhãn giả           2.20    0.069   ← T2
  U4  HOPE + tự luyện      _____           gradient ×12     103.4     2.178   ← T3 ⭐
  arm1 trần có nhãn        _____           nhãn thật          1.50    0.007   ← T4

  U5 − U1  =  ____ ± ____    giá trị của TỰ LUYỆN
  U4 − U5  =  ____ ± ____    giá trị của NESTED LEARNING (cùng chế độ)  ⭐
  D1 − D3  =  ____           giá trị của CMS ĐA TẦN SỐ
```

---

# LỊCH CHẠY

| | Việc | Thời gian | Máy |
|---|---|---|---|
| **Hôm nay** | T0 phát biểu bài toán | 30 phút | — |
| | T1 hạ tầng + test | ~3h | Mac |
| **Đêm 1** | T2.2 dò ngưỡng (3 run) | ~3h | `uavcl-slda` |
| **Đêm 1** | (song song) T4 oracle 3 seed | ~3h | `uavcl-titans` |
| **Đêm 2** | T2.3 U5 ba seed | ~3h | `uavcl-slda` |
| **Đêm 2** | T3.2 D1/D2/D3 | ~10h | `uavcl-titans` |
| **Đêm 3** | T3.3 U4 ba seed | ~10h | `uavcl-titans` |
| Cuối | T5 bench · T6 dọn nợ | ~1h | Mac |

Tổng ~3 đêm, hai máy song song.

---

# RỦI RO

| # | Rủi ro | Mức | Khử |
|---|---|---|---|
| R1 | **Tích luỹ lỗi** — tự luyện cải thiện rồi sụp | 🔴 cao | Ngưỡng tin cậy (T2.2) · vế neo CE pha 1 · theo dõi đường chéo 12 chuyến |
| R2 | U5 không hơn U1 → tự luyện tự nó vô dụng | 🟡 | **Vẫn là kết quả.** Và T3 mất ý nghĩa → dừng sớm, tiết kiệm 20h |
| R3 | U4 vẫn thua U5 dù NL có việc làm | 🟡 | Kết luận mạnh cho khung B: *"NL được thi đấu đúng luật vẫn thua, và đây là ba lý do"* |
| R4 | Hết giờ | 🟡 | Thứ tự T0→T1→T2→T3 là **đường găng**. T4/T5/T6 cắt được |
| R5 | Rò rỉ nhãn vào vế học | 🔴 | **Test #4** (xáo `y`, `μ_c` phải giống hệt) — bắt buộc |

---

# BA LUẬN ĐIỂM SAU KHI XONG

**① NL là khung đúng, nhưng hiện thực hoá gradient cần một chế độ mà bài toán gốc không có.**
Với luật *"không cập nhật ở pha 2"*, CMS chỉ có **1 đợt gradient** trên 12 chuyến → đa tần số không có việc làm; α bò lên **96% trần** → ký ức không tích luỹ (`norm` phẳng 54,4–54,7); kết quả thua công thức đóng **5,3 điểm** (t=−5,23).

**② Định nghĩa lại bài toán cho sát thực tế thì cả ba đóng góp NL đều được thi đấu.**
Cho phép cập nhật giữa chuyến (không nhãn mới, chỉ đặc trưng nén 0,69 MB) → 12 đợt gradient → CMS chạy suốt, và ta **đo được đóng góp riêng của đa tần số** (D1−D3) — con số chưa ai có.

**③ Chi phí vẫn là ranh giới cứng.**
Ngay cả khi thi đấu đúng luật, HOPE tốn **103 MB** so với **2,2 MB** — vượt ngân sách drone **10,3×**. Kết luận về **khả năng triển khai** độc lập với kết luận về **độ chính xác**.

---

*Cập nhật file này sau mỗi mốc. Đường găng: T0 → T1 → T2 → T3. Mọi thứ khác cắt được.*
