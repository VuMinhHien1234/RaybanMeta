# Kế hoạch vá cổng η trước khi chạy D11

Lập 2026-08-03, sau khi rà `fix/nl-gates-alive` (xem `DIEM_YEU_NHANH_GATES_2026-08-03.md`)

---

## ⛔ Điều đầu tiên: ĐỪNG `git pull` trên VM lúc này

D10 (15 run SLDA) đang chạy. Vòng lặp gọi `run_g1.py` **lại từ đầu mỗi run**, nên `git pull`
giữa chừng sẽ khiến các arm còn lại nạp code mới. Arm 1 và arm 2–5 đo bằng hai bản code khác
nhau, **không có gì báo lỗi**, 10 tiếng thành vô nghĩa.

Toàn bộ kế hoạch dưới đây làm **trên Mac**. Chỉ `git pull` lên VM sau khi thấy `=== D10 XONG`.

Tin tốt: không việc nào dưới đây đụng vào SLDA, nên D10 đang chạy vẫn hợp lệ.

---

## Vấn đề, phát biểu ngắn gọn

`eta_logit_limit: 2.1972` chọn khi còn tưởng `max_lr = 1e-2` → dự định η ∈ [1e-3, 9e-3].
`max_lr` thật là **1,0** → thực tế η ∈ [0,1 · 0,9], **gấp 100 lần**.

Log 9 task: η_real đi `0,648 → … → 0,896`, trần `0,900` → **99,5% của trần**.
α đi về sàn: `0,068` với sàn `0,050` → cách sàn **2%** của khoảng.

Cả hai cổng đều bão hoà ở biên. Cổng bão hoà = hằng số = **không còn phụ thuộc dữ liệu** —
mà đó chính là Eq 76, là toàn bộ luận điểm của self-modifying memory.

Nếu chạy D11 nguyên trạng và Titans thua, ta không phân biệt được **ý tưởng NL yếu** với
**một tham số lệch đơn vị**. Đó là nhiễu biến phải loại trước, không phải sau.

---

# Các task

Ký hiệu: ⛔ chặn D11 · ⚪ nên làm nhưng không chặn

---

## ⛔ G1 — `gate_bound` không được phép tắt âm thầm

**Vì sao.** `memory.py:155` trả `installed or None` khi không tìm thấy
`to_adaptive_step`/`to_decay_factor`. `titans_head.py:127` chỉ `if _gb: print(...)`. Kết quả:
config ghi `gate_bound:` đầy đủ, test xanh, run chạy tới cuối — mà bản vá **đã tắt**. Phải để
ý sự *vắng mặt* của một dòng log mới biết.

**Sửa.** Trong `_install_gate_bounds`, sau vòng cài hook:

```python
if not installed:
    raise RuntimeError(
        "memory.gate_bound được yêu cầu nhưng không tìm thấy to_adaptive_step / "
        "to_decay_factor trên đối tượng memory — thư viện đã đổi tên? "
        "Config yêu cầu chặn cổng mà không chặn được là lỗi, không phải cảnh báo.")
```

Và ở `titans_head.py`, đổi `if _gb: print(...)` thành in **cả khi tắt**:

```python
print(f"[titans] gate_bound {'BẬT — ' + _gb if _gb else 'TẮT'}")
```

Để log luôn có một dòng khẳng định, không phải suy ra từ sự vắng mặt.

| File | Thời gian | Rủi ro |
|---|---|---|
| `models/memory.py`, `models/titans_head.py` | 15 phút | thấp |

---

## ⛔ G2 — Cho phép trần **lệch tâm**

**Vì sao.** Trần đối xứng `tanh(x/L)·L` luôn có tâm ở logit 0, tức `η = sigmoid(0)·max_lr = 0,5`.
Muốn η nhỏ thì **không thể** chỉ chỉnh độ rộng — phải dời tâm. Đây là hạn chế cấu trúc của
cài đặt hiện tại, không phải chọn sai hằng số.

**Sửa.** Đổi `_make(limit)` thành `_make(nua, tam=0.0)`:

```python
def _make(nua, tam=0.0):
    def _bound_hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if not torch.is_tensor(t):
            return None
        b = torch.tanh((t - tam) / nua) * nua + tam       # tâm=0 -> y hệt công thức cũ
        return (b,) + tuple(out[1:]) if isinstance(out, tuple) else b
    return _bound_hook
```

Config nhận thêm hai khoá, **mặc định 0,0 → hành vi cũ bất biến**:

```yaml
gate_bound:
  alpha_logit_limit: 2.9444
  alpha_logit_center: 0.0
  eta_logit_limit: 2.1972
  eta_logit_center: 0.0
```

`gate_bound_report()` phải in khoảng **tuyệt đối** đã tính từ (tâm, nửa) chứ không giả định
tâm 0 như hiện nay.

| File | Thời gian | Rủi ro |
|---|---|---|
| `models/memory.py` | 45 phút | trung bình — phải giữ bất biến khi tâm=0 |

---

## ⛔ G3 — Tự cảnh báo khi cổng bão hoà ở biên

**Vì sao.** Lỗi này tồn tại 1 ngày mà không ai thấy, vì log in η=0,896 mà **không in trần**.
Người đọc không có mốc để biết 0,896 là "sát trần" hay "giữa dải". Máy phải tự nói ra.

**Sửa.** Trong `methods.py` chỗ log cổng, tính vị trí tương đối trong khoảng:

```python
frac = (eta_real - lo) / (hi - lo)          # 0 = sàn, 1 = trần
print(f"[titans]   η={eta_real:.4f} ({frac*100:.0f}% khoảng [{lo:.3f},{hi:.3f}])")
if frac > 0.90:  print("[titans]   ⚠️ η BÃO HOÀ Ở TRẦN — cổng thành hằng số, hết phụ thuộc dữ liệu")
if frac < 0.10:  print("[titans]   ⚠️ η BÃO HOÀ Ở SÀN")
```

Tương tự cho α. Kiểm ngược trên số đã có: η task 8 cho `frac = (0,896−0,1)/0,8 = 0,995` → phải
kêu. α cho `frac = (0,068−0,05)/0,9 = 0,020` → phải kêu.

| File | Thời gian | Rủi ro |
|---|---|---|
| `methods.py` | 30 phút | thấp |

---

## ⛔ G4 — Sửa comment sai 100× trong config

```yaml
eta_logit_limit: 2.1972       # SAI: "η ∈ [1e-3, 9e-3]"   -> ĐÚNG: η ∈ [0.1, 0.9] khi max_lr=1.0
init_adaptive_step_bias: 0.0  # SAI: "η khởi đầu = 5e-3"   -> ĐÚNG: η khởi đầu = 0.5
```

Sửa ở `g2_titans_resisc45_selfmod_m3_gates.yaml`, `_cms.yaml`, `drift_titans_gates.yaml`.
`memory.py:36` đã ghi đúng — đây là sửa cho hai chỗ khỏi mâu thuẫn.

| File | Thời gian | Rủi ro |
|---|---|---|
| 3 config | 5 phút | không |

---

## ⛔ G5 — Test bắt được lỗi thang **tuyệt đối**

**Vì sao.** `ETA_FRAC_LO, ETA_FRAC_HI = 0.01, 0.95` đo η **theo phân số của max_lr** — chính
cách đóng khung đó khiến sai số thang đo trở nên vô hình. Trần hiện tại cho phân số 0,1–0,9,
nằm gọn trong ngưỡng → test xanh dù thang tuyệt đối lệch 100 lần.

Cùng loại lỗ hổng với thang λ sáng nay: **cài đúng, chỉnh sai**. Test chỉ kiểm cơ chế thì
không bao giờ bắt được sai hiệu chỉnh.

**Thêm các test:**

| Test | Kiểm gì |
|---|---|
| `test_bound_lech_tam_cho_dung_khoang` | tâm=−2,2 nửa=2,2 → η ∈ [0,0121 · 0,5], sai số <1% |
| `test_tam_0_trung_khop_ban_cu` | tâm=0 → **bằng bit** với công thức `tanh(x/L)·L` |
| `test_report_in_khoang_tuyet_doi` | báo cáo in đúng khoảng đã dời tâm, không giả định tâm 0 |
| `test_raise_khi_khong_cai_duoc_hook` | G1 — memory giả không có 2 thuộc tính → `RuntimeError` |
| `test_canh_bao_bao_hoa_kich_hoat` | nạp η=0,896 với trần 0,9 → phải sinh cảnh báo |
| `test_khoang_eta_tuyet_doi_hop_ly` | ⭐ η_max < 0,5·max_lr — **test chặn đúng lỗi hôm nay** |

| File | Thời gian | Rủi ro |
|---|---|---|
| `tests/test_gate_bound.py` | 45 phút | thấp |

---

## ⚪ G6 — Đệm bằng zero thay vì lặp frame cuối

`memory.py:217` đệm bằng `seq[:, -1:, :].expand(-1, pad, -1)`. Output được cắt lại, nhưng
`state` thì không — bộ nhớ **đã học** từ frame lặp. Với `reset: never`, ảnh cuối mỗi batch lẻ
được ghi nhiều lần hơn mọi ảnh khác.

Nhỏ (`pad=0` cho ~65/66 batch) nhưng **có hệ thống**, không ngẫu nhiên. Không chặn D11 vì nó
ảnh hưởng ctrl và gates như nhau. Ghi lại để sửa sau.

---

## ⚪ G7 — Probe cộng `sum`/`numel` thay vì `mean`

`_eta_sum += float(d.mean())` là trung bình không trọng số của các trung bình chunk. Sai lệch
nhỏ, không đổi kết luận nào. Sửa khi cần số η chính xác.

---

# Thứ tự và cửa chặn

```
G4 (5')  →  G1 (15')  →  G2 (45')  →  G3 (30')  →  G5 (45')
                                                      ↓
                                            pytest phải xanh hết
                                                      ↓
                                        ⛔ CỬA CHẶN: smoke 3 task
```

**Cửa chặn — smoke trên Mac, ~20 phút:**

```bash
cd ~/Desktop/Raybanmeta/uav-continual-learning
.venv/bin/python -m pytest tests/test_gate_bound.py -q

.venv/bin/python scripts/run_g1.py \
  --config configs/drift_titans_gates.yaml \
  --set seed=0 data.num_tasks=3 log.dir=./artifacts_smoke_eta \
  2>&1 | tee run_smoke_eta.log
```

Ba dòng phải thấy:

| Dòng | Phải là |
|---|---|
| `[titans] gate_bound BẬT — η∈[...]` | khoảng **mới**, không phải [0,100 · 0,900] |
| `[titans] η=… (X% khoảng […])` | có in phần trăm — G3 đã chạy |
| `[drift] BẬT` | trôi vẫn bật |

Tổng: **~2,5 giờ code + 20 phút kiểm.**

---

# Quyết định cần bạn chọn: chạy D11 thế nào

Sửa xong code thì vẫn còn một câu **không phải câu về code**: đặt trần η ở đâu?

Bốn ứng viên, cùng nửa rộng nhưng khác tâm:

| Tâm | Nửa | η ∈ | Tâm η | Nhận xét |
|---:|---:|---|---:|---|
| 0,0 | 2,20 | [0,100 · 0,900] | 0,500 | **nguyên trạng** — đang bão hoà ở trần |
| −2,20 | 2,20 | [0,012 · 0,500] | 0,100 | thấp hơn 5×, còn rộng 40× |
| −3,00 | 2,20 | [0,005 · 0,269] | 0,047 | thấp hơn 10× |
| −5,81 | 1,10 | [0,001 · 0,009] | 0,003 | **dự định ban đầu** — rất hẹp, chỉ rộng 9× |

Tôi không biết đáp án. Có thể η cao **đúng là** thứ Titans cần trên stream trôi — trôi nhanh
thì phải ghi nhanh. Nhưng phải là lựa chọn có ý thức, không phải tai nạn về đơn vị.

**Đề nghị: thử 1 seed × 2 cấu hình trước (2 run, ~2,7 giờ), rồi mới dồn 3 seed.**

Đúng cách làm đã cứu ta hôm nay ở `calibrate_drift`: kiểm rẻ trước, đốt giờ máy sau.
Nhìn hai thứ ở 2 run đó — η có còn bão hoà không, và accuracy chênh bao nhiêu — rồi mới quyết
dồn 3 seed vào cấu hình nào.

| | Cách chạy D11 | Chi phí | Đổi lại |
|---|---|---|---|
| A | 1 seed × 2 trần → chọn → 3 seed | ~2,7h + ~4h | Có cơ sở chọn, không đoán |
| B | 3 seed × 2 trần luôn | ~8h | Tách được ảnh hưởng của trần, xong trong một đêm |
| C | 3 seed × 1 trần (chọn tay ngay) | ~4h | Rẻ nhất, nhưng nếu chọn sai thì mất cả 4h |
