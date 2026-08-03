# Điểm yếu trong code nhánh `fix/nl-gates-alive`

Rà lại ngày 2026-08-03 · `memory.py` (246 dòng), `titans_head.py`, config `*_gates.yaml`,
đối chiếu với log thật `res_titans_gates/run_9t_gates_s0.log`

> `memory.py` **y hệt** giữa `fix/nl-gates-alive` và `feat/drift-lambda` (diff trống),
> nên mọi điểm dưới đây áp dụng cho **cả hai nhánh**, kể cả `drift_titans_gates.yaml`
> sắp dùng cho D11.

---

## ⛔ 1. Trần η bị chỉnh sai thang **100 lần** — cổng đã ngừng là cổng

Đây là điểm nặng nhất.

`eta_logit_limit: 2.1972` được chọn với giả định `max_lr = 1e-2`, cho η ∈ [1e-3, 9e-3].
Nhưng `max_lr` **thật là 1,0** (đã phát hiện hôm 08-02: `neural_memory.py:272` đặt
`default_step_transform_max_lr = 1.` và `:457` luôn ghi đè). Cùng một trần logit giờ cho:

```
sigmoid(±2.1972) × 1.0  =  η ∈ [0.100, 0.900]      ← gấp 100× dự định
```

**Bằng chứng từ log thật** — η_real qua 9 task:

```
0,6483 → 0,6343 → 0,7204 → 0,8348 → 0,8426 → 0,8456 → 0,8751 → 0,8782 → 0,8957
                                                                          ↑
                                                              trần = 0,900
```

Task 8 đạt **99,5% của trần**. η vẫn **trôi đơn điệu**, chỉ là đâm vào trần thay vì bay đi.

Hệ quả thật sự không phải "con số to" mà là: **η ở task cuối không còn phụ thuộc dữ liệu
nữa — nó là hằng số 0,9.** Mà learning-rate phụ thuộc dữ liệu chính là Eq 76, là toàn bộ
điểm bán hàng của self-modifying memory. Cơ chế đã thoái hoá thành một hằng số.

Điều này giải thích tại sao cơ chế "sống lại" (α dao động thật) mà accuracy chỉ nhích:
một trong hai cổng đã chết theo kiểu khác — chết vì bão hoà ở trần.

### Sửa thế nào

Trần đối xứng quanh 0 luôn cho tâm `sigmoid(0)·max_lr = 0,5`. Muốn η nhỏ thì phải **lệch tâm**:

```python
# muốn η ∈ [1e-3, 9e-3] với max_lr = 1.0  ->  logit ∈ [−6.91, −4.71], tâm −5.81
logit_bounded = tanh((x − tam) / nua) * nua + tam        # tam=−5.81, nua=1.10
```

Hoặc đơn giản hơn: truyền `max_lr=1e-2` xuống `NeuralMemory` để trần đối xứng chạy đúng như
thiết kế ban đầu. **Nên chọn cách nào là quyết định thí nghiệm, không phải quyết định code** —
vì η=0,9 có thể là điều Titans *cần* trên stream trôi. Nhưng phải là lựa chọn có ý thức,
không phải tai nạn về đơn vị.

---

## 2. Cả hai cổng vẫn trôi về biên — bản vá đổi biên, không đổi hướng

α qua 9 task (sàn = 0,050):

```
0,2637 → 0,0891 → 0,2625 → 0,2135 → 0,2311 → 0,0876 → 0,0869 → 0,1801 → 0,0678
```

Dao động thật (đó là kết quả dương), nhưng xu thế đi **xuống sàn**. `keep = 1−α = 0,9322` ở
task 8: bộ nhớ giữ lại 93% mỗi chunk, gần như không quên.

Cộng với mục 1: **η ép trần, α ép sàn.** Cả hai cổng đều đang bị gradient đẩy vào biên, đúng
như đã cảnh báo — `gate_bound` chỉ dời biên ra xa, không đổi được **hướng** áp lực. Giờ có số
để nói điều đó thay vì chỉ suy đoán.

---

## 3. `gate_bound` có thể tự tắt **âm thầm**

`memory.py:155`:

```python
return installed or None
```

Nếu thư viện đổi tên `to_adaptive_step` / `to_decay_factor`, hoặc bản self-modifying bọc lại
module, thì `hasattr` = False → không cài hook nào → trả `None`.

Rồi `titans_head.py:125-127`:

```python
_gb = self.memory.gate_bound_report() ...
if _gb:
    print(f"[titans] gate_bound BẬT — {_gb}")
```

`None` → **không in gì**. Bạn phải nhận ra sự **vắng mặt** của một dòng log mới biết bản vá
đã tắt. Config vẫn ghi `gate_bound:` đầy đủ, test vẫn xanh, run vẫn chạy tới cuối.

**Nên `raise`**: config yêu cầu `gate_bound` mà không cài được hook nào là lỗi cấu hình,
không phải trường hợp im lặng bỏ qua.

---

## 4. Comment trong config sai 100× — bẫy cho người đọc sau

`configs/g2_titans_resisc45_selfmod_m3_gates.yaml`:

```yaml
eta_logit_limit: 2.1972     # η ∈ [1e-3, 9e-3]        ← SAI, thật là [0.1, 0.9]
init_adaptive_step_bias: 0.0  # η khởi đầu = 0.5*max_lr = 5e-3   ← SAI, thật là 0.5
```

Cả hai comment còn giữ giả định `max_lr=1e-2` cũ. Ai đọc config sẽ tưởng η rất nhỏ và kết luận
sai về hành vi. `memory.py:36` thì ghi **đúng** (`η ∈ [0.1, 0.9] * max_lr`) — hai chỗ mâu thuẫn
nhau, và chỗ sai lại là chỗ người ta đọc trước.

---

## 5. Bộ test không thể bắt được mục 1

`tests/test_gate_bound.py` có 16 test, trong đó:

```python
ETA_FRAC_LO, ETA_FRAC_HI = 0.01, 0.95     # ngưỡng tính bằng PHÂN SỐ của max_lr
def test_gate_bound_giu_eta_trong_vung_lanh_manh(bias): ...
```

Đo η **theo phân số của max_lr** — chính là cách đóng khung khiến sai số thang đo trở nên vô
hình. Trần hiện tại cho phân số 0,1–0,9, nằm gọn trong [0,01, 0,95] → **test xanh**, dù thang
tuyệt đối lệch 100 lần.

Các test kiểm *cơ chế* (tanh chặn đúng, gradient không chết, vẫn phụ thuộc dữ liệu) — đều tốt.
Nhưng **không test nào kiểm sự hiệu chỉnh**: "chặn ở đâu thì hợp lý". Cùng loại lỗ hổng với
thang λ hôm nay — cài đúng, chỉnh sai.

---

## 6. Frame đệm bị **ghi vào bộ nhớ** và không cắt lại được

`memory.py:214-218, 239-240`:

```python
pad = (-L) % self.chunk_size
if pad:
    seq = torch.cat([seq, seq[:, -1:, :].expand(-1, pad, -1)], dim=1)   # lặp frame cuối
...
if pad:
    out = out[:, :L, :]      # cắt OUTPUT
```

Output được cắt, nhưng `state` thì không — bộ nhớ **đã học** từ các frame lặp đó. Với
`reset: never`, ảnh cuối của mỗi batch lẻ được ghi vào ký ức nhiều lần hơn mọi ảnh khác.

Mức độ: `chunk_size=32`, `batch_size=32` → `pad=0` cho hầu hết batch; chỉ batch lẻ cuối mỗi
task mới dính (~1/66 batch). Nhỏ, nhưng **có hệ thống** chứ không ngẫu nhiên, và thống kê η/α
cũng bị đếm trùng theo.

Sửa rẻ nhất: đệm bằng **zero** thay vì lặp frame cuối, hoặc bỏ hẳn batch lẻ khi `reset: never`.

---

## 7. Probe là "trung bình của trung bình" theo chunk

`_eta_sum += float(d.mean())` rồi `_eta_cnt += 1` — mỗi lần hook chạy là một chunk, và các chunk
không cùng số phần tử (chunk cuối có phần đệm). Nên `_eta_sum / _eta_cnt` là trung bình không
trọng số của các trung bình chunk.

Sai lệch nhỏ và không đổi kết luận nào hiện có. Ghi lại để nếu sau này cần số η chính xác thì
biết chỗ sửa: cộng `d.sum()` và `d.numel()` thay vì `d.mean()`.

---

# Tóm tắt và đề nghị

| # | Điểm yếu | Mức | Ảnh hưởng kết luận đã có? |
|---|---|:-:|---|
| 1 | Trần η lệch thang 100×, η ghim ở trần | ⛔ nặng | **Có** — giải thích vì sao accuracy chỉ nhích |
| 2 | Cả hai cổng vẫn trôi vào biên | ⚠️ | Không — đã nói đúng, giờ có số |
| 3 | `gate_bound` tắt âm thầm | ⚠️ | Không lần này (log xác nhận đã bật) |
| 4 | Comment config sai 100× | ⚠️ | Không, nhưng dễ gây sai về sau |
| 5 | Test không bắt được #1 | ⚠️ | Không |
| 6 | Frame đệm ghi vào state | nhẹ | Không |
| 7 | Probe trung bình theo chunk | nhẹ | Không |

**Điều quan trọng nhất cần quyết trước D11:** `drift_titans_gates.yaml` đang dùng **đúng cái
trần η lệch thang này**. Chạy D11 nguyên trạng thì Titans sẽ vào stream trôi với η ghim ở 0,9 —
tức phần "learning rate thích nghi theo dữ liệu" của Nested Learning **không thực sự hoạt động**.

Nếu D11 mà thua, ta sẽ không biết đó là do ý tưởng NL yếu, hay do một tham số bị lệch đơn vị.
Đây đúng là loại nhiễu biến cần loại **trước** khi chạy, không phải sau.

Ba lựa chọn, chi phí khác nhau:

| | Làm gì | Chi phí | Đổi lại |
|---|---|---|---|
| A | Sửa trần η lệch tâm rồi mới chạy D11 | ~1h code + test | D11 đo đúng cơ chế NL |
| B | Chạy D11 nguyên trạng, ghi rõ hạn chế | 0 | Kết quả có chú thích lớn, khó tuyên bố |
| C | Chạy **cả hai** trần trong D11 (thêm 3 run) | +4h máy | Tách được ảnh hưởng của trần η |
