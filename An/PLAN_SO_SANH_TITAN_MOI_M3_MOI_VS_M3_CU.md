# Plan so sánh Titan mới + M3 mới với Titan mới + M3 cũ

## 1. Câu hỏi nghiên cứu

Mục tiêu duy nhất:

> Khi giữ nguyên Titan Task 4 v2, M3 mới có chính xác hơn, ít quên hơn và ổn định
> số tốt hơn M3 cũ hay không?

Không dùng kết quả `Titan cũ + M3 cũ` để trả lời câu hỏi này. Kết quả đó chỉ là
mốc lịch sử. Hai đối tượng phải chạy lại trên cùng nhánh `Titan_M3`:

| Nhóm | Titan | M3 | Vai trò |
|---|---|---|---|
| Control | Task 4 v2, self-modifying value-only | M3 cũ | Mốc đối chứng |
| Treatment | Task 4 v2, self-modifying value-only | M3 mới | Phương án cần kiểm tra |

## 2. Vì sao không thể chỉ chạy một seed

M3 cũ từng cho hai hành vi rất khác nhau trên RESISC45:

| Seed | Linear Acc / Forgetting | `norm(state)` |
|---|---:|---:|
| 0 | 0.6214 / 0.2420 | khoảng 54, ổn định |
| 1 | 0.2508 / 0.5643 | khoảng 1.644.728, nổ |

Seed 0 một mình có thể tạo kết luận quá lạc quan. Vì vậy:

- Seed 1 là stress seed, dùng để phát hiện nổ sớm.
- Kết luận cuối phải có ít nhất seed 0, 1, 2.
- Báo cáo cả từng seed và `mean ± std`; không chỉ báo cáo seed tốt nhất.

## 3. Định nghĩa M3 cũ và M3 mới

Hai config đã có:

- `configs/g2_titans_resisc45_selfmod_m3_legacy.yaml`
- `configs/g2_titans_resisc45_selfmod_m3_improved.yaml`

Các phần được khóa giống nhau:

- RESISC45 đủ 31.500 ảnh, split 80/10/10 tái lập được.
- ViT-S pretrained và frozen.
- Titan depth 3, heads 1.
- Self-modifying chỉ ở value; `self_modifying_readpath=false`.
- Memory sống xuyên task: `reset=never`.
- 9 task, 3 epoch/task, batch 32.
- M3 delta-approx, frequency 16, cùng alpha/eta và weight decay.

Gói M3 cũ:

- `update_norm=rms`.
- Tạo lại optimizer sau mỗi task.
- Không có outer gradient clipping.

Gói M3 mới:

- `update_norm=clip`.
- Giữ state optimizer xuyên task.
- `grad_clip_norm=1.0`.
- Có kiểm tra NaN/Inf và bản vá zero-norm cho internal Titans clipping.

Lưu ý: so hai config trên trả lời “recipe M3 mới có tốt hơn recipe cũ không”.
Nó chưa cho biết thành phần nào tạo ra cải thiện. Phần ablation ở mục 7 sẽ trả lời
câu hỏi đó.

## 4. Chỉ số đánh giá

### Chỉ số chính

1. Linear-head Average Accuracy: cao hơn là tốt hơn.
2. Linear-head Average Forgetting: thấp hơn là tốt hơn.
3. Độ ổn định:
   - Không NaN/Inf trong logits, loss, gradient và memory state.
   - `norm(state) < 10.000` ở mọi task.
   - Không có bước nhảy norm lớn hơn 10 lần giữa hai task liên tiếp.

Linear head là chỉ số chính vì head và memory parameters được M3 cập nhật trực tiếp.

### Chỉ số phụ

1. NCM-head Average Accuracy và Forgetting trên cùng feature sau memory.
2. Accuracy của từng task trong `acc_matrix.csv`.
3. `eta_t`, `alpha_t`, `self-mod branch strength` theo task.
4. Backward Transfer.
5. Thời gian chạy và peak memory nếu runner đo được.

NCM-head phải được bật ở cả hai nhóm bằng `train.eval_ncm_head=true`.

## 5. Điều kiện để gọi M3 mới là tốt hơn

Không kết luận chỉ dựa trên một con số accuracy.

### Kết luận “M3 mới tốt hơn”

Phải thỏa cả ba:

1. Cả 3 seed hữu hạn và không có `norm(state) >= 10.000`.
2. Mean Linear Accuracy tăng ít nhất 0.02 tuyệt đối so với M3 cũ.
3. Mean Forgetting không tăng quá 0.02.

### Kết luận “M3 mới ổn định hơn nhưng accuracy tương đương”

Áp dụng khi:

1. M3 mới hết nổ state trong khi M3 cũ còn nổ ở ít nhất một seed.
2. Accuracy mới không thấp hơn cũ quá 0.01.
3. Forgetting mới không xấu hơn quá 0.02.

### Kết luận “chưa đủ bằng chứng”

- Chênh lệch accuracy dưới 0.02 và độ lệch chuẩn còn lớn.
- Hai bên đều có seed nổ.
- Kết quả thay đổi ngược chiều giữa Linear và NCM mà chưa giải thích được.

Với chỉ 3 seed, không tuyên bố “có ý nghĩa thống kê”. Báo paired difference theo
từng seed, mean, std và range.

## 6. Quy trình chạy chính

### Phase 0 - Kiểm tra trước khi train

- Checkout `Titan_M3`.
- Ghim cùng version Python, PyTorch, torchvision, timm và `titans-pytorch==0.5.5`.
- Chạy toàn bộ test; yêu cầu `83 passed`.
- Kiểm tra MPS thực sự được chọn khi config là `device:auto`.
- Kiểm tra RESISC45 có đúng 31.500 ảnh, 700 ảnh/class.
- In config cuối cùng của từng run và kiểm tra seed override có hiệu lực.

Nếu Phase 0 lỗi thì không chạy benchmark.

### Phase 1 - Smoke tích hợp

Chạy hai recipe trên synthetic, 3 task, seed 0:

- Legacy: RMS, optimizer reset per task.
- Improved: clip, optimizer giữ xuyên task, grad clip 1.0.

Yêu cầu:

- Chạy hết stream.
- Không NaN/Inf.
- Có `metrics.json`, `train_log.json`, `memory_state.pt`.
- Tên artifact phân biệt rõ `delta_rms` và `delta_clip`.

Phase này đã chạy thành công; không dùng accuracy synthetic để kết luận.

### Phase 2 - Matched-LR trên stress seed

Chạy RESISC45 seed 1 với LR `1e-3`:

| Run | Recipe | LR | Seed |
|---|---|---:|---:|
| S2-A | M3 cũ | 1e-3 | 1 |
| S2-B | M3 mới | 1e-3 | 1 |

Đây là phép so drop-in sạch nhất: cùng Titan, data, LR và seed.

Gate:

- Nếu M3 mới NaN/Inf hoặc `norm(state) >= 10.000`, dừng và phân tích trước khi sweep.
- Nếu M3 mới hữu hạn còn M3 cũ tái hiện vụ nổ, đã có bằng chứng ổn định ban đầu,
  nhưng vẫn chưa được kết luận accuracy trước khi chạy 3 seed.

### Phase 3 - Tìm LR phù hợp cho M3 mới

Chạy M3 mới trên stress seed 1:

`LR = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]`

Thứ tự chạy để giảm rủi ro:

`3e-4 -> 1e-3 -> 3e-3 -> 5e-3 -> 1e-4`

Mỗi run dùng đủ 9 task. Không chọn LR từ smoke hoặc stream rút gọn vì vụ nổ của
Titan cũ thường chỉ lộ khi state đã tích lũy lâu.

Score chọn LR:

`score = accuracy - max(forgetting, 0) - instability_penalty`

Trong đó:

- `instability_penalty = 10` nếu có NaN/Inf hoặc norm >= 10.000.
- `instability_penalty = 0.05` nếu norm tăng hơn 10 lần giữa hai task.
- Nếu score gần nhau trong 0.01, ưu tiên LR có norm thấp hơn và std loss nhỏ hơn.

Không tune M3 cũ lại ở phase này; M3 cũ là baseline lịch sử tại LR `1e-3`.

### Phase 4 - Chạy xác nhận 3 seed

Chạy các nhóm sau với seed `0, 1, 2`:

| Nhóm | Recipe | LR | Số run |
|---|---|---:|---:|
| A | M3 cũ | 1e-3 | 3 |
| B | M3 mới matched-LR | 1e-3 | 3 |
| C | M3 mới best-LR từ Phase 3 | LR được chọn | 3 |

Nếu best-LR chính là `1e-3`, nhóm B và C là một, tổng còn 6 run.

Hai phép so cần báo riêng:

1. A vs B: M3 mới có tốt hơn khi thay trực tiếp ở cùng LR không?
2. A vs C: Sau khi tune hợp lý, M3 mới đạt trần tốt hơn M3 cũ không?

Không được lấy seed tốt nhất của C so với mean của A.

## 7. Ablation để biết cải thiện đến từ đâu

Chỉ làm sau Phase 4, hoặc làm sớm trên seed 1 nếu cần chẩn đoán.

Giữ LR cố định ở LR tốt nhất của M3 mới:

| Ablation | Norm update | Optimizer qua task | Grad clip |
|---|---|---|---|
| A0 - legacy | RMS | reset | off |
| A1 - clip only | clip | reset | off |
| A2 - clip + carry | clip | giữ | off |
| A3 - full improved | clip | giữ | 1.0 |

Chạy seed 1 trước. Nếu hai cấu hình đứng đầu chênh accuracy dưới 0.02, chạy thêm
seed 0 và 2 cho hai cấu hình đó.

Cách diễn giải:

- A1 tốt hơn A0: `update_norm=clip` là thành phần chính.
- A2 tốt hơn A1: memory gradient của M3 cần sống xuyên task.
- A3 ổn định hơn A2: outer gradient clipping có tác dụng bảo vệ.
- A3 không hơn A2: grad clipping có thể không cần thiết ở recipe cuối.

## 8. Logging và giám sát

Mỗi run phải lưu:

- Config cuối cùng sau override.
- `metrics.json` và `metrics_ncm.json`.
- `acc_matrix.csv` và ma trận NCM.
- `train_log.json`.
- Stdout/stderr riêng.
- `failure.json` nếu dừng vì NaN, Inf hoặc norm threshold.
- Commit SHA, hostname, device, package versions và thời gian bắt đầu/kết thúc.

Sau mỗi task, kiểm:

- train loss có giảm hợp lý không;
- accuracy từng task;
- `state_finite`;
- `norm(state)`;
- `eta_t`, `alpha_t`;
- `beta` và `|W_state|` của nhánh self-modifying.

Nếu norm vượt 10.000, run bị đánh dấu fail. Không dùng metrics của run đó để tính
mean accuracy như một run hợp lệ.

## 9. Tự động hóa cần implement

Tạo `scripts/run_titan_m3_study.py` với các phase:

1. `check`: test môi trường, config và dataset.
2. `stress`: Phase 2.
3. `sweep`: Phase 3.
4. `replicate`: Phase 4.
5. `ablate`: Phase 7.
6. `summarize`: đọc mọi artifact và sinh bảng Markdown/CSV.

Runner phải:

- Resume run đã có metrics hợp lệ.
- Không ghi đè artifact cũ.
- Dừng riêng run lỗi nhưng tiếp tục campaign.
- Lưu command và failure reason.
- Xác minh config legacy/improved chỉ khác whitelist cho phép.

## 10. Bảng kết quả cuối cần có

| Recipe | LR | Valid seeds | Exploded seeds | Linear Acc mean±std | Linear Fgt mean±std | NCM Acc mean±std | NCM Fgt mean±std | Max state norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M3 cũ | 1e-3 | | | | | | | |
| M3 mới matched | 1e-3 | | | | | | | |
| M3 mới tuned | TBD | | | | | | | |

Thêm bảng paired difference:

| Seed | New - Old Linear Acc | New - Old Forgetting | New - Old NCM Acc | Norm cũ | Norm mới |
|---:|---:|---:|---:|---:|---:|
| 0 | | | | | |
| 1 | | | | | |
| 2 | | | | | |

## 11. Thứ tự thực hiện khuyến nghị

1. Implement study runner và test logic command/config.
2. Chạy `check` và smoke.
3. Chạy stress seed 1 ở LR 1e-3.
4. Nếu M3 mới ổn định, chạy LR sweep trên seed 1.
5. Chọn LR bằng rule đã khóa trước, không chọn bằng cảm tính.
6. Chạy legacy, matched và tuned trên seed 0/1/2.
7. Tổng hợp Linear, NCM, forgetting và norm.
8. Chạy ablation nếu cần giải thích nguyên nhân.
9. Viết báo cáo kết luận, vấn đề còn lại và config đề xuất.

## 12. Kết quả mong đợi

Kết quả tốt nhất không nhất thiết là accuracy cao nhất ở một seed. Mục tiêu thực tế là:

- M3 mới loại được vụ nổ state của seed 1.
- Accuracy trung bình tăng hoặc ít nhất không giảm đáng kể.
- Forgetting giảm.
- NCM-head tiếp tục đạt mốc `Acc >= 0.72` và `Forgetting <= 0.10`.
- Kết quả nhất quán qua seed để có thể dùng làm kết quả chính của project.

