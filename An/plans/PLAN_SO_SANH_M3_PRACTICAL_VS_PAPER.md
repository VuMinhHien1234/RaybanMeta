# Kế hoạch so sánh M3 thực dụng và M3 theo NL.pdf

Ngày lập: 2026-07-25  
Nhánh thực nghiệm: `M3-origin`  
Commit nền khi lập kế hoạch: `4b9a736`  
Project: `/Users/an/Documents/Do An/RaybanMeta/uav-continual-learning`

## 1. Mục tiêu

Giữ nguyên bản M3 hiện tại làm kết quả chính, sau đó triển khai và đánh giá một
nhánh M3 bám sát Algorithm 1, Section 7.2 của `NL.pdf` để trả lời:

1. M3 thực dụng hiện tại hay M3 theo pseudocode của paper đạt accuracy cao hơn?
2. Bản nào chống quên tốt hơn trên EuroSAT và RESISC45?
3. M3 paper có ổn định số trên ViT pretrained + HOPE + CMS không?
4. Nếu M3 paper không ổn định, nguyên nhân đến từ cách tích lũy momentum hay do
   không có update clipping?
5. Chi phí thời gian và bộ nhớ của hai bản khác nhau như thế nào?

Mục tiêu của nhánh này không phải triển khai Delta Momentum Eq. 48-49 đầy đủ.
Delta Momentum là một thuật toán khác với M3 Algorithm 1. Baseline hiện tại vẫn
phải được gọi là `M3-delta-approx-clip`.

## 2. Kết quả nền đã có

Bản thực dụng hiện tại:

```text
M3-delta-approx-clip
LR = 5e-3 nếu ưu tiên accuracy
LR = 3e-3 nếu ưu tiên retention
alpha = 0.5
frequency = 16
optimizer state được giữ xuyên task
```

Kết quả 3 seed đã xác nhận:

| Dataset | Cấu hình | Accuracy, mean +/- sd | Forgetting, mean +/- sd |
|---|---|---:|---:|
| EuroSAT | M3-delta-approx-clip, LR 5e-3 | 0.7103 +/- 0.0465 | 0.0674 +/- 0.0820 |
| RESISC45 | M3-delta-approx-clip, LR 5e-3 | 0.7262 +/- 0.0189 | 0.0412 +/- 0.0254 |
| RESISC45 | M3-delta-approx-clip, LR 3e-3 | 0.6717 +/- 0.0137 | -0.0190 +/- 0.0162 |

Đây là mốc phải giữ nguyên để so sánh. Không được sửa nghĩa của
`beta_style=delta`, không ghi đè artifacts cũ và không dùng tên “Delta Momentum
đầy đủ” cho baseline này.

## 3. M3 paper thực sự là gì?

Algorithm 1 trong `NL.pdf` mô tả:

```text
M1_t = M1_(t-1) + beta1 * g_t
V_t  = V_(t-1)  + beta2 * g_t^2

mỗi f bước:
M2 = M2 + beta3 * sum(gradient trong chunk)

O1 = NewtonSchulz(M1)
O2 = NewtonSchulz(M2)

theta_t = theta_(t-1)
          - lr * (O1 + alpha*O2) / (sqrt(V) + eps)
```

Khác biệt chính với baseline:

| Thành phần | M3-delta-approx-clip hiện tại | M3 Algorithm 1 |
|---|---|---|
| Fast memory M1 | `(delta_alpha-delta_eta)*M1 + delta_eta*g` | `M1 + beta1*g` |
| Slow memory M2 | Delta-approx trên trung bình chunk | Cộng `beta3*sum(chunk)` |
| Second moment V | EMA, có bias correction | Cộng dồn, không bias correction |
| Update norm | Clip Frobenius norm về tối đa 1 | Không có clip trong pseudocode |
| Weight decay | Decoupled, giống AdamW | Không xuất hiện trong Algorithm 1 |
| Global grad clip | Có, norm 1 | Không xuất hiện trong Algorithm 1 |
| Tensor 1D | Không chạy Newton-Schulz | Paper không nói rõ |
| Step clock | Theo tensor/tier được CMS cập nhật | Pseudocode dùng một timeline `t` |

Do paper không công bố đầy đủ mọi hyperparameter và cách xử lý tensor 1D, ta
chỉ có thể gọi bản mới là `M3-paper-pseudocode`, không nên tuyên bố “tái lập
100% code gốc của tác giả”.

## 4. Các biến thể bắt buộc phải so

### P0 - Baseline chính

Tên báo cáo:

```text
M3-delta-approx-clip
```

Cấu hình:

```yaml
train:
  lr: 0.005
  weight_decay: 0.01
  grad_clip_norm: 1.0
  optimizer_per_task: false
  m3:
    beta_style: delta
    update_norm: clip
    alpha: 0.5
    frequency: 16
    delta:
      alpha: [0.999, 0.9999]
      eta: [0.1, 0.05]
```

LR `3e-3` được giữ như preset retention, không thay thế preset accuracy `5e-3`.

### P1 - Paper memory, giữ ổn định của project

Tên báo cáo:

```text
M3-paper-stabilized
```

Mục đích: chỉ thay quy tắc M1/M2/V sang Algorithm 1, còn giữ cùng pipeline để
xác định bản thân cách tích lũy của paper có hữu ích không.

```yaml
train:
  weight_decay: 0.01
  grad_clip_norm: 1.0
  optimizer_per_task: false
  m3:
    beta_style: paper
    update_norm: clip
    alpha: 0.5
    frequency: 16
```

Đây không phải bản paper-strict vì vẫn có hai cơ chế ổn định ngoài pseudocode.

### P2 - Paper pseudocode nghiêm ngặt

Tên báo cáo:

```text
M3-paper-strict
```

Cấu hình ban đầu:

```yaml
train:
  weight_decay: 0.0
  grad_clip_norm: null
  optimizer_per_task: false
  m3:
    beta_style: paper
    update_norm: none
    alpha: 0.5
    frequency: 16
```

P2 dùng để kiểm tra độ trung thành với Algorithm 1. Nó không phải đối chứng
hiệu quả hoàn toàn công bằng với P0 vì đồng thời bỏ weight decay và hai lớp
clipping. Vì vậy không được chỉ chạy P0 và P2 rồi kết luận nguyên nhân.

### P3 - Đối chứng chẩn đoán

Tên báo cáo:

```text
M3-EMA-clip
```

P3 không phải mục tiêu chính, nhưng nên chạy ở LR thắng để biết khác biệt đến từ
paper accumulation hay chỉ do baseline delta-approx gần EMA.

## 5. Nguyên tắc so sánh công bằng

Mọi run so trực tiếp phải giữ giống nhau:

- Dataset và split protocol.
- Thứ tự class/task theo từng seed.
- Model initialization.
- Backbone và pretrained weights.
- Batch size, augmentation và số epoch/task.
- HOPE/Titans flags.
- CMS tiers, period, eta, order và `grad_agg`.
- `optimizer_per_task=false`.
- Cách đánh giá và accuracy matrix.
- Cùng seed theo cặp: seed 0 so với seed 0, seed 1 so với seed 1.

Hai câu hỏi phải được báo cáo riêng:

1. **Hiệu quả trong project:** so P0 với P1 vì các cơ chế ổn định bên ngoài được
   giữ giống nhau.
2. **Độ trung thành paper:** đánh giá P2 riêng và so P1 với P2 để đo tác động của
   clipping/weight decay.

Không chọn LR bằng test accuracy cuối cùng. Runner cần có đường chọn
hyperparameter bằng validation stream hoặc một validation score riêng. Test chỉ
được dùng sau khi đã khóa cấu hình thắng.

## 6. Công việc code trước khi chạy

### 6.1 Đóng băng baseline

- Ghi commit nền, branch, config và package versions vào metadata mỗi run.
- Không đổi hành vi của `beta_style=delta`.
- Thêm regression test để P0 cho cùng quỹ đạo tham số với commit nền trên toy
  problem cố định.
- Dùng log root mới, ví dụ:

```text
artifacts/m3_paper_study/
```

### 6.2 Rà soát mode `paper` hiện có

Mode hiện tại đã cộng dồn M1/V/M2, nhưng cần kiểm tra:

- Slow memory có dùng đúng chunk trước khi bước vào chunk tiếp theo không.
- Có off-by-one ở mốc `step % frequency == 0` không.
- `O2` nào được dùng cho từng chunk.
- `beta3` nhân với tổng chunk, không phải trung bình chunk.
- Step clock của từng CMS tier có đúng với quy ước `m3_frequency_unit` không.
- Tensor không có gradient ở một CMS step không được làm timeline sai âm thầm.

Nếu cần sửa timing, phải tạo implementation mới hoặc version/tag mới; không
được làm thay đổi khả năng tái lập artifacts cũ mà không ghi nhận.

### 6.3 Chốt chính sách tensor 1D

Paper không nói rõ Newton-Schulz cho bias/LayerNorm vector. Plan dùng:

- Weight matrix/Conv tensor: M3 + Newton-Schulz.
- Bias, LayerNorm và tensor 1D: đường update không Newton-Schulz như code hiện
  tại để tránh phép toán không xác định.

Đây phải được ghi rõ là một quyết định triển khai, không phải chi tiết được paper
quy định. Có thể làm thêm ablation hybrid AdamW cho tensor 1D sau khi P0-P2 xong,
không trộn vào so sánh chính.

### 6.4 Tách tên run để không ghi đè

Tên thư mục hiện chưa chứa đủ:

- `weight_decay`.
- `grad_clip_norm`.
- Bộ `betas`.
- Nhãn `paper-strict` hay `paper-stabilized`.

Runner mới cần `experiment.tag` hoặc đưa các trường trên vào run ID. Hai run có
cùng style/LR nhưng khác clipping tuyệt đối không được dùng chung thư mục.

### 6.5 Thêm optimizer diagnostics

Mỗi task và một số step đầu cần log:

- Norm gradient trước và sau global clipping.
- Norm `M1`, `M2`, `V`.
- Norm `O1`, `O2`.
- Norm `sqrt(V)`.
- Norm `step_dir` trước update clipping.
- Norm `step_dir` sau update clipping.
- Tỷ lệ tensor bị clip.
- `||delta_w|| / (||w|| + eps)` theo tensor và CMS tier.
- Min/max của adaptive denominator.
- Số lần gặp NaN/Inf.
- Runtime và peak memory nếu thiết bị hỗ trợ.

Không lưu toàn bộ tensor mỗi step vì artifacts sẽ quá lớn. Chỉ lưu thống kê
min/median/p95/max theo task và log chi tiết khoảng 20-50 step đầu.

## 7. Unit test bắt buộc

### T1 - Công thức một bước

Với gradient cố định nhỏ, kiểm tra bằng tay:

```text
M1_1 = beta1*g1
V_1  = beta2*g1^2
M1_2 = beta1*g1 + beta1*g2
V_2  = beta2*g1^2 + beta2*g2^2
```

### T2 - Slow chunk timing

Với `frequency=2`, gradient `g1, g2, g3, g4`, xác nhận:

```text
chunk 1 chứa đúng g1+g2
chunk 2 chứa đúng g3+g4
M2 không ăn lặp hoặc bỏ sót gradient
```

### T3 - Paper không bias-correct V

Xác nhận P1/P2 dùng `sqrt(V)+eps`, không dùng `V/(1-beta2^t)`.

### T4 - Strict không update-clip

Xác nhận `update_norm=none` không âm thầm chuẩn hóa hoặc clip `step_dir`.

### T5 - CMS clock

Với tiers có period `[1, 8, 64]`, kiểm tra M3 slow-memory update đúng lịch đã
quy đổi `[16, 2, 1]` khi dùng `m3_frequency_unit=global_step`.

### T6 - Non-finite fail-fast

Gradient, momentum, denominator hoặc update có NaN/Inf phải dừng run và ghi
`failure.json`, không được skip rồi tạo accuracy giả.

### T7 - Baseline regression

P0 phải giữ nguyên quỹ đạo toy problem trước và sau khi thêm P1/P2.

### T8 - Run identity

P1 và P2 cùng LR phải tạo hai output directory khác nhau.

Sau unit test, chạy toàn bộ test suite hiện có. Không bắt đầu EuroSAT nếu test
M3, CMS, HOPE hoặc logging thất bại.

## 8. Các phase thực nghiệm

## Phase A - Static audit và toy problems

Chạy trên:

1. Quadratic vector.
2. Linear regression với weight matrix.
3. Tiny Conv/ViT-like network có matrix, bias và LayerNorm.
4. Chuỗi gradient cố định, đổi hướng và trực giao.

Mục tiêu:

- Xác nhận công thức.
- Xem tốc độ tăng norm M1/M2/V của paper accumulation.
- Phát hiện sớm limit-cycle và denominator quá nhỏ.
- So raw update của P0, P1, P2 trước khi tốn thời gian với ảnh thật.

Điều kiện qua:

- Không NaN/Inf.
- Loss giảm trên matrix problem.
- Slow memory được kích hoạt.
- Không có lỗi chunk/off-by-one.

## Phase B - Smoke HOPE + CMS

Chạy synthetic và một mini-stream EuroSAT:

```text
seed = 0
1 task trước, sau đó 2 task
1 epoch/task
ít batch
num_workers = 0
```

Chạy P0, P1, P2. Kiểm tra:

- Loss từng batch.
- Titans state hữu hạn.
- M1/M2/V hữu hạn.
- Raw step norm.
- Parameter drift theo tier.
- Output directory và config lưu đúng.

P2 phải dừng ngay nếu:

- Có NaN/Inf.
- Loss tăng hơn 10 lần và không hồi phục trong nhiều batch liên tiếp.
- Relative parameter update tăng đột biến hơn 100 lần P0.
- State norm hoặc optimizer norm tăng mất kiểm soát qua từng step.

Không tự thêm clip vào P2 để “cứu run”. Nếu cần clip, đó là P1.

## Phase C - Đo scale để chọn LR cho paper-strict

Không dùng thẳng LR `5e-3` cho P2. Trước tiên đo:

```text
raw_step_norm
relative_update = lr * raw_step_norm / (weight_norm + eps)
```

LR khởi đầu cho P2:

```text
1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4
```

Chỉ mở rộng lên `3e-4`, `1e-3`, `3e-3` nếu smoke cho thấy update vẫn hữu hạn và
không quá lớn. Khoảng này là điểm bắt đầu an toàn, không phải kết luận trước.

P1 có clip nên sweep ban đầu:

```text
1e-4, 3e-4, 1e-3, 3e-3, 5e-3
```

## Phase D - EuroSAT seed 0 sweep

Chạy:

- P0 LR `3e-3` và `5e-3` làm mốc.
- P1 trên LR grid đã nêu.
- P2 trên LR grid vượt qua Phase C.
- P3 tại LR tốt nhất của nhóm clip để chẩn đoán.

Mỗi run phải kiểm tra:

- Config đầy đủ.
- `train_log.json`.
- `metrics.json`.
- `acc_matrix.csv`.
- `failure.json` nếu thất bại.
- Finite state ở mọi task.

Chọn tối đa hai cấu hình paper:

1. Cấu hình có validation average accuracy cao nhất.
2. Cấu hình có forgetting thấp nhất nhưng accuracy không thấp hơn cấu hình tốt
   nhất quá 2 điểm phần trăm.

Không dùng một score tự chế để che trade-off accuracy/forgetting. Hai metric
phải được trình bày riêng.

## Phase E - EuroSAT đủ seed

Chạy seed `0, 1, 2` cho:

- P0 accuracy preset.
- P0 retention preset.
- P1 thắng seed 0.
- P2 thắng seed 0 nếu P2 qua stability gate.
- P3 nếu cần xác định đóng góp của delta-approx.

Báo cáo:

- Mean và standard deviation.
- Chênh lệch paired-seed giữa P0 và từng paper variant.
- Accuracy, forgetting, BWT, FWT.
- Runtime, peak memory, clipping rate.
- Stability diagnostics.

Nếu chênh lệch accuracy giữa P0 và P1 nhỏ hơn 1 điểm phần trăm, chạy thêm seed
`3, 4` trước khi tuyên bố một bản thắng.

## Phase F - RESISC45

Chỉ chạy khi một paper variant:

- Hữu hạn trên mọi EuroSAT seed.
- Không có optimizer/state norm tăng mất kiểm soát.
- Accuracy hợp lý và không thất bại ở task giữa stream.

Chạy RESISC45 3 seed cho:

- P0 LR `5e-3`.
- Paper variant tốt nhất.
- P0 LR `3e-3` chỉ cần dùng lại kết quả retention đã có.
- P2 chỉ được chạy nếu đã ổn định đủ 3 seed EuroSAT.

Protocol bắt buộc:

```text
31.500 ảnh
45 class
700 ảnh/class
split_protocol = combined31500_v2
9 task
```

Không sử dụng lại bất kỳ kết quả RESISC45 cũ nào dùng 18.900 ảnh.

## Phase G - Phân tích và kết luận

Tạo bảng:

| Variant | LR | Acc | Forgetting | BWT | FWT | NaN | Runtime | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

Tạo thêm:

- Accuracy matrix trung bình.
- Accuracy theo task.
- Train loss theo task.
- Norm M1/M2/V theo task.
- Raw/post-clip update norm.
- Relative parameter drift theo CMS tier.
- Tỷ lệ update bị clip của P0/P1.

Kết luận phải tách:

1. Bản có accuracy tốt nhất.
2. Bản có retention tốt nhất.
3. Bản ổn định nhất.
4. Bản gần paper nhất.
5. Chi phí để đổi lấy accuracy/retention.

## 9. Quy tắc ra quyết định

### Giữ P0 làm mặc định nếu

- P1/P2 không ổn định; hoặc
- Paper variant có accuracy thấp hơn P0 rõ ràng; hoặc
- Chênh lệch dưới 1 điểm phần trăm nhưng paper tốn thời gian/bộ nhớ hơn; hoặc
- Paper chỉ thắng một seed và không lặp lại ở các seed khác.

### Chọn P1 làm bản thực dụng mới nếu

- Mean accuracy tăng ít nhất khoảng 1 điểm phần trăm qua paired seeds;
- Không làm forgetting xấu hơn quá 2 điểm phần trăm;
- Không tăng NaN/Inf hoặc parameter drift;
- Chi phí runtime/bộ nhớ chấp nhận được;
- Kết quả lặp lại trên cả EuroSAT và RESISC45.

Nếu P1 thắng, vẫn phải gọi là `M3-paper-stabilized`, không gọi là
`M3-paper-strict`.

### Công nhận P2 hiệu quả nếu

- Không cần update clipping.
- Không cần global grad clipping.
- Hữu hạn trên mọi seed.
- Kết quả cạnh tranh với P0/P1 sau khi tune LR riêng.

Nếu P2 thất bại nhưng P1 tốt, kết luận hợp lệ là:

> Quy tắc momentum tích lũy của paper có ích trong pipeline, nhưng cần cơ chế ổn
> định của project để dùng trên continual ViT.

Nếu cả P1 và P2 thua P0:

> Bản M3-delta-approx-clip thực dụng phù hợp hơn với HOPE+CMS trên remote sensing;
> kết quả không phủ nhận paper vì dataset, model scale và training regime khác.

## 10. Runner đề xuất

Tạo runner riêng, không sửa ý nghĩa các phase cũ:

```text
scripts/run_m3_paper_study.py
```

Interface dự kiến:

```bash
cd "/Users/an/Documents/Do An/RaybanMeta/uav-continual-learning"

python3 scripts/run_m3_paper_study.py audit
python3 scripts/run_m3_paper_study.py smoke
python3 scripts/run_m3_paper_study.py eurosat-sweep
python3 scripts/run_m3_paper_study.py eurosat-replicate
python3 scripts/run_m3_paper_study.py resisc45
python3 scripts/run_m3_paper_study.py summarize
```

Runner phải có:

- `--dry-run`.
- `--skip-existing`.
- Resume theo từng cấu hình/seed.
- Stability gate trước khi sang phase tốn thời gian.
- Không tự chọn run failed hoặc thiếu `train_log.json`.
- Kiểm tra split protocol trước RESISC45.
- Tạo summary JSON/Markdown tự động.

## 11. Deliverables

Code:

- Mode paper đã audit và có test công thức/timing.
- Runner riêng cho study.
- Logging optimizer diagnostics.
- Run naming không collision.
- Validation-based model selection.

Artifacts:

```text
artifacts/m3_paper_study/
  results/
  summaries/
  failures/
```

Báo cáo cuối:

```text
/Users/an/Documents/Do An/RaybanMeta/An/
  KET_QUA_SO_SANH_M3_PRACTICAL_VS_PAPER.md
```

Báo cáo cuối phải ghi:

- Commit và environment.
- Công thức từng variant.
- Mọi LR/seed đã chạy, kể cả run thất bại.
- Kết quả EuroSAT và RESISC45.
- Stability và runtime.
- Giới hạn của thí nghiệm.
- Kết luận bản nào nên dùng làm kết quả chính.

## 12. Checklist thực hiện

- [x] Ghi commit baseline `4b9a736` vào metadata/plan (chưa tạo Git tag).
- [x] Thêm regression test cho P0.
- [x] Audit chunk timing của paper mode.
- [x] Chốt policy tensor 1D.
- [x] Tách run ID cho strict/stabilized.
- [x] Thêm optimizer diagnostics.
- [x] Thêm validation-based selection.
- [x] Chạy toàn bộ unit tests (`81 passed`, 2026-07-25).
- [x] Chạy Phase A toy audit.
- [x] Chạy Phase B smoke.
- [x] Chọn LR range P2 bằng raw update scale và stability gate.
- [x] Chạy EuroSAT seed 0 sweep.
- [x] Chạy EuroSAT 3 seed.
- [x] Không chạy thêm seed 3-4: chênh lệch practical-paper lớn hơn 1 điểm phần trăm trên cả hai dataset.
- [x] Chỉ chạy RESISC45 cho paper variant vượt stability gate (P1; P2 bị loại khỏi cross-dataset vì accuracy/forgetting kém).
- [x] Tổng hợp accuracy, forgetting, BWT, FWT, norm và runtime.
- [x] Tạo báo cáo tự động tại `RaybanMeta/An` (hiện là báo cáo trạng thái smoke;
      kết luận cuối chỉ xuất hiện sau khi đủ EuroSAT/RESISC45).

Ghi chú implementation 2026-07-25: P1/P2 dùng `paper_timing=next_chunk`, đúng
Algorithm 1 (O2 từ chunk vừa hoàn tất được dùng cho chunk kế tiếp). Mode
`legacy_boundary` giữ mặc định trong API cũ chỉ để tái lập artifact trước đây;
runner study luôn override rõ `next_chunk`.

## 13. Phạm vi không làm trong study này

- Không triển khai Delta Momentum Eq. 48-49 đầy đủ.
- Không đồng thời thay kiến trúc HOPE/Titans/CMS.
- Không tune lại CMS tiers, eta hoặc order.
- Không thay dataset protocol.
- Không dùng kết quả test để chọn LR.
- Không xóa hay ghi đè artifacts của study M3 trước.
- Không kết luận paper sai nếu P2 không phù hợp với pipeline hiện tại.

Thứ tự trên giúp mỗi khác biệt đều có nguyên nhân rõ ràng: trước tiên xác nhận
công thức, sau đó đo ổn định, tiếp theo tune LR công bằng, cuối cùng mới tiêu tốn
thời gian cho nhiều seed và RESISC45.
