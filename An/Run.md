Bạn cần chạy theo **2 mức**: trước hết chạy **quick EuroSAT** để kiểm tra toàn bộ G1→G4, sau đó mới chạy **RESISC45 full** nếu máy chịu được.

**Bước 0: kiểm tra môi trường**
```bash
cd "/Users/an/Documents/Do An/RaybanMeta/uav-continual-learning"

python3 scripts/check_env.py
pytest -q
```

**Bước 1: chạy đủ 5 baseline EuroSAT**
```bash
for m in finetune ewc replay lwf ncm; do
  python3 scripts/run_g1.py --config configs/g1_eurosat.yaml --method $m \
    --set device=mps data.num_workers=0
done

python3 scripts/compare_g1.py
```

**Bước 2: chạy G2 Titans A/B/C trên EuroSAT**
```bash
# A: reset mỗi ảnh
python3 scripts/run_g1.py --config configs/g2_titans_eurosat.yaml \
  --set device=mps data.num_workers=0 memory.reset=image

# B: nhớ trong task
python3 scripts/run_g1.py --config configs/g2_titans_eurosat.yaml \
  --set device=mps data.num_workers=0 memory.reset=task

# C: nhớ xuyên task + FWT
python3 scripts/run_g1.py --config configs/g2_titans_eurosat.yaml \
  --set device=mps data.num_workers=0 memory.reset=never train.eval_future=true

python3 scripts/compare_g1.py
```

**Bước 3: chạy G3 CMS ablation EuroSAT**
```bash
python3 scripts/run_g1.py --config configs/g3_cms_eurosat.yaml \
  --set device=mps data.num_workers=0 cms.order=late_slow

python3 scripts/run_g1.py --config configs/g3_cms_eurosat.yaml \
  --set device=mps data.num_workers=0 cms.order=early_slow

python3 scripts/run_g1.py --config configs/g3_cms_eurosat.yaml \
  --set device=mps data.num_workers=0 cms.order=late_slow "cms.tiers=[[4,1],[4,8],[4,64]]"

python3 scripts/run_g1.py --config configs/g3_cms_eurosat.yaml \
  --set device=mps data.num_workers=0 cms.order=early_slow "cms.tiers=[[4,1],[4,8],[4,64]]"

python3 scripts/compare_g1.py
```

**Bước 4: chạy G4 HOPE EuroSAT**
```bash
python3 scripts/run_g1.py --config configs/g4_hope_eurosat.yaml \
  --set device=mps data.num_workers=0

python3 scripts/run_g1.py --config configs/g4_hope_eurosat.yaml \
  --set device=mps data.num_workers=0 train.optimizer=adamw

python3 scripts/compare_g1.py
```

**Bước 5: RESISC45 full**
Cái này nặng. Trên Mac M4 nên chạy từng dòng trước, không bấm full matrix ngay:

```bash
python3 scripts/run_g1.py --config configs/g1_resisc45.yaml --method ncm \
  --set device=mps data.num_workers=0 backbone.freeze=true
```

Nếu ổn rồi mới chạy các method khác:

```bash
for m in finetune ewc replay lwf ncm; do
  python3 scripts/run_g1.py --config configs/g1_resisc45.yaml --method $m \
    --set device=mps data.num_workers=0
done
```

Có script gộp tất cả:
```bash
bash scripts/run_all.sh --quick
```

Nhưng trên MacBook mình khuyên **chạy từng bước như trên** để dễ biết lỗi nằm ở G1/G2/G3/G4. `--quick` tiện nhưng nếu lỗi giữa chừng thì khó đọc hơn.




## ------------------------------------------------------------------------
### CÁCH ĐỌC RESULT
Đây là các **thư mục kết quả của từng lần chạy experiment**. Tên được ghép theo công thức gần như:

```text
<dataset>_<method>_seed<seed>_<optimizer>_<memory/reset>_<cms_order>_<periods>
```

Ví dụ:

```text
eurosat_cms_seed0_m3_early_slow_p1-8-64
```

Đọc là:

- `eurosat`: dataset dùng là EuroSAT.
- `cms`: method/mô hình là CMS.
- `seed0`: seed random = 0.
- `m3`: optimizer là M3.
- `early_slow`: cách chia tầng CMS, các block đầu được update chậm hơn.
- `p1-8-64`: chu kỳ update các tier là 1, 8, 64 bước.

**Ý nghĩa từng nhóm file/thư mục**

```text
eurosat_finetune_seed0
```
Baseline fine-tune thường. Model học tuần tự, dễ quên. Đây là mốc “đáy”.

```text
eurosat_ewc_seed0
```
Baseline EWC. Có regularization để giữ tham số cũ.

```text
eurosat_replay_seed0
```
Baseline Replay. Lưu một ít ảnh cũ để ôn lại.

```text
eurosat_lwf_seed0
```
Baseline Learning without Forgetting. Dùng teacher/distillation.

```text
eurosat_ncm_seed0
```
Baseline NCM. Backbone đóng băng, mỗi class có prototype trung bình. Rẻ, ít quên.

```text
eurosat_titans_seed0_rnever
```
Titans chạy với reset `never`, tức memory sống xuyên task. Có thể là bản dùng AdamW hoặc tên cũ chưa ghi optimizer.

```text
eurosat_titans_seed0_m3_rnever
```
Titans + optimizer M3, memory reset `never`. Đây là bản G2 quan trọng.

```text
eurosat_cms_seed0_m3_late_slow_p1-4-16
```
CMS + M3. `late_slow`: block cuối update chậm. Chu kỳ tier là 1, 4, 16.

```text
eurosat_cms_seed0_m3_early_slow_p1-4-16
```
CMS + M3. `early_slow`: block đầu update chậm. Chu kỳ 1, 4, 16.

```text
eurosat_cms_seed0_m3_late_slow_p1-8-64
```
CMS + M3. Block cuối chậm, chu kỳ chậm hơn: 1, 8, 64.

```text
eurosat_cms_seed0_m3_early_slow_p1-8-64
```
CMS + M3. Block đầu chậm, chu kỳ 1, 8, 64. Đây thường là một ứng viên CMS-best nếu số tốt.

```text
eurosat_hope_seed0_m3_rnever_early_slow_p1-8-64
```
HOPE = Titans + CMS + M3. Memory Titans reset `never`, CMS dùng `early_slow`, periods `1-8-64`.

**Trong mỗi thư mục có gì**
Mở một thư mục bất kỳ, thường có:

```text
metrics.json
acc_matrix.csv
config.yaml
```

Đọc nhanh nhất là mở `metrics.json`.

Ví dụ:

```bash
cat artifacts/results/eurosat_hope_seed0_m3_rnever_early_slow_p1-8-64/metrics.json
```

Các chỉ số cần xem:

- `average_accuracy`: càng cao càng tốt.
- `average_forgetting`: càng thấp càng tốt.
- `backward_transfer`: gần 0 hoặc dương là tốt; âm là quên.
- `forward_transfer`: chỉ có ý nghĩa nếu bật eval future, G2/G4 nên xem.
- `runtime_sec`: thời gian chạy.
- `method_extra_floats`: bộ nhớ thêm.

**Hai file tổng**
```text
baseline_table.csv
baseline_table.md
```

Đây là bảng gộp tất cả thư mục kết quả.

Mở file này để so sánh nhanh:

```bash
cat artifacts/results/baseline_table.md
```

Hoặc mở `baseline_table.csv` bằng Excel/VS Code table viewer.

**Cách đọc theo mục tiêu đồ án**
So theo từng nhóm:

1. **G1 baseline**
So:
```text
finetune / ewc / replay / lwf / ncm
```
Mục tiêu: biết baseline mạnh nhất là ai.

2. **G2 Titans**
So:
```text
eurosat_titans_seed0_m3_rnever
```
với:
```text
eurosat_ncm_seed0
eurosat_replay_seed0
eurosat_finetune_seed0
```

3. **G3 CMS**
So 4 thư mục CMS:
```text
early_slow_p1-4-16
early_slow_p1-8-64
late_slow_p1-4-16
late_slow_p1-8-64
```
Chọn cái có `accuracy` cao và `forgetting` thấp nhất.

4. **G4 HOPE**
So:
```text
hope
```
với:
```text
titans
cms-best
replay
ncm
```

Nếu HOPE tốt hơn Titans-only và CMS-only thì có tín hiệu “ghép 2 thành phần có ích”.