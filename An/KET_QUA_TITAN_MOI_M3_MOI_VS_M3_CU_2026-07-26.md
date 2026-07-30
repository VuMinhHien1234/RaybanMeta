# Kết quả Titan mới + M3 mới so với Titan mới + M3 cũ

Ngày tổng hợp: 2026-07-26

## 1. Kết luận ngắn

Campaign đã chạy hết toàn bộ plan, sinh summary và tự tắt VM thành công.

Kết quả chính:

- **M3 cũ không ổn định:** cả 3 seed tại LR `1e-3` đều dừng vì gradient
  NaN/Inf. Tỷ lệ hợp lệ là `0/3`.
- **M3 mới ổn định:** cả 3 seed tại LR `1e-3` và `5e-3` đều hoàn thành, state
  hữu hạn và cách rất xa ngưỡng nổ `10.000`. Tỷ lệ hợp lệ là `3/3`.
- LR tốt nhất trong grid đã thử là **`5e-3`**.
- Với LR `5e-3`, M3 mới đạt:
  - Linear Accuracy: **`0.6152 +/- 0.0094`**.
  - Linear Forgetting: **`-0.0300 +/- 0.0147`**.
  - NCM Accuracy: **`0.7403 +/- 0.0043`**.
  - NCM Forgetting: **`0.0635 +/- 0.0054`**.
- Cả 3 seed LR `5e-3` đều vượt mốc NCM của plan:
  `Accuracy >= 0.72` và `Forgetting <= 0.10`.

Kết luận khoa học phù hợp nhất là:

> Trên Titan Task 4 v2, recipe M3 mới đã chứng minh được độ ổn định số vượt
> trội và là recipe duy nhất chạy hết cả 3 seed. Chưa thể tính chênh lệch mean
> accuracy công bằng với M3 cũ vì không seed M3 cũ nào tạo được kết quả cuối
> hợp lệ.

Do đó không nên viết rằng “M3 mới tăng accuracy X điểm so với M3 cũ”. Có thể
viết rằng “M3 mới loại bỏ failure NaN trên 3/3 seed và đạt accuracy ổn định
61,52% Linear, 74,03% NCM”.

## 2. Thiết lập thí nghiệm

Hai nhóm dùng chung:

- Nhánh Git: `Titan_M3`.
- Commit kết quả chính: `9b7e496a5630901483c933d0f3023efda452b324`.
- Dataset: RESISC45 đủ 31.500 ảnh, 700 ảnh/class.
- Split: `combined31500_v2`, 80/10/10.
- Backbone: ViT-S pretrained và frozen.
- Titan: Task 4 v2, self-modifying value-only, `reset=never`.
- 9 task, 3 epoch/task, batch size 32.
- M3 delta-approx, frequency 16, cùng alpha/eta và weight decay.
- GPU: NVIDIA Tesla T4, PyTorch `2.9.1+cu129`.

Khác biệt recipe:

| Thành phần | M3 cũ | M3 mới |
|---|---|---|
| `m3.update_norm` | `rms` | `clip` |
| Optimizer qua task | reset | giữ lại |
| Outer gradient clipping | tắt | `grad_clip_norm=1.0` |
| Kiểm tra numerical health | có | có |

Đây là so sánh hai **recipe**. Không được quy toàn bộ khác biệt cho riêng một
thành phần nếu không dùng kết quả ablation.

## 3. Kiểm tra độ đầy đủ

Campaign hoàn thành các phase:

1. Environment và toàn bộ test.
2. Synthetic smoke cho legacy và improved.
3. Stress seed 1.
4. Sweep LR `1e-4, 3e-4, 1e-3, 3e-3, 5e-3`.
5. Replicate seed `0, 1, 2`.
6. Ablation seed 1 ở LR tốt nhất.
7. Tổng hợp `runs.csv` và `summary.md`.
8. Tự shutdown VM.

Kiểm kê artifact:

| Loại | Số lượng |
|---|---:|
| `metrics.json` | 13 |
| `metrics_ncm.json` | 11 |
| `train_log.json` | 13 |
| `failure.json` | 4 |
| Tổng file đã tải | 121 |

Hai smoke run cố ý tắt NCM, nên 13 metrics nhưng 11 NCM metrics là đúng. Bốn
failure gồm 3 seed legacy LR `1e-3` và một ablation legacy LR `5e-3`.

## 4. Kết quả chính qua 3 seed

| Recipe | LR | Valid | Exploded | Linear Acc | Linear Fgt | NCM Acc | NCM Fgt | Max state norm |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| M3 cũ | `1e-3` | 0/3 | 0, 1, 2 | - | - | - | - | FAIL |
| M3 mới matched | `1e-3` | 3/3 | không | 0.3931 +/- 0.0282 | -0.0469 +/- 0.0174 | 0.7038 +/- 0.0015 | 0.0833 +/- 0.0032 | 126.18 |
| **M3 mới tuned** | **`5e-3`** | **3/3** | **không** | **0.6152 +/- 0.0094** | **-0.0300 +/- 0.0147** | **0.7403 +/- 0.0043** | **0.0635 +/- 0.0054** | **120.45** |

Chi tiết M3 mới LR `5e-3`:

| Seed | Linear Acc | Linear Fgt | BWT | NCM Acc | NCM Fgt | Max norm |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6089 | -0.0293 | 0.2243 | 0.7362 | 0.0696 | 96.22 |
| 1 | 0.6108 | -0.0157 | 0.1886 | 0.7400 | 0.0611 | 120.45 |
| 2 | 0.6260 | -0.0450 | 0.2375 | 0.7448 | 0.0596 | 92.34 |

Khoảng accuracy rất hẹp, từ `0.6089` đến `0.6260`. Điều này tốt hơn kiểu hành
vi “seed ổn, seed nổ” đã quan sát ở M3 cũ.

## 5. LR sweep

Sweep dùng stress seed 1:

| LR | Linear Acc | Linear Fgt | NCM Acc | NCM Fgt | Max norm | Hợp lệ |
|---:|---:|---:|---:|---:|---:|---|
| `1e-4` | 0.0210 | 0.0786 | 0.6902 | 0.0914 | 186.59 | có |
| `3e-4` | 0.0959 | 0.0354 | 0.6987 | 0.0868 | 197.16 | có |
| `1e-3` | 0.3971 | -0.0579 | 0.7022 | 0.0854 | 126.18 | có |
| `3e-3` | 0.5413 | -0.0371 | 0.7178 | 0.0689 | 151.92 | có |
| **`5e-3`** | **0.6108** | **-0.0157** | **0.7400** | **0.0611** | **120.45** | **có** |

Rule đã khóa trước chọn đúng LR `5e-3`. So với matched LR `1e-3`, kết quả 3
seed ở `5e-3` thay đổi:

- Linear Accuracy: `+0.2221` tuyệt đối.
- NCM Accuracy: `+0.0365`.
- NCM Forgetting: `-0.0199`, tốt hơn.
- Linear Forgetting: `+0.0169`, kém hơn nhẹ nhưng vẫn nằm trong tolerance
  `0.02` của plan và vẫn là giá trị âm.
- Mean max state norm: giảm từ `108.95` xuống `103.00`.

M3 mới rất nhạy với LR. LR thấp không gây nổ nhưng gần như không học được
linear head. Không nên dùng lại mặc định `1e-3` cho cấu hình này.

## 6. M3 cũ thất bại như thế nào

| Seed/LR | Điểm dừng | Lỗi |
|---|---|---|
| seed 0, `1e-3` | sau task 4, khi train task 5 | gradient NaN/Inf |
| seed 1, `1e-3` | sau task 3, khi train task 4 | gradient NaN/Inf |
| seed 2, `1e-3` | sau task 1, khi train task 2 | gradient NaN/Inf |
| seed 1, `5e-3` | ngay sau task 0 | `norm(state)=10039.94` |

Ở LR `1e-3`, norm trước khi lỗi của legacy còn khoảng `64-85`, nhưng gradient
đột ngột thành NaN/Inf. Vì vậy chỉ nhìn `norm(state)` chưa đủ để dự báo lỗi;
fail-fast trên gradient/loss/logits vẫn cần được giữ.

Các failure được xem là kết quả thí nghiệm, không bị runner âm thầm bỏ qua hoặc
chạy lặp vô hạn.

## 7. Ablation

Ablation chạy seed 1 tại LR `5e-3`:

| Recipe | Linear Acc | Linear Fgt | BWT | NCM Acc | NCM Fgt | Max norm |
|---|---:|---:|---:|---:|---:|---:|
| Legacy: RMS + reset + no grad clip | FAIL | - | - | - | - | 10039.94 |
| Clip only: clip + reset + no grad clip | 0.5568 | 0.1961 | -0.1961 | 0.7327 | 0.0761 | 87.42 |
| Clip + carry + no grad clip | 0.5571 | -0.0329 | 0.2193 | 0.7330 | 0.0664 | 87.42 |
| **Full improved: clip + carry + grad clip** | **0.6108** | **-0.0157** | **0.1886** | **0.7400** | **0.0611** | 120.45 |

Diễn giải:

1. Đổi `rms` sang `clip` là thành phần chính giúp run không nổ ở LR `5e-3`.
2. Giữ optimizer qua task gần như không đổi final Linear Accuracy trên seed 1
   (`+0.0003`), nhưng cải thiện forgetting `0.2289` và đổi BWT từ âm sang dương.
3. Outer gradient clipping tăng Linear Accuracy thêm `0.0537` và NCM Accuracy
   thêm `0.0070` so với clip + carry.
4. Outer clipping không làm max memory-state norm nhỏ hơn trong run này. Vai trò
   quan sát được của nó là cải thiện quá trình tối ưu, không chỉ chặn state norm.

Ablation mới có một seed, đủ để chẩn đoán nhưng chưa đủ để tuyên bố quan hệ nhân
quả mạnh cho từng thành phần.

## 8. Vấn đề linear head

Accuracy cuối trung bình theo task của M3 mới LR `5e-3`:

| Task | Linear | NCM |
|---:|---:|---:|
| 0 | 0.8524 | 0.7590 |
| 1 | 0.7695 | 0.7514 |
| 2 | 0.7886 | 0.7448 |
| 3 | 0.7762 | 0.7400 |
| 4 | 0.7495 | 0.7295 |
| 5 | 0.5857 | 0.7305 |
| 6 | 0.6257 | 0.7972 |
| 7 | 0.3381 | 0.7305 |
| 8 | **0.0515** | **0.6800** |

Feature sau Titan vẫn tốt vì NCM đạt `0.68-0.80` trên các task cuối, nhưng
linear head giảm mạnh, đặc biệt task 8 chỉ đạt `5.15%`.

Đây là nút thắt lớn còn lại. Average Forgetting âm không phản ánh đầy đủ vấn đề
này vì metric forgetting loại task cuối và chỉ đo task cũ tụt từ đỉnh. Một model
có forgetting đẹp vẫn có thể học task cuối bằng linear head rất kém.

## 9. Trả lời câu hỏi nghiên cứu

### M3 mới có ổn định hơn M3 cũ không?

**Có, bằng chứng rõ.**

- M3 mới: `3/3` seed hợp lệ ở cả LR matched và LR tuned.
- M3 cũ: `0/3` seed hợp lệ tại LR matched.
- Norm M3 mới tối đa `126.18`, thấp hơn rất xa ngưỡng `10.000`.

### M3 mới có accuracy cao hơn M3 cũ không?

**Chưa thể định lượng trực tiếp trong benchmark hiện tại.**

M3 cũ không tạo được metrics cuối ở seed 0, 1 hoặc 2, nên không tồn tại mean
accuracy hợp lệ để trừ. Việc dùng accuracy ngay trước lúc nổ cũng không công
bằng vì các run dừng ở task khác nhau.

Mốc lịch sử seed 0 của M3 cũ từng đạt Linear Accuracy `0.6214`, gần với M3 mới
seed 0 LR `5e-3` là `0.6089`, nhưng mốc lịch sử không cùng run/guard hiện tại và
không đại diện cho độ ổn định qua seed. Chỉ nên dùng nó làm bối cảnh.

### Có nên dùng M3 mới cho kết quả chính không?

**Có.**

M3 cũ không phải baseline vận hành được trên giao thức hiện tại. M3 mới LR
`5e-3` là cấu hình hợp lệ, tái lập tốt qua 3 seed và vượt mốc NCM của project.

## 10. Cấu hình đề xuất

```yaml
train:
  optimizer: m3
  lr: 0.005
  weight_decay: 0.01
  grad_clip_norm: 1.0
  optimizer_per_task: false
  max_state_norm: 10000
  max_state_norm_growth: 10
  m3:
    alpha: 0.5
    frequency: 16
    beta_style: delta
    update_norm: clip
```

Tên khoa học nên dùng:

**M3-delta-approx-clip, practical recipe của project.**

Không gọi đây là Delta Momentum đầy đủ 100% theo paper.

## 11. Hạn chế và việc nên làm tiếp

1. Best LR `5e-3` nằm ở biên trên của grid. Điều này cho thấy sweep chưa chắc đã
   bao được đỉnh; có thể smoke/stress thêm `7e-3` và `1e-2`, rồi chỉ replicate
   nếu vượt `5e-3` mà vẫn ổn định.
2. Ưu tiên xử lý linear-head bias ở task 7-8. Nên thử classifier calibration,
   replay/cân bằng logits, AdamW riêng cho bias/norm/head, hoặc dùng NCM/hybrid
   head khi inference.
3. Nếu cần tuyên bố đóng góp riêng của từng thành phần M3, chạy ablation full và
   hai cấu hình gần nhất thêm seed 0, 2. Kết quả một seed hiện chỉ là chẩn đoán.
4. Thêm Titan mới + AdamW 3 seed làm baseline hữu hạn. Vì M3 cũ nổ hoàn toàn,
   AdamW sẽ giúp trả lời accuracy gain so với một optimizer thực sự chạy được.
5. Chỉ có 3 seed; không tuyên bố ý nghĩa thống kê cho chênh lệch nhỏ.
6. Tiếp tục giữ fail-fast gradient/loss/logits/state. Norm state một mình không
   phát hiện sớm các failure legacy.

## 12. Vị trí kết quả

Archive nguyên gốc tải từ VM:

`RaybanMeta/An/titan_m3_results_2026-07-26.tar.gz`

- Kích thước: `727,307,022` byte, khoảng 694 MB.
- SHA-256:
  `c9540db4c426925d19f780ef4cb9a9b33c2f686cd8c2a42bb51a976dd2ac729f`.

Thư mục đã giải nén:

`RaybanMeta/An/Titan_M3_results_2026-07-26`

Các file chính:

- `artifacts/titan_m3_study/summary.md`
- `artifacts/titan_m3_study/runs.csv`
- `artifacts/titan_m3_study/results/<run>/metrics.json`
- `artifacts/titan_m3_study/results/<run>/metrics_ncm.json`
- `artifacts/titan_m3_study/results/<run>/acc_matrix*.csv`
- `artifacts/titan_m3_study/results/<run>/train_log.json`
- `artifacts/titan_m3_study/results/<run>/memory_state.pt`
- `titan_m3_campaign.log`

VM đã được tắt lại sau khi tải kết quả để tránh tiếp tục tính phí. Persistent
disk vẫn có `autoDelete=false`.
