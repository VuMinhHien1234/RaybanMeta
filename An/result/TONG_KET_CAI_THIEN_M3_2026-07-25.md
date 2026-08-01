# Tổng kết cải thiện M3 - 2026-07-25

## 1. Mục tiêu

Kiểm tra M3 hiện tại có đúng và ổn định không, tìm nguyên nhân accuracy thấp,
sửa các lỗi làm kết quả không đáng tin, rồi so công bằng với AdamW trên EuroSAT
và RESISC45 bằng ít nhất 3 seed.

## 2. Đã làm được gì

### Sửa lỗi độ ổn định

- Sửa NaN trong Titans do phép chia cho norm bằng 0.
- Thêm kiểm tra fail-fast cho gradient, loss, logits, M3 update và Titans state.
- Giữ optimizer xuyên task để M1/M2/V không bị xóa sau mỗi task.
- Quy đổi M3 frequency theo CMS period để tier chậm không phải chờ 1.024 batch.
- Thêm global gradient clipping.
- Sửa `update_norm=rms`, nguyên nhân làm tensor lớn bị update quá mạnh và sinh
  NaN. Mode mặc định hiện là `clip`: chỉ co update khi norm lớn hơn 1.
- Thêm test cho parameter ma trận, Newton-Schulz, slow momentum và logging.

### Sửa tính công bằng của dữ liệu

- EuroSAT validation/test dùng transform xác định.
- RESISC45 trước đây cho AdamW và M3 số ảnh khác nhau. Loader mới gộp đủ 31.500
  ảnh, kiểm tra 700 ảnh/class rồi chia lại 80/10/10.
- Kết quả RESISC45 hợp lệ có hậu tố `splitcombined31500_v2`. Run cũ không được
  dùng để kết luận.

### Thí nghiệm đã chạy

- Sweep EMA và Delta-approx với LR `1e-4, 3e-4, 1e-3, 3e-3`.
- Refine LR `2e-3, 3e-3, 5e-3`.
- Ablation alpha `0.05, 0.1, 0.25, 0.5`.
- Ablation frequency `8, 16, 32`.
- So EMA và Delta-approx đủ 3 seed.
- So AdamW, M3 LR `3e-3`, M3 LR `5e-3` đủ 3 seed.
- Xác nhận cấu hình LR `5e-3` trên RESISC45 đủ 3 seed.
- Toàn bộ run được lưu config, metrics, accuracy matrix và train log để tái lập.

## 3. Kết quả chính

### RESISC45

| Cấu hình | Accuracy, mean +/- sd | Forgetting, mean +/- sd | BWT mean |
|---|---:|---:|---:|
| AdamW | 31,17% +/- 4,28% | 76,61% +/- 4,51% | -76,61% |
| M3 clip, LR 3e-3 | 67,17% +/- 1,37% | **-1,90% +/- 1,62%** | **+19,11%** |
| M3 clip, LR 5e-3 | **72,62% +/- 1,89%** | 4,12% +/- 2,54% | +10,05% |

So với AdamW, preset accuracy LR `5e-3` tăng **41,45 điểm phần trăm**. So với
M3 LR `3e-3`, nó tăng 5,45 điểm accuracy nhưng forgetting xấu hơn 6,02 điểm.

### EuroSAT

| Cấu hình | Accuracy, mean +/- sd | Forgetting, mean +/- sd |
|---|---:|---:|
| AdamW | 38,20% +/- 11,12% | 76,58% +/- 13,78% |
| M3 clip, LR 3e-3 | 70,10% +/- 3,13% | 7,95% +/- 5,98% |
| M3 clip, LR 5e-3 | **71,03% +/- 4,65%** | **6,74% +/- 8,20%** |

Mọi seed trong bảng đều hữu hạn, không có NaN/Inf.

## 4. Cấu hình chốt

Preset ưu tiên accuracy, hiện đặt làm mặc định G4:

```yaml
train:
  optimizer: m3
  lr: 0.005
  grad_clip_norm: 1.0
  optimizer_per_task: false
  m3:
    alpha: 0.5
    frequency: 16
    beta_style: delta
    update_norm: clip
```

Preset ưu tiên retention: đổi riêng `train.lr=0.003`.

Không có một LR tối ưu tuyệt đối:

- `5e-3`: accuracy cao nhất, phù hợp vấn đề ban đầu là model học kém.
- `3e-3`: giữ task cũ tốt hơn, BWT cao hơn.

## 5. Kết luận về thuật toán

M3 hiện tại đã là một optimizer thực dụng, ổn định và có cải thiện lớn so với
AdamW trong pipeline HOPE+CMS. Phần cải thiện được chứng minh đến từ:

1. Chuẩn hóa update bằng `clip` thay cho `rms`.
2. Tune LR riêng cho M3.
3. Giữ optimizer state xuyên task.
4. Đồng bộ frequency M3 với CMS.
5. Sửa NaN Titans và protocol dữ liệu.

Nhánh đang tên `delta` chỉ là xấp xỉ:

```text
M <- alpha*M + eta*(g-M)
```

Nó không phải Delta Momentum Eq. 48-49 đầy đủ. So 3 seed ở LR `3e-3`:

| Style | Accuracy | Forgetting |
|---|---:|---:|
| EMA | 70,15% | 7,86% |
| Delta-approx | 70,10% | 7,95% |

Hai style thực tế bằng nhau; Delta-approx không có đóng góp riêng được chứng
minh. Trong báo cáo khoa học phải gọi đúng là `M3-delta-approx`.

## 6. Vấn đề còn tồn tại

1. Delta Momentum Eq. 48-49 đầy đủ chưa được cài trong M3.
2. `beta_style=paper, update_norm=none` gần pseudocode nhất nhưng chưa ổn định
   trên ViT; kết quả tốt hiện tại là biến thể thực dụng.
3. Bias/norm vector chưa bias-correct M1 giống repo tham khảo.
4. Chưa thử hybrid: M3 cho weight matrix, AdamW cho bias/norm/vector.
5. EuroSAT seed 2 có Titans state norm tăng tạm thời tới `474,27`. State vẫn
   hữu hạn và giảm lại, nhưng cần theo dõi nếu tăng LR hoặc đổi kiến trúc.
6. Ba seed đủ cho kết luận thực nghiệm hiện tại, chưa đủ để khẳng định các chênh
   lệch rất nhỏ dưới 1 điểm phần trăm.
7. LR `5e-3` tăng accuracy nhưng giảm khả năng giữ task cũ so với `3e-3`.

## 7. Nên làm gì tiếp

1. Dùng LR `5e-3` cho bảng ưu tiên accuracy và LR `3e-3` cho ablation retention.
2. Không tiếp tục tune alpha/frequency quanh grid vừa chạy; chưa có bằng chứng
   chúng tốt hơn `alpha=0.5`, `frequency=16`.
3. Nếu cần đóng góp Delta Momentum trong bài báo, cài nó thành style mới và so
   riêng với EMA; không sửa nghĩa mode `delta` cũ.
4. Thử hybrid M3/AdamW và bias correction vector như một nghiên cứu tiếp theo,
   không trộn vào baseline đã chốt.
5. Khi viết báo cáo, tách rõ đóng góp của HOPE+CMS, M3-clip và Delta-approx.

## 8. Vị trí kết quả

- Runner: `uav-continual-learning/scripts/run_m3_study.py`
- Báo cáo kỹ thuật: `uav-continual-learning/docs/KIEM_TRA_M3_2026-07-23.md`
- Artifacts: `uav-continual-learning/artifacts/m3_study/results`
- Config EuroSAT: `uav-continual-learning/configs/g4_hope_eurosat.yaml`
- Config RESISC45: `uav-continual-learning/configs/g4_hope_resisc45.yaml`

Lệnh tái lập phase tối ưu:

```bash
cd "/Users/an/Documents/Do An/RaybanMeta/uav-continual-learning"
HF_DATASETS_OFFLINE=1 python3 scripts/run_m3_study.py optimize
```
