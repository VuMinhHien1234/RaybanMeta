# Báo cáo kết quả cải thiện Titans + NCM không replay

Ngày hoàn tất: 2026-08-01  
Nhánh: `NCM_Head`  
Commit tại máy local khi tổng hợp: `49b084e6db3ed3120ae031fb6a8e72cfa3ffbb2e`

## 1. Mục tiêu

Kiểm tra xem có thể cải thiện `ViT + Titans + NCM Online` trong điều kiện online
thực tế, không đọc lại dữ liệu train của task cũ, để vượt `Frozen ViT + NCM`
hay không.

Campaign kiểm tra các tầng sau:

1. Anchored Feature Blend.
2. Blend + Prototype Transport.
3. Blend + Feature Distillation không replay.
4. Blend + Transport + Distillation.

## Giải thích nhanh các phương pháp

**Frozen ViT + NCM:** ViT được giữ nguyên, không học thêm khi task mới đến. NCM
lưu một vector đại diện, gọi là prototype, cho mỗi lớp và dự đoán ảnh mới thuộc
prototype gần nhất. Cách này đơn giản, ổn định và là baseline chính, nhưng không
thích nghi được với thông tin mới trong stream.

**Titans + NCM Online:** Titans tạo thêm memory để feature có thể thay đổi và
thích nghi sau mỗi task. NCM chỉ cập nhật prototype bằng dữ liệu task đang học,
không đọc lại ảnh task cũ. Nhược điểm là feature space thay đổi nhưng prototype
lớp cũ có thể vẫn nằm ở feature space cũ, gây giảm accuracy và tăng forgetting.

**Anchored Feature Blend:** Trộn feature ổn định của Frozen ViT với feature thích
nghi của Titans. `gamma` quyết định tỷ lệ Titans:

- `gamma=0`: 100% Frozen, không dùng tín hiệu Titans.
- `gamma=0.25`: khoảng 75% Frozen + 25% Titans.
- `gamma=1`: 100% Titans.

Mục tiêu là lấy một phần khả năng thích nghi của Titans nhưng vẫn giữ không gian
feature đủ ổn định cho các prototype cũ. Đây là phương pháp cho kết quả tốt nhất
trong campaign khi dùng `gamma=0.25`.

**Prototype Transport:** Khi feature space thay đổi, phương pháp này ước lượng
một phép biến đổi để di chuyển prototype cũ sang feature space mới. Có thể hiểu
như cập nhật lại vị trí các “mốc đại diện lớp” mà không cần đọc lại ảnh cũ. Nếu
ước lượng sai, prototype có thể bị di chuyển lệch; vì vậy code dùng safety gate
để chỉ chấp nhận transport khi phép biến đổi đạt kiểm tra alignment.

**Feature Distillation không replay:** Frozen ViT đóng vai trò teacher. Khi học
ảnh của task hiện tại, Titans được phạt nếu feature lệch quá xa feature của
teacher trên chính các ảnh mới đó. Cách này giảm feature drift mà không lưu hoặc
đọc lại ảnh task cũ, nhưng regularization quá mạnh có thể hạn chế khả năng thích
nghi của Titans.

**Full Combination:** Kết hợp Anchored Blend, Prototype Transport và Feature
Distillation. Về lý thuyết, blend giữ feature ổn định, distillation giảm drift và
transport sửa prototype cũ. Trong lần chạy 1, current-task validation chọn
`gamma=1`, khiến full combination cải thiện nhiều so với Titans Online cũ nhưng
vẫn chưa vượt Frozen NCM.

**Titans + NCM Post-hoc:** Sau khi model học xong, phương pháp này đọc lại toàn bộ
train data cũ để tính lại prototype trong feature space cuối cùng. Nó cho biết
model có tiềm năng tốt đến đâu nếu được phép làm mới toàn bộ prototype, nhưng vi
phạm điều kiện online/no-replay nên chỉ là oracle chẩn đoán, không phải kết quả
triển khai công bằng.

**No replay:** Model không được lưu rồi đọc lại ảnh train của task cũ. Đây là
điều kiện quan trọng để mô phỏng hệ thống online có giới hạn dữ liệu và bộ nhớ.

**Accuracy và Forgetting:** Accuracy càng cao càng tốt. Forgetting đo mức giảm
khả năng nhận diện các task cũ sau khi học task mới; với cách tính trong báo cáo
này, giá trị dương càng nhỏ càng tốt.

Mốc so sánh đã được xác nhận từ campaign NCM Head trước:

| Phương pháp | Accuracy | Forgetting | Vai trò |
|---|---:|---:|---|
| Frozen ViT + NCM | 0.7114 +/- 0.0000 | 0.0874 +/- 0.0041 | Baseline chính |
| Titans + NCM Online cũ | 0.6190 +/- 0.0147 | 0.2671 +/- 0.0186 | Online baseline |
| Titans + NCM Post-hoc | 0.7459 +/- 0.0075 | 0.0656 +/- 0.0033 | Oracle có đọc lại old train, không dùng để tuyên bố thắng |

## 1.1. Bảng 1 - Lần chạy khám phá, seeds 0, 1, 2

Đây là **bảng tổng quan cũ của lần chạy đầu**, được giữ riêng và không trộn số
liệu của seeds mới vào.

| Phương pháp | Seeds | Accuracy mean +/- std | Forgetting mean +/- std | So với Frozen | So với Online cũ | Trạng thái kết luận |
|---|---:|---:|---:|---:|---:|---|
| Frozen ViT + NCM | 3 | 0.7114 +/- 0.0000 | 0.0874 +/- 0.0041 | Mốc | +9.24 điểm | **Baseline production** |
| Titans + NCM Online cũ | 3 | 0.6190 +/- 0.0147 | 0.2671 +/- 0.0186 | -9.24 điểm | Mốc | Baseline online cũ |
| Anchored Blend, gamma 0.25 | 3 | **0.7196 +/- 0.0045** | 0.0913 +/- 0.0007 | **+0.81 điểm** | +10.06 điểm | Hứa hẹn, nhưng mới là exploratory/test-hindsight |
| Anchored Blend, gamma 1.0 | 3 | 0.6166 +/- 0.0185 | 0.2689 +/- 0.0269 | -9.48 điểm | -0.24 điểm | Gamma được current-task validation chọn |
| Full combination được validation chọn | 3 | **0.6898 +/- 0.0100** | **0.1711 +/- 0.0113** | **-2.16 điểm** | **+7.08 điểm** | Kết quả online/no-replay chính thức |
| Titans + NCM Post-hoc | 3 | 0.7459 +/- 0.0075 | 0.0656 +/- 0.0033 | +3.45 điểm | +12.69 điểm | Oracle; đọc lại old train, không deployable theo protocol |

**Kết luận riêng của lần 1:** full combination do current-task validation chọn
vẫn thua Frozen NCM. Gamma `0.25` cho kết quả tốt nhất khi nhìn lại test grid,
nhưng lúc này chỉ tạo ra giả thuyết cần xác nhận vì chưa được khóa từ trước.

## 1.2. Bảng 2 - Lần chạy xác nhận, seeds 3, 4, 5

Gamma `0.25` được khóa trước khi chạy; campaign không sweep lại gamma, không dùng
transport, không distillation và không đọc lại old train.

| Seed | Gamma 0.25 Accuracy | Forgetting | Frozen control | Gain over Frozen (điểm %) |
|---:|---:|---:|---:|---:|
| 3 | 0.7210 | 0.0979 | 0.7114 | +0.95 |
| 4 | 0.7219 | 0.0968 | 0.7114 | +1.05 |
| 5 | 0.7190 | 0.0857 | 0.7114 | +0.76 |
| **Mean +/- std** | **0.7206 +/- 0.0015** | **0.0935 +/- 0.0067** | **0.7114 +/- 0.0000** | **+0.92 +/- 0.15** |

- Cả 3/3 seed xác nhận đều cao hơn Frozen NCM.
- Mean gain là `+0.0092`, tức khoảng **+0.92 điểm phần trăm**.
- 95% t-confidence interval của gain xấp xỉ **+0.56 đến +1.28 điểm phần trăm**;
  khoảng này không chứa 0.
- Forgetting của gamma 0.25 là `0.0935`; Frozen control cùng ba seed là `0.0937`.
  Chênh lệch `-0.0002` là rất nhỏ, tức không thấy đánh đổi forgetting đáng kể.
- Gộp cả discovery và confirmatory thành 6 seed: accuracy `0.7201 +/- 0.0030`,
  cao hơn Frozen `+0.87` điểm; 95% CI của gain khoảng `+0.55` đến `+1.19` điểm.

Kết luận hợp lệ ở đây là gamma 0.25 tạo cải thiện nhỏ nhưng nhất quán trên nhiều
class order của RESISC45. Kết quả chưa chứng minh khả năng khái quát sang dataset
khác và chưa tự động chứng minh lợi ích đủ lớn so với chi phí chạy Titans.

## 1.3. Hai lần chạy khác nhau ở đâu?

**Lần chạy 1 - khám phá:**

- Dùng seeds `0,1,2`.
- Thử gamma grid `[0, 0.1, 0.25, 0.5, 0.75, 1]`, transport, distillation và full
  combination.
- Current-task validation chọn `gamma=1`, nhưng sau khi xem test mới nhận ra
  `gamma=0.25` đạt `0.7196` và tốt nhất trong gamma grid.
- Vì test đã tham gia vào việc nhận ra ứng viên tốt, kết quả gamma `0.25` ở lần
  này chỉ là **exploratory/test-hindsight**.
- Giá trị của lần chạy 1 là phát hiện giả thuyết: nên khóa `gamma=0.25` và chạy
  lại trên seeds chưa dùng.

**Lần chạy 2 - xác nhận:**

- Dùng seeds mới `3,4,5`; không dùng lại seeds discovery để xác nhận.
- Khóa `gamma=0.25` trước khi chạy và chỉ giữ `gamma=0` làm Frozen control.
- Không sweep/chọn lại gamma, không transport, không distillation và không replay
  dữ liệu task cũ.
- Đạt `0.7206 +/- 0.0015`; cả `3/3` seed đều cao hơn Frozen `0.7114`.
- Vì cấu hình đã được khóa trước, đây là **confirmatory evidence hợp lệ trong
  phạm vi RESISC45**.

**Kết luận chung từ hai lần:**

- Lần 1 phát hiện gamma `0.25` là ứng viên tốt; lần 2 xác nhận tín hiệu đó không
  chỉ là may mắn của seeds `0,1,2`.
- Gộp để mô tả độ ổn định trên cả 6 seed, gamma `0.25` đạt
  `0.7201 +/- 0.0030`, cao hơn Frozen khoảng `0.87` điểm phần trăm; 6/6 seed đều
  cao hơn Frozen.
- Có thể kết luận Anchored Blend `75% Frozen + 25% Titans` cải thiện accuracy nhỏ
  nhưng nhất quán trên RESISC45, đồng thời forgetting gần như không đổi.
- Chưa thể kết luận phương pháp sẽ thắng trên dataset khác. Mức tăng dưới 1 điểm
  phần trăm cũng cần được cân nhắc với chi phí tính toán của Titans.

## 2. Cấu hình và protocol

- Dataset: RESISC45, split `combined31500_v2`, 31,500 ảnh, 45 lớp.
- Continual stream: 9 task, 5 lớp mỗi task, class order thay đổi theo seed.
- Lần chạy khám phá dùng seeds `0,1,2`; lần confirmatory dùng seeds `3,4,5`.
- Backbone: pretrained `vit_small_patch16_224`, backbone đóng băng.
- Titans: memory bật, `reset=never`, state tiếp tục qua các task.
- Optimizer: M3 delta practical, LR `0.005`, frequency `16`, update clip.
- Training: 3 epoch/task, CUDA trên GPU T4.
- NCM feature protocol: `independent_image`.
- Prototype chỉ được cập nhật bằng ảnh của task hiện tại.
- Không đọc lại train data của task cũ.
- Ở lần 1, hyperparameter được chọn bằng mean accuracy trên validation của task
  hiện tại; không dùng old validation. Ở lần 2, gamma `0.25` được khóa trước và
  không thực hiện lựa chọn hyperparameter mới.

Cấu hình được protocol chọn cho full combination:

- Blend gamma: `1.0`.
- Transport: diagonal, regularization `10`, beta `0.5`.
- Feature distillation: frozen-backbone teacher, cosine loss, weight `0.1`.

## 3. Hậu kiểm kỹ thuật

Campaign đã hoàn tất và artifact hợp lệ:

- `121 passed`; không có test fail.
- 11/11 result directory có `metrics.json`, `train_log.json` và `checkpoint.pt`.
- Không có `failure.json`.
- Không còn `progress.pt`, tức mọi run đã đóng kết quả hoàn chỉnh.
- Tất cả JSON parse hợp lệ.
- Không tìm thấy NaN/Inf trong JSON hoặc CSV.
- Cả 27 state checkpoint của 3 full seeds đều finite.
- State norm full combination nằm trong khoảng `64.20` đến `109.24`, rất xa
  ngưỡng lỗi `10000`.
- `old_samples_revisited=0` ở cả ba full seeds.
- Feature distillation ghi nhận `old_data_revisited=false` ở mọi task.
- Prototype transport được safety gate chấp nhận ở 8/8 lần có thể transport
  trong mỗi seed; task đầu không có feature space cũ để transport.

VM Spot bị preempt trong campaign nhưng runner checkpoint/resume đã tiếp tục đúng
task. Không có run bị tính trùng trong summary cuối. Sau khi tải artifact, VM đã
được dừng và xác nhận trạng thái `TERMINATED`.

## 4. Kết quả Anchored Blend

Đây là kết quả ba seed của toàn bộ gamma grid đã định trước:

| Gamma | Accuracy mean +/- std | Forgetting mean +/- std |
|---:|---:|---:|
| 0.00 | 0.7114 +/- 0.0000 | 0.0874 +/- 0.0041 |
| 0.10 | 0.7141 +/- 0.0012 | 0.0875 +/- 0.0031 |
| 0.25 | **0.7196 +/- 0.0045** | 0.0913 +/- 0.0007 |
| 0.50 | 0.7171 +/- 0.0091 | 0.1144 +/- 0.0085 |
| 0.75 | 0.6793 +/- 0.0089 | 0.1810 +/- 0.0114 |
| 1.00 | 0.6166 +/- 0.0185 | 0.2689 +/- 0.0269 |

Ý nghĩa gamma:

- `gamma=0`: chỉ dùng feature frozen, tái lập đúng Frozen NCM.
- `gamma=1`: chỉ dùng feature Titans.
- Giá trị ở giữa trộn tín hiệu ổn định của frozen ViT với phần thích nghi của
  Titans.

Gamma `0.25` đạt accuracy cao nhất khi nhìn lại test: cao hơn Frozen NCM khoảng
`0.0081`, tức `0.81` điểm phần trăm. Cả ba seed đều cao hơn baseline lần lượt
`+0.54`, `+0.57`, `+1.33` điểm phần trăm. Tuy nhiên đây là quan sát sau khi đã
nhìn toàn bộ test grid, nên chỉ là **kết quả khám phá**, chưa phải thắng lợi được
xác nhận độc lập. Mức tăng trung bình cũng chưa đạt tiêu chí mạnh `+1` điểm phần
trăm và uncertainty của ba seed còn chồng baseline.

Lượt confirmatory seeds `3,4,5` sau đó đã khóa gamma trước và xác nhận lại tín
hiệu này; xem Mục 1.2. Vì vậy hạn chế test-hindsight chỉ áp dụng cho lượt discovery
ban đầu, không áp dụng cho kết luận confirmatory mới.

## 5. Kết quả lựa chọn hyperparameter

Validation task hiện tại chọn gamma `1.0`:

| Gamma | Mean current-task validation |
|---:|---:|
| 0.00 | 0.7975 |
| 0.10 | 0.8000 |
| 0.25 | 0.8076 |
| 0.50 | 0.8238 |
| 0.75 | 0.8403 |
| 1.00 | **0.8549** |

Đây là phát hiện quan trọng nhất của campaign: validation trên task mới thích
feature Titans mạnh vì nó giúp phân lớp task hiện tại, nhưng không đo được việc
prototype của lớp cũ đang trở nên stale. Vì vậy thứ tự validation gần như ngược
với khả năng giữ lớp cũ trên test continual.

Nói cách khác, implementation chạy đúng protocol, nhưng **objective dùng để chọn
hyperparameter chưa đại diện đúng mục tiêu chống quên**. Đây là lý do cấu hình
chính thức chọn gamma `1.0` dù gamma `0.25` cho kết quả continual tốt hơn.

## 6. Transport và distillation

### Transport screening

Theo current-task validation, transport được chọn là diagonal `r=10, beta=0.5`,
validation `0.8476`. Trên seed 0, readout này đạt test accuracy `0.6597`.

Trong tám ứng viên, identity-ridge `r=10, beta=1.0` đạt test accuracy cao nhất
`0.7349`, nhưng validation chỉ `0.8057` và xếp cuối. Không được đổi sang phương
án này rồi tuyên bố kết quả chính, vì đó sẽ là chọn hyperparameter bằng test.
Nó chỉ tạo ra một giả thuyết tốt cho campaign xác nhận kế tiếp.

### Distillation screening

Trên seed 0 với gamma `1.0`:

| Distillation weight | Accuracy | Forgetting |
|---:|---:|---:|
| 0.01 | 0.6216 | 0.2654 |
| 0.05 | 0.6267 | 0.2550 |
| 0.10 | **0.6387** | **0.2411** |

Distillation weight `0.1` cải thiện rõ so với blend gamma `1.0` seed 0
(`0.6133`), cho thấy neo feature vào frozen backbone có tác dụng. Tuy vậy mức
regularization này chưa đủ bù cho drift khi dùng 100% feature Titans.

## 7. Full combination ba seed

Full combination dùng đúng hyperparameter được current-task validation chọn:
gamma `1.0` + diagonal transport + distillation `0.1`.

| Seed | Accuracy | Forgetting | BWT | So với Frozen NCM |
|---:|---:|---:|---:|---:|
| 0 | 0.6879 | 0.1757 | -0.1754 | -0.0235 |
| 1 | 0.6810 | 0.1793 | -0.1761 | -0.0305 |
| 2 | 0.7006 | 0.1582 | -0.1554 | -0.0108 |
| Mean +/- std | **0.6898 +/- 0.0100** | **0.1711 +/- 0.0113** | **-0.1689 +/- 0.0118** | **-0.0216** |

So với Titans + NCM Online cũ:

- Accuracy tăng từ `0.6190` lên `0.6898`: **+7.08 điểm phần trăm**.
- Forgetting giảm từ `0.2671` xuống `0.1711`: cải thiện khoảng **9.60 điểm**.

So với Frozen NCM:

- Accuracy vẫn thấp hơn **2.16 điểm phần trăm**.
- Forgetting vẫn cao hơn khoảng **8.37 điểm**.
- Không seed nào vượt Frozen NCM.

So với Post-hoc oracle:

- Còn cách khoảng **5.60 điểm phần trăm accuracy**.
- Full combination vẫn là online/no-replay; Post-hoc không phải phép so triển
  khai công bằng.

## 8. Kết luận

### Kết luận được xác nhận

1. Code adaptation, checkpoint/resume, finite guard và no-replay audit hoạt động
   đúng trong campaign này.
2. Transport + distillation cải thiện mạnh Titans NCM Online cũ, tăng 7.08 điểm
   accuracy và giảm forgetting đáng kể.
3. Cấu hình full combination được protocol hiện tại chọn vẫn **chưa vượt Frozen
   NCM**; vì vậy chưa nên thay Frozen NCM trong cấu hình triển khai chính.
4. Nút thắt lớn nhất không chỉ là năng lực của transport/distillation, mà là tiêu
   chí current-task validation đang chọn cấu hình tốt cho lớp mới nhưng kém giữ
   prototype lớp cũ.
5. Anchored Blend `gamma=0.25` được khóa trước đã đạt `0.7206 +/- 0.0015` trên
   seeds `3,4,5`; cả ba seed vượt Frozen NCM, mean gain 0.92 điểm phần trăm và
   không làm forgetting xấu đi đáng kể.
6. Do đó có thể kết luận trong phạm vi RESISC45 rằng thêm một lượng nhỏ feature
   Titans vào feature frozen **có hiệu quả**, nhưng lợi ích còn nhỏ và chưa vượt
   tiêu chí practical win mạnh `+1` điểm phần trăm.

### Tín hiệu còn cần xác nhận thêm

1. Identity-ridge `r=10, beta=1.0` đạt `0.7349` trên seed 0 transport screen.
2. Distillation `0.1` cải thiện nhất quán trong ba weight đã thử.
3. Kết hợp transport/distillation với gamma `0.25` chưa được chạy confirmatory.

Không được dùng identity-ridge làm kết luận thắng chính vì nó được nhận ra sau
khi xem test candidates.

## 9. Việc nên làm tiếp theo

Ưu tiên theo thứ tự:

1. **Sửa selection protocol.** Không dùng accuracy của task hiện tại làm
   tiêu chí duy nhất. Thêm stability proxy không cần old data: feature-anchor
   drift trên current inputs, transport held-out alignment, prototype movement,
   condition number và penalty cho gamma lớn.
2. **Xác nhận liên-dataset**, ưu tiên khóa gamma `0.25` và chạy EuroSAT trước khi
   tuyên bố cải thiện khái quát. Không tuning lại gamma trên test EuroSAT.
3. Nếu tiếp tục transport/distillation, khóa gamma `0.25` và dùng một shortlist
   đã đăng ký trước; không chọn identity-ridge bằng test RESISC45 lần nữa.
4. Đo thêm latency, memory và năng lượng. Gain 0.92 điểm có thể không bù được chi
   phí Titans nếu mục tiêu là deployment gọn nhẹ.
5. Frozen NCM vẫn là lựa chọn production bảo thủ. Gamma `0.25` là research
   candidate đã được xác nhận trên RESISC45 và có thể trở thành production nếu
   vượt kiểm tra liên-dataset hoặc chứng minh chi phí triển khai chấp nhận được.

## 10. Artifact

Toàn bộ artifact đã tải về:

`RaybanMeta/An/Titans_NCM_NoReplay_results_2026-08-01/artifacts_ncm_adaptation`

Artifact confirmatory seeds `3,4,5`:

`RaybanMeta/An/Gamma025_confirmatory_results_2026-08-01/artifacts_ncm_gamma025_confirmatory`

Dung lượng local khoảng `2.7 GB`, gồm:

- `selection.json`: hyperparameter được protocol chọn.
- `summary.csv`: kết quả tổng hợp mọi run.
- `campaign_logs/`: environment, pytest và log từng experiment.
- `results/`: config, metrics, accuracy matrix, train log, model checkpoint và
  Titans memory state của 11 run.
- `REPORT_TITANS_NCM_NO_REPLAY.md`: report tự sinh từ runner.
