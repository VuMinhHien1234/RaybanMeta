# Kế hoạch cải thiện Titans + NCM không replay

## 1. Thông tin chung

- Nhánh thực hiện: `NCM_Head`
- Project: `RaybanMeta/uav-continual-learning`
- Dataset chính: RESISC45, 9 task, 5 class/task
- Backbone: ViT-S/16 pretrained, đóng băng
- Model học liên tục: Titans mới + M3 improved/delta-approx
- Head đánh giá chính: NCM Online
- Ràng buộc chính: không lưu và không đọc lại ảnh/feature của task cũ
- Ngày lập kế hoạch: 2026-07-31

Mục tiêu khoa học cần trả lời:

> Trong giao thức class-incremental online, không replay dữ liệu cũ, Titans có thể
> bổ sung thông tin hữu ích để NCM đạt accuracy cao hơn Frozen ViT + NCM hay không?

Plan này ưu tiên kết luận đúng và tái lập được. Không xem một run seed 0 tốt hoặc
một cấu hình được chọn bằng test set là bằng chứng đủ.

---

## 2. Baseline đã xác nhận

Kết quả RESISC45 hiện tại, trung bình 3 seed:

| Readout | Accuracy | Forgetting | Vai trò |
|---|---:|---:|---|
| Frozen ViT + NCM | 0.7114 | 0.0874 | Baseline chính cần vượt |
| Titans Linear | 0.6103 | -0.0251 | Readout phụ |
| Titans + NCM Online | 0.6190 | 0.2671 | Phương pháp online hiện tại |
| Titans + NCM Post-hoc | 0.7459 | 0.0656 | Oracle, có đọc lại old train |

Các bằng chứng đã có:

1. NCM Online thấp hơn Frozen NCM khoảng 9.24 điểm phần trăm.
2. Post-hoc cao hơn Online khoảng 12.69 điểm phần trăm.
3. Cosine giữa prototype Online và Post-hoc cuối stream còn khoảng 0.9158.
4. Titans state hữu hạn; norm lớn nhất theo seed khoảng 92-120.
5. Không có NaN/Inf trong campaign hoàn chỉnh.
6. Protocol `independent_image`, deterministic prototype loader, unseen-class
   mask và checkpoint NCM đã được kiểm thử.

Kết luận xuất phát:

- Nút thắt chính là representation drift làm prototype lớp cũ bị stale.
- Đây không còn chủ yếu là lỗi công thức NCM hay lỗi nổ số của M3.
- Post-hoc 0.7459 cho thấy Titans có tín hiệu hữu ích, nhưng head Online chưa giữ
  được tín hiệu đó qua toàn bộ stream.
- Khoảng trần quan sát so với Frozen NCM chỉ khoảng 3.45 điểm phần trăm. Vì vậy
  mọi tuyên bố "Titans giúp NCM" phải dựa trên nhiều seed và sai số đi kèm.

---

## 3. Phạm vi và định nghĩa giao thức

### 3.1. Được phép lưu

- Prototype, count và seen-mask: `O(C x D)`.
- Titans model, memory state, optimizer state và M3 state.
- Snapshot teacher trong thời gian train task hiện tại.
- Feature tạm thời của **task hiện tại** để ước lượng transport; phải giải phóng
  sau khi kết thúc task.
- Ma trận transport tạm thời hoặc transform tích lũy có kích thước bị chặn.
- Metrics, diagnostics và checkpoint phục vụ tái lập.

### 3.2. Không được phép

- Lưu ảnh của task cũ.
- Lưu feature theo từng ảnh của task cũ để dùng lại ở task sau.
- Mở lại train/prototype/validation loader của task cũ trong đường phương pháp
  no-replay.
- Chọn hyperparameter bằng test accuracy.
- Dùng Post-hoc làm kết quả chính.

### 3.3. Phân biệt evaluation và replay

Đánh giá trên test của các task cũ sau mỗi task vẫn cần thiết để lập ma trận
accuracy/forgetting và không được coi là training replay. Tuy nhiên test chỉ dùng
để báo cáo, không được điều khiển gamma, lambda, beta hoặc quyết định safety gate.

Post-hoc được giữ làm oracle chẩn đoán, nhưng campaign tuning chỉ nên chạy oracle
ở các mốc cần thiết để giảm thời gian.

---

## 4. Các giả thuyết cần kiểm chứng

### H1 - Titans có ích ở mức đóng góp nhỏ

`gamma=1` làm NCM kém, nhưng một lượng nhỏ feature Titans có thể bổ sung thông
tin mà vẫn giữ nền ổn định của Frozen ViT.

### H2 - Prototype transport thu hồi được phần accuracy mất do stale prototype

Nếu drift trên class hiện tại đại diện đủ tốt cho drift toàn cục, phép biến đổi
gần identity có thể đưa prototype cũ sang feature space mới mà không cần old data.

### H3 - Feature anchoring giảm drift tích lũy

Ép feature Titans không rời quá xa feature ViT cố định hoặc Titans trước task sẽ
giảm khoảng cách Online/Post-hoc, nhưng regularization quá mạnh có thể làm Titans
không học được task mới.

### H4 - Transport và distillation bổ sung cho nhau

Distillation giảm lượng drift; transport sửa phần drift còn lại. Kết hợp có thể
tốt hơn từng thành phần đơn lẻ.

### H5 - Một cơ chế phức tạp không mặc nhiên tốt hơn

Full transform hoặc lambda lớn có thể tăng current-task accuracy nhưng làm hỏng
old-task geometry. Mọi thành phần phải có ablation và safety gate.

---

## 5. Kiến trúc thí nghiệm mục tiêu

Ba feature cần được tách rõ:

```text
image
  |
  +--> frozen ViT --------------------> z_base
  |
  +--> frozen ViT --> Titans memory --> z_titans

z_blend(gamma) = normalize(
    (1 - gamma) * normalize(z_base)
    + gamma * normalize(z_titans)
)
```

Mỗi readout NCM phải ghi metadata đầy đủ:

- feature source và `gamma`;
- transport type, `lambda_transport`, `beta` và safety-gate decision;
- distillation teacher, `lambda_distill` và loss từng epoch/task;
- prototype sample count;
- old-data revisit policy;
- extra memory và runtime;
- seed, config hash và git commit.

---

## 6. Workstream A - Feature API và baseline bảo toàn

### A1. Tách feature component

Mở rộng `TitansClassifier` bằng API rõ ràng, dự kiến:

```text
feature_components(x, protocol) -> {
    base: z_base,
    titans: z_titans
}
features(x, protocol, blend_gamma=1.0) -> z_blend
```

Yêu cầu:

- `z_base` phải đúng feature Frozen ViT đang dùng bởi baseline NCM.
- `z_titans` với `gamma=1` phải tái lập đường NCM Online hiện tại.
- `gamma=0` phải tái lập Frozen NCM trong tolerance số học.
- Không để gọi component API làm thay đổi Titans state trong eval.
- `independent_image` vẫn độc lập với permutation và batch boundary.
- Hỗ trợ đúng `image_seq`; `token_seq` phải có định nghĩa pooling rõ hoặc fail
  sớm nếu chưa được hỗ trợ.

### A2. Forward training chỉ chạy student một lần

Feature distillation cần student feature và logits cùng batch. Không được gọi
`model(x)` rồi gọi lại `model.features(x)`, vì Titans ở train mode có thể cập
nhật state hai lần.

Thiết kế API dự kiến:

```text
forward_with_features(x) -> logits, feature_bundle
```

Engine dùng một forward duy nhất; auxiliary loss nhận `feature_bundle` đã tạo.
Các method cũ vẫn tương thích và không thay đổi hành vi.

### A3. Kiểm thử bảo toàn baseline

- `gamma=0` gần bằng Frozen NCM trên cùng ảnh và cùng backbone weights.
- `gamma=1` gần bằng feature Titans legacy mới trước refactor.
- Không đổi số lần cập nhật memory state mỗi train batch.
- Kết quả eval lặp lại không làm bẩn state.
- Feature và logits phải finite.

Tiêu chí hoàn thành A:

- Toàn bộ test cũ qua.
- Test mới chứng minh hai endpoint gamma đúng.
- Legacy seed/smoke nằm trong tolerance đã định trước.

---

## 7. Workstream B - Anchored Feature Blend

### B1. Multi-head gamma sweep trong cùng run

Tạo một `PrototypeHead` riêng cho mỗi gamma:

```text
gamma = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
```

Sau mỗi task, chỉ đọc deterministic prototype loader của task hiện tại một lần.
Từ cùng `z_base` và `z_titans`, tính mọi `z_blend(gamma)` rồi cập nhật các head.

Lợi ích:

- Không phải train model sáu lần.
- Các gamma dùng đúng cùng model, dữ liệu và task order.
- `gamma=0` là internal control cho Frozen NCM.
- `gamma=1` là internal control cho Titans NCM hiện tại.

### B2. Chọn gamma

Không dùng final test accuracy để chọn gamma. Hai chế độ cần báo cáo riêng:

1. **Global validation selection:** chọn một gamma duy nhất bằng validation
   aggregate của seed tuning; sau đó khóa gamma cho các seed replicate.
2. **Online task-adaptive gamma** chỉ là nghiên cứu phụ; nếu làm, quyết định tại
   task `t` chỉ được dùng dữ liệu current-task train/val và state đã có.

Ưu tiên global fixed gamma vì dễ tái lập và không thêm policy phức tạp.

### B3. Diagnostics

- Accuracy/forgetting theo gamma.
- Accuracy current task và old tasks riêng.
- Cosine `z_blend` với `z_base` và `z_titans`.
- Prototype alignment Online/Post-hoc ở các task mốc.
- Confusion matrix cuối stream cho gamma tốt nhất, 0 và 1.

Tiêu chí đi tiếp:

- Nếu không gamma nào cải thiện rõ so với `gamma=0` trên validation, vẫn giữ
  blend làm control nhưng không tuyên bố Titans có ích.
- Chỉ đưa tối đa hai gamma tốt nhất sang transport để tránh grid explosion.

---

## 8. Workstream C - Prototype Transport

### C1. Thu cặp feature không replay

Tại đầu task `t > 0`:

1. Snapshot model/state trước task.
2. Dùng deterministic loader của **task hiện tại** để tính `X_before`.
3. Train task như bình thường.
4. Dùng lại chính loader hiện tại để tính `X_after`.
5. Fit transform từ `X_before` sang `X_after`.
6. Transport prototype của class cũ.
7. Cập nhật prototype class mới từ feature sau task.
8. Xóa feature tạm thời và snapshot không còn cần thiết.

Không mở loader task cũ ở bất kỳ bước nào.

### C2. Thứ tự cập nhật prototype

Prototype class cũ phải được transport **trước** khi thêm prototype class mới.
Không transport class vừa học bằng một transform đã được fit trên chính class
đó nếu prototype mới có thể tính trực tiếp trong space hiện tại.

Nếu `PrototypeHead` tiếp tục lưu `proto_sum`, sau transport cần bảo toàn count và
đặt lại hướng sum một cách nhất quán:

```text
proto_sum[c] = normalize(p_transported[c]) * proto_count[c]
```

Do logits NCM chỉ dùng hướng cosine, norm/count không được vô tình thay đổi quyết
định nhưng vẫn phải hợp lệ cho checkpoint và diagnostics.

### C3. Các estimator cần thử theo tầng

#### C3.1. Identity

Không transport. Đây là control bắt buộc.

#### C3.2. Mean translation

Ước lượng vector dịch chuyển trung bình. Ít tham số, variance thấp, làm baseline
transport đơn giản.

#### C3.3. Diagonal affine

Mỗi chiều feature có scale/shift riêng; mạnh hơn translation nhưng vẫn bị chặn.

#### C3.4. Geometry-preserving hoặc identity-regularized map

Ứng viên chính:

```text
A = argmin ||X_before A - X_after||^2 + lambda * ||A - I||^2
p' = normalize((1 - beta) * p + beta * pA)
```

Grid ban đầu:

```text
lambda_transport = [1, 10, 100]
beta = [0.25, 0.5, 1.0]
```

Orthogonal Procrustes có thể dùng làm ablation vì bảo toàn góc, nhưng không dùng
full unconstrained rotation làm mặc định. Nếu transform phức tạp không thắng
translation/diagonal trên validation thì loại sớm.

### C4. Safety gate

Transport chỉ được áp dụng khi trên current-task validation hoặc held-out portion:

- prediction error `X_before -> X_after` tốt hơn identity một margin tối thiểu;
- output finite;
- condition number/rank nằm trong ngưỡng;
- norm và cosine shift không vượt ngưỡng cấu hình;
- current-task validation NCM không giảm quá tolerance.

Nếu gate fail, task đó dùng identity và log rõ `transport_skipped=true` cùng lý do.
Safety gate tuyệt đối không đọc old validation/test.

### C5. Diagnostics bắt buộc

- Fit error train và current validation.
- Identity error để so sánh.
- Transform norm, rank, singular values/condition number.
- Mean prototype cosine trước/sau transport.
- Số task gate accept/skip.
- Accuracy old/current task sau từng task.
- Online/Post-hoc prototype alignment ở task mốc.
- Runtime và peak memory của fit transform.

---

## 9. Workstream D - Feature Distillation không replay

### D1. Biến thể chính: fixed-backbone anchor

ViT đã đóng băng và Frozen NCM là baseline ổn định. Loss chính:

```text
L_total = L_CE + lambda_anchor * L_anchor
L_anchor = 1 - cosine(normalize(z_titans), normalize(z_base))
```

Grid ban đầu:

```text
lambda_anchor = [0.01, 0.05, 0.1]
```

Nếu scale loss chênh lệch lớn, log riêng CE và anchor loss trước khi thay grid.
Không tăng số lambda một cách mù quáng.

Ưu điểm:

- Teacher cố định, không tích lũy drift.
- Không cần giữ bản sao toàn model.
- So sánh trực tiếp với feature space đã giúp Frozen NCM đạt 0.7114.

### D2. Biến thể phụ: previous-Titans teacher

Trước task `t`, chụp teacher Titans và state. Trên ảnh task hiện tại:

```text
L_temporal = 1 - cosine(z_student, stopgrad(z_teacher_previous))
```

Chỉ chạy sau khi fixed-anchor đã có kết quả. Teacher phải:

- ở eval mode;
- không cập nhật state;
- dùng cùng định nghĩa feature/pooling;
- được giải phóng sau task;
- được đưa vào footprint/runtime report.

Không dùng LwF logits hiện tại để thay thế vì logit distillation không trực tiếp
bảo vệ hệ tọa độ feature NCM.

### D3. Kiểm soát stability-plasticity

Theo dõi riêng:

- current-task accuracy: khả năng học cái mới;
- old-task accuracy/forgetting: khả năng giữ cái cũ;
- cosine student-base và student-previous-teacher;
- CE loss và distillation loss;
- gradient norm trước/sau clipping;
- Titans state norm và finite check.

Loại cấu hình nếu current-task learning giảm mạnh dù forgetting tốt hơn. Mục tiêu
không phải đóng băng Titans trá hình.

---

## 10. Workstream E - Kết hợp có kiểm soát

Không chạy full Cartesian grid của gamma x transport x beta x lambda-distill.
Chọn tuần tự bằng validation:

1. Chọn tối đa hai gamma từ Workstream B.
2. Chọn một transport estimator và tối đa hai cặp `(lambda_transport, beta)`.
3. Chọn một `lambda_anchor` từ Workstream D.
4. Chỉ sau đó chạy tổ hợp.

Các ablation bắt buộc:

| ID | Blend | Transport | Distillation | Mục đích |
|---|---|---|---|---|
| A0 | gamma=0 | Không | Không | Frozen control trong cùng pipeline |
| A1 | gamma=1 | Không | Không | Titans Online hiện tại |
| A2 | gamma tốt nhất | Không | Không | Hiệu quả riêng của Blend |
| A3 | gamma tốt nhất | Có | Không | Hiệu quả bổ sung của Transport |
| A4 | gamma tốt nhất | Không | Fixed anchor | Hiệu quả riêng của Distillation |
| A5 | gamma tốt nhất | Có | Fixed anchor | Phương pháp đầy đủ |
| A6 | gamma tốt nhất | Có | Previous Titans | Ablation teacher, nếu cần |

`Transport + Distillation` với `gamma=1` có thể chạy như ablation, nhưng không
phải ứng viên chính nếu gamma sweep cho thấy Titans full-strength không ổn định.

---

## 11. Phương án dự phòng: Dual-space NCM

Nếu Anchored Blend không cho một gamma ổn định qua seed, thử một baseline thực
dụng với hai head:

```text
score = (1 - w) * cosine(z_base, p_base)
      + w * calibrated_cosine(z_titans, p_titans)
```

- `p_base` luôn ổn định vì ViT frozen.
- `p_titans` có thể dùng transport.
- Không replay old data.
- Bộ nhớ prototype tăng khoảng `C x D`, vẫn rất nhỏ.

Chỉ thử sau A2-A5 vì cần calibration temperature/scale và khó diễn giải hơn
feature blend. Phải báo cáo rõ đây là ensemble/fusion, không được dùng để che
việc nhánh Titans riêng kém.

---

## 12. Thiết kế config và output

Đề xuất config có namespace riêng, không làm thay đổi run cũ:

```yaml
train:
  ncm:
    enabled: true
    feature_protocol: independent_image
    blend:
      enabled: true
      gammas: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    transport:
      enabled: false
      type: identity_ridge
      regularization: 10.0
      beta: 0.5
      safety_gate: true
    feature_distillation:
      enabled: false
      teacher: frozen_backbone
      weight: 0.05
      loss: cosine
```

Output tách riêng:

- `metrics_ncm_blend_gamma_*.json`
- `acc_matrix_ncm_blend_gamma_*.csv`
- `ncm_transport_diagnostics.json`
- `ncm_distillation_diagnostics.json`
- `ncm_method_summary.json`
- checkpoint chứa mọi PrototypeHead và transport metadata cần cho inference.

Tên run phải encode ít nhất seed, gamma, transport type/beta/regularization,
distillation teacher/weight và protocol version, hoặc lưu config hash chống đè.

---

## 13. Kế hoạch kiểm thử

### 13.1. Unit tests

- Blend endpoint: gamma 0 và 1.
- Blend normalization và finite guard.
- Một lần forward train chỉ cập nhật state một lần.
- Transport identity không đổi prototype/logits.
- Translation/diagonal/ridge khôi phục đúng synthetic transform đã biết.
- Transport chỉ tác động class cũ, không đụng unseen/current class.
- Count/seen-mask/checkpoint được bảo toàn.
- Ill-conditioned hoặc non-finite transform bị reject.
- Safety gate accept/skip đúng.
- Distillation loss bằng 0 khi student bằng teacher.
- Gradient chỉ chảy vào student, không vào backbone/teacher.

### 13.2. Protocol tests

- No-replay method không truy cập loader task cũ.
- Feature tạm của current task không tồn tại sau task.
- Không dùng test loader để fit/gate/select hyperparameter.
- `independent_image` không phụ thuộc permutation/batch size.
- Resume từ task-boundary checkpoint cho kết quả tương đương run liên tục.
- Checkpoint inference tái tạo đúng logits cho blend/transport head.

### 13.3. Integration smoke

- Tiny synthetic stream ít nhất 3 task để old prototype thực sự trải qua nhiều
  lần transport.
- RESISC45 smoke với số class/task nhỏ hoặc epoch giảm.
- Quét JSON đệ quy để fail khi có NaN/Inf.
- Xác minh mọi readout ghi đúng `revisit_old_train=false`.

### 13.4. Regression

- Toàn bộ test suite hiện tại phải qua.
- Legacy NCM post-hoc seed/smoke trong tolerance đã chốt.
- Cấu hình không bật tính năng mới giữ nguyên hành vi.

---

## 14. Chiến lược thí nghiệm

### Phase 0 - Cơ chế và smoke local

1. Chạy unit/protocol tests.
2. Chạy synthetic stream.
3. Chạy RESISC45 smoke trên MPS/local nếu phù hợp.
4. Kiểm tra state update count, NaN/Inf và artifact schema.

Không đưa lên campaign lớn nếu Phase 0 chưa qua.

### Phase 1 - Gamma sweep, một training run seed 0

- Train model hiện tại một lần.
- Đánh giá toàn bộ gamma đồng thời.
- Chọn tối đa hai gamma bằng validation.
- Post-hoc chỉ cần chạy gamma 0, gamma 1 và gamma tốt nhất ở cuối stream hoặc các
  task mốc 0/4/8.

### Phase 2 - Transport screening, seed 0

Thử theo successive halving:

1. Identity, translation, diagonal và một identity-ridge mặc định.
2. Loại estimator fail safety gate nhiều hoặc không cải thiện validation.
3. Với estimator tốt nhất, thử grid nhỏ regularization/beta.
4. Chọn tối đa hai cấu hình transport.

### Phase 3 - Distillation screening, seed 0

- Chạy fixed-backbone anchor với ba lambda.
- Chỉ thử previous-Titans teacher nếu fixed anchor không đủ hoặc cần ablation.
- Chọn lambda bằng validation và kiểm tra current-task plasticity.

### Phase 4 - Combination, seed 0

Chạy A0-A5 bằng hyperparameter đã khóa từ các phase trước. Không retune dựa trên
test của phase này.

### Phase 5 - Replicate 3 seed

Chọn trước tối đa hai ứng viên:

- phương pháp đầy đủ tốt nhất;
- phương pháp đơn giản hơn nếu accuracy gần tương đương.

Chạy seed `0, 1, 2` và so với đúng Frozen NCM, Titans Online hiện tại và Post-hoc
oracle trên cùng split/protocol.

Nếu chênh lệch với Frozen NCM nhỏ hơn khoảng 1 điểm phần trăm hoặc variance lớn,
mở rộng lên 5 seed trước khi kết luận.

### Phase 6 - Robustness và ablation cuối

- Báo cáo per-task matrix, forgetting, BWT và current/old accuracy.
- Ablation từng thành phần.
- Sensitivity quanh gamma/lambda/beta đã chọn.
- Runtime, VRAM, checkpoint size và footprint.
- Ít nhất một lần resume Spot checkpoint.
- Không cần chạy full oracle sau mọi task cho mọi seed.

---

## 15. Tiêu chí lựa chọn và kết luận

### 15.1. Điều kiện hợp lệ

- 0 NaN/Inf.
- 0 truy cập old train/prototype/val loader trong phương pháp no-replay.
- Checkpoint round-trip đúng.
- Mọi config/seed/artifact đầy đủ.
- Không chọn hyperparameter bằng test.
- Ít nhất 3 seed hoàn chỉnh.

### 15.2. Mức kết luận

#### Không đạt

- Mean accuracy không vượt Frozen NCM 0.7114, hoặc chỉ một seed vượt.
- Cải thiện accuracy nhưng forgetting/runtime/memory tăng không chấp nhận được.

Kết luận: Titans chưa chứng minh được lợi ích cho NCM online no-replay.

#### Có tín hiệu

- Mean cao hơn Frozen NCM nhưng dưới +1 điểm phần trăm hoặc confidence còn chồng
  lấn mạnh.

Kết luận: có tín hiệu, cần 5 seed; chưa tuyên bố chắc chắn.

#### Thành công

- Mean vượt Frozen NCM ít nhất khoảng +1 điểm phần trăm.
- Phần lớn hoặc mọi seed đều vượt baseline.
- Không đánh đổi bằng forgetting bất thường.
- No-replay audit và checkpoint đều qua.

#### Thành công mạnh

- Tiến gần Post-hoc oracle 0.7459, ví dụ đạt khoảng 0.73-0.74 ổn định.
- Ablation chứng minh đóng góp của Titans/transport/distillation, không chỉ nhờ
  quay về gần `gamma=0`.

### 15.3. Kiểm tra Titans thực sự đóng góp

Một phương pháp vượt Frozen nhưng gamma gần 0 chưa đủ để nói Titans đóng góp lớn.
Cần đồng thời báo cáo:

- gamma được chọn;
- chênh lệch A2 - A0;
- chênh lệch A3/A5 - A2;
- feature/prototype alignment;
- kết quả bỏ nhánh Titans.

---

## 16. Campaign GCP và vận hành an toàn

Thực hiện theo `RaybanMeta/An/GCP_TRAINING_GUIDE.md`.

Yêu cầu campaign runner:

- preflight environment, CUDA và dataset integrity;
- `--dry-run` để kiểm tra toàn bộ command/config trước khi train;
- skip run đã hoàn chỉnh;
- task-boundary checkpoint và resume idempotent;
- log stdout/stderr theo từng run;
- fail-fast khi NaN/Inf hoặc artifact thiếu;
- summary có số run valid/failed/skipped;
- tải artifact về local và kiểm tra checksum/JSON;
- chỉ dừng VM sau khi artifact local đã xác nhận đầy đủ;
- không xóa VM hoặc persistent disk;
- Spot preemption phải tiếp tục được từ checkpoint.

Để kiểm soát chi phí, thứ tự chạy là:

```text
tests -> smoke -> seed0 screening -> khóa config -> seeds 0/1/2 -> robustness
```

Không gửi toàn bộ grid lên VM ngay từ đầu.

---

## 17. Deliverables

### Code

- Feature component/blend API.
- Multi-gamma PrototypeHead manager hoặc cấu trúc tương đương.
- Prototype transport với estimator, regularization và safety gate.
- Fixed-backbone feature distillation.
- Previous-Titans teacher ablation nếu Phase 3 yêu cầu.
- Checkpoint/resume cho toàn bộ state mới.
- Campaign runner và report generator.

### Tests

- Unit, protocol, integration và regression tests mô tả ở mục 13.

### Artifacts

- Config cuối của mọi run.
- Train log và auxiliary-loss log.
- Accuracy matrices và metrics từng readout.
- Transport/distillation diagnostics.
- Full inference checkpoint.
- Campaign manifest, environment và git commit.

### Báo cáo cuối

Báo cáo trong `RaybanMeta/An` phải trả lời rõ:

1. Titans + NCM no-replay có vượt Frozen NCM không?
2. Phần tăng/giảm đến từ Blend, Transport hay Distillation?
3. Bao nhiêu seed hợp lệ và variance ra sao?
4. Có đọc lại old data hay không?
5. Chi phí memory/runtime tăng bao nhiêu?
6. Khoảng cách còn lại tới Post-hoc oracle là bao nhiêu?
7. Cấu hình nào nên dùng thực tế và giới hạn của nó là gì?

---

## 18. Thứ tự triển khai đề xuất

- [x] P0. Đóng băng baseline artifact và schema so sánh.
- [x] P1. Implement feature component API và one-forward training API.
- [x] P2. Unit/regression test endpoint gamma và state lifecycle.
- [x] P3. Implement multi-gamma Anchored Feature Blend.
- [x] P4. Implement no-replay audit và diagnostics cho blend.
- [x] P5. Chạy smoke và gamma sweep seed 0.
- [x] P6. Implement transport estimators, safety gate và checkpoint.
- [x] P7. Test synthetic transform, loader-access và resume.
- [x] P8. Chạy transport screening seed 0.
- [x] P9. Implement fixed-backbone feature distillation bằng một student forward.
- [x] P10. Test gradient/state lifecycle và chạy distillation screening seed 0.
- [x] P11. Chạy ablation A0-A5, khóa cấu hình bằng validation.
- [x] P12. Chạy 3 seed trên GCP Spot; không mở rộng 5 seed vì full combination thua baseline 2.16 điểm phần trăm.
- [x] P13. Tải và hậu kiểm artifact; dừng VM sau khi xác nhận dữ liệu local.
- [x] P14. Viết báo cáo cuối và khuyến nghị cấu hình production/research.

---

## 19. Quyết định hiện tại

Ưu tiên kỹ thuật và khoa học:

1. **Anchored Feature Blend** là bước đầu và control bắt buộc.
2. **Prototype Transport gần identity có safety gate** là cơ chế sửa drift chính.
3. **Fixed-backbone feature distillation** được ưu tiên hơn previous-Titans
   distillation vì teacher ViT ổn định và rẻ hơn.
4. **Blend + Transport + Distillation** là ứng viên đầy đủ, nhưng chỉ chạy sau
   khi từng thành phần đã chứng minh có ích.
5. **Dual-space NCM** là phương án dự phòng thực dụng.
6. Replay exemplar chỉ dùng làm benchmark riêng, không trộn vào kết luận no-replay.

Plan hoàn thành khi có kết luận 3 seed hợp lệ về câu hỏi Titans có giúp NCM
Online no-replay vượt Frozen NCM hay không, chứ không chỉ khi code chạy xong.
