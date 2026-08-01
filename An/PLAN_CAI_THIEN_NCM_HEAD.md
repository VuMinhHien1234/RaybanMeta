# Kế hoạch cải thiện NCM Head

## 1. Thông tin và mục tiêu

- Nhánh thực hiện: `NCM_Head`
- Project: `RaybanMeta/uav-continual-learning`
- Ngày audit: 2026-07-30
- Cấu hình chính cần phục vụ: Titans mới + M3 improved trên RESISC45, 9 task
- Mục tiêu cuối:
  1. Giữ lại kết quả NCM hậu kiểm hiện tại để làm mốc phân tích.
  2. Xây dựng một NCM head online đúng giao thức continual learning, có bộ nhớ bị chặn.
  3. Đảm bảo NCM có thể lưu, nạp và dùng để inference mà không cần đọc lại toàn bộ ảnh train cũ.
  4. So sánh công bằng Linear head, NCM online và NCM hậu kiểm trên cùng model, cùng seed.
  5. Kết luận chính xác phần tăng accuracy đến từ feature Titans/M3 hay từ việc đọc lại dữ liệu cũ.

Plan này ưu tiên độ đúng của giao thức và khả năng tái lập. Không lấy việc làm tăng một con số accuracy
đơn lẻ làm tiêu chuẩn hoàn thành.

---

## 2. Kết luận audit hiện trạng

### 2.1. Project đang có hai cơ chế NCM khác nhau

#### A. NCM baseline G1

Các file chính:

- `src/uavcl/models/ncm.py`
- `src/uavcl/methods.py`
- `tests/test_ncm.py`

Cách hoạt động:

1. Backbone được đóng băng.
2. Mỗi feature được chuẩn hóa L2.
3. Với mỗi class, model giữ `proto_sum` và `proto_count`.
4. Prototype là feature trung bình của class rồi được chuẩn hóa L2.
5. Inference dùng cosine similarity giữa feature và prototype.
6. Khi task mới đến, chỉ prototype của class mới được cập nhật.

Đánh giá:

- Đây là NCM online bounded tương đối đúng.
- Bộ nhớ là `O(C x D)`, với `C` là số class và `D` là chiều feature.
- Vì backbone đóng băng nên feature space không trôi; prototype cũ vẫn có cùng ý nghĩa.
- `proto_sum` và `proto_count` là buffer nên có thể đi theo `state_dict`.
- Test hiện tại đã kiểm tra forgetting thấp, footprint bị chặn và kết quả hữu hạn.

Phần này không phải nguồn chính của vấn đề NCM Head hiện tại.

#### B. NCM-head của Titans/M3

Các file chính:

- `src/uavcl/engine.py`, hàm `_memory_prototypes` và `_evaluate_ncm`
- `scripts/run_g1.py`, phần ghi `metrics_ncm.json`
- Cờ cấu hình `train.eval_ncm_head=true`

Cách hoạt động sau task `t`:

1. Lấy model Titans/M3 hiện tại.
2. Đọc lại train loader của tất cả task từ `0..t`.
3. Chạy lại toàn bộ ảnh train qua `model.features`.
4. Dựng lại toàn bộ prototype từ đầu trong feature space hiện tại.
5. Đánh giá cosine NCM trên test của tất cả task đã thấy.
6. Không giữ prototype này làm state lâu dài; sang lần đánh giá sau lại dựng lại.

Tên hiện tại trong output là `ncm_head_posthoc`. Tên này gần đúng, nhưng chưa nói rõ việc dùng lại
toàn bộ train của các task đã thấy.

Tên đề xuất:

`ncm_posthoc_full_seen_train`

### 2.2. Những phần đang làm đúng

- Không dùng test set để dựng prototype.
- Không dùng class hoặc dữ liệu của task tương lai.
- Dùng class-incremental evaluation: khi test chỉ cho phép các class đã thấy.
- Feature và prototype đều được chuẩn hóa L2.
- Dùng trung bình class và cosine similarity đúng công thức NCM cơ bản.
- Ba seed tốt nhất của M3 improved cho kết quả NCM tương đối ổn định:
  - Accuracy: `0.7403 +/- 0.0043`
  - Forgetting: `0.0635 +/- 0.0054`

Do đó kết quả 0.7403 không phải số giả. Nó cho thấy feature của Titans mới + M3 improved chứa tín hiệu
phân lớp tốt hơn Linear head hiện tại khai thác được.

### 2.3. Những phần chưa hoàn chỉnh hoặc chưa công bằng

#### Vấn đề P0: NCM Titans hiện tại không phải NCM online bounded

Sau mỗi task, code đọc lại toàn bộ ảnh train của mọi task cũ. Điều này không phải test leakage theo
nghĩa thông thường, nhưng vi phạm giao thức “không revisit dữ liệu cũ nếu không có replay”.

Hệ quả:

- Không thể coi kết quả này là một head deploy được mà không giữ dataset cũ.
- Chi phí giữ/truy cập ảnh cũ không được tính trong `method_extra_floats`.
- So sánh trực tiếp với method không được revisit dữ liệu cũ là không hoàn toàn công bằng.
- NCM forgetting hiện tại trộn lẫn representation drift và nhiễu do dựng lại prototype.

#### Vấn đề P0: Feature Titans phụ thuộc thứ tự và ranh giới batch

Trong `image_seq`, adapter đổi tensor `(B, D)` thành `(1, B, D)`. Vì vậy cả batch ảnh được coi là một
chuỗi thời gian. Feature của một ảnh có thể phụ thuộc các ảnh đứng trước nó.

Trong kiểm tra cô lập, cùng một tập ảnh nhưng đổi thứ tự batch rồi đổi prediction về thứ tự cũ đã tạo
ra sai khác feature tối đa khoảng `1.918`. Đây là sai khác lớn, không chỉ là nhiễu số học.

Hệ quả:

- Prototype phụ thuộc shuffle và cách chia batch.
- Test accuracy có thể phụ thuộc thứ tự DataLoader.
- Train prototype đang shuffle, test không shuffle nên hai bên dùng hai ngữ cảnh chuỗi khác nhau.
- Thay đổi `batch_size` có thể làm thay đổi kết quả dù model và ảnh không đổi.

Đây là vấn đề giao thức của Titans image classification nói chung, không chỉ riêng NCM. Tuy nhiên NCM
làm vấn đề lộ rõ hơn vì prototype là trung bình của các feature phụ thuộc ngữ cảnh batch.

#### Vấn đề P1: Prototype đang dùng augmentation ngẫu nhiên

`task_loaders[tid]["train"]` dùng:

- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `shuffle=True`

Trong khi test dùng resize cố định và không shuffle.

Hệ quả:

- Dựng prototype không hoàn toàn deterministic.
- Kết quả phụ thuộc trạng thái RNG và thứ tự các lần gọi.
- Prototype train và feature test không dùng cùng một pipeline ảnh xác định.

#### Vấn đề P1: Không lưu NCM head hoàn chỉnh

Prototype hậu kiểm hiện tại là biến local và bị bỏ sau đánh giá. Runner chỉ lưu:

- metrics
- accuracy matrix
- train log
- Titans `memory_state.pt` nếu có

Runner chưa lưu full model weights và chưa lưu prototype/count của NCM Titans.

Hệ quả:

- Không thể nạp một run đã hoàn thành và inference bằng NCM ngay.
- Chỉ có `memory_state.pt` là chưa đủ vì còn trọng số memory, post-norm, Linear head và NCM state.
- Muốn tái tạo NCM phải đọc lại ảnh train cũ.

#### Vấn đề P1: Thiếu kiểm tra numerical health trong đường NCM

Các hàm NCM hậu kiểm chưa kiểm tra:

- feature có NaN/Inf hay không;
- `proto_sum`, count và prototype có hữu hạn hay không;
- mỗi class được phép đánh giá có count lớn hơn 0 hay không;
- logits cosine có NaN/Inf hay không.

Nếu logits có NaN, `argmax` vẫn có thể trả về class và tạo ra accuracy nhìn có vẻ hợp lệ.

#### Vấn đề P1: Test chưa bao phủ đúng rủi ro

Các test liên quan hiện tại đã qua: `13 passed`.

Nhưng chưa có test cho:

- NCM online không truy cập loader task cũ;
- prototype loader deterministic;
- NCM checkpoint round-trip;
- class chưa có prototype bị mask đúng;
- NaN/Inf phải làm run thất bại;
- kết quả không đổi khi lặp lại evaluation;
- độ nhạy với shuffle, batch size và thứ tự ảnh;
- nhãn output phân biệt rõ online và post-hoc.

---

## 3. Định nghĩa ba readout cần tách riêng

Không dùng chung tên “NCM head” cho ba chế độ sau.

### 3.1. `ncm_frozen_baseline`

- NCM G1 hiện tại.
- Feature từ backbone pretrained đóng băng.
- Cập nhật một lần khi dữ liệu hiện tại đến.
- Không revisit task cũ.
- Bộ nhớ `O(C x D)`.
- Dùng làm baseline độc lập.

### 3.2. `ncm_online_current_task`

- NCM mới cần triển khai cho Titans/M3.
- Sau khi học xong task `t`, chỉ đọc dữ liệu train của task `t`.
- Mỗi mẫu được dùng một lần trong pass dựng prototype xác định.
- Prototype cũ không được dựng lại từ ảnh cũ.
- Lưu `proto_sum`, `proto_count`, `seen_mask`.
- Bộ nhớ `O(C x D)`.

Đây là kết quả NCM chính nếu team muốn tuyên bố một NCM head continual-learning thật sự.

Lưu ý khoa học: feature extractor Titans/M3 thay đổi theo thời gian. Vì vậy prototype cũ có thể bị
“stale”, tức được tạo trong feature space cũ trong khi ảnh mới được biểu diễn trong feature space mới.
Không được che giấu hạn chế này bằng cách tự động đọc lại toàn bộ ảnh cũ.

### 3.3. `ncm_posthoc_full_seen_train`

- Giữ logic hậu kiểm hiện tại.
- Sau mỗi task, đọc lại toàn bộ train của các task đã thấy.
- Dựng lại prototype trong feature space mới nhất.
- Phải ghi rõ `revisit_old_train=true`.
- Không dùng làm kết quả online chính.

Vai trò:

- Là oracle/readout chẩn đoán chất lượng feature.
- Cho biết “nếu có lại toàn bộ dữ liệu cũ thì NCM tốt đến đâu”.
- Khoảng cách giữa post-hoc và online đo ảnh hưởng của feature drift/prototype staleness.

---

## 4. Quyết định thiết kế

### 4.1. Không xóa hoặc âm thầm thay đổi kết quả cũ

- Giữ khả năng tái tạo `ncm_head_posthoc` cũ trong một chế độ legacy.
- Output mới phải có tên rõ hơn.
- Report cũ không được tự động diễn giải lại thành online NCM.
- Backward compatibility cho `train.eval_ncm_head=true`: map sang post-hoc và in deprecation warning.

### 4.2. Tách “head state” khỏi “cách thu thập feature”

Tạo một module prototype dùng lại được, dự kiến trong `src/uavcl/models/ncm.py`:

```text
PrototypeHead
  - proto_sum: (num_classes, feat_dim)
  - proto_count: (num_classes,)
  - seen_mask: (num_classes,)
  - update(features, labels)
  - prototypes()
  - logits(features, allowed_classes)
  - reset()
  - extra_floats()
```

Module không tự đọc DataLoader và không biết task cũ/task mới. Engine chịu trách nhiệm giao thức.

Lợi ích:

- Unit test công thức độc lập.
- Dùng được cho frozen NCM và Titans NCM.
- Buffer đi theo `state_dict`.
- Không trộn logic model, data và evaluation.

### 4.3. Có loader riêng để dựng prototype

Mỗi task nên có thêm loader `prototype`:

- dùng đúng train indices của task;
- dùng eval transform cố định;
- `shuffle=False`;
- `drop_last=False`;
- batch size cấu hình rõ;
- không dùng test/val;
- không augmentation ngẫu nhiên.

Không tái sử dụng loader `"train"` để dựng prototype.

### 4.4. Đặt feature-evaluation protocol thành cấu hình tường minh

Trước khi chốt kết quả, cần hỗ trợ và so sánh hai semantics:

1. `stream_batch_legacy`
   - Giữ hành vi hiện tại: một batch ảnh là một chuỗi.
   - Chỉ dùng để tái tạo kết quả cũ.
   - Phải log batch size và shuffle.

2. `independent_image`
   - Mỗi ảnh test/prototype bắt đầu từ cùng một bản sao Titans state.
   - Ảnh trong cùng batch không tác động feature của nhau.
   - Kết quả phải bất biến theo permutation và batch size trong sai số số học.
   - Đây là ứng viên mặc định cho class-incremental image classification.

Không đổi semantics train trong cùng task ở bước đầu. Việc đổi train từ batch-sequence sang
independent-image là một ablation Titans riêng, có blast radius lớn hơn NCM.

Nếu thư viện memory chưa vector hóa được `(B, 1, D)` với state độc lập, bản đầu có thể chạy từng ảnh
để xác nhận độ đúng. Sau đó mới tối ưu vector hóa. Phải benchmark vì cách từng ảnh có thể chậm.

### 4.5. Online NCM cập nhật đúng một deterministic pass sau task

Không cập nhật prototype trong mọi epoch train vì như vậy:

- cùng ảnh bị đếm nhiều lần;
- augmentation khác nhau làm ý nghĩa count không rõ;
- thay đổi `epochs_per_task` làm thay đổi head dù dữ liệu giống nhau.

Quy trình đề xuất:

1. Train task `t` xong.
2. Chuyển model sang eval.
3. Đóng băng snapshot state dùng cho feature extraction.
4. Chạy đúng một pass loader `prototype` của task `t`.
5. Update only class thuộc task `t`.
6. Đánh giá online NCM.

### 4.6. Checkpoint phải đủ cho inference

Checkpoint tối thiểu cần lưu:

- `model.state_dict()`;
- Titans exported memory state;
- online NCM `state_dict()` gồm sum/count/seen mask;
- danh sách class đã thấy;
- config đã resolve;
- git commit;
- format version.

Nên dùng một file có schema rõ, ví dụ `checkpoint.pt`, thay vì nhiều file rời không có contract.

Không cần lưu optimizer trong checkpoint inference. Nếu sau này cần resume training chính xác thì tạo
training checkpoint riêng có optimizer/M3 state và RNG state.

---

## 5. Config contract đề xuất

Ví dụ:

```yaml
train:
  ncm:
    enabled: true
    readouts:
      - online_current_task
      - posthoc_full_seen_train
    feature_protocol: independent_image
    prototype_transform: eval
    prototype_batch_size: 32
    fail_on_nonfinite: true
    save_state: true
```

Quy tắc:

- `readouts` là danh sách để một lần train có thể chấm nhiều head.
- `posthoc_full_seen_train` phải log `revisit_old_train=true`.
- `online_current_task` phải log `revisit_old_train=false`.
- `prototype_transform` bản chính chỉ cho phép `eval`.
- `feature_protocol` luôn xuất hiện trong metrics.
- Cờ cũ `eval_ncm_head` chỉ tồn tại trong giai đoạn chuyển đổi.

---

## 6. Kế hoạch triển khai theo giai đoạn

### Giai đoạn 0 - Khóa protocol và dựng test tái hiện

Mục tiêu: biến các rủi ro hiện tại thành test đo được trước khi sửa.

Task:

1. Thêm test tái hiện độ nhạy permutation của `stream_batch_legacy`.
2. Thêm test đổi batch size làm feature hoặc accuracy thay đổi.
3. Thêm test cho thấy post-hoc có truy cập loader của task cũ.
4. Thêm test lặp dựng prototype với train augmentation có thể khác nhau.
5. Ghi một file audit machine-readable nhỏ cho smoke run:
   - feature protocol;
   - prototype source;
   - shuffle;
   - revisit count;
   - prototype counts.

Tiêu chí hoàn thành:

- Test phải mô tả đúng hành vi hiện tại.
- Test legacy không được viết theo kiểu mong nó pass tính bất biến; nó phải đánh dấu đây là hành vi
  đã biết, để khi thêm protocol mới có thể so sánh rõ.

### Giai đoạn 1 - Refactor PrototypeHead an toàn

Mục tiêu: có core NCM đúng, tái sử dụng, không thay đổi kết quả baseline ngoài sai số nhỏ.

Task:

1. Tạo `PrototypeHead` với registered buffers.
2. Thêm finite checks cho input, sum, count, prototype và logits.
3. Mask class chưa có prototype bằng `-inf`.
4. Validate label range và shape.
5. Cho `NCMClassifier` G1 dùng core mới hoặc bọc tương thích.
6. Giữ API `update_prototypes`, `prototypes`, `extra_floats`.

Test bắt buộc:

- prototype bằng mean tính tay;
- normalize đúng;
- update nhiều batch bằng update một batch;
- class unseen không thể thắng argmax;
- label ngoài range báo lỗi;
- NaN/Inf báo lỗi;
- state_dict save/load cho logits giống nhau;
- test NCM G1 cũ tiếp tục pass.

Gate:

- Không sang giai đoạn 2 nếu G1 regression.

### Giai đoạn 2 - Prototype loader deterministic

Mục tiêu: prototype không phụ thuộc augmentation và shuffle.

Task:

1. Mở rộng `build_task_loaders` để có loader `"prototype"`.
2. Dùng train split/train indices nhưng eval transform.
3. `shuffle=False`, `drop_last=False`.
4. Cho phép batch size riêng nhưng mặc định bằng eval batch size.
5. Log số mẫu prototype theo class.
6. Đổi đường `gradient_free` của NCM G1 sang loader `"prototype"` để baseline cũng không phụ thuộc
   augmentation ngẫu nhiên; giữ mode legacy nếu cần reproduce artifact cũ.

Test bắt buộc:

- indices của prototype loader đúng bằng train indices;
- transform không chứa random crop/flip;
- hai lần iterate cho tensor/label giống nhau;
- không lấy mẫu từ val/test;
- không mất mẫu ở batch cuối.

### Giai đoạn 3 - Feature protocol độc lập với batch

Mục tiêu: tạo feature dùng cho NCM/eval không phụ thuộc ảnh khác trong batch.

Task:

1. Thêm API tường minh, ví dụ:
   - `model.features(x, protocol="stream_batch_legacy")`
   - `model.features(x, protocol="independent_image")`
2. Với `independent_image`, mỗi ảnh đọc từ cùng snapshot state.
3. Không được ghi hoặc thay đổi `_state` thật khi dựng prototype/test.
4. Tối ưu vector hóa sau khi bản đúng đã có.
5. Benchmark runtime trên CPU/MPS/CUDA và batch 1/8/32.

Test bắt buộc:

- permutation invariance;
- batch-size invariance;
- lặp evaluate không làm đổi Titans state;
- output shape đúng;
- feature hữu hạn;
- legacy protocol vẫn tái tạo hành vi cũ;
- independent protocol không làm model state thay đổi.

Gate:

- Sai khác khi reorder/unbatch phải nhỏ hơn `1e-5` trên CPU float32.
- Nếu không đạt, chưa được dùng `independent_image` làm kết quả chính.

### Giai đoạn 4 - Online NCM đúng giao thức

Mục tiêu: có readout `ncm_online_current_task`.

Task:

1. Khởi tạo một `PrototypeHead` online cùng model/run.
2. Sau task `t`, chỉ truyền loader `"prototype"` của task `t`.
3. Không truy cập loader train/prototype của task `< t`.
4. Cập nhật count đúng số ảnh một lần, không nhân theo epoch.
5. Tạo ma trận accuracy riêng `ncm_online_R`.
6. Log:
   - count từng class;
   - norm prototype;
   - seen mask;
   - finite status;
   - memory floats;
   - số ảnh cũ được revisit, phải bằng 0.

Test bắt buộc:

- dùng poison loader cho task cũ; online run vẫn phải hoàn thành;
- count bằng đúng số mẫu train của class;
- không update prototype class cũ khi học class mới;
- output matrix có đúng shape;
- class chưa thấy bị mask;
- save/load giữ nguyên prediction.

### Giai đoạn 5 - Giữ post-hoc oracle nhưng ghi nhãn trung thực

Mục tiêu: không mất mốc 0.7403, đồng thời tránh hiểu nhầm.

Task:

1. Đổi tên output mới thành `ncm_posthoc_full_seen_train`.
2. Dùng prototype loader deterministic.
3. Chọn feature protocol bằng config.
4. Log:
   - `revisit_old_train=true`;
   - tổng số sample encode lại sau từng task;
   - estimated raw-data dependency;
   - runtime riêng của readout.
5. Có chế độ legacy để tái tạo loader train augmentation cũ khi cần đối chiếu.

Output đề xuất:

- `metrics_ncm_online.json`
- `acc_matrix_ncm_online.csv`
- `metrics_ncm_posthoc.json`
- `acc_matrix_ncm_posthoc.csv`
- `ncm_diagnostics.json`

Không tiếp tục dùng một file mơ hồ `metrics_ncm.json` cho các run mới.

### Giai đoạn 6 - Checkpoint và inference round-trip

Mục tiêu: model đã train có thể dùng lại mà không cần dataset train cũ.

Task:

1. Thêm checkpoint schema version 1.
2. Lưu model weights, Titans state, online NCM state, seen classes và config.
3. Thêm hàm load trên CPU, MPS, CUDA với `map_location`.
4. Tạo smoke inference từ checkpoint chỉ cần test image.
5. Xác nhận không cần truy cập train loader sau load.

Test bắt buộc:

- prediction trước và sau load giống nhau;
- prototype count giống nhau;
- Titans state norm giống nhau trong tolerance;
- checkpoint thiếu key quan trọng báo lỗi rõ;
- checkpoint version không hỗ trợ báo lỗi rõ.

### Giai đoạn 7 - Metrics và report công bằng

Mục tiêu: report nói đúng khả năng của từng readout.

Mỗi run phải ghi:

- Linear accuracy/forgetting/BWT;
- Online NCM accuracy/forgetting/BWT;
- Post-hoc NCM accuracy/forgetting/BWT;
- `oracle_online_gap`;
- feature protocol;
- prototype transform;
- revisit old train;
- prototype memory floats;
- full runtime và readout runtime;
- per-task prototype counts;
- finite flags;
- batch size;
- seed, config, commit, device.

Lưu ý diễn giải:

- Online forgetting phản ánh cả prototype staleness và representation drift.
- Post-hoc forgetting phản ánh representation ở từng thời điểm sau khi đã refit prototype.
- `posthoc - online` là chỉ báo drift/staleness, không phải lợi ích miễn phí.
- Linear vs post-hoc cho biết Linear head có phải nút thắt không.
- Linear vs online mới là so sánh deployable gần công bằng hơn.

---

## 7. Ma trận thực nghiệm

### 7.1. Bước A - Unit và synthetic smoke

Chạy:

```bash
pytest -q
python3 scripts/run_g1.py --config configs/g2_titans_smoke.yaml ...
```

Readout:

- Linear
- Online NCM
- Post-hoc NCM

Protocol:

- `stream_batch_legacy`
- `independent_image`

Mục tiêu:

- không crash;
- không NaN/Inf;
- checkpoint load được;
- online không revisit old data;
- output được đặt tên đúng.

### 7.2. Bước B - Reproduce số cũ

Dùng đúng cấu hình Titan mới + M3 improved LR `5e-3`, seed 0:

- Post-hoc legacy transform/protocol.
- Kỳ vọng gần NCM accuracy cũ của seed 0: `0.7362`.

Mục tiêu:

- xác nhận refactor không vô tình thay đổi lịch sử;
- tolerance ban đầu `+/- 0.005`, sau khi cố định RNG nên chặt hơn nếu khả thi.

### 7.3. Bước C - Protocol audit một seed

Cùng một training run seed 0, so:

1. Linear.
2. Post-hoc legacy.
3. Post-hoc deterministic + stream-batch.
4. Post-hoc deterministic + independent-image.
5. Online deterministic + independent-image.
6. Frozen NCM G1 làm mốc.

Quan sát:

- accuracy từng task;
- forgetting;
- batch/order sensitivity;
- online-oracle gap;
- runtime;
- prototype norm/count;
- state norm.

Không chọn phương án chỉ dựa trên final accuracy. Phải qua toàn bộ correctness gates.

### 7.4. Bước D - Ba seed chính

Sau khi chốt protocol:

- Dataset: RESISC45.
- Titan: cấu hình mới trên nhánh `Titan_M3`.
- M3: improved/delta-approx, LR `5e-3`.
- Seed: `0, 1, 2`.
- Cùng một model run sinh Linear, Online NCM và Post-hoc NCM.

Báo cáo mean, std và từng seed. Không chỉ báo mean.

### 7.5. Bước E - Exemplar NCM, chỉ làm sau core

Nếu online NCM kém post-hoc nhiều do feature drift, thử một nhánh có replay feature/exemplar bị chặn:

- `1`, `5`, `20` exemplar mỗi class;
- random selection trước, herding là ablation sau;
- tính đúng memory của exemplar;
- không gọi đây là “không replay”.

Mục đích:

- tìm trade-off accuracy/bộ nhớ;
- kiểm tra cần bao nhiêu dữ liệu cũ để bù prototype staleness.

Không thực hiện bước này trước khi online NCM thuần được đo đúng.

---

## 8. Tiêu chí nghiệm thu

### 8.1. Correctness

- Online NCM không đọc loader của task cũ.
- Prototype không dùng val/test.
- Count mỗi class đúng số mẫu đã hấp thụ.
- Unseen class không thể được dự đoán.
- NaN/Inf làm run fail rõ ràng.
- Evaluation không thay đổi Titans state thật.
- Independent-image protocol bất biến theo permutation và batch size trong tolerance.

### 8.2. Reproducibility

- Cùng checkpoint, cùng input cho cùng prediction.
- Dựng prototype deterministic cho cùng model/data/config.
- Output chứa config resolved, git commit, seed, device và protocol.
- Reproduce được post-hoc legacy seed 0 trong tolerance.

### 8.3. Continual-learning fairness

- Từng readout ghi rõ có revisit old train hay không.
- Memory cost của prototype/exemplar được tính.
- Kết luận chính không dùng post-hoc oracle như online head.
- Ba seed cho cấu hình cuối.

### 8.4. Engineering quality

- Toàn bộ test cũ pass.
- Test mới bao phủ core, protocol, integration và checkpoint.
- Không nhân đôi công thức NCM ở nhiều file.
- API/config có validation và lỗi dễ hiểu.
- Không làm thay đổi mặc định của các experiment cũ nếu không bật config mới.

---

## 9. Thứ tự ưu tiên

### P0 - Phải làm trước

1. Đặt tên đúng cho post-hoc NCM.
2. Prototype loader deterministic.
3. Finite/count/unseen-class guards.
4. Đo và xử lý feature protocol phụ thuộc batch.
5. Online NCM chỉ cập nhật current task.
6. Test chứng minh online không revisit task cũ.

### P1 - Cần có trước kết luận cuối

1. Checkpoint đầy đủ.
2. Ba readout trên cùng training run.
3. Metrics chi phí và oracle-online gap.
4. Ba seed RESISC45.
5. Report cập nhật thuật ngữ.

### P2 - Tối ưu sau khi core đúng

1. Vector hóa independent-image feature extraction.
2. Exemplar budget ablation.
3. Herding hoặc prototype correction.
4. Temperature calibration bằng validation set.
5. Multi-prototype cho class đa cụm.

Không nên làm P2 để “cứu accuracy” trước khi P0/P1 hoàn thành.

---

## 10. Rủi ro và cách xử lý

### Rủi ro 1: Online NCM thấp hơn nhiều so với 0.7403

Đây không nhất thiết là bug. Nguyên nhân hợp lý là feature space Titans/M3 trôi làm prototype cũ stale.

Cách xử lý:

- báo cả online và post-hoc;
- đo oracle-online gap theo task;
- đo cosine drift của anchor/prototype;
- sau đó mới thử exemplar hoặc prototype correction.

### Rủi ro 2: Independent-image làm accuracy giảm

Accuracy giảm có thể vì kết quả cũ đang hưởng ngữ cảnh từ các ảnh khác trong batch.

Cách xử lý:

- không quay lại protocol order-sensitive chỉ vì số đẹp hơn;
- báo cả legacy và canonical;
- kiểm tra semantics nào phù hợp mục tiêu deployment.

### Rủi ro 3: Independent-image quá chậm

Cách xử lý:

- tạo bản đúng chạy từng ảnh làm reference;
- benchmark;
- vector hóa `(B, 1, D)` với state tách biệt nếu thư viện hỗ trợ;
- test output vectorized khớp reference.

### Rủi ro 4: Refactor làm mất khả năng so với kết quả cũ

Cách xử lý:

- giữ mode legacy;
- reproduce seed 0 trước;
- output mới dùng tên file mới, không ghi đè artifact cũ.

### Rủi ro 5: Một prototype/class chưa đủ

Class ảnh vệ tinh có thể đa dạng góc nhìn, mùa, độ phân giải.

Cách xử lý:

- core vẫn dùng một prototype để có baseline rõ;
- multi-prototype chỉ là ablation P2;
- phải tính thêm memory và chọn số cluster chỉ bằng train/val.

---

## 11. Dự kiến file code bị tác động

Core:

- `src/uavcl/models/ncm.py`
- `src/uavcl/engine.py`
- `src/uavcl/data/loaders.py`
- `src/uavcl/models/titans_head.py`
- `src/uavcl/models/seq_adapter.py`
- `scripts/run_g1.py`

Config/campaign:

- một config NCM smoke mới;
- một config RESISC45 NCM audit mới;
- có thể mở rộng `scripts/run_titan_m3_study.py` sau khi core ổn định.

Tests:

- mở rộng `tests/test_ncm.py`;
- thêm `tests/test_ncm_readouts.py`;
- thêm `tests/test_ncm_checkpoint.py`;
- mở rộng `tests/test_g2_titans.py`;
- mở rộng `tests/test_run_logging.py`.

Docs/report:

- cập nhật README hoặc tài liệu protocol;
- tạo report kết quả riêng sau campaign;
- không sửa số trong report cũ, chỉ thêm chú thích phân loại post-hoc.

---

## 12. Ước lượng công việc

Ước lượng thực tế, không tính thời gian chờ review:

- Giai đoạn 0-2: 0.5-1 ngày.
- Giai đoạn 3: 1-2 ngày vì cần xác minh semantics Titans và benchmark.
- Giai đoạn 4-6: 1-2 ngày.
- Giai đoạn 7 + smoke + sửa lỗi: 0.5-1 ngày.
- RESISC45 3 seed trên VM GPU: phụ thuộc GPU và feature protocol; model campaign cũ khoảng 25 phút/run,
  nhưng independent-image có thể tăng thời gian đáng kể. Cần benchmark seed-0 trước khi báo ETA chính xác.

Tổng coding, test và audit hợp lý: khoảng 3-6 ngày làm việc. Training cuối có thể chạy trong vài giờ
trên VM nếu vector hóa tốt, hoặc lâu hơn nếu phải xử lý từng ảnh độc lập.

---

## 13. Definition of Done

Task NCM Head chỉ được coi là hoàn thành khi:

1. Có `ncm_online_current_task` dùng được mà không cần ảnh train cũ.
2. Có `ncm_posthoc_full_seen_train` được ghi nhãn oracle rõ ràng.
3. Có deterministic prototype loader.
4. Có feature protocol đã test về batch/order.
5. Có guard NaN/Inf/count.
6. Có checkpoint round-trip.
7. Toàn bộ test cũ và mới pass.
8. Reproduce được kết quả post-hoc cũ trong tolerance.
9. Có kết quả RESISC45 ba seed cho Linear/Online/Post-hoc trên cùng cấu hình.
10. Report cuối nêu rõ accuracy, forgetting, memory, runtime, revisit policy và giới hạn của kết luận.

---

## 14. Kết luận định hướng

Không nên bỏ NCM hiện tại vì nó đã cung cấp một kết quả chẩn đoán có giá trị: feature Titans mới + M3
improved đủ tốt để một cosine prototype classifier đạt khoảng 74% trên RESISC45. Tuy nhiên không nên
đưa 74% đó thành kết quả của một “online NCM head” vì code hiện tại đọc lại toàn bộ train đã thấy.

Hướng đúng là giữ cả hai:

- post-hoc NCM làm oracle đo chất lượng feature;
- online NCM bounded làm head chính có thể triển khai.

Khoảng cách giữa hai bản mới là kết quả khoa học quan trọng. Nếu khoảng cách nhỏ, team có một NCM head
đơn giản, rẻ và thực tế. Nếu khoảng cách lớn, project đã xác định đúng vấn đề tiếp theo: feature drift
của Titans/M3 và prototype staleness, thay vì tiếp tục tối ưu một con số hậu kiểm chưa phản ánh giao
thức continual learning thật.
