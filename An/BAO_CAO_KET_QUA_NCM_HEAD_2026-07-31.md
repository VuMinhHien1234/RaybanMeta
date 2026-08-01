# Báo cáo hoàn thiện và đánh giá NCM Head

Ngày hoàn tất: 2026-07-31  
Nhánh: `NCM_Head`  
Dataset chính: RESISC45, 9 task, 5 class/task  
Thiết bị training: GCP Spot VM, NVIDIA Tesla T4, PyTorch CUDA  
Commit được ghi trong kết quả: `9b7e496a5630901483c933d0f3023efda452b324`

## 1. Mục tiêu

Mục tiêu của đợt này là hoàn thiện NCM Head về cả thuật toán, giao thức
continual learning, khả năng tái lập và checkpoint; sau đó trả lời công bằng câu
hỏi:

> Titans mới + M3 cải thiện + NCM Head có tốt hơn NCM dùng feature extractor
> frozen hay không?

Phép so chính phải là `Titans NCM Online` với `Frozen NCM`. Kết quả NCM
post-hoc chỉ là oracle phân tích vì được phép đọc lại toàn bộ train data cũ.

## 2. Những phần đã triển khai

### 2.1. Prototype Head dùng chung

- Tách `PrototypeHead` thành thành phần độc lập, tái sử dụng được.
- Theo dõi tổng feature, số mẫu và seen-mask cho từng class.
- Chuẩn hóa prototype trước khi tính cosine similarity.
- Mask class chưa từng thấy để chúng không thể được dự đoán nhầm.
- Kiểm tra shape, label range và NaN/Inf tại biên cập nhật.
- Giữ `NCMClassifier` tương thích với code cũ.

### 2.2. Ba readout được tách rõ

- `ncm_frozen_baseline`: backbone frozen, không Titans, không đọc lại dữ liệu cũ.
- `ncm_online_current_task`: sau mỗi task chỉ dùng một deterministic pass của
  train data task hiện tại để cập nhật prototype.
- `ncm_posthoc_full_seen_train`: dựng lại prototype bằng toàn bộ train data đã
  thấy; chỉ dùng làm oracle, không coi là phương pháp continual deployable.

Mỗi output ghi rõ `readout`, `revisit_old_train`, feature protocol và prototype
loader để không thể nhầm hai giao thức khi tổng hợp.

### 2.3. Feature và dữ liệu prototype

- Có loader `prototype` riêng với eval transform, không augmentation ngẫu nhiên
  và không shuffle.
- Thêm protocol `independent_image`, trong đó feature của một ảnh không phụ
  thuộc ảnh đứng trước/sau hay ranh giới batch.
- Giữ `stream_batch_legacy` để tái lập kết quả cũ khi cần.
- Online và post-hoc dùng cùng cách mã hóa feature; khác nhau duy nhất ở quyền
  đọc lại dữ liệu cũ.

### 2.4. Checkpoint và khả năng resume

- Checkpoint inference lưu model, Titans state, optimizer và trạng thái NCM.
- Có checkpoint tiến độ ở ranh giới task, gồm RNG state và ma trận đánh giá.
- Campaign có thể tiếp tục sau khi Spot VM bị preempt mà không chạy lại từ đầu.
- Đã sửa việc CUDA RNG state bị map nhầm lên GPU khi load checkpoint.
- Checkpoint tạm được xóa sau khi một run hoàn thành thành công.

### 2.5. Metrics và artifact

- Xuất riêng metrics và accuracy matrix cho Linear, NCM Online và NCM Post-hoc.
- Ghi prototype count/norm, state finite, số sample encode, thời gian, quyền
  revisit và cosine alignment giữa online/post-hoc.
- Có script campaign chạy smoke, seed 0, ba seed và tạo report tổng hợp.
- Artifact cuối có config, train log, metric, matrix, checkpoint và memory state.

## 3. Kiểm thử và hậu kiểm

- Unit/integration test local: **111 passed, 1 warning**.
- Warning nằm trong test M3 khi ép tensor có gradient sang `float`; không làm
  test fail và không liên quan đường NCM.
- Ba seed Titans hợp lệ: **3/3**.
- Ba seed Frozen NCM hợp lệ: **3/3**.
- JSON được parse và quét đệ quy sau cả run legacy: **26/26 hợp lệ**, không
  NaN/Inf.
- `failure.json`: **0**.
- `progress_checkpoint.pt` còn sót sau campaign: **0**.
- State Titans finite ở mọi task của cả ba seed.
- VM sau khi tải kết quả: **TERMINATED**.
- Persistent disk: `autoDelete=false`.

## 4. Kết quả RESISC45

| Seed | Frozen NCM Acc/Fgt | Titans Linear Acc/Fgt | Titans NCM Online Acc/Fgt | Titans NCM Post-hoc Acc/Fgt | Online - Frozen | Post-hoc - Online |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7114 / 0.0846 | 0.6079 / -0.0307 | 0.6143 / 0.2725 | 0.7454 / 0.0671 | -0.0971 | +0.1311 |
| 1 | 0.7114 / 0.0854 | 0.6006 / -0.0068 | 0.6073 / 0.2825 | 0.7387 / 0.0679 | -0.1041 | +0.1314 |
| 2 | 0.7114 / 0.0921 | 0.6222 / -0.0379 | 0.6356 / 0.2464 | 0.7537 / 0.0618 | -0.0759 | +0.1181 |

Tổng hợp mean +/- sample standard deviation:

| Readout | Accuracy | Forgetting |
|---|---:|---:|
| Frozen NCM | 0.7114 +/- 0.0000 | 0.0874 +/- 0.0041 |
| Titans Linear | 0.6103 +/- 0.0110 | -0.0251 +/- 0.0163 |
| Titans NCM Online | 0.6190 +/- 0.0147 | 0.2671 +/- 0.0186 |
| Titans NCM Post-hoc | 0.7459 +/- 0.0075 | 0.0656 +/- 0.0033 |

Các chẩn đoán chính:

- Online thấp hơn Frozen NCM trung bình **0.0924**, tức khoảng **9.24 điểm phần
  trăm**.
- Post-hoc cao hơn Online trung bình **0.1269**, tức khoảng **12.69 điểm phần
  trăm**.
- Post-hoc cao hơn Frozen NCM khoảng **3.45 điểm phần trăm**, nhưng không phải
  phép so deployable vì post-hoc đọc lại toàn bộ train data cũ.
- Cosine giữa prototype online và post-hoc cuối stream là
  **0.9158 +/- 0.0063**.
- Online PrototypeHead dùng **17,325 float**, tương đương khoảng **67.7 KiB**
  nếu lưu FP32; không lưu exemplar hoặc ảnh train cũ.
- State norm lớn nhất theo seed là 96.22, 120.45 và 92.34; tất cả vẫn finite,
  không có hiện tượng nổ NaN/Inf.

### 4.1. Nghiệm thu tương thích legacy

Đã chạy lại seed 0 bằng đúng đường tương thích cũ:

- M3 improved/delta-approx, LR `5e-3`.
- `train.eval_ncm_head=true`.
- Prototype dùng train loader/transform legacy.
- Feature protocol `stream_batch_legacy`.

Kết quả:

| Chỉ số | Mốc lịch sử | Run sau refactor | Sai lệch tuyệt đối |
|---|---:|---:|---:|
| NCM post-hoc accuracy | 0.7362 | 0.7361904762 | 0.0000095238 |
| NCM post-hoc forgetting | 0.0696 | 0.0696428571 | 0.0000428571 |

Sai lệch accuracy nhỏ hơn rất nhiều tolerance `0.005`. Run này tái lập thực tế
mốc cũ, xác nhận refactor không làm hỏng đường legacy. Số mới 0.7459 của protocol
chính cao hơn vì đã chuyển sang deterministic prototype loader và
`independent_image`; không phải do âm thầm thay đổi kết quả legacy.

## 5. Phân tích

### 5.1. NCM Head hiện đã đúng giao thức hơn

Các lỗi thiết kế chính trong plan đã được xử lý: prototype không còn lấy từ
augmentation ngẫu nhiên, feature evaluation không phụ thuộc batch, class unseen
được mask, online không đọc lại old train, post-hoc được ghi nhãn oracle và
checkpoint chứa đủ trạng thái NCM.

Vì vậy kết quả thấp của Online hiện không còn nên được giải thích đơn giản là
"NCM Head code chưa hoàn chỉnh". Đây là tín hiệu thực nghiệm về sự không tương
thích giữa feature space thay đổi liên tục và prototype cũ.

### 5.2. Representation drift là nút thắt chính

Prototype online của class cũ được tạo tại thời điểm class đó xuất hiện. Titans
và model tiếp tục thay đổi ở các task sau, nhưng prototype cũ không được cập nhật
vì giao thức online không cho đọc lại dữ liệu cũ. Feature hiện tại và prototype
cũ vì vậy dần lệch nhau.

Ba bằng chứng cùng chỉ về drift:

- Online forgetting cao: khoảng 0.267.
- Post-hoc dựng lại prototype bằng feature space hiện tại giảm forgetting xuống
  khoảng 0.066.
- Khoảng cách post-hoc - online ổn định ở cả ba seed, từ 0.118 đến 0.131.

Cosine prototype cuối vẫn khá cao, khoảng 0.916, nhưng NCM quyết định bằng biên
gần nhất giữa nhiều class. Một thay đổi góc vừa phải có thể đổi class thắng khi
các prototype nằm gần nhau; vì thế cosine cao không mâu thuẫn với accuracy giảm
12.69 điểm phần trăm.

### 5.3. Không được dùng Post-hoc để tuyên bố thắng

Accuracy 0.7459 của Post-hoc là kết quả tốt nhất, nhưng nó dùng lại 25,200 ảnh
train ở task cuối và trước đó cũng dựng lại trên toàn bộ dữ liệu đã thấy sau mỗi
task. Đây là oracle giúp chứng minh feature hiện tại còn phân loại tốt nếu
prototype được refresh; nó không đáp ứng ràng buộc không lưu/replay dữ liệu cũ.

Kết luận chính phải dùng Online 0.6190. Theo phép so này, cấu hình Titans mới +
M3 cải thiện + NCM Online **chưa tốt hơn** Frozen NCM 0.7114.

### 5.4. Chi phí đánh giá

Ba run Titans ghi nhận tổng runtime khoảng 5.57 giờ, chưa tính toàn bộ thời gian
chờ do Spot preemption và tải artifact. Post-hoc có chi phí tăng dần theo số task;
ở cuối stream phải encode toàn bộ train set đã thấy. Đây là nguyên nhân campaign
chạy lâu hơn dự kiến, không phải deadlock.

Spot VM bị preempt nhiều lần. Cơ chế checkpoint ranh giới task đã giữ được tiến
độ, nhưng task đang chạy dở vẫn phải chạy lại. VM dùng termination action `STOP`
và disk không auto-delete nên không mất kết quả.

## 6. Kết luận

1. Phần core của plan NCM Head đã được triển khai và kiểm thử đầy đủ.
2. Campaign RESISC45 đã hoàn thành 3 seed hợp lệ, không NaN/Inf; run legacy seed
   0 cũng tái lập mốc lịch sử trong tolerance.
3. NCM Post-hoc cho thấy feature Titans có tiềm năng: 0.7459 accuracy.
4. NCM Online thực tế chỉ đạt 0.6190 và thua Frozen NCM 9.24 điểm phần trăm.
5. Nút thắt hiện tại là prototype cũ bị stale do representation drift, không còn
   chủ yếu là lỗi cài đặt NCM.
6. Chưa nên chọn Titans NCM Online hiện tại làm head chính nếu tiêu chí là
   accuracy và không replay dữ liệu cũ. Frozen NCM hiện đáng tin cậy hơn.

## 7. Hướng cải thiện tiếp theo

Ưu tiên thực tế theo thứ tự:

1. **Exemplar replay có budget cố định:** lưu một số ít ảnh hoặc feature mỗi
   class và refresh prototype sau task. Cần báo cáo accuracy theo các budget như
   1, 5, 10, 20 exemplar/class.
2. **Prototype transport/alignment:** học phép biến đổi từ feature space cũ sang
   feature space mới bằng dữ liệu task hiện tại, rồi dịch prototype cũ mà không
   cần giữ toàn bộ old train.
3. **Ổn định representation:** thêm distillation hoặc regularization giữa
   feature trước/sau task để giảm drift ngay từ backbone/Titans.
4. **EMA prototype không đủ nếu đứng một mình:** EMA chỉ hữu ích khi class còn
   xuất hiện hoặc có replay; nó không tự sửa prototype của class cũ không còn
   dữ liệu.
5. **Giảm chi phí oracle:** chỉ chạy post-hoc ở cuối stream hoặc ở một số task
   mốc khi tuning; không cần dựng lại toàn bộ sau mọi task trong campaign lớn.
6. Sau khi chọn một cơ chế bounded ở trên, chạy ablation 1 seed trước, rồi mới
   chạy 3 seed và so với Frozen NCM bằng cùng split/protocol.

## 8. Vị trí kết quả

- Toàn bộ artifact:
  `RaybanMeta/An/NCM_Head_results_2026-07-31/artifacts_ncm_head_study/`
- Bảng tổng hợp gốc:
  `RaybanMeta/An/NCM_Head_results_2026-07-31/artifacts_ncm_head_study/summary.md`
- Bảng từng run:
  `RaybanMeta/An/NCM_Head_results_2026-07-31/artifacts_ncm_head_study/runs.csv`
- Log campaign:
  `RaybanMeta/An/NCM_Head_results_2026-07-31/ncm_campaign_master.log`
- Artifact nghiệm thu legacy:
  `RaybanMeta/An/NCM_Head_results_2026-07-31/artifacts_ncm_head_legacy_repro/`
- Log nghiệm thu legacy:
  `RaybanMeta/An/NCM_Head_results_2026-07-31/ncm_legacy_repro.log`

Tổng dung lượng kết quả local khoảng 884 MB, gồm đầy đủ checkpoint của ba seed
Titans và memory state của run legacy.
