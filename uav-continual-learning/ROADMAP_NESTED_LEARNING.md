# Roadmap: đưa Titans về đúng "Deep Self-Referential Titans" (NL.pdf §8.1)

Mục tiêu: nâng bản Titans hiện tại (neural memory cơ bản, Eq 76) lên bản self-modifying
của paper, theo cái thang Eq 76 → 79 → 82. Mỗi bậc là 1 task độc lập, có tiêu chí xong rõ.

> Lưu ý xuyên suốt: "giống paper hơn" là mục tiêu **độ trung thành nghiên cứu**, chưa chắc
> tăng accuracy trên EuroSAT 5-task (feature đóng băng -> NCM đã 74.9%). Cơ chế self-modifying
> tỏa sáng ở **chuỗi dài / nhiều task / streaming**. Vì vậy có TASK 0 (chọn regime đo) đi kèm.

---

## BẬC 1 — Deep memory ✅ ĐÃ LÀM (2026-07-21)
- **Đã sửa:** `src/uavcl/models/titans_head.py` đọc `memory.depth`, `memory.heads` từ config
  và truyền xuống `NeuralMemory` (`default_model_kwargs.depth`, `heads`+`dim_head`).
  Mặc định depth=2, heads=1 = giữ nguyên hành vi cũ.
- **Config mới:** `configs/g2_titans_eurosat_deep.yaml` (depth=3, heads=2).
- **Chạy:** `.venv/bin/python scripts/run_g1.py --config configs/g2_titans_eurosat_deep.yaml`
  rồi so với P1 (depth2/heads1 = 39.4% / F 0.462).
- **Trước khi tin số:** chạy `pytest tests/test_g2_titans.py -q` (xác nhận sửa không phá G2).

---

## TASK 0 — Chọn regime đo cho đúng (làm song song, độ khó THẤP)
- **Vì sao:** EuroSAT 5-task không phân biệt được self-modifying vs vanilla. Cần bài khó hơn.
- **Việc:** thêm config chuỗi dài: `data.num_tasks` cao hơn (vd RESISC45 45 class / 9-15 task),
  hoặc lặp lại nhiều lần class để tạo stream dài. Thêm `configs/g2_titans_resisc45_long.yaml`.
- **Tiêu chí xong:** có 1 stream ≥ 9 task chạy được; đo được đường cong forgetting theo thời gian.

## TASK 2 — η/α data-dependent (Eq 76) — độ khó THẤP (chủ yếu xác minh)
- **Mục tiêu:** learning-rate η_t và forget-gate α_t phụ thuộc dữ liệu (x_t·W_η, x_t·W_α),
  không phải hằng số.
- **Việc:**
  1. Đọc `titans_pytorch/neural_memory.py`: xác nhận `adaptive_lr` (per-token lr từ input) và
     `decay_factor`/forget gate đang BẬT theo mặc định (nhiều khả năng đã có).
  2. Expose cờ `memory.per_parameter_lr_modulation` qua `_stability_kwargs` (giống bậc 1).
  3. Log giá trị η_t, α_t trung bình/task để xác nhận chúng thay đổi theo dữ liệu.
- **File:** `models/titans_head.py` (thêm cờ), đọc `neural_memory.py` (xác minh).
- **Tiêu chí xong:** log cho thấy η_t/α_t biến thiên theo input; 1 run bật/tắt cho khác biệt đo được.
- **Ước lượng:** 0.5 ngày.

## TASK 3 — Self-referential projections (Eq 79) — độ khó CAO (fork thư viện)
- **Mục tiêu:** W_k, W_v, W_q (hiện là `nn.Linear` cố định) trở thành **memory con tự cập nhật
  in-context**, init meta-học. Đây là cốt lõi của "self-referential".
- **Việc:**
  1. Tạo `src/uavcl/models/self_ref_memory.py`: lớp `SelfRefNeuralMemory` fork/kế thừa
     `NeuralMemory`, thay `to_keys/to_values/to_queries` bằng module memory nhỏ cập nhật theo Eq 79.
  2. Dùng "simple version" của paper (chia sẻ value cho các thành phần) để đỡ chi phí.
  3. `models/memory.py`: thêm cờ `memory.self_referential: true/false` để chọn vanilla vs self-ref.
- **File mới:** `models/self_ref_memory.py`; sửa `models/memory.py`, `models/titans_head.py`.
- **Test:** `tests/test_self_ref_memory.py` — projections thay đổi theo chuỗi (khác vanilla cố định),
  forward giữ shape (1,L,D), không NaN trên quick config.
- **Tiêu chí xong:** chạy được trên `g4_hope_eurosat_quick`; projections chứng minh động (không cố định).
- **Ước lượng:** 2-4 ngày (cần đọc kỹ `store_memories`/`retrieve_memories` của titans_pytorch).
- **Phụ thuộc:** là tiền đề cho TASK 4.

## TASK 4 — Self-modifying value generation (§8.1 cuối) — độ khó RẤT CAO
- **Mục tiêu:** model **tự sinh value v̂_t** theo trạng thái memory M_{t-1} + ngữ cảnh (không phải
  x_t·W_v cố định) — self-modifying kiểu Schmidhuber. Chính là phần `optim/m3.py` docstring đã ghi
  "chưa làm, cần fork NeuralMemory".
- **Việc:** mở rộng `SelfRefNeuralMemory` (TASK 3): value sinh bởi hàm phụ thuộc M_{t-1},
  tạo vòng tự tham chiếu. Cần cơ chế ổn định số (tái dùng 4 cờ: spectral_norm, gated_transition).
- **File:** mở rộng `models/self_ref_memory.py`.
- **Test:** v_t phụ thuộc M_{t-1} (đổi state -> đổi value); norm(state) không nổ qua ≥5 task.
- **Tiêu chí xong:** đo trên stream dài (TASK 0); norm(state) ổn định; báo cáo so vanilla.
- **Ước lượng:** 4-7 ngày. **Rủi ro số học cao** — bắt buộc bám log norm(state).
- **Phụ thuộc:** TASK 3.

### ✅ ĐÃ CÀI (2026-07-23) — nhánh `task4-self-modifying-value`
- **Cách làm (bám triết lý Task 3: BỌC, không rebuild → version-agnostic với titans-pytorch 0.5.5):**
  value không còn tĩnh `v=to_values(x)` mà `v = ContextGate(x) + β·tanh(W_state·summary(M_{t-1}))`.
  `summary(M_{t-1})` = [mean, rms, mean_abs, max_abs] nén-log của trọng số memory, được BƠM vào
  value-projection ngay trước mỗi `store_memories` (bọc method, đọc tham số `weights` = M_{t-1}).
- **An toàn số:** `W_state` init = 0 → nhánh Task 4 = 0 lúc đầu → **output TRÙNG Task 3** (so 1-biến
  sạch, khởi đầu ổn định); `tanh` chặn nhánh state → bounded; summary được **detach** (không mở
  thêm đường gradient vào M_{t-1}, giữ nguyên per_sample_grad_fn của titans).
- **File:** `models/self_ref_memory.py` (+`SelfModifyingValueProjection`, `summarize_memory_state`,
  `make_self_modifying`, `_install_store_hook`, `build_self_modifying_neural_memory`);
  sửa `models/memory.py`, `models/titans_head.py`; config `configs/g2_titans_resisc45_selfmod.yaml`
  (cờ `memory.self_modifying: true`, bao trùm self_referential); test `tests/test_self_modifying_memory.py`.
### ✅ KẾT QUẢ (2026-07-23) — Task 4 XONG, tiêu chí đạt
- **v1 (summary gộp 4 số):** avg_acc 0.2474 / forget 0.7127 — hoà với selfref (nhánh gần như chưa kích hoạt).
- **v2 "tăng lực" (summary NỐI 4 thống kê/ma trận, ~24 chiều):** **avg_acc 0.6214 / forget 0.2420** — nhảy vọt.
- **Kiểm chứng seed0:** `|W_state|` tăng đều 0→26.4 (nhánh sống thật); β~1.0; norm(state) **phẳng ~54**
  (v1 phình 392→2785) → vòng self-modifying tự ổn định. 0.62 < NCM 0.69/replay 0.79 (không phi lý).
- **Bài học cốt lõi:** "chất lượng biểu diễn M" là trục thống trị (4→24 chiều = +0.37 acc).
- **Đang xác nhận:** seed 1 (robust), và hướng-1 (self-mod cho k/q, nhánh `memory_titan_task4_v2`).

---

## TRƯỚC TASK 5 — 3 đòn bẩy đóng khoảng cách 0.62 → mốc thắng (Acc ≥ 0.72, Forget ≤ 0.1)

> Mốc từ `docs/KET_LUAN_G1.md`: PHẢI vượt **NCM 0.6933**; mơ tới **replay 0.7937**; Forget ≤ 0.1;
> floats ≪ 135M (selfmod ~25M — đã đạt trục chi phí). Làm TỪNG cái, **so 1-biến**, **đa seed** rồi mới tin.

### ĐÒN A — NCM-head trên feature sau memory (nghi phạm forgetting = head Linear) ✅ ĐÃ CÀI (test rẻ)
- **Giả thuyết:** `head = nn.Linear` train-liên-tục tự quên; NCM (prototype class-mean) gần như không quên.
- **Test rẻ (không train lại):** sau mỗi task, dựng prototype từ FEATURE SAU MEMORY của train đã thấy,
  phân loại test bằng cosine → prototype. So Acc/Forget với head Linear và với NCM gốc 0.6933.
- **File:** `models/titans_head.py` (+`features()`); `engine.py` (`_memory_prototypes`, `_evaluate_ncm`,
  hook trong `run_continual` sau mỗi task); `scripts/run_g1.py` (in + lưu `metrics_ncm.json`/`acc_matrix_ncm.csv`).
  Bật bằng cờ `train.eval_ncm_head=true` (mặc định TẮT → run cũ bất biến).
- **Chạy:** `.venv/bin/python scripts/run_g1.py --config configs/g2_titans_resisc45_selfmod.yaml \`
  `--set train.eval_ncm_head=true --set log.dir=./artifacts_titans_resisc45_ncmhead`
- **Đọc:** NCM-head **> head Linear (0.62)** và tiến gần/qua **0.69** → xác nhận head là nút thắt →
  làm bản đầy đủ (readout NCM cố định thay Linear, hoặc cosine/mask head). Nếu KHÔNG hơn → head không phải
  thủ phạm, dồn sang đòn B.

### ĐÒN B — tín hiệu M giàu hơn nữa (trục đã chứng minh thống trị)
- **Ý:** thay/bổ sung summary 24-chiều bằng biểu diễn M mạnh hơn: (i) đọc **nội dung ký ức truy hồi thật**
  (retrieved memory), không chỉ thống kê trọng số; (ii) một **phép chiếu học-được** (low-rank) của trọng số M;
  (iii) biến nhánh self-mod từ 1 `Linear` → **MLP nhỏ** (Linear→SiLU→Linear, lớp cuối init 0 giữ trung tính).
- **File:** `models/self_ref_memory.py` (`summarize_memory_state` + `SelfModifyingProjection`).
- **An toàn:** giữ init trung tính + tanh-bounded + detach như hiện tại → xấu nhất = bằng bản đang có.

### ĐÒN C — gỡ khoá heads>1 (thêm capacity memory)
- **Vấn đề:** `heads>1` sinh overlapping-memory kẹt weight-decay của M3 (đã hạ về heads=1 ở `9bd1888`).
- **Việc:** cho M3 **bỏ weight-decay trên tham số memory đa đầu** (hoặc tách nhóm param), rồi bật heads=2;
  kèm thử `depth=4`. So 1-biến với bản heads=1 tốt nhất.
- **Ưu tiên:** sau A/B (A/B kỳ vọng cao hơn).

> **Thứ tự:** chờ seed1 + hướng-1 → **A** (rẻ, xác nhận nút thắt) → nếu ăn, làm A đầy đủ; song song **B** →
> rồi **C** → rồi mới **Task 5**. Chạm Acc ≥ 0.72 & Forget ≤ 0.1 = "thắng có ý nghĩa" → chốt, dừng tối ưu.

## TASK 5 — Meta-learned initial state (Eq 72, 79-82) — độ khó CAO (vòng ngoài)
- **Mục tiêu:** trạng thái khởi tạo memory M_0 được **meta-học qua các task** (thay vì zero/cố định).
  Paper nói init meta-học là thiết yếu cho fast-adaptation + ổn định + chống nhiễu.
- **Việc:**
  1. `titans_head.py`: biến init state của memory thành tham số học được (hoặc buffer cập nhật).
  2. `engine.py`: thêm hook meta-update — sau mỗi task, cập nhật M_0 (bản đơn giản: EMA của state
     cuối task; bản đầy đủ: gradient meta kiểu Reptile qua nhiều task).
- **File:** `engine.py`, `models/titans_head.py`.
- **Test/Tiêu chí xong:** M_0 ≠ 0 sau training; accuracy task mới ở epoch đầu cao hơn (fast-adapt).
- **Ước lượng:** 2-3 ngày.
- **Phụ thuộc:** độc lập, nhưng hưởng lợi nếu có TASK 3.

---

## Thứ tự đề xuất
1. **TASK 0 + TASK 2** trước (rẻ, tạo nền đo + xác minh η/α).
2. **TASK 3** (self-referential) — bước NL thật sự đầu tiên.
3. **TASK 4** (self-modifying) — đỉnh độ trung thành, dựng trên TASK 3.
4. **TASK 5** (meta-init) — có thể xen vào bất cứ lúc nào sau TASK 0.

## Kỷ luật chung
- Mỗi task: 1 nhánh git, 1 config riêng, 1 unit test, so 1-biến/lần với mốc trước.
- Luôn đọc log `norm(state)` (memory) và `‖Δw‖` (nếu chạy trong HOPE) trước khi tin accuracy.
- Regime đo: dùng stream dài của TASK 0, KHÔNG chỉ EuroSAT 5-task.
