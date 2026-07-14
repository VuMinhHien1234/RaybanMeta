# PLAN G2 — Bộ nhớ Titans, chuỗi XUYÊN TASK (4 tuần)

**Quyết định đã chốt:** chuỗi = xuyên task (memory KHÔNG reset qua task); backbone FREEZE.
**Mục tiêu:** model "frozen ViT + Titans memory + head" chạy end-to-end trên stream UAV,
memory mang thông tin từ task cũ sang task mới; có bảng số so với 5 baseline G1.

## 0. Điều kiện vào (tuần 0 phải xong)
`docs/KET_LUAN_G1.md` đã có, trả lời: Forgetting của finetune (≈ bao nhiêu %), Avg Acc của
NCM vs finetune. Nếu NCM đã ~trần accuracy → luận điểm G2 nhắm vào: *Titans thích nghi
feature theo ngữ cảnh stream, thứ NCM (tĩnh) không làm được* — và FWT là metric khoe được.

## 1. Kiến trúc (chốt interface TUẦN 1 để N2/N3 làm song song)
```
ảnh (B,3,H,W) ─ frozen ViT ─▶ token features (B, P, D) hoặc CLS (B, D)
                                   │  adapter (N3): sắp ảnh trong batch theo THỨ TỰ stream
                                   ▼
                    chuỗi (1, B·k, D) ──▶ TitansMemory (N2) ──▶ (1, B·k, D), state
                                   ▼
                              head Linear (C=45) ──▶ logits ──▶ engine G1 (mask/eval như cũ)
```
- **TitansMemory** (`src/uavcl/models/memory.py`, N2): bọc `titans_pytorch.NeuralMemory`.
  API bắt buộc: `forward(seq, state=None) -> (seq_out, state)`; `state` pickle/lưu được.
- **Adapter** (`src/uavcl/models/seq_adapter.py`, N3): (B,P,D) → chuỗi; 2 chế độ:
  `token_seq` (P token của 1 ảnh nối nhau) và `image_seq` (mỗi ảnh 1 bước = mean-pool token).
- **TitansClassifier** (`src/uavcl/models/titans_head.py`): ghép cả 3, giữ đúng giao thức
  logits (B, C) để cắm vào engine — KHÔNG sửa engine/metrics.
- Config mới (thêm vào g1_*.yaml khi chạy): `memory: {enabled, dim, chunk_size, reset: image|task|never, detach_every: N}`.

## 2. Bậc thang 3 bước (một tuần một bậc — C là ĐÍCH, A/B là để sống sót)
**A. `reset: image` (tuần 1–2, sanity):** memory reset mỗi ảnh (chuỗi = token trong ảnh).
   Mục đích duy nhất: shape đúng, loss giảm, không NaN. Kỳ vọng ≈ NCM/linear-probe.
**B. `reset: task` (tuần 2–3):** state truyền qua các batch TRONG 1 task, reset khi sang
   task mới. Loader phải cho ảnh theo thứ tự deterministic (`shuffle=True` trong task vẫn
   được — state không phụ thuộc thứ tự lý tưởng). Kỹ thuật: **truncated BPTT** — `state.detach()`
   sau mỗi `detach_every` batch để không backprop xuyên lịch sử → tránh nổ RAM.
**C. `reset: never` (tuần 3–4, ĐÍCH):** state sống XUYÊN task — trước khi eval task j,
   snapshot state hiện tại và dùng nó (eval không được sửa state: clone). Đây là chỗ
   "học liên tục bằng bộ nhớ" thật sự: task mới đến, memory tự cập nhật online.

## 3. Việc theo người
- **N2:** tuần 1: toy NeuralMemory chạy lại (smoke_titans.py có sẵn) + viết TitansMemory
  với quản lý state (init/detach/clone/save); tuần 2–3: chế độ B/C + xử lý ổn định số
  (norm state, clip); tuần 4: hỗ trợ N1 đọc kết quả, viết note cơ chế.
- **N3:** tuần 1: adapter 2 chế độ + TitansClassifier; tuần 2: nối vào run_g1 (thêm nhánh
  model như đã làm với NCM); tuần 3–4: profiling tốc độ, thử token_seq vs image_seq.
- **N1:** tuần 1: thêm method "titans" vào configs + chạy A; tuần 2–4: chạy B, C trên
  EuroSAT trước rồi RESISC45; **bật `train.eval_future: true`** (FWT giờ có ý nghĩa — memory
  có thể giúp task chưa học); vẽ forgetting-theo-task; bảng Titans vs 5 baseline.

## 4. Thí nghiệm phải có (định nghĩa "xong")
| Run | Dataset | So với |
|---|---|---|
| A/B/C lần lượt | EuroSAT | nhau + finetune/ncm (cùng frozen) |
| C (+ B nếu C thắng) | RESISC45 | đủ 5 baseline G1 |
| Ablation: reset image/task/never | EuroSAT | nhau — chứng minh "xuyên task" đáng giá |

Train chỉ memory + head (backbone frozen) → nhanh; ngân sách ~15–25 run, GPU NVIDIA thoải mái.

## 5. Rủi ro & đường lùi
- State NaN/explode ở C → giảm η nội bộ, thêm LayerNorm sau memory, detach dày hơn. Còn xấu → nộp kết quả B (vẫn qua cổng, C ghi "phân tích thất bại" — trung thực).
- Memory không giúp gì (≈ NCM) → kiểm tra: memory có thật sự được update lúc eval-stream không; thử image_seq thay token_seq; tăng dim/chunk_size. Kết quả âm + giải thích vẫn qua cổng (plan gốc cho phép).
- Loader thứ tự không tái lập → seed đã cố định trong harness, kiểm bằng chạy 2 lần so md5 acc_matrix.

## 6. Cổng chuyển sang G3
- [ ] Chế độ C chạy end-to-end không NaN trên cả 2 dataset.
- [ ] Bảng Titans (A/B/C) vs 5 baseline + 1 đoạn kết luận "memory xuyên task giúp/không giúp, ở đâu" (`docs/KET_LUAN_G2.md`).
- [ ] FWT của C so với baseline (kỳ vọng: nhỉnh hơn vì memory mang ngữ cảnh sang task mới).
