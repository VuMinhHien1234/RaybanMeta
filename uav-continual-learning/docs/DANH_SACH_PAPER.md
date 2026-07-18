# Danh sách paper cần đọc (xếp theo mục đích, không phải đọc hết một lượt)

> Cách đọc mỗi paper (30–45 phút nắm 80%): abstract → hình vẽ → phần method.
> Bỏ qua thí nghiệm chi tiết và phụ lục ở lần đọc đầu. Tick khi xong + ghi 3 dòng
> "mình rút được gì" vào cuối file — 3 dòng đó chính là nguyên liệu chương Related Work.

## Nhóm A — BẮT BUỘC, phục vụ G2 (đọc trong lúc VM đang chạy)

- [ ] **A1. Titans: Learning to Memorize at Test Time** — Behrouz, Zhong, Mirrokni (Google), 2025. arXiv:2501.00663.
      *Vì sao:* paper NGUỒN của `NeuralMemory` đang chạy trong G2 — surprise metric, momentum
      của ký ức, forget gate, 3 cách gắn memory (MAC/MAG/MAL). Đọc xong sẽ hiểu từng dòng
      log `norm(state)`. **Quan trọng nhất danh sách.**

- [ ] **A2. Transformer Feed-Forward Layers Are Key-Value Memories** — Geva, Schuster, Berant, Levy. EMNLP 2021.
      *Vì sao:* chứng minh thực nghiệm "MLP = bộ nhớ key-value, kiến thức sống trong đó" —
      nền móng khiến CMS chọn khối MLP để chia tần số. Ngắn, dễ đọc, trả lời luôn câu hỏi
      "MLP là gì trong vai trò bộ nhớ".

## Nhóm B — NÊN ĐỌC, phục vụ G3/G4

- [ ] **B1. NL.pdf (đã có trong folder)** — đọc CHỌN LỌC: §4 (optimizer là bộ nhớ, Delta
      Momentum Eq. 48–49), §7 (CMS Eq. 70–71 + M3 Algorithm 1), §8 (HOPE). ~15 trang lõi.
- [ ] **B2. Muon optimizer** — Keller Jordan, 2024 (bài blog "Muon: An optimizer for hidden
      layers of neural networks" + repo github.com/KellerJordan/Muon).
      *Vì sao:* giải thích Newton–Schulz trong M3, và quy ước "tham số 1D dùng Adam" —
      thứ dự án đã học được bằng một bug thật (limit-cycle trên vector).
- [ ] **B3. Linear Transformers Are Secretly Fast Weight Programmers** — Schlag, Irie, Schmidhuber. ICML 2021.
      *Vì sao:* cầu nối lịch sử attention tuyến tính ↔ fast weights ↔ delta rule; đọc xong
      thấy Titans là mắt xích của dòng ý tưởng 30 năm chứ không phải từ trên trời rơi xuống.
- [ ] **B4. (Tuỳ chọn) Learning to (Learn at Test Time): RNNs with Expressive Hidden States**
      — Sun et al., 2024 ("TTT"). Anh em song song với Titans về test-time learning; đọc nếu
      muốn viết Related Work dày hơn.

## Nhóm C — PHỤC VỤ VIẾT BÁO CÁO (cite đối thủ + metric cho chuẩn)

- [ ] **C1. GEM: Gradient Episodic Memory for Continual Learning** — Lopez-Paz, Ranzato. NeurIPS 2017.
      *Vì sao:* nguồn gốc ma trận R và metric Avg Acc / BWT / FWT mà harness đang đo.
      Chỉ cần phần định nghĩa metric.
- [ ] **C2. Three scenarios for continual learning** — van de Ven, Tolias, 2019 (arXiv).
      *Vì sao:* ngắn; phân biệt task-/domain-/class-incremental — giúp phát biểu chính xác
      setting của đồ án khi bảo vệ (mình đang làm class-incremental).
- [ ] **C3. iCaRL: Incremental Classifier and Representation Learning** — Rebuffi et al. CVPR 2017.
      *Vì sao:* tổ tiên của HAI baseline cùng lúc: replay buffer + phân loại nearest-class-mean
      → nguồn cite chuẩn cho cả `replay` lẫn `ncm`.
- [ ] **C4. Overcoming catastrophic forgetting in neural networks (EWC)** — Kirkpatrick et al. PNAS 2017.
      Lướt 15 phút phần method — đủ để mô tả và cite đúng.
- [ ] **C5. Learning without Forgetting (LwF)** — Li, Hoiem. ECCV 2016 / TPAMI 2017.
      Tương tự C4. (Tiện thể soi lại vì LwF của mình bất thường trên EuroSAT.)

## Nhóm D — "PAPER BẰNG CODE" (đọc kèm B1, đã nằm trong kế hoạch từ G0)

- [ ] **D1. github.com/obekt/HOPE-nested-learning** — bản CMS dễ hiểu nhất (`cms_tiers=[[n,period]]`).
- [ ] **D2. github.com/kmccleary3301/nested_learning** — bản bám công thức nhất, có
      `docs/PAPER_COMPLIANCE.md` map phương trình → code, và script đo forgetting.

## Lịch đọc gợi ý (khớp tiến độ máy đang chạy)

| Khi nào | Đọc | Để làm gì |
|---|---|---|
| Đêm nay (VM đang train) | A1 | Mai đọc số G2 thấm gấp đôi |
| Trước khi đọc số G3 | A2 + B1 (§7) + D1 | Hiểu ‖Δw‖ và ablation đầu-chậm/cuối-chậm |
| Trước khi tune M3 / viết phần optimizer | B2 + B1 (§4) | Giải thích được NS + delta momentum |
| Tuần viết báo cáo | C1→C5 + B3 | Related Work + cite chuẩn |

## Ghi chú sau khi đọc (điền dần)

> Đã có bản tóm tắt + giải thích chi tiết + câu hỏi tự kiểm tra cho cả 13 nguồn ở
> `docs/TOM_TAT_PAPER.md` (đọc trực tiếp NL.pdf từng trang + code thật trong `src/uavcl/`,
> không suy diễn chung chung). 3 dòng dưới đây là bản rút gọn từ đó — tự đọc lại phần chi
> tiết + tự tick ô ở trên khi đã thấm, đừng tick hộ dựa trên 3 dòng này không thôi.

### A1 Titans:
- Neural memory tự cập nhật trọng số ngay trong forward qua 3 khái niệm: surprise (=−∇ loss
  key→value), momentum của surprise (α_t/η_t học từ data), forget gate (tổng quát hoá gate
  của LSTM/RNN hiện đại).
- 3 kiến trúc chính MAC/MAG/MAL (memory làm context/gate/layer) + biến thể LMM-only (Appendix
  C, chỉ memory không attention) — G2 hiện tại (`TitansClassifier`) gần LMM-only nhất vì
  không có attention nào chạy song song/nối tiếp với `TitansMemory`.
- Chi tiết + câu hỏi tự kiểm tra: `docs/TOM_TAT_PAPER.md`, mục A1.

### A2 Geva:
- MLP trong Transformer = bộ nhớ key-value thật (W1=key khớp pattern input, W2=value=phân
  phối token kế tiếp) — chứng minh bằng thực nghiệm, không phải suy diễn lý thuyết.
- Đây là lý do CMS chỉ chia tần số cho khối MLP chứ không phải attention — khớp đúng cách
  `build_cms_param_groups` trong `models/cms.py` tách attn/norm ra khỏi tier MLP.
- Chi tiết: `docs/TOM_TAT_PAPER.md`, mục A2.

### B1 NL.pdf:
- Luận điểm 1 câu: mọi thành phần (trọng số/optimizer/memory) đều là associative memory,
  chỉ khác tần số ghi — đã đọc trực tiếp §3/§4/§7/§8, đối chiếu từng phương trình với
  `m3.py`/`cms.py`/`hope.py` (Eq. 48–49 Delta Momentum, Eq. 70–71 CMS, Algorithm 1 M3,
  Eq. 83–93 self-modifying Titans).
- 2 điểm đáng cân nhắc phát hiện được: (1) `cms.py` khởi tạo weight giống §7.3 "Ad-hoc Level
  Stacking" hơn là đúng nghĩa "Independent variant Eq. 74" như ghi trong `LOGIC_NL`; (2)
  `hope.py` ghép Titans **gốc** (A1) chứ chưa dùng self-modifying Titans của §8.1 — không sai,
  nhưng nên phát biểu chính xác lúc bảo vệ.
- Chi tiết đầy đủ: `docs/TOM_TAT_PAPER.md`, mục B1 (dài nhất, có bảng đối chiếu equation↔code).

### B2 Muon:
- Trực giao hoá update bằng Newton-Schulz (5 bước, hệ số 3.4445/-4.7750/2.0315) thay SVD —
  chỉ áp cho tham số 2D; tham số 1D/embedding/head phải dùng AdamW.
- Quy ước "1D→Adam" chính là gốc rễ lý thuyết của bug limit-cycle đã sửa trong M3 (không chỉ
  là vá bug, có căn cứ từ chính blog gốc).
- Chi tiết: `docs/TOM_TAT_PAPER.md`, mục B2.

### B3 Schlag/Irie/Schmidhuber:
- Linear attention ≡ fast weight programmer (Schmidhuber đầu 1990s); delta rule cho phép SỬA
  giá trị đã lưu thay vì chỉ cộng dồn — cầu nối lịch sử 30 năm tới Titans/Delta Momentum.
- Chi tiết: `docs/TOM_TAT_PAPER.md`, mục B3.

### B4 TTT (tuỳ chọn):
- Hidden state = chính một mô hình ML (tuyến tính/MLP), cập nhật bằng 1 bước tự giám sát ngay
  test-time — ý tưởng song song Titans, cùng thời điểm giữa 2024.
- Chi tiết: `docs/TOM_TAT_PAPER.md`, mục B4.

### C1 GEM:
- Ma trận R (T×T) là nguồn gốc `acc_matrix`; Avg Acc/BWT/FWT trong `metrics/continual.py`
  định nghĩa đúng GEM; thuật toán GEM gốc (episodic memory + chiếu gradient) KHÔNG được cài
  trong dự án, chỉ mượn metric.
- Chi tiết: `docs/TOM_TAT_PAPER.md`, mục C1.

### C2 van de Ven & Tolias:
- 3 kịch bản task-/domain-/class-incremental phân theo việc task-ID có biết lúc test không;
  dự án đang làm **class-incremental** (khó nhất) — khớp đúng `mask_logits` + eval trên mọi
  class đã học trong `engine.py`.
- Chi tiết: `docs/TOM_TAT_PAPER.md`, mục C2.

### C3 iCaRL:
- Nguồn gốc CHUNG của 2 baseline trong dự án: Replay (buffer chọn bằng herding trong bản gốc,
  dự án dùng random — đơn giản hoá có chủ đích) + NCM (phân loại bằng khoảng cách tới
  prototype, tránh FC bị lệch về class mới học).
- Chi tiết: `docs/TOM_TAT_PAPER.md`, mục C3.

### C4 EWC:
- Fisher information (≈ bình phương gradient) đo "độ quan trọng" từng tham số với task cũ;
  phạt bậc hai kéo tham số quan trọng về giá trị cũ; nhược điểm mỗi task một anchor riêng →
  bộ nhớ phình tuyến tính (khớp đúng `footprint_floats` trong `methods.py::EWC`).
- Chi tiết: `docs/TOM_TAT_PAPER.md`, mục C4.

### C5 LwF:
- Không lưu ảnh cũ; dùng bản sao model cũ làm teacher, ép logits class cũ bám theo qua
  KL-distillation có nhiệt độ T; nghi vấn F=0.609 bất thường trên EuroSAT nên soi lại λ/T
  trước khi nghi ngờ chỗ khác.
- Chi tiết: `docs/TOM_TAT_PAPER.md`, mục C5.

### D1+D2 (2 repo GitHub):
- D1 (`obekt/HOPE-nested-learning`): bản demo dễ đọc, `cms_tiers=[[n,period]]` áp cho TOÀN BỘ
  layer của một LM tự huấn luyện từ đầu — khác dự án (chỉ áp cho MLP-block retrofit trên ViT
  pretrained).
- D2 (`kmccleary3301/nested_learning`): bản nghiêm túc nhất, có `docs/PAPER_COMPLIANCE.md`
  đối chiếu phương trình↔code y hệt tinh thần `LOGIC_NESTED_LEARNING.md` của dự án; xác nhận
  chéo số phương trình Eq. 83–93 cho self-modifying Titans (khớp với B1).
- Chi tiết: `docs/TOM_TAT_PAPER.md`, mục D1+D2.
