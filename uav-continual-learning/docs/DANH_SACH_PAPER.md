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

### A1 Titans:
-
### A2 Geva:
-
(...)
