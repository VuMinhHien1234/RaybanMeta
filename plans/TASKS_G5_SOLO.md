# TASK LIST G5 SOLO — Đánh giá chặt (bám PLAN_G5, cập nhật theo 4 quyết định 2026-07-17)

> Quyết định đã chốt: dataset 3 = **UCM** · máy = **GCP CPU nhiều đêm** (chưa GPU) ·
> robustness = **cả 3 bài** (đảo thứ tự task + corruption + cross-dataset) ·
> vòng chung kết = 6 model lõi **+ cả 4 mục mở rộng** (optimizer_per_task, HOPE+AdamW,
> EWC/LwF đa seed, η adaptive).
>
> ⚠ THÀNH THẬT VỀ NGÂN SÁCH MÁY: bạn chọn gói "full option" — trên CPU, ma trận đầy đủ
> là ~2 TUẦN đêm. Task list dưới đây xếp theo BẬC ƯU TIÊN: hết thời gian thì cắt từ dưới
> lên, bảng vẫn đứng vững. (Nâng billing → L4 bất kỳ lúc nào: toàn bộ co về 1–2 đêm, ~$5–10.)

**Điều kiện vào (P):**
- [ ] P1: `--quick` sau fix M3-RMS chạy xong sạch, đọc số EuroSAT, chốt CMS-winner + Titans bậc thắng.
- [ ] P2: `KET_LUAN_G2/G3/G4.md` (mỗi cái nửa trang) — không có kết luận thì đa seed là vô nghĩa.
- [ ] P3: η adaptive (task S9 của G4) đã cài + có tín hiệu trên EuroSAT — vì bạn muốn đưa nó vào đa seed.

## Phần C — Code (Claude làm được ~90%, làm TRƯỚC khi đốt đêm nào)

- [ ] **C1. Nguồn UCM** — `_load_ucm` vào `sources.py` (21 class × 100 ảnh, 7 task × 3 class)
      + `configs/g5_ucm.yaml`. Nguồn tải: HF hub hoặc link UC Merced (viết fallback 2 nguồn,
      kiểm khi chạy thật). KHÔNG tinh chỉnh gì theo UCM — chạy y cấu hình thắng của RESISC45.
- [ ] **C2. Tách `stream_seed` khỏi `seed`** ⭐ phát hiện phương pháp luận quan trọng:
      hiện `seed` điều khiển CẢ khởi tạo model LẪN thứ tự chia class → đa seed đang trộn lẫn
      "phương sai model" với "phương sai thứ tự task". Sửa: `data.stream_seed` riêng
      (mặc định = seed để không phá kết quả cũ). Đa seed G5: stream_seed CỐ ĐỊNH, seed đổi.
      Đảo-thứ-tự: seed cố định, stream_seed đổi. → hai trục phương sai đo TÁCH BẠCH.
- [ ] **C3. Corruption eval** — `--set eval.corruption=blur|noise|fog` áp transform lúc EVAL
      (không train lại): đo model đã train chịu điều kiện bay xấu tới đâu (~60 dòng + test).
- [ ] **C4. Cross-dataset zero-shot** — script `scripts/eval_transfer.py`: nạp checkpoint
      RESISC45, map thủ công ~8–10 class tương đương RESISC45↔UCM (forest, river, beach...),
      đo acc trên UCM KHÔNG train. Kèm file map `configs/class_map_resisc_ucm.yaml` để ai
      cũng soi được tính công bằng.
- [ ] **C5. Bộ chạy G5** — `scripts/run_g5.py`: ma trận {model × seed × dataset} theo BẬC
      (--tier 1|2|3, resumable như run_all) + `scripts/aggregate_g5.py`: gộp mean±std
      (groupby dataset+method+optimizer), xuất bảng LaTeX/markdown + biểu đồ errorbar.

## Phần R — Chạy (xếp theo bậc; mỗi ô ghi ước lượng CPU)

- [ ] **R1 (BẬC 1 — EuroSAT đa seed, ~4–6 đêm CPU):** 
      6 model lõi + 4 mở rộng × seed {0,1,2}, stream_seed=0 cố định.
      Baselines resnet18 (nhanh, ~10ph/run); NL models vit_tiny (~2–4h/run).
      Gồm luôn: HOPE+AdamW ×3 seed, HOPE optimizer_per_task=false ×3 seed, η-adaptive ×3 seed.
- [ ] **R2 (BẬC 1 — UCM đa seed, ~2–3 đêm):** cùng danh sách, dataset UCM (nhỏ nên rẻ).
- [ ] **R3 (BẬC 2 — RESISC45, ~3–5 đêm, MỖI ĐÊM 1–2 RUN):** hoàn thiện seed-0 đủ bộ
      (finetune/cms-best/hope nếu --quick chưa phủ) + đa seed cho nhóm RẺ (ncm, titans-frozen
      — CPU chịu được). Full fine-tune đa seed RESISC45 = ghi rõ "cần GPU" nếu không kịp.
- [ ] **R4 (BẬC 1.5 — robustness, ~2 đêm):** đảo thứ tự: 3 stream_seed × {hope, cms, ncm}
      trên EuroSAT; corruption: eval lại các checkpoint sẵn có (rẻ, không train);
      zero-shot: chạy `eval_transfer.py` trên checkpoint RESISC45 sẵn có.
- [ ] **R5 (BẬC 3 — nếu có GPU):** full đa seed RESISC45 + AID bổ sung nếu muốn 4 dataset.

## Phần A — Phân tích & sản phẩm

- [ ] **A1.** Bảng cuối mean±std (aggregate_g5) — BẢNG TRUNG TÂM phiên bản thống kê;
      đánh dấu đậm khi khác biệt > 1 std.
- [ ] **A2.** Biểu đồ: errorbar Acc/Forgetting theo method từng dataset; heatmap ‖Δw‖
      tier×task (từ log); đường norm(state) theo task; cột FWT.
- [ ] **A3.** Case study định tính: confusion matrix trước/sau khi học task mới trên 2–3 cặp
      class dễ lẫn (River↔SeaLake, Highway↔Freeway...) + 4–6 hình ảnh minh hoạ.
- [ ] **A4.** `docs/FINDINGS.md` — 5–8 phát hiện chính, mỗi cái 1 câu + bảng/hình dẫn chứng.
      Nhớ đưa vào: chuỗi 8 bug phần cứng thật + phát hiện M3-scale (Algorithm 1 nguyên văn
      nổ trên vision) — chúng là findings, không phải chuyện bếp núc.
- [ ] **A5.** Đóng băng: tag `v0.9-results`; từ đây KHÔNG chạy lại gì trừ khi phát hiện bug làm sai số.

## Cổng chuyển sang G6
- [ ] Bảng mean±std ổn định trên ≥2 dataset (std không nuốt khác biệt giữa các method chính).
- [ ] Cả 3 bài robustness có số + 1 đoạn diễn giải mỗi bài.
- [ ] FINDINGS.md xong — đủ để VIẾT báo cáo mà không cần chạy thêm bất kỳ thí nghiệm nào.

## Thứ tự cắt nếu thiếu thời gian (cắt từ trên xuống)
1. Cross-dataset zero-shot (tham vọng nhất, dễ tranh cãi nhất) → chuyển future work.
2. EWC/LwF đa seed → giữ seed 0 + ghi chú.
3. AID / R5.
4. TUYỆT ĐỐI KHÔNG cắt: R1 (EuroSAT đa seed) + A4 (findings) — xương sống thống kê của báo cáo.
