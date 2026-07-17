# G2 SOLO — làm một mình từ A đến Z (thay bản chia 3 người)

> Nguyên tắc bản solo: đi TUYẾN TÍNH (không có việc song song), cắt mọi thứ "nice-to-have",
> code nặng có thể giao Claude viết — bạn tập trung CHẠY + ĐỌC SỐ + QUYẾT ĐỊNH.
> Ước lượng: 3–4 tuần (1–2h/ngày code nhờ hỗ trợ + máy chạy qua đêm).
> Luật chống kẹt: mắc ở 1 bước quá 2 ngày → dừng, hỏi/hạ bậc, không cố.

## Ngày 0 — đóng mốc G1 (30 phút) 🔴 chặn
- [ ] `cd uav-continual-learning && git add -A && git commit -m "G1 done" && git tag v0.1-g1-baseline`
- [ ] `python scripts/smoke_titans.py` trên máy GPU → phải [OK]. Lỗi thì sửa env trước.

## Ngày 1 — tự chốt 3 quyết định thiết kế (không cần họp ai)
Ghi 5 dòng vào `docs/G2_LOG.md` (file nhật ký chạy, tự tạo):
- [ ] Chế độ chuỗi chính: **image_seq** (mỗi ảnh = 1 bước thời gian — khớp mục tiêu xuyên task; token_seq để dành ablation nếu dư thời gian).
- [ ] `memory.dim = 384` (bằng feat_dim ViT-S — khỏi lớp chiếu).
- [ ] `chunk_size = 64`, `detach_every = 8` (mặc định, chỉ đụng khi OOM/chậm).

## Tuần 1 — Bậc A: pipeline chạy (phần code lớn nhất, giao Claude được ~90%)
- [ ] **S1** Viết 3 file: `src/uavcl/models/memory.py` (TitansMemory bọc NeuralMemory, forward(seq,state)->(out,state), state detach/clone được) · `seq_adapter.py` (image_seq + token_seq) · `titans_head.py` (TitansClassifier: frozen ViT → adapter → memory → head, ra logits (B,C)).
- [ ] **S2** Nối `scripts/run_g1.py`: nhánh model khi `cfg.memory.enabled` (bắt chước nhánh ncm có sẵn).
- [ ] **S3** Tạo `configs/g2_titans_smoke.yaml` (từ g1_smoke + memory) và `configs/g2_titans_eurosat.yaml` (từ g1_eurosat + memory + `backbone.freeze: true`).
- [ ] **S4** Test: `tests/test_titans_memory.py` (shape đúng; chạy 2 lần cùng state → cùng output; state có thay đổi sau forward). `pytest -q` xanh.
- [ ] **S5** Chạy: smoke → rồi `python scripts/run_g1.py --config configs/g2_titans_eurosat.yaml` (reset: image). ✅ Đạt khi: chạy hết, không NaN, Acc ≈ ncm (0.6–0.75). Ghi số vào G2_LOG.md.

## Tuần 2 — Bậc B: nhớ trong task
- [ ] **S6** Thêm `reset: task`: state giữ qua các batch trong 1 task, `state.detach()` mỗi 8 batch, reset khi sang task mới.
- [ ] **S7** Luật "chấm thi không ghi trí nhớ": trong evaluate dùng `state.clone()`; test: eval 2 lần liên tiếp → acc y hệt.
- [ ] **S8** Chạy B EuroSAT: `--set memory.reset=task`. ✅ Đạt khi B ≥ A về Avg Acc. Kém hơn hẳn → debug (thường do state rò qua eval hoặc detach sai nhịp) trước khi đi tiếp.

## Tuần 3 — Bậc C: nhớ xuyên task (ĐÍCH) + chạy lớn
- [ ] **S9** Thêm `reset: never`: state đi xuyên ranh giới task; TRƯỚC mỗi lượt eval → snapshot clone; log `norm(state)` mỗi task vào G2_LOG.md (phát hiện "phình").
- [ ] **S10** Lưu state vào checkpoint (torch.save cùng model) — cần cho G4.
- [ ] **S11** Bảng bán hàng của G2 — 3 run EuroSAT cùng seed: `reset=image|task|never`, bật `train.eval_future=true`. Kỳ vọng: never ≥ task ≥ image, FWT của never nhỉnh nhất.
- [ ] **S12** Chạy đêm RESISC45: `reset=never` (+ `task` nếu kịp), eval_future=true.
  Mốc so (từ KET_LUAN_G1): **phải vượt NCM 0.693; mơ tới replay 0.794; Forgetting ≤ 0.1.**
- Cắt-được nếu thiếu thời gian: sweep chunk_size, token_seq ablation, profiling.

## Tuần 4 — Chốt sổ
- [ ] **S13** `python scripts/compare_g1.py` (Titans lẫn vào bảng chung vì cùng harness); vẽ 2 hình: forgetting-theo-task (C vs finetune vs ncm) + cột FWT. (Script vẽ giao Claude.)
- [ ] **S14** Viết `docs/KET_LUAN_G2.md` (nửa trang): C được bao nhiêu / có lấy được phần nào trong "10 điểm khoảng trống NCM↔replay" không / norm state có ổn định / FWT có nhích / giá bộ nhớ.
- [ ] **S15** Tự họp cổng = đối chiếu mục "Cổng chuyển sang G3" trong PLAN_G2_TITANS.md → GO/NO-GO, tag `v0.2-g2-titans`.

## Khác gì bản 3 người?
Bỏ: họp interface (I1 → bạn tự quyết ở Ngày 1), B5/C6 (profiling, sweep — chỉ làm khi đau),
phân vai. Giữ nguyên: 3 bậc A→B→C, các luật an toàn (detach, clone khi eval, log norm state),
và cổng số liệu — vì chúng bảo vệ BẠN khỏi tự lừa mình bằng số sai.

## Phân bổ sức một mình
~40% code (Claude gánh phần lớn S1–S4, S6–S10) · ~40% máy chạy (S5, S8, S11–S12 — set rồi đi ngủ)
· ~20% đọc số + viết kết luận (S14 — phần KHÔNG nên giao ai, vì đây là hiểu biết của bạn khi bảo vệ).
