# TASK LIST G2 — Titans xuyên task (bám PLAN_G2_TITANS.md)

> Cách dùng: tick `[x]` + ghi ngày/người sau mỗi task. Task có mã (A1, B2...) để nhắc
> trong commit message: `git commit -m "A3: TitansMemory forward+state"`.
> Quy ước: mỗi task ≤ 2 ngày công; nếu ước lượng > 2 ngày → tách nhỏ trước khi làm.

**Trạng thái:** ⬜ chưa bắt đầu | Bậc A ⬜ → Bậc B ⬜ → Bậc C ⬜ → Đo & kết luận ⬜

---

## 0. Điều kiện vào (chặn — làm xong mới mở G2)
- [ ] **P1** (cả team): bảng G1 đủ 2 dataset × 5 method — `python scripts/compare_g1.py` ra `baseline_table.md`.
- [ ] **P2** (N1): viết `uav-continual-learning/docs/KET_LUAN_G1.md` (~1 trang: finetune quên ?%, NCM/replay đứng đâu, seed 0).
- [ ] **P3** (N1): tag mốc: `git tag v0.1-g1-baseline && git push --tags`.
- [ ] **P4** (N2): chạy lại `python scripts/smoke_titans.py` — xác nhận titans-pytorch còn chạy sau khi cài env mới; ghi version vào KET_LUAN_G1.

## 1. Chốt interface (Tuần 1, đầu tuần — N2+N3 họp 1 buổi)
- [ ] **I1** (N2+N3): chốt chữ ký 2 API, ghi thẳng vào docstring file mới:
      `TitansMemory.forward(seq: (1,L,D), state=None) -> (out: (1,L,D), state)` và
      `SeqAdapter(feats: (B,P,D), mode) -> seq: (1,B,D) [image_seq] | (1,B*P,D) [token_seq]`.
- [ ] **I2** (N3): chốt `feat_dim` thực tế: in shape output `vit_small_patch16_224` với
      `num_classes=0` — quyết định D=384 đi thẳng vào memory hay chiếu Linear về `memory.dim`.
- [ ] **I3** (N1): thêm block config mẫu vào `configs/g2_titans_eurosat.yaml` (copy từ g1_eurosat):
      `memory: {enabled: true, dim: 384, chunk_size: 64, reset: image, detach_every: 8}` + `backbone.freeze: true`.

## 2. Bậc A — reset mỗi ảnh (Tuần 1–2): mục tiêu = pipeline chạy, không NaN
- [ ] **A1** (N2): tạo `src/uavcl/models/memory.py` — class `TitansMemory` bọc
      `titans_pytorch.NeuralMemory`: forward + trả state; state có `.detach_()`, `.clone()`.
- [ ] **A2** (N2): unit test `tests/test_titans_memory.py` (skipif no torch):
      (1) forward shape đúng; (2) chạy 2 lần cùng input + cùng state → cùng output (determinism);
      (3) state sau forward KHÁC state trước (memory có update).
- [ ] **A3** (N3): tạo `src/uavcl/models/seq_adapter.py` — 2 mode `token_seq`/`image_seq` + test shape.
- [ ] **A4** (N3): tạo `src/uavcl/models/titans_head.py` — `TitansClassifier(backbone, adapter, memory, head)`,
      forward trả logits (B, C); backbone tự freeze như NCMClassifier (copy pattern từ `ncm.py`).
- [ ] **A5** (N3): nối vào `scripts/run_g1.py`: nhánh model thứ 3 khi `cfg.memory.enabled`
      (giữ nguyên engine/metrics — chỉ thêm model). Method dùng `finetune` (train memory+head).
- [ ] **A6** (N1): smoke synthetic: `run_g1.py --config configs/g2_titans_smoke.yaml` (tạo config
      từ g1_smoke + memory.enabled) → chạy hết, loss giảm, không NaN.
- [ ] **A7** (N1): chạy A trên EuroSAT: `--config configs/g2_titans_eurosat.yaml` → so nhanh với
      NCM/finetune-frozen. Kỳ vọng ≈ linear-probe. Ghi số vào `docs/G2_LOG.md` (bảng chạy dần).
- **Cổng bậc A:** A1–A7 xong + `pytest -q` xanh toàn bộ.

## 3. Bậc B — state sống trong task (Tuần 2–3)
- [ ] **B1** (N2): thêm chế độ `reset: task` vào TitansClassifier/engine-hook: state giữ qua các
      batch trong 1 task; **`state.detach()` mỗi `detach_every` batch** (truncated BPTT).
- [ ] **B2** (N2): xử lý biên: batch cuối task ngắn hơn; eval KHÔNG được sửa state → dùng
      `state.clone()` trong `evaluate` (kiểm tra bằng test: eval 2 lần → acc y hệt).
- [ ] **B3** (N2): unit test `tests/test_state_lifecycle.py`: state None lúc đầu task mới (mode task);
      state khác None giữa task; detach đúng nhịp (không giữ graph — check `state.requires_grad == False` sau detach).
- [ ] **B4** (N1): chạy B trên EuroSAT (`--set memory.reset=task`), so A vs B trong G2_LOG.md.
- [ ] **B5** (N3): đo tốc độ A vs B (giây/epoch) — nếu B chậm > 2× A, profile chỗ nghẽn trước khi sang C.
- **Cổng bậc B:** B ≥ A về Avg Acc trên EuroSAT (nếu kém hơn hẳn → debug trước khi sang C).

## 4. Bậc C — XUYÊN TASK, đích của G2 (Tuần 3–4)
- [ ] **C1** (N2): chế độ `reset: never`: state truyền qua ranh giới task trong `run_continual`;
      trước MỖI lượt eval → snapshot `state.clone()`, eval xong vứt (stream không bị eval làm bẩn).
- [ ] **C2** (N2): ổn định số: LayerNorm sau memory output; theo dõi `norm(state)` mỗi task —
      log ra G2_LOG.md; nếu tăng không chặn → thêm clip/decay.
- [ ] **C3** (N2): save/load state vào checkpoint (`torch.save` cùng model) — cần cho resume + G4.
- [ ] **C4** (N1): chạy C trên EuroSAT + RESISC45, **bật `train.eval_future=true`** (FWT là điểm
      khoe của xuyên-task). Lệnh mẫu:
      `python scripts/run_g1.py --config configs/g2_titans_resisc45.yaml --set memory.reset=never train.eval_future=true`
- [ ] **C5** (N1): ablation reset: 3 run EuroSAT `reset=image|task|never` cùng seed → bảng nhỏ
      "giá trị của trí nhớ dài" (đây là bảng bán hàng của G2).
- [ ] **C6** (N3): kiểm tra chunk_size 32/64/128 trên RESISC45 (tốc độ vs acc) — chọn 1 ghi vào config chính thức.

## 5. Đo, viết, đóng giai đoạn (Tuần 4)
- [ ] **D1** (N1): bảng tổng G2: Titans A/B/C vs 5 baseline (EuroSAT + RESISC45, seed 0) —
      `compare_g1.py` gộp được vì cùng harness; thêm cột FWT.
- [ ] **D2** (N1): 2 biểu đồ: (i) forgetting-theo-task C vs finetune vs ncm; (ii) FWT so sánh.
- [ ] **D3** (N2): `docs/KET_LUAN_G2.md`: memory xuyên task giúp/không, ở đâu, vì sao (kèm log norm state).
- [ ] **D4** (N3): dọn code: docstring 3 file mới, xoá nhánh chết, `pytest` + smoke xanh trên cả 3 máy.
- [ ] **D5** (cả team): họp cổng — đối chiếu checklist "Cổng chuyển sang G3" trong PLAN_G2 → quyết định GO/NO-GO.

## Rủi ro nhắc nhanh (chi tiết trong PLAN_G2 §5)
NaN ở C → giảm η/detach dày hơn/LayerNorm; C ≈ NCM → thử image_seq↔token_seq, tăng dim;
kẹt quá 3 ngày ở 1 task → hạ mục tiêu về bậc B, ghi phân tích C thất bại (vẫn qua cổng).

## Tổng kết khối lượng
| Ai | Số task | Nặng nhất |
|---|---|---|
| N2 | 9 (A1,A2,B1,B2,B3,C1,C2,C3,D3) | C1–C2 (state xuyên task + ổn định số) |
| N3 | 7 (I2,A3,A4,A5,B5,C6,D4) | A4–A5 (wiring vào harness) |
| N1 | 8 (I3,A6,A7,B4,C4,C5,D1,D2) | C4–C5 (ma trận run + đọc số) |
