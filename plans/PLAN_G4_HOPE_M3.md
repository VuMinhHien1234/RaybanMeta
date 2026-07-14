# PLAN G4 — Ghép HOPE + optimizer M3 (5 tuần, M3 BẮT BUỘC theo quyết định team)

**Mục tiêu:** hợp nhất 2 mảnh đã kiểm chứng riêng — Titans-xuyên-task (G2, thích nghi nhanh)
+ CMS (G3, bền chống quên) — thành khối HOPE; thay optimizer bằng **M3/Delta-Momentum**;
ra bảng so sánh ĐẦY ĐỦ.

## 1. Kiến trúc HOPE-UAV (đơn giản hoá có kiểm soát từ §8 paper)
```
ảnh ─▶ ViT-CMS (G3: MLP đa tần số, attention frozen/chậm)   ← tầng TRUNG + CHẬM
              │ token features
              ▼
      TitansMemory xuyên task (G2, state never-reset)        ← tầng NHANH NHẤT
              ▼
            head
```
Tức: **Titans = tầng tần số cao nhất NGỒI TRÊN stack CMS** — đúng quan hệ bổ sung mà paper
mô tả (Titans: luật học phức tạp/capacity nhỏ; CMS: luật đơn giản/capacity lớn). Ghi rõ trong
báo cáo đây là bản *feature-level HOPE*, không phải bản token-level đầy đủ của paper.
- File mới: `src/uavcl/models/hope.py` = ghép TitansClassifier(G2) + backbone-CMS(G3);
  gần như không code mới ngoài wiring + config `hope: {enabled, ...}` gộp 2 block config cũ.

## 2. Optimizer M3 / Delta-Momentum (phần nặng của N2 — 2 tuần)
- Nguồn bám: công thức trong NL.pdf + bản mechanism-level của kmccleary
  (`docs/PAPER_COMPLIANCE.md` map phương trình→code — đã xác minh tồn tại từ G0).
- Cài `src/uavcl/optim/m3.py` như một `torch.optim.Optimizer` chuẩn để cắm vào cả
  train thường lẫn CMSOptimizer (wrapper nhận inner optimizer — thiết kế sẵn từ G3).
- **Unit test bắt buộc:** trên bài toán lồi nhỏ (quadratic), M3 hội tụ; so quỹ đạo với AdamW.
- So sánh 2×2: {HOPE, CMS-only} × {AdamW, M3} — tách bạch "kiến trúc giúp" vs "optimizer giúp".

## 3. Việc theo người & tuần
- **N2 (chủ trì):** T1: wiring HOPE + smoke; T2–3: M3 (đọc công thức → cài → unit test);
  T4: 2×2 optimizer runs; T5: phân tích, note cơ chế.
- **N3:** T1: tích hợp hope.py vào run_g1 (nhánh model thứ 3); T2: **tối ưu tốc độ/bộ nhớ**
  (chunk-wise cho memory, AMP/bf16, gradient checkpointing nếu cần); T3–5: giữ pipeline ổn
  cho N1 chạy, profiling bảng thời gian train từng model (vào báo cáo).
- **N1:** T2–5: **bảng so sánh đầy đủ** trên RESISC45 + EuroSAT, seed 0:
  5 baseline G1 / Titans-C / CMS-best / **HOPE** / HOPE+M3 — cùng harness, cùng stream;
  ablation bật/tắt: Titans-only, CMS-only, HOPE (chứng minh "1+1>2" hoặc không).

## 4. Định nghĩa "xong" (bảng trung tâm của toàn dự án)
| Model | Forgetting ↓ | Avg Acc ↑ | FWT | Extra mem | Train time |
|---|---|---|---|---|---|
| finetune / ewc / replay / lwf / ncm | (từ G1) | | | | |
| Titans-C (G2) | | | | | |
| CMS-best (G3) | | | | | |
| **HOPE** | | | | | |
| **HOPE + M3** | | | | | |

## 5. Rủi ro & đường lùi
- HOPE không hơn CMS-only → ablation chỉ ra thành phần nào thừa; báo cáo "CMS là đủ, Titans
  thêm giá trị ở FWT/thích nghi" nếu số nói vậy — vẫn là kết luận công bố được.
- M3 khó cài/không hội tụ → gắn cờ đỏ CUỐI TUẦN 3: nếu chưa qua unit test lồi → thu hẹp:
  M3 chỉ áp cho head+memory (ít tham số), backbone giữ AdamW; ghi rõ giới hạn.
- Chậm/OOM khi ghép → N3 chunk-wise + giảm batch; tệ nhất: HOPE chỉ chạy RESISC45 với
  backbone.freeze=true cho bảng chính, full fine-tune để phụ lục.

## 6. Cổng chuyển sang G5
- [ ] HOPE end-to-end trên 2 dataset; bảng đầy đủ ở §4 kín số (seed 0).
- [ ] Ablation Titans-only/CMS-only/HOPE + 2×2 optimizer.
- [ ] HOPE vượt (hoặc ngang, kèm phân tích) baseline mạnh nhất về Forgetting — điều kiện cổng của Team_Plan.
- [ ] `docs/KET_LUAN_G4.md`.
