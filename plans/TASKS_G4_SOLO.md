# TASK LIST G4 SOLO — Ghép HOPE (bám PLAN_G4_HOPE_M3.md, đã cập nhật theo 4 quyết định)

> Quyết định (2026-07-14): kiến trúc **nối tiếp feature-level** (Titans ngồi trên backbone-CMS) ·
> **η adaptive** làm SAU khi HOPE-fixed có số · ablation **đầy đủ 2×2 + thành phần** ·
> **code wiring NGAY**, chạy số sau khi G2/G3 xong.
>
> ĐÃ XONG TRƯỚC (không còn trong G4): M3 optimizer + Delta Momentum + test hội tụ ✓
> → G4 rút từ 5 tuần xuống ~3 tuần.
>
> Câu hỏi G4 phải trả lời: **"Titans (thích nghi nhanh) + CMS (bền) — 1+1 có > 2 không?"**

**Trạng thái:** Wiring ⬜ → Chạy ghép ⬜ → η adaptive ⬜ → Bảng trung tâm ⬜

---

## Phần A — Wiring (làm NGAY, không chờ số G2/G3)
- [ ] **S1** `src/uavcl/models/hope.py` — **HOPEClassifier**:
      ảnh → backbone ViT (MỞ BĂNG, nhịp update do CMS lo) → SeqAdapter (image_seq)
      → TitansMemory (state `reset: never` như G2-C) → post_norm + residual → head.
      Tái dùng tối đa: SeqAdapter/TitansMemory/state_utils của G2 — code mới chủ yếu là wiring.
- [ ] **S2** Mở rộng `models/cms.py`: nếu model có `.memory` → tham số memory vào **tier NHANH nhất**
      (memory chính là tầng tần số cao nhất của HOPE — đúng lý thuyết); head vẫn tier nhanh.
- [ ] **S3** Method `hope` trong methods.py = TitansCL (reset/log norm state) + CMS (log ‖Δw‖ per-tier)
      gộp lại; nhánh model trong run_g1: `memory.enabled && cms.enabled` → HOPEClassifier.
- [ ] **S4** Configs `g4_hope_eurosat.yaml` + `g4_hope_resisc45.yaml`: gộp block memory (G2)
      + cms (G3) + method hope + optimizer m3 (beta_style delta); test `tests/test_g4_hope.py`
      (vit_tiny pretrained=False, 2 task bé: forward chạy, state sống xuyên task, Δw tier chậm ≈ 0).
- [ ] **S5** `pytest -q` xanh + chạy thử EuroSAT 1 epoch: đọc tier_report + `[titans] norm(state)`
      + `[cms] ‖Δw‖` cùng xuất hiện — 3 hệ log sống chung không giẫm nhau.

## Phần B — Chạy ghép (SAU khi có số G2 bậc C + G3 ablation)
- [ ] **S6** Ghép cấu hình THẮNG: `cms.order` = quán quân S9/G3; giữ tiers/etas thắng;
      memory như G2-C. Chạy HOPE EuroSAT đầy đủ → so ngay với Titans-C và CMS-best trên G4_LOG.
- [ ] **S7** RESISC45 — bảng ablation đã chốt (chạy đêm, ~5 run, không ghi đè nhau):
      | run | model | optimizer |
      |---|---|---|
      | 1 | Titans-only (G2-C) | m3-delta |  ← đã có từ G2 nếu chạy rồi
      | 2 | CMS-only (best)    | m3-delta |  ← đã có từ G3
      | 3 | **HOPE**           | m3-delta |
      | 4 | CMS-only (best)    | adamw    |
      | 5 | **HOPE**           | adamw    |
- [ ] **S8** Đọc chéo 3 hệ bằng chứng: FWT (memory có giúp task chưa học?), norm(state)
      (ký ức ổn không khi backbone TRÔI dưới chân nó?), ‖Δw‖ (tier chậm còn bất động?).

## Phần C — η tự tính (CHỈ sau khi S7 có số — mỗi lần một biến số)
- [x] **S9** (code, 07-18 — làm SỚM hơn lịch vì đọc số G4 --quick 07-17 xấu, cần sửa cơ chế
      trước khi chạy lại — xem `docs/TIEN_DO_2026-07-18.md`) `cms.eta_mode: adaptive`: η_tier =
      η_base × (1 − cos(grad chu kỳ này, hướng update trước)), clip tự nhiên [0, η_base×2].
      Cài trong `CMSOptimizer._apply_adaptive_eta` + 3 unit test (`tests/test_g3_cms.py`:
      surprise 0 → η→nhỏ; ngược hướng → η→lớn; default vẫn `fixed`, không đổi hành vi cũ).
      Tắt mặc định. ⚠ Unit test mới viết CHƯA chạy được thật (sandbox không có torch) — bắt
      buộc `pytest -q tests/test_g3_cms.py` trên máy có torch trước khi tin.
- [ ] **S10** Ablation fixed-vs-adaptive: 2 run EuroSAT; đẹp thì +1 run RESISC45.
      (Đây là "self-modifying nhẹ" — một đóng góp riêng viết được vào báo cáo.)

## Phần D — Chốt sổ G4
- [ ] **S11** BẢNG TRUNG TÂM CỦA DỰ ÁN (điền bản mean seed 0; G5 sẽ thêm ±std):
      5 baseline G1 / Titans-C / CMS-best / HOPE / HOPE+adamw (+ adaptive nếu có)
      — cột: Acc, Forgetting, BWT, FWT, extra_floats, runtime. Kèm 2 biểu đồ.
- [ ] **S12** `docs/KET_LUAN_G4.md`: 1+1 có >2? Kiến trúc góp mấy điểm, optimizer góp mấy điểm
      (đọc từ 2×2)? Nếu HOPE ≤ CMS-only: thành phần nào thừa, vì sao — kết quả âm + giải thích
      vẫn qua cổng. Tag `v0.4-g4-hope` → GO G5.

## Rủi ro đặc thù HOPE (khác G2/G3 cộng lại)
- **Feature drift dưới chân memory**: backbone được CMS update → không gian feature TRÔI,
  ký ức cũ của Titans lệch dần. Đây là căng thẳng cốt lõi của HOPE. Theo dõi bằng norm(state)
  + acc task cũ; thuốc: giảm η tier nhanh của CMS, hoặc cho memory "ôn" lại sau mỗi task.
  Nếu quan sát được hiện tượng này + đo được thuốc → tự nó là một phát hiện cho báo cáo.
- OOM (CMS mở băng + memory + ViT-S): giảm batch 32→16, bật AMP; tệ nhất chạy EuroSAT làm chính.
- HOPE thua CMS-only: KHÔNG giấu — ablation chỉ ra Titans thừa ở đâu; báo cáo trung thực
  kiểu Meta-Rayban/CPM (định vị lại đóng góp) luôn đứng vững hơn số đẹp không giải thích được.

## Phân bổ sức solo (~3 tuần)
T1: Phần A (Claude gánh S1–S4, bạn đọc log S5) · T2: S6–S8 (máy chạy đêm, bạn đọc số)
· T3: S9–S12 (η adaptive + bảng trung tâm + kết luận — phần đọc số S11–S12 là của riêng bạn).
