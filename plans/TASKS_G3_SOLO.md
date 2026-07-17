# TASK LIST G3 SOLO — Retrofit CMS vào backbone ⭐ (bám PLAN_G3_CMS.md)

> Quyết định đã chốt (2026-07-14): làm SOLO · chạy CẢ 2 giả thuyết đầu-chậm/cuối-chậm ·
> tier mặc định **[[4,1],[4,4],[4,16]]** (ViT-S 12 block) · **attention+LayerNorm vào TẦNG
> CHẬM NHẤT** (không đóng băng hẳn — kèm núm `attn: freeze` làm đường lùi nếu bất ổn).
> Khác G2: backbone MỞ BĂNG — CMS chính là "fine-tune có kiểm soát", đối thủ trực tiếp
> là finetune (Acc 0.4144 / F 0.6133) và phải tiến về phía ncm 0.6933 / replay 0.7937.
> Luật chống kẹt: mắc 1 bước > 2 ngày → dừng, hỏi, hoặc hạ về cấu hình đơn giản hơn.

**Trạng thái:** ⬜ | Code ⬜ → Ablation EuroSAT ⬜ → RESISC45 ⬜ → Cơ chế + kết luận ⬜

---

## Ngày 0 — điều kiện vào 🔴
- [ ] **P1** G2 đã có số bậc C (ít nhất EuroSAT); `docs/KET_LUAN_G2.md` viết xong (nửa trang là đủ).
      (Được phép chồng lấn: code S1–S4 trong lúc máy còn chạy RESISC45 của G2.)
- [ ] **P2** Tag `v0.2-g2-titans`. Tạo `docs/G3_LOG.md`, chép 4 quyết định ở đầu file này vào.

## Tuần 1 — Code (giao Claude được ~80%)
- [ ] **S1** `src/uavcl/optim/cms_optimizer.py` — **CMSOptimizer** bọc AdamW:
      mỗi param-group có (period p, eta_scale η); gradient TÍCH LŨY mỗi bước;
      group chỉ `.step()` khi `global_step % p == 0` với **grad trung bình** của p bước.
      Đây là toàn bộ "phép màu" CMS bản optimizer — không đụng forward của ViT.
- [ ] **S2** Unit test cho S1 (quan trọng nhất G3): (a) param tier chậm KHÔNG đổi giữa các
      chu kỳ; (b) grad được trung bình đúng (so tay trên tensor bé); (c) η scale đúng;
      (d) sau `zero_grad` bộ đệm tích lũy không rò sang chu kỳ sau.
- [ ] **S3** `src/uavcl/models/cms.py` — `build_cms_param_groups(model, cfg)`:
      gom `backbone.blocks[i].mlp` thành 3 tier theo `order: early_slow | late_slow`;
      attention + norm → tier chậm nhất (quyết định của bạn; `attn: freeze` = đường lùi);
      patch_embed/pos_embed → đóng băng; head → tier nhanh nhất.
      Kèm hàm `tier_report(model)` in ra block nào thuộc tier nào (kiểm bằng mắt 1 lần).
- [ ] **S4** Nối engine: `train_one_task` dùng CMSOptimizer khi `cfg.cms.enabled`
      (~10 dòng, không đổi giao thức); method mới `cms` (chỉ để đặt tên run + log Δw).
- [ ] **S5** Log **‖Δw‖ per-tier per-task** (bằng chứng cơ chế, bắt buộc cho báo cáo):
      chụp weight đầu task, cuối task tính `‖w_end − w_start‖` từng tier, in + ghi G3_LOG.
      Kỳ vọng: tier chậm ≈ 0, tier nhanh lớn — nếu ngược lại là cài sai.
- [ ] **S6** Configs: `g3_cms_eurosat.yaml`, `g3_cms_resisc45.yaml`
      (`backbone.freeze: false`, `cms: {enabled, tiers: [[4,1],[4,4],[4,16]], etas: [1.0,0.5,0.1], order: late_slow, attn: slow}`)
      + `g3_cms_smoke.yaml` (tinycnn không có blocks → smoke dùng ViT tí hon hoặc bỏ qua smoke, chạy thẳng EuroSAT 1 epoch).
- [ ] **S7** `pytest -q` xanh; chạy thử EuroSAT 1 epoch (`--set train.epochs_per_task=1`) — không NaN, tier_report đúng.

## Tuần 2 — Ablation EuroSAT (trái tim của cổng G3: ≥2 cấu hình, chuẩn là 4)
- [ ] **S8** Ma trận 4 run, seed 0, config đầy đủ 3 epoch:
      {order: early_slow, late_slow} × {periods: [1,4,16], [1,8,64]}.
      Lệnh mẫu: `python scripts/run_g1.py --config configs/g3_cms_eurosat.yaml --method cms --set cms.order=early_slow`
- [ ] **S9** Đọc số + ghi G3_LOG: cấu hình nào Forgetting thấp nhất mà Acc không sập?
      So ngay với finetune EuroSAT (0.4213 / F 0.2566) — CMS phải thắng rõ cả hai cột.
- [ ] **S10** (cắt được) η tầng chậm: 0.1 vs 0.0 (đóng băng tuyệt đối) — 1 run thêm.

## Tuần 3 — RESISC45 + bằng chứng cơ chế
- [ ] **S11** Chạy đêm RESISC45 với cấu hình THẮNG ở S9 (+ á quân nếu máy rảnh).
      Mốc: thắng finetune (0.4144/F 0.6133) đậm; đích đẹp: Acc ≥ 0.65 & F ≤ 0.2, `method_extra_floats ≈ 0`.
- [ ] **S12** Vẽ 2 hình từ log: (i) bar ‖Δw‖ theo tier × task (tầng chậm bất động);
      (ii) forgetting: cms vs finetune vs ncm vs titans-C.
- [ ] **S13** (cắt được) `attn: freeze` chạy 1 lần đối chứng — nếu attn-trong-tier-chậm gây bất ổn thì đây là đường lùi đã có số.

## Tuần 4 — Chốt sổ
- [ ] **S14** `docs/KET_LUAN_G3.md`: CMS giảm quên bao nhiêu so finetune; đầu-chậm hay
      cuối-chậm thắng (phát hiện nghiên cứu!); chi phí ≈ 0 so replay 135M floats; Δw nói gì.
- [ ] **S15** Tự họp cổng (PLAN_G3 §6): retrofit chạy + ≥2 ablation + bảng so đủ → GO/NO-GO
      sang G4; tag `v0.3-g3-cms`.

## Rủi ro nhanh (chi tiết PLAN_G3 §5)
- CMS không giảm quên → soi ‖Δw‖ trước: tầng chậm vẫn trôi → tăng p/giảm η; tầng chậm đứng
  yên mà vẫn quên → quên nằm ở head/tier nhanh → thử mask-grad head cột class cũ.
- Attn trong tier chậm gây loss dựng đứng → `--set cms.attn=freeze` (S13 làm sẵn đối chứng).
- timm đổi cấu trúc `.blocks[i].mlp` → tier_report (S3) bắt được ngay ngày đầu.

## Phân bổ sức solo
~45% code S1–S7 (Claude gánh chính, bạn review tier_report) · ~35% máy chạy S8/S11 ·
~20% đọc số + viết S9/S12/S14 (phần của riêng bạn — đây là chương Phương pháp khi bảo vệ).
