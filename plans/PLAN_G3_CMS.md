# PLAN G3 — Retrofit CMS vào backbone ⭐ (4 tuần, mấu chốt dự án)

**Mục tiêu:** biến các khối MLP của ViT thành bộ nhớ ĐA TẦN SỐ (tầng nhanh update mỗi bước,
tầng chậm update hiếm) → fine-tune trên stream mà KHÔNG quên. Khác G2: ở đây **mở băng
backbone** — CMS chính là "fine-tune có kiểm soát", nên đối thủ trực tiếp là finetune/EWC/replay.

## 1. Cách cài (bản ĐA-TẦN-SỐ-OPTIMIZER — dễ, đã chốt trong Team_Plan là điểm khởi đầu)
Không mổ forward của ViT. Chỉ điều khiển NHỊP UPDATE qua optimizer wrapper:
```
CMSOptimizer(bọc AdamW):
  - nhóm param theo tier: tier i có (danh sách block, chu kỳ p_i, lr nội bộ η_i)
  - mỗi step: gradient các tier đều được TÍCH LŨY;
    tier i chỉ .step() khi global_step % p_i == 0, dùng grad TRUNG BÌNH của p_i bước
  - tier chậm nhất có thể η→0 (gần như đóng băng — giữ kiến thức pretrained)
```
- File mới: `src/uavcl/models/cms.py` (chia tier từ `model.blocks[i].mlp` của timm ViT) +
  `src/uavcl/optim/cms_optimizer.py`. Engine chỉ cần 1 nhánh: nếu `cms.enabled` → dùng
  CMSOptimizer thay AdamW (thêm ~5 dòng vào `train_one_task`).
- Attention + LayerNorm + patch-embed: **ĐÓNG BĂNG** giai đoạn đầu (đúng trọng tâm paper:
  MLP là bộ nhớ). Head: tier nhanh nhất.
- Config (đã có chỗ trong default.yaml): `cms: {enabled: true, tiers: [[4,1],[4,4],[4,16]], etas: [1.0, 0.5, 0.1]}`
  (ViT-S có 12 block → 4 nhanh / 4 vừa / 4 chậm là điểm xuất phát).

## 2. Câu hỏi nghiên cứu phải trả lời bằng ABLATION (không tra được)
1. **Block nào chậm?** — 2 giả thuyết đối nghịch: (a) block ĐẦU chậm (giữ đặc trưng thấp
   phổ quát), (b) block CUỐI chậm (giữ ngữ nghĩa). Chạy cả hai + bản đảo (nhanh↔chậm).
2. **Mấy tầng?** — 2 tiers vs 3 tiers.
3. **Chu kỳ & η** — [[1,4,16]] vs [[1,8,64]]; η tầng chậm 0.1 vs 0.0 (đóng băng hẳn).
Cổng yêu cầu ≥2 cấu hình; plan này đặt chuẩn 4–6 cấu hình × EuroSAT (nhanh) rồi 2 cấu hình
tốt nhất × RESISC45.

## 3. Việc theo người
- **N2 (chủ trì):** tuần 1: đọc lại obekt (`cms_tiers`) + kmccleary (gộp gradient) — note
  đã có từ G0; viết CMSOptimizer + unit test (param tier chậm KHÔNG đổi khi chưa tới chu kỳ;
  grad được trung bình đúng). Tuần 2: nối cms.py + config. Tuần 3–4: ablation, tinh chỉnh.
- **N3:** tuần 1–2: map block ViT → tier (kiểm chứng `model.blocks[i].mlp` tồn tại với
  vit_small_patch16_224), freeze attention/norm đúng chỗ, đảm bảo `pytest` + smoke chạy;
  tuần 3: **log ‖Δw‖ per-tier per-task** (bằng chứng "tầng chậm gần như bất động" — bắt buộc
  cho báo cáo); tuần 4: thử mở attention vào tier chậm (nếu còn thời gian).
- **N1:** tuần 2–4: chạy ma trận ablation qua harness (mỗi cấu hình = 1 yaml trong
  `configs/g3/`); vẽ: forgetting vs cấu hình tier, đường ‖Δw‖ theo thời gian; bảng
  **CMS vs 5 baseline vs Titans-C** (cùng stream, cùng seed 0).

## 4. So sánh công bằng (tránh bị hội đồng bẻ)
- CMS dùng CÙNG epochs_per_task + lr gốc như finetune G1 (nó là biến thể của fine-tune).
- Báo cáo cả thời gian train/bộ nhớ: CMS gần như không thêm tham số (`method_extra_floats≈0`)
  — đây là điểm bán hàng so với replay (tốn RAM ảnh) và EWC (2×P mỗi task).

## 5. Rủi ro & đường lùi
- CMS không giảm quên so với finetune → soi ‖Δw‖: nếu tầng chậm vẫn trôi → giảm η/tăng p;
  nếu tầng chậm đứng yên mà vẫn quên → quên nằm ở head/tầng nhanh → thêm mask-grad cho head cũ.
- Kẹt kỹ thuật với timm internals → đường lùi: áp CMS chỉ lên 6 block cuối + head, phần còn
  lại freeze (vẫn đúng tinh thần §7.3 retrofit).
- Nếu bản optimizer chạy tốt và CÒN ≥1 tuần: thử bản "self-modifying" (η học được) — KHÔNG
  bắt buộc, đó là lãnh địa G4.

## 6. Cổng chuyển sang G4
- [ ] CMS retrofit chạy trên RESISC45, có ≥2 (chuẩn: 4+) cấu hình ablation.
- [ ] Bảng CMS vs baseline vs Titans + biểu đồ ‖Δw‖ chứng minh cơ chế.
- [ ] `docs/KET_LUAN_G3.md`: CMS giảm quên bao nhiêu, cấu hình nào tốt nhất, vì sao.
