# PLAN G5 — Đánh giá chặt: 3 seed + dataset thứ 3 (3 tuần)

**Quyết định đã chốt:** 3 seed {0,1,2}; thêm dataset aerial thứ 3 (**UCM hoặc AID**) để
chứng minh khái quát hoá. Mục tiêu: biến "số của seed 0" thành "kết luận đáng tin".

## 1. Ma trận chạy chính (N1 chủ trì)
- Model vào vòng chung kết (chọn từ G4, tối đa 6): finetune, replay (baseline mạnh nhất
  thường), ncm, CMS-best, HOPE, HOPE+M3. (EWC/LwF chỉ seed 0 nếu thiếu thời gian máy.)
- 3 dataset × ~6 model × 3 seed ≈ **54 run** — GPU NVIDIA + backbone freeze cho nhóm frozen
  → ước 3–6 ngày máy. Viết `scripts/run_matrix.py` (vòng for gọi run_g1 + resume: bỏ qua
  run đã có metrics.json) để bấm 1 lệnh chạy qua đêm.
- `compare_g1.py` nâng cấp: gộp theo (dataset, method) → **mean±std**; xuất bảng LaTeX/markdown.

## 2. Dataset thứ 3 (N1 + N3, tuần 1)
- **UCM (UC-Merced, 21 class, 2.100 ảnh)** — nhỏ, tải nhanh, 7 task × 3 class → chọn làm mặc
  định; AID (30 class, 10k ảnh) nếu muốn nặng hơn. Viết `_load_ucm` vào
  `src/uavcl/data/sources.py` theo đúng khuôn `DataSource` (nguồn HF hub hoặc link trực tiếp
  — xác minh link trước khi code). Config `configs/g5_ucm.yaml`.
- Chú ý: KHÔNG tinh chỉnh gì theo UCM — chạy y nguyên cấu hình tốt nhất từ RESISC45 để đo
  khái quát hoá thật.

## 3. Phân tích định tính + robustness (N3)
- Confusion matrix trước/sau khi học task mới: class cũ nào bị "ăn" bởi class mới giống nó.
- Robustness kiểu UAV không cần dataset mới: (a) **đảo thứ tự task** (3 order theo seed —
  đã có sẵn qua `shuffle_classes`); (b) test-time corruption (Gaussian blur/noise/fog bằng
  torchvision transforms) trên model đã train — mô phỏng điều kiện bay xấu.
- 4–6 hình case study (ảnh bị quên/không bị quên) cho báo cáo.

## 4. Phân tích cơ chế (N2)
- Từ log ‖Δw‖ per-tier (G3): tầng nào giữ gì — vẽ heatmap tier × task.
- Soi state Titans: norm state theo thời gian, có "trôi" khi qua task không.
- Ablation sâu còn thiếu từ G3/G4 (chu kỳ cực đoan, η=0 tuyệt đối...) — chỉ trên EuroSAT.
- Trả lời câu bắt buộc của cổng: **"vì sao nó hoạt động/không hoạt động"** (1–2 trang).

## 5. Tuần
- T1: run_matrix + UCM source + bắt đầu chạy đêm; N3 dựng script confusion/corruption.
- T2: chạy tiếp + N2 phân tích cơ chế; N1 vẽ toàn bộ biểu đồ (forgetting curve mean±std,
  bar chart per-dataset, heatmap).
- T3: chốt "danh sách phát hiện chính" (findings.md — mỗi phát hiện 1 câu + hình/bảng dẫn
  chứng); đóng băng số liệu, tag `v0.9-results`.

## 6. Cổng chuyển sang G6
- [ ] Bảng cuối mean±std trên 3 seed × 3 dataset, ổn định (std không nuốt chửng khác biệt).
- [ ] Bộ biểu đồ + case study + phân tích cơ chế đủ để VIẾT mà không cần chạy thêm gì.
- [ ] `docs/FINDINGS.md` — 5–8 phát hiện chính, mỗi cái có dẫn chứng.
