# PLAN G6 — Báo cáo + Draft paper (3 tuần, viết nháp từ G5)

**Quyết định đã chốt:** đầu ra = báo cáo đồ án ĐẦY ĐỦ + draft paper (workshop/arXiv).

## 1. Hai văn bản, một bộ số
Cùng dùng số liệu đóng băng ở tag `v0.9-results`. KHÔNG chạy lại thí nghiệm trong G6
(trừ bug làm sai số — khi đó phải ghi changelog).

### Báo cáo đồ án (dài, tiếng Việt) — khung chương
1. Giới thiệu: bài toán UAV học liên tục, catastrophic forgetting.
2. Nền tảng: continual learning, các baseline; Nested Learning/Titans/CMS/HOPE (N2 viết, đã có note từ G0–G4).
3. Phương pháp: harness class-incremental (N1); Titans-xuyên-task (G2); CMS retrofit (G3); HOPE + M3 (G4). Nhấn: các quyết định thiết kế + vì sao (lấy từ các file KET_LUAN_G*.md).
4. Thực nghiệm: 3 dataset, 3 seed, bảng trung tâm (PLAN_G4 §4 bản mean±std), ablation, cơ chế (‖Δw‖), robustness.
5. Phát hiện & bàn luận: từ FINDINGS.md — kể cả kết quả âm, TRUNG THỰC (bài học từ project CPM: định vị đúng đóng góp thì hội đồng không bẻ được).
6. Kết luận + hướng mở (detection/YOLO, memory dài hơn, on-device).

### Draft paper (ngắn, tiếng Anh, 6–8 trang) — luận điểm bán
- **Đóng góp:** (1) benchmark class-incremental trên ảnh trên không với 5 baseline trung thực
  (kèm NCM-frozen — baseline hay bị bỏ qua); (2) Titans memory XUYÊN TASK cho vision stream —
  điểm mới nhất, khoe FWT; (3) CMS retrofit lên ViT + bằng chứng cơ chế ‖Δw‖; (4) HOPE + M3
  trên bài toán thị giác. Title hướng: *"Nested Learning for Continual Aerial Scene
  Recognition: Cross-Task Neural Memory and Multi-Frequency Backbones"*.
- Chọn đích: workshop CL/remote-sensing (deadline nào gần thì N1 tra sau) hoặc arXiv trước.

## 2. Việc theo người & tuần
- **N1:** T1: chương Thực nghiệm (báo cáo) + section Experiments (paper) — bảng/hình đã có
  sẵn từ G5; T2: rà số liệu chéo (mỗi số trong văn bản phải trace được về 1 metrics.json).
- **N2:** T1–2: chương Phương pháp + related work phần NL; T3: rà thuật ngữ/công thức khớp NL.pdf.
- **N3:** T1: **phụ lục tái lập**: `REPRODUCE.md` + script one-click (env → data → run_matrix
  → bảng) chạy thử trên máy sạch; dọn repo (xoá code chết, pin versions trong requirements);
  T2: phần related work vision/CL + hình kiến trúc (vẽ lại 3 sơ đồ từ các plan); T3: video/demo
  ngắn nếu cần bảo vệ.
- **Cả team T3:** đọc chéo toàn văn; mock defense 1 buổi (mỗi người bị hỏi 5 câu — lấy từ
  `CAU_HOI_THEO_GIAI_DOAN.md` những câu đã trả lời bằng số).

## 3. Checklist chất lượng trước khi nộp
- [ ] Mỗi claim có số/hình dẫn chứng; mỗi số tái lập bằng 1 lệnh.
- [ ] Kết quả âm (nếu có) được nêu + giải thích, không giấu.
- [ ] Repo: tag `v1.0`, README hướng dẫn 15 phút chạy lại được smoke + 1 bảng nhỏ.
- [ ] Đạo văn/trích dẫn: NL.pdf, GEM metrics, EWC/LwF/replay, timm/titans-pytorch, obekt/kmccleary đủ cite.
- [ ] Slide bảo vệ: 15–20 trang, 1 slide "bảng trung tâm", 1 slide cơ chế ‖Δw‖, 1 slide hạn chế.

## 4. Xong = nộp
Báo cáo PDF + draft paper PDF + repo v1.0 + slide. Dự án đóng. 🎓
