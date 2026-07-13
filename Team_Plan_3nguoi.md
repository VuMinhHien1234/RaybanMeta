# Kế hoạch phân công team 3 người — UAV Học Liên Tục (Nested Learning / Titans / HOPE)

> Đi kèm `UAV_NestedLearning_Roadmap.md`. Chia nhiệm vụ theo 7 giai đoạn (G0–G6), mỗi giai đoạn có: mục tiêu, phân công 3 người, **kết quả cần đạt**, và **cổng chuyển giai đoạn** (điều kiện để sang giai đoạn sau).

## Giả định
- Tác vụ mặc định: **phân loại ảnh trên không theo class-incremental** (thêm lớp dần). Nếu chuyển sang **object detection** → đổi backbone sang YOLO, metric thêm mAP; cấu trúc phân công không đổi.
- Hạ tầng: ≥1 GPU dùng chung, 1 repo git chung, quản lý experiment (W&B hoặc JSON logs).

## Vai trò 3 người (xuyên suốt dự án)
- **N1 — Data & Khung học liên tục:** dataset, loader stream nhiều task, metrics (Forgetting/BWT/FWT), baseline CL, chạy thí nghiệm & biểu đồ. *(Vai trò "hạ tầng thí nghiệm".)*
- **N2 — Kiến trúc bộ nhớ (Titans + CMS):** phần lõi & khó nhất — module bộ nhớ Titans, retrofit CMS, ghép HOPE, optimizer. *(Nên là người mạnh ML nhất.)*
- **N3 — Backbone thị giác & Tích hợp:** backbone pretrained, trích đặc trưng, adapter 2D→chuỗi, head tác vụ, ghép bộ nhớ vào ống thị giác, tối ưu tốc độ.

## Bảng tổng quan (ai chủ trì gì)
| Giai đoạn | N1 (Data/CL) | N2 (Bộ nhớ) | N3 (Vision) |
|---|---|---|---|
| G0 Nền tảng | Repo + môi trường chung, shortlist dataset | Chạy Titans toy, đọc CMS | Backbone + trích feature |
| G1 Baseline | **Chủ trì**: loader + metrics | Baseline EWC/replay/LwF | Backbone + head + vòng train |
| G2 Titans | Chạy thí nghiệm + đo | **Chủ trì**: module bộ nhớ | Adapter feature→chuỗi |
| G3 CMS ⭐ | Continual-adapt + ablation | **Chủ trì**: module CMS | Gắn CMS vào backbone thật |
| G4 HOPE | Chạy so sánh đầy đủ | **Chủ trì**: ghép HOPE + optimizer | Tích hợp + tối ưu |
| G5 Đánh giá | **Chủ trì**: benchmark đa seed | Phân tích ablation/cơ chế | Phân tích định tính/robustness |
| G6 Báo cáo | Phần thí nghiệm | Phần phương pháp | Phụ lục tái lập + dọn repo |

---

## G0 — Nền tảng (1–2 tuần)
**Mục tiêu:** môi trường chạy được + hiểu code tham khảo. *Chưa có gì UAV.*

- **N1:** dựng repo chung (cấu trúc thư mục, git, conda/requirements chuẩn cho cả team, CI nhẹ chạy test); khảo sát & lập shortlist dataset UAV (VD: remote-sensing scene classification, VisDrone…).
- **N2:** cài `titans-pytorch`, chạy toy `NeuralMemory`/MAC (forward+backward); đọc `obekt/HOPE-nested-learning` và `kmccleary3301/nested_learning`, viết **note cơ chế CMS/Titans** (tier–chu kỳ, update rule) cho team.
- **N3:** cài backbone pretrained (timm ViT hoặc DINOv2), viết script trích đặc trưng từ ảnh mẫu, xác nhận pipeline GPU.

**Kết quả cần đạt:** repo chung + môi trường tái lập; 3 "hello-world" chạy (bộ nhớ toy, CMS đọc-hiểu, backbone trích feature); 1 doc ngắn "cơ chế CMS/Titans"; shortlist dataset.
**Cổng chuyển:** cả 3 chạy được môi trường như nhau; chốt được dataset.

## G1 — Baseline & Khung đo (1–2 tuần)
**Mục tiêu:** dựng "cây thước đo" + bảng baseline.

- **N1 (chủ trì):** chốt dataset & tách **task stream** (task1→…→taskN); viết **DataLoader stream nhiều giai đoạn**; viết **metrics** (Average Accuracy, Forgetting/Backward Transfer, Forward Transfer); tích hợp thư viện Avalanche nếu dùng.
- **N2:** cài các **baseline CL kinh điển**: EWC, replay buffer, LwF/knowledge-distillation (qua Avalanche hoặc tự viết), đảm bảo tái lập.
- **N3:** chốt backbone + **head phân loại**; **vòng lặp train/eval per-task**; chạy **fine-tune tuần tự (naive)** để lấy mức quên "đáy".

**Kết quả cần đạt:** bảng baseline (naive/EWC/replay/LwF) với Avg Acc + Forgetting; harness tái sử dụng cho mọi giai đoạn sau; seed cố định, số liệu tái lập.
**Cổng chuyển:** có ít nhất 3 baseline chạy ổn định + bảng số so sánh được.

## G2 — Tích hợp bộ nhớ Titans (2–4 tuần)
**Mục tiêu:** lần đầu đưa ý tưởng paper vào pipeline UAV; lấy tín hiệu đầu tiên.

- **N2 (chủ trì):** tái lập/hiểu `NeuralMemory` ở toy; đóng gói thành **module dùng được** (API rõ: chuỗi vào → chuỗi ra + trạng thái).
- **N3:** viết **adapter feature→chuỗi** (chốt "trục thời gian": patch / frame theo thời gian / task stream); gắn `NeuralMemory` + head; nối vào harness G1.
- **N1:** chạy thí nghiệm Titans qua harness; đo vs baseline; quản lý experiment tracking + biểu đồ.

**Kết quả cần đạt:** model "backbone + Titans" chạy end-to-end trên stream UAV; **bảng số Titans vs baseline**; kết luận sơ bộ (giúp/không giúp, ở đâu).
**Cổng chuyển:** pipeline Titans chạy end-to-end + có số so với baseline (kể cả kết quả âm cũng chấp nhận, miễn giải thích được).

## G3 — Retrofit CMS vào backbone (2–4 tuần) ⭐ mấu chốt
**Mục tiêu:** gắn bộ nhớ đa tần số (CMS) lên backbone — mảnh chống quên cốt lõi.

- **N2 (chủ trì):** viết **module CMS** (chia tầng + chu kỳ update + gộp gradient giữa các lần), learning-rate nội bộ η per-tier; phỏng theo obekt (`cms_tiers=[[n,period]...]`) và kmccleary.
- **N3:** **tích hợp CMS vào backbone thật** (timm ViT): gom khối MLP thành tầng tần số, để attention/norm ở tầng chậm giai đoạn đầu; đảm bảo forward/inference ổn.
- **N1:** **continual-adapt nhẹ** trên stream; đo forgetting; **ablation** số tầng / chu kỳ / η; log xác nhận tầng chậm gần như bất động.

**Kết quả cần đạt:** backbone-CMS chạy; **bảng CMS vs baseline vs Titans-only**; ablation tầng tần số; bằng chứng CMS giảm quên (hoặc không, có lý giải).
**Cổng chuyển:** CMS retrofit chạy + có ablation ≥2 cấu hình tần số.

## G4 — Ghép thành HOPE (2–4 tuần)
**Mục tiêu:** kết hợp Titans (thích nghi nhanh) + CMS (bền, chống quên).

- **N2 (chủ trì):** ghép self-modifying Titans + CMS thành khối **HOPE**; (tùy chọn) thay optimizer **M3 / Delta Momentum**.
- **N3:** tích hợp HOPE vào pipeline UAV; **tối ưu tốc độ/bộ nhớ** (chunk-wise).
- **N1:** chạy HOPE qua harness; **so sánh đầy đủ**; ablation bật/tắt từng thành phần (Titans-only / CMS-only / HOPE).

**Kết quả cần đạt:** HOPE-UAV chạy end-to-end; **bảng so sánh đầy đủ** (baseline / Titans / CMS / HOPE); ablation thành phần.
**Cổng chuyển:** HOPE vượt (hoặc ngang, có phân tích) baseline mạnh nhất trên forgetting.

## G5 — Đánh giá chặt (1–2 tuần)
**Mục tiêu:** kết quả đáng tin, đủ để kết luận.

- **N1 (chủ trì):** chạy toàn bộ benchmark **nhiều seed**, thống kê mean±std; biểu đồ forgetting/accuracy.
- **N3:** phân tích **định tính** (case tốt/xấu) + **robustness** (domain shift: ngày↔đêm, đô thị↔nông thôn).
- **N2:** phân tích **ablation sâu** + giải thích cơ chế (tầng nào giữ kiến thức gì).

**Kết quả cần đạt:** bộ kết quả cuối + biểu đồ; bảng ablation hoàn chỉnh; danh sách phát hiện chính.
**Cổng chuyển:** kết quả ổn định qua nhiều seed; hiểu được *vì sao* nó hoạt động/không.

## G6 — Tổng hợp & Báo cáo
- **N1:** viết phần thí nghiệm/kết quả, gom số liệu.
- **N2:** viết phần phương pháp (NL/Titans/CMS/HOPE cho UAV).
- **N3:** viết phần liên quan + **phụ lục tái lập**; dọn repo, README, script one-click tái lập.

**Kết quả cần đạt:** báo cáo/notebook tái lập; repo sạch có hướng dẫn; (tùy chọn) bản nháp paper.

---

## Nguyên tắc phối hợp
- **Harness của N1 là "khuôn" chung:** mọi model (Titans/CMS/HOPE) phải cắm vừa cùng loader + metrics để so sánh công bằng.
- **API rõ ràng giữa 3 mảng:** backbone→feature (N3), feature→bộ nhớ (N2), bộ nhớ→đo (N1). Chốt interface sớm ở G1–G2 để làm song song.
- **Không sang giai đoạn mới khi chưa qua "cổng chuyển".**
- **N2 là nút cổ chai** (giữ phần khó nhất) → N3 co-own tích hợp ở G2–G4 để chia tải; N1 giữ nhịp thí nghiệm.

## Rủi ro chính & người canh
- *Tích hợp bộ nhớ vào vision khó hơn dự kiến* (N2+N3): bắt đầu bản CMS đa-tần-số-optimizer (dễ) trước bản self-modifying (khó).
- *Baseline/metrics sai → so sánh vô nghĩa* (N1): kiểm bằng seed cố định + đối chiếu số công bố.
- *Phạm vi phình to* (cả team): bám 1 dataset + 1 tác vụ tới khi có tín hiệu dương.

## Mốc thời gian (ước lượng, có thể chồng lấn)
G0 (1–2t) → G1 (1–2t) → G2 (2–4t) → G3 (2–4t) → G4 (2–4t) → G5 (1–2t) → G6 (1t). Tổng ~10–19 tuần. G2 và một phần G3 có thể chạy song song nếu interface đã chốt ở G1.
