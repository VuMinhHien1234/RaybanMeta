# Kết luận G1 — Bảng baseline (seed 0, 2026-07-14)

> Nguồn số: `artifacts/results/*/metrics.json`, gộp bằng `scripts/compare_g1.py`.
> Dòng `synthetic` trong bảng là smoke test, không tính vào kết luận.

## Bảng số

**EuroSAT** (resnet18, 5 task × 2 class):

| Method | Avg Acc ↑ | Forgetting ↓ | BWT | Extra floats | Time (s) |
|---|---|---|---|---|---|
| finetune | 0.4213 | 0.2566 | −0.2566 | 0 | 434 |
| ewc | 0.4371 | 0.1902 | −0.1902 | 112M | 482 |
| lwf | 0.3739 | **0.6089 ⚠** | −0.6089 | 11M | 407 |
| **ncm** | **0.7486** | 0.1113 | −0.1113 | **5K** | **237** |
| replay | 0.6191 | **0.0027** | **+0.2253** | 2.5M | 463 |

**RESISC45** (ViT-S/16, 9 task × 5 class) — bảng chính:

| Method | Avg Acc ↑ | Forgetting ↓ | BWT | Extra floats | Time (s) |
|---|---|---|---|---|---|
| finetune | 0.4144 | **0.6133** | −0.6133 | 0 | 2 187 |
| ewc | 0.4259 | 0.5999 | −0.5999 | **390M (~1.5GB)** | **10 681 ⚠** |
| lwf | 0.5900 | 0.2777 | −0.2777 | 21.7M | 2 249 |
| ncm | 0.6933 | 0.1000 | −0.1000 | 17K | 837 |
| **replay** | **0.7937** | **0.0787** | −0.0718 | 135M | 3 164 |

## 6 phát hiện

1. **"Bệnh" được xác nhận.** Fine-tune thuần trên RESISC45 quên 61% (mất gần 2/3 kiến thức cũ sau 9 task). Bài toán của dự án có thật, stream đủ khó — thước đo hoạt động đúng.

2. **Replay là baseline mạnh nhất** (đúng dự đoán): RESISC45 Acc 0.794, Forgetting 0.079. Trên EuroSAT còn có **BWT dương** (+0.225) — học task mới làm task cũ *tốt lên*. Nhưng giá phải trả: lưu 135M floats ảnh thô (RAM + riêng tư — điều UAV thực tế khó chấp nhận).

3. **NCM: rẻ vô địch, mạnh bất ngờ** (đúng bài học từ project CPM): 0 tham số train, chạy nhanh nhất (837s vs 2–10k s), mà đứng **nhì** RESISC45 (0.693) và **nhất** EuroSAT (0.749). Mọi phương pháp NL sau này thua NCM là vô nghĩa.

4. **EWC gần như vô dụng và đắt nhất:** chỉ +0.01 Acc so với finetune trên RESISC45, nhưng tốn ~1.5GB anchor và **3 giờ** (5× finetune) vì penalty quét 8 anchor × 21.7M tham số mỗi bước. Giữ trong bảng làm bằng chứng "regularization cổ điển không đủ"; nếu muốn cứu: giảm số anchor (chỉ giữ K gần nhất) hoặc online-EWC — ưu tiên thấp.

5. **LwF bất thường trên EuroSAT** ⚠: Forgetting 0.609 — *tệ hơn cả finetune* (0.257), trong khi trên RESISC45 nó chạy tốt (0.278). Nghi vấn: task chỉ 2 class → distillation (λ=1.0, T=2) đè tín hiệu CE. Việc theo dõi (không chặn cổng): thử `lwf_lambda: 0.1–0.5` trên EuroSAT, 2 run là rõ.

6. **Khoảng trống cho Nested Learning đã lộ ra — đây là luận điểm G2/G3:** trên RESISC45, chênh lệch NCM (frozen, 0.693) ↔ replay (được chỉnh feature + ôn ảnh, 0.794) ≈ **10 điểm accuracy = giá trị của việc được thích nghi feature**. Đích của Titans (G2) và CMS (G3): **giành lấy (một phần) 10 điểm đó mà KHÔNG lưu ảnh thô** — CMS = fine-tune có kiểm soát đa tần số, Titans = mang ngữ cảnh xuyên task. Nếu đạt Acc ≥ ~0.72 với Forgetting ≤ 0.1 và extra floats ≪ 135M → thắng có ý nghĩa.

## Phán quyết cổng G1

**ĐẠT** — 5 baseline chạy ổn định trên 2 dataset, số tái lập (seed 0, config lưu kèm từng run).

Việc còn lại trước khi mở G2 (theo `plans/TASKS_G2_TITANS.md`):
- [x] P1 bảng số — file này
- [x] P2 kết luận — file này
- [ ] P3: `git tag v0.1-g1-baseline && git push --tags`
- [ ] P4: `python scripts/smoke_titans.py` chạy [OK] trên máy GPU
