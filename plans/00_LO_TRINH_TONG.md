# Lộ trình tổng — UAV Học Liên Tục (Nested Learning)

> File điều phối cho bộ plan chi tiết: `PLAN_G2_TITANS.md` → `PLAN_G6_BAOCAO_PAPER.md`.
> Đi kèm `../Team_Plan_3nguoi.md` (phân công gốc) và `../CAU_HOI_THEO_GIAI_DOAN.md`.

## Các quyết định ĐÃ CHỐT (2026-07-14)
| # | Quyết định | Lựa chọn | Hệ quả trong plan |
|---|---|---|---|
| 1 | Chuỗi cho Titans (G2) | **Xuyên task (cả stream)** | G2 đi theo bậc thang A→B→C (xem PLAN_G2), C là đích. Đây là điểm nhấn nghiên cứu mạnh nhất của dự án. |
| 2 | Backbone ở G2 | **Freeze** | Cô lập đóng góp Titans; so công bằng với NCM; fine-tune để dành cho G3 (CMS chính là fine-tune có kiểm soát). |
| 3 | Thời gian | **5+ tháng (~22 tuần)** | Làm đủ G2→G6, không cắt. Có 2 tuần đệm. |
| 4 | Compute | **GPU NVIDIA** | Full fine-tune ViT trên RESISC45 khả thi; batch/epochs theo config hiện tại. |
| 5 | Optimizer M3 (G4) | **Bắt buộc** | G4 kéo dài 5 tuần, N2 gánh thêm — kế hoạch đã tính. |
| 6 | Số seed (G5) | **3 seed** | Bảng cuối = mean±std trên seed {0,1,2}. |
| 7 | Dataset thứ 3 (G5) | **Thêm AID hoặc UCM** | Viết thêm source cho AID/UCM ở G5; chứng minh khái quát hoá. |
| 8 | Đầu ra (G6) | **Báo cáo + draft paper** | G6 = 3 tuần, viết song song từ G5. |

## Timeline (22 tuần, bắt đầu = tuần chốt số G1)
```
Tuần   0   1  2  3  4   5  6  7  8   9 10 11 12 13   14 15 16   17 18 19   20 21
       G1↘ [---- G2 ----] [---- G3 ----] [----- G4 -----] [-- G5 --] [-- G6 --] (đệm 2t)
       chốt số            CMS ⭐          HOPE + M3        3 seed     báo cáo
                                                           +AID/UCM   +paper
```
- G2 và tuần đầu G3 có thể chồng lấn (N2 viết CMS trong lúc N1 đo Titans) — plan gốc cho phép.
- Viết báo cáo bắt đầu NHÁP từ G5, G6 chỉ hoàn thiện.

## Tuần 0 — chốt sổ G1 (đang làm dở)
- [ ] Chạy xong 5 method × RESISC45 (EuroSAT đã xong) → `python scripts/compare_g1.py`.
- [ ] Đọc bảng cùng cả team, trả lời 2 câu: finetune quên bao nhiêu? NCM/replay đứng đâu?
- [ ] Ghi 1 trang `docs/KET_LUAN_G1.md` trong repo uav-continual-learning (N1 viết, mẫu trong PLAN_G2 §0).
- [ ] Tag git `v0.1-g1-baseline` — số baseline không được đổi từ đây (trừ khi sửa bug).
- **Cổng:** bảng ≥ 2 dataset × 5 method, finetune có forgetting rõ (nếu không → dừng, gọi Claude debug trước khi sang G2).

## Nguyên tắc xuyên suốt (nhắc lại từ Team_Plan)
1. Mọi model mới (Titans/CMS/HOPE) phải chạy qua **đúng harness G1** (`run_g1.py` + engine + metrics) — không viết vòng eval riêng.
2. **Không sang giai đoạn mới khi chưa qua cổng** ghi ở cuối mỗi file plan.
3. Mỗi thí nghiệm = 1 config yaml + 1 dòng lệnh, kết quả vào `artifacts/results/` — không có số "chạy tay không tái lập được".
4. Kết quả ÂM vẫn chấp nhận, miễn có phân tích vì sao (đã ghi trong từng plan phần "nếu kết quả xấu").

## Phân công vai (giữ nguyên từ Team_Plan)
- **N1** — hạ tầng thí nghiệm: chạy đo, biểu đồ, ablation runner, seed.
- **N2** — lõi bộ nhớ: Titans module (G2), CMS (G3), HOPE + M3 (G4), phân tích cơ chế (G5).
- **N3** — vision & tích hợp: adapter chuỗi (G2), gắn CMS vào ViT (G3), tối ưu (G4), robustness (G5).
