# Nhận xét 2026-08-01 (bản 2, sau `git fetch`) — Optimizer M3 mới & NCM-head giảm đọc data cũ

> Đính chính bản 1: so sánh working-tree hai folder thì m3.py giống nhau (RB2 checkout ở `main` cũ 18-07).
> Sau khi fetch, trên remote lộ các branch team — và **m3.py CÓ bản mới thật** (hash `aa854262`, nằm trên
> `Feat-M3-An` → `Titan_M3` → `NCM_Head`), khác bản dự án (`897071c0`). Kèm kết quả đo của An ngày 07-26 & 07-31
> commit ngay trong branch (`An/…`). Bản 1 sai ở kết luận "giống hoàn toàn" — đúng với local, sai với remote.

## 1. Bản M3 mới khác gì bản dự án

Khác biệt cốt lõi nằm ở **cách giới hạn bước cập nhật** (`update_norm`): bản dự án mặc định `rms` — ép RMS của
bước ma trận về 1 rồi nhân lr, nghĩa là bước của tensor lớn phình theo √(số phần tử) và update nhỏ bị **khuếch
đại lên**; bản mới mặc định `clip` — chỉ **co** update khi Frobenius norm > 1 (`step/max(‖step‖,1)`), không bao
giờ khuếch đại, khớp implementation tham khảo. `rms` giữ lại làm chế độ legacy để tái lập run cũ. Ngoài ra bản
mới thêm guard NaN/Inf cho gradient lẫn bước cập nhật (fail sớm thay vì lặng lẽ hỏng), validate đủ siêu tham số,
sửa docstring cho trung thực ("delta" của dự án không phải Delta Momentum Eq. 48–49 đầy đủ), và
`cms_optimizer.py` sửa một bug tinh vi: khi M3 chạy trong CMS, tần số f của ký ức chậm được quy đổi per-tier
(`cms.m3_frequency_unit`) để hai lịch thưa không nhân nhau (trước đây tier p=64 × f=16 → 1024 batch mới có M2).
Recipe "improved" đi kèm còn đổi `optimizer_per_task: false` (giữ ký ức optimizer xuyên task), `lr: 5e-3`
(clip làm bước nhỏ hơn nên cần lr lớn hơn — sweep của An xác nhận lr ≤ 3e-4 sập), và `grad_clip_norm: 1.0`.

**Đã port vào dự án (branch `memory_titan_task4_v2`, working tree):** `optim/m3.py`, `optim/__init__.py`,
`optim/cms_optimizer.py`, `tests/test_m3.py` (chép nguyên từ `origin/NCM_Head`, md5 khớp 100%), 2 config A/B
`g2_titans_resisc45_selfmod_m3_improved.yaml` / `_legacy.yaml`, và thêm ~8 dòng hỗ trợ `train.grad_clip_norm`
vào `engine.py` (không đặt cờ → hành vi cũ bất biến; không đụng phần SDC/CosineHead mới thêm hôm 01-08).
Lưu ý hệ quả: mặc định mới là `clip` — run M3 nào muốn tái lập số cũ phải set `m3.update_norm: rms`.
Chưa port: engine NCM mới + `ncm_adaptation.py` + `checkpoint.py` + 4 script study của branch (đụng nặng vào
engine.py sẽ conflict với SDC/CosineHead vừa viết — nên quyết sau khi đọc mục 2).

## 2. M3 mới có đem lại kết quả không? — CÓ, rõ rệt

Số của An (RESISC45, ViT-S, 9 task, Titans, split mới `combined31500_v2`, M3 clip lr5e-3 optkeep), so cùng bảng:

| Head đọc kết quả | seed 0 | seed 1 | seed 2 | Forget |
|---|---|---|---|---|
| Linear (M3 **clip**) | 0.609 | 0.611 | 0.626 | **≈ 0 (−0.02…−0.05)** |
| Linear (M3 **rms** cũ, split cũ — tham khảo) | 0.608 | 0.241 (nổ) | 0.226 (nổ) | 0.26 / 0.57 / 0.72 |

Bản rms cũ sập 2/3 seed vì nổ norm; bản clip **ổn định cả 3 seed, forgetting về ≈0**. Ablation seed 1 của An tách
đóng góp: clip-only 0.557/F0.20 → clip+optkeep 0.611/F−0.02 — cả hai thành phần đều cần. (So chéo hai split là
tham khảo; nhưng kết luận "clip ổn định qua seed" đứng vững ngay trong bảng của An.) Vụ "M3 gây nổ norm ở một số
seed → không dùng làm bản chốt" trong HUONG_DAN_CHAY.md giờ có ứng viên sửa: **M3-clip đưa M3 quay lại cuộc đua
với AdamW**.

## 3. NCM-head "giảm phụ thuộc đọc data cũ" có đem lại kết quả không? — MỘT NỬA

Cùng study của An (M3-clip, 3 seed, mốc baseline **NCM frozen cùng split = 0.7114 / F 0.085**):

| Readout | seed 0 | seed 1 | seed 2 | Forget | Đọc lại data cũ? |
|---|---|---|---|---|---|
| NCM **posthoc rebuild** | **0.745** | **0.739** | **0.754** | 0.06–0.07 | CÓ (quét mọi task cũ) |
| NCM **online** (chỉ data task hiện tại) | 0.614 | 0.607 | 0.636 | 0.25–0.28 | KHÔNG |

Phần đã chứng minh: NCM-head rebuild giờ **vượt baseline 0.7114 trên cả 3 seed** — điều bản M3-rms không làm được
(seed 1–2 tụt còn 0.61–0.62). Phát hiện "head Linear là nút thắt" đứng vững và giờ robust qua seed.
Phần CHƯA đạt: bản **không đọc data cũ** (NCM online thuần) mất ~12–14 điểm so với rebuild và thua cả NCM frozen —
tự nó chưa dùng được. Các cơ chế bù đắp thông minh hơn thì **chưa có số**: prototype-transport/feature-blending
(`ncm_adaptation.py` của branch — có config, không thấy artifacts kết quả) và SDC + CosineHead (nhánh chính, cài
01-08, mới có unit test). Vậy: câu "thay NCM head để bớt đọc data cũ" hiện mới đem lại kết quả ở dạng *đắt*
(rebuild); dạng *rẻ* đang là khoảng trống đo đạc — chính là thí nghiệm đáng chạy nhất tiếp theo.

## 4. Việc cần làm

Trên Mac/VM (sandbox không cài được torch): (1) `pytest -q` — test_m3.py mới phải xanh với code vừa port;
(2) A/B `configs/g2_titans_resisc45_selfmod_m3_improved.yaml` vs `_legacy.yaml` trên split hiện hành của dự án
để xác nhận lại clip-vs-rms ngoài môi trường của An; (3) đo nhóm "không đọc data cũ" trên cùng M3-clip:
`ncm_head_mode=sdc` vs `rebuild` (tách `log.dir`!), `head=cosine` vs `linear`, và nếu lấy engine mới của branch
thì thêm transport của `ncm_adaptation.py` — mục tiêu: đóng khoảng cách 0.61→0.74 mà không quét lại data cũ;
(4) merge branch `NCM_Head` vào `memory_titan_task4_v2` sẽ conflict ở engine.py (881 dòng đổi, đè lên SDC/CosineHead
01-08) — nên hợp nhất có chủ đích thay vì merge mù.

*Kiểm chứng phiên này: mọi file port md5-khớp `origin/NCM_Head`; py_compile sạch (m3, __init__, cms_optimizer,
test_m3, engine); mọi con số đọc trực tiếp từ metrics*.json trong repo/branch (An/NCM_Head_results_2026-07-31,
An/Titan_M3_results_2026-07-26, result_test/res1–3).*
