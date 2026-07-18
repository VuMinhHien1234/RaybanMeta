# Logic triển khai Nested Learning trong dự án (bản tóm tắt kỹ thuật)

> Trả lời một câu: *các đoạn xử lý NL trong code này hoạt động theo logic gì, ở đâu, vì sao*.
> Dùng làm sườn cho chương "Phương pháp" của báo cáo. Đối chiếu công thức: NL.pdf.

## 0. Ý tưởng trung tâm — một câu

Paper NL nói: **mọi thành phần của hệ học (trọng số, optimizer, memory) đều là BỘ NHỚ,
chỉ khác nhau ở TẦN SỐ được ghi**. Quên thảm khốc xảy ra khi mọi thứ bị ghi cùng một nhịp.
Dự án này cài đúng tư tưởng đó thành **4 tầng nhớ lồng nhau**, mỗi tầng một nhịp ghi:

| Tầng (nhanh → chậm) | Là gì | Nhịp ghi | Công thức paper | File |
|---|---|---|---|---|
| 1. Titans memory | MLP "vừa chạy vừa tự ghi" | mỗi chunk ảnh, NGAY trong forward | §7 Titans, delta-rule | `models/memory.py`, `titans_head.py` |
| 2. Ký ức gradient (M3) | momentum = bộ nhớ của gradient | mỗi bước (nhanh) + mỗi f bước (chậm) | Algorithm 1 §7.2, Eq. 48–49 | `optim/m3.py` |
| 3. Trọng số MLP (CMS) | kiến thức trong ViT | tier nhanh mỗi bước / chậm mỗi 16 bước | Eq. 70–71 §7.1 | `optim/cms_optimizer.py`, `models/cms.py` |
| 4. Nền pretrained | patch_embed, pos_embed... | KHÔNG BAO GIỜ (đóng băng) | §7.3 retrofit | `models/cms.py` |

## 1. Tầng 1 — Titans memory (G2): học online bằng delta-rule

**Logic xử lý mỗi lần forward** (`titans_head.py`):
1. Ảnh → ViT (frozen ở G2) → feature; `SeqAdapter` xếp B ảnh trong batch thành CHUỖI
   thời gian (1, B, D) — "mỗi ảnh = một lần UAV nhìn".
2. `NeuralMemory` đọc chuỗi theo chunk; với mỗi chunk nó **tự đoán trước** (M·k), đo
   **surprise** = giá trị thật − đoán, rồi **chỉ ghi phần surprise** vào trọng số ký ức:
   `M ← α·M + η·update(surprise)` — α, η ở tầng này là **hàm học được theo dữ liệu**
   (thư viện titans-pytorch lo). Không có `optimizer.step()` nào ở đây — ghi xảy ra
   NGAY trong forward, cả lúc train lẫn lúc gặp dữ liệu mới.
3. `state` (gói {trọng số ký ức + đà}) được trả về và truyền sang forward sau.

**Vòng đời state — 3 chế độ, 2 luật an toàn** (một dòng yaml `memory.reset`):
- `image`: không giữ state giữa các forward (sanity, bậc A) · `task`: giữ trong task,
  xoá khi sang task mới (bậc B) · `never`: sống xuyên task — **đích của dự án** (bậc C).
- Luật 1 — truncated BPTT: state lưu lại LUÔN bị `detach` (cắt gradient, GIỮ giá trị) —
  không thì backward lần 2 nổ + RAM tràn. Chú ý kỹ thuật: state chứa `TensorDict`,
  phải detach xuyên qua kiểu dữ liệu này (`state_utils._tree_map`).
- Luật 2 — "chấm thi không ghi trí nhớ": khi eval, forward xuất phát từ **bản clone**
  của state và không lưu state mới → đo bao nhiêu lần cũng ra một kết quả.

## 2. Tầng 2 — M3 + Delta Momentum: optimizer cũng là bộ nhớ

**Logic mỗi `step()`** (`optim/m3.py`, bám Algorithm 1):
1. Ký ức NHANH M¹ ăn gradient mỗi bước; ký ức CHẬM M² chỉ ăn mỗi `f=16` bước
   (gộp cả chunk gradient — chính là CMS áp vào optimizer).
2. Cập nhật theo **Delta Momentum (Eq. 48–49)**: `M ← α·M + η·(g − M)` — chỉ ghi phần
   gradient LỆCH so với cái đã nhớ; **cổng quên α tách riêng tốc độ ghi η**, mỗi tầng
   một cặp (α, η) riêng (đúng chuẩn per-level của paper). α=1 thì trùng EMA (có test chứng minh).
3. Ma trận: trực giao hoá **Newton–Schulz** kiểu Muon rồi cộng 2 tầng `O¹ + α·O²`,
   chia `√V̂` kiểu Adam (V có bias-correction).
4. **Quy ước Muon (học từ debug thật)**: NS + ký ức chậm CHỈ áp cho tham số ma trận;
   bias/norm 1D rơi về update Adam thuần — áp nguyên M3 lên vector gây limit-cycle.

## 3. Tầng 3 — CMS trên trọng số (G3): đa tần số bằng optimizer wrapper

**Logic (Eq. 71, không mổ forward của ViT)** (`cms_optimizer.py` + `models/cms.py`):
1. 12 block MLP của ViT chia 3 tier `[[4,1],[4,4],[4,16]]` theo `order`
   (late_slow: block cuối bền / early_slow: block đầu bền — ablation bắt buộc);
   attention+norm vào tier chậm nhất; head + Titans memory vào tier nhanh nhất;
   nền pretrained đóng băng.
2. Mỗi batch: gradient MỌI tier được tích lũy; tier chỉ được inner-optimizer (M3)
   bước khi `step % chu_kỳ == 0`, với gradient **GỘP TỔNG** của cả cửa sổ
   (`grad_agg: sum` — nguyên văn Eq. 71; `mean` là biến thể ổn định để ablate);
   η per-tier scale lr (tier chậm học rất khẽ).
3. Hệ quả chống quên: học task mới, "mực" đổ chủ yếu vào tier nhanh; tier chậm
   gần bất động → kiến thức cũ sống sót. Kiểm chứng bằng log `[cms] ‖Δw‖` mỗi task.
4. `f(·)` trong Eq. 71 là "error của optimizer BẤT KỲ" → CMS bọc được cả M3 lẫn AdamW
   — đúng câu chữ paper, và cho bảng 2×2 tách công kiến trúc/optimizer.

## 4. Tầng 4 — HOPE (G4): ghép tất cả thành một phổ tần số

**Luồng xử lý MỘT batch khi chạy HOPE** (`models/hope.py`):
```
ảnh → ViT (mở băng, nhịp do CMS quyết)     ← tầng trọng số: nhanh/vừa/chậm/đóng băng
    → SeqAdapter → TitansMemory (state xuyên task)   ← tầng nhanh NHẤT, ghi trong forward
    → post_norm + residual → head → logits
loss = CE(logits mask theo giao thức class-incremental)
backward → CMSOptimizer tick chu kỳ từng tier → M3 (delta momentum) bước các tier đến hạn
state ← detach rồi lưu (mang sang batch/task sau)
```
Rủi ro đặc thù đã dự phòng: backbone thay đổi → không gian feature TRÔI dưới chân
memory ("feature drift") — theo dõi bằng `[hope] norm(state)` mỗi task.

## 5. Sai khác CÓ CHỦ ĐÍCH so với paper (phần "trung thực" của báo cáo)

| # | Sai khác | Lý do |
|---|---|---|
| 1 | Delta Momentum dùng key hằng (không key-ma-trận đầy đủ) | bản đầy đủ cần momentum dim² cho TỪNG tham số — bất khả thi cho optimizer |
| 2 | State detach sau MỖI batch (TBPTT) | gradient xuyên nhiều batch → double-backward + nổ RAM; ký ức GIÁ TRỊ vẫn nối dài |
| 3 | Tham số 1D dùng Adam, không M3 | quy ước Muon; đo được limit-cycle nếu làm ngược |
| 4 | HOPE ghép ở feature-level (không token-level như §8) | tái dùng 2 mảnh đã kiểm chứng, quy được trách nhiệm khi đọc số |
| 5 | α/η của CMS & M3 là HẰNG SỐ per-tier (chưa data-dependent) | đúng mức Algorithm 1; bản tự-tính (η adaptive) là bước sau, để tách bạch biến số |
| 6 | `grad_agg: mean` tồn tại cạnh `sum` | sum = nguyên văn; mean giữ ngữ nghĩa lr đồng nhất giữa các chu kỳ — ablate được |

### Cập nhật 07-18 — đọc số G4 --quick xấu (HOPE Forgetting 0.956), thu hẹp 3 khoảng cách trên

Xem `docs/TIEN_DO_2026-07-18.md` cho đầy đủ. Tóm tắt: #1 và #5 KHÔNG còn "chưa làm" mà đã có
bản cài, tắt mặc định (opt-in) — #4 vẫn giữ nguyên (feature-level), nhưng bệnh feature-drift
của nó được giảm nhẹ bằng cờ có sẵn trong thư viện, không phải sửa kiến trúc.
- **#1 (key hằng số)** — thêm `m3.key_proj_eta` (mặc định 0.0): xấp xỉ RANK-1 (không phải
  ma trận đầy đủ, vẫn bất khả thi) của P_i — trừ đúng phần `m1` nằm dọc hướng gradient hiện
  tại, "quên có chọn lọc theo hướng" thay vì đều mọi hướng. `src/uavcl/optim/m3.py`.
- **#5 (η hằng số)** — `cms.eta_mode: adaptive` (S9) nay đã CÀI xong (trước đây chỉ có trong
  kế hoạch): η_tier = η_base×(1−cos(surprise)). `src/uavcl/optim/cms_optimizer.py`.
- **Mảnh MỚI, không nằm trong 6 sai khác gốc**: titans-pytorch (0.5.5, đã cài sẵn) có 4 cờ
  ổn định đúng tinh thần "tự điều chỉnh theo ngữ cảnh" của §8 mà project chưa bật:
  `gated_transition` (cổng học được quyết định ghi đè bao nhiêu %), `spectral_norm_surprises`
  (chuẩn hoá Newton-Schulz cho surprise trước khi ghi — cùng họ `update_norm=rms` đã cứu M3),
  `qk_rmsnorm` (đọc/ghi bền hơn khi feature trôi), `max_grad_norm`. Đã bật cả 4 trong
  `configs/g4_hope_*.yaml` và `configs/g2_titans_*.yaml`. `src/uavcl/models/titans_head.py`.
- **Đã dò, KHÔNG khả thi trong đợt này**: tự sinh giá trị mục tiêu v̂ kiểu self-modifying
  Titans (§8.1) — titans_pytorch 0.5.5 không có hook, `to_values` là linear projection cố
  định. Cần fork/viết lại `NeuralMemory`, không phải việc chỉnh config — để future work.
- ⚠ Toàn bộ trên mới qua rà soát code + unit test viết tay (sandbox không có torch để chạy
  thật) — bắt buộc `pytest -q` trên máy/VM có torch trước khi tin số, rồi mới train lại.

## 6. Bằng chứng cơ chế = 3 hệ log (đọc mỗi lần chạy)

1. `[titans]/[hope] norm(state)` — ký ức có phình/nổ không, feature drift có xảy ra không.
2. `[cms] ‖Δw‖ per-tier` — tier chậm phải ≈ 0; ngược lại là mapping/chu kỳ sai, DỪNG đọc accuracy.
3. `FWT` (bật `eval_future`) — memory xuyên task có giúp cả task CHƯA HỌC không; đây là
   metric mà NCM/EWC/replay không bao giờ cải thiện được → điểm khoe độc quyền của NL.

**Tiêu chí thành công cuối cùng** (từ bảng G1): Acc ≥ ~0.72 & Forgetting ≤ 0.1 trên
RESISC45 với bộ nhớ thêm ≪ 135M floats và KHÔNG lưu ảnh thô — tức giành phần điểm của
replay bằng trí nhớ cấu trúc thay vì bằng kho ảnh.
