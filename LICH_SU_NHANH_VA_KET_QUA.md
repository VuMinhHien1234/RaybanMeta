# Lịch sử nhánh Git và kết quả từng nhánh

Repo `VuMinhHien1234/RaybanMeta` · lập ngày 2026-08-03

Mỗi mục ghi rõ **mức bằng chứng**, vì không phải nhánh nào cũng có số đo:

| Ký hiệu | Nghĩa |
|---|---|
| 📊 | Có số đo 3 seed, đã kiểm chéo, tin được |
| 📝 | Chỉ có commit message — chưa tìm thấy log kết quả tương ứng |
| ⏳ | Đang chạy, chưa có kết quả |

---

## 1. Repo tách làm HAI dòng, không gộp lại

```
                              (lịch sử cũ: CLAM / TCGA — không liên quan UAV)
                                        │
                                    24b362b
                                   ╱        ╲
                    ┌─────────────╱          ╲──────────────┐
                    │  DÒNG A: main            DÒNG B: memory_*
                    │  (07-13 → 07-18)         (07-22 → nay)
                    │                          │
              95711f5 First commit        925d6bf Task 3
              f9521f8 plans g2            54f2d38 memory_task_titans
              070da1a/c7ff0eb g4          9bd1888 bỏ heads=2
              a307307 M3 update              │
              568e04d fix 0718 ⭐             ├─ task4-self-modifying-value
              e3ad69c config                 │   a76f154 Task 4: v=f(x,M)
              4e75683 cms, titan memory      │   26d17ed richer state summary
              5c51b6b Completed              │
                    │                        └─ memory_titan_task4_v2
              ┌─────┴─────┐                      fd5b666 → 96114dd → fc1c09b
        136bdbb          │                       bd0dde8 → 77d1035  📊
        4b9a736 Feat-M3-An                            │
        7367cc5 M3-origin                       2311e75 fix/nl-gates-alive  📊
              │                                       │
        Titan_M3 ──→ NCM_Head                   0974b9b feat/slda-ablation  📊
        (7750bb1 … 49b084e)                           │
                                                8d8e436 ┐
                                                1cb40ed ├ feat/drift-lambda  ⏳
                                                d0e0927 ┘  ← ĐANG Ở ĐÂY
```

**Điểm quan trọng nhất của sơ đồ này:** hai dòng tách nhau ở `24b362b` và **xây `src/uavcl/`
độc lập với nhau**. `main` không phải tổ tiên của nhánh đang làm việc — 6 commit 07-13→07-18
của `main` (2.550 dòng, gồm `cms_optimizer.py`, `hope.py`, `state_utils.py`, 4 cờ ổn định
titans-pytorch) **chưa bao giờ có mặt** trong `feat/drift-lambda`.

Ngược lại, nhánh làm việc hiện có nhiều code hơn `main` **4.684 dòng**. Nên đây không phải
"quên pull main" — mà là hai bản cài đặt song song của cùng một package. Gộp lại sẽ không
tự động được, phải đối chiếu tay từng file. Xem mục 5.

---

## 2. Bảng tổng — nhánh nào cho kết quả gì

| Nhánh | Ngày | Đóng góp chính | Kết quả đo được | BC |
|---|---|---|---|:-:|
| `main` | 07-13→07-18 | Dựng pipeline, CMS, HOPE, M3, 4 cờ ổn định titans-pytorch | không tìm thấy log | 📝 |
| `Feat-M3-An` | 07-22→07-25 | Optimizer M3 | không tìm thấy log | 📝 |
| `M3-origin` | 07-26 | So M3 hiện tại với bản trong paper | không tìm thấy log | 📝 |
| `memory_task_titans` | 07-22 | Task 3 — bọc projection version-agnostic; bỏ `heads=2` | không tìm thấy log | 📝 |
| `task4-self-modifying-value` | 07-23 | Task 4 — `v = f(x, M_{t−1})` (self-modifying value) | không tìm thấy log | 📝 |
| `Titan_M3` | 07-26→07-30 | Tự động hoá nghiên cứu so sánh Titan-M3; xử lý lỗi số học khi resume | không tìm thấy log | 📝 |
| `NCM_Head` | 08-01 | Cải tiến NCM head | không tìm thấy log | 📝 |
| **`memory_titan_task4_v2`** | 08-01 | SLDA (#21), latent replay (#22), checkpoint (#23), AAA (#24), open-set (#27), bench (#28) | **Báo cáo 08-01** — xem mục 3.1 | 📊 |
| **`fix/nl-gates-alive`** | 08-02 | Vá 3 lỗi Nested Learning: α đọc ngược, η log logit thô, cổng trôi ra biên | **Cổng hồi sinh** — mục 3.2 | 📊 |
| **`feat/slda-ablation`** | 08-02 | Ablation `cov_mode` (B2) + đo chi phí triển khai (B6) | **+13,81 điểm là do Σ** — mục 3.3 | 📊 |
| **`feat/drift-lambda`** | 08-03 | Hệ số quên λ + dataset trôi domain-incremental | đang chạy 15 run | ⏳ |

---

## 3. Chi tiết ba nhánh có số đo

### 3.1 `memory_titan_task4_v2` — báo cáo Titans/NCM không replay

Nguồn: `BAO_CAO_KET_QUA_TITANS_NCM_NO_REPLAY_2026-08-01.md`, 3 seed.

| Phương pháp | Accuracy | Forgetting | Vai trò |
|---|---:|---:|---|
| Frozen ViT + NCM | 0,7114 ± 0,0000 | 0,0874 | Baseline chính |
| Titans + NCM Online (cũ) | 0,6190 ± 0,0147 | 0,2671 | −9,24 điểm |
| Anchored Blend γ=0,25 | **0,7196 ± 0,0045** | 0,0913 | +0,81 điểm — nhưng là *test-hindsight* |
| Anchored Blend γ=1,0 | 0,6166 ± 0,0185 | 0,2689 | ⚠️ γ này do **validation chọn** |
| Tổ hợp validation chọn | 0,6898 ± 0,0100 | 0,1711 | **−2,16 điểm** — kết quả chính thức |
| Titans + NCM Post-hoc | 0,7459 ± 0,0075 | 0,0656 | Oracle, đọc lại train cũ — không triển khai được |

**Bài học lớn nhất từ nhánh này:** γ=0,25 tốt hơn baseline, nhưng validation trên task hiện
tại lại chọn γ=1,0 — cấu hình **tệ nhất**. Cùng một cơ chế sau này được nhận ra ở cổng α:
tối ưu theo loss của task hiện tại thì *quên* luôn là phần thưởng. Hai hiện tượng, một
nguyên nhân.

### 3.2 `fix/nl-gates-alive` — vá cổng quên của Nested Learning

Ba lỗi được vá, chẩn đoán trong `CHAN_DOAN_NL_2026-08-02.md`:

1. **α bị đọc ngược 180°.** Dự án ghi chú `alpha≈1 = không quên`; thật ra retention là `1 − α`.
   Bằng chứng: run `m3_s1` có α=0,0000 mà norm state lên 1.757.552.
2. **η log ra logit thô** thay vì giá trị thật sau `adaptive_step_transform`.
3. **Cổng trôi ra biên.** `trace_gates` trên log CŨ (không tốn giờ máy) cho thấy **9/13 run**
   thuộc loại `TROI-DAN`: α khởi đầu lành mạnh 0,10–0,89 rồi dạt tới biên trong 2–4 task.
   **0 run** bão hoà ngay từ đầu → thuốc đúng là `gate_bound` (chặn logit bằng tanh),
   không phải `pre_norm` như dự đoán ban đầu.

Kết quả sau khi vá:

| | `ctrl` (không vá) | `gates` (có vá) |
|---|---|---|
| `trace_gates` xếp loại | `TROI-DAN` ⚠️ | **`LANH-MANH`** ✅ |
| α task 0 → 8 | 0,4475 → 0,9993 (kẹt biên từ task 5) | 0,2637 → 0,0678 |
| α span | — | **0,1959** |
| norm(state) | 54,3 (phẳng = không tích luỹ) | **99,2** (dao động 56–163) |
| Linear | 0,5453 | **0,5704** (+2,51 điểm) |
| AAA | 0,6290 | **0,6547** (+2,57 điểm) |

Cơ chế đã sống lại, nhưng vẫn kém SLDA 11,6 điểm.

### 3.3 `feat/slda-ablation` — tách nguồn gốc của +13,81 điểm

Bốn arm, cùng split `combined31500_v2`, 9 task × 5 lớp, 3 seed:

| Arm | Accuracy | std | Forgetting | AAA |
|---|---:|---:|---:|---:|
| NCM thuần | 0,6924 | 0,0010 | 0,0979 | 0,7778 |
| SLDA **Σ = I** | 0,6885 | 0,0010 | 0,0988 | 0,7758 |
| SLDA **Σ đóng băng** | 0,7682 | 0,0077 | **0,0579** | 0,8437 |
| SLDA **Σ streaming** | **0,8266** | 0,0017 | 0,0768 | **0,8941** |

**Bẫy đã kiểm:** NCM và SLDA Σ=I đi *hai đường code khác nhau* nhưng về toán phải tương đương.
Đo được `|0,6924 − 0,6885| = 0,0040` → ĐẠT. Nếu lệch nhiều thì mọi kết luận đều vô nghĩa.

Tách đóng góp:

| Thành phần | Đóng góp | Tỷ lệ |
|---|---:|---:|
| Ước lượng Σ **một lần** sau task 0 | +7,97 điểm | 58% |
| **Cập nhật tiếp** Σ qua 9 task | +5,84 điểm | 42% |

Chi phí triển khai (B6):

| | SLDA (fp64, CPU) | Titans (self-mod, depth 3) | Tỷ lệ |
|---|---:|---:|---:|
| Bộ nhớ ViT-S | 1,388 MB | **102,682 MB** | **74×** |
| Bộ nhớ ViT-L | 8,942 MB | **729,948 MB** | **82×** |
| Suy luận ViT-S | 0,0071 ms (MPS) | 2,1782 ms (MPS) | **307×** |
| Suy luận ViT-L | 0,0081 ms | 28,3667 ms | 3.500× |

Ở ViT-L, Titans chiếm **85% ngân sách một khung hình 30 fps** — chỉ riêng phần bộ nhớ.

### 3.4 `feat/drift-lambda` — đang chạy ⏳

Regime hoàn toàn mới: **domain-incremental** (mọi task đủ 45 lớp) + **trôi điều kiện quan sát**
tăng dần 0% → 100% (độ sáng, nhiệt độ màu, tương phản, mờ, nhiễu).

Đã hiệu chỉnh trước khi chạy: NCM trên feature đóng băng tụt 0,6900 → 0,5310 ở mức trôi 100%
— nằm trong vùng đề nghị, có chỗ cho λ thể hiện mà feature chưa vỡ.

15 run đang chạy, 5 arm × 3 seed:

| Arm | λ_μ | λ_Σ | n_c thực tế | % so λ=1 | Trải mấy task |
|---|---:|---:|---:|---:|---:|
| 1 | 1,0 | 1,0 | 420 | 100% | 9,0 |
| 2 | 0,999 | 0,9999 | 343 | 81,7% | 7,4 |
| 3 ⭐ | 0,99 | 0,9999 | 99 | 23,5% | 2,1 |
| 4 | 0,97 | 0,9999 | 33 | 7,9% | 0,7 |
| 5 | 0,99 | 0,99 | 99 | 23,5% | Σ chỉ 100 mẫu < D=384 → suy biến |

Còn lại: **D11 — Titans (cổng đã vá) trên đúng stream này**. Đó mới là so sánh thật giữa
*cổng quên học được* (α của NL) và *cổng quên đặt tay* (λ).

---

## 4. Bốn lỗi đã bắt được, ghi lại để không lặp

| Nhánh | Lỗi | Vì sao nguy hiểm |
|---|---|---|
| `fix/nl-gates-alive` | `max_lr` thật là **1,0** chứ không phải 1e-2 | Tôi ghi nhầm trong chẩn đoán; chính test của mình bắt được. Sai theo hướng **xấu hơn** — bộ nhớ đang ghi ở bước nhảy tối đa |
| `feat/drift-lambda` | Σ dùng đồng hồ toàn cục, μ_c dùng đồng hồ theo-lớp → đẳng thức `Σ=(G−Σn_cμμᵀ)/N` sai | Σ mất xác định dương, `inv(Σ)` ra rác. **Không báo lỗi gì** — run vẫn chạy tới cuối |
| `feat/drift-lambda` | `.gitignore` gốc có `data/` không neo → nuốt `src/uavcl/data/drift.py` | Commit xanh, push xanh, chỉ vỡ trên VM. File cũ vẫn được theo dõi nên không ai nghi ngờ |
| `feat/drift-lambda` | Thang λ chỉnh cho dataset lớn hơn thực tế | λ_μ=0,9999 giữ 97,9% trí nhớ → arm 2 trùng arm 1, phí 3 run |

---

## 5. Việc còn nợ

1. **Hai dòng nhánh chưa gộp.** `main` có 2.550 dòng (07-13→07-18: `cms_optimizer.py`,
   `hope.py`, `state_utils.py`, 4 cờ ổn định titans-pytorch) không có trong nhánh làm việc.
   Cần đối chiếu tay xem có gì đáng cứu không, hoặc quyết định bỏ hẳn dòng A.
2. **`0,6933` so với `0,7114`** — hai biến thể NCM chênh 1,9 điểm, chưa rõ khác nhau chỗ nào.
   Phải chốt trước khi viết báo cáo, không thì bảng baseline không đáng tin.
3. **Bảy nhánh chưa có log kết quả** (📝 ở mục 2). Nếu có log ở đâu đó thì nên gom về
   `result_test/` và ghi bổ sung; nếu không thì coi như công việc trung gian, không trích dẫn được.
4. **Metrics JSON chưa commit** — kết quả campaign đang nằm ở file rời, không đi theo git.
