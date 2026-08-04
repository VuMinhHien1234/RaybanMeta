# ĐÃ TRIỂN KHAI — toàn bộ nhóm A + P + DL1 + M1 + M2 (2026-08-04)

Thực hiện theo `KE_HOACH_SUA_2026-08-04.md`, bám phát biểu gốc `BAI_TOAN_VA_MUC_TIEU`.
Từ bản này, **track chính chạy đúng bài toán: pha 2 KHÔNG nhãn, dòng thời gian thật,
cùng khu vực quay lại, đo đủ O1/O2/O3/O4.**

---

## Cái gì đã vào code

| Nhóm | Nội dung | File |
|---|---|---|
| A1 | Cờ `memory.post_norm` (mặc định bật = hành vi cũ) — mở khoá ablation D11 | `models/titans_head.py` |
| A2 | Guard: `stream_type=revisit` bắt buộc `drift.enabled=true`, chết sớm | `scripts/run_g1.py` |
| A3 | `lich_tu_cfg()` — MỘT nguồn lịch bay duy nhất cho run_g1 + loaders | `data/revisit.py`, 2 call site |
| A4 | Dọn config oracle arm1/arm2 (bỏ m3/memory/eval_future, n_mode 4→3) | `configs/revisit_arm{1,2}_*.yaml` |
| P1 ⛔ | **Pha 2 không nhãn**: engine gác ranh giới `chuyen_hieu_chinh`; từ đó chỉ `hap_thu_khong_nhan` (không một nhãn nào chạm update — test chứng minh bằng `count_raw` đứng yên) | `engine.py`, `models/slda.py` |
| P2 | **Prequential** (test-then-train, đúng "trả lời ŷ_t ngay") nuôi O2 + **trôi TRONG chuyến** + `thu_tu_thoi_gian` (shuffle=false, transform eval) | `models/slda.py`, `data/loaders.py` (`TaskDatasetDongThoiGian`) |
| P3 | Bench O4: MB + ms/khung hình cho hệ ba tầng, đối chiếu trần 10 MB / 5% × 33,3 ms | `scripts/bench_ba_tang.py` |
| DL1 | `pha2.test_chung`: mọi chuyến chấm trên CÙNG tập test đầy đủ — đúng nghĩa "cùng khu vực", σ giảm ~3,5× | `scripts/run_g1.py` |
| M1 | **TangNhanh** — m_t/v_t chạy, không nhãn, căn chỉnh `day_du` hoặc `truc` (chống bẫy T2), ~3 KB | `models/tang_nhanh.py` (mới) |
| M2 | **NganHangCheDo** — nhớ điều kiện đã gặp, khớp trên trục, NẠP snapshot khi gặp lại (cơ chế O3), gộp khi tràn K | `models/ngan_hang_che_do.py` (mới) |
| M3 | Hợp nhất ba tầng vào SLDA: update/forward đi qua căn chỉnh; μ_c/Σ sống trong hệ toạ độ e₀; mặc định TẮT — SLDA cũ trùng bit | `models/slda.py` |
| Đ/O2 | `thoi_gian_hoi_phuc` nối vào run_g1 từ chuỗi prequential; báo cáo ba tầng in kèm; tất cả vào `metrics_revisit.json` | `scripts/run_g1.py` |

Config track chính (mới): `revisit_U0_dongbang.yaml` (đóng băng — mốc) →
`revisit_U1_tangnhanh.yaml` (+M1) → `revisit_U2_nganhang.yaml` (+M2).
**Con số công bố = O3(U2) − O3(U1), 3 seed** — phép so một biến, tự khử biến nhiễu
"học thêm thì tự khắc tốt lên".

Test mới: `test_ba_tang.py` (14 test — gồm 3 bẫy ⭐ của M4), `test_pha2_engine.py` (3),
`test_dong_thoi_gian.py` (7, gồm A3), `test_post_norm_flag.py` (3). Sửa 2 test CŨ vốn hỏng
sẵn trong `test_revisit.py` (một cái tự mâu thuẫn luật "đợt", một cái thiếu guard torch).

Đã kiểm trong sandbox (không có torch): py_compile 13 file ✅ · YAML 5 config ✅ ·
27 test thuần Python ✅. **Test torch chưa chạy — bước 0 dưới đây bắt buộc.**

---

## KIỂM ĐẦU-CUỐI (audit cùng ngày — dataset → stream → ba tầng → thước đo)

Chạy `scripts/mo_phong_ba_tang.py` (thuần numpy, mọi máy đều chạy được): mô phỏng ĐẦY ĐỦ
giao thức bằng lịch bay THẬT + thước đo THẬT của dự án, ba tầng phản chiếu đúng công thức.
Kết quả (regime trôi ≈ khoảng cách lớp, đúng cỡ hiệu chỉnh D8):

| | acc chéo TB | O3 chéo | ⭐ O3 preq10 | O2 hồi phục |
|---|---:|---:|---:|---:|
| U0 đóng băng | 0,596 | +0,026 | −0,017 | (0,8)* |
| U1 +tầng nhanh | 0,998 | −0,001 | +0,077 | 12,3 batch |
| U2 +ngân hàng | 0,998 | +0,003 | **+0,166** | **5,7 batch** |

Chuỗi thiết kế RA ĐÚNG TÍN HIỆU: U1 cứu 40 điểm khỏi trôi (O1/O2), ngân hàng chế độ nhân
đôi lợi ích quay lại và giảm nửa thời gian hồi phục (O3). Audit đồng thời bắt được **1 bug
thật + 2 lỗ thước đo**, đã vá:

1. **Bug khớp ngân hàng bị nhiễm chuyến trước** — tại batch `cho_khop_sau`, EMA `m_t` còn
   giữ 20–67% điều kiện chuyến TRƯỚC → có thể nạp nhầm chính chế độ vừa rời khỏi. Vá: khớp
   bằng `m_tuoi` (trung bình CHỈ của chuyến hiện tại) — `ngan_hang_che_do.py` + `slda.py`,
   test hồi quy `test_khop_bang_trung_binh_TUOI...`.
2. **O3 trên đường chéo R bị MÙ với ngân hàng** (cột "O3 chéo" ≈ 0 ở cả U1/U2): R[t][t]
   chấm SAU khi hấp thụ cả chuyến, lúc tầng nhanh đã tự hội tụ. Lợi ích "nhận ra ngay" nằm
   ở ĐẦU chuyến. Vá: run_g1 giờ tính thêm `o3_preq10_loi_ich` (trung bình prequential 10
   batch đầu) — **luật công bố O3 = preq10(U2) − preq10(U1) cùng seed** (U1 làm đối chứng
   khử carryover EMA; cột U1 +0,077 cho thấy carryover tự nó cũng tạo "lợi ích giả").
3. **O2 tính so với mức ổn định CỦA CHÍNH arm** — U0 đứng im ở mức thấp cũng "hồi phục 0,8
   batch"(*). Không so O2 giữa các arm khi acc khác xa nhau; run_g1 in kèm cảnh báo.

Giới hạn đã soát, chấp nhận có chủ đích (ghi vào phần hạn chế của báo cáo):

- Pha 1 chỉ có ~1.837 mẫu nhãn (~40/lớp, 1/12 tập train) — đủ cho μ_c + Σ (nhờ shrinkage),
  và MỌI arm dùng chung pha 1 nên phép so vẫn một-biến. Muốn trần đẹp hơn: tăng
  `chuyen_hieu_chinh` (nhưng chuyến 1+ không còn ở điều kiện e₀ — cân nhắc).
- Test của chuyến chấm ở mức trôi ĐẠI DIỆN chuyến (không áp `trong_chuyen` lên test) —
  lệch ≤ bien_do/2 = 12,5%, có chủ đích để "điều kiện chuyến" được định nghĩa sạch.
- Trôi tổng hợp là dịch photometric toàn cục → thiên vị giả định "trục chung" (T1 gần như
  chắc đạt). Kết luận tổng quát hoá cần DL2 (SkyScenes) — vẫn nằm ở mục "Còn lại".

## Chạy theo thứ tự này (trên Mac / VM có torch)

```bash
cd uav-continual-learning

# 0a) Mô phỏng đầu-cuối (10 giây, không cần GPU/dataset) — nhìn lại tín hiệu thiết kế
python scripts/mo_phong_ba_tang.py

# 0b) BẮT BUỘC — toàn bộ suite, phải xanh trước khi đốt giờ máy (~1 phút)
pytest -q

# 1) Nhóm B — T1/T2, cửa chặn 15 phút. cos<0,3 -> DỪNG ba tầng (xem KE_HOACH_SUA)
mkdir -p artifacts_t1
python scripts/do_truc_dieu_kien.py --config configs/drift_slda_arm3_dexuat.yaml \
  --json artifacts_t1/t1_t2.json

# 2) P3 — bench O4 (~1 phút, CPU): xác nhận ba tầng nằm lọt trần 10 MB / 1,67 ms
python scripts/bench_ba_tang.py

# 3) Track chính U0 -> U1 -> U2, 3 seed (GPU khuyến nghị vì test_chung; 1 đêm)
for s in 0 1 2; do
  python scripts/run_g1.py --config configs/revisit_U0_dongbang.yaml  --set seed=$s
  python scripts/run_g1.py --config configs/revisit_U1_tangnhanh.yaml --set seed=$s
  python scripts/run_g1.py --config configs/revisit_U2_nganhang.yaml  --set seed=$s
done
# ⛔ cửa chặn giữa chừng: U1 không hơn U0 (acc đường chéo + O2) -> dừng, khỏi chạy U2

# 4) Oracle có nhãn (trần trên, nối mạch D10) — song song đêm khác
for s in 0 1 2; do
  python scripts/run_g1.py --config configs/revisit_arm1_lam1.yaml --set seed=$s
  python scripts/run_g1.py --config configs/revisit_arm2_quen.yaml --set seed=$s
done

# 5) N1 — ablation post_norm (đêm máy rảnh, 0 code nhờ A1)
python scripts/run_g1.py --config configs/drift_titans_gates_eta_thap.yaml \
  --set memory.post_norm=false seed=0   # lặp seed 1,2
```

Đọc kết quả: mỗi run in sẵn khối `== BAY LẶP LẠI` (O1, O3 ⭐, O2, báo cáo ba tầng) và ghi
`metrics_revisit.json` (kèm chuỗi prequential từng chuyến để vẽ đường hồi phục).

## Hai chỗ CẦN HIỆU CHỈNH TAY sau khi có số

1. **`nguong` của U2** (tham số nguy hiểm nhất): đọc log U1 dòng
   `[ba_tang] tầng nhanh: ... độ trôi hiện tại X` ở các chuyến trôi 100% → đặt `nguong ≈ X/2`
   trong `revisit_U2_nganhang.yaml`. Cửa chặn: log U2 in `số chế độ (số điều kiện THẬT: N)`
   — lệch gấp đôi là hỏng, có cảnh báo ⚠️ tự động.
2. **`kieu: truc`**: nếu T1 (bước 1) kết luận TRUC_CHUNG → đổi `tang_nhanh.kieu: truc` +
   `truc_json: artifacts_t1/t1_t2.json` trong U1/U2 (đã để sẵn comment trong config) —
   chống bẫy tỷ lệ lớp T2.

## Còn lại (chưa làm trong đợt này)

- **DL2** — chạy T1 trên SkyScenes (`--theo-viewpoint`, cần tải dữ liệu, nửa ngày): kiểm
  "trục chung" trên điều kiện THẬT thay vì trôi tổng hợp vốn cài sẵn đáp án.
- **N2** — rerun D11 với `eval_ncm_head: true` (13 h máy, chỉ khi cần tuyên bố về Titans).
- **N3** — chốt 0,6933 vs 0,7114 (1 giờ người, trước khi viết báo cáo).
- Resume checkpoint CHƯA lưu state ba tầng (m_t/ngân hàng) — run revisit ngắn nên chấp nhận;
  nếu cần chạy theo ca thì nối `export_state/import_state` (đã có sẵn trên model) vào
  `checkpoint.py`.
