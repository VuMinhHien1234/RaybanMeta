# Kế hoạch SỬA chi tiết — từ review code 2026-08-04

> **TRẠNG THÁI (cập nhật cùng ngày): nhóm A + P + DL1 + M1 + M2 + O2 ĐÃ CÀI XONG** —
> xem `DA_TRIEN_KHAI_SUA_2026-08-04.md` (danh sách file + lệnh chạy theo thứ tự).
> Còn lại cho người/máy: chạy pytest → T1/T2 → bench → U0/U1/U2 → oracle → N1;
> tuỳ chọn: DL2 (SkyScenes), N2, N3.

Lập sau khi rà toàn bộ `uav-continual-learning` đối chiếu với `KE_HOACH_BA_TANG_2026-08-04.md`.
Bổ sung cùng ngày: **nhóm P** — sau khi đối chiếu lại `BAI_TOAN_VA_MUC_TIEU`, giao thức thí
nghiệm hiện tại còn 3 chỗ KHÔNG khớp phát biểu gốc; không sửa thì arm4 có thắng cũng chưa trả
lời đúng bài toán.

Nguyên tắc giữ nguyên từ dự án: **mặc định TẮT mọi thứ mới** (run cũ bất biến), **chặn cấu hình
vô nghĩa trước khi đốt giờ máy**, **cửa chặn trước phần rủi ro**.

Trạng thái đã xác minh: nhóm R (stream lặp lại) ✅ · nhóm Đ (O1/O3) ✅ · SLDA hai λ ✅ ·
script T1/T2 ✅ nhưng **chưa chạy** · M1/M2 ❌ · O2 ❌ chưa nối · ablation post_norm ❌ bị chặn.

## Đối chiếu với bài toán gốc — cái gì đã khớp, cái gì chưa

| Mục tiêu gốc | Hạ tầng hiện tại | Còn hở gì | Vá ở |
|---|---|---|---|
| O1 giữ danh tính lớp | ✅ SLDA + `acc_dieu_kien_hien_tai` | — | — |
| O2 bám trôi không nhãn | ◐ A5 đo được | điều kiện KHÔNG đổi trong chuyến; loader xáo trộn; có aug ngẫu nhiên — tầng nhanh chưa thật sự bị thử | **P2** |
| O3 nhớ điều kiện | ◐ stream lặp + thước đo có | pha 2 đang **CÓ NHÃN mọi chuyến** — trái phát biểu §2/§3 ("nhãn: chỉ pha 1") | **P1** ⛔ |
| O4 chạy trên drone | ❌ | chưa có bước đo MB/ms cho hệ ba tầng | **P3** |
| Không lưu ảnh thô | ✅ chỉ thống kê | — | — |
| Trả lời ŷ_t ngay lập tức | ◐ | đường chéo R[t][t] chấm SAU khi hấp thụ cả chuyến — không phải prequential | **P2** |

---

# NHÓM A — Sửa code nhỏ (~4 giờ, làm TRƯỚC mọi lần chạy máy)

## A1 — Cờ `memory.post_norm` (30 phút, mở khoá Ưu tiên 2 của D11)

**Vấn đề.** `titans_head.py:130` bật cứng `nn.LayerNorm(dim)`. Giả thuyết D11 §2.1: chính lớp
này triệt tiêu η (đổi trần η 100× → Δacc = 0,00002). Ablation cần tắt nó — hiện không tắt được.

```python
# titans_head.py:130 — thay
self.post_norm = nn.LayerNorm(dim)
# bằng (mặc định BẬT -> mọi run cũ bất biến):
_pn = bool(memory_cfg.get("post_norm", True))
self.post_norm = nn.LayerNorm(dim) if _pn else nn.Identity()
print(f"[titans] post_norm {'BẬT (mặc định)' if _pn else 'TẮT — ablation D11'}")
```

- `nn.Identity()` giữ nguyên đường gọi ở dòng 192, không sửa chỗ nào khác.
- Test thêm vào `tests/test_g2_titans.py`: (a) không khai báo key → là LayerNorm; (b)
  `post_norm: false` → là Identity; (c) hai model chỉ khác cờ này cho output khác nhau.
- **Ghi chú cho lúc đọc kết quả:** residual `+ seq` (dòng 192) cũng pha loãng tín hiệu memory.
  Nếu tắt post_norm mà η vẫn vô dụng → nghi phạm kế tiếp là residual, làm cờ `memory.residual`
  tương tự (nhưng ĐỪNG làm trước — mỗi lần một biến).

## A2 — Guard: `stream_type=revisit` bắt buộc `drift.enabled=true` (15 phút)

**Vấn đề.** `run_g1.py` nhánh revisit không kiểm drift. Quên bật drift → mọi chuyến cùng điều
kiện, nhưng `mode_that` vẫn được tính và báo cáo O3 vẫn in ra như thật — đo một hiện tượng
không tồn tại.

```python
# run_g1.py, đầu nhánh revisit:
if not bool((cfg["data"].get("drift") or {}).get("enabled", False)):
    raise ValueError("stream_type=revisit cần data.drift.enabled=true — không có trôi thì "
                     "mọi chuyến cùng điều kiện, thước đo O3 vô nghĩa")
```

## A3 — Một nguồn `lich_bay` duy nhất (30 phút)

**Vấn đề.** `run_g1.py:132` và `loaders.py:91` cùng dựng lịch bay, mỗi nơi tự đọc tham số +
tự điền mặc định. Hiện khớp nhau; sửa mặc định một nơi là lệch ngầm — bug loại khó thấy nhất.

```python
# revisit.py — thêm:
def lich_tu_cfg(n_chuyen: int, drift_cfg: dict) -> List[ChuyenBay]:
    """MỘT chỗ đọc tham số duy nhất — run_g1 và loaders đều phải gọi qua đây."""
    return lich_bay(n_chuyen,
                    che_do=str(drift_cfg.get("che_do", "tuan_hoan")),
                    chu_ky=int(drift_cfg.get("chu_ky", 4)),
                    severity=float(drift_cfg.get("severity", 1.0)),
                    troi_dai_han=float(drift_cfg.get("troi_dai_han", 0.0)),
                    n_mode=int(drift_cfg.get("n_mode", 4)))
```

Hai call site đổi về `lich_tu_cfg(...)`. Test: lịch từ run_g1 và từ loaders trùng nhau
từng phần tử với cùng cfg.

## A4 — Dọn config `revisit_arm1/arm2` (15 phút)

| Sửa | Lý do |
|---|---|
| `train.optimizer: m3` → `adamw` (hoặc xoá) | SLDA gradient-free không dùng optimizer, nhưng `run_dir_name` gắn hậu tố `_m3` vào tên thư mục → đọc kết quả sau này tưởng nhầm có M3 |
| Xoá khối `train.m3:` | đi theo dòng trên |
| Xoá khối `memory:` (giữ `enabled: false` một dòng nếu muốn) | 12 dòng tham số Titans không dùng, gây nhiễu khi diff config giữa các arm |
| `eval_future: true` → `false` | thước đo revisit chỉ dùng đường chéo; bỏ tiết kiệm 12 lần eval/run |
| `n_mode: 4` → `3` | lịch sin chu_ky=4 chỉ sinh 3 mức {0, 50, 100} → mode 1 không bao giờ xuất hiện; để 4 thì `so_che_do` đọc dễ nhầm |

Các arm chưa chạy lần nào (chưa có `artifacts_revisit_*`) → đổi tên thư mục/config không phá
tính so sánh với kết quả cũ nào.

## A5 — Nối O2 (thời gian hồi phục) vào engine (2 giờ — mục lớn nhất nhóm A)

**Vấn đề.** `thoi_gian_hoi_phuc()` cần chuỗi accuracy THEO BƯỚC trong một chuyến; engine chỉ
chấm sau mỗi chuyến (R theo task) và `tom_tat_revisit` không hề gọi hàm này. O2 hiện là thước
đo trên giấy.

**Thiết kế** (mặc định TẮT):

```yaml
train:
  eval_trace:            # chỉ có nghĩa với method gradient-free (slda/ncm)
    enabled: true
    every_batches: 10    # chấm 1 lần mỗi 10 batch hấp thụ
    val_batches: 4       # 4 batch val đầu (~128 mẫu) — nhiễu chấp nhận được cho đường cong
```

- `methods.py` — `SLDA.fit_task_trace(model, loader, val_loader, device, trace_cfg)`: hấp thụ
  như `fit_task`, cứ `every_batches` lại chấm nhanh trên bộ val nhỏ (cache tensor val MỘT lần,
  không đọc đĩa lặp lại), trả về `list[float]`.
- `engine.py` — nhánh `gradient_free`: nếu trace bật và method có `fit_task_trace` thì gọi nó,
  lưu `log["trace"][t]`; ngược lại đi đường cũ nguyên vẹn.
- `run_g1.py` — nếu có `log["trace"]` và `mode_that`: với mỗi chuyến t mà mode đổi so với t−1,
  tính `thoi_gian_hoi_phuc(log["trace"][t])`, ghi trung bình + từng chuyến vào
  `metrics_revisit.json` (`o2_hoi_phuc_tb`, `o2_hoi_phuc_theo_chuyen`).
- Test: trace tắt → `log` không có key `trace`, kết quả bit-trùng đường cũ; series toàn 1.0 →
  hồi phục = 0; series tăng dần → số bước đúng công thức.

**Chi phí:** ~+20% thời gian một chuyến với 10/4. Chỉ bật cho run cần O2.

## A6 — Vệ sinh: chạy lại pytest (10 phút, trên Mac)

`.pytest_cache` còn ghi `test_streaming_KHAC_identity_tren_du_lieu_bat_dang_huong` fail —
test này đã được viết lại thành `..._tren_nhieu_tuong_quan` (dữ liệu cũ quá dễ). Chạy
`pytest -q` xác nhận toàn bộ xanh, chốt lại mốc sạch trước khi sửa gì thêm.

---

# NHÓM B — Chẩn đoán T1/T2 (⛔ CỬA CHẶN, 15 phút máy, 0 code)

Script đã viết xong (`scripts/do_truc_dieu_kien.py`) nhưng **chưa có kết quả JSON nào**.
Đây là cửa chặn số 1 do chính kế hoạch ba tầng đặt ra — chạy TRƯỚC khi viết M1.

```bash
mkdir -p artifacts_t1
python scripts/do_truc_dieu_kien.py \
  --config configs/drift_slda_arm3_dexuat.yaml \
  --json artifacts_t1/t1_t2.json
```

| Kết quả | Hành động |
|---|---|
| T1 = TRUC_CHUNG (cos ≥ 0,8 · PC1 ≥ 70%) | ✅ đi tiếp M1 bản "chiếu lên trục" |
| T1 = MOT_PHAN | M1 căn chỉnh đầy đủ (m_t, v_t chéo); tầng trung gánh phần riêng |
| T1 = KHONG_CO_TRUC (cos < 0,3) | ⛔ DỪNG hướng ba tầng → quay lại Titans (và lúc đó A1 + ablation post_norm thành đường chính) |
| T2 = CAN_CAN_BANG_LOP (nhiễu ≥ 30%) | M1 phải chiếu lên trục điều kiện, KHÔNG căn toàn vector |

File `t1_t2.json` chứa `truc_dieu_kien` — M1 và M2 nạp lại trục này, không tính lại.

---

# NHÓM P — KHỚP GIAO THỨC VỚI BÀI TOÁN GỐC (~6 giờ code, làm trước track chính)

Đây là phần trả lời câu hỏi *"kế hoạch có giải được bài toán ban đầu không"*. Ba lỗ hổng,
xếp theo mức nghiêm trọng.

## P1 — ⛔ Pha 2 phải KHÔNG NHÃN (2 giờ — lỗ hổng lớn nhất)

**Vấn đề.** Phát biểu gốc (§2, §3): nhãn CHỈ có ở pha 1 hiệu chỉnh; pha 2 là dòng không nhãn.
Nhưng engine hiện cho SLDA hấp thụ `(x, y)` CÓ NHÃN ở **mọi** chuyến — arm1/arm2 đang giải một
bài dễ hơn hẳn bài toán thật. Mọi kết quả track chính phải chạy ở chế độ không nhãn.

**Thiết kế** (mặc định TẮT — run cũ bất biến):

```yaml
data:
  pha2:
    khong_nhan: true        # từ chuyến 1 trở đi: KHÔNG đưa nhãn vào bất kỳ update nào
    chuyen_hieu_chinh: 1    # số chuyến đầu CÓ nhãn (pha 1); mặc định 1
```

- `engine.py` — nhánh `gradient_free`: chuyến `t < chuyen_hieu_chinh` → `fit_task` như cũ
  (pha 1, có nhãn). Chuyến sau: KHÔNG gọi `fit_task`; thay bằng
  `model.hap_thu_khong_nhan(loader)` nếu model có (tầng nhanh/trung tự cập nhật từ feature,
  không đụng μ_c/Σ), không có thì bỏ qua (= đóng băng hoàn toàn).
- Nhãn của test set vẫn dùng để CHẤM — cùng nguyên tắc với `mode_that` ("chỉ để chấm điểm").
- Hệ quả lên thiết kế arm: ở chế độ không nhãn, λ_μ **không còn tồn tại** (μ_c không update
  thì không có gì để quên). Track chính vì vậy là:

| Arm chính | Cấu hình (đều `khong_nhan: true`) | Trả lời |
|---|---|---|
| U0 | đóng băng hoàn toàn sau pha 1 | mốc "không làm gì" — mọi cải thiện đo từ đây |
| U1 | + tầng nhanh (M1) | căn chỉnh không nhãn đáng giá bao nhiêu (O2) |
| U2 | + tầng trung (M2) | ⭐ nhớ điều kiện đáng giá bao nhiêu (O3 = U2 − U1) |
| O-lab | arm1 λ=1 **có nhãn** mọi chuyến | TRẦN TRÊN oracle — khoảng cách tới nó = giá của việc mất nhãn |

- (tuỳ chọn, arm riêng, mặc định TẮT) `pha2.gia_nhan: true` — hấp thụ `(f, ŷ)` bằng nhãn giả
  tự tin cao vào tầng chậm. Rủi ro vòng lặp tự nhiễm — chỉ chạy sau khi U1/U2 có kết quả.
- Test: `khong_nhan: false` → bit-trùng đường cũ; bật → `count_raw` của SLDA đứng yên từ
  chuyến 1 (bằng chứng không có nhãn nào lọt vào).

## P2 — Trôi TRONG chuyến + thứ tự thời gian + prequential (3 giờ)

**Vấn đề.** Ba chỗ lệch nhỏ nhưng cùng một gốc — pha 2 hiện được xử lý như "tập train tĩnh",
không phải "dòng thời gian":

1. Điều kiện KHÔNG đổi bên trong một chuyến (mỗi chuyến một mức trôi cố định + jitter) —
   trong khi yêu cầu (c) của bài toán là *"nắng gắt dần suốt 10 phút"*, tức đổi **trong**
   chuyến. Tầng nhanh chưa thật sự bị thử đúng việc của nó.
2. Loader pha 2 `shuffle=True` — dòng thật không ai xáo trộn thời gian.
3. Pha 2 vẫn dùng aug train (RandomResizedCrop/Flip) — dữ liệu triển khai không phải dữ liệu
   augment.

**Thiết kế** (mặc định TẮT từng cờ):

```yaml
data:
  pha2:
    thu_tu_thoi_gian: true   # shuffle=false + transform kiểu eval (không aug) cho chuyến pha 2
  drift:
    trong_chuyen:
      enabled: true
      bien_do: 0.25          # mức trôi đi từ (muc − bđ/2) -> (muc + bđ/2) theo vị trí trong chuyến
```

- `loaders.py`: với chuyến pha 2, `TaskDataset` nhận thêm hàm `muc_theo_vi_tri(k/len)`; drift
  áp theo vị trí mẫu thay vì một mức chung. `shuffle=false` để vị trí = thời gian.
- Prequential (nối dài A5, đúng nghĩa "trả lời ŷ_t trước khi thấy x_{t+1}"): trong
  `hap_thu_khong_nhan`, **chấm trên chính batch sắp hấp thụ TRƯỚC khi cập nhật**
  (test-then-train — chuẩn streaming). Rẻ hơn A5 (không cần val riêng) và trung thực hơn.
  Giữ A5 làm phương án cho method có aug/không streaming.
- Test: `bien_do=0` ≡ hành vi cũ; mức trôi mẫu cuối − mẫu đầu = đúng `bien_do`.

## P3 — Bước đo O4 cho hệ ba tầng (1 giờ)

`bench_edge.py`/`bench_cost.py` đã có cho SLDA/Titans — thêm đường đo cho U2 (SLDA + M1 + M2):
MB thêm (kỳ vọng: M1 ~3 KB + M2 ~K·2D·8B, K=8 → ~50 KB) và ms/khung hình (căn chỉnh O(D) +
khớp chế độ O(K·D)). Ghi vào bảng cùng định dạng B6, đối chiếu ràng buộc **≤ 10 MB, ≤ 5%
ngân sách 33,3 ms**. Chạy một lần sau khi U2 chạy được — trước khi viết bất kỳ báo cáo nào.

---

# NHÓM C — Mốc arm1/arm2 trên stream lặp lại (1 đêm máy, 0 code)

**Vai trò sau khi có P1: đây là track ORACLE có nhãn** — giữ mạch so sánh với D10 và làm trần
trên cho track chính U0–U2. Kết quả λ ở đây KHÔNG được suy ra cho chế độ không nhãn.

Chạy SAU nhóm A (để hưởng A2/A3/A4), 3 seed mỗi arm:

```bash
for s in 0 1 2; do
  python scripts/run_g1.py --config configs/revisit_arm1_lam1.yaml --set seed=$s
  python scripts/run_g1.py --config configs/revisit_arm2_quen.yaml --set seed=$s
done
```

Ba câu hỏi bảng này trả lời:

1. **+3,91 điểm của λ=0,99 (D10) còn sống không khi điều kiện quay lại?** Dự đoán trong
   `KE_HOACH_BA_TANG`: có thể đảo dấu. Đây là kết quả công bố được dù ra chiều nào.
2. **`loi_ich_quay_lai` của arm1 (λ=1, KHÔNG có tầng nhớ điều kiện)** = độ lớn của biến nhiễu
   "học thêm dữ liệu thì tự khắc tốt lên". Con số này là MẪU SỐ khi diễn giải mọi arm sau.
3. O2 (nếu bật `eval_trace`): λ=0,99 hồi phục nhanh hơn λ=1 bao nhiêu bước khi mode đổi.

**⚠️ Luật báo cáo O3 (ghi vào mọi bảng kết quả từ giờ):** `loi_ich_quay_lai` thô LẪN biến
"model đã thấy nhiều dữ liệu hơn". Không bao giờ báo cáo nó đứng một mình làm bằng chứng O3 —
chỉ báo **hiệu giữa arm có/không tầng trung** (arm4 − arm3), cùng seed, cùng lịch bay.

---

# NHÓM D — M1: Tầng NHANH + arm3 (4 giờ code + 1 đêm máy)

Chỉ bắt đầu sau khi B đạt. File mới `src/uavcl/models/tang_nhanh.py`:

```python
class TangNhanh:
    """Thống kê điều kiện hiện tại — KHÔNG nhãn, O(D) bộ nhớ (~3 KB, thoả ràng buộc 10 MB).

    Trạng thái:  m_t (D,) trung bình chạy · v_t (D,) phương sai ĐƯỜNG CHÉO chạy
    Mốc pha 1:   m0, v0 — chụp cuối chuyến hiệu chỉnh (chuyến 0)
    Cập nhật mỗi batch:  m_t ← λ·m_t + (1−λ)·mean(f)     (λ=0,99 ≈ cửa sổ 100 mẫu)
                         v_t ← λ·v_t + (1−λ)·var(f)
    Căn chỉnh:
      bản đầy đủ : f' = (f − m_t) · sqrt(v0 / v_t) + m0
      bản trục   : f' = f − ((m_t − m0)·u) u        # u = trục điều kiện từ t1_t2.json
                                                     # phần ⊥ trục (thông tin lớp) GIỮ NGUYÊN
    """
```

Chọn bản nào do T1/T2 quyết định (bảng nhóm B). Tích hợp vào `SLDAClassifier`:

- Config: khối `slda.tang_nhanh: {enabled: false, decay: 0.99, kieu: truc|day_du, truc_json: ...}`.
- Vị trí căn chỉnh: feature căn chỉnh xong mới vào `update()` VÀ `forward()` — thống kê tầng
  chậm sống trong hệ toạ độ e₀, không bị trôi kéo đi.
- `m0/v0`: chốt tại `on_task_end()` của chuyến 0 (pha hiệu chỉnh). Trước khi chốt: căn chỉnh
  là no-op (chuyến 0 tự nó là mốc).
- Tắt cờ → **bit-trùng** SLDA hiện tại (bất biến ngược, test bắt buộc).

Test (M4 một phần):

| Test | Chặn cái gì |
|---|---|
| `enabled: false` → output trùng bit SLDA cũ | bất biến ngược |
| dữ liệu KHÔNG trôi → căn chỉnh ≈ identity | không tự phá khi không có bệnh |
| `m_t` hội tụ đúng λ lý thuyết (như `window_report`) | cài đúng công thức |
| trôi nhân tạo biết trước → `m_t − m0` khớp hướng trôi | căn đúng chiều |

Config mới theo track chính P1: `configs/revisit_U0_dongbang.yaml` (không nhãn, không tầng
mới) và `configs/revisit_U1_tangnhanh.yaml` (= U0 + `tang_nhanh` bật). Chạy 3 seed, bật
`pha2.khong_nhan` + `thu_tu_thoi_gian` + `trong_chuyen` (P2).

**⛔ CỬA CHẶN:** so U1 với U0 trên `acc_dieu_kien_hien_tai` và O2 (prequential). Tầng nhanh —
thứ đơn giản, chắc ăn nhất — mà không cải thiện thì M2 phức tạp hơn không cứu được: DỪNG,
viết kết quả âm, không đốt 8 giờ vào M2.

---

# NHÓM E — M2: Tầng TRUNG + arm4/5 (8 giờ code + 1 đêm máy, rủi ro CAO NHẤT)

File mới `src/uavcl/models/ngan_hang_che_do.py` — đúng thiết kế trong `KE_HOACH_BA_TANG`
(mỗi chế độ `(m_k, v_k, so_lan_gap)`; khớp bằng khoảng cách TRÊN TRỤC điều kiện; `nguong` tạo
mới; gộp khi K > K_max). Ba tham số nguy hiểm (`nguong`, `K_max`, λ trong chế độ) — cửa chặn
riêng: chạy với `mode_that` đã biết, số chế độ hệ tự tạo phải khớp số điều kiện thật.

Test bắt buộc (3 bẫy ⭐ từ M4): gặp lại điều kiện cũ → khớp chế độ CŨ không tạo mới · 4 điều
kiện tách biệt → đúng 4 chế độ · đổi tỷ lệ lớp giữ điều kiện → KHÔNG tạo chế độ mới.

U2 = U1 + ngân hàng chế độ (không nhãn). U2-lin = U2 với `che_do: troi_dan` (nối lại D10).
**Con số của cả dự án = O3(U2) − O3(U1)**, 3 seed, kèm σ — đo đúng chế độ không nhãn của
bài toán gốc, và nhờ so một-biến nên tự khử biến nhiễu "học thêm dữ liệu" (luật nhóm C).

---

# NHÓM DL — DỮ LIỆU: đúng nhưng chưa đủ (rà 2026-08-04)

**Phán quyết.** RESISC45 + trôi tổng hợp GIỮ làm testbed chính — nó là bộ duy nhất cho phép
điều khiển trôi trong chuyến (P2), prequential, và lặp điều kiện chính xác. Nhưng nó có 3 lỗ
hổng so với bài toán gốc; DL1 sửa được bằng code, DL2/DL3 cần dữ liệu ngoài.

## DL1 — `test_chung`: "cùng khu vực" thật sự + tăng power thống kê (1 giờ code)

**Vấn đề.** Bài toán nói *bay lại CÙNG khu vực*. Nhưng `build_domain_stream` →
`chia_deu_theo_lop` chia mẫu RỜI NHAU cho từng chuyến: mỗi chuyến nhìn ảnh KHÁC nhau — "quay
lại" hiện chỉ là quay lại *điều kiện*, không phải *địa điểm*. Hệ quả phụ nghiêm trọng: test
mỗi chuyến chỉ còn ~1/12 tập test (~500 ảnh) → nhiễu đường chéo cỡ ±2 điểm, trong khi hiệu
ứng cần đo (O3) kỳ vọng 1–4 điểm — sát nút nguy hiểm.

**Sửa** (`data.pha2.test_chung: true`, mặc định tắt): MỌI chuyến chấm trên **cùng một tập
test đầy đủ**, chỉ khác drift của chuyến đó. Cùng ảnh + khác điều kiện = đúng nghĩa "cùng khu
vực quay lại", và test to gấp 12 lần → σ giảm ~3,5×. Sửa ở `run_g1.py`/`loaders.py`: nhánh
revisit dùng `test_idx` = toàn bộ split test cho mọi TaskSpec. Train/val vẫn chia rời (mỗi
chuyến thấy mẫu mới — đúng thực tế bay).

## DL2 — Kiểm T1 trên điều kiện THẬT (nửa ngày, ⚠️ đe doạ tính hợp lệ của cả hướng)

**Vấn đề.** Trôi tổng hợp là 5 phép photometric áp GIỐNG NHAU cho mọi ảnh → gần như **cài sẵn
đáp án "có trục chung"** cho T1. Điều kiện thật (mùa) đổi cả ngữ nghĩa: cây đổi màu, tuyết,
bóng đổ đổi hướng — không chắc còn là một trục. Nếu chỉ kiểm T1 trên trôi tổng hợp thì kết
luận "căn chỉnh tuyến tính đủ dùng" có thể không chuyển sang đời thật.

**Sửa.** Chạy T1 lần hai trên [SkyScenes](https://hoffman-group.github.io/SkyScenes/) (ECCV
2024, CARLA): ảnh UAV thật sự, **mỗi viewpoint có 60 biến thể weather/daytime/height/pitch với
toạ độ camera tái lập được** — chính xác "cùng chỗ, khác điều kiện". T1 phiên bản này còn sạch
hơn bản gốc: vector dịch tính **theo viewpoint** (cùng cảnh, 2 điều kiện) thay vì theo lớp,
không cần nhãn. Chỉ cần tải 2–3 biến thể × vài trăm viewpoint (~vài GB), sửa
`do_truc_dieu_kien.py` thêm chế độ `--theo-viewpoint`.

| Kết quả T1-thật | Nghĩa là |
|---|---|
| Trục chung như bản tổng hợp | ✅ kết quả RESISC45 có giá trị ngoài — ghi vào báo cáo làm bằng chứng ngoại suy |
| Không có trục chung | ⚠️ hướng ba tầng chỉ đúng cho trôi photometric; phần điều kiện phi tuyến phải ghi rõ là giới hạn (hoặc quay lại Titans cho phần đó) |

## DL3 — (tuỳ chọn, sau khi U2 thắng) Benchmark trên điều kiện thật

Hai ứng viên, chỉ làm MỘT khi cần bằng chứng "chạy trên dữ liệu thật" cho báo cáo:

| Bộ | Vì sao hợp | Giá phải trả |
|---|---|---|
| [SkyScenes](https://arxiv.org/abs/2312.06719) | cùng viewpoint × điều kiện có kiểm soát; UAV thật sự | nhãn là segmentation → phải suy nhãn phân loại (lớp trội mỗi ảnh), ~1 buổi code |
| [fMoW](https://sustainlab-group.github.io/SatMAE/) / fMoW-Sentinel | chuỗi thời gian THẬT cùng địa điểm, 62 lớp, mùa thật quay vòng | rất nặng (phải subsample); điều kiện không kiểm soát → không đo được O2 trong chuyến |

RESISC45/EuroSAT là ảnh vệ tinh (Google Earth/Sentinel), không phải ảnh drone — ghi rõ vào
mục hạn chế của báo cáo dù có hay không có DL3.

---

# NỢ CŨ — xếp lịch luôn cho khỏi trôi

| # | Việc | Chi phí | Khi nào |
|---|---|---|---|
| N1 | Ablation post_norm: `drift_titans_gates_eta_thap` + `memory.post_norm=false`, 3 seed | 4h máy, 0 code (sau A1) | đêm nào máy rảnh, song song nhóm C/D |
| N2 | D11 chạy lại với `train.eval_ncm_head: true` — phép so Titans/SLDA hiện KHÔNG hợp lệ | 13h máy | CHỈ khi cần tuyên bố về Titans; sau N1 |
| N3 | Chốt 0,6933 vs 0,7114 (hai biến thể NCM) — diff config 2 run, tìm biến khác nhau | 1h người | trước khi viết báo cáo |

---

# Thứ tự tổng và cửa chặn

```
A1..A6  sửa code + vệ sinh          ~4h      (A5 có thể dời sau B nếu vội)
   ↓
B       T1/T2                        15ph    ⛔ cos < 0,3 -> DỪNG ba tầng, rẽ sang Titans (N1 thành đường chính)
   ↓
P1..P3  khớp giao thức bài toán gốc  ~6h     ⛔ track chính BẮT BUỘC chạy chế độ không nhãn
   +
DL1     test_chung                   1h      (cùng đợt code với P — trước khi chạy bất kỳ arm nào)
   ↓
C       oracle arm1/arm2 × 3 seed    1 đêm    (song song: N1 nếu còn máy · DL2 T1-thật nếu còn người)
   ↓
D       M1 + U0/U1                   4h + 1 đêm   ⛔ U1 không hơn U0 -> DỪNG, viết kết quả âm
   ↓
E       M2 + U2                      8h + 1 đêm
   ↓
P3      đo O4 (MB/ms) cho U2         1h
```

Tổng: **~23 giờ code + 3 đêm máy** (+ nửa ngày DL2; chưa tính N2/DL3 tuỳ chọn).

Hai điểm khác so với `KE_HOACH_BA_TANG`: (1) thêm nhóm A — sửa chữa mở đường; (2) thêm nhóm P
— ép giao thức về đúng phát biểu gốc: pha 2 không nhãn (P1), dòng thời gian thật + trôi trong
chuyến + prequential (P2), và bước đo phần cứng O4 (P3). Bảng kết quả trung tâm của dự án đổi
từ "arm3 vs arm4 có nhãn" thành **U0 → U1 → U2 không nhãn, oracle có nhãn làm trần trên**.
