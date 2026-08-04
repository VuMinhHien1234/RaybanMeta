# Chạy code MỚI (hệ ba tầng) trên máy ảo — từ đầu đến cuối

Ngày 2026-08-04 · sau `DA_TRIEN_KHAI_SUA_2026-08-04.md` (nhóm A + P + DL1 + M1 + M2)

| | |
|---|---|
| Nhánh | `feat/cms-ba-tang` |
| Trạng thái code | ⚠️ **CHƯA COMMIT, CHƯA LÊN GITHUB** — xem PHẦN A |
| Track chính | U0 (đóng băng) → U1 (+tầng nhanh) → U2 (+ngân hàng chế độ), 3 seed |
| Số run | 9 (track chính) + 6 oracle + 3 ablation N1 |
| Thời gian | ~1,1–1,4 h/run trên e2-standard-8 → track chính ~10–13 h, để qua đêm |
| Máy | CPU `e2-standard-8` (như mọi campaign trước) |
| Repo | `https://github.com/VuMinhHien1234/RaybanMeta.git` |

## ⚙️ Làm MỘT LẦN trước khi copy-paste bất cứ lệnh nào

zsh ở chế độ tương tác **không hiểu `#` là chú thích** (khác bash). Dán một dòng có chú thích
đuôi vào Terminal sẽ ra `ERROR: unrecognized arguments: # đổi tên/zone...`. Bật một lần cho
xong, áp dụng cho mọi file hướng dẫn trong dự án:

```bash
echo 'setopt interactive_comments' >> ~/.zshrc
source ~/.zshrc
```

Chưa bật thì khi dán nhớ **cắt bỏ phần từ dấu `#` trở đi**.

---

**Điểm khác mọi lần trước — đọc trước khi làm gì:** campaign này **KHÔNG chạy được bằng một
vòng for duy nhất**. Có 2 tham số phải hiệu chỉnh TAY *giữa chừng*, đọc từ log của bước trước:

```
T1 (15 phút) ──► quyết định `tang_nhanh.kieu` ──► U0 + U1 (3 seed) ──► đọc log U1 lấy
`độ trôi hiện tại` ──► đặt `ngan_hang.nguong` ──► U2 (3 seed)
```

Chạy U2 trước khi có số của U1 = ngưỡng đặt mò = **đốt 4 giờ máy cho kết quả vô nghĩa**.

---

# PHẦN A — Trên Mac: đưa code mới lên GitHub

## A0. Gỡ lock git còn kẹt

Thư mục đang có `.git/index.lock` rỗng (sandbox để lại) — không xoá thì mọi lệnh `git add`
đều chết:

```bash
cd ~/Desktop/Raybanmeta
rm -f .git/index.lock .git/HEAD.lock .git/objects/maintenance.lock
git status --short | head
```

## A1. ⛔ Chạy pytest trên Mac TRƯỚC (bắt buộc, ~1 phút)

Toàn bộ 49 test mới (`test_ba_tang` 17 · `test_revisit` 19 · `test_dong_thoi_gian` 7 ·
`test_pha2_engine` 3 · `test_post_norm_flag` 3) **chưa từng chạy với torch** — sandbox viết
code không có torch, chỉ kiểm được `py_compile` + 27 test thuần Python.

```bash
cd ~/Desktop/Raybanmeta/uav-continual-learning
python scripts/mo_phong_ba_tang.py      # 10 giây, không cần dataset — xem lại tín hiệu thiết kế
pytest -q                                # PHẢI XANH TOÀN BỘ
```

`pytest` đỏ → **dừng ở đây, báo lại**, đừng tạo VM. Số của `mo_phong_ba_tang.py` phải khớp
bảng trong `DA_TRIEN_KHAI_SUA` (U2 preq10 ≈ +0,166 · hồi phục 5,7 batch).

> **Lần chạy đầu (2026-08-04) đỏ 1/228 test — đã vá, xem `VA_TEST_M1_2026-08-04.md`.**
> Lỗi nằm ở `_du_lieu()` của `test_ba_tang.py`, không phải M1. Chạy lại `pytest -q` phải ra
> **228 passed**.

## A2. Commit + push

```bash
cd ~/Desktop/Raybanmeta

git add \
  uav-continual-learning/src/uavcl/models/tang_nhanh.py \
  uav-continual-learning/src/uavcl/models/ngan_hang_che_do.py \
  uav-continual-learning/src/uavcl/models/slda.py \
  uav-continual-learning/src/uavcl/models/titans_head.py \
  uav-continual-learning/src/uavcl/data/revisit.py \
  uav-continual-learning/src/uavcl/data/loaders.py \
  uav-continual-learning/src/uavcl/metrics/ \
  uav-continual-learning/src/uavcl/engine.py \
  uav-continual-learning/scripts/run_g1.py \
  uav-continual-learning/scripts/bench_ba_tang.py \
  uav-continual-learning/scripts/do_truc_dieu_kien.py \
  uav-continual-learning/scripts/mo_phong_ba_tang.py \
  uav-continual-learning/configs/revisit_*.yaml \
  uav-continual-learning/tests/test_ba_tang.py \
  uav-continual-learning/tests/test_revisit.py \
  uav-continual-learning/tests/test_dong_thoi_gian.py \
  uav-continual-learning/tests/test_pha2_engine.py \
  uav-continual-learning/tests/test_post_norm_flag.py \
  KE_HOACH_SUA_2026-08-04.md DA_TRIEN_KHAI_SUA_2026-08-04.md \
  BAI_TOAN_VA_MUC_TIEU_2026-08-04.md CHAY_VM_BA_TANG_2026-08-04.md

git commit -m "feat(ba-tang): pha 2 khong nhan + tang nhanh (M1) + ngan hang che do (M2)

- P1: engine gac ranh gioi chuyen_hieu_chinh, tu chuyen 1 chi hap_thu_khong_nhan
- P2: prequential test-then-train + troi TRONG chuyen + thu tu thoi gian
- DL1: pha2.test_chung — moi chuyen cham tren CUNG tap test day du
- M1 tang_nhanh.py (~3 KB, m_t/v_t khong nhan) · M2 ngan_hang_che_do.py (nap snapshot khi gap lai)
- A1 co memory.post_norm · A2 guard revisit can drift · A3 lich_tu_cfg mot nguon
- O2 thoi_gian_hoi_phuc + O3 o3_preq10_loi_ich vao metrics_revisit.json
- fix: khop ngan hang bang m_tuoi (chong nhiem chuyen truoc)
- 49 test moi; mo_phong_ba_tang.py xac nhan chuoi U0->U1->U2 ra dung tin hieu"

git push -u origin feat/cms-ba-tang
```

## A3. Kiểm đã lên

```bash
git log origin/feat/cms-ba-tang --oneline -1
```

Phải in đúng commit vừa tạo. `git push` hỏi mật khẩu → dán **Personal Access Token** GitHub.

---

# PHẦN B — Bật máy ảo

## B1. Xem máy đang có

```bash
gcloud compute instances list
```

| STATUS | Nghĩa | Làm gì |
|---|---|---|
| `RUNNING` | đang chạy, đang tính tiền | vào luôn (B3) |
| `TERMINATED` | đã tắt, chỉ tính tiền ổ đĩa | `start` (B2) — **ưu tiên**: repo + venv + dataset 630 MB còn nguyên |
| không thấy gì | đã xoá | tạo mới (B2b) |

## B2. Bật máy cũ

Thay `uavcl-slda` / `us-east1-b` bằng **đúng NAME và ZONE mà B1 in ra** — `uavcl-slda` chỉ là
tên của campaign trước, rất có thể đã xoá:

```bash
gcloud compute instances start uavcl-slda --zone=us-east1-b
```

Đợi ~30 giây rồi SSH. Báo `was not found` = tên/zone sai hoặc máy đã xoá → quay lại B1, hoặc
tạo mới ở B2b.

## B2b. Hoặc tạo máy mới

```bash
gcloud compute instances create uavcl-batang \
  --zone=us-east1-b --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB
```

Kẹt quota (free-trial 8 vCPU/region) → đổi `--zone` sang `asia-east1-a` hoặc `us-central1-a`.

## B3. Vào máy

```bash
gcloud compute ssh uavcl-batang --zone=us-east1-b
```

`Connection refused` ngay sau khi tạo = chưa boot xong, đợi 60–90 s.

---

# PHẦN C — Lấy code mới về VM

## C1. Máy cũ (đã có repo)

```bash
cd ~/RaybanMeta
git fetch origin
git checkout feat/cms-ba-tang
git pull origin feat/cms-ba-tang
```

Báo *"local changes would be overwritten"* → `git stash` rồi checkout lại.

## C2. Máy mới tinh

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git tmux
cd ~ && git clone https://github.com/VuMinhHien1234/RaybanMeta.git
cd RaybanMeta && git checkout feat/cms-ba-tang
cd uav-continual-learning
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

## C3. ⛔ Kiểm ĐÚNG code — đừng bỏ qua

```bash
cd ~/RaybanMeta/uav-continual-learning
git branch --show-current
ls configs/revisit_*.yaml | wc -l
ls src/uavcl/models/tang_nhanh.py src/uavcl/models/ngan_hang_che_do.py src/uavcl/data/revisit.py
grep -c "m_tuoi" src/uavcl/models/ngan_hang_che_do.py
grep -c "hap_thu_khong_nhan" src/uavcl/engine.py
grep -c "o3_preq10" scripts/run_g1.py
```

| Lệnh | Phải ra | Ra sai nghĩa là |
|---|---|---|
| `git branch --show-current` | `feat/cms-ba-tang` | sai nhánh |
| `ls configs/revisit_* \| wc -l` | `5` | pull thiếu |
| `ls ...tang_nhanh.py...` | không lỗi | chưa có M1/M2 |
| `grep -c m_tuoi` | **3** | **ra 0 = code CŨ, ngân hàng khớp bằng EMA nhiễm chuyến trước → O3 sai mà vẫn chạy trót lọt** |
| `grep -c hap_thu_khong_nhan` | **3** | ra 0 = pha 2 vẫn ăn nhãn → toàn bộ bài toán sai |
| `grep -c o3_preq10` | **6** | ra 0 = không có số O3 chính |

Hai dòng in đậm quan trọng nhất. Ra `0` → quay lại C1, đừng chạy tiếp.

## C4. Kiểm môi trường

```bash
.venv/bin/python scripts/check_env.py                 # Device: cpu
.venv/bin/python -m pytest -q                         # PHẢI XANH (~1 phút)
```

---

# PHẦN D — ⛔ Smoke test trên VM (~40–60 phút, bắt buộc)

**Năm** chuyến, một seed. Không xem accuracy — xem **đường ống có nối đúng không**.

⚠️ **Tối thiểu là 5 chuyến, không được ít hơn.** Với `chu_ky=4` thì lịch bay chỉ bắt đầu lặp
điều kiện ở chuyến thứ 4 (`c3:50%/m1(lần 2)`). Đặt `num_tasks=3` sẽ bị guard chặn ngay:

```
ValueError: Lịch bay KHÔNG có chuyến nào lặp lại chế độ điều kiện cũ
            -> không đo được 'Lợi ích khi quay lại' (mục tiêu O3)
```

Đó là guard chạy ĐÚNG (chết sớm thay vì chạy xong mới biết O3 rỗng), không phải bug.

| `num_tasks` | Lịch | Bank thấy gì |
|---|---|---|
| 3 | c0 c1 c2 | ⛔ bị chặn — không có lặp |
| 4 | + `c3:m1(lần 2)` | 1 lần GẶP LẠI |
| **5** | + `c4:m0` (chế độ MỚI với bank) | 1 lần GẶP LẠI **+ 1 lần phải KHÔNG khớp** ← chọn cái này |

Chọn 5 vì nó kiểm cả hai chiều: bank phải nhận ra m1 ở c3, và phải **không** nhận nhầm m0 ở
c4 thành m1/m2 — đúng cái bẫy ⭐ của M4.

```bash
cd ~/RaybanMeta/uav-continual-learning

.venv/bin/python scripts/mo_phong_ba_tang.py

.venv/bin/python scripts/run_g1.py \
  --config configs/revisit_U2_nganhang.yaml \
  --set seed=0 data.num_tasks=5 log.dir=./artifacts_smoke_batang \
  2>&1 | tee run_smoke_batang.log
```

Lần đầu tải dataset ~630 MB — chậm, bình thường.

**Bấm giờ chuyến đầu.** `test_chung` bắt mỗi chuyến chấm trên đủ 6300 ảnh test, cộng ~6300
ảnh train → ~12.600 lượt forward ViT-S mỗi chuyến. Đo thời gian một chuyến rồi **× 12 × 9 run**
để ra ETA thật của track chính, thay cho con số ước lượng ~1,1–1,4 h/run ở đầu file.

## Năm dòng phải thấy trong log

```bash
grep -E "\[revisit\] test_chung|\[ba_tang\] tầng NHANH bật|ĐÃ CHỐT mốc pha 1|\[ba_tang\] tầng trung|BAY LẶP LẠI" run_smoke_batang.log
```

| Dòng | Nghĩa | Không thấy = |
|---|---|---|
| `[revisit] test_chung BẬT — … ảnh test` | DL1 đang chạy, mọi chuyến chấm cùng tập | `pha2` chưa được đọc |
| `[ba_tang] tầng NHANH bật: kieu=day_du` | M1 sống | config sai |
| `[ba_tang] ĐÃ CHỐT mốc pha 1 (m0/v0)` | P1 gác đúng ranh giới chuyến hiệu chỉnh | pha 2 vẫn ăn nhãn ⛔ |
| `[ba_tang] tầng trung: N chế độ` | M2 sống | `ngan_hang.enabled` không tới |
| `== BAY LẶP LẠI` | chạy tới cuối, có thước đo O1/O2/O3 | xem lỗi cuối log |

Thêm một kiểm tra chống bịp — **chứng minh pha 2 thật sự không nhãn**:

```bash
grep -n "count_raw\|hap_thu_khong_nhan" run_smoke_batang.log | head
```

Dọn sạch trước khi chạy thật: `rm -rf artifacts_smoke_batang run_smoke_batang.log`

---

# PHẦN E — Chạy thật

## E0. ⚡ Cách nhanh: MỘT lệnh, để qua đêm

`scripts/chay_dem_ba_tang.sh` chạy hết E1→E5, **tự làm cả hai chỗ trước đây phải hiệu chỉnh
tay** và tự dừng ở hai cửa chặn:

```bash
cd ~/RaybanMeta/uav-continual-learning
tmux new -s dem
nohup bash scripts/chay_dem_ba_tang.sh > dem.log 2>&1 &
tail -f dem.log
```

Rời tmux: `Ctrl+B` rồi `D`. Tắt Mac không sao.

| Nó tự làm gì | Cách làm |
|---|---|
| `tang_nhanh.kieu` | đọc `ket_luan_t1` trong `artifacts_t1/t1_t2.json` → `TRUC_CHUNG` thì `truc` + nối `truc_json`, còn lại giữ `day_du` |
| `ngan_hang.nguong` | `dich_dieu_kien / 4` lấy từ chính file T1 (giải thích ở E4) |
| ⛔ cửa chặn T1 | `KHONG_CO_TRUC` → thoát mã 2, không chạy gì thêm |
| ⛔ cửa chặn U1/U0 | chạy `tom_tat_ba_tang.py --cua-chan`; U1 không hơn U0 → thoát mã 3, **không** chạy U2 |
| tổng kết | in bảng + con số công bố `preq10(U2) − preq10(U1)` ghép cặp theo seed |

Tuỳ chọn:

```bash
SEEDS="0 1"  bash scripts/chay_dem_ba_tang.sh      # chạy ít seed cho nhanh
ORACLE=1     bash scripts/chay_dem_ba_tang.sh      # chạy thêm arm1/arm2 có nhãn
NGUONG=0.37  bash scripts/chay_dem_ba_tang.sh      # ép ngưỡng, bỏ phần tự tính
```

**Chạy lại lần hai là an toàn.** Mọi run dùng `--skip-existing`, run nào đã có `metrics.json`
thì bỏ qua. VM chết giữa chừng → SSH lại, chạy y hệt lệnh cũ, nó đi tiếp từ chỗ dở.

Xem tiến độ từ Mac bất cứ lúc nào:

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'tail -40 ~/RaybanMeta/uav-continual-learning/dem.log'
```

Muốn tự tay từng bước (hoặc cần gỡ lỗi) thì đọc tiếp E1–E6 dưới đây — đó chính là những gì
script làm.

---

## E1–E6 — làm tay, ĐÚNG THỨ TỰ

## E1. T1/T2 — trục điều kiện (~15 phút) · CỬA CHẶN ĐẦU TIÊN

```bash
cd ~/RaybanMeta/uav-continual-learning
mkdir -p artifacts_t1
.venv/bin/python scripts/do_truc_dieu_kien.py \
  --config configs/drift_slda_arm3_dexuat.yaml \
  --json artifacts_t1/t1_t2.json 2>&1 | tee run_t1.log
```

| Kết luận in ra | Nghĩa | Làm gì |
|---|---|---|
| `✅ TRUC_CHUNG` (cos ≥ ngưỡng tốt, PC1 cao) | điều kiện dịch theo MỘT trục | sửa `tang_nhanh.kieu: truc` + bỏ comment `truc_json: artifacts_t1/t1_t2.json` trong **U1 và U2** — chống bẫy tỷ lệ lớp T2 |
| `⚠️ CÓ TRỤC NHƯNG KHÔNG TRỌN` | trục yếu | giữ `kieu: day_du`, ghi vào phần hạn chế |
| `❌ KHÔNG CÓ TRỤC CHUNG (cos < 0,3)` | ⛔ | **DỪNG hệ ba tầng** — căn chỉnh một trục không có cơ sở, báo lại trước khi đốt đêm máy |

## E2. Bench O4 (~1 phút)

```bash
.venv/bin/python scripts/bench_ba_tang.py 2>&1 | tee bench_ba_tang.log
```

Phải thấy `TỔNG thêm` **< 10 MB** và `TỔNG thêm/khung` **< 1,67 ms** (5% của 33,3 ms).
Vượt trần → hệ không chạy được trên biên, phải báo trong kết quả.

## E3. U0 + U1, 3 seed (~7–9 giờ, để qua đêm) · CỬA CHẶN THỨ HAI

```bash
tmux new -s batang
cd ~/RaybanMeta/uav-continual-learning

nohup bash -c '
for S in 0 1 2; do
  for U in U0_dongbang U1_tangnhanh; do
    echo "=== $U seed $S  bat dau $(date +%H:%M:%S)"
    .venv/bin/python scripts/run_g1.py \
      --config configs/revisit_${U}.yaml \
      --set seed=$S log.dir=./artifacts_revisit_${U}_s$S \
      > run_${U}_s$S.log 2>&1 || echo "    !!! LOI o $U seed $S"
    echo "    xong $(date +%H:%M:%S)  ->  $(grep -o "Lợi ích quay lại PREQ10.*" run_${U}_s$S.log | head -1)"
  done
done
echo "=== U0+U1 XONG $(date +%H:%M:%S)"
' > batang_campaign.log 2>&1 &

tail -f batang_campaign.log
```

Rời tmux: `Ctrl+B` rồi `D`. Tắt Mac không sao.

⛔ **Cửa chặn:** U1 **không hơn** U0 (acc đường chéo + O2 hồi phục) → **DỪNG, không chạy U2**.
Tầng nhanh không cứu được trôi thì ngân hàng chế độ không có gì để nhớ.

## E4. Hiệu chỉnh `nguong` — tính từ T1, không phải từ log U1

> ⚠️ **Sửa so với bản đầu.** Tôi từng ghi "đọc `độ trôi hiện tại` trong log U1" — **không dùng
> được**: dòng đó chỉ in **một lần ở cuối run**, phản ánh chuyến CUỐI. Với `chu_ky=4` và 12
> chuyến thì chuyến cuối là `c11:50%`, không phải 100%. Lấy số đó làm ngưỡng là sai cỡ.

Nguồn đúng là `artifacts_t1/t1_t2.json`, trường `dich_dieu_kien` — độ dịch **trên trục điều
kiện** khi trôi đi từ 0% lên 100%, đo bằng chính backbone trên chính dataset:

```bash
python3 -c "import json; d=json.load(open('artifacts_t1/t1_t2.json')); \
print('dich=%.4f -> nguong=%.4f' % (d['dich_dieu_kien'], d['dich_dieu_kien']/4))"
```

Vì sao chia 4, với `n_mode: 3` (các mức 0% · 50% · 100%):

```
khoảng cách hai chế độ KỀ NHAU trên trục = dich_dieu_kien / (n_mode − 1) = dich/2
ngưỡng đặt ở NỬA khoảng đó                                              = dich/4
```

Đủ rộng để tha thứ trôi trong-chuyến (`bien_do=0.25` → ±12,5%) cộng nhiễu ước lượng `m_tuoi`;
đủ hẹp để **không gộp hai chế độ kề nhau**. Đây là tham số nguy hiểm nhất — đặt nhỏ thì nở
chế độ vô tội vạ, đặt lớn thì gộp hết làm một.

Cửa chặn hậu kiểm: log U2 phải in `số chế độ (số điều kiện THẬT: 3)` với con số gần 3.
Lệch gấp đôi là hỏng — run_g1 tự in cảnh báo ⚠️.

## E5. U2, 3 seed (~3,5–4 giờ)

Truyền `nguong` bằng `--set`, khỏi sửa file config (chạy lại tái lập được, và log lưu luôn
giá trị đã dùng):

```bash
NGUONG=$(python3 -c "import json; print(round(json.load(open('artifacts_t1/t1_t2.json'))['dich_dieu_kien']/4, 4))")
echo "nguong = $NGUONG"

nohup bash -c '
for S in 0 1 2; do
  echo "=== U2 seed $S  bat dau $(date +%H:%M:%S)"
  .venv/bin/python scripts/run_g1.py \
    --config configs/revisit_U2_nganhang.yaml --skip-existing \
    --set seed=$S log.dir=./artifacts_revisit_U2_nganhang_s$S slda.ngan_hang.nguong='"$NGUONG"' \
    > run_U2_nganhang_s$S.log 2>&1 || echo "    !!! LOI seed $S"
  echo "    xong $(date +%H:%M:%S)  ->  $(grep -o "Lợi ích quay lại PREQ10.*" run_U2_nganhang_s$S.log | head -1)"
done
echo "=== U2 XONG $(date +%H:%M:%S)"
' > batang_u2.log 2>&1 &
```

Nếu T1 kết luận `TRUC_CHUNG` thì thêm vào cùng cụm `--set`:
`slda.tang_nhanh.kieu=truc slda.tang_nhanh.truc_json=artifacts_t1/t1_t2.json`

## E6. Oracle có nhãn + ablation N1 (máy khác / đêm khác, không chặn track chính)

```bash
# trần trên có nhãn — nối mạch D10
for S in 0 1 2; do
  .venv/bin/python scripts/run_g1.py --config configs/revisit_arm1_lam1.yaml \
    --set seed=$S log.dir=./artifacts_revisit_arm1_s$S > run_arm1_s$S.log 2>&1
  .venv/bin/python scripts/run_g1.py --config configs/revisit_arm2_quen.yaml \
    --set seed=$S log.dir=./artifacts_revisit_arm2_s$S > run_arm2_s$S.log 2>&1
done

# N1 — ablation post_norm (0 code, nhờ A1)
for S in 0 1 2; do
  .venv/bin/python scripts/run_g1.py --config configs/drift_titans_gates_eta_thap.yaml \
    --set memory.post_norm=false seed=$S log.dir=./artifacts_n1_s$S > run_n1_s$S.log 2>&1
done
```

> **Bẫy `--set`:** dùng **MỘT** `--set` rồi liệt kê mọi cặp phía sau. Lặp `--set` hai lần sẽ
> **rớt** các cặp trước đó (argparse chỉ giữ lần cuối).

---

# PHẦN F — Theo dõi từ Mac

```bash
VM=uavcl-batang; Z=us-east1-b
gcloud compute ssh $VM --zone=$Z -- 'tail -30 ~/RaybanMeta/uav-continual-learning/batang_campaign.log'
gcloud compute ssh $VM --zone=$Z -- 'pgrep -af run_g1.py | wc -l'      # 0 = xong
```

Vòng lặp tự in O3 PREQ10 sau mỗi run → chỉ cần nhìn `batang_campaign.log`.

---

# PHẦN G — Lấy kết quả về Mac

```bash
gcloud compute ssh uavcl-batang --zone=us-east1-b -- 'sudo bash -c "shopt -s nullglob; D=\$(ls -d /home/*/RaybanMeta/uav-continual-learning 2>/dev/null | head -1); cd \$D && tar czf /tmp/res_batang.tgz --exclude=resume_checkpoint.pt artifacts_revisit_* artifacts_t1 artifacts_n1_* *.log && chmod 644 /tmp/res_batang.tgz && echo GOI_XONG \$D"'

mkdir -p ~/Desktop/Raybanmeta/result_test/batang
gcloud compute scp uavcl-batang:/tmp/res_batang.tgz ~/Desktop/Raybanmeta/result_test/batang/ --zone=us-east1-b
cd ~/Desktop/Raybanmeta/result_test/batang && tar xzf res_batang.tgz

cd ~/Desktop/Raybanmeta/result_test/batang
python3 ~/Desktop/Raybanmeta/uav-continual-learning/scripts/tom_tat_ba_tang.py .
```

`GOI_XONG /home/...` = gói thành công (`Connection closed` sau đó là bình thường).

`tom_tat_ba_tang.py` chỉ dùng thư viện chuẩn → chạy bằng `python3` thường, không cần venv,
không cần torch. Nó in bảng O1/O3/O2/số chế độ theo từng seed **và** con số công bố
`preq10(U2) − preq10(U1)` ghép cặp theo seed, kèm σ.

## Đọc kết quả — luật công bố

| Đọc gì | Ở đâu |
|---|---|
| ⭐ **Con số công bố = `o3_preq10_loi_ich`(U2) − `o3_preq10_loi_ich`(U1), cùng seed, 3 seed + σ** | `artifacts_revisit_U*/results/*/metrics_revisit.json` |
| O1 (chống trôi) | acc đường chéo U1 vs U0 |
| O2 (hồi phục) | `thoi_gian_hoi_phuc` — **chỉ so khi acc hai arm gần nhau**; U0 đứng im ở mức thấp cũng "hồi phục nhanh" giả |
| O4 (chi phí) | `bench_ba_tang.log` |
| Trần có nhãn | arm1/arm2 |

Vì sao trừ U1 chứ không trừ U0: cột U1 trong mô phỏng đã cho +0,077 **chỉ nhờ carryover EMA** —
tức "lợi ích giả". U1 làm đối chứng khử đúng phần đó, thành phép so một biến.

**Không** dùng `O3 chéo` (R[t][t]) làm số chính — nó chấm SAU khi hấp thụ cả chuyến, lúc tầng
nhanh đã tự hội tụ, nên mù với ngân hàng.

---

# PHẦN H — Xoá / tắt VM

```bash
gcloud compute instances stop   uavcl-batang --zone=us-east1-b   # còn dùng tiếp: giữ đĩa (~$4/tháng)
gcloud compute instances delete uavcl-batang --zone=us-east1-b   # xong hẳn
```

---

# Phụ lục — bẫy thường gặp

| Triệu chứng | Xử |
|---|---|
| `git add` chết vì `index.lock` | `rm -f .git/index.lock .git/HEAD.lock .git/objects/maintenance.lock` |
| `stream_type=revisit cần data.drift.enabled=true` | Guard A2 cố ý chết sớm — revisit không trôi thì mọi chuyến cùng điều kiện, `mode_that` vô nghĩa |
| `Lịch bay KHÔNG có chuyến nào lặp lại chế độ` | `num_tasks` < 4 với `chu_ky=4`. Smoke phải để `num_tasks=5`. Guard chạy đúng |
| `grep -c m_tuoi` ra 0 trên VM | VM chạy code cũ → O3 nhiễm chuyến trước, kết quả rác mà không báo lỗi |
| log U2 in `số chế độ` lệch gấp đôi số điều kiện thật | `nguong` sai → quay lại E4 |
| seed override không ăn | Lặp `--set` hai lần → rớt. Gộp **1 `--set` nhiều cặp** |
| `Connection closed` sau lệnh ssh 1 dòng | BÌNH THƯỜNG |
| `Connection refused` ngay sau tạo VM | chưa boot xong, đợi 60–90 s |
| VM chết giữa chừng | SSH lại, chạy đúng lệnh cũ thêm `--resume` — ⚠️ **resume CHƯA lưu state ba tầng (m_t/ngân hàng)**, run revisit ngắn nên chạy lại từ đầu thì sạch hơn |

## Đường dẫn nhanh

- Nhánh: `feat/cms-ba-tang`
- Track chính: `configs/revisit_U{0,1,2}_*.yaml` · Oracle: `configs/revisit_arm{1,2}_*.yaml`
- Kết quả: `<log.dir>/results/<...>/metrics.json` + **`metrics_revisit.json`** (O1/O2/O3 + chuỗi prequential)
- Bối cảnh: `BAI_TOAN_VA_MUC_TIEU_2026-08-04.md` (§2/§3 pha 2 không nhãn) ·
  `KE_HOACH_SUA_2026-08-04.md` (cửa chặn) · `DA_TRIEN_KHAI_SUA_2026-08-04.md` (audit + giới hạn)
