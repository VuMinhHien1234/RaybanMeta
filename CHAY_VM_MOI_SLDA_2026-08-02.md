# VM mới → campaign B2 (SLDA ablation) — hướng dẫn từng bước

Ngày: 2026-08-02 · Nhánh: `feat/slda-ablation` · VM: `uavcl-slda`

10 bước, từ commit tới xoá máy. **Bước 6 là chạy thử — đừng bỏ qua**, nó bắt được
mọi lỗi hạ tầng trong 15 phút thay vì phát hiện sau 8 tiếng.

| Bước | Việc | Thời gian | Ở đâu |
|---|---|---|---|
| 0 | Commit + push nhánh | 2 ph | Mac |
| 1 | Tạo VM | 2 ph | Mac |
| 2 | Cài gói hệ thống | 2 ph | VM |
| 3 | Clone + chọn nhánh | 2 ph | VM |
| 4 | Cài Python env | 6 ph | VM |
| 5 | **Kiểm tra trước khi đốt giờ** | 3 ph | VM |
| 6 | **CHẠY THỬ** — 1 arm, 3 task | 15 ph | VM |
| 7 | Chạy thật — 12 run | 4–8 h | VM |
| 8 | Theo dõi | — | Mac |
| 9 | Lấy kết quả về | 5 ph | Mac |
| 10 | Xoá VM | 1 ph | Mac |

---

# BƯỚC 0 — Trên Mac: commit + push

```bash
cd ~/Desktop/Raybanmeta
rm -f .git/index.lock .git/HEAD.lock 2>/dev/null

git checkout -b feat/slda-ablation
git add -A
git commit -m "feat(slda): ablation cov_mode + do chi phi trien khai (B2, B6)

- slda.py: cov_mode = streaming | identity | frozen
  · identity: Sigma=I -> TUONG DUONG NCM (chung duong code) = doi chung sach nhat
  · frozen:   Sigma dong bang sau cov_freeze_after task, mu_c van cap nhat
- slda.py: stats_dtype float64|float32 (MPS/NPU bien khong ho tro fp64)
- slda.py: memory_report() do byte THAT theo dtype
- methods.py: SLDA.end_task goi on_task_end + in bo nho moi task
- run_g1.py: doc slda.cov_mode / cov_freeze_after tu config
- 4 config arm: ncm_frozen + slda_{streaming,identity,frozen}
- scripts/bench_cost.py: do bo nho/do tre, khong can dataset
- tests/test_slda_cov_mode.py: 19 test

Ket qua B6: Titans 102.7 MB vs SLDA 1.39 MB (74x); do tre 307x cung thiet bi.
Mac dinh cov_mode=streaming + stats_dtype=float64 -> ket qua 0.8263 bat bien."

git push -u origin feat/slda-ablation
```

**Xác nhận đã lên remote:**

```bash
git branch -a | grep slda-ablation
```

Phải thấy **cả hai** dòng:

```
* feat/slda-ablation
  remotes/origin/feat/slda-ablation
```

Thiếu dòng `remotes/origin/...` = chưa push thành công, **đừng đi tiếp**.

---

# BƯỚC 1 — Tạo VM

```bash
gcloud compute instances create uavcl-slda \
  --zone=us-east1-b --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB
```

- `e2-standard-8` = 8 vCPU / 32 GB, đủ cho ViT-S + RESISC45
- Kẹt quota → đổi `--zone` sang `asia-east1-a` hoặc `us-central1-a`,
  và nhớ đổi ở **mọi lệnh sau**

Đợi **60–90 giây** rồi vào máy:

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b
```

`Connection refused` = chưa boot xong, đợi thêm rồi thử lại.

---

# BƯỚC 2 — Cài gói hệ thống (trong VM)

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git tmux
```

---

# BƯỚC 3 — Clone và chọn nhánh (trong VM)

```bash
git clone https://github.com/VuMinhHien1234/RaybanMeta.git
cd RaybanMeta
git checkout feat/slda-ablation
```

Repo private → `git clone` hỏi mật khẩu → dán **Personal Access Token** GitHub
(không phải mật khẩu tài khoản).

## Xác nhận đúng nhánh và đúng code

```bash
git branch --show-current
ls uav-continual-learning/configs/ | grep -E "slda_resisc45|ncm_resisc45"
```

Phải ra:

```
feat/slda-ablation
ncm_resisc45_frozen.yaml
slda_resisc45_frozen.yaml
slda_resisc45_identity.yaml
slda_resisc45_streaming.yaml
```

**Thiếu file config nào = sai nhánh.** Quay lại `git fetch origin && git checkout feat/slda-ablation`.

---

# BƯỚC 4 — Cài Python env (trong VM, ~6 phút)

```bash
cd ~/RaybanMeta/uav-continual-learning

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install titans-pytorch==0.5.5
pip install -e .
```

`titans-pytorch` ghim đúng `0.5.5` — bản đã kiểm chứng. Đổi bản khác thì mọi kết quả
Titans không so được.

---

# BƯỚC 5 — Kiểm tra trước khi đốt giờ (CHẶN)

```bash
python scripts/check_env.py
pytest -q
git branch --show-current
```

| Kiểm | Phải ra |
|---|---|
| `check_env.py` | `Device: cpu` |
| `pytest -q` | **xanh hết**, không fail nào |
| `git branch` | `feat/slda-ablation` |

**Bất kỳ dòng nào sai → dừng lại, gửi lỗi cho tôi.** Đừng chạy campaign trên env hỏng.

---

# BƯỚC 6 — CHẠY THỬ (~15 phút) ⭐ ĐỪNG BỎ QUA

Chạy **một arm, ba task, một seed** để kiểm toàn bộ đường ống trước khi đốt 8 tiếng.

Lần chạy này còn **tải dataset RESISC45 từ HuggingFace** (31.500 ảnh) — đây là chỗ hay
hỏng nhất và tốn thời gian nhất. Tải xong một lần thì 11 run sau dùng lại.

```bash
cd ~/RaybanMeta/uav-continual-learning && source .venv/bin/activate

python scripts/run_g1.py \
  --config configs/slda_resisc45_frozen.yaml \
  --set seed=0 data.num_tasks=3 train.eval_future=false log.dir=./artifacts_smoke_b2 \
  2>&1 | tee run_smoke_b2.log
```

Chọn arm `frozen` để chạy thử vì nó dùng **nhiều đường code nhất** — vừa có `cov_mode`,
vừa có `on_task_end`, vừa có `memory_report`. Chạy được arm này thì ba arm kia chắc chắn chạy.

## Năm thứ phải thấy trong log

```
[slda] shrinkage=0.0001 cov_mode=frozen cov_freeze_after=1
[slda] ĐÓNG BĂNG Σ sau task 0 (cov_mode=frozen, cov_freeze_after=1)
[slda] cov_mode=frozen | bộ nhớ = 1.xxx MB (gram 1.180 + means 0.138 + cache 0.069)
== Ma trận accuracy ...
  Average Accuracy   : 0.8xxx
```

| Dấu hiệu | Nghĩa |
|---|---|
| Có đủ 5 dòng trên | ✅ Đường ống thông, sang bước 7 |
| Thiếu `cov_mode=frozen` | ❌ Sai nhánh — quay lại bước 3 |
| Thiếu `ĐÓNG BĂNG Σ` | ❌ `on_task_end` không được gọi — báo tôi |
| Accuracy < 0,5 | ⚠️ Bất thường với 3 task × 15 lớp — báo tôi trước khi chạy tiếp |
| Lỗi tải dataset | ❌ Kiểm mạng / HF token — xem mục Sự cố cuối file |

## Dọn dẹp sau khi thử

```bash
rm -rf artifacts_smoke_b2 run_smoke_b2.log
```

Xoá để không lẫn vào kết quả thật khi gộp sau này. **Dataset đã tải thì giữ nguyên.**

---

# BƯỚC 7 — Chạy thật: 4 arm × 3 seed

```bash
tmux new -s slda
cd ~/RaybanMeta/uav-continual-learning && source .venv/bin/activate

nohup bash -c '
for ARM in ncm_resisc45_frozen slda_resisc45_streaming slda_resisc45_identity slda_resisc45_frozen; do
  for S in 0 1 2; do
    echo "=== $ARM seed $S  bat dau $(date +%H:%M)"
    .venv/bin/python scripts/run_g1.py \
      --config configs/$ARM.yaml \
      --set seed=$S train.eval_future=false log.dir=./artifacts_b2_${ARM}_s$S \
      > run_b2_${ARM}_s$S.log 2>&1
    echo "    xong $(date +%H:%M)  ->  $(grep -o "Average Accuracy.*" run_b2_${ARM}_s$S.log | head -1)"
  done
done
echo "=== XONG TAT CA $(date +%H:%M)"
' > b2_campaign.log 2>&1 &

tail -f b2_campaign.log
```

Rời tmux: `Ctrl+B` rồi `D`. Thoát VM: `exit`. Run vẫn chạy tiếp.

Vòng lặp tự in accuracy sau mỗi run — chỉ cần nhìn `b2_campaign.log` là thấy tiến độ
và kết quả, không phải mở từng log.

## Cũng nên chạy bench trên máy này (2 phút)

VM là **CPU thuần**, không có MPS — nên bench ở đây **sạch hơn trên Mac**: mọi dòng
cùng một thiết bị, so chéo được cả cột thời gian chứ không chỉ bộ nhớ.

```bash
# tab khác, hoặc sau khi campaign xong
python scripts/bench_cost.py --json bench_vm_vits.json | tee bench_vm_vits.log
python scripts/bench_cost.py --dim 1024 --json bench_vm_vitl.json | tee bench_vm_vitl.log
```

Đây là số **nên đưa vào báo cáo** thay cho số đo trên Mac.

---

# BƯỚC 8 — Theo dõi từ Mac

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'tail -20 ~/RaybanMeta/uav-continual-learning/b2_campaign.log'
```

Còn chạy hay xong:

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'pgrep -af run_g1.py | wc -l'
```

`0` = xong hết.

---

# BƯỚC 9 — Lấy kết quả về Mac

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'sudo bash -c "shopt -s nullglob; D=\$(ls -d /home/*/RaybanMeta/uav-continual-learning 2>/dev/null | head -1); cd \$D && tar czf /tmp/res_b2.tgz artifacts_b2_* run_b2_*.log b2_campaign.log bench_vm_* && chmod 644 /tmp/res_b2.tgz && echo GOI_XONG \$D"'

mkdir -p ~/Desktop/Raybanmeta/result_test/b2
gcloud compute scp uavcl-slda:/tmp/res_b2.tgz ~/Desktop/Raybanmeta/result_test/b2/ --zone=us-east1-b

cd ~/Desktop/Raybanmeta/result_test/b2 && tar xzf res_b2.tgz

cd ~/Desktop/Raybanmeta
python3 uav-continual-learning/scripts/compare_all.py result_test/b2
```

---

# BƯỚC 10 — Xoá VM

```bash
gcloud compute instances delete uavcl-slda --zone=us-east1-b
```

Dùng `delete` chứ không `stop` — máy tạo riêng cho campaign này, xong là bỏ.
Máy tắt vẫn tính tiền ổ đĩa (~$4/tháng/100GB).

---

# Bảng cần điền

| Arm | Bộ đọc | Acc (3 seed) | Forget | AAA | Bộ nhớ |
|---|---|---|---|---|---|
| 1 | NCM thuần | | | | ~0,14 MB |
| 2 | SLDA Σ streaming | | | | 1,39 MB |
| 3 | SLDA Σ = I | | | | ~0,14 MB |
| 4 | SLDA Σ đóng băng | | | | 1,39 MB |

## Bốn kịch bản

| Quan sát | Kết luận |
|---|---|
| arm 3 ≈ arm 1 ≈ 0,71 · arm 2 = 0,83 | ✅ **Hiệp phương sai là nguyên nhân** |
| arm 3 ≈ arm 2 | ❌ Σ không quan trọng — cải thiện đến từ chỗ khác |
| **arm 4 ≈ arm 2** | Σ **không cần** streaming → bỏ được `gram`, **cắt 85% bộ nhớ** |
| arm 4 < arm 2 rõ rệt | Σ **cần** streaming → chi phí O(D²) là bắt buộc |

**Bẫy phải kiểm:** arm 1 và arm 3 đi **hai đường code khác nhau** nhưng về toán tương đương
(đã kiểm 0/300 sai lệch). Nếu chạy thật mà lệch nhiều → **có bug**, phải tìm ra trước khi
tin bất kỳ kết luận nào.

---

# Sự cố thường gặp

| Triệu chứng | Xử lý |
|---|---|
| `git clone` hỏi mật khẩu | Dán **Personal Access Token**, không phải mật khẩu tài khoản |
| Thiếu file `slda_resisc45_*.yaml` | Sai nhánh → `git fetch origin && git checkout feat/slda-ablation` |
| `ModuleNotFoundError: uavcl` | Quên `pip install -e .` hoặc quên `source .venv/bin/activate` |
| Lỗi tải dataset HuggingFace | Thử lại; nếu bị rate-limit thì đặt `export HF_TOKEN=...` |
| `Connection closed` sau lệnh ssh một dòng | **Bình thường** — chạy xong tự đóng |
| `Connection refused` ngay sau khi tạo VM | Chưa boot xong → đợi 60–90 giây |
| Không thấy dòng `[slda] ...` | Sai nhánh, hoặc config không có khối `slda:` |
| Máy hết RAM | Giảm `--set data.batch_size=16` |
