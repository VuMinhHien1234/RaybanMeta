# Chạy campaign trôi + λ trên VM — từ đầu đến cuối

Ngày 2026-08-03 · sau khi cửa chặn ĐẠT (NCM 0,6900 → 0,5310, ảnh nhìn đúng)

| | |
|---|---|
| Nhánh mới | `feat/drift-lambda` (tách từ `feat/slda-ablation`) |
| Số run | 15 (5 arm × 3 seed) |
| Thời gian | ~10 giờ nếu 1 máy · ~3,5 giờ nếu chia 3 máy |
| Repo | `https://github.com/VuMinhHien1234/RaybanMeta.git` |

---

# PHẦN A — Trên Mac: đẩy code lên GitHub

## A1. Bỏ qua file rác

```bash
cd ~/Desktop/Raybanmeta
grep -q "^.DS_Store" .gitignore || echo ".DS_Store" >> .gitignore
git rm --cached .DS_Store 2>/dev/null
```

## A2. Tạo nhánh mới và commit

```bash
git checkout -b feat/drift-lambda

git add .gitignore \
  uav-continual-learning/src/uavcl/models/slda.py \
  uav-continual-learning/src/uavcl/data/drift.py \
  uav-continual-learning/src/uavcl/data/stream.py \
  uav-continual-learning/src/uavcl/data/loaders.py \
  uav-continual-learning/src/uavcl/methods.py \
  uav-continual-learning/scripts/run_g1.py \
  uav-continual-learning/scripts/calibrate_drift.py \
  uav-continual-learning/scripts/collect_vms.sh \
  uav-continual-learning/tests/test_slda_decay.py \
  uav-continual-learning/configs/drift_*.yaml \
  uav-continual-learning/troi_mau.png \
  uav-continual-learning/cua_chan.log \
  *.md

git commit -m "feat(slda): he so quen lambda + dataset troi domain-incremental

- decay_mean (theo-lop) / decay_cov (toan cuc), mac dinh 1.0 -> bat bien voi ket qua cu
- SUA LOI: them cap dem toan cuc feat_sum_g/count_g cho dang thuc Sigma.
  Truoc do gram phan ra toan cuc con feat_sum phan ra theo-lop -> trong so lech ->
  phep tru tru qua tay -> Sigma mat xac dinh duong -> inv(Sigma) ra rac.
- data/drift.py: 5 phep bien doi vat ly, troi TAT DINH + rung +-15%
- stream domain-incremental: moi task du 45 lop, chi khac dieu kien quan sat
- scripts/calibrate_drift.py: hieu chinh cuong do truoc khi dot gio may
- 45 test xanh; hieu chinh: NCM 0.6900 -> 0.5310 tai severity=1.0 (DAT)"

git push -u origin feat/drift-lambda
```

## A3. Kiểm đã lên

```bash
git log origin/feat/drift-lambda --oneline -1
```

Phải in đúng commit vừa tạo. Nếu `git push` hỏi mật khẩu → dán **Personal Access Token**
GitHub, không phải mật khẩu tài khoản.

---

# PHẦN B — Bật máy ảo

## B1. Xem máy đang có

```bash
gcloud compute instances list
```

| Cột STATUS | Nghĩa | Làm gì |
|---|---|---|
| `RUNNING` | đang chạy, đang tính tiền | vào luôn, sang B3 |
| `TERMINATED` | đã tắt, chỉ tính tiền ổ đĩa | `start` ở B2 |
| không thấy máy nào | đã xoá | tạo mới ở B2b |

## B2. Bật máy đã có

```bash
gcloud compute instances start uavcl-slda --zone=us-east1-b
```

Đợi khoảng 30 giây rồi mới SSH được. **Ổ đĩa còn nguyên** — repo, venv, dataset 630 MB
vẫn ở đó, không phải cài lại. Đây là lý do nên `stop` thay vì `delete` nếu còn dùng tiếp.

## B2b. Hoặc tạo máy mới

```bash
gcloud compute instances create uavcl-drift \
  --zone=us-east1-b --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB
```

Kẹt quota → đổi `--zone` sang `asia-east1-a` hoặc `us-central1-a`.
Máy mới thì phải làm thêm B4b (cài đặt từ đầu).

## B3. Vào máy

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b
```

---

# PHẦN C — Lấy code mới về máy ảo

## C1. Nếu repo đã có sẵn (máy cũ)

```bash
cd ~/RaybanMeta
git fetch origin
git checkout feat/drift-lambda
git pull origin feat/drift-lambda
```

`git checkout` báo *"local changes would be overwritten"* → có sửa tay trên VM lúc trước:

```bash
git stash          # cất đi, lấy lại sau bằng git stash pop
git checkout feat/drift-lambda
```

## C2. Nếu máy mới tinh

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git tmux
cd ~
git clone https://github.com/VuMinhHien1234/RaybanMeta.git
cd RaybanMeta
git checkout feat/drift-lambda
```

## C3. ⛔ Kiểm đúng code — đừng bỏ qua bước này

```bash
cd ~/RaybanMeta/uav-continual-learning
git branch --show-current
ls configs/drift_*.yaml | wc -l
ls src/uavcl/data/drift.py scripts/calibrate_drift.py
grep -c "feat_sum_g" src/uavcl/models/slda.py
```

| Lệnh | Phải ra |
|---|---|
| `git branch --show-current` | `feat/drift-lambda` |
| `ls configs/drift_* \| wc -l` | `6` |
| `ls ... drift.py ...` | không lỗi |
| `grep -c feat_sum_g` | **≥ 6** |

Dòng cuối quan trọng nhất: `feat_sum_g` là bản vá Σ sáng nay. **Ra `0` nghĩa là VM đang
chạy code cũ có lỗi** — Σ mất xác định dương, 15 run sẽ cho kết quả rác mà vẫn chạy trót lọt
không báo lỗi gì. Ra 0 thì dừng lại, quay về C1.

## C4. Môi trường Python

Máy cũ đã có `.venv` → bỏ qua. Máy mới:

```bash
cd ~/RaybanMeta/uav-continual-learning
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

Kiểm nhanh:

```bash
.venv/bin/python -c "import torch, timm; print(torch.__version__, timm.__version__)"
```

---

# PHẦN D — ⛔ Smoke test trên VM (10 phút, bắt buộc)

Chạy **một arm, ba task, một seed**. Mục đích không phải xem accuracy — mà xem **đường ống
có nối đúng không**.

```bash
cd ~/RaybanMeta/uav-continual-learning

.venv/bin/python -m pytest tests/test_slda_decay.py -q

.venv/bin/python scripts/run_g1.py \
  --config configs/drift_slda_arm3_dexuat.yaml \
  --set seed=0 data.num_tasks=3 train.eval_future=false log.dir=./artifacts_smoke_drift \
  2>&1 | tee run_smoke_drift.log
```

Lần đầu sẽ tải dataset (~630 MB) — chậm, bình thường.

## Bốn dòng phải thấy trong log

```bash
grep -E "\[drift\] BẬT|\[stream\] DOMAIN|cua so nho|Average Accuracy" run_smoke_drift.log
```

| Dòng | Nghĩa | Không thấy = |
|---|---|---|
| `[drift] BẬT · mode=linear severity=1.0` | trôi đang bật | config sai / nhánh sai |
| `[stream] DOMAIN-incremental: 3 task × 45 lớp` | mọi task đủ 45 lớp | `stream_type` chưa được đọc |
| `n_c` trung bình gần `1/(1−λ)` | λ đang chạy đúng | ❌ báo tôi ngay |
| `Average Accuracy` | chạy tới cuối | xem lỗi cuối log |

⚠️ Nếu thấy cảnh báo `n_c lệch >30%` → **dừng**, đừng chạy tiếp, báo tôi.

Xong thì dọn:

```bash
rm -rf artifacts_smoke_drift run_smoke_drift.log
```

---

# PHẦN E — Chạy thật

## E1. Một máy (~10 giờ, để qua đêm)

```bash
tmux new -s drift
cd ~/RaybanMeta/uav-continual-learning

nohup bash -c '
for ARM in arm1_lam1 arm2_cham arm3_dexuat arm4_nhanh arm5_doichung; do
  for S in 0 1 2; do
    echo "=== $ARM seed $S  bat dau $(date +%H:%M:%S)"
    .venv/bin/python scripts/run_g1.py \
      --config configs/drift_slda_${ARM}.yaml \
      --set seed=$S train.eval_future=false log.dir=./artifacts_drift_${ARM}_s$S \
      > run_drift_${ARM}_s$S.log 2>&1 \
      || echo "    !!! LOI o $ARM seed $S"
    echo "    xong $(date +%H:%M:%S)  ->  $(grep -o "Average Accuracy.*" run_drift_${ARM}_s$S.log | head -1)"
  done
done
echo "=== D10 XONG $(date +%H:%M:%S)"
' > drift_campaign.log 2>&1 &

tail -f drift_campaign.log
```

Rời ra: `Ctrl+B` rồi `D`, sau đó `exit`. Tắt Mac cũng không sao.

## E2. Hoặc chia 3 máy (~3,5 giờ)

Cùng lệnh trên, chỉ đổi dòng `for ARM in ...` ở mỗi máy:

| Máy | Dòng `for ARM in` |
|---|---|
| 1 | `for ARM in arm1_lam1 arm2_cham; do` |
| 2 | `for ARM in arm3_dexuat arm4_nhanh; do` |
| 3 | `for ARM in arm5_doichung; do` |

Máy 3 nhẹ hơn → chạy thêm D11 (Titans) ngay sau, xem PHẦN G.

**Mỗi máy đều phải qua PHẦN C3 và PHẦN D riêng.** Đừng tin máy này giống máy kia.

---

# PHẦN F — Theo dõi từ Mac

```bash
# tiến độ + accuracy từng run
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'tail -30 ~/RaybanMeta/uav-continual-learning/drift_campaign.log'

# còn chạy không (0 = xong)
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'pgrep -af run_g1.py | wc -l'
```

Vòng lặp tự in accuracy sau mỗi run, nên chỉ cần nhìn `drift_campaign.log`, không phải mở
15 file log riêng.

---

# PHẦN G — D11: Titans trên cùng stream trôi (~4 giờ)

Đây mới là so sánh thật: **cổng quên học được (α) so với cổng quên đặt tay (λ)**.
Chạy máy riêng, hoặc nối tiếp sau D10:

```bash
cd ~/RaybanMeta/uav-continual-learning

nohup bash -c '
while pgrep -f run_g1.py > /dev/null; do sleep 60; done      # đợi D10 xong
for S in 0 1 2; do
  echo "=== titans seed $S  bat dau $(date +%H:%M:%S)"
  .venv/bin/python scripts/run_g1.py \
    --config configs/drift_titans_gates.yaml \
    --set seed=$S train.eval_future=false log.dir=./artifacts_drift_titans_s$S \
    > run_drift_titans_s$S.log 2>&1 || echo "    !!! LOI seed $S"
  echo "    xong $(date +%H:%M:%S)  ->  $(grep -o "Average Accuracy.*" run_drift_titans_s$S.log | head -1)"
done
echo "=== D11 XONG $(date +%H:%M:%S)"
' > titans_drift.log 2>&1 &
```

---

# PHẦN H — Lấy kết quả về Mac

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'cd ~/RaybanMeta/uav-continual-learning && tar czf /tmp/res_drift.tgz artifacts_drift_* run_drift_*.log drift_campaign.log titans_drift.log 2>/dev/null; chmod 644 /tmp/res_drift.tgz; echo GOI_XONG'

mkdir -p ~/Desktop/Raybanmeta/result_test/drift
gcloud compute scp uavcl-slda:/tmp/res_drift.tgz ~/Desktop/Raybanmeta/result_test/drift/ --zone=us-east1-b
cd ~/Desktop/Raybanmeta/result_test/drift && tar xzf res_drift.tgz

cd ~/Desktop/Raybanmeta
python3 uav-continual-learning/scripts/compare_all.py result_test/drift
```

Chia nhiều máy thì đổi tên file gói (`res_drift1.tgz`, ...) và giải nén vào cùng thư mục.

---

# PHẦN I — Tắt máy

```bash
gcloud compute instances stop uavcl-slda --zone=us-east1-b     # giữ ổ đĩa, còn dùng tiếp
gcloud compute instances delete uavcl-slda --zone=us-east1-b   # xong hẳn, khỏi tốn
```

`stop` vẫn tính tiền ổ đĩa (~4 USD/tháng cho 100 GB) nhưng giữ nguyên dataset và venv —
đáng nếu còn chạy tiếp trong vài ngày tới.

---

# Bảng cần điền

| Arm | λ_μ | λ_Σ | Acc cuối | AAA | Forget | AAA nửa sau (task 5–8) |
|---|---|---|---|---|---|---|
| 1 | 1,0 | 1,0 | | | | |
| 2 | 0,9999 | 0,9999 | | | | |
| 3 | 0,999 | 0,9999 | | | | |
| 4 | 0,99 | 0,999 | | | | |
| 5 | 0,99 | 0,99 | | | | |

## Đọc kết quả thế nào

**Kiểm miễn phí trước tiên — task 1 đến 3.** Hiệu chỉnh cho thấy ở mức trôi 12–38%, NCM chỉ
đổi ±0,15 điểm, tức **nằm trong nhiễu đo** (sai số chuẩn ±1,0 điểm trên 2000 mẫu). Không có
gì để thích nghi ở đó, nên **5 arm phải giống nhau**. Nếu chúng khác nhau ngay từ task 1–3
thì có bug, không phải λ đang hoạt động.

Trôi chỉ thực sự cắn từ **task 5** trở đi. Vậy nên cột cuối bảng — AAA nửa sau — mới là chỗ
λ tách nhau, không phải accuracy cuối.

| Quan sát | Kết luận |
|---|---|
| Chữ U ngược, đỉnh ở arm 3 hoặc 4 | ✅ Đúng giả thuyết: có λ tối ưu, quên quá ít và quá nhiều đều dở |
| arm 1 ≈ arm 3 | ❌ Trôi vẫn quá nhẹ, hoặc λ chưa tác dụng — kiểm lại `window_report` |
| arm 5 < arm 4 rõ rệt | ✅ Xác nhận **phải tách hai λ**: Σ cần cửa sổ dài hơn μ |
| arm 5 ≈ arm 4 | Một λ chung là đủ — thiết kế hai λ không cần thiết |
| Titans > arm tốt nhất | Cổng **học được** thắng cổng đặt tay — luận điểm NL đứng vững |
| Titans < arm tốt nhất | Cổng học được vẫn thua, dù đã vá — kèm chi phí 67× bộ nhớ |
