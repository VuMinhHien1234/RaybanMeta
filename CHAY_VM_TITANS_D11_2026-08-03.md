# Chạy D11 (Titans trên stream trôi) trên VM thứ hai

Lập 2026-08-03, sau khi vá cổng η (G1–G5)

| | |
|---|---|
| VM | **`uavcl-titans`** — máy MỚI, không phải `uavcl-slda` |
| Nhánh | `feat/drift-lambda` |
| Số run | 6 (2 trần η × 3 seed) hoặc 2 (cửa chặn rẻ trước) |
| Thời gian | ~8 giờ nếu 6 run · ~2,7 giờ nếu 2 run |

---

## ⛔ Vì sao PHẢI dùng máy thứ hai

`uavcl-slda` đang chạy D10 (15 run SLDA). Vòng lặp gọi `run_g1.py` **lại từ đầu mỗi run**,
nên `git pull` ở đó giữa chừng sẽ khiến arm 2–5 nạp code mới. Arm 1 và phần còn lại đo bằng
hai bản code khác nhau, **không có gì báo lỗi**, mất trọn 10 tiếng.

Không đụng gì vào `uavcl-slda` cho tới khi thấy `=== D10 XONG`.

---

# PHẦN A — Trên Mac: chuyển nhánh, rồi đẩy code

## A1. Chuyển 6 file đã sửa sang đúng nhánh

```bash
cd ~/Desktop/Raybanmeta

git stash push -m "va cong eta G1-G5" \
  uav-continual-learning/configs/g2_titans_resisc45_selfmod_m3_cms.yaml \
  uav-continual-learning/configs/g2_titans_resisc45_selfmod_m3_gates.yaml \
  uav-continual-learning/src/uavcl/methods.py \
  uav-continual-learning/src/uavcl/models/memory.py \
  uav-continual-learning/src/uavcl/models/titans_head.py \
  uav-continual-learning/tests/test_gate_bound.py

git checkout feat/drift-lambda
git stash pop
git status --short
```

5/6 file giống hệt giữa hai nhánh nên sang sạch. `methods.py` khác nhưng khác **vùng**
(tôi sửa khối log cổng Titans; nhánh này thêm log `window_report` của SLDA) → gộp được.
Báo xung đột thì gửi tôi output.

## A2. ⏸ Dừng ở đây, nhắn tôi

Tôi cần tạo nốt `configs/drift_titans_gates_eta_thap.yaml` (arm trần η dời tâm) và
`scripts/run_titans_d11.sh`. Hai file này phải nằm trên `feat/drift-lambda`, mà lúc tôi sửa
code thì cây làm việc đang ở nhánh cũ nên chưa tạo được.

## A3. Chạy test + commit

```bash
cd ~/Desktop/Raybanmeta/uav-continual-learning
.venv/bin/python -m pytest tests/test_gate_bound.py tests/test_slda_decay.py -q
```

Xanh hết mới commit:

```bash
cd ~/Desktop/Raybanmeta
git add -A uav-continual-learning *.md
git commit -m "fix(gates): tran eta lech tam duoc + canh bao bao hoa theo tran that

Loi: eta_logit_limit=2.1972 chon khi con tuong max_lr=1e-2 -> du dinh eta in [1e-3,9e-3].
max_lr THAT = 1.0 -> thuc te [0.1,0.9], gap 100 lan. Log 9 task: eta bo toi 0.8957/0.900
= 99.5% tran -> cong thanh HANG SO, het phu thuoc du lieu (mat Eq 76).

G1 raise khi khong cai duoc hook (truoc do tu tat am tham); log in ca khi TAT
G2 tran lech tam: tanh((x-tam)/nua)*nua+tam, tam=0 bat bien voi ban cu
G3 canh bao bao hoa theo % vi tri trong TRAN THAT, khong phai hang so co dinh
G4 sua comment sai 100x trong 3 config
G5 9 test, gom test do thang TUYET DOI (bo test cu do theo phan so nen khong bat duoc)"

git push
```

---

# PHẦN B — Tạo VM thứ hai

```bash
gcloud compute instances list          # xem uavcl-slda vẫn RUNNING, đừng đụng vào

gcloud compute instances create uavcl-titans \
  --zone=us-east1-b --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB
```

Kẹt quota → đổi `--zone` sang `asia-east1-a` hoặc `us-central1-a`. Đợi ~30 giây rồi:

```bash
gcloud compute ssh uavcl-titans --zone=us-east1-b
```

---

# PHẦN C — Lấy code

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git tmux

cd ~
git clone https://github.com/VuMinhHien1234/RaybanMeta.git
cd RaybanMeta
git checkout feat/drift-lambda
```

Repo private → dán **Personal Access Token** GitHub khi hỏi mật khẩu.

---

# PHẦN D — Môi trường Python

```bash
cd ~/RaybanMeta/uav-continual-learning
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` **cố ý không liệt kê torch** (macOS và Linux khác nhau) — nên phải cài torch
trước, đúng thứ tự trên. `titans-pytorch` nằm trong requirements, đây là thư viện D11 bắt buộc
cần mà D10 (SLDA) không cần.

```bash
.venv/bin/python -c "import torch, timm, titans_pytorch; print('ok', torch.__version__)"
```

---

# PHẦN E — ⛔ Kiểm đúng code (30 giây, đừng bỏ)

```bash
cd ~/RaybanMeta/uav-continual-learning
git branch --show-current
ls configs/drift_titans_gates.yaml configs/drift_titans_gates_eta_thap.yaml
grep -c "gate_bound_range" src/uavcl/models/memory.py
grep -c "GATE_SAT_FRAC" src/uavcl/methods.py
grep -c "feat_sum_g" src/uavcl/models/slda.py
```

| Lệnh | Phải ra |
|---|---|
| `git branch --show-current` | `feat/drift-lambda` |
| `ls configs/drift_titans_*` | 2 file, không lỗi |
| `grep -c gate_bound_range` | **≥ 2** — bản vá G2 có mặt |
| `grep -c GATE_SAT_FRAC` | **≥ 2** — cảnh báo G3 có mặt |
| `grep -c feat_sum_g` | ≥ 6 — bản vá Σ |

Ra `0` ở bất kỳ dòng nào = chưa pull được bản mới. Chạy tiếp là phí giờ máy.

---

# PHẦN F — ⛔ Smoke 3 task (~15 phút)

```bash
.venv/bin/python -m pytest tests/test_gate_bound.py -q

.venv/bin/python scripts/run_g1.py \
  --config configs/drift_titans_gates_eta_thap.yaml \
  --set seed=0 data.num_tasks=3 train.eval_future=false log.dir=./artifacts_smoke_d11 \
  2>&1 | tee run_smoke_d11.log
```

Lần đầu tải dataset ~630 MB, chậm là bình thường.

```bash
grep -E "gate_bound|khoảng \[|BÃO HOÀ|\[drift\] BẬT|\[stream\] DOMAIN" run_smoke_d11.log
```

| Dòng | Phải là |
|---|---|
| `[titans] gate_bound BẬT — … η∈[1.21e-02,5.00e-01], tâm −2.2` | trần **MỚI**, không phải `[1.00e-01,9.00e-01]` |
| `[titans]   η=… (X% khoảng […])` | có in phần trăm → G3 chạy |
| `[drift] BẬT · mode=linear severity=1.0` | trôi bật |
| `[stream] DOMAIN-incremental: 3 task × 45 lớp` | stream đúng |

Thấy `η∈[1.00e-01,9.00e-01]` nghĩa là đang chạy config cũ — quay lại PHẦN E.

Dọn: `rm -rf artifacts_smoke_d11 run_smoke_d11.log`

---

# PHẦN G — Chạy thật

## G-a. Rẻ trước: 1 seed × 2 trần (~2,7 giờ) — ĐỀ NGHỊ

Cùng logic đã cứu ta ở `calibrate_drift` sáng nay: kiểm rẻ trước, đốt giờ máy sau.

```bash
tmux new -s titans
cd ~/RaybanMeta/uav-continual-learning
SEEDS="0" nohup bash scripts/run_titans_d11.sh > titans_d11.log 2>&1 &
sleep 15 && cat titans_d11.log
```

Xong thì xem hai thứ: η **còn bão hoà không** (có dòng `⚠️ BÃO HOÀ` không), và accuracy chênh
bao nhiêu. Rồi mới dồn 3 seed vào trần thắng.

## G-b. Hoặc chạy thẳng 6 run qua đêm (~8 giờ)

```bash
tmux new -s titans
cd ~/RaybanMeta/uav-continual-learning
nohup bash scripts/run_titans_d11.sh > titans_d11.log 2>&1 &
sleep 15 && cat titans_d11.log
```

Rời ra: `Ctrl+B` rồi `D`, sau đó `exit`. Tắt Mac thoải mái.

---

# PHẦN H — Theo dõi từ Mac

```bash
gcloud compute ssh uavcl-titans --zone=us-east1-b -- 'tail -30 ~/RaybanMeta/uav-continual-learning/titans_d11.log'
gcloud compute ssh uavcl-titans --zone=us-east1-b -- 'pgrep -af run_g1.py | wc -l'
```

Script tự trích dòng `BÃO HOÀ` của mỗi run vào `titans_d11.log`, nên nhìn một file là biết cổng
có sống không, khỏi mở 6 log.

---

# PHẦN I — Lấy kết quả

```bash
gcloud compute ssh uavcl-titans --zone=us-east1-b -- 'cd ~/RaybanMeta/uav-continual-learning && tar czf /tmp/res_d11.tgz artifacts_drift_titans_* run_drift_titans_*.log titans_d11.log && chmod 644 /tmp/res_d11.tgz && echo GOI_XONG'

mkdir -p ~/Desktop/Raybanmeta/result_test/d11
gcloud compute scp uavcl-titans:/tmp/res_d11.tgz ~/Desktop/Raybanmeta/result_test/d11/ --zone=us-east1-b
cd ~/Desktop/Raybanmeta/result_test/d11 && tar xzf res_d11.tgz

cd ~/Desktop/Raybanmeta
python3 uav-continual-learning/scripts/compare_all.py result_test/d11
python3 uav-continual-learning/scripts/trace_gates.py result_test/d11
```

---

# PHẦN J — Tắt máy

```bash
gcloud compute instances delete uavcl-titans --zone=us-east1-b
```

Máy tạo riêng cho D11, xong là bỏ. `stop` vẫn tính tiền ổ đĩa ~4 USD/tháng/100 GB.

---

# Bảng cần điền

| Trần η | Seed | Acc | AAA | Forget | η cuối (% khoảng) | Có `⚠️ BÃO HOÀ`? |
|---|---|---|---|---|---|---|
| [0,100 · 0,900] tâm 0 | 0,1,2 | | | | | |
| [0,012 · 0,500] tâm −2,2 | 0,1,2 | | | | | |
| **SLDA arm tốt nhất (D10)** | 0,1,2 | | | | — | — |

## Đọc thế nào

**Nhìn cột η cuối TRƯỚC cột accuracy.** Nếu cả hai trần đều bão hoà thì trần không phải là
vấn đề — nguyên nhân nằm ở chỗ khác (gradient vẫn đẩy cổng đi, chặn ở đâu cũng dính), và
kết luận sẽ là *`gate_bound` không đủ, cần lực kéo ngược như weight-decay trên cổng*.

| Quan sát | Kết luận |
|---|---|
| Trần thấp hết bão hoà, acc tăng | ✅ Trần lệch thang **đúng là** thủ phạm |
| Trần thấp hết bão hoà, acc không đổi | Cổng sống nhưng không giúp — điểm yếu ở chỗ khác |
| Cả hai vẫn bão hoà | `gate_bound` không đủ; cần lực kéo ngược, không phải chặn biên |
| Titans > SLDA arm tốt nhất | Cổng **học được** thắng cổng đặt tay, đổi lại 74× bộ nhớ |
| Titans < SLDA ngay cả ở sân nhà | Kết luận mạnh và tiêu cực cho NL — nhưng giờ mới nói được |
