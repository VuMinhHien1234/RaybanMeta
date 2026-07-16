# Chạy toàn bộ thí nghiệm trên GCP — một lệnh, ra docx

> Kịch bản: tạo VM GPU → đẩy code lên → `bash scripts/run_all.sh --shutdown` → lấy về
> `artifacts/BAO_CAO_KET_QUA.docx`. Script RESUMABLE: máy đứt giữa chừng thì chạy lại
> đúng lệnh cũ, các run đã xong tự bị bỏ qua.

## 0. Điều kiện trước khi tạo được GPU (quan trọng với tài khoản mới)
1. **Tài khoản free-trial KHÔNG tạo được GPU** (lỗi: *"billing account is currently in the
   free tier where non-TPU accelerators are not available"*). Nâng cấp — CẦN THẺ tín dụng/ghi nợ:
   Console -> **Billing** -> banner "Free trial" -> **Activate full account** (hoặc link trong
   thông báo lỗi). Credit $300 còn lại VẪN được dùng tiếp sau khi nâng cấp; chỉ bị trừ tiền
   thật khi xài hết credit.
2. Sau nâng cấp, xin quota GPU (duyệt vài phút–vài giờ): Console -> IAM & Admin ->
   Quotas & System limits -> lọc "GPUs (all regions)" -> Edit -> 1.
   (L4 SPOT có thể cần thêm quota "Preemptible NVIDIA L4 GPUs" theo region.)
3. Cảnh báo `Gaia id not found...` trong Cloud Shell là nhiễu, bỏ qua được — lỗi thật nằm ở dòng ERROR.

> **Không muốn nhập thẻ / chờ duyệt?** Dùng **Google Colab** (T4 miễn phí): mở notebook
> `notebooks/colab_run_all.ipynb` trong repo — cùng một lệnh `run_all.sh --quick`, kết quả
> lưu về Google Drive. Đủ để xem hết mọi task trên EuroSAT trong ~2–3h.

## 1. Tạo VM GPU (một lần) — L4, nhanh ~3× T4, hợp mục tiêu "1–2h xem hết task"
Image family cũ `pytorch-latest-gpu` ĐÃ BỊ XOÁ. Tên hiện hành (PyTorch 2.9 + CUDA 12.9 cài sẵn):
```bash
gcloud compute instances create uavcl-train \
  --zone=asia-southeast1-b \
  --machine-type=g2-standard-8 \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT
```
- `g2-standard-8` ĐÃ KÈM SẴN 1× GPU L4 — **không cần** cờ `--accelerator` (khác dòng n1+T4).
- Zone hết L4 (`ZONE_RESOURCE_POOL_EXHAUSTED`) -> thử `asia-southeast1-c`, `asia-east1-a/c`,
  `us-central1-a`. Muốn rẻ hơn nữa: quay lại T4
  (`--machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1`, chậm ~3×).
- Kiểm tra family còn tồn tại/mới hơn:
  `gcloud compute images list --project deeplearning-platform-release --no-standard-images --format="value(family)" | sort -u | grep pytorch`
- SSH lần đầu nếu được hỏi cài driver NVIDIA -> gõ `y`.

## 1b. SSH key ("genkey") — khi nào cần, khi nào không
- **Dùng `gcloud compute ssh/scp` (khuyên dùng): KHÔNG cần tự tạo key.** Lần chạy đầu gcloud
  tự sinh cặp key `~/.ssh/google_compute_engine(.pub)` và tự đẩy public key lên VM —
  cứ Enter qua các câu hỏi passphrase là xong.
- Chỉ cần `ssh-keygen` thủ công trong 2 tình huống:
  1. **Clone repo GitHub private trên VM:**
     ```bash
     ssh-keygen -t ed25519 -C "uavcl-gcp" -f ~/.ssh/id_ed25519 -N ""
     cat ~/.ssh/id_ed25519.pub    # copy dán vào GitHub -> Settings -> SSH keys
     git clone git@github.com:<user>/<repo>.git
     ```
  2. Muốn `ssh` thẳng không qua gcloud: tạo key như trên rồi thêm public key vào
     Console -> Compute Engine -> Metadata -> SSH Keys.

### 1c. Từ Mac SSH thẳng vào VM — trọn bộ 5 bước (không cần cài gcloud trên Mac)
**B1 — tạo key trên Mac** (Terminal; `-C` chính là USERNAME sẽ dùng để đăng nhập):
```bash
ssh-keygen -t ed25519 -C "vum257792" -f ~/.ssh/gcp_uavcl -N ""
cat ~/.ssh/gcp_uavcl.pub        # copy TOÀN BỘ 1 dòng này
```
**B2 — nạp public key cho VM** (chọn 1 trong 2):
- Web: Console -> Compute Engine -> **Metadata** -> tab **SSH Keys** -> Edit -> Add item
  -> dán dòng vừa copy -> Save. (Nạp mức project = mọi VM đều nhận.)
- Hoặc trong Cloud Shell (thay `ssh-ed25519 AAAA...` bằng nội dung pubkey):
```bash
gcloud compute instances add-metadata uavcl-cpu --zone=asia-southeast1-b \
  --metadata=ssh-keys="vum257792:ssh-ed25519 AAAA... vum257792"
```
**B3 — lấy External IP của VM:**
```bash
gcloud compute instances describe uavcl-cpu --zone=asia-southeast1-b \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```
(hoặc nhìn cột External IP trong Console -> VM instances.)
**B4 — SSH từ Mac:**
```bash
ssh -i ~/.ssh/gcp_uavcl vum257792@<EXTERNAL_IP>     # lần đầu hỏi fingerprint -> yes
```
**B5 — (tiện) alias để gõ ngắn + scp thẳng từ Mac, khỏi vòng qua Cloud Shell:**
```bash
cat >> ~/.ssh/config <<'EOF'
Host uavcl
  HostName <EXTERNAL_IP>
  User vum257792
  IdentityFile ~/.ssh/gcp_uavcl
EOF
ssh uavcl                                   # đăng nhập
scp uavcl.tgz uavcl:~                       # đẩy code thẳng Mac -> VM
scp uavcl:~/uav-continual-learning/artifacts/BAO_CAO_KET_QUA.docx ~/Desktop/   # lấy kết quả
```
⚠ **IP đổi mỗi lần stop/start VM** (IP tạm) — chạy lại B3 và sửa `HostName` trong ~/.ssh/config.
Không SSH được: kiểm tra VM đang RUNNING + firewall `default-allow-ssh` tồn tại (mặc định có).

## 2. Đưa code lên
Cách A — qua Cloud Shell (không cần cài gì trên Mac):
1. Trên Mac nén project (⚠ exclude phải NEO đường dẫn — `--exclude='data'` trần sẽ nuốt nhầm cả `src/uavcl/data`!):
   `cd /Users/minhvu/Desktop/Raybanmeta && tar czf uavcl.tgz --exclude='uav-continual-learning/.venv' --exclude='uav-continual-learning/data' uav-continual-learning`
   (GIỮ `artifacts/` trong gói — mang theo số G1 đã chạy để VM tự skip toàn bộ G1, tiết kiệm ~2h.)
2. Cloud Shell: nút ⋮ (3 chấm) -> **Upload** -> chọn `uavcl.tgz`.
3. Từ Cloud Shell đẩy sang VM và giải nén:
```bash
gcloud compute scp uavcl.tgz uavcl-train:~ --zone=asia-southeast1-b
gcloud compute ssh uavcl-train --zone=asia-southeast1-b
tar xzf uavcl.tgz && cd uav-continual-learning
```
Cách B — git (nếu repo đã push): `git clone <URL_REPO> && cd uav-continual-learning`
(repo private thì tạo SSH key theo mục 1b.1).

## 3. Cài môi trường (trên VM, ~5 phút)
```bash
cd uav-continual-learning
pip install -r requirements.txt && pip install -e . && pip install python-docx
python scripts/check_env.py          # mọi dòng [OK]; Device phải ra: cuda
pytest -q                            # toàn bộ test phải xanh trước khi đốt giờ GPU
```

## 4. Chạy — MỘT LỆNH (trong tmux để rớt SSH không chết job)

**Kế hoạch "1–2 giờ xem hết mọi task" (trên L4):**
```bash
tmux new -s train
bash scripts/run_all.sh --quick 2>&1 | tee run_quick.log
```
`--quick` = chạy ĐỦ mọi giai đoạn G1→G4 + báo cáo docx, nhưng chỉ trên EuroSAT —
trên L4 hết ~1–2h (nếu đã upload kèm `artifacts/` từ Mac thì G1 được skip, còn ~1h).
Bạn sẽ thấy trọn: 5 baseline, Titans A/B/C, 4 ablation CMS + auto-pick winner, HOPE,
và `BAO_CAO_KET_QUA.docx` đầu tiên.

**Bảng chính thức (RESISC45) — chạy đêm sau khi quick ổn:**
```bash
bash scripts/run_all.sh --shutdown 2>&1 | tee run.log        # ~3-5h trên L4, xong tự tắt máy
# rời tmux: Ctrl+B rồi D · quay lại xem: tmux attach -t train
```
Thứ tự script tự làm: G1 (5 baseline × 2 dataset) → G2 (Titans A/B/C + đối chứng AdamW)
→ G3 (4 ablation CMS, TỰ CHỌN cấu hình thắng, chạy RESISC45) → G4 (HOPE với cấu hình
thắng + bảng 2×2 optimizer) → `compare_g1` + `make_report` → **BAO_CAO_KET_QUA.docx**.

## 5. Lấy kết quả về máy
```bash
gcloud compute scp uavcl-train:~/uav-continual-learning/artifacts/BAO_CAO_KET_QUA.docx . --zone=asia-southeast1-b
gcloud compute scp --recurse uavcl-train:~/uav-continual-learning/artifacts ./artifacts_gcp --zone=asia-southeast1-b
gcloud compute scp uavcl-train:~/uav-continual-learning/run.log . --zone=asia-southeast1-b   # log norm(state)/‖Δw‖
```

## 6. Tiền nong & vệ sinh
- T4 SPOT ≈ $0.10–0.15/giờ (+VM ≈ $0.10/h) → full run ~8–14h ≈ **$2–4**. On-demand ≈ gấp 3.
- SPOT có thể bị GCP thu hồi giữa chừng → nhờ resume, chỉ việc `start` lại VM và chạy lại lệnh.
- `--shutdown` đã tự tắt máy, nhưng máy tắt VẪN TÍNH TIỀN Ổ ĐĨA (~$4/tháng/100GB). Xong hẳn thì xoá:
```bash
gcloud compute instances delete uavcl-train --zone=asia-southeast1-b
```

## Phương án CPU — free trial, KHÔNG cần nâng billing (chậm, chấp nhận được cho --quick)
Free trial cho tạo VM CPU thoải mái (credit $300 tự trả, giới hạn 8 vCPU/region).
Kỳ vọng thật: `--quick` ≈ 12–18h trên 8 vCPU (chạy qua đêm + ngày); RESISC45 full = NHIỀU NGÀY — đừng làm trên CPU.
```bash
# 1) Tạo VM CPU (không SPOT cho đỡ rắc rối quota; ~$0.27/h trừ vào credit)
gcloud compute instances create uavcl-cpu \
  --zone=asia-southeast1-b --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB

# 2) Đưa code lên (mục 2 — upload uavcl.tgz qua Cloud Shell rồi scp, nhớ kèm artifacts/)
gcloud compute scp uavcl.tgz uavcl-cpu:~ --zone=asia-southeast1-b
gcloud compute ssh uavcl-cpu --zone=asia-southeast1-b

# 3) Trên VM: cài môi trường (torch bản CPU cho nhẹ)
sudo apt update && sudo apt install -y python3-pip python3-venv tmux
tar xzf uavcl.tgz && cd uav-continual-learning
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e . && pip install python-docx
python scripts/check_env.py && python -m pytest -q     # device sẽ là: cpu — đúng dự kiến

# 4) Chạy trong tmux, xong tự tắt máy
tmux new -s train
bash scripts/run_all.sh --quick --shutdown 2>&1 | tee run.log
# Ctrl+B rồi D để rời; hôm sau: tmux attach -t train hoặc xem run.log

# 5) Lấy kết quả (chạy từ Cloud Shell; máy đã tắt thì start lại trước: gcloud compute instances start uavcl-cpu --zone=...)
gcloud compute scp uavcl-cpu:~/uav-continual-learning/artifacts/BAO_CAO_KET_QUA.docx . --zone=asia-southeast1-b
```
Xong hẳn nhớ xoá VM (`gcloud compute instances delete uavcl-cpu --zone=asia-southeast1-b`).

## Sự cố thường gặp
| Hiện tượng | Xử lý |
|---|---|
| `Quota 'GPUS_ALL_REGIONS' exceeded` | Xin quota (mục 1), chờ duyệt ~vài giờ |
| Zone hết GPU (`ZONE_RESOURCE_POOL_EXHAUSTED`) | Đổi zone/region hoặc đổi loại GPU |
| SPOT bị thu hồi | `gcloud compute instances start uavcl-train ...` rồi chạy lại lệnh cũ (resume) |
| OOM trên T4 (16GB) khi chạy G4 | thêm `--set data.batch_size=16` vào config g4 hoặc chạy `--quick` |
| Tải RESISC45 chậm lần đầu | bình thường (~426MB từ HF); chỉ tải 1 lần vào `data/hf_cache` |
