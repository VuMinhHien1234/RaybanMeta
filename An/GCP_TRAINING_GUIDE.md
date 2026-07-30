# Hướng Dẫn Kết Nối Google Cloud VM Và Chạy Training

Tài liệu này dùng cho máy ảo Google Cloud hiện tại của bạn và repo nhóm:

```text
Repo: https://github.com/VuMinhHien1234/RaybanMeta.git
```

Thông tin VM hiện tại:

```text
Project ID: project-95a0d104-9d0f-4aa1-ba0
Zone: us-central1-b
Instance name: instance-20260725-122154
SSH user: vanan05092004
Image: pytorch-2-9-cu129-ubuntu-2204-nvidia-580-stage
Hệ điều hành: Ubuntu 22.04 Deep Learning VM
CUDA: 12.9
PyTorch: 2.9
Ổ đĩa khả dụng: khoảng 96.73 GB
```

Mục tiêu:

1. SSH vào máy ảo.
2. Kiểm tra GPU CUDA.
3. Clone project từ GitHub.
4. Cài môi trường Python.
5. Chạy test để chắc project ổn.
6. Chạy training quick/full.
7. Tự động tắt VM sau khi training xong để tiết kiệm tiền.
8. Lấy kết quả về máy cá nhân.

---

## 0. Ghi Chú Cho Agent Khi Hỗ Trợ Chạy Trên VM

Nếu dùng agent/Codex để hỗ trợ chạy training, ưu tiên thao tác từ **Terminal trên Mac** bằng `gcloud compute ssh`, vì như vậy agent có thể gửi lệnh trực tiếp vào phiên SSH thay vì phải thao tác qua SSH-in-browser.

Thông tin cố định để agent dùng:

```text
Project ID: project-95a0d104-9d0f-4aa1-ba0
Zone: us-central1-b
Instance name: instance-20260725-122154
Repo: https://github.com/VuMinhHien1234/RaybanMeta.git
Project path on VM: ~/RaybanMeta/uav-continual-learning
```

Lệnh SSH chuẩn từ Terminal Mac:

```bash
gcloud compute ssh instance-20260725-122154 \
  --zone=us-central1-b \
  --project=project-95a0d104-9d0f-4aa1-ba0
```

Nếu VM đang tắt, start trước:

```bash
gcloud compute instances start instance-20260725-122154 \
  --zone=us-central1-b \
  --project=project-95a0d104-9d0f-4aa1-ba0
```

Sau đó SSH lại bằng lệnh trên.

Không chạy:

```bash
sudo do-release-upgrade
```

Lý do: VM đang dùng Ubuntu 22.04 Deep Learning VM với CUDA/PyTorch đã được cấu hình sẵn. Upgrade lên Ubuntu 24.04 có thể làm hỏng driver/CUDA/PyTorch và mất thời gian sửa.

Khi training dài, luôn dùng:

```bash
tmux
```

và ưu tiên thêm:

```bash
--shutdown
```

để VM tự tắt sau khi chạy xong.

---

## 1. Cách SSH Vào Máy Ảo

### Cách khuyên dùng cho agent: SSH từ Terminal Mac

Trên Terminal Mac, chạy:

```bash
gcloud compute ssh instance-20260725-122154 \
  --zone=us-central1-b \
  --project=project-95a0d104-9d0f-4aa1-ba0
```

Nếu lệnh báo VM đang tắt, start VM:

```bash
gcloud compute instances start instance-20260725-122154 \
  --zone=us-central1-b \
  --project=project-95a0d104-9d0f-4aa1-ba0
```

Rồi SSH lại:

```bash
gcloud compute ssh instance-20260725-122154 \
  --zone=us-central1-b \
  --project=project-95a0d104-9d0f-4aa1-ba0
```

### Cách nhanh nhất: dùng nút SSH trên Google Cloud Console

Vào:

```text
Google Cloud Console -> Compute Engine -> VM instances
```

Ở dòng máy:

```text
instance-20260725-122154
```

bấm nút:

```text
SSH
```

Bạn sẽ vào terminal dạng trình duyệt như ảnh hiện tại.

Nếu thấy dòng dạng:

```text
vanan05092004@instance-20260725-122154:~$
```

thì bạn đã SSH thành công.

### Cách SSH bằng Cloud Shell

Mở Cloud Shell rồi chạy:

```bash
gcloud compute ssh instance-20260725-122154 --zone=us-central1-b
```

Nếu cần chỉ rõ project:

```bash
gcloud compute ssh instance-20260725-122154 \
  --zone=us-central1-b \
  --project=project-95a0d104-9d0f-4aa1-ba0
```

---

## 2. Kiểm Tra GPU Trên VM

Sau khi SSH vào VM, chạy:

```bash
nvidia-smi
```

Nếu đúng, bạn sẽ thấy thông tin GPU NVIDIA, ví dụ:

```text
NVIDIA T4
```

hoặc:

```text
NVIDIA L4
```

Nếu `nvidia-smi` chạy được, nghĩa là driver NVIDIA và CUDA đã ổn.

Nếu không thấy GPU, cần kiểm tra lại trong Google Cloud:

```text
1. VM có gắn GPU chưa?
2. VM có đang dùng đúng image Deep Learning VM with CUDA + PyTorch không?
3. Zone có đúng là us-central1-b không?
```

---

## 3. Kiểm Tra Python, CUDA Và PyTorch

Chạy:

```bash
python3 --version
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Kết quả mong muốn:

```text
torch.cuda.is_available() = True
GPU name = NVIDIA T4 hoặc NVIDIA L4
```

Nếu kết quả là `False`, chưa nên training. Cần xử lý CUDA/PyTorch trước.

Không cần nâng cấp Ubuntu dù hệ thống báo:

```text
New release '24.04.x LTS' available.
Run 'do-release-upgrade' to upgrade to it.
```

Cứ giữ Ubuntu 22.04 để tránh lệch CUDA/driver.

---

## 4. Clone Repo Nhóm Từ GitHub

Trên VM, chạy:

```bash
cd ~
git clone https://github.com/VuMinhHien1234/RaybanMeta.git
cd ~/RaybanMeta/uav-continual-learning
```

Nếu repo là private và GitHub yêu cầu đăng nhập:

```text
Username: VuMinhHien1234
Password: dán GitHub Personal Access Token vào đây
```

Lưu ý: GitHub không cho dùng mật khẩu tài khoản thường để clone bằng HTTPS. Nếu repo private, bạn cần dùng **Personal Access Token**.

---

## 5. Nếu Repo Đã Clone Rồi Thì Cập Nhật Code

Không clone lại nếu thư mục `~/RaybanMeta` đã tồn tại. Chạy:

```bash
cd ~/RaybanMeta
git pull
cd ~/RaybanMeta/uav-continual-learning
```

Nếu đang đứng trong thư mục project:

```bash
git pull
```

---

## 6. Cài Môi Trường Python Cho Project

Vào thư mục project:

```bash
cd ~/RaybanMeta/uav-continual-learning
```

Cài dependencies:

```bash
pip install -r requirements.txt
pip install -e .
pip install python-docx
```

Nếu gặp lỗi quyền ghi package, dùng:

```bash
pip install --user -r requirements.txt
pip install --user -e .
pip install --user python-docx
```

Nếu muốn dùng môi trường riêng sạch hơn, dùng virtual environment:

```bash
cd ~/RaybanMeta/uav-continual-learning
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install python-docx
```

Khi SSH lại vào VM lần sau, nếu dùng `.venv`, nhớ kích hoạt lại:

```bash
cd ~/RaybanMeta/uav-continual-learning
source .venv/bin/activate
```

---

## 7. Kiểm Tra Project Trước Khi Training

Chạy:

```bash
cd ~/RaybanMeta/uav-continual-learning
python scripts/check_env.py
```

Cần thấy device là:

```text
cuda
```

Sau đó chạy test:

```bash
pytest -q
```

Nếu test xanh, lúc đó mới nên chạy training dài.

---

## 8. Dùng tmux Để Training Không Bị Dừng Khi Tắt Tab SSH

`tmux` giúp training tiếp tục chạy kể cả khi bạn đóng tab SSH trong trình duyệt.

Cài `tmux` nếu chưa có:

```bash
sudo apt update
sudo apt install -y tmux
```

Tạo session training:

```bash
tmux new -s train
```

Rời khỏi tmux mà không dừng job:

```text
Ctrl+B rồi nhấn D
```

Quay lại session:

```bash
tmux attach -t train
```

Xem danh sách session:

```bash
tmux ls
```

---

## 9. Chạy Quick EuroSAT Trước

Quick dùng để kiểm tra toàn bộ pipeline G1 -> G4 trên EuroSAT. Đây là bước nên chạy trước full RESISC45.

Chạy quick và tự tắt VM sau khi xong:

```bash
cd ~/RaybanMeta/uav-continual-learning
tmux new -s train
bash scripts/run_all.sh --quick --shutdown 2>&1 | tee run_quick.log
```

Nếu không muốn tự tắt VM sau quick:

```bash
bash scripts/run_all.sh --quick 2>&1 | tee run_quick.log
```

Kết quả quick nằm ở:

```text
artifacts/results/
artifacts/baseline_table.csv
artifacts/baseline_table.md
artifacts/BAO_CAO_KET_QUA.docx
run_quick.log
```

---

## 10. Chạy Full Training

Sau khi quick ổn, chạy full:

```bash
cd ~/RaybanMeta/uav-continual-learning
tmux new -s train
bash scripts/run_all.sh --shutdown 2>&1 | tee run.log
```

`--shutdown` giúp VM tự tắt khi training xong.

Điều này giúp ngừng tính tiền:

```text
CPU
GPU
RAM
```

Nhưng vẫn còn tính rất ít tiền cho:

```text
Persistent disk
```

Nếu không muốn tự tắt:

```bash
bash scripts/run_all.sh 2>&1 | tee run.log
```

---

## 11. Lệnh Tắt VM Thủ Công

Nếu đang ở trong VM:

```bash
sudo shutdown -h now
```

Nếu dùng Cloud Shell:

```bash
gcloud compute instances stop instance-20260725-122154 --zone=us-central1-b
```

Nếu cần chỉ rõ project:

```bash
gcloud compute instances stop instance-20260725-122154 \
  --zone=us-central1-b \
  --project=project-95a0d104-9d0f-4aa1-ba0
```

Khi stop VM:

```text
File vẫn còn.
Code vẫn còn.
Kết quả trong artifacts vẫn còn.
CPU/GPU/RAM ngừng tính tiền.
Disk vẫn tính tiền nhẹ.
```

---

## 12. Start Lại VM Sau Khi Đã Stop

Trong Google Cloud Console:

```text
Compute Engine -> VM instances -> chọn instance-20260725-122154 -> Start
```

Hoặc dùng Cloud Shell:

```bash
gcloud compute instances start instance-20260725-122154 --zone=us-central1-b
```

Sau khi start, bấm SSH lại. Nếu dùng SSH từ Mac bằng IP public, lưu ý IP có thể đã đổi.

---

## 13. Nếu Spot VM Bị Google Thu Hồi Giữa Chừng

Nếu dùng Spot, Google có thể dừng VM khi họ cần tài nguyên. Khi đó:

1. Start VM lại.
2. SSH vào VM.
3. Vào thư mục project.
4. Chạy lại đúng lệnh cũ.

Ví dụ quick:

```bash
cd ~/RaybanMeta/uav-continual-learning
tmux new -s train
bash scripts/run_all.sh --quick --shutdown 2>&1 | tee -a run_quick.log
```

Ví dụ full:

```bash
cd ~/RaybanMeta/uav-continual-learning
tmux new -s train
bash scripts/run_all.sh --shutdown 2>&1 | tee -a run.log
```

Script được thiết kế để resume/skip những run đã có kết quả, nên không cần xóa tay thư mục kết quả.

---

## 14. Xem Kết Quả Trên VM

Danh sách các run:

```bash
cd ~/RaybanMeta/uav-continual-learning
ls artifacts/results
```

Xem bảng tổng hợp Markdown:

```bash
cat artifacts/baseline_table.md
```

Xem log 100 dòng cuối:

```bash
tail -n 100 run.log
```

Theo dõi log khi đang chạy:

```bash
tail -f run.log
```

Nếu đang chạy quick:

```bash
tail -f run_quick.log
```

Mỗi thư mục run thường có:

```text
metrics.json
acc_matrix.csv
config.yaml
```

Cách đọc nhanh:

```text
metrics.json   = accuracy, forgetting, runtime, memory của run
acc_matrix.csv = ma trận accuracy qua từng task
config.yaml    = cấu hình đã dùng để chạy run đó
```

---

## 15. Lấy Kết Quả Về Máy

### Cách A: Tải bằng nút Download File trong SSH Browser

Trong cửa sổ SSH trên trình duyệt có nút:

```text
DOWNLOAD FILE
```

Cách này hợp để tải từng file như:

```text
~/RaybanMeta/uav-continual-learning/run.log
~/RaybanMeta/uav-continual-learning/artifacts/BAO_CAO_KET_QUA.docx
```

Nếu muốn tải cả thư mục `artifacts`, nên nén trước:

```bash
cd ~/RaybanMeta/uav-continual-learning
tar czf artifacts_gcp.tgz artifacts run.log run_quick.log
```

Sau đó dùng nút **DOWNLOAD FILE** để tải:

```text
~/RaybanMeta/uav-continual-learning/artifacts_gcp.tgz
```

### Cách B: Tải bằng Cloud Shell

Từ Cloud Shell:

```bash
gcloud compute scp --recurse instance-20260725-122154:~/RaybanMeta/uav-continual-learning/artifacts ./artifacts_gcp --zone=us-central1-b
gcloud compute scp instance-20260725-122154:~/RaybanMeta/uav-continual-learning/run.log ./run.log --zone=us-central1-b
```

Nếu cần chỉ rõ project:

```bash
gcloud compute scp --recurse instance-20260725-122154:~/RaybanMeta/uav-continual-learning/artifacts ./artifacts_gcp \
  --zone=us-central1-b \
  --project=project-95a0d104-9d0f-4aa1-ba0
```

### Cách C: Tải trực tiếp từ Mac nếu đã cài gcloud

Trên Mac:

```bash
gcloud compute scp --recurse instance-20260725-122154:~/RaybanMeta/uav-continual-learning/artifacts "/Users/an/Documents/Do An/artifacts_gcp" --zone=us-central1-b
gcloud compute scp instance-20260725-122154:~/RaybanMeta/uav-continual-learning/run.log "/Users/an/Documents/Do An/run_gcp.log" --zone=us-central1-b
```

---

## 16. Xóa VM Khi Xong Hẳn

Chỉ xóa VM sau khi đã tải hết kết quả quan trọng về máy.

Cloud Shell:

```bash
gcloud compute instances delete instance-20260725-122154 --zone=us-central1-b
```

Nếu lúc tạo VM bạn chọn:

```text
Delete boot disk
```

thì khi xóa VM, disk cũng bị xóa và không còn tính tiền.

Nếu chỉ **Stop** VM, file vẫn còn nhưng disk vẫn tính tiền nhẹ.

---

## 17. Checklist Chạy Nhanh Trên VM

Checklist này dành cho agent hoặc người chạy từ Terminal Mac.

### 17.1. Từ Terminal Mac: start và SSH vào VM

```bash
gcloud compute instances start instance-20260725-122154 \
  --zone=us-central1-b \
  --project=project-95a0d104-9d0f-4aa1-ba0
```

```bash
gcloud compute ssh instance-20260725-122154 \
  --zone=us-central1-b \
  --project=project-95a0d104-9d0f-4aa1-ba0
```

Nếu VM đã chạy sẵn, lệnh start có thể bỏ qua.

### 17.2. Trên VM: kiểm tra, clone/cập nhật, cài và chạy quick

Copy lần lượt các lệnh này trong phiên SSH:

```bash
nvidia-smi
cd ~
if [ -d RaybanMeta ]; then
  cd RaybanMeta
  git pull
else
  git clone https://github.com/VuMinhHien1234/RaybanMeta.git
  cd RaybanMeta
fi
cd ~/RaybanMeta/uav-continual-learning
pip install -r requirements.txt
pip install -e .
pip install python-docx
python scripts/check_env.py
pytest -q
```

Chạy quick trong `tmux`:

```bash
tmux new -s train
bash scripts/run_all.sh --quick --shutdown 2>&1 | tee run_quick.log
```

Lưu ý: sau dòng `tmux new -s train`, terminal sẽ vào màn hình tmux. Khi đó mới chạy dòng `bash scripts/run_all.sh ...`.

Nếu muốn rời tmux mà training vẫn chạy:

```text
Ctrl+B rồi nhấn D
```

Quay lại:

```bash
tmux attach -t train
```

### 17.3. Sau khi quick ổn: chạy full

Start VM lại nếu quick đã tự shutdown, SSH vào VM, rồi chạy:

```bash
cd ~/RaybanMeta/uav-continual-learning
tmux new -s train
bash scripts/run_all.sh --shutdown 2>&1 | tee run.log
```

### 17.4. Lệnh một cụm cho agent sau khi đã SSH vào VM

Nếu agent đã ở trong VM và muốn chạy nhanh toàn bộ bước chuẩn bị + quick:

```bash
nvidia-smi
cd ~
if [ -d RaybanMeta ]; then
  cd RaybanMeta
  git pull
else
  git clone https://github.com/VuMinhHien1234/RaybanMeta.git
  cd RaybanMeta
fi
cd ~/RaybanMeta/uav-continual-learning
pip install -r requirements.txt
pip install -e .
pip install python-docx
python scripts/check_env.py
pytest -q
tmux new -s train
```

Sau khi vào tmux, chạy:

```bash
bash scripts/run_all.sh --quick --shutdown 2>&1 | tee run_quick.log
```

---

## 18. Checklist Kiểm Tra Trước Khi Training Dài

Trước khi chạy full, nên kiểm tra:

```text
[ ] nvidia-smi thấy GPU
[ ] python scripts/check_env.py báo device cuda
[ ] pytest -q chạy xanh
[ ] Đã chạy quick ít nhất một lần
[ ] Dùng tmux
[ ] Có --shutdown để VM tự tắt sau khi xong
```

---

## 19. Gợi Ý Khi Gặp Lỗi

### Lỗi: `git clone` hỏi password

Repo có thể đang private. Dùng GitHub Personal Access Token thay password.

### Lỗi: `CUDA out of memory`

Giảm batch size trong config hoặc chạy quick trước. Với T4 16GB, HOPE/RESISC45 có thể cần batch nhỏ hơn.

### Lỗi: SSH tab bị đóng

Nếu đang chạy trong `tmux`, không sao. SSH lại rồi:

```bash
tmux attach -t train
```

### Lỗi: VM bị stop do Spot

Start lại VM rồi chạy lại lệnh cũ. Script sẽ bỏ qua các run đã xong.

### Lỗi: hết dung lượng disk

Kiểm tra:

```bash
df -h
du -sh ~/RaybanMeta/uav-continual-learning/data
du -sh ~/RaybanMeta/uav-continual-learning/artifacts
```

Nếu disk chỉ khoảng 100GB, không nên lưu quá nhiều run cũ không cần thiết.
