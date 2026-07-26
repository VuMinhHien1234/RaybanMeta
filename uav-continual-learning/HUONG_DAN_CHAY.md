# Hướng dẫn chạy — từ tạo VM đến lấy kết quả về Mac

Quy trình đầy đủ để chạy thí nghiệm Titans (G2) trên một VM CPU của GCP, rồi kéo kết quả về Mac
phân tích. Gộp mọi bài học đã gặp (đường dẫn theo user, cú pháp `--set`, bug lock, cách kéo file).

> **Cấu hình đang thắng (2026-07-24):** `optimizer=adamw` + **NCM-head** → ~0.74 Acc / ~0.13 Forget,
> ỔN ĐỊNH qua seed, VƯỢT NCM gốc 0.6933. (M3 gây nổ norm ở một số seed → không dùng làm bản chốt.)

---

## Bước 0 — (trên Mac) push code mới nhất lên GitHub

VM sẽ `git clone`, nên mọi commit phải nằm trên GitHub trước. Nếu commit trong sandbox bị kẹt lock,
xoá lock rồi commit tay:
```bash
cd ~/Desktop/Raybanmeta
rm -f .git/index.lock .git/HEAD.lock .git/objects/maintenance.lock 2>/dev/null
git add -A
git commit -m "cập nhật trước khi chạy"
git push
```

---

## Bước 1 — Tạo VM CPU

⚠️ Tài khoản free-trial giới hạn **8 vCPU/region**. Nếu region đã có VM 8-vCPU, chọn **zone khác**
(hoặc xoá VM cũ). Đổi `TÊN_VM` và `ZONE` theo ý:
```bash
gcloud compute instances create uavcl-cpu3 \
  --zone=us-east1-b --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB
```
- `e2-standard-8` = 8 vCPU / 32GB (đủ cho ViT-S + RESISC45).
- Zone gợi ý nếu kẹt quota: `us-east1-b`, `asia-east1-a`, `us-central1-a`.

---

## Bước 2 — SSH vào (đợi ~1 phút cho máy boot)
```bash
gcloud compute ssh uavcl-cpu3 --zone=us-east1-b
```
- Lỗi `Connection refused` ngay sau khi tạo = máy chưa boot xong → đợi 60–90s, thử lại.
- Sau này chạy lệnh 1 dòng kiểu `gcloud compute ssh ... -- "lệnh"` thì thấy **"Connection closed"
  là BÌNH THƯỜNG** (chạy xong tự đóng), không phải lỗi.

---

## Bước 3 — Cài môi trường (trên VM, ~6–8 phút)
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git tmux

git clone https://github.com/VuMinhHien1234/RaybanMeta.git
cd RaybanMeta && git checkout memory_titan_task4_v2      # nhánh có Task 4 + NCM-head + fix data
cd uav-continual-learning

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # torch bản CPU
pip install -r requirements.txt
pip install titans-pytorch==0.5.5        # GHIM đúng bản đã kiểm chứng
pip install -e .
```
- Repo private → `git clone` hỏi mật khẩu → dán **Personal Access Token** GitHub (không phải mật khẩu tài khoản).

---

## Bước 4 — Kiểm tra trước khi đốt giờ
```bash
python scripts/check_env.py                                 # Device: cpu
python -m pytest tests/test_self_modifying_memory.py -q     # phải xanh
```

---

## Bước 5 — Chạy thí nghiệm

**Quy tắc vàng về `--set`:** dùng **MỘT** `--set` rồi liệt kê mọi cặp phía sau, cách nhau bằng dấu cách.
Lặp `--set` hai lần sẽ **rớt** các cặp trước đó (argparse chỉ giữ lần cuối).
```
--set train.optimizer=adamw seed=1 train.eval_ncm_head=true log.dir=./artifacts_adamw_s1   ✅ ĐÚNG
--set train.optimizer=adamw --set seed=1                                                    ❌ SAI (rớt optimizer)
```

### 5a. Bản THẮNG (khuyên dùng) — AdamW + NCM-head, 3 seed
```bash
tmux new -s run
source .venv/bin/activate
nohup bash -c 'for s in 0 1 2; do \
  .venv/bin/python scripts/run_g1.py --config configs/g2_titans_resisc45_selfmod.yaml \
    --set train.optimizer=adamw seed=$s train.eval_ncm_head=true log.dir=./artifacts_adamw_s$s \
    > run_adamw_s$s.log 2>&1 ; done' > campaign.log 2>&1 &
tail -f run_adamw_s0.log
```

### 5b. So M3 vs AdamW (2 optimizer × 2 seed) — để đối chứng
```bash
nohup bash -c 'for opt in m3 adamw; do for s in 0 1; do \
  .venv/bin/python scripts/run_g1.py --config configs/g2_titans_resisc45_selfmod.yaml \
    --set train.optimizer=$opt seed=$s train.eval_ncm_head=true log.dir=./artifacts_${opt}_s${s} \
    > run_${opt}_s${s}.log 2>&1 ; done ; done' > campaign.log 2>&1 &
```

- Rời tmux: `Ctrl+B` rồi `D`. Quay lại: `tmux attach -t run`.
- Chạy tuần tự nên mỗi run ~2–3h (có `eval_ncm_head` thì lâu hơn). Cả loạt ~8–12h → để qua đêm.
- **Không chạy 2 job train cùng lúc trên MỘT máy CPU** (tranh nhân → cả hai chậm). Muốn song song thì
  tách ra 2 máy.

---

## Bước 6 — Theo dõi tiến độ
```bash
pgrep -af run_g1.py                              # còn chạy không (im = xong)
tail -20 run_adamw_s0.log                        # xem run hiện tại tới task mấy
ls -d artifacts_*/results/*/metrics.json 2>/dev/null   # đã xong mấy run
```
Đọc log mỗi task: `[task N] test acc so far` (Linear), `[task N] NCM-head acc so far` (NCM),
`norm(state) sau task` (sức khoẻ bộ nhớ — >1e4 là NỔ), `eta_t/alpha_t` (tốc độ ghi / cổng quên).

---

## Bước 7 — Lấy kết quả về Mac (khi xong)

Repo trên VM có thể nằm ở `/home/<user>/RaybanMeta` với user khác nhau tuỳ máy. Lệnh dưới **tự dò**
đường dẫn + dùng `sudo` để đọc được mọi user. Chạy **trên Mac** (đổi TÊN_VM, ZONE, số tarball):
```bash
# đóng gói trên VM
gcloud compute ssh uavcl-cpu3 --zone=us-east1-b -- 'sudo bash -c "shopt -s nullglob; D=\$(ls -d /home/*/RaybanMeta/uav-continual-learning 2>/dev/null | head -1); cd \$D && tar czf /tmp/res.tgz artifacts_* *.log && chmod 644 /tmp/res.tgz && echo GOI_XONG \$D"'
# kéo về Mac
gcloud compute scp uavcl-cpu3:/tmp/res.tgz ~/Desktop/Raybanmeta/result_test/ --zone=us-east1-b
```
Thấy `GOI_XONG /home/.../uav-continual-learning` = đóng gói thành công. Nhiều máy thì đổi tên tarball
(`res1.tgz`, `res2.tgz`...) cho khỏi đè.

Giải nén (tách thư mục để 3 máy không trùng tên đè nhau):
```bash
cd ~/Desktop/Raybanmeta/result_test
mkdir -p from_cpu3 && tar xzf res.tgz -C from_cpu3
```

---

## Bước 8 — Phân tích: in bảng tổng

Script chỉ dùng thư viện chuẩn → chạy bằng `python3` thường (khỏi venv):
```bash
cd ~/Desktop/Raybanmeta
python3 uav-continual-learning/scripts/compare_all.py result_test
```
Nó quét mọi `metrics.json` + `metrics_ncm.json` + `*.log`, in bảng **Linear/NCM Acc-Forget +
norm(state) + eta/alpha**, đánh dấu `<-- đạt mốc thắng` khi NCM Acc≥0.72 & Forget≤0.10.

Mốc so: **NCM gốc 0.6933 · replay 0.7937 · THẮNG khi Acc≥0.72 & Forget≤0.10.**

---

## Bước 9 — Xoá VM (kẻo tính tiền ổ đĩa)
```bash
gcloud compute instances delete uavcl-cpu3 --zone=us-east1-b
```
(Máy tắt vẫn tính tiền ổ đĩa ~$4/tháng/100GB. Xong hẳn thì xoá.)

---

## Phụ lục — các bẫy thường gặp

| Triệu chứng | Nguyên nhân + cách xử |
|---|---|
| `cd: /home/minhvu/...: No such file` khi kéo file | Repo nằm ở user khác (`/home/vum257792/...`). Dùng lệnh tự-dò + `sudo` ở Bước 7. |
| seed override không ăn (thư mục vẫn `seed0`) | Lặp `--set` hai lần → rớt. Gộp **1 `--set` nhiều cặp**. |
| `train.eval_ncm_head` không có tác dụng | Cùng lỗi lặp `--set`. Kiểm tra: có dòng `NCM-head acc so far` trong log không. |
| `ModuleNotFoundError: uavcl.data` | Clone thiếu (đã fix ở nhánh mới) → `git pull` lại nhánh `memory_titan_task4_v2`. |
| `Connection closed` sau lệnh ssh 1 dòng | BÌNH THƯỜNG — chạy xong tự đóng, không phải lỗi. |
| `Connection refused` ngay sau tạo VM | Máy chưa boot xong → đợi 60–90s, thử lại. |
| `norm(state)` > 1e4 trong log | Bộ nhớ NỔ (thường do M3). Dùng `optimizer=adamw` → norm phẳng ~54. |
| git commit kẹt `index.lock`/`HEAD.lock` | `rm -f .git/index.lock .git/HEAD.lock .git/objects/maintenance.lock` rồi commit lại. |

## Đường dẫn nhanh
- **Nhánh chính:** `memory_titan_task4_v2`
- **Config chạy:** `configs/g2_titans_resisc45_selfmod.yaml`
- **Bản v2 gốc (0.62):** commit `26d17ed`
- **Kết quả lưu ở:** `<log.dir>/results/<dataset>_titans_seed<N>[_<opt>]_rnever/`
  (metrics.json = Linear-head; metrics_ncm.json = NCM-head)
