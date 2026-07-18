# Hướng dẫn chạy fix 07-18 trên GCP

Mục tiêu vòng này: đo xem **4 cờ ổn định titans-pytorch** + **CMS thắng G3** + **ablation M3** cải thiện được bao nhiêu so với kết quả cũ (`result_test/artifacts_gcp`): HOPE 21.7%/F 0.956, Titans+M3 30.2%/F 0.72.

## Bước 1 — Đẩy code từ Mac lên GitHub

Trên Mac, tại thư mục `Raybanmeta`:

```bash
git add uav-continual-learning plans
git commit -m "fix 0718: 4 co on dinh titans + CMS thang G3 cho HOPE + ablation M3 (key_proj/eta-adaptive)"
git push origin main    # nếu nhánh khác main thì thay tên nhánh
```

## Bước 2 — Trên VM GCP: kéo code + kiểm tra môi trường

```bash
cd ~/RaybanMeta
git pull
cd uav-continual-learning
.venv/bin/pip install -U titans-pytorch    # cần bản có đủ 4 cờ (>=0.5.x)
```

## Bước 3 — Chạy

```bash
# Vòng nhanh (Titans, ~40 phút): sanity + mốc mới + 4 ablation M3
bash scripts/run_gcp_fix0718.sh

# Ổn rồi thì chạy cả HOPE (~2h), tự tắt máy khi xong:
nohup bash scripts/run_gcp_fix0718.sh --hope --shutdown > run_fix0718.log 2>&1 &
tail -f run_fix0718.log
```

Script **resumable**: VM chết giữa chừng thì chạy lại y nguyên lệnh, run nào có `metrics.json` rồi sẽ tự SKIP. Kết quả nằm ở `artifacts_fix0718*/results/` — không đè artifacts cũ.

Các run trong script (1 biến/lần):

| # | Run | So với | Đọc gì |
|---|---|---|---|
| P1 | titans + 4 cờ | 30.2% / F 0.72 | 4 cờ tự nó cứu được bao nhiêu |
| 2a | + key_proj_eta=0.5 | P1 | "quên có hướng" của M3 |
| 2b | + lr 3e-4 | P1 | bớt trôi cuối task |
| 2c | + delta.eta thấp | P1 | ghi nhẹ có đỡ quên |
| 2d | + optimizer_per_task=false | P1 | ký ức chậm m2 xuyên task |
| P3 | hope config mới | 21.7% / F 0.956 | CMS thắng G3 + 4 cờ |
| 3b | hope + optkeep | P3 | như 2d nhưng cho HOPE |

## Bước 4 — Kéo kết quả về Mac

Trên Mac (thay `VM_NAME`, `ZONE` của bạn):

```bash
cd ~/Desktop/Raybanmeta/result_test
gcloud compute scp --recurse VM_NAME:~/RaybanMeta/uav-continual-learning/artifacts_fix0718 . --zone=ZONE
# có các thư mục _kp05/_lr3e4/_etalow thì kéo thêm tương tự, hoặc dùng wildcard qua ssh+tar:
gcloud compute ssh VM_NAME --zone=ZONE -- "cd ~/RaybanMeta/uav-continual-learning && tar czf /tmp/fix0718.tgz artifacts_fix0718*"
gcloud compute scp VM_NAME:/tmp/fix0718.tgz . --zone=ZONE && tar xzf fix0718.tgz
```

Xong thì quay lại chat, đưa tôi thư mục kết quả mới — tôi so với bảng cũ và đề xuất vòng fix tiếp theo.

## Đọc nhanh kết quả — quyết định vòng sau

Ba tín hiệu cần nhìn (theo thứ tự):

1. `norm(state)` trong log HOPE: còn nhảy loạn (kiểu 262→302→120) → hạ `memory.max_grad_norm` 1.0 → 0.5. Ổn định dần → 4 cờ ăn.
2. Forgetting: P1 vs cũ là câu trả lời cho 4 cờ; 2a–2d cái nào thắng thì mang sang HOPE vòng sau.
3. Accuracy: nếu HOPE mới vẫn < CMS-best 38.8% → vấn đề không còn ở ổn định số, chuyển hướng cấu trúc (NCM/prototype head thay Linear head, hoặc HOPE + replay nhỏ).

Vòng sau (đã có sẵn trong code, chỉ bật config): `cms.eta_mode=adaptive` (S9), tổ hợp winner của 2a–2d, rồi seed 1/2 + RESISC45 để chốt bảng báo cáo.
