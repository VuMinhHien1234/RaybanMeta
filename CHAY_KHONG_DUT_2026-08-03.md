# Chạy campaign không đứt khi tắt máy

Ngày 2026-08-03 · cho D10 (5 arm × 3 seed) và D11 (Titans trên stream trôi)

---

## Nguyên lý — vì sao run bị chết khi đóng máy

Khi bạn `gcloud compute ssh`, mọi lệnh bạn gõ đều là **con của phiên SSH**. Đóng laptop →
SSH đứt → hệ điều hành gửi tín hiệu `SIGHUP` xuống toàn bộ tiến trình con → run chết giữa chừng.

Chặn được bằng hai lớp, nên dùng **cả hai**:

| lớp | chặn cái gì | mất khi nào |
|---|---|---|
| `nohup` | bỏ qua tín hiệu `SIGHUP` | không mất — trừ khi VM reboot |
| `tmux` | giữ phiên sống trên VM để quay lại xem | mất khi VM reboot |

`nohup` là lớp **bắt buộc**. `tmux` là tiện lợi — để lúc SSH lại còn xem được màn hình cũ.

> **VM vẫn chạy khi laptop tắt.** Laptop bạn chỉ là cái màn hình từ xa. Miễn là bạn **không**
> `gcloud compute instances stop/delete`, máy vẫn quay CPU và vẫn tính tiền.

---

## BƯỚC 1 — Vào VM, mở tmux

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b

tmux new -s drift
```

Nếu tmux báo đã có phiên tên `drift` (do lần trước) thì vào lại bằng `tmux attach -t drift`.

---

## BƯỚC 2 — D10: 5 arm × 3 seed

```bash
cd ~/RaybanMeta/uav-continual-learning

nohup bash -c '
for ARM in arm1_lam1 arm2_cham arm3_dexuat arm4_nhanh arm5_doichung; do
  for S in 0 1 2; do
    echo "=== $ARM seed $S  bat dau $(date +%H:%M:%S)"
    .venv/bin/python scripts/run_g1.py \
      --config configs/drift_slda_${ARM}.yaml \
      --set seed=$S train.eval_future=false log.dir=./artifacts_drift_${ARM}_s$S \
      > run_drift_${ARM}_s$S.log 2>&1 \
      || echo "    !!! LOI o $ARM seed $S — xem run_drift_${ARM}_s$S.log"
    echo "    xong $(date +%H:%M:%S)  ->  $(grep -o "Average Accuracy.*" run_drift_${ARM}_s$S.log | head -1)"
  done
done
echo "=== D10 XONG TAT CA $(date +%H:%M:%S)"
' > drift_campaign.log 2>&1 &

tail -f drift_campaign.log
```

Ba chỗ có chủ đích trong khối trên:

- **`.venv/bin/python`** chứ không `python`. `nohup bash -c` mở shell mới, **không kế thừa
  `source .venv/bin/activate`** — gọi `python` trơn sẽ chạy nhầm Python hệ thống và lỗi
  `No module named torch`. Đây là lỗi hay gặp nhất ở bước này.
- **`|| echo "!!! LOI"`** — một arm hỏng thì 14 run còn lại vẫn chạy tiếp, không mất cả đêm.
- **Vòng lặp tự in accuracy** sau mỗi run → chỉ nhìn `drift_campaign.log` là thấy hết tiến độ
  lẫn kết quả, không phải mở 15 file log.

---

## BƯỚC 3 — Rời máy

```
Ctrl+B   rồi   D          ← rời tmux (run vẫn chạy)
exit                      ← thoát SSH
```

Giờ đóng laptop được. Tắt hẳn cũng được.

---

## BƯỚC 4 — Quay lại xem (từ bất kỳ đâu)

Xem nhanh không cần vào tmux:

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'tail -30 ~/RaybanMeta/uav-continual-learning/drift_campaign.log'
```

Còn chạy hay xong:

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'pgrep -af run_g1.py | wc -l'
```

`0` = xong hết. Số khác `0` = đang chạy.

Muốn xem lại màn hình cũ:

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b
tmux attach -t drift
```

---

## BƯỚC 5 — D11: Titans trên cùng stream trôi

Chạy **máy khác** cho song song, hoặc chạy nối tiếp trên cùng máy sau khi D10 xong.
Titans nặng hơn nhiều (3 epoch/task thay vì 1) — tính khoảng 4 tiếng.

```bash
cd ~/RaybanMeta/uav-continual-learning

nohup bash -c '
for S in 0 1 2; do
  echo "=== titans seed $S  bat dau $(date +%H:%M:%S)"
  .venv/bin/python scripts/run_g1.py \
    --config configs/drift_titans_gates.yaml \
    --set seed=$S train.eval_future=false log.dir=./artifacts_drift_titans_s$S \
    > run_drift_titans_s$S.log 2>&1 \
    || echo "    !!! LOI o seed $S"
  echo "    xong $(date +%H:%M:%S)  ->  $(grep -o "Average Accuracy.*" run_drift_titans_s$S.log | head -1)"
done
echo "=== D11 XONG $(date +%H:%M:%S)"
' > titans_drift.log 2>&1 &
```

**Muốn nối tiếp tự động sau D10** (không phải canh giờ): thay `nohup bash -c '` bằng

```bash
nohup bash -c 'while pgrep -f run_g1.py > /dev/null; do sleep 60; done
for S in 0 1 2; do
  ...
```

Nó tự đợi D10 xong rồi mới bắt đầu.

---

## Nếu chạy trên Mac thay vì VM

Khác hẳn: **Mac tắt là chết thật**, không cứu được. Chỉ chặn được máy *ngủ*:

```bash
cd ~/Desktop/Raybanmeta/uav-continual-learning
caffeinate -i nohup .venv/bin/python scripts/run_g1.py \
  --config configs/drift_slda_arm3_dexuat.yaml \
  --set seed=0 log.dir=./artifacts_drift_arm3_s0 \
  > run_arm3_s0.log 2>&1 &
```

`caffeinate -i` giữ máy thức khi đang chạy; `nohup` giữ run sống khi đóng Terminal. Nhưng
**đóng nắp máy hoặc hết pin thì vẫn mất**. Campaign 15 run nên để trên VM.

---

## Lấy kết quả về

```bash
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'cd ~/RaybanMeta/uav-continual-learning && tar czf /tmp/res_drift.tgz artifacts_drift_* run_drift_*.log drift_campaign.log titans_drift.log 2>/dev/null; chmod 644 /tmp/res_drift.tgz; echo GOI_XONG'

mkdir -p ~/Desktop/Raybanmeta/result_test/drift
gcloud compute scp uavcl-slda:/tmp/res_drift.tgz ~/Desktop/Raybanmeta/result_test/drift/ --zone=us-east1-b
cd ~/Desktop/Raybanmeta/result_test/drift && tar xzf res_drift.tgz

cd ~/Desktop/Raybanmeta
python3 uav-continual-learning/scripts/compare_all.py result_test/drift
```

---

## Sự cố hay gặp

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `No module named torch` trong log | gọi `python` trơn trong `nohup bash -c` | dùng `.venv/bin/python` — xem BƯỚC 2 |
| `drift_campaign.log` đứng yên rất lâu ở run đầu | đang tải dataset lần đầu | bình thường, đợi |
| Không thấy dòng `[drift] BẬT` | config sai / chạy nhầm nhánh | `git branch --show-current` |
| `tmux: no sessions` sau khi SSH lại | VM đã reboot | run vẫn sống nhờ `nohup`; kiểm bằng `pgrep -af run_g1.py` |
| `pgrep` = 0 mà log chưa có dòng XONG | run bị kill (thường do hết RAM) | xem cuối file log, đổi máy nhiều RAM hơn |

---

## ⛔ Nhắc lại: đừng chạy BƯỚC 2 trước khi qua cửa chặn

```bash
python -m pytest tests/test_slda_decay.py -q
python scripts/calibrate_drift.py --config configs/drift_slda_arm3_dexuat.yaml --dump-samples troi_mau.png
```

Nếu `calibrate_drift.py` báo **TRÔI QUÁ NHẸ** thì 15 run kia sẽ cho 5 arm giống hệt nhau và
bạn mất trọn một đêm máy để biết điều đó. 15 phút hiệu chỉnh đổi lấy 6 tiếng — đáng.
