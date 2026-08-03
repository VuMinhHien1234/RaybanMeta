# Lấy kết quả từ 3 VM & phân tích

Dùng cho ngày 03-08, sau khi cả ba máy chạy xong.

## Ba máy đang chạy gì

| VM | Zone | Chạy gì | Artifact |
|---|---|---|---|
| `uavcl-cos` | `us-east1-b` | Titans **có** `gate_bound`, 9 task | `artifacts_9t_gates_s0` |
| `uavcl-sdc` | `us-central1-a` | Titans **đối chứng**, 9 task | `artifacts_9t_ctrl_s0` |
| `uavcl-slda` | `us-east1-b` | SLDA B2, 4 arm × 3 seed | `artifacts_b2_*` |

---

# Cách nhanh — một lệnh

```bash
cd ~/Desktop/Raybanmeta/uav-continual-learning
bash scripts/collect_vms.sh
```

Script tự làm ba việc:

1. **Kiểm cả ba máy đã xong chưa** — dừng lại nếu còn máy đang chạy
2. **Đóng gói + kéo về**, mỗi máy một tarball riêng
3. **Giải nén vào ba thư mục tách biệt** — `titans_gates/`, `titans_ctrl/`, `slda_b2/`

Tách thư mục là bắt buộc: hai máy Titans đều sinh `artifacts_9t_*`, để chung sẽ đè nhau.

## Hai chế độ phụ

```bash
bash scripts/collect_vms.sh --check    # chỉ xem máy nào xong, không kéo
bash scripts/collect_vms.sh --force    # kéo kể cả khi còn đang chạy (lấy dở dang)
```

`--check` tiện để chạy vài lần trong ngày mà không đụng gì.

---

# Cách thủ công — nếu script lỗi

## Bước 1 — Kiểm đã xong chưa

```bash
for VM in "uavcl-cos us-east1-b" "uavcl-sdc us-central1-a" "uavcl-slda us-east1-b"; do
  set -- $VM
  echo -n "$1: "
  gcloud compute ssh $1 --zone=$2 --command='pgrep -f run_g1.py | wc -l' 2>/dev/null
done
```

Cả ba ra `0` = xong hết.

## Bước 2 — Đóng gói + kéo về từng máy

```bash
mkdir -p ~/Desktop/Raybanmeta/result_test/final

# --- máy 1: Titans có gate_bound ---
gcloud compute ssh uavcl-cos --zone=us-east1-b -- 'sudo bash -c "shopt -s nullglob; D=\$(ls -d /home/*/RaybanMeta/uav-continual-learning 2>/dev/null | head -1); cd \$D && tar czf /tmp/res_titans_gates.tgz artifacts_9t_* run_9t_*.log && chmod 644 /tmp/res_titans_gates.tgz && echo GOI_XONG \$D"'
gcloud compute scp uavcl-cos:/tmp/res_titans_gates.tgz ~/Desktop/Raybanmeta/result_test/final/ --zone=us-east1-b

# --- máy 2: Titans đối chứng ---
gcloud compute ssh uavcl-sdc --zone=us-central1-a -- 'sudo bash -c "shopt -s nullglob; D=\$(ls -d /home/*/RaybanMeta/uav-continual-learning 2>/dev/null | head -1); cd \$D && tar czf /tmp/res_titans_ctrl.tgz artifacts_9t_* run_9t_*.log && chmod 644 /tmp/res_titans_ctrl.tgz && echo GOI_XONG \$D"'
gcloud compute scp uavcl-sdc:/tmp/res_titans_ctrl.tgz ~/Desktop/Raybanmeta/result_test/final/ --zone=us-central1-a

# --- máy 3: SLDA B2 ---
gcloud compute ssh uavcl-slda --zone=us-east1-b -- 'sudo bash -c "shopt -s nullglob; D=\$(ls -d /home/*/RaybanMeta/uav-continual-learning 2>/dev/null | head -1); cd \$D && tar czf /tmp/res_slda_b2.tgz artifacts_b2_* run_b2_*.log b2_campaign.log bench_vm_* && chmod 644 /tmp/res_slda_b2.tgz && echo GOI_XONG \$D"'
gcloud compute scp uavcl-slda:/tmp/res_slda_b2.tgz ~/Desktop/Raybanmeta/result_test/final/ --zone=us-east1-b
```

Thấy `GOI_XONG /home/.../uav-continual-learning` là đóng gói ok.
`Connection closed` sau đó là **bình thường**.

## Bước 3 — Giải nén tách thư mục

```bash
cd ~/Desktop/Raybanmeta/result_test/final
mkdir -p titans_gates titans_ctrl slda_b2
tar xzf res_titans_gates.tgz -C titans_gates
tar xzf res_titans_ctrl.tgz  -C titans_ctrl
tar xzf res_slda_b2.tgz      -C slda_b2
```

---

# Phân tích

```bash
cd ~/Desktop/Raybanmeta

# 1. Bảng tổng cả 3 máy
python3 uav-continual-learning/scripts/compare_all.py result_test/final

# 2. Quỹ đạo cổng η/α — chỉ áp dụng cho 2 máy Titans
python3 uav-continual-learning/scripts/trace_gates.py result_test/final

# 3. Xem nhanh 12 run SLDA
grep -h "Average Accuracy" result_test/final/slda_b2/run_b2_*.log
```

## Đọc kết quả — hai câu hỏi, hai bảng

### Câu 1 (hướng NL): bản vá `gate_bound` có ăn không?

So `titans_gates` với `titans_ctrl`, cùng seed 0, cùng 9 task, khác đúng một khối config.

| Chỉ số | Ngưỡng |
|---|---|
| `trace_gates` xếp loại run `gates` | phải là **`LANH-MANH`**, không phải `DUNG-YEN`/`TROI-DAN` |
| α qua 9 task | **không kẹt ở biên** (không phải toàn 1.0 hay toàn 0.05) |
| Linear / NCM | ≥ đối chứng |
| `norm(state)` | không leo vô hạn |

Tiêu chí đầu quan trọng nhất. **9 task mới đủ dài để lộ đà trôi** — smoke 3 task không đủ.

### Câu 2 (hướng SLDA): hiệp phương sai có phải nguyên nhân không?

| Arm | Acc kỳ vọng |
|---|---|
| 1 · NCM thuần | 0,71 |
| 2 · SLDA Σ streaming | 0,83 |
| 3 · SLDA Σ = I | **≈ arm 1** |
| 4 · SLDA Σ đóng băng | ? |

| Quan sát | Kết luận |
|---|---|
| arm 3 ≈ arm 1 ≈ 0,71 · arm 2 = 0,83 | ✅ **Hiệp phương sai là nguyên nhân** |
| arm 3 ≈ arm 2 | ❌ Σ không quan trọng — tìm lại nguyên nhân |
| **arm 4 ≈ arm 2** | Σ **không cần** streaming → bỏ `gram`, **cắt 85% bộ nhớ** |
| arm 4 < arm 2 rõ rệt | Σ **cần** streaming → chi phí O(D²) bắt buộc |

**Kiểm bẫy trước tiên:** arm 1 và arm 3 đi hai đường code khác nhau nhưng về toán tương
đương (đã kiểm 0/300 sai lệch). Lệch nhiều = **có bug**, phải tìm ra trước khi tin gì khác.

---

# Bảng cuối cùng cần điền

Tất cả cùng split `combined31500_v2`, 9 task × 5 lớp — **so trực tiếp, không cần chú thích**.

| Phương pháp | Acc | Forget | AAA | Bộ nhớ | Nguồn |
|---|---|---|---|---|---|
| Frozen ViT + NCM | 0,7114 | 0,087 | — | ~0,14 MB | đã có |
| Titans + NCM online | 0,6190 | 0,267 | — | ~100 MB | đã có ⚠️ cổng hỏng |
| Titans **có gate_bound** | | | | ~100 MB | **máy 1** |
| Titans **đối chứng** | | | | ~100 MB | **máy 2** |
| SLDA Σ = I | | | | ~0,14 MB | **máy 3** |
| SLDA Σ đóng băng | | | | 1,39 MB | **máy 3** |
| **SLDA Σ streaming** | **0,8263** | 0,076 | 0,894 | 1,39 MB | đã có + **máy 3** |

Cột chi phí lấy từ `KET_QUA_B6_CHI_PHI_2026-08-02.md`, hoặc tốt hơn là từ
`bench_vm_*.json` của máy 3 (đo trên CPU thuần, mọi dòng cùng thiết bị).

---

# Sau khi lấy xong: tắt/xoá máy

```bash
gcloud compute instances stop uavcl-cos  --zone=us-east1-b
gcloud compute instances stop uavcl-sdc  --zone=us-central1-a
gcloud compute instances delete uavcl-slda --zone=us-east1-b   # máy tạo riêng, xoá luôn
```

Máy tắt vẫn tính tiền ổ đĩa (~$4/tháng/100GB). Hai máy Titans nên `stop` nếu còn định
chạy 3 seed; xong hẳn thì `delete`.

**Chỉ tắt sau khi đã giải nén và chạy `compare_all` thành công** — kiểm tarball đủ dữ liệu
trước, kẻo xoá mất.

---

# Việc nên làm ngay sau đó

Commit metrics vào git để kết quả tái lập được (vài MB, `.pt` đã bị `.gitignore` chặn):

```bash
cd ~/Desktop/Raybanmeta
git add result_test/final/*/**/metrics*.json result_test/final/*/**/acc_matrix*.csv \
        result_test/final/*/**/config.yaml result_test/final/*/*.log
git commit -m "results: campaign 03-08 (titans gate_bound vs ctrl, SLDA B2 ablation)"
git push
```

Đây là việc còn tồn đọng từ đầu — 2,7 GB artifact của báo cáo 01-08 vẫn không có trong
bất kỳ nhánh nào, nên không ai kiểm chứng được số. Đừng lặp lại với campaign này.
