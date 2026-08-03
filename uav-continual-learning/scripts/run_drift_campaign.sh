#!/usr/bin/env bash
# D10 — campaign 5 arm × 3 seed trên stream TRÔI (domain-incremental).
#
# Vì sao là file script chứ không phải vòng lặp dán thẳng vào terminal:
# dán nhiều dòng dễ bị gộp thành MỘT dòng (terminal/bracketed-paste nuốt ký tự xuống dòng).
# Khi đó `echo` nuốt luôn lệnh python thành đối số của nó -> cả campaign chạy xong trong
# nháy mắt mà không train gì, log chỉ chứa nguyên văn dòng lệnh. Rất khó nhận ra.
# Dùng file thì lúc chạy chỉ phải dán MỘT dòng.
#
# Dùng:
#   nohup bash scripts/run_drift_campaign.sh > drift_campaign.log 2>&1 &
#
# Chia nhiều máy — đặt biến ARMS trước lệnh:
#   ARMS="arm1_lam1 arm2_cham"          nohup bash scripts/run_drift_campaign.sh > drift_campaign.log 2>&1 &
#   ARMS="arm3_dexuat arm4_nhanh"       nohup bash scripts/run_drift_campaign.sh > drift_campaign.log 2>&1 &
#   ARMS="arm5_doichung"                nohup bash scripts/run_drift_campaign.sh > drift_campaign.log 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1

PY="${PY:-.venv/bin/python}"
ARMS="${ARMS:-arm1_lam1 arm2_cham arm3_dexuat arm4_nhanh arm5_doichung}"
SEEDS="${SEEDS:-0 1 2}"

# ---- KIỂM TRƯỚC KHI CHẠY: thà chết ngay còn hơn chạy 10 tiếng ra kết quả rác --------------
[ -x "$PY" ] || { echo "❌ không thấy $PY — sai thư mục, hoặc chưa tạo venv"; exit 1; }
for ARM in $ARMS; do
  CFG="configs/drift_slda_${ARM}.yaml"
  [ -f "$CFG" ] || { echo "❌ thiếu $CFG"; exit 1; }
done
# Bản vá Σ (feat_sum_g) phải có mặt. Thiếu nó thì Σ mất xác định dương, run vẫn chạy trót lọt
# tới cuối mà accuracy là rác — kiểu lỗi tệ nhất vì không báo gì.
grep -q "feat_sum_g" src/uavcl/models/slda.py || {
  echo "❌ src/uavcl/models/slda.py thiếu feat_sum_g — code cũ, chạy git pull trước"; exit 1; }
[ -f src/uavcl/data/drift.py ] || { echo "❌ thiếu src/uavcl/data/drift.py — chạy git pull"; exit 1; }

echo "=== BAT DAU $(date +%F' '%H:%M:%S)"
echo "    arm : $ARMS"
echo "    seed: $SEEDS"
echo "    ban code: $(git rev-parse --short HEAD 2>/dev/null || echo 'khong ro')"
echo

T0=$(date +%s)
LOI=0
for ARM in $ARMS; do
  for S in $SEEDS; do
    LOG="run_drift_${ARM}_s${S}.log"
    echo "=== $ARM seed $S  bat dau $(date +%H:%M:%S)"
    if "$PY" scripts/run_g1.py \
        --config "configs/drift_slda_${ARM}.yaml" \
        --set seed="$S" train.eval_future=false log.dir="./artifacts_drift_${ARM}_s${S}" \
        > "$LOG" 2>&1
    then
      echo "    xong $(date +%H:%M:%S)  ->  $(grep -o 'Average Accuracy.*' "$LOG" | head -1)"
      # λ có thật sự hoạt động không — bằng chứng nằm ngay trong log của chính run đó
      grep -o 'giữ [0-9.]*% trí nhớ.*' "$LOG" | tail -1 | sed 's/^/    /'
      grep -o '⚠️.*' "$LOG" | sort -u | sed 's/^/    /'
    else
      LOI=$((LOI + 1))
      echo "    !!! LOI o $ARM seed $S — xem $LOG"
      tail -3 "$LOG" | sed 's/^/        /'
    fi
  done
done

GIO=$(( ($(date +%s) - T0) / 3600 ))
PHUT=$(( (($(date +%s) - T0) % 3600) / 60 ))
echo
echo "=== XONG $(date +%F' '%H:%M:%S) — mat ${GIO}h${PHUT}m, ${LOI} run loi"
