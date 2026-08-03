#!/usr/bin/env bash
# D11 — Titans (cổng đã vá) trên stream TRÔI, so HAI TRẦN η.
#
# Đây là so sánh chưa ai làm: cổng quên HỌC ĐƯỢC (α của Titans) với cổng quên ĐẶT TAY
# (λ của SLDA), trên cùng một dòng dữ liệu. Và trong chính Titans, so hai trần η để tách
# xem trần lệch thang có phải thủ phạm khiến accuracy không nhích hay không.
#
#   arm A  gates       η ∈ [0.100, 0.500·1.8]  tâm 0.500   ← nguyên trạng, đang bão hoà
#   arm B  eta_thap    η ∈ [0.012, 0.500]      tâm 0.100   ← trần dời tâm
#
# Dùng script file thay vì dán vòng lặp: dán nhiều dòng dễ bị gộp thành MỘT dòng, khi đó
# `echo` nuốt luôn lệnh python thành đối số và cả campaign chạy xong trong nháy mắt mà
# không train gì. Đã sập một lần hôm nay.
#
# Dùng:
#   nohup bash scripts/run_titans_d11.sh > titans_d11.log 2>&1 &          # 2 arm × 3 seed
#   SEEDS="0"     nohup bash scripts/run_titans_d11.sh > ... &            # cửa chặn rẻ, 2 run
#   ARMS="gates"  nohup bash scripts/run_titans_d11.sh > ... &            # chỉ một arm
set -u

cd "$(dirname "$0")/.." || exit 1

PY="${PY:-.venv/bin/python}"
ARMS="${ARMS:-gates eta_thap}"
SEEDS="${SEEDS:-0 1 2}"

_cfg() { [ "$1" = "gates" ] && echo "configs/drift_titans_gates.yaml" \
                            || echo "configs/drift_titans_gates_${1}.yaml"; }

# ---- KIỂM TRƯỚC: thà chết ngay còn hơn chạy 8 tiếng ra kết quả không dùng được ------------
[ -x "$PY" ] || { echo "❌ không thấy $PY — sai thư mục, hoặc chưa tạo venv"; exit 1; }
for A in $ARMS; do [ -f "$(_cfg "$A")" ] || { echo "❌ thiếu $(_cfg "$A")"; exit 1; }; done
"$PY" -c "import titans_pytorch" 2>/dev/null || {
  echo "❌ chưa cài titans-pytorch — D11 bắt buộc cần (D10/SLDA thì không)"; exit 1; }
# Bản vá cổng η hôm nay. Thiếu thì run vẫn chạy tới cuối với trần lệch thang 100×, và ta sẽ
# không phân biệt được "ý tưởng NL yếu" với "một tham số lệch đơn vị" — hỏng cả thí nghiệm.
grep -q "gate_bound_range" src/uavcl/models/memory.py || {
  echo "❌ memory.py thiếu gate_bound_range — code cũ, chạy git pull trước"; exit 1; }
grep -q "GATE_SAT_FRAC" src/uavcl/methods.py || {
  echo "❌ methods.py thiếu GATE_SAT_FRAC — code cũ, chạy git pull trước"; exit 1; }
[ -f src/uavcl/data/drift.py ] || { echo "❌ thiếu drift.py — chạy git pull"; exit 1; }

echo "=== BAT DAU $(date +%F' '%H:%M:%S)"
echo "    arm : $ARMS"
echo "    seed: $SEEDS"
echo "    ban code: $(git rev-parse --short HEAD 2>/dev/null || echo 'khong ro')"
echo

T0=$(date +%s); LOI=0
for A in $ARMS; do
  for S in $SEEDS; do
    LOG="run_drift_titans_${A}_s${S}.log"
    echo "=== $A seed $S  bat dau $(date +%H:%M:%S)"
    if "$PY" scripts/run_g1.py --config "$(_cfg "$A")" \
        --set seed="$S" train.eval_future=false \
              log.dir="./artifacts_drift_titans_${A}_s${S}" > "$LOG" 2>&1
    then
      echo "    xong $(date +%H:%M:%S)  ->  $(grep -o 'Average Accuracy.*' "$LOG" | head -1)"
      # Trần đang áp + vị trí cổng ở task cuối. ĐỌC HAI DÒNG NÀY TRƯỚC cột accuracy:
      # nếu cả hai arm đều bão hoà thì trần không phải thủ phạm, và kết luận sẽ là
      # `gate_bound` về bản chất không đủ — cần lực kéo ngược chứ không phải chặn biên.
      grep -o 'gate_bound BẬT.*' "$LOG" | head -1 | sed 's/^/    /'
      grep -oE '(η|α)=[^ ]+ \([0-9]+% khoảng[^)]*\)' "$LOG" | tail -2 | sed 's/^/    /'
      grep -o '⚠️.*BÃO HOÀ.*' "$LOG" | sort -u | sed 's/^/    /'
    else
      LOI=$((LOI + 1)); echo "    !!! LOI o $A seed $S — xem $LOG"
      tail -3 "$LOG" | sed 's/^/        /'
    fi
  done
done

D=$(( $(date +%s) - T0 ))
echo
echo "=== D11 XONG $(date +%F' '%H:%M:%S) — mat $((D/3600))h$(((D%3600)/60))m, ${LOI} run loi"
