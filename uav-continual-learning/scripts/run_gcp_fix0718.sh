#!/usr/bin/env bash
# ===========================================================================
# FIX 07-18 — chạy lại HOPE/Titans với 4 cờ ổn định + ablation M3, theo đúng
# kỷ luật 1-BIẾN/LẦN. Resumable: chạy lại script thì run đã xong tự SKIP.
#
# Cách dùng trên GCP (sau khi git pull — xem HUONG_DAN_GCP_FIX0718.md):
#   bash scripts/run_gcp_fix0718.sh              # Phase 1+2 (Titans, ~40')
#   bash scripts/run_gcp_fix0718.sh --hope       # thêm Phase 3 (HOPE, ~2h)
#   bash scripts/run_gcp_fix0718.sh --hope --shutdown   # tự tắt VM khi xong
#
# Kết quả gom về ./artifacts_fix0718/results/ (KHÔNG đè artifacts cũ).
# Các run chỉ khác lr/key_proj (tên thư mục trùng nhau) được tách bằng log.dir riêng.
# ===========================================================================
set -e
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
RUN="$PY scripts/run_g1.py --skip-existing"
OUT=./artifacts_fix0718

DO_HOPE=0; DO_SHUTDOWN=0
for a in "$@"; do
  case "$a" in
    --hope) DO_HOPE=1 ;;
    --shutdown) DO_SHUTDOWN=1 ;;
  esac
done

echo "=== Phase 0: sanity (thư viện + test các fix mới) ==="
$PY - <<'EOF'
import inspect
from titans_pytorch import NeuralMemory
need = ("gated_transition", "spectral_norm_surprises", "qk_rmsnorm", "max_grad_norm")
have = inspect.signature(NeuralMemory.__init__).parameters
missing = [k for k in need if k not in have]
assert not missing, f"titans-pytorch THIẾU {missing} -> pip install -U titans-pytorch"
print("titans-pytorch OK: đủ 4 cờ ổn định")
EOF
$PY -m pytest tests/test_m3.py tests/test_g3_cms.py tests/test_g4_hope.py -q

echo
echo "=== Phase 1: Titans + 4 cờ ổn định (MỐC MỚI — so với 30.2%/F0.72 cũ) ==="
$RUN --config configs/g2_titans_eurosat.yaml \
  --set memory.reset=never train.eval_future=true log.dir=$OUT

echo
echo "=== Phase 2: ablation M3 trên Titans (1 biến/lần, mỗi run ~10') ==="
# 2a. key_proj_eta=0.5 — "quên có hướng" (rank-1 P_i)
$RUN --config configs/g2_titans_eurosat.yaml \
  --set memory.reset=never train.eval_future=true \
        train.m3.key_proj_eta=0.5 log.dir=${OUT}_kp05
# 2b. lr 1e-3 -> 3e-4 (update_norm=rms: lr = độ dịch MỖI bước -> hạ để bớt trôi)
$RUN --config configs/g2_titans_eurosat.yaml \
  --set memory.reset=never train.eval_future=true \
        train.lr=3e-4 log.dir=${OUT}_lr3e4
# 2c. ghi nhẹ hơn: delta.eta [0.1,0.05] -> [0.05,0.01]
$RUN --config configs/g2_titans_eurosat.yaml \
  --set memory.reset=never train.eval_future=true \
        train.m3.delta.eta=[0.05,0.01] log.dir=${OUT}_etalow
# 2d. giữ M3 xuyên task (ký ức chậm m2 không bị reset) — tên tự thêm _optkeep
$RUN --config configs/g2_titans_eurosat.yaml \
  --set memory.reset=never train.eval_future=true \
        train.optimizer_per_task=false log.dir=$OUT

echo
echo "=== Bảng Titans sau Phase 1+2 ==="
for d in $OUT ${OUT}_kp05 ${OUT}_lr3e4 ${OUT}_etalow; do
  [ -d "$d/results" ] && $PY scripts/compare_g1.py --dir "$d" || true
done

if [ "$DO_HOPE" = "1" ]; then
  echo
  echo "=== Phase 3: HOPE full — config đã sửa (CMS thắng G3) + 4 cờ (so 21.7%/F0.956) ==="
  $RUN --config configs/g4_hope_eurosat.yaml --set log.dir=$OUT
  # 3b. HOPE + giữ M3 xuyên task (ablation, tên _optkeep)
  $RUN --config configs/g4_hope_eurosat.yaml \
    --set train.optimizer_per_task=false log.dir=$OUT
  echo "=== Bảng cuối ==="
  $PY scripts/compare_g1.py --dir $OUT || true
fi

echo
echo "XONG. Kết quả trong ${OUT}*/results/ — kéo về máy Mac để phân tích tiếp."
[ "$DO_SHUTDOWN" = "1" ] && sudo shutdown -h now
