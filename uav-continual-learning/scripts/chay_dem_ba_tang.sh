#!/usr/bin/env bash
# Chạy TOÀN BỘ PHẦN E của CHAY_VM_BA_TANG bằng MỘT lệnh, để qua đêm.
#
# Tự làm luôn hai chỗ trước đây phải hiệu chỉnh tay giữa chừng:
#   1. `tang_nhanh.kieu`      <- kết luận T1 (TRUC_CHUNG -> truc, còn lại -> day_du)
#   2. `ngan_hang.nguong`     <- dich_dieu_kien của T1 chia 4  (xem ghi chú NGUONG bên dưới)
# và tự dừng ở hai cửa chặn: T1=KHONG_CO_TRUC, và U1 không hơn U0.
#
# Dùng:
#   cd ~/RaybanMeta/uav-continual-learning
#   tmux new -s dem
#   nohup bash scripts/chay_dem_ba_tang.sh > dem.log 2>&1 &
#   tail -f dem.log            # Ctrl+B rồi D để rời tmux, tắt máy Mac cũng không sao
#
# Biến môi trường (không bắt buộc):
#   SEEDS="0 1 2"     seed cần chạy
#   ORACLE=1          chạy thêm arm1/arm2 có nhãn sau khi xong track chính
#   NGUONG=0.37       ép ngưỡng thủ công, bỏ qua phần tự tính
#
# Chạy lại lần hai là AN TOÀN: mọi run dùng --skip-existing, run nào đã có metrics.json
# thì bỏ qua. VM chết giữa chừng -> ssh lại, chạy y hệt lệnh này, nó đi tiếp.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
SEEDS="${SEEDS:-0 1 2}"
T1JSON=artifacts_t1/t1_t2.json

log()  { printf '\n=== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
chet() { printf '\n⛔ DỪNG: %s\n' "$1"; exit "${2:-1}"; }

[ -x "$PY" ] || chet "không thấy $PY — tạo venv trước (PHẦN C4 của hướng dẫn)"
log "BẮT ĐẦU · seeds=[$SEEDS] · $(git branch --show-current 2>/dev/null)"

# ---------------------------------------------------------------- E1: T1/T2 + cửa chặn
if [ -f "$T1JSON" ]; then
  log "E1 T1/T2 — đã có $T1JSON, bỏ qua (xoá file nếu muốn chạy lại)"
else
  log "E1 T1/T2 — trục điều kiện (~15 phút)"
  mkdir -p artifacts_t1
  $PY scripts/do_truc_dieu_kien.py --config configs/drift_slda_arm3_dexuat.yaml \
      --json "$T1JSON" 2>&1 | tee run_t1.log
  [ -f "$T1JSON" ] || chet "T1 không sinh được $T1JSON — xem run_t1.log"
fi

KETLUAN=$($PY -c "import json,sys; print(json.load(open('$T1JSON'))['ket_luan_t1'])") || chet "không đọc được $T1JSON"
DICH=$($PY -c "import json,sys; print(json.load(open('$T1JSON'))['dich_dieu_kien'])")

EXTRA=()
case "$KETLUAN" in
  TRUC_CHUNG)
    KIEU=truc
    EXTRA=(slda.tang_nhanh.kieu=truc "slda.tang_nhanh.truc_json=$T1JSON")
    log "T1 = TRUC_CHUNG -> kieu=truc (chống bẫy tỷ lệ lớp T2)" ;;
  MOT_PHAN)
    KIEU=day_du
    EXTRA=(slda.tang_nhanh.kieu=day_du)
    log "T1 = MOT_PHAN -> giữ kieu=day_du. ⚠️ Trục yếu: ghi vào phần hạn chế của báo cáo." ;;
  KHONG_CO_TRUC)
    chet "T1 = KHONG_CO_TRUC (cos < 0,3) — mỗi lớp phản ứng với điều kiện một kiểu, căn chỉnh
     tuyến tính không có cơ sở. Đừng đốt đêm máy; xem KE_HOACH_SUA nhóm B." 2 ;;
  *)
    chet "không hiểu ket_luan_t1='$KETLUAN'" ;;
esac

# NGUONG: khoảng cách trên trục giữa hai chế độ KỀ NHAU = dich_dieu_kien/(n_mode−1) = dich/2
# (n_mode=3: các mức trôi 0% · 50% · 100%). Đặt ngưỡng ở NỬA khoảng đó -> vừa đủ rộng để
# tha thứ trôi trong-chuyến (bien_do=0,25 -> ±12,5%) + nhiễu ước lượng m_tuoi, vừa đủ hẹp
# để KHÔNG gộp hai chế độ kề nhau. Ép tay bằng: NGUONG=0.37 bash scripts/chay_dem_ba_tang.sh
if [ -z "${NGUONG:-}" ]; then
  NGUONG=$($PY -c "print(round(float('$DICH')/4.0, 4))")
  log "ngưỡng tự tính: dich_dieu_kien=$DICH / 4 -> nguong=$NGUONG"
else
  log "ngưỡng ép tay: nguong=$NGUONG"
fi
$PY -c "import sys; sys.exit(0 if float('$NGUONG') > 0 else 1)" || chet "nguong='$NGUONG' không hợp lệ"

# ---------------------------------------------------------------- E2: bench O4
log "E2 bench ba tầng (~1 phút)"
$PY scripts/bench_ba_tang.py 2>&1 | tee bench_ba_tang.log

# ---------------------------------------------------------------- hàm chạy 1 run
chay() {
  local U=$1 S=$2; shift 2
  local DIR=./artifacts_revisit_${U}_s$S
  local LOG=run_${U}_s$S.log
  log "$U seed $S  ->  $LOG"
  if $PY scripts/run_g1.py --config "configs/revisit_${U}.yaml" --skip-existing \
        --set seed="$S" log.dir="$DIR" "$@" > "$LOG" 2>&1; then
    printf '    xong %s  |  %s\n' "$(date +%H:%M:%S)" \
      "$(grep -o 'Lợi ích quay lại PREQ10.*' "$LOG" | head -1)"
  else
    printf '    !!! LỖI %s seed %s — 20 dòng cuối:\n' "$U" "$S"
    tail -20 "$LOG" | sed 's/^/        /'
    return 1
  fi
}

# ---------------------------------------------------------------- E3: U0 + U1
log "E3 track chính U0 + U1 (${SEEDS// /,}) — phần dài nhất"
LOI=0
for S in $SEEDS; do
  chay U0_dongbang  "$S"                || LOI=1
  chay U1_tangnhanh "$S" "${EXTRA[@]}"  || LOI=1
done
[ "$LOI" -eq 0 ] || log "⚠️ có run lỗi ở E3 — vẫn đi tiếp để chấm phần đã xong"

# ---------------------------------------------------------------- cửa chặn U1 > U0
log "CỬA CHẶN — U1 có hơn U0 không?"
$PY scripts/tom_tat_ba_tang.py . --cua-chan
if [ $? -ne 0 ]; then
  chet "U1 không hơn U0 (hoặc thiếu kết quả) -> KHÔNG chạy U2. Kết quả đã có vẫn nằm nguyên
     trong artifacts_revisit_U{0,1}_*; đọc bảng phía trên rồi báo lại." 3
fi

# ---------------------------------------------------------------- E5: U2
log "E5 U2 + ngân hàng chế độ (nguong=$NGUONG)"
for S in $SEEDS; do
  chay U2_nganhang "$S" "${EXTRA[@]}" "slda.ngan_hang.nguong=$NGUONG" || LOI=1
done

# ---------------------------------------------------------------- E6: oracle (tuỳ chọn)
if [ "${ORACLE:-0}" = "1" ]; then
  log "E6 oracle có nhãn (arm1/arm2) — trần trên"
  for S in $SEEDS; do
    chay arm1_lam1 "$S" || LOI=1
    chay arm2_quen "$S" || LOI=1
  done
fi

# ---------------------------------------------------------------- tổng kết
log "XONG — bảng tổng hợp"
$PY scripts/tom_tat_ba_tang.py .
grep -h "số chế độ" run_U2_nganhang_s*.log 2>/dev/null | sed 's/^/  /'
echo
echo "Kéo về Mac: xem PHẦN G của CHAY_VM_BA_TANG_2026-08-04.md"
[ "$LOI" -eq 0 ] || echo "⚠️ có ít nhất một run lỗi — soát lại các file run_*.log"
exit "$LOI"
