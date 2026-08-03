#!/usr/bin/env bash
# Gom kết quả từ 3 VM về Mac, mỗi máy một thư mục riêng (không đè nhau).
#
# Vì sao cần script: ba máy khác ZONE, khác kiểu tên artifact, và repo trên mỗi VM nằm
# dưới USER khác nhau (/home/vum257792/... chứ không phải /home/minhvu/...). Dán tay dễ
# sai một trong ba chỗ đó.
#
# Dùng:
#   bash scripts/collect_vms.sh            # kiểm trạng thái + kéo về nếu đã xong
#   bash scripts/collect_vms.sh --check    # CHỈ kiểm, không kéo
#   bash scripts/collect_vms.sh --force    # kéo kể cả khi còn đang chạy (lấy dở dang)
set -uo pipefail

DICH="${HOME}/Desktop/Raybanmeta/result_test/final"

# tên_vm | zone | nhãn thư mục | pattern file cần đóng gói
MAY=(
  "uavcl-cos|us-east1-b|titans_gates|artifacts_9t_* run_9t_*.log"
  "uavcl-sdc|us-central1-a|titans_ctrl|artifacts_9t_* run_9t_*.log"
  "uavcl-slda|us-east1-b|slda_b2|artifacts_b2_* run_b2_*.log b2_campaign.log bench_vm_*"
)

CHE_DO="${1:-}"

echo
echo "=============================================================="
echo " 1/3  KIỂM TRA TRẠNG THÁI 3 MÁY"
echo "=============================================================="
CON_CHAY=0
for m in "${MAY[@]}"; do
  IFS='|' read -r VM ZONE NHAN _ <<< "$m"
  printf "  %-12s (%s) ... " "$VM" "$ZONE"
  N=$(gcloud compute ssh "$VM" --zone="$ZONE" --command='pgrep -f run_g1.py | wc -l' 2>/dev/null | tr -d '[:space:]')
  if [[ -z "$N" ]]; then
    echo "KHÔNG KẾT NỐI ĐƯỢC (máy tắt? sai zone?)"
    CON_CHAY=$((CON_CHAY + 1))
  elif [[ "$N" == "0" ]]; then
    echo "✅ đã xong"
  else
    echo "⏳ còn $N tiến trình đang chạy"
    CON_CHAY=$((CON_CHAY + 1))
  fi
done

if [[ "$CHE_DO" == "--check" ]]; then
  echo; echo "  (chế độ --check: dừng ở đây)"; exit 0
fi

if [[ "$CON_CHAY" -gt 0 && "$CHE_DO" != "--force" ]]; then
  echo
  echo "  ⚠️  Còn $CON_CHAY máy chưa xong hoặc không kết nối được."
  echo "      Đợi thêm, hoặc chạy lại với --force để lấy kết quả dở dang."
  exit 1
fi

mkdir -p "$DICH"

echo
echo "=============================================================="
echo " 2/3  ĐÓNG GÓI TRÊN VM + KÉO VỀ"
echo "=============================================================="
LOI=0
for m in "${MAY[@]}"; do
  IFS='|' read -r VM ZONE NHAN PATTERN <<< "$m"
  echo
  echo "--- $VM -> $NHAN"

  # Tự dò đường dẫn repo (mỗi VM một user khác nhau) + sudo để đọc được mọi user.
  gcloud compute ssh "$VM" --zone="$ZONE" --command="sudo bash -c 'shopt -s nullglob; \
    D=\$(ls -d /home/*/RaybanMeta/uav-continual-learning 2>/dev/null | head -1); \
    if [ -z \"\$D\" ]; then echo KHONG_TIM_THAY_REPO; exit 1; fi; \
    cd \$D && tar czf /tmp/res_${NHAN}.tgz ${PATTERN} 2>/dev/null; \
    chmod 644 /tmp/res_${NHAN}.tgz; \
    echo GOI_XONG \$D \$(du -h /tmp/res_${NHAN}.tgz | cut -f1)'" 2>/dev/null

  if [[ $? -ne 0 ]]; then
    echo "  ❌ đóng gói thất bại"; LOI=$((LOI + 1)); continue
  fi

  gcloud compute scp "${VM}:/tmp/res_${NHAN}.tgz" "$DICH/" --zone="$ZONE" 2>/dev/null \
    && echo "  ✅ đã kéo về $DICH/res_${NHAN}.tgz" \
    || { echo "  ❌ scp thất bại"; LOI=$((LOI + 1)); }
done

echo
echo "=============================================================="
echo " 3/3  GIẢI NÉN (mỗi máy một thư mục riêng)"
echo "=============================================================="
cd "$DICH" || exit 1
for m in "${MAY[@]}"; do
  IFS='|' read -r _ _ NHAN _ <<< "$m"
  [[ -f "res_${NHAN}.tgz" ]] || { echo "  bỏ qua $NHAN (không có tarball)"; continue; }
  mkdir -p "$NHAN"
  tar xzf "res_${NHAN}.tgz" -C "$NHAN" && echo "  ✅ $NHAN/"
done

echo
echo "=============================================================="
echo " XONG. Phân tích:"
echo "=============================================================="
cat <<EOF

  cd ~/Desktop/Raybanmeta

  # bảng tổng cả 3 máy (cùng split, so trực tiếp được)
  python3 uav-continual-learning/scripts/compare_all.py result_test/final

  # quỹ đạo cổng η/α của 2 máy Titans — cổng có sống qua 9 task không?
  python3 uav-continual-learning/scripts/trace_gates.py result_test/final

  # riêng arm SLDA, xem nhanh accuracy 12 run
  grep -h "Average Accuracy" result_test/final/slda_b2/run_b2_*.log

EOF
if [[ "$LOI" -gt 0 ]]; then
  echo "  ⚠️  Có $LOI máy lỗi — kiểm lại ở trên."
fi
