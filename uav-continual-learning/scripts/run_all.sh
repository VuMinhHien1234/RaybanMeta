#!/usr/bin/env bash
# MỘT LỆNH chạy tất cả + xuất báo cáo docx. Truyền thêm cờ nếu cần:
#   bash scripts/run_all.sh --quick        # chỉ EuroSAT (thử trước)
#   bash scripts/run_all.sh --shutdown     # GCP: tự tắt máy khi xong
set -e
cd "$(dirname "$0")/.."
exec python scripts/run_all.py "$@"
