# UAV Continual Learning — Nested Learning / Titans / HOPE

Bộ khung dự án (Giai đoạn 0) cho việc áp dụng ý tưởng **Nested Learning / Titans / HOPE**
(Behrouz et al., 2025) vào bài toán **UAV học liên tục**. Thư mục này là *môi trường chung*
cho team 3 người, đã kèm sẵn script kiểm tra và test chạy được.

> Xem `../UAV_NestedLearning_Roadmap.md` (lộ trình) và `../Team_Plan_3nguoi.md` (phân công).

## Cấu trúc thư mục
```
uav-continual-learning/
├── README.md              # file này
├── requirements.txt       # deps (torch cài riêng, xem bước 3)
├── environment.yml        # tùy chọn: tạo env bằng conda
├── pyproject.toml         # cài package `uavcl` + cấu hình pytest
├── configs/
│   ├── default.yaml       # "hợp đồng" config chung cả team
│   ├── g1_smoke.yaml      # G1: dữ liệu giả + tinycnn — kiểm tra pipeline (~1 phút, CPU)
│   ├── g1_eurosat.yaml    # G1: EuroSAT nhẹ (10 class, 5 task) — chạy nhanh
│   └── g1_resisc45.yaml   # G1: RESISC45 chính (45 class, 9 task) — bảng số mốc
├── src/uavcl/
│   ├── data/              # G1 (N1) ✓: sources (resisc45/eurosat/synthetic) + stream + loaders
│   ├── models/            # backbone ✓ + classifier/mask ✓ + ncm ✓ (G1); memory/cms/hope (G2–G4)
│   ├── methods.py         # G1 ✓: baseline FineTune + EWC + NCM (kèm footprint bộ nhớ)
│   ├── engine.py          # G1 ✓: vòng lặp train/eval CL -> ma trận R
│   ├── metrics/           # forgetting/BWT/accuracy + open-set (AUC/EER/TAR@FAR)
│   └── utils/             # seed, config yaml
├── scripts/
│   ├── check_env.py       # "bác sĩ" môi trường
│   ├── smoke_titans.py    # smoke: bộ nhớ Titans chạy được
│   ├── smoke_backbone.py  # smoke: backbone trích feature
│   ├── run_g1.py          # G1 ✓: chạy 1 thí nghiệm từ config
│   └── compare_g1.py      # G1 ✓: gộp các run thành bảng số mốc
└── tests/
    ├── test_metrics.py    # test thuần numpy (chạy pass ngay)
    ├── test_stream.py     # test chia task — thuần Python (chạy pass ngay)
    └── test_g1_smoke.py   # end-to-end trên dữ liệu giả (cần torch, tự skip nếu thiếu)
```

## Yêu cầu
- Python 3.10 (khuyến nghị), git, và conda **hoặc** venv.
- GPU tùy chọn: NVIDIA/CUDA (Linux) hoặc Apple Silicon/MPS (Mac). CPU vẫn chạy được smoke test.

## Cài đặt từ đầu

**Bước 1 — Lấy code & khởi tạo git**
```bash
cd uav-continual-learning
git init && git add -A && git commit -m "G0: project skeleton"
```

**Bước 2 — Tạo môi trường ảo** (chọn 1 trong 2)

Conda:
```bash
conda create -n uavcl python=3.10 -y
conda activate uavcl
```
hoặc venv:
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**Bước 3 — Cài PyTorch (tùy nền tảng, cài TRƯỚC)**
- macOS (Apple Silicon/Intel): `pip install torch torchvision`
- Linux + CUDA 12.1: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- CPU-only (Linux/Windows): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

  Kiểm tra trang chính thức nếu CUDA khác: https://pytorch.org/get-started/locally/

**Bước 4 — Cài phần còn lại + package**
```bash
pip install -r requirements.txt
pip install -e .                 # cài `uavcl` ở chế độ editable
```

## Kiểm tra môi trường (bắt buộc trước khi sang G1)
```bash
python scripts/check_env.py       # mọi dòng nên là [OK]
python scripts/smoke_titans.py    # -> [OK] NeuralMemory forward ... backward ok
python scripts/smoke_backbone.py  # -> [OK] features ... -> embedding ...
pytest -q                         # -> 20 passed (15 nếu chưa cài torch)
```
`check_env.py` và `pytest` đã được xác minh chạy đúng. Hai smoke test cần torch nên chỉ chạy
sau bước 3–4 (chúng tự báo `[SKIP]` kèm hướng dẫn nếu thiếu thư viện).

## Ai sở hữu phần nào (map với `Team_Plan_3nguoi.md`)
- `src/uavcl/data/` → **N1** (G1): tách task stream + loader tuần tự.
- `src/uavcl/metrics/` → **N1**: đã có `average_accuracy`, `backward_transfer`, `average_forgetting`, `forward_transfer` + bộ open-set (AUC/EER/TAR@FAR).
- `src/uavcl/models/backbone.py` → **N3** (G1): ảnh → chuỗi đặc trưng.
- `src/uavcl/models/memory.py` → **N2** (G2): bọc `NeuralMemory` của Titans.
- `src/uavcl/models/cms.py` → **N2 + N3** (G3 ⭐): khối MLP đa tần số retrofit lên backbone.
- `src/uavcl/models/hope.py` → **N2** (G4): ghép Titans + CMS.
- `configs/default.yaml` → cả team: chỉnh chung, giữ là "hợp đồng" giao tiếp giữa các module.

## G1 — khung thí nghiệm + baseline (CODE ĐÃ CÓ, cần chạy để lấy số)

Dataset đã chốt: **RESISC45** (45 class ảnh trên không, tự tải từ HuggingFace ~426MB);
**EuroSAT** làm bộ nhẹ để debug; **synthetic** để smoke test không cần mạng.

Giao thức: class-incremental — chia class thành các task học tuần tự; train task t chỉ
thấy class của t; eval sau task t chấm trên *mọi* class đã thấy (mask logits, xem
`src/uavcl/models/classifier.py`). Ma trận R -> `uavcl.metrics` (accuracy/forgetting/BWT).

```bash
# 0) pipeline có chạy không? (~1 phút, CPU, không cần mạng)
python scripts/run_g1.py --config configs/g1_smoke.yaml
pytest -q                                   # giờ phải là: 20 passed (khi có torch)

# 1) bộ nhẹ trước (EuroSAT, resnet18) — đủ 5 baseline theo Team_Plan G1
for m in finetune ewc replay lwf ncm; do
  python scripts/run_g1.py --config configs/g1_eurosat.yaml --method $m
done

# 2) bảng số mốc chính (RESISC45, ViT-S) — cần GPU thì mới nhanh (ncm thì CPU cũng ổn)
for m in finetune ewc replay lwf ncm; do
  python scripts/run_g1.py --config configs/g1_resisc45.yaml --method $m
done

# 3) gộp thành bảng baseline (artifacts/results/baseline_table.md)
python scripts/compare_g1.py
```
Chỉnh nhanh không sửa file: `--set train.lr=5e-5 data.num_tasks=5 backbone.freeze=true`.

**Kỳ vọng đọc số (5 baseline):** finetune quên nặng nhất (Forgetting cao, BWT âm sâu);
LwF đỡ một phần (không cần dữ liệu cũ); EWC đỡ một phần (phạt tham số); **replay thường là
baseline kinh điển MẠNH NHẤT** (ôn lại ảnh cũ trực tiếp); **NCM (backbone đóng băng +
prototype) gần như KHÔNG quên** — bài học lấy từ project Meta-Rayban/CPM: trên đặc trưng
đóng băng, forgetting gần như biến mất. Vì vậy replay và NCM là 2 baseline "khó chịu" nhất:
từ G2 (Titans) và G3 (CMS), model phải thắng **cả 5 dòng này** — hoặc chỉ ra chỗ chúng gãy
(NCM: feature không đủ tách class mới / domain shift; replay: tốn RAM lưu ảnh + vấn đề
riêng tư) — thì dự án mới có ý nghĩa. Bảng so sánh đã kèm **chi phí** (`trainable_params`,
`method_extra_floats`): cùng accuracy thì method rẻ hơn thắng (EWC phình 2×P mỗi task;
replay phình theo buffer; LwF giữ 1 bản teacher lúc train; NCM bị chặn C×D).
FWT (forward transfer) đo được khi bật `train.eval_future: true`.

## Checklist G1
- [ ] Cả 3 máy: `pytest -q` -> 20 passed; `run_g1.py --config configs/g1_smoke.yaml` chạy hết.
- [ ] N1: chạy EuroSAT đủ 5 method, đọc hiểu `acc_matrix.csv` (hàng i = sau khi học task i); bật thử `train.eval_future=true` để có FWT.
- [ ] N3: chạy RESISC45 đủ 5 method; thử `backbone.freeze=true` vs `false`, ghi lại thời gian.
- [ ] N2: tinh chỉnh núm của 3 baseline chống quên: `ewc_lambda` (100/1000/10000), `replay.buffer_per_class` (5/20/50), `lwf_lambda` (0.5/1/2); đọc `engine.py` + `methods.py` (hook `begin_task`/`penalty`/`extra_batch_loss`/`end_task`) — G2 sẽ cắm Titans vào đúng khung này. Tham khảo delta-rule đa tầng dễ đọc: `Meta-Rayban/cpm/memory.py` (TierMemory).
- [ ] Cả team: `compare_g1.py` ra bảng ≥ 10 dòng (2 dataset x 5 method) -> **cổng G1 đạt** (plan yêu cầu ≥3 baseline ổn định).
- [ ] Thảo luận: replay & NCM đứng ở đâu? Nếu NCM ~ finetune-full về accuracy và không quên -> luận điểm G2/G3 phải nhắm vào chỗ chúng yếu (feature không đủ tách class mới / thích nghi domain shift / chi phí bộ nhớ).

## Checklist G0 (đánh dấu khi xong)
- [ ] N1: repo + môi trường chung chạy; `check_env.py` toàn `[OK]`; chốt shortlist dataset.
- [ ] N2: `smoke_titans.py` chạy `[OK]`; đọc obekt + kmccleary, viết note cơ chế CMS/Titans.
- [ ] N3: `smoke_backbone.py` chạy `[OK]` (thử cả `--pretrained`); xác nhận pipeline GPU.
