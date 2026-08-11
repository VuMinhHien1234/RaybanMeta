# Hướng dẫn đọc code — dự án UAV Continual Learning

> **Đối tượng:** người biết Python, chưa quen machine learning / continual learning.
> **Cập nhật:** 2026-08-04 (bao trùm cả phần mới: drift, revisit, ba tầng, pha 2 không nhãn).
> **Thay cho:** `HUONG_DAN_DOC_CODE.md` (24/07 — vẫn đúng nhưng chỉ tới G4, thiếu 40% code hiện tại).

Cách dùng: đọc Phần 1–3 một mạch (~20 phút) để có bản đồ. Sau đó theo **lộ trình ở Phần 4**,
mỗi buổi một chặng. Phần 5–9 là từ điển tra cứu — không cần đọc tuần tự.

---

## Mục lục

1. [Dự án giải bài toán gì](#1)
2. [Từ vựng tối thiểu](#2)
3. [Kiến trúc: dữ liệu chảy như thế nào](#3)
4. [Lộ trình đọc — 6 chặng](#4)
5. [Giải thích từng file: tầng dữ liệu](#5)
6. [Giải thích từng file: tầng mô hình](#6)
7. [Giải thích từng file: engine, method, metric, optimizer](#7)
8. [scripts/ — chạy cái gì bằng lệnh nào](#8)
9. [Đọc kết quả · debug · mở rộng](#9)

---

<a name="1"></a>
## 1. Dự án giải bài toán gì

### 1.1. Kịch bản gốc

Một UAV bay tuần tra **một khu vực cố định**.

- **Lần đầu** — bay qua, dữ liệu được **gán nhãn** thủ công. Đây là *pha hiệu chỉnh*, làm dưới đất, một lần, không giới hạn tính toán.
- **Những lần sau** — bay lại **cùng khu vực**, nhưng điều kiện quan sát đã khác: giờ trong ngày, mùa, mây, sương, bụi bám ống kính, độ cao. **Không có nhãn. Không có người.**

Hệ phải làm ba việc, và ba việc này đổi với **tần số lệch nhau hàng triệu lần**:

| | Việc | Đổi khi nào |
|---|---|---|
| a | Nhận đúng *"đây là sân bay, kia là ruộng"* | gần như không bao giờ |
| b | Nhận ra *"điều kiện hôm nay giống chuyến tháng trước"* | mỗi chuyến bay |
| c | Bám theo *"nắng đang gắt dần suốt 10 phút"* | mỗi khung hình |

Ba tần số đó chính là lý do **kiến trúc BA TẦNG** ra đời (§6.6). Nếu chỉ nhớ một điều từ tài liệu này thì nhớ bảng trên — mọi lựa chọn thiết kế trong code đều quy về nó.

Nguồn: `BAI_TOAN_VA_MUC_TIEU_2026-08-04.md` (ở thư mục gốc, ngoài repo code).

### 1.2. Vấn đề kỹ thuật trung tâm

Mạng nơ-ron học cái mới thì **ghi đè** cái cũ — gọi là **catastrophic forgetting** (quên thảm hoạ). Dự án thử nhiều chiến lược chống quên rồi **chấm điểm** xem chiến lược nào tốt nhất, dưới hai chế độ khác nhau:

- **class-incremental** — task mới = **lớp mới** (đợt 1 học 5 loại địa hình, đợt 2 học 5 loại khác). Đây là chế độ chuẩn của ngành, dùng cho G1–G4.
- **domain-incremental / revisit** — tập lớp **không đổi**, task mới = **điều kiện quan sát mới**. Đây mới là bài toán UAV thật (§1.1), thêm vào từ 03/08.

### 1.3. Các giai đoạn (giải mã tiền tố G1/G2/G3/G4 gặp khắp nơi)

| Tên | Là gì | Trạng thái |
|---|---|---|
| **G1** | 5 baseline kinh điển: finetune, EWC, Replay, LwF, NCM | ✅ có số, xem `docs/KET_LUAN_G1.md` |
| **G2** | **Titans** — bộ nhớ nơ-ron tự ghi lúc chạy (NL.pdf §8) | ✅ có số |
| **G3** | **CMS** — Continuum Memory System: chia mạng thành tầng cập nhật với chu kỳ khác nhau | ✅ có số |
| **G4** | **HOPE** — Titans + CMS chạy chung | ✅ có số |
| **#21/#22/#23…** | các task lẻ: SLDA, latent replay, checkpoint, open-set | ✅ |
| **D10/D11** | thí nghiệm λ (hệ số quên) trên stream trôi | ✅ 04/08 |
| **M1/M2/M3, U0/U1/U2** | kiến trúc **BA TẦNG** + track thí nghiệm hiện tại | 🚧 đang chạy |

**Kết quả G1 làm mốc** (RESISC45, 9 task × 5 lớp) — con số bạn sẽ thấy trích dẫn khắp code:

| Method | Avg Acc ↑ | Forgetting ↓ | Extra floats |
|---|---:|---:|---:|
| finetune | 0.4144 | 0.6133 | 0 |
| ewc | 0.4259 | 0.5999 | 390M ⚠ |
| lwf | 0.5900 | 0.2777 | 21.7M |
| **ncm** | 0.6933 | 0.1000 | **17K** |
| **replay** | **0.7937** | **0.0787** | 135M ⚠ |

Đọc bảng này là hiểu định hướng cả dự án: **NCM rẻ gấp 8000 lần replay mà chỉ kém 10 điểm.** Mọi phương pháp phức tạp về sau phải thắng NCM mới có ý nghĩa. Khoảng 10 điểm giữa NCM và replay chính là "giá trị của việc được thích nghi feature" — thứ G2–G4 đi giành lấy mà không lưu ảnh thô.

---

<a name="2"></a>
## 2. Từ vựng tối thiểu

Bạn biết Python rồi, nên chỉ cần các từ chuyên ngành:

| Từ | Nghĩa trong dự án này |
|---|---|
| **backbone** | mạng trích đặc trưng: ảnh (3,224,224) → vector 384 chiều. Ở đây là ViT-S pretrained. |
| **feature / đặc trưng** | vector 384-d nói trên. Ký hiệu `feats`, `h`, `f` trong code. |
| **head** | lớp cuối cùng: feature → điểm số cho mỗi lớp (`logits`). Ở đây là `nn.Linear(384, 45)`. |
| **logits** | điểm số thô trước softmax, shape `(B, C)`. |
| **task / chuyến** | một đợt học. `TaskSpec` = mô tả một task. Với revisit, "task" = "một chuyến bay". |
| **stream** | danh sách các task theo thứ tự thời gian. |
| **frozen / đóng băng** | `requires_grad=False` — tham số không được cập nhật. |
| **prototype** | vector trung bình của một lớp trong không gian feature. |
| **replay** | lưu lại mẫu cũ để "ôn bài". |
| **ma trận R** | `R[i][j]` = accuracy trên task `j` **sau khi** học xong task `i`. Toàn bộ kết quả dự án gói trong ma trận này. |
| **state (Titans)** | "cục ký ức" mà bộ nhớ Titans mang theo giữa các lần forward. |
| **λ (lambda/decay)** | hệ số quên: thống kê cũ nhân λ mỗi bước. λ=1 → không quên bao giờ. |

**Quy ước đặt tên:** dự án trộn tiếng Anh và **tiếng Việt không dấu**. File/hàm tiếng Việt là phần mới (từ 02/08): `tang_nhanh.py` (tầng nhanh), `ngan_hang_che_do.py` (ngân hàng chế độ), `hap_thu_khong_nhan` (hấp thụ không nhãn), `chuyen` (chuyến bay), `troi` (trôi/drift), `muc_troi` (mức trôi). Không phải lỗi — là chủ ý để phân biệt phần "bài toán UAV thật" với phần "baseline chuẩn ngành".

**Code đã được chú thích rất dày.** Mọi dòng quan trọng có comment `# ↳ ...` giải thích. Tài liệu này không lặp lại chúng — nó cho bạn **thứ tự đọc** và **bức tranh nối các file**, thứ mà comment trong file không làm được.

---

<a name="3"></a>
## 3. Kiến trúc: dữ liệu chảy như thế nào

### 3.1. Cây thư mục

```
uav-continual-learning/
├── configs/           38 file yaml — MỖI FILE = MỘT THÍ NGHIỆM
├── scripts/           điểm vào: chạy, đo, so sánh, sinh báo cáo
├── src/uavcl/         toàn bộ thư viện
│   ├── data/          ảnh → task stream → DataLoader  (+ drift, revisit)
│   ├── models/        các kiến trúc: classifier, NCM, SLDA, Titans, HOPE, ba tầng
│   ├── metrics/       chấm điểm: continual, open-set, revisit
│   ├── optim/         M3 optimizer + CMSOptimizer
│   ├── utils/         config, seed, device
│   ├── engine.py      ⭐ VÒNG LẶP CHÍNH — mọi thí nghiệm đi qua đây
│   ├── methods.py     ⭐ 10 CHIẾN LƯỢC chống quên
│   ├── checkpoint.py  lưu/khôi phục giữa chừng
│   └── openset_eval.py chấm "biết nói không biết"
├── tests/             29 file pytest
└── docs/              ghi chú nghiên cứu, kết luận từng giai đoạn
```

Tổng: **~5.000 dòng** trong `src/`, **~2.500 dòng** trong `scripts/`. Không lớn — đọc hết được.

### 3.2. Sơ đồ luồng (đường đi của một lần chạy)

```
     configs/xxx.yaml
            │
            ▼
   scripts/run_g1.py  ────────────────────────────────┐
            │                                          │
   ┌────────┴────────┐                                 │
   │  1. DỮ LIỆU     │                                 │
   │  get_source()   │  data/sources.py    ảnh + nhãn  │
   │  build_stream() │  data/stream.py     chia task   │
   │  build_task_    │  data/loaders.py    DataLoader  │
   │    loaders()    │  (+ data/drift.py: thêm sương/nắng)
   └────────┬────────┘                                 │
   ┌────────┴────────┐                                 │
   │  2. MÔ HÌNH     │  models/backbone.py   ViT-S     │
   │  build_backbone │  models/classifier.py head      │
   │  + chọn lớp bọc │  hoặc ncm/slda/titans_head/hope │
   └────────┬────────┘                                 │
   ┌────────┴────────┐                                 │
   │  3. CHIẾN LƯỢC  │  methods.py                     │
   │  build_method() │  finetune|ewc|replay|lwf|ncm|   │
   │                 │  slda|titans|cms|hope|latent_…  │
   └────────┬────────┘                                 │
            ▼                                          │
   ┌─────────────────────────────────────┐             │
   │  4. engine.run_continual()   ⭐      │             │
   │  for t in tasks:                     │             │
   │      method.begin_task()             │             │
   │      train_one_task()  ← optim/      │             │
   │      method.end_task()               │             │
   │      for j in 0..t: R[t,j]=evaluate()│             │
   └────────┬─────────────────────────────┘             │
            ▼                                          │
   ┌─────────────────┐                                 │
   │  5. CHẤM ĐIỂM   │  metrics/continual.py           │
   │                 │  metrics/revisit.py (O1/O2/O3)  │
   └────────┬────────┘                                 │
            ▼                                          ▼
   artifacts/results/<tên_run>/{acc_matrix.csv, metrics.json, config.yaml}
                              │
                              ▼
              scripts/compare_all.py → bảng so sánh
              scripts/make_report.py → báo cáo .docx
```

### 3.3. Ba hợp đồng giữ toàn bộ dự án dính vào nhau

Hiểu ba điều này là hiểu vì sao đổi model không phải sửa engine:

**Hợp đồng 1 — nhãn luôn là id toàn cục.**
Head luôn có đủ `num_classes` cột (45 với RESISC45), kể cả ở task 0. Không remap nhãn về `0..4`. Việc giới hạn "task này chỉ được đoán lớp nào" do **mask logits** làm ở engine, không do model. Xem `models/classifier.py::mask_logits`.

**Hợp đồng 2 — mọi model chỉ cần `forward(x) -> logits (B, C)`.**
Nhờ vậy `ContinualClassifier`, `NCMClassifier`, `SLDAClassifier`, `TitansClassifier`, `HOPEClassifier` cắm vào cùng một `engine.run_continual` mà không sửa một dòng engine. Các phương thức phụ (`features()`, `export_state()`, `hap_thu_khong_nhan()`) đều **tuỳ chọn** — engine kiểm tra bằng `hasattr` trước khi gọi.

**Hợp đồng 3 — method chỉ cài 5 hook.**

```python
method.begin_task(model, device, allowed)                  # trước task
method.penalty(model)                     -> tensor | None  # phạt tham số (EWC)
method.extra_batch_loss(model, x, logits_full, device)      # loss thêm (Replay, LwF)
method.end_task(model, loader, device, allowed)            # sau task
method.footprint_floats(model)            -> int            # chi phí bộ nhớ thêm
```

`FineTune` là lớp cha cài cả 5 hook rỗng. Mọi method khác kế thừa và **chỉ ghi đè hook mình cần**. Khi đọc một method mới, chỉ cần xem nó ghi đè cái gì.

---

<a name="4"></a>
## 4. Lộ trình đọc — 6 chặng

Đánh dấu `[x]` khi xong. Mỗi chặng nên chạy được test tương ứng trước khi sang chặng sau.

### Chặng 0 — chạy được đã (30 phút)

```bash
cd uav-continual-learning
python scripts/check_env.py            # kiểm môi trường
pytest tests/test_stream.py tests/test_metrics.py -q   # test thuần Python, không cần torch
python scripts/run_g1.py --config configs/g1_smoke.yaml   # end-to-end ~1 phút, dữ liệu giả
```

Xong bước này bạn có `artifacts/results/synthetic_finetune_seed0/acc_matrix.csv`. **Mở nó ra nhìn.** Đó là ma trận R — đích đến của mọi thứ còn lại.

- [ ] Chặng 0

### Chặng 1 — hiểu "task" và "chấm điểm" (thuần Python, không cần torch)

| # | File | Dòng | Đọc để biết |
|---|---|---:|---|
| 1 | `src/uavcl/data/stream.py` | 246 | task được chia ra sao |
| 2 | `src/uavcl/metrics/continual.py` | 83 | 5 chỉ số đọc từ ma trận R |
| 3 | `tests/test_stream.py`, `tests/test_metrics.py` | | test viết bằng số tính tay — đối chiếu để chắc mình hiểu đúng |

Đây là hai file **quan trọng nhất mà dễ nhất**. Chúng cố ý không import torch để chạy được ở mọi máy.

- [ ] Chặng 1

### Chặng 2 — đường dữ liệu (cần torch)

| # | File | Dòng | Đọc để biết |
|---|---|---:|---|
| 4 | `src/uavcl/data/sources.py` | 194 | 3 dataset được bọc về cùng một hình dạng |
| 5 | `src/uavcl/data/loaders.py` | 230 | Dataset/DataLoader mỗi task |
| 6 | `src/uavcl/utils/config.py` | 63 | yaml → dict, và `--set a.b=c` |
| 7 | `src/uavcl/models/backbone.py` + `classifier.py` | 143 | ảnh → feature → logits, và `mask_logits` |

- [ ] Chặng 2

### Chặng 3 — trái tim ⭐ (đọc kỹ, có thể mất 2 buổi)

| # | File | Dòng | Đọc để biết |
|---|---|---:|---|
| 8 | `src/uavcl/engine.py` | 358 | ⭐ vòng lặp học liên tục |
| 9 | `src/uavcl/methods.py` | 548 | ⭐ 10 chiến lược chống quên |
| 10 | `scripts/run_g1.py` | 420 | mọi mảnh trên được nối lại thế nào |

**Cách đọc `engine.py`:** đọc `run_continual` trước (dòng ~199 trở đi), bỏ qua mọi nhánh `if` phụ (checkpoint, SDC, NCM-head, pha 2). Bộ khung chỉ có:

```python
for t, spec in enumerate(stream):
    train_one_task(...)          # hoặc method.fit_task nếu gradient_free
    method.end_task(...)
    seen += spec.classes
    for j in range(t + 1):
        R[t, j] = evaluate(model, task_loaders[j]["test"], device, sorted(seen))
```

15 dòng. Phần còn lại là tuỳ chọn bật/tắt bằng config. Đọc xong khung rồi mới quay lại từng nhánh.

**Cách đọc `methods.py`:** đọc `FineTune` (lớp cha) → `EWC` → `Replay` → `LwF`. Bốn cái đó là 80% giá trị. `NCM` và `SLDA` là loại khác hẳn (`gradient_free = True`) — engine đi nhánh riêng cho chúng.

- [ ] Chặng 3

### Chặng 4 — các kiến trúc thay thế

Chọn nhánh theo thứ tự lợi ích/công sức:

| # | File | Dòng | Ghi chú |
|---|---|---:|---|
| 11 | `models/ncm.py` | 63 | **đọc đầu tiên** — 63 dòng mà đứng nhì bảng G1 |
| 12 | `models/slda.py` | 408 | NCM + hiệp phương sai. Là **model chủ lực hiện tại** |
| 13 | `models/seq_adapter.py` + `memory.py` + `titans_head.py` | 375 | Titans (G2) |
| 14 | `models/cms.py` + `optim/cms_optimizer.py` | 315 | CMS (G3) |
| 15 | `models/hope.py` | 57 | HOPE (G4) — chỉ 57 dòng vì kế thừa Titans |
| 16 | `optim/m3.py` | 232 | optimizer đa tầng ký ức |

- [ ] Chặng 4

### Chặng 5 — phần mới: bài toán UAV thật (từ 03/08)

Đây là phần **file cũ không có**, và là hướng đang chạy:

| # | File | Dòng | Ghi chú |
|---|---|---:|---|
| 17 | `data/drift.py` | 139 | mô phỏng nắng/sương/bụi |
| 18 | `data/revisit.py` | 134 | lịch bay tuần hoàn — điều kiện **quay lại** |
| 19 | `metrics/revisit.py` | 116 | ⭐ O1/O2/**O3** — thước đo trung tâm hiện tại |
| 20 | `models/tang_nhanh.py` | 158 | M1 — tầng nhanh |
| 21 | `models/ngan_hang_che_do.py` | 149 | M2 — tầng trung |
| 22 | `slda.py::hap_thu_khong_nhan` | | pha 2 không nhãn |

- [ ] Chặng 5

### Chặng 6 — phần tra cứu khi cần

`checkpoint.py`, `openset_eval.py`, `metrics/openset.py`, `models/state_utils.py`, `models/self_ref_memory.py`, `scripts/bench_*.py`. Đọc khi gặp, không cần đọc trước.

- [ ] Chặng 6

---

<a name="5"></a>
## 5. Giải thích từng file — tầng dữ liệu (`src/uavcl/data/`)

### 5.1. `sources.py` (194 dòng) — nguồn dữ liệu

**Nhiệm vụ:** ba dataset khác nhau về cách lưu, bọc về **cùng một hình dạng** để phần còn lại không cần biết đang dùng bộ nào.

```python
@dataclass
class DataSource:
    name: str
    num_classes: int
    class_names: List[str]
    splits: Dict[str, object]   # 'train' | 'val' | 'test'
```

Mỗi `split` là một đối tượng có đúng 3 thứ: `__len__()`, `get_image(i)`, `.labels`. Có ba cài đặt tuỳ **ảnh nằm ở đâu**:

| Lớp | Ảnh ở đâu | Dùng cho |
|---|---|---|
| `ImagePathSplit` | trên đĩa, lưu đường dẫn | EuroSAT |
| `HFSplit` | trong parquet của HuggingFace | RESISC45 |
| `ArraySplit` | trong RAM (numpy uint8) | synthetic |

Ba hàm tải: `_load_resisc45` (45 lớp, 31.500 ảnh, dataset chính), `_load_eurosat` (10 lớp, nhẹ, để debug — tự chia 80/10/10 vì không có split chính thức), `_load_synthetic` (mỗi lớp = một màu nền + nhiễu Gauss; chạy được không cần mạng).

Điểm vào duy nhất:

```python
source = get_source(cfg["data"])    # đọc cfg["data"]["name"] → tra _REGISTRY
```

Thêm dataset mới = viết một hàm `_load_xxx` + thêm một dòng vào `_REGISTRY`. Không đụng file khác.

> **Lưu ý thiết kế:** mọi lớp ở đây **picklable** (không dùng closure). Cần thế thì `DataLoader(num_workers>0)` mới chạy được trên macOS (dùng `spawn` chứ không `fork`).

### 5.2. `stream.py` (246 dòng) — chia task

**Thuần Python, không import torch** — chủ ý, để test logic chia ở mọi máy.

```python
@dataclass
class TaskSpec:
    task_id: int
    classes: List[int]      # lớp của task này
    train_idx: List[int]    # chỉ số vào split train
    val_idx:   List[int]
    test_idx:  List[int]
```

Bốn hàm dựng stream — **khác nhau ở chỗ "task mới" nghĩa là gì**:

| Hàm | Task mới nghĩa là | Dùng cho |
|---|---|---|
| `build_stream` | **lớp mới** | G1–G4 (chuẩn ngành) |
| `build_stream_with_holdout` | như trên, + giữ lại N lớp không bao giờ train | open-set (#27) |
| `build_domain_stream` | **điều kiện mới**, lớp không đổi | D5 domain-incremental |
| `build_domain_stream` + `stream_type=revisit` | điều kiện mới **và quay lại** | ⭐ bài toán UAV thật |

Ba hàm phụ đáng chú ý:

- `split_classes(num_classes, num_tasks, seed, shuffle=True)` — chia 45 lớp thành 9 nhóm 5. `shuffle` xáo theo seed vì **thứ tự lớp ảnh hưởng kết quả CL** → phải cố định để tái lập được.
- `stratified_split(indices, labels, fraction, seed)` — cắt tỉ lệ giữ cân bằng lớp. Dùng khi dataset không có split sẵn.
- `chia_deu_theo_lop(labels, num_tasks, seed)` — chia theo **mẫu** chứ không theo **lớp**: mọi task đều thấy đủ 45 lớp. Đây là nền của domain-incremental.

> **Vì sao tách `build_domain_stream` khỏi `build_stream`?** Docstring nói thẳng: nếu để lẫn "lớp mới" và "điều kiện trôi" trong cùng một stream thì khi accuracy tụt, **không tách được nguyên nhân**. Đây là kiểu quyết định thiết kế xuất hiện nhiều trong dự án — mỗi thí nghiệm chỉ đổi một biến.

### 5.3. `loaders.py` (230 dòng) — DataLoader

**Đây là chỗ đầu tiên cần torch.**

`build_transforms(image_size, train)` — augment nhẹ + chuẩn hoá theo ImageNet (phải khớp backbone pretrained của timm, sai chuẩn hoá là mất vài điểm accuracy không rõ lý do).

`TaskDataset` — một split của một task. Điểm quan trọng ghi ngay trong docstring:

> Nhãn giữ nguyên **id toàn cục** (không remap) — head có đủ cột cho mọi class, việc giới hạn class nào được dùng do **mask logits ở engine** quyết định.

Đây chính là Hợp đồng 1 ở §3.3.

`TaskDatasetDongThoiGian` — bản pha 2: mức trôi tính **theo vị trí mẫu trong chuyến**, không phải hằng số cả chuyến. Ứng với yêu cầu (c) "nắng gắt dần suốt 10 phút" — điều kiện đổi **trong** chuyến.

```python
def muc_tai(self, k):    # mức trôi của mẫu thứ k
    # đi tuyến tính từ (muc − biên_độ/2) đến (muc + biên_độ/2)
```

`build_task_loaders(source, stream, data_cfg)` → `[{'train':…, 'val':…, 'test':…}, …]`, một dict mỗi task. Đây là thứ engine nhận vào.

`build_eval_loader(...)` — loader không aug, không xáo, trên danh sách chỉ số tuỳ ý. Dùng cho open-set.

### 5.4. `drift.py` (139 dòng) — mô phỏng trôi điều kiện

**Vì sao có file này:** mọi kết quả trước 03/08 đo ở chế độ "lớp mới, điều kiện **tĩnh**". Nhưng câu hỏi ứng dụng thật là *"drone bay hàng tháng, nắng/mùa/sương đổi dần — model có bám theo được không?"*

Năm phép biến đổi, mỗi phép mô phỏng một hiện tượng vật lý:

| Phép | Hiện tượng thật |
|---|---|
| đổi độ sáng | giờ trong ngày |
| đổi tương phản | mây / độ trong khí quyển |
| ám màu | sáng vàng chiều muộn vs xanh giữa trưa |
| làm mờ | sương, ống kính bẩn |
| thêm nhiễu | ISO cao lúc thiếu sáng |

```python
class ApDungTroi:
    def __init__(self, muc, jitter=0.15, seed=0, sinh_ngau=True)
    def __call__(self, x)         # x: tensor (C,H,W) trong [0,1]
```

Tất định theo `muc`, cộng rung ngẫu nhiên `jitter` để không phải hai ảnh cùng mức trôi thì y hệt nhau.

`muc_troi(task_idx, num_tasks, mode, severity)` → mức trôi của task thứ t, trong `[0, severity]`:
- `linear` (mặc định) — tăng đều 0 → 1 qua các task, giống mùa chuyển dần
- `step` — nhảy bậc ở giữa stream, để so sánh

**Thứ tự trong `build_drift_transform` là có chủ đích:**

```
hình học → ToTensor → TRÔI → Normalize
```

Trôi phải nằm trên thang `[0,1]` mới đúng nghĩa vật lý. Đặt sau Normalize là sai (nhiễu cộng vào thang đã chuẩn hoá không tương ứng với hiện tượng nào cả).

`bang_muc_troi()` in bảng mức trôi từng task ra log — để về sau đọc log có **bằng chứng** trôi đúng như thiết kế.

### 5.5. `revisit.py` (134 dòng) — bay lặp lại ⭐

**Đây là file làm rõ nhất tư duy của dự án.** Đọc docstring đầu file:

> Stream trôi của D10 tăng **đơn điệu** 0% → 100%, điều kiện cũ **không bao giờ quay lại**. Ở đó quên điều kiện cũ gần như miễn phí — đó chính là lý do arm λ=0,99 thắng λ=1 tới 3,91 điểm. Nhưng bài toán thật thì drone **bay lại cùng chỗ**.

Nói cách khác: thí nghiệm D10 đo đúng, nhưng đo một tình huống không tồn tại trong thực tế. File này sửa chỗ đó.

```python
@dataclass
class ChuyenBay:      # một chuyến: đi qua TOÀN BỘ khu vực, dưới MỘT điều kiện
    ...
    mode_that: int    # nhãn điều kiện THẬT — chỉ để chấm điểm, model KHÔNG thấy
    lan_gap_mode: int # đây là lần thứ mấy gặp điều kiện này
```

`lich_bay(n_chuyen, che_do, chu_ky, ...)` sinh lịch điều kiện:
- `tuan_hoan` — hình sin, chu kỳ `chu_ky` chuyến. Chuyến `chu_ky+1` lặp lại điều kiện chuyến 1. **← kịch bản chính**
- `troi_dan` — tăng đều, không lặp (giống D10, để nối lại kết quả cũ)

`lich_tu_cfg(n_chuyen, drift_cfg)` — **một chỗ đọc tham số duy nhất**. Docstring giải thích vì sao cần:

> trước đây `run_g1.py` và `loaders.py` mỗi nơi tự đọc `drift_cfg` và tự điền mặc định. Hiện khớp nhau, nhưng sửa mặc định một nơi là hai lịch lệch **ngầm** — model train trên lịch này, chấm trên lịch khác.

`kiem_lich(lich)` — **chặn cấu hình vô nghĩa trước khi đốt giờ máy**. Lịch không có chuyến nào lặp lại điều kiện cũ thì thước đo O3 không tính được, cả thí nghiệm mất mục tiêu. Ném lỗi ngay thay vì chạy 8 tiếng rồi mới biết.

> Kiểu "cửa chặn" này lặp lại nhiều nơi trong dự án (`run_g1.py` cũng chặn `stream_type=revisit` mà `drift.enabled=false`). Đó là bài học rút ra sau vài lần mất giờ máy — đáng học theo.

---

<a name="6"></a>
## 6. Giải thích từng file — tầng mô hình (`src/uavcl/models/`)

### 6.1. `backbone.py` (63 dòng)

```python
backbone, feat_dim = build_backbone(cfg["backbone"])   # module(x) -> (B, feat_dim)
```

Hai chế độ: tên model timm (`vit_small_patch16_224` → `feat_dim=384`, dùng `num_classes=0` để lấy feature thay vì logits) hoặc `tinycnn` (3 khối conv + GAP, không cần mạng, chỉ để smoke test).

`cfg["backbone"]["freeze"]` quyết định backbone có được train không. Với SLDA/NCM luôn `true`.

### 6.2. `classifier.py` (80 dòng) — quan trọng hơn kích thước

```python
class ContinualClassifier(nn.Module):
    def forward(self, x)              -> logits (B, num_classes)
    def forward_from_feats(self, feats)  # bỏ qua backbone — dùng cho latent replay
```

Hai thứ đáng nhớ:

**`mask_logits(logits, allowed)`** — giữ nguyên cột trong `allowed`, đè các cột khác bằng `MASK_FILL` (số rất âm). Quy ước cả dự án:
- khi **train** task t → mask về đúng lớp của task t
- khi **eval** sau task t → mask về lớp **đã thấy** (task 0..t)

**`CosineHead`** — chuẩn hoá cả feature lẫn trọng số trước khi nhân, nên logit chỉ phụ thuộc **hướng**, bất biến **độ lớn**. Sửa "recency bias" của `nn.Linear`: trọng số lớp mới phình to lấn lớp cũ — một nguyên nhân forgetting mà thí nghiệm NCM-head đã phơi ra. Bật bằng `head: cosine` trong yaml.

### 6.3. `ncm.py` (63 dòng) — rẻ nhất, mạnh bất ngờ

Backbone **đóng băng**, mỗi lớp giữ một **prototype** = trung bình feature của lớp đó. Dự đoán = cosine gần nhất.

```python
def update_prototypes(self, feats, ys)   # cộng dồn tổng feature theo lớp
def prototypes(self)                     # trung bình đã chuẩn hoá
```

Vì sao gần như không quên: **prototype lớp cũ không bao giờ bị ghi đè** khi học lớp mới. Không gradient, bộ nhớ chặn ở `O(C·D)` = 45×384 ≈ 17K float.

`train(mode)` được ghi đè để giữ backbone ở `eval()` kể cả khi gọi `.train()` — BatchNorm không được cập nhật.

> Docstring nói thẳng: *"Nếu Titans (G2) / CMS (G3) không thắng nổi cái này thì phải xem lại."* Đó là vai trò của NCM trong dự án: **mốc dưới không được phép thua**.

### 6.4. `slda.py` (408 dòng) — model chủ lực hiện tại ⭐

Streaming Linear Discriminant Analysis (Hayes & Kanan 2020). Như NCM nhưng học thêm **ma trận hiệp phương sai chung** của feature → ranh giới lớp xét cả "hình dạng" đám mây feature, không chỉ khoảng cách tới tâm. Trên RESISC45 hơn NCM **+11,5 điểm**.

Ba bộ đếm tích luỹ, không giữ lại mẫu nào:

```
s_c   tổng feature của lớp c        →  μ_c = s_c / n_c
n_c   số mẫu lớp c
G     ma trận Gram (D×D)            →  Σ_w = (G − Σ_c n_c μ_c μ_cᵀ) / N
```

Rồi `Λ = (Σ_w + ε I)⁻¹`, dự đoán `argmax_c (μ_c·Λ·f − ½ μ_c·Λ·μ_c)`.

Các núm điều khiển đáng biết:

| Tham số | Tác dụng |
|---|---|
| `cov_mode` | `streaming` (mặc định) \| `identity` (ép Σ=I, khi đó SLDA ≡ NCM) \| `frozen` (đóng băng Σ sau N task) |
| `decay_mean` / `decay_cov` (λ) | hệ số **quên**. λ=1 → không quên. Đây là biến của thí nghiệm D10 |
| `shrinkage` (ε) | regularize nghịch đảo, mặc định 1e-4 |
| `stats_dtype` | `float64` — cố ý, vì cộng dồn Gram lâu dài mà float32 thì mất chính xác |
| `tang_nhanh` / `ngan_hang` | bật kiến trúc BA TẦNG (§6.6) |

Hai phương thức "bằng chứng chạy đúng" đáng học theo:

- `window_report()` — với λ<1, `n_c` của lớp gặp thường xuyên **hội tụ về `1/(1−λ)`**. So số đo với kỳ vọng lý thuyết là cách rẻ nhất bắt lỗi cài đặt.
- `memory_report()` — chi phí byte **thật** theo dtype. `extra_floats` không phản ánh dtype, mà chi phí bị `gram` chi phối và là **O(D²)** chứ không phải O(C·D).

`hap_thu_khong_nhan(loader, device, allowed)` — **pha 2 của bài toán gốc**. Trả về chuỗi accuracy **prequential** (test-then-train): mỗi batch **dự đoán trước** bằng trạng thái hiện có, **rồi mới** cho các tầng không nhãn cập nhật. Đúng nghĩa "phải trả lời ŷ_t ngay, trước khi thấy x_{t+1}".

### 6.5. Titans (G2): `seq_adapter.py` + `memory.py` + `titans_head.py`

Ý tưởng Titans: một module bộ nhớ **vừa đọc vừa tự ghi lúc chạy** (test-time). Trọng số bộ nhớ tự cập nhật theo "độ bất ngờ" của đầu vào, không cần gradient từ loss.

**`seq_adapter.py` (61 dòng) — người phiên dịch.** Titans chỉ ăn chuỗi `(1, L, D)`. Adapter quyết định *"thời gian" nghĩa là gì*:
- `image_seq` (mặc định) — **mỗi ảnh = 1 bước thời gian**. `(B, D)` → `(1, B, D)`. Chuỗi = "từng lần UAV nhìn thấy cảnh".
- `token_seq` — mỗi patch = 1 bước (chuỗi dài 196×, chậm hơn nhiều; để ablation).

**`memory.py` (281 dòng) — bọc `titans_pytorch.NeuralMemory`** sau **một** chữ ký cố định:

```python
out, state = memory(seq, state=None)      # (1,L,D) -> (1,L,D)
```

Vì sao phải bọc: cả dự án chỉ phụ thuộc chữ ký này, thư viện đổi version thì chỉ sửa file này.

File này cũng chứa phần chẩn đoán quan trọng: **hai cổng η (tốc độ ghi) và α (tốc độ quên)**. Nếu chúng bão hoà ở biên thì bộ nhớ chết:
- α → 1: xoá sạch mỗi chunk, memory không tích luỹ gì
- α → 0: không quên gì, norm state **nổ** (đã gặp: 1.75e6)

`_install_gate_bounds()` chặn logit hai cổng không cho trôi ra biên; `_install_eta_alpha_probes()` gắn forward-hook ghi lại trung bình; `eta_alpha_stats()` trả `(η thô, η THẬT, α)`. η THẬT `= sigmoid(logit) × max_lr` mới là con số có ý nghĩa vật lý.

**`titans_head.py` (253 dòng) — ghép lại:**

```
ảnh → [ViT đóng băng] → feature → [SeqAdapter] → chuỗi
    → [TitansMemory + state] → post_norm + residual → [head] → logits
```

Ba **chế độ reset state**, đổi bằng một dòng yaml `memory.reset`:

| Chế độ | Ký ức sống bao lâu | Vai trò |
|---|---|---|
| `image` (A) | mỗi batch reset | sanity check — không có trí nhớ |
| `task` (B) | trong một task | bậc trung gian |
| `never` (C) | xuyên toàn bộ stream | ⭐ đích thật |

**`state_utils.py` (106 dòng)** — state do thư viện trả về là cấu trúc lồng nhau (namedtuple/dict/TensorDict). Các hàm ở đây đi đệ quy áp phép biến đổi lên **từng tensor**, giữ nguyên khung: `detach_state` (cắt gradient nhưng giữ giá trị — truncated BPTT), `clone_state`, `state_to_cpu`, `state_norm`, `count_floats`.

> Bẫy đã gặp và đã vá: `TensorDict` **không phải** dict thường. Bỏ sót nó = state còn dính graph = double-backward error.

**`self_ref_memory.py` (277 dòng)** — phần nghiên cứu sâu (NL.pdf §8.1, Eq 79): cho các projection k/v/q **tự điều chỉnh theo ngữ cảnh** thay vì đóng băng sau pretrain. Hai mức: `ContextGatedProjection` (Task 3, nhân output với cổng tính từ chuỗi) và `SelfModifyingProjection` (Task 4, cộng thêm nhánh sinh từ **tóm tắt trạng thái memory** `M_{t-1}`). Đọc sau cùng.

### 6.6. Kiến trúc BA TẦNG — `tang_nhanh.py` + `ngan_hang_che_do.py` ⭐

Đây là hướng **đang chạy**, ánh xạ thẳng vào ba tần số ở §1.1:

| Tầng | File | Nhớ gì | Cập nhật | Chi phí |
|---|---|---|---|---|
| **chậm** | `slda.py` (μ_c, Σ) | danh tính lớp | 1 lần ở pha 1 có nhãn | O(D²) |
| **NHANH** (M1) | `tang_nhanh.py` | "điều kiện lúc này" | **mỗi batch**, không nhãn | O(D) ~3 KB |
| **TRUNG** (M2) | `ngan_hang_che_do.py` | "các điều kiện đã từng gặp" | 1 lần/chuyến | O(k·D) |

**`tang_nhanh.py` (158 dòng)** — thống kê EMA của feature thô:

```python
def cap_nhat(self, feats)      # hấp thụ 1 batch, KHÔNG đụng nhãn
    # r = λ^B;  m ← r·m + (1−r)·mean(batch)
def chot_moc(self)             # cuối pha 1: đóng băng mốc (m0, v0)
def can_chinh(self, feats)     # đưa feature về hệ toạ độ pha 1
```

Ý tưởng: điều kiện quan sát dịch feature **cùng một hướng** cho mọi lớp → ước lượng được **không cần nhãn** chỉ bằng trung bình và phương sai. Trước khi `chot_moc`, `can_chinh` là identity.

Cố ý **không phải `nn.Module`**: không tham số học được, không đi vào optimizer. State là tensor thường; `export_state`/`import_state` phục vụ checkpoint.

**`ngan_hang_che_do.py` (149 dòng)** — thứ làm mục tiêu **O3** khả thi. Vấn đề nó giải:

> Tầng nhanh bám điều kiện tốt nhưng **không nhớ**. Gặp lại mùa đông sau 6 tháng, nó thích nghi lại từ đầu — mất đúng "thời gian hồi phục" như lần đầu.

Ngân hàng lưu **snapshot trạng thái tầng nhanh** của từng điều kiện đã gặp:

```python
def khop_va_nap(self, tn, m_tuoi=None) -> bool
    # gọi MỘT lần mỗi chuyến, sau `cho_khop_sau` batch
    # gặp lại → NẠP snapshot cũ vào tầng nhanh = "nhận ra ngay, không học lại"
def ghi_lai(self, tn)
    # cuối chuyến: cập nhật chế độ đã khớp, hoặc ghi chế độ MỚI
def _gop_gan_nhat(self, truc)
    # k tràn → gộp HAI chế độ gần nhau nhất (trung bình có trọng số)
```

Tham số nguy hiểm nhất là `nguong` (ngưỡng khớp): nhỏ quá → sợ chế độ, mỗi chuyến tạo một chế độ mới; lớn quá → gộp hết làm một. Cách hiệu chỉnh ghi trong `configs/revisit_U2_nganhang.yaml`.

**Track thí nghiệm** so ba mốc, mỗi mốc thêm đúng một tầng:

| Track | Config | Nội dung |
|---|---|---|
| U0 | `revisit_U0_dongbang.yaml` | SLDA đóng băng hoàn toàn ở pha 2 |
| U1 | `revisit_U1_tangnhanh.yaml` | + tầng nhanh |
| U2 | `revisit_U2_nganhang.yaml` | + ngân hàng chế độ ⭐ |

**Con số công bố = O3(U2) − O3(U1)**, 3 seed, kèm σ. Phép so **một biến** nên tự khử nhiễu kiểu "học thêm dữ liệu thì tự khắc tốt lên".

### 6.7. `cms.py` + `hope.py`

**`cms.py` (137 dòng)** — chia backbone ViT thành các **tầng tần số**. Không đụng forward, chỉ nhóm tham số:

```python
tiers: [[4,1], [4,4], [4,16]]    # [số block, chu kỳ update] — nhanh → chậm
```

12 block ViT chia làm 3 tầng: 4 block cập nhật mỗi bước, 4 block mỗi 4 bước, 4 block mỗi 16 bước. Quyết định của team: attention + LayerNorm → tầng **chậm nhất**; `patch_embed`/`pos_embed`/`cls_token`/norm cuối → **đóng băng** (nền pretrained); head phân loại → tầng **nhanh nhất**.

`tier_report(groups)` in bảng "block nào ở tier nào" — chạy lần đầu **phải đọc**.

**`hope.py` (57 dòng)** — G4. Chỉ 57 dòng vì kế thừa `TitansClassifier`, khác đúng 2 điểm:
1. backbone **không** đóng băng — gradient chảy về, nhịp update do CMSOptimizer kiểm soát
2. `train()` được ghi đè cho phù hợp

```
ảnh → ViT (mở băng, CMS kiểm soát) → SeqAdapter → TitansMemory (xuyên task) → post_norm → head
```

---

<a name="7"></a>
## 7. Giải thích từng file — engine, method, metric, optimizer

### 7.1. `engine.py` (358 dòng) ⭐⭐

**Bộ khung — 15 dòng thật sự:**

```python
def run_continual(model, method, stream, task_loaders, device, train_cfg):
    T = len(stream)
    R = np.zeros((T, T))
    seen = []
    for t, spec in enumerate(stream):
        if getattr(method, "gradient_free", False):
            method.fit_task(model, task_loaders[t]["train"], device)   # NCM, SLDA
        else:
            train_one_task(model, method, task_loaders[t]["train"],
                           device, spec.classes, train_cfg)
        method.end_task(model, task_loaders[t]["train"], device, spec.classes)
        seen += list(spec.classes)
        for j in range(t + 1):
            R[t, j] = evaluate(model, task_loaders[j]["test"], device, sorted(seen))
    return R, log
```

**`train_one_task` — vòng train một task:**

```python
method.begin_task(model, device, allowed)          # LwF chụp teacher tại đây
for ep in range(epochs):
    for x, y in loader:
        logits = mask_logits(model(x), allowed)
        loss = F.cross_entropy(logits, y)
        pen = method.penalty(model)                        # EWC
        if pen is not None:   loss = loss + pen
        extra = method.extra_batch_loss(model, x, logits_full, device)  # Replay/LwF
        if extra is not None: loss = loss + extra
        opt.zero_grad(); loss.backward(); opt.step()
```

Tham số `opt`: mặc định mỗi task tạo optimizer **mới**. Truyền `opt` có sẵn (`train.optimizer_per_task: false`) → **ký ức gradient** (M1/M2/V của M3) và pha chu kỳ CMS sống **xuyên task** — đúng tinh thần "optimizer cũng là bộ nhớ dài hạn" của Nested Learning.

**Sáu nhánh tuỳ chọn** bọc quanh bộ khung, tất cả **mặc định TẮT** (nguyên tắc: bật cờ mới thì run cũ phải bất biến):

| Nhánh | Cờ config | Làm gì |
|---|---|---|
| Checkpoint/resume | `train.checkpoint_path` + `resume` | lưu sau mỗi task, chạy lại nhảy thẳng tới task kế |
| Dừng sớm | `train.stop_after_task` | chạy theo ca |
| Forward transfer | `train.eval_future` | đo `R[t, t+1]` — acc task kế **trước khi** học nó |
| NCM-head shadow | `train.eval_ncm_head` | chấm song song bằng prototype thay vì head Linear |
| SDC | `train.ncm_head_mode: sdc` | dịch prototype lớp cũ theo trôi feature, **không đọc lại data cũ** |
| Pha 2 không nhãn | `train.pha2.khong_nhan` | từ chuyến N trở đi không dùng nhãn |

**SDC (Semantic Drift Compensation)** đáng chú ý về mặt ý tưởng: thay vì dựng lại prototype từ mọi task đã thấy (tốn), nó (1) chụp model **trước** khi train task này, (2) đo `δ = feature_mới − feature_cũ` **chỉ trên data task hiện tại**, (3) dịch prototype lớp cũ theo trung bình δ có trọng số Gaussian theo khoảng cách. Chi phí: một lượt data hiện tại.

**Nhánh pha 2** là mã hoá của bài toán gốc:

```python
if khong_nhan and t >= chuyen_hieu_chinh:
    if hasattr(model, "hap_thu_khong_nhan"):
        log["trace"][t] = model.hap_thu_khong_nhan(loader, device, allowed)
    else:
        pass    # đóng băng hoàn toàn = mốc U0
else:
    method.fit_task(model, loader, device)     # pha 1, CÓ nhãn
```

Trước bản vá này, `fit_task` nhận `(x, y)` **có nhãn ở mọi chuyến** — tức đang giải một bài **dễ hơn** bài thật. Đây là ví dụ điển hình của loại lỗi mà tài liệu ghi chú trong dự án hay gọi là "đo đúng nhưng đo nhầm thứ".

### 7.2. `methods.py` (548 dòng) ⭐⭐

10 chiến lược, tra trong `_METHODS`:

| Tên trong yaml | Lớp | Ý tưởng một câu | Chi phí |
|---|---|---|---|
| `finetune` | `FineTune` | không làm gì — mốc dưới | 0 |
| `ewc` | `EWC` | phạt tham số quan trọng rời giá trị cũ: `L = CE + (λ/2)Σ F_i(θ_i−θ*_i)²` | Fisher, rất đắt |
| `replay` | `Replay` | buffer ảnh cũ, mỗi bước cộng thêm CE "ôn bài" | RAM ảnh thô |
| `lwf` | `LwF` | chụp teacher trước task, ép logits lớp cũ bám theo (KL, temperature T) | 1 bản model |
| `ncm` | `NCM` | gradient-free, prototype | O(C·D) |
| `slda` | `SLDA` | gradient-free, prototype + hiệp phương sai | O(D²) |
| `titans` | `TitansCL` | quản lý vòng đời state Titans | state |
| `cms` | `CMS` | train như finetune, optimizer là CMSOptimizer; log ‖Δw‖ per-tier | 0 |
| `hope` | `HOPE` | Titans + CMS | cả hai |
| `latent_replay` | `LatentReplay` | replay bằng **feature 384-d** thay vì ảnh — rẻ hơn ~400 lần | O(N·D) |

**Cách đọc:** `FineTune` cài cả 5 hook rỗng. Mỗi method con chỉ ghi đè hook mình cần. Ví dụ `LwF`:

```python
def begin_task(...):     # chụp teacher = deepcopy model, đóng băng
def extra_batch_loss(...): # KL(student/T || teacher/T) trên cột lớp cũ, nhân λT²
```

Không đụng `penalty` và `end_task` → đọc xong 2 hàm là hiểu hết LwF.

`LatentReplay` là ví dụ đẹp của tư duy chi phí: backbone đóng băng → feature cũ không bao giờ "ôi" → chỉ cần lưu vector 384-d float16 (~0,75 KB/mẫu) thay vì ảnh 224² (~300 KB). **Rẻ hơn 400 lần.** Điều kiện: model phải có `forward_from_feats` (method kiểm tra và ném `TypeError` rõ ràng nếu thiếu).

Đầu file còn có ba hằng số chẩn đoán cổng Titans: `ALPHA_SAT_HI/LO` và `GATE_SAT_FRAC` — dùng cảnh báo khi cổng η/α bão hoà.

### 7.3. `metrics/` — chấm điểm

**`continual.py` (83 dòng)** — thuần numpy, 5 chỉ số đọc từ ma trận R:

| Hàm | Công thức | Nghĩa |
|---|---|---|
| `average_accuracy(R)` | `mean(R[-1, :])` | hàng cuối — trung bình sau khi học hết |
| `average_forgetting(R)` | `mean_{j<T-1}( max_{j≤i≤T-2} R[i,j] − R[-1,j] )` | ⭐ tụt bao nhiêu so với **đỉnh từng đạt** (trước giai đoạn cuối). **Thấp = tốt — đây là số chính của dự án** |
| `backward_transfer(R)` | `mean_{j<T-1}(R[-1,j] − R[j,j])` | học cái mới làm cái cũ tốt lên (dương) hay tệ đi (âm) |
| `forward_transfer(R)` | `mean_{j≥1}(R[j-1,j] − chance)` | biết gì về task chưa học. Cần `eval_future: true` |
| `average_anytime_accuracy(R)` | `mean_t(mean_{j≤t} R[t,j])` | ⭐ trung bình tại **mọi mốc**, không chỉ mốc cuối |

**Vì sao có AAA:** `average_accuracy` chỉ nhìn hàng cuối. Nhưng UAV dùng model **liên tục**, không đợi hết stream — chất lượng ở giữa đường cũng tính.

**`revisit.py` (116 dòng)** — ⭐ thước đo cho bài toán thật. Đọc docstring:

> `AAA`/`Average Accuracy`/`Forgetting` **không đo được O3**. Chúng chỉ nói mô hình đúng bao nhiêu, không nói nó có **ghi nhớ điều kiện** hay đang thích nghi lại từ đầu mỗi lần.

| Hàm | Mục tiêu | Đo cái gì |
|---|---|---|
| `acc_hien_tai(R)` | **O1** | trung bình **đường chéo** — đúng bao nhiêu trên chính điều kiện **đang bay** |
| `thoi_gian_hoi_phuc(acc, nguong)` | **O2** | bao nhiêu **bước** để hồi về 95% mức ổn định sau khi điều kiện đổi |
| `loi_ich_quay_lai(R, mode_that)` | ⭐ **O3** | `Acc(gặp lại chế độ X) − Acc(lần ĐẦU gặp X)` |

O3 là **thước đo trung tâm của dự án hiện tại**. Nếu tầng ghi nhớ điều kiện hoạt động, chuyến gặp **lại** một chế độ đã biết phải tốt hơn hẳn lần đầu.

**Vì sao phải viết thước đo mới:** bài học trực tiếp từ D10 — λ hoạt động tốt (+3,91 điểm ở mức trôi 100%, 3,10σ) nhưng thước đo chuẩn **che mất**, vì `AAA` trung bình trên mọi task đã thấy: sau task 8, mô hình bị chấm trên 9 bộ test mà **8 bộ là điều kiện quá khứ**. Thước đo đó thưởng cho việc nhớ lịch sử. *Drone thì bay bây giờ.*

> Đây là bài học phương pháp luận đáng giá nhất trong dự án: **thước đo sai thì kết luận sai, dù thí nghiệm chạy đúng.**

**`openset.py` (77 dòng)** — khi UAV gặp lớp **chưa học**, phải biết nói "không biết". `roc_points` / `auc` / `eer` / `tar_at_far`. Quy ước: `genuine` = điểm tin cậy trên mẫu lớp đã học (mong cao), `impostor` = mẫu lớp chưa học (mong thấp).

### 7.4. `optim/` — optimizer

**`m3.py` (232 dòng)** — Multi-scale Momentum Muon, Algorithm 1 §7.2 của NL.pdf. Ý tưởng: **optimizer cũng là bộ nhớ** — momentum là ký ức của gradient. M3 có **hai tầng ký ức gradient** (nhanh và chậm) + trực giao hoá kiểu Muon:

```
g_t = gradient
M1 = update(M1, g_t)        # ký ức NHANH
V  = update(V, g_t²)        # moment bậc 2 kiểu Adam
O1 = NewtonSchulz(M1)       # trực giao hoá — giữ "hướng", bỏ "độ lớn lệch trục"
M2 = update(M2, O1)         # ký ức CHẬM
```

`newton_schulz(m, steps=5)` — xấp xỉ trực giao hoá bằng iteration, không cần SVD.

**`cms_optimizer.py` (178 dòng)** — CMS ở mức **optimizer**, không đụng forward của ViT:

- tham số chia thành **tier**, mỗi tier có `(chu kỳ p, hệ số học η)`
- gradient của **mọi** tier tích luỹ sau mỗi batch
- tier chỉ được cập nhật khi `global_step % p == 0`, với gradient **gộp** của p bước vừa qua

Bọc quanh optimizer bất kỳ → CMS chạy được trên cả AdamW lẫn M3.

`_apply_adaptive_eta`: `η_tier = η_base × (1 − cos(grad hiện tại, hướng update trước))` — gradient đổi hướng nhiều thì học mạnh hơn.

### 7.5. `checkpoint.py` (64 dòng)

Vì sao cần: một run RESISC45 9 task mất 2–6h trên VM CPU; VM bị preempt / SSH đứt / OOM giữa task 7 là **mất trắng**.

`save_run_checkpoint` lưu sau **mỗi task**, **atomic** (ghi file tạm rồi rename). Chụp: `model.state_dict()`, state Titans, **object method** (buffer/anchor/teacher — pickle nguyên), R, log, optimizer state, RNG state của torch và Python.

Chụp cả RNG là chi tiết quan trọng: nhờ vậy chạy lại với `--resume` cho số **giống hệt** chạy liền. Test: `tests/test_checkpoint.py`.

---

<a name="8"></a>
## 8. `scripts/` — chạy cái gì bằng lệnh nào

### 8.1. Chạy thí nghiệm

```bash
# một thí nghiệm từ một config
python scripts/run_g1.py --config configs/g1_smoke.yaml
python scripts/run_g1.py --config configs/g1_resisc45.yaml --method ewc --set train.lr=5e-5

# toàn bộ chuỗi G1→G4 + sinh báo cáo docx
python scripts/run_all.py --quick          # chỉ EuroSAT, ~2-4h
python scripts/run_all.py --shutdown       # tự tắt VM GCP khi xong
```

`run_g1.py` là **file dài nhất trong scripts** (420 dòng) nhưng đọc dễ vì tuyến tính: parse args → load config → chọn stream → dựng loader → dựng model → dựng method → `run_continual` → tính metric → ghi file.

Điểm đáng chú ý: `run_dir_name(cfg, method)` sinh tên thư mục kết quả từ config. Nó cố ý thêm hậu tố cho từng biến thể (`_r{reset}`, `_optkeep`, `_{order}_p{periods}`) để **hai cấu hình khác nhau không đè kết quả của nhau** — bẫy đã gặp thật.

### 8.2. Kiểm tra trước khi đốt giờ máy

```bash
python scripts/check_env.py           # môi trường đủ chưa
python scripts/smoke_backbone.py      # ảnh → feature chạy chưa
python scripts/smoke_titans.py        # Titans forward+backward chạy chưa
python scripts/calibrate_drift.py     # ⭐ cường độ trôi có nằm vùng đo được không
python scripts/do_truc_dieu_kien.py   # ⭐ CỬA CHẶN: "điều kiện" có phải trục chung không
python scripts/mo_phong_ba_tang.py    # mô phỏng numpy đầu-cuối, không cần GPU
```

Hai script có ⭐ thể hiện văn hoá làm việc của dự án: **kiểm setup 15 phút, tránh mất 8 tiếng.**

`do_truc_dieu_kien.py` trả lời câu hỏi quyết định cả kiến trúc: *"khi điều kiện đổi, feature của 45 lớp có dịch cùng một hướng không?"* Nếu **có** → điều kiện là một trục chung, ước lượng được **không cần nhãn** chỉ bằng vài số → toàn bộ tầng nhanh hợp lý. Nếu **không** → phải thiết kế lại. Chạy 15 phút, trước khi viết dòng code nào.

### 8.3. Đọc kết quả

```bash
python scripts/compare_g1.py                # gộp artifacts/results/*/metrics.json
python scripts/compare_all.py               # quét mọi artifacts_*/ + log
python scripts/tom_tat_ba_tang.py --cua-chan  # U0/U1/U2 + cửa chặn (exit 3 nếu U1 không hơn U0)
python scripts/trace_gates.py               # trích quỹ đạo η/α từ log đã có
python scripts/make_report.py               # → artifacts/BAO_CAO_KET_QUA.docx
```

### 8.4. Đo chi phí

```bash
python scripts/bench_cost.py      # bộ nhớ (byte thật) · độ trễ · thông lượng: SLDA vs Titans
python scripts/bench_edge.py      # chi phí optimizer + head cho UAV on-board
python scripts/bench_ba_tang.py   # MB thêm + ms/khung hình của ba tầng
```

Ràng buộc bài toán: tổng bộ nhớ thêm **≤ 10 MB**, độ trễ thêm **≤ 5%** ngân sách khung hình 30 fps (= 1,67 ms). Ba script này kiểm ràng buộc đó.

---

<a name="9"></a>
## 9. Đọc kết quả · debug · mở rộng

### 9.1. Mỗi run sinh ra gì

```
artifacts/results/<dataset>_<method>_seed<N>[_hậu_tố]/
├── acc_matrix.csv          ma trận R
├── metrics.json            các chỉ số + chi phí + runtime
├── config.yaml             config ĐÃ DÙNG (tái lập được)
├── metrics_openset.json    (nếu bật #27)
└── resume_checkpoint.pt    (nếu bật checkpoint)
```

**Cách đọc `acc_matrix.csv`:**
- **đường chéo** `R[t,t]` — học task t xong thì làm task t tốt cỡ nào (khả năng học)
- **cột j đọc từ trên xuống** — task j tụt dần thế nào khi học thêm task khác (**quên**)
- **hàng cuối** — trạng thái cuối cùng

### 9.2. Bẫy hay gặp

| Triệu chứng | Nguyên nhân thường gặp |
|---|---|
| `KeyError: Unknown dataset/method` | sai tên trong yaml — xem `_REGISTRY` / `_METHODS` |
| accuracy ~ 1/C (đoán mò) | quên `mask_logits`, hoặc lr sai thang |
| kết quả đè lên nhau | `run_dir_name` chưa phân biệt biến thể → thêm hậu tố |
| norm state Titans **nổ** | cổng α → 0. Xem `eta_alpha_stats()`, bật `gate_bound` |
| accuracy Titans phẳng lì | cổng α → 1 (xoá sạch mỗi chunk) |
| `float32` mất chính xác | SLDA phải dùng `stats_dtype: float64` |
| `double-backward error` | quên `detach_state` — TensorDict còn dính graph |
| số hai máy lệch nhau | chưa `seed_everything`, hoặc `shuffle_classes` khác seed |
| revisit chạy nhưng O3 vô nghĩa | quên `drift.enabled: true` — nay đã có cửa chặn ném lỗi |

### 9.3. Thêm cái mới — chỗ cần sửa

| Muốn thêm | Sửa ở đâu |
|---|---|
| dataset mới | `data/sources.py`: viết `_load_xxx` + một dòng vào `_REGISTRY` |
| chiến lược chống quên mới | `methods.py`: kế thừa `FineTune`, ghi đè hook cần + một dòng vào `_METHODS` |
| kiến trúc model mới | `models/`: chỉ cần `forward(x) -> (B, C)`; thêm nhánh chọn trong `run_g1.py` |
| chỉ số mới | `metrics/continual.py` + export trong `metrics/__init__.py` |
| kiểu stream mới | `data/stream.py`: hàm `build_*_stream` + nhánh `stream_type` trong `run_g1.py` |
| kiểu trôi mới | `data/drift.py`: thêm phép biến đổi vào `ApDungTroi.__call__` |

**Nguyên tắc bất di bất dịch của dự án:** tính năng mới phải **mặc định TẮT**, để mọi run cũ đi đúng đường cũ, kết quả **bất biến**. Bạn sẽ thấy câu này lặp lại trong hầu hết docstring — đó là lý do so sánh giữa các run cách nhau vài tuần vẫn tin được.

### 9.4. Chạy test

```bash
pytest -q                              # toàn bộ (test tự skip nếu thiếu torch/timm/titans)
pytest tests/test_stream.py tests/test_metrics.py -q      # không cần torch
pytest tests/test_ba_tang.py -q        # ba tầng (phần mới nhất)
pytest tests/test_revisit.py -q        # lịch bay + O1/O3
```

29 file test. Nhiều test **tự skip** khi thiếu thư viện nặng — chủ ý, để CI và máy dev yếu vẫn chạy được phần logic thuần Python.

### 9.5. Tài liệu nên đọc kèm

Trong repo `uav-continual-learning/`:

| File | Nội dung |
|---|---|
| `README.md` | cài đặt + lệnh chạy từng giai đoạn |
| `HUONG_DAN_CHAY.md` | quy trình VM GCP: tạo máy → chạy → lấy kết quả về |
| `ROADMAP_NESTED_LEARNING.md` | lộ trình theo paper |
| `docs/KET_LUAN_G1.md` | ⭐ bảng số mốc + 6 kết luận |
| `docs/TOM_TAT_PAPER.md` | tóm tắt NL.pdf |
| `docs/LOGIC_NESTED_LEARNING.md` | logic ý tưởng Nested Learning |

Ở thư mục gốc (ngoài repo) — ghi chú nghiên cứu theo ngày:

| File | Nội dung |
|---|---|
| `BAI_TOAN_VA_MUC_TIEU_2026-08-04.md` | ⭐ **đọc trước tiên** — phát biểu bài toán hiện hành |
| `KE_HOACH_BA_TANG_2026-08-04.md` | thiết kế ba tầng |
| `KET_QUA_D10_D11_2026-08-04.md` | kết quả λ + Titans trên stream trôi |
| `CHAY_VM_BA_TANG_2026-08-04.md` | quy trình chạy track hiện tại |
| `LICH_SU_NHANH_VA_KET_QUA.md` | lịch sử các nhánh git + kết quả |

---

## Phụ lục A — Bảng tra file nhanh

| File | Dòng | Một câu |
|---|---:|---|
| `engine.py` | 358 | ⭐ vòng lặp học liên tục — mọi thí nghiệm đi qua đây |
| `methods.py` | 548 | ⭐ 10 chiến lược chống quên (hook `begin/penalty/extra/end/footprint`) |
| `models/slda.py` | 408 | model chủ lực: prototype + hiệp phương sai, gradient-free |
| `models/self_ref_memory.py` | 277 | projection tự điều chỉnh theo ngữ cảnh (nghiên cứu sâu) |
| `models/memory.py` | 281 | bọc `titans_pytorch.NeuralMemory` + chẩn đoán cổng η/α |
| `models/titans_head.py` | 253 | ghép backbone + adapter + memory + head |
| `data/stream.py` | 246 | chia task (4 kiểu stream), thuần Python |
| `optim/m3.py` | 232 | optimizer hai tầng ký ức gradient |
| `data/loaders.py` | 230 | DataLoader mỗi task (+ dòng thời gian pha 2) |
| `data/sources.py` | 194 | 3 dataset → cùng một `DataSource` |
| `optim/cms_optimizer.py` | 178 | CMS ở mức optimizer (chu kỳ + gộp gradient) |
| `models/tang_nhanh.py` | 158 | M1 — tầng nhanh, EMA feature, không nhãn |
| `models/ngan_hang_che_do.py` | 149 | M2 — tầng trung, nhớ các điều kiện đã gặp |
| `data/drift.py` | 139 | 5 phép mô phỏng nắng/sương/bụi |
| `models/cms.py` | 137 | chia ViT thành tầng tần số |
| `data/revisit.py` | 134 | lịch bay tuần hoàn — điều kiện quay lại |
| `metrics/revisit.py` | 116 | ⭐ O1/O2/O3 |
| `models/state_utils.py` | 106 | thao tác đệ quy trên state Titans |
| `metrics/continual.py` | 83 | 5 chỉ số từ ma trận R |
| `models/classifier.py` | 80 | head + `mask_logits` + CosineHead |
| `metrics/openset.py` | 77 | ROC/AUC/EER cho "biết nói không biết" |
| `openset_eval.py` | 75 | chấm open-set bằng khoảng cách prototype |
| `checkpoint.py` | 64 | lưu/khôi phục atomic sau mỗi task |
| `models/backbone.py` | 63 | ViT-S pretrained hoặc TinyCNN |
| `models/ncm.py` | 63 | prototype, 63 dòng mà đứng nhì bảng G1 |
| `utils/config.py` | 63 | yaml + `--set a.b=c` |
| `models/seq_adapter.py` | 61 | ảnh → chuỗi cho Titans |
| `models/hope.py` | 57 | G4 = Titans + CMS |
| `utils/seed.py` | 23 | cố định mọi nguồn ngẫu nhiên |

## Phụ lục B — Bản đồ config

| Khối yaml | Đọc bởi | Núm chính |
|---|---|---|
| `seed`, `device` | `run_g1.py` | |
| `data` | `sources.py`, `stream.py`, `loaders.py` | `name`, `num_tasks`, `stream_type`, `batch_size` |
| `data.drift` | `drift.py`, `revisit.py` | `enabled`, `che_do`, `chu_ky`, `severity`, `mode` |
| `data.pha2` | `run_g1.py` → `engine.py` | `khong_nhan`, `chuyen_hieu_chinh`, `test_chung` |
| `backbone` | `backbone.py` | `name`, `pretrained`, `freeze` |
| `head` | `classifier.py` | `linear` \| `cosine` |
| `memory` | `titans_head.py`, `memory.py` | `enabled`, `reset` (image/task/never), `chunk_size`, `seq` |
| `cms` | `cms.py`, `cms_optimizer.py` | `enabled`, `tiers`, `order` |
| `train` | `engine.py` | `method`, `epochs_per_task`, `lr`, `optimizer`, `eval_future`, `eval_ncm_head` |
| `slda` | `slda.py` | `cov_mode`, `decay_mean/cov` (λ), `shrinkage`, `tang_nhanh`, `ngan_hang` |
| `ewc`/`replay`/`lwf` | `methods.py` | tham số riêng mỗi method (`build_method` truyền `cfg[tên_method]`) |
| `openset` | `stream.py`, `openset_eval.py` | `enabled`, `holdout` |
| `log` | `run_g1.py` | `dir` |

---

*Hết. Nếu chỗ nào trong tài liệu này lệch với code, tin code — và sửa tài liệu.*
