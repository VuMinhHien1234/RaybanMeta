# Báo cáo Tổng quan Dự án: UAV Học Liên Tục (Nested Learning / Titans / HOPE)

Tài liệu này cung cấp một cái nhìn toàn diện về cấu trúc, thiết kế, phân công công việc và trạng thái hiện tại của dự án **UAV Continual Learning** (Học liên tục trên Thiết bị bay không người lái), dựa trên việc nghiên cứu mã nguồn và các tài liệu kế hoạch có trong repository.

---

## 1. Tổng quan Dự án & Mục tiêu Nghiên cứu

Dự án nhằm ứng dụng các nghiên cứu mới nhất về **Nested Learning (Học lồng nhau)**, **Titans (Kiến trúc bộ nhớ nơ-ron tuần tự phục hồi)**, và **HOPE** (hợp nhất Titans và CMS - Continuum Memory System) vào bài toán **phân loại ảnh trên không theo dạng học liên tục tăng dần lớp (Class-Incremental Aerial Scene Recognition)**.

### Thách thức cốt lõi
*   **Catastrophic Forgetting (Quên thảm họa):** Khi huấn luyện mô hình trên các tác vụ mới (lớp ảnh mới), mô hình có xu hướng mất đi khả năng nhận diện các lớp ảnh đã học trước đó.
*   **Ràng buộc tài nguyên của UAV:** Các thuật toán học liên tục phải cân bằng giữa độ chính xác, khả năng chống quên, và chi phí tính toán/bộ nhớ (ví dụ: việc lưu trữ quá nhiều ảnh cũ ở Replay Buffer hoặc nhân đôi tham số ở EWC là không tối ưu cho thiết bị nhúng).

### Các mô hình & phương pháp đề xuất
Mô hình sẽ tiến hóa qua các giai đoạn từ đơn giản đến phức tạp:
1.  **Baselines (G1):** Đánh giá các phương pháp học liên tục kinh điển (Fine-tune tuần tự, EWC, Replay, LwF, NCM).
2.  **Titans Memory (G2):** Tích hợp bộ nhớ Titans stateful xuyên suốt luồng dữ liệu (cross-task) giúp thích nghi nhanh với ngữ cảnh mới và tăng khả năng chuyển giao tri thức về sau (Forward Transfer - FWT).
3.  **CMS Retrofit (G3):** Cải tiến kiến trúc Vision Transformer (ViT) thành mạng đa tần số (Multi-Frequency) bằng cách tối ưu hóa chu kỳ cập nhật trọng số MLP (tần số nhanh/chậm).
4.  **HOPE (G4):** Kết hợp cả hai thế mạnh: Titans (thích nghi nhanh, tần số cực cao) và CMS (bền vững chống quên, tần số thấp) đi kèm optimizer **M3 / Delta-Momentum**.

---

## 2. Phát hiện Lỗi Nghiêm Trọng (Critical Issue) cần xử lý ngay 🚨

Trong quá trình phân tích mã nguồn, tôi đã phát hiện một lỗi cấu hình nghiêm trọng trong file `.gitignore` dẫn đến **mất toàn bộ mã nguồn xử lý dữ liệu**.

### Nguyên nhân
Tại file [uav-continual-learning/.gitignore](file:///Users/an/Documents/Do%20An/RaybanMeta/uav-continual-learning/.gitignore) dòng số 11, cấu hình ghi:
```gitignore
data/
```
Quy tắc này nhằm mục đích bỏ qua thư mục chứa dữ liệu thô tải về (`uav-continual-learning/data/`). Tuy nhiên, trong Git, viết `data/` không có đường dẫn tuyệt đối bắt đầu bằng dấu `/` sẽ **bỏ qua đệ quy bất kỳ thư mục nào tên là `data`** ở bất kỳ cấp nào trong dự án.
Hệ quả là thư mục mã nguồn quan trọng **`uav-continual-learning/src/uavcl/data/`** (chứa loader, stream split và datasets) **bị Git bỏ qua hoàn toàn và không được đẩy lên GitHub**.

### Hậu quả hiện tại
*   Khi chạy lệnh `pytest`, hệ thống báo lỗi: `ModuleNotFoundError: No module named 'uavcl.data'`.
*   Các module `run_g1.py` và `test_stream.py` bị lỗi import ngay lập tức vì thiếu:
    *   `src/uavcl/data/sources.py`
    *   `src/uavcl/data/stream.py`
    *   `src/uavcl/data/loaders.py`

### Giải pháp khắc phục
1.  Cập nhật file `.gitignore` thay thế `data/` bằng đường dẫn cụ thể từ gốc dự án, ví dụ:
    ```gitignore
    /data
    /uav-continual-learning/data
    ```
2.  Yêu cầu thành viên **N1** (người viết code phần Data) kiểm tra lại thư mục cục bộ của họ, thực hiện git add và commit thư mục `src/uavcl/data/` lên repository chung.

---

## 3. Cấu trúc Thư mục và Vai trò của từng File

Dưới đây là sơ đồ chi tiết các thành phần trong dự án và vai trò tương ứng:

```
RaybanMeta/
├── CAU_HOI_THEO_GIAI_DOAN.md       # Tổng hợp câu hỏi thiết kế cần thống nhất trước mỗi giai đoạn
├── Team_Plan_3nguoi.md              # Phân công vai trò chi tiết cho N1, N2, N3 từ G0 đến G6
├── plans/                           # Chứa kế hoạch và checklist chi tiết từng giai đoạn
│   ├── 00_LO_TRINH_TONG.md          # Lộ trình tổng thể (~22 tuần) và các quyết định đã chốt
│   ├── PLAN_G2_TITANS.md            # Kế hoạch chi tiết tích hợp Titans Memory
│   ├── PLAN_G3_CMS.md               # Kế hoạch chi tiết tích hợp CMS (Đa tần số)
│   ├── PLAN_G4_HOPE_M3.md           # Kế hoạch chi tiết ghép HOPE và Optimizer M3
│   ├── PLAN_G5_DANH_GIA.md          # Kế hoạch benchmark 3 seeds + dataset thứ 3
│   ├── PLAN_G6_BAOCAO_PAPER.md      # Kế hoạch viết báo cáo và dự thảo bài báo khoa học
│   └── TASKS_G2_TITANS.md           # Bảng phân chia đầu việc (WBS) cho giai đoạn G2
│
└── uav-continual-learning/          # Thư mục chứa mã nguồn Python chính
    ├── environment.yml / requirements.txt # Định nghĩa môi trường conda / pip
    ├── pyproject.toml               # Cấu hình cài đặt package local `uavcl`
    ├── configs/                     # Chứa các file cấu hình thí nghiệm (.yaml)
    │   ├── default.yaml             # Cấu hình mặc định dùng chung làm "hợp đồng" giữa các thành viên
    │   ├── g1_smoke.yaml            # Config chạy thử nhanh trên CPU bằng dữ liệu giả
    │   ├── g1_eurosat.yaml          # Config chạy thử nghiệm EuroSAT (10 classes, 5 tasks)
    │   └── g1_resisc45.yaml         # Config benchmark chính RESISC45 (45 classes, 9 tasks)
    │
    ├── src/uavcl/                   # Source code của thư viện uavcl
    │   ├── __init__.py
    │   ├── engine.py                # Vòng lặp học liên tục chính (train_one_task, evaluate, run_continual)
    │   ├── methods.py               # Cài đặt thuật toán CL baselines (EWC, Replay, LwF, NCM)
    │   ├── data/                    # [BỊ THIẾU TRÊN GIT] Chứa luồng xử lý dữ liệu và DataLoader
    │   ├── metrics/                 # Bộ thư viện đo đạc kết quả học liên tục
    │   │   ├── __init__.py
    │   │   ├── continual.py         # Đo Avg Accuracy, Forgetting, BWT, FWT
    │   │   └── openset.py           # Đo các chỉ số nhận diện lớp lạ (AUC, EER, TAR@FAR)
    │   │
    │   ├── models/                  # Định nghĩa các kiến trúc mạng
    │   │   ├── __init__.py
    │   │   ├── backbone.py          # Chứa backbone (timm ViT/ResNet hoặc TinyCNN cho smoke test)
    │   │   ├── classifier.py        # Module Classifier kèm mặt nạ logits (masking) cho Incremental Learning
    │   │   └── ncm.py               # Classifier dựa trên Nearest Class Mean (NCM) không gradient
    │   │
    │   └── utils/
    │       ├── __init__.py          # Export helper get_device()
    │       ├── config.py            # Đọc/ghi cấu hình YAML và override parameters bằng CLI
    │       └── seed.py              # Đảm bảo tính tái lập (seed_everything)
    │
    ├── scripts/                     # Các kịch bản thực thi thí nghiệm
    │   ├── check_env.py             # Script kiểm tra tính sẵn sàng của môi trường
    │   ├── smoke_titans.py          # Test nhanh module bộ nhớ Titans (forward & backward)
    │   ├── smoke_backbone.py        # Test nhanh module vision backbone trích xuất đặc trưng
    │   ├── run_g1.py                # Script chạy chính cho 1 thí nghiệm
    │   └── compare_g1.py            # Script tổng hợp kết quả của các run thành bảng markdown
    │
    └── tests/                       # Thư mục chứa mã kiểm thử tự động (Pytest)
        ├── test_fwt.py              # Kiểm thử tính toán Forward Transfer
        ├── test_g1_smoke.py         # Kiểm thử end-to-end trên luồng dữ liệu giả lập
        ├── test_methods_g1.py       # Kiểm thử các hàm cập nhật và phạt (penalty) của baselines
        ├── test_metrics.py          # Kiểm thử tính toán độ quên/chuyển giao bằng numpy
        ├── test_ncm.py              # Kiểm thử cơ chế cập nhật prototype của NCM
        ├── test_openset.py          # Kiểm thử khả năng phân biệt lớp lạ (open-set)
        └── test_stream.py           # Kiểm thử logic phân chia task (đang lỗi do thiếu uavcl.data)
```

---

## 4. Kế hoạch & Phân công Vai trò (Team 3 người)

Dự án được thực hiện bởi 3 thành viên với các mảng chuyên môn chuyên biệt, phối hợp nhịp nhàng thông qua các giao diện lập trình (API) đã chốt trước.

### Phân vai thành viên
1.  **N1 (Data & Khung học liên tục):**
    *   *Nhiệm vụ:* Chuẩn bị dữ liệu, phân chia luồng task stream, xây dựng DataLoader, cài đặt hệ thống metrics (Accuracy, Forgetting, Open-set), thực hiện chạy ma trận thí nghiệm, vẽ biểu đồ và quản lý log (W&B / JSON).
    *   *Sản phẩm chính:* Pipeline dữ liệu ổn định, bảng kết quả tổng hợp.
2.  **N2 (Kiến trúc bộ nhớ - Lõi ML):**
    *   *Nhiệm vụ:* Người nắm phần giải thuật lõi khó nhất. Thiết kế và cài đặt module bộ nhớ Titans, phân tầng đa tần số CMS, kết hợp HOPE, và cài đặt thuật toán tối ưu hóa cải tiến **M3 / Delta Momentum**.
    *   *Sản phẩm chính:* Các module bộ nhớ nơ-ron tuần tự, optimizer cải tiến.
3.  **N3 (Backbone thị giác & Tích hợp):**
    *   *Nhiệm vụ:* Cài đặt vision backbone pretrained (timm ViT/DINOv2), viết bộ adapter chuyển đổi đặc trưng không gian (2D) sang chuỗi thời gian cho Titans, tích hợp CMS vào kiến trúc ViT thực tế, tối ưu hóa bộ nhớ GPU (AMP, chunk-wise, checkpointing) và kiểm thử độ bền bỉ (Robustness).
    *   *Sản phẩm chính:* Mô hình ViT-CMS tích hợp bộ nhớ tối ưu hóa tốc độ.

### Lộ trình 7 Giai đoạn (Tổng thời gian ~22 tuần)

```
G0: Nền tảng (1-2t) ➔ G1: Baselines (1-2t) ➔ G2: Titans (2-4t) ➔ G3: CMS (2-4t) ➔ G4: HOPE (2-4t) ➔ G5: Đánh giá (3t) ➔ G6: Báo cáo (3t)
```

*   **G0 — Nền tảng (Đã xong):** Thiết lập môi trường chung, chạy thử nghiệm Titans toy và trích xuất đặc trưng của backbone.
*   **G1 — Baselines & Khung đo (Code xong - Chờ lấy số):** Xây dựng xong 5 thuật toán nền tảng. Hiện đang cần chạy thí nghiệm để lấy bảng số mốc (baseline_table.md) trên EuroSAT và RESISC45.
*   **G2 — Tích hợp bộ nhớ Titans (4 tuần):**
    *   Chốt API giữa Adapter (N3) và TitansMemory (N2).
    *   Hiện thực hóa chuỗi xuyên task (Cross-task stream): trạng thái bộ nhớ Titans được truyền tiếp từ task này sang task khác mà không reset, giúp mô hình tích lũy tri thức động.
    *   Đạt được 3 mức độ reset bộ nhớ: `reset: image` (Sanity check) ➔ `reset: task` (BPTT cắt ngắn) ➔ `reset: never` (Mục tiêu tối thượng).
*   **G3 — Retrofit CMS vào Backbone (4 tuần):**
    *   Can thiệp vào kiến trúc MLP của ViT để biến chúng thành bộ nhớ đa tần số được điều khiển qua chu kỳ cập nhật trọng số của Optimizer Wrapper (`CMSOptimizer`).
    *   Thực hiện các thí nghiệm loại trừ (ablation) để tìm ra block nào của ViT nên cập nhật nhanh hay chậm.
*   **G4 — Ghép thành HOPE (5 tuần):**
    *   Tích hợp Titans (tầng phản xạ nhanh) đứng trên stack CMS (tầng lưu trữ chậm) tạo nên kiến trúc HOPE hoàn chỉnh.
    *   Cài đặt Optimizer M3 / Delta-Momentum độc quyền của bài báo.
*   **G5 — Đánh giá chặt chẽ (3 tuần):**
    *   Chạy toàn bộ 54 thí nghiệm (3 dataset × 6 model × 3 seeds) để lấy thống kê $mean \pm std$.
    *   Bổ sung bộ dữ liệu thứ 3 (**UCM** hoặc **AID**) nhằm chứng minh tính tổng quát hóa mà không tối ưu hyperparameters trên đó.
    *   Đánh giá độ bền bỉ dưới tác động của nhiễu ảnh thời tiết (blur/fog/noise) và sự xáo trộn thứ tự task.
*   **G6 — Tổng hợp & Báo cáo (3 tuần):**
    *   Hoàn thiện báo cáo đồ án tiếng Việt chi tiết.
    *   Dự thảo bài báo tiếng Anh (Draft paper) hướng tới các hội thảo về CL/Remote-Sensing.
    *   Đóng gói repo với kịch bản chạy lại bằng 1 dòng lệnh (one-click reproduction).

---

## 5. Chi tiết Thiết kế Kỹ thuật trong Codebase hiện tại

### Cơ chế Class-Incremental Learning (`uavcl/engine.py` & `classifier.py`)
Mô hình sử dụng chung một mạng phân loại tuyến tính có số cột bằng tổng số lớp của toàn bộ dataset (ví dụ: 45 cột cho RESISC45).
*   **Khi huấn luyện Task $t$:** Logits đầu ra từ mô hình sẽ đi qua hàm `mask_logits`, đè giá trị $-1.0e4$ lên tất cả các cột không thuộc nhóm lớp của task $t$. Điều này giúp cô lập quá trình tối ưu hóa chỉ tập trung vào các lớp hiện tại.
*   **Khi đánh giá sau Task $t$:** Logits sẽ được mask để chỉ mở các cột thuộc những lớp **đã từng xuất hiện** từ Task $0$ đến Task $t$. Điều này đảm bảo đánh giá chính xác khả năng chống quên trên toàn bộ lịch sử đã qua.

### Bộ chỉ số đánh giá (`uavcl/metrics/continual.py`)
Mã nguồn tính toán dựa trên ma trận độ chính xác $R$ kích thước $T \times T$ (với $T$ là số task):
*   **Average Accuracy ($A_{avg}$):** Điểm trung bình hàng cuối cùng của ma trận $R$, thể hiện độ chính xác của mô hình trên mọi tác vụ sau khi kết thúc stream.
*   **Backward Transfer (BWT) / Average Forgetting ($F_{avg}$):** Đo lường mức độ suy giảm hiệu năng của các tác vụ cũ sau khi mô hình học thêm tác vụ mới. Dự án tập trung tối ưu hóa để kéo $F_{avg}$ về sát $0$.
*   **Forward Transfer (FWT):** Đo lường khả năng mô hình tận dụng bộ nhớ tích lũy để nhận diện trước các tác vụ tương lai chưa được học (kích hoạt qua config `train.eval_future=true`).

---

## 6. Các bước hành động khuyến nghị tiếp theo (Next Actions)

Để tiếp tục đẩy nhanh tiến độ dự án mà không bị tắc nghẽn, nhóm nên thực hiện các công việc sau:

1.  **Sửa lỗi `.gitignore`:** Cập nhật quy tắc loại trừ như hướng dẫn ở Mục 2 để có thể đẩy thư mục `src/uavcl/data/` lên Git.
2.  **Đóng cổng G1:**
    *   Sau khi có mã nguồn dữ liệu, chạy script `python scripts/run_g1.py` trên môi trường GPU của đội ngũ đối với 5 baselines trên cả 2 tập dữ liệu EuroSAT và RESISC45.
    *   Dùng `python scripts/compare_g1.py` để xuất file `baseline_table.md`.
    *   Cả team thảo luận để viết tài liệu kết luận baseline `docs/KET_LUAN_G1.md`.
3.  **Họp Interface G2:** Chốt định dạng của sequence đầu vào cho module Titans để N2 và N3 có thể bắt đầu lập trình song song module `memory.py` và `seq_adapter.py`.
