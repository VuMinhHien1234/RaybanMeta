# Hướng dẫn đọc code dự án UAV Continual Learning

> Dành cho người **mới hoàn toàn**, chưa biết lập trình. Mục tiêu: đọc dần và hiểu được toàn bộ dự án mà không bị choáng.
>
> Cách dùng file này: đọc từ trên xuống. Tới **Phần 4 (Lộ trình đọc)** thì làm theo đúng thứ tự, đọc xong file nào thì đánh dấu `[x]` vào ô vuông trước nó.

---

## Phần 1 — Dự án này làm gì? (bằng lời, chưa cần code)

Tưởng tượng một chiếc **drone (UAV)** phải học nhận diện mọi thứ nó nhìn thấy từ trên cao: đồng ruộng, sân bay, rừng, sông, khu dân cư...

Vấn đề: khi AI học thứ **mới**, nó thường **quên** thứ **cũ** — giống như bạn ôn thi môn mới rồi quên sạch môn tuần trước. Ngành học máy gọi đây là **"quên thảm họa" (catastrophic forgetting)**.

Dự án này **thử nhiều chiến lược khác nhau** để giúp AI *"học cái mới mà không quên cái cũ"*, rồi **chấm điểm** xem chiến lược nào tốt nhất. Gần như mọi dòng code đều phục vụ đúng một mục tiêu đó.

Dự án chia thành nhiều **giai đoạn** (bạn sẽ thấy chữ **G1, G2, G3, G4** ở khắp nơi — tên file, config, test):

- **G1** = các cách làm **cơ bản** (gọi là *baseline*) để làm mốc so sánh.
- **G2 / G3 / G4** = các cách làm **nâng cao**, tên là **Titans / CMS / HOPE** (lấy ý tưởng từ một bài báo khoa học năm 2025).

Khi mới đọc, bạn chỉ cần quan tâm **G1**. G2–G4 để dành cho sau cùng.

---

## Phần 2 — Vài khái niệm cần biết trước

Không cần hiểu sâu, chỉ cần "à, từ này nghĩa là vậy":

- **Continual learning (học liên tục):** dạy AI theo từng **đợt** nối tiếp nhau, thay vì dạy tất cả một lần.
- **Task (đợt/nhiệm vụ):** một đợt học. Ví dụ đợt 1 dạy 5 loại địa hình, đợt 2 dạy 5 loại khác.
- **Model (mô hình) / backbone (xương sống):** chính là "bộ não AI" — mạng nơ-ron nhận ảnh vào và đoán ra nhãn.
- **Baseline (mốc cơ bản):** cách làm đơn giản để so sánh. Cách nâng cao phải thắng được baseline mới có ý nghĩa.
- **Config (cấu hình):** file cài đặt kiểu `tên: giá_trị`, quyết định chạy thí nghiệm nào, bao nhiêu vòng...
- **Metric (chỉ số):** con số chấm điểm — học tốt không, quên bao nhiêu.
- **Python:** ngôn ngữ lập trình dùng trong dự án. File code có đuôi `.py`.

---

## Phần 3 — Bản đồ thư mục (mỗi thứ một câu)

Toàn bộ code nằm trong thư mục `uav-continual-learning/`. Bên trong:

```
uav-continual-learning/
├── README.md            # Giới thiệu + cách cài đặt
├── configs/             # Các file CÀI ĐẶT (chạy thí nghiệm gì) — dễ đọc nhất
├── src/uavcl/           # TOÀN BỘ code chính nằm ở đây
│   ├── data/            #   Lo phần DỮ LIỆU: lấy ảnh ở đâu, chia thành các "đợt học"
│   ├── models/          #   Các BỘ NÃO AI (mạng nơ-ron), từ đơn giản đến phức tạp
│   ├── methods.py       #   Các CHIẾN LƯỢC chống quên (trái tim ý tưởng)
│   ├── engine.py        #   VÒNG LẶP huấn luyện (trái tim vận hành — nối mọi thứ lại)
│   ├── metrics/         #   CHẤM ĐIỂM: học tốt không, quên bao nhiêu
│   ├── optim/           #   Phần nâng cao (G3–G4), để sau cùng
│   └── utils/           #   Tiện ích lặt vặt (cố định hạt ngẫu nhiên, đọc config)
├── scripts/             # Các nút "CHẠY" để khởi động thí nghiệm
└── tests/               # Kiểm tra code chạy đúng
```

Ghi nhớ 2 file **trái tim** để không lạc: `methods.py` (chiến lược chống quên) và `engine.py` (vòng lặp chạy tất cả).

---

## Phần 4 — Lộ trình đọc (làm theo đúng thứ tự này)

Chia thành 5 chặng, **dễ trước khó sau**. Đừng nhảy cóc. Đọc xong đánh dấu `[x]`.

### Chặng 0 — Hiểu ý tưởng (chưa cần đọc code)

- [ ] `README.md` — dự án là gì, cách cài đặt.
- [ ] `plans/00_LO_TRINH_TONG.md` — bức tranh tổng thể các giai đoạn.
- [ ] `CAU_HOI_THEO_GIAI_DOAN.md` — các câu hỏi/mục tiêu theo từng giai đoạn.

### Chặng 1 — File code dễ nhất, làm quen

- [ ] `configs/g1_smoke.yaml` — chỉ là danh sách cài đặt. Tập đọc kiểu `khóa: giá_trị`, chưa cần hiểu hết.
- [ ] `src/uavcl/data/stream.py` — **file đầu tiên nên đọc kỹ.** Nó giải thích ý tưởng cốt lõi (chia dữ liệu thành từng "đợt học"), viết bằng Python thuần, không phức tạp, chú thích rất đầy đủ.
  - Cần để ý: hàm `split_classes` (chia lớp thành các đợt) và `build_stream` (dựng chuỗi các đợt học).

### Chặng 2 — Đi theo luồng dữ liệu

- [ ] `src/uavcl/data/sources.py` — ảnh đến từ đâu (dữ liệu giả, EuroSAT, RESISC45).
- [ ] `src/uavcl/data/loaders.py` — nạp ảnh vào để huấn luyện ra sao.
- [ ] `src/uavcl/models/backbone.py` — "xương sống" trích đặc trưng từ ảnh.
- [ ] `src/uavcl/models/classifier.py` — bộ phận đưa ra dự đoán cuối cùng.
- [ ] `src/uavcl/models/ncm.py` — một mô hình đơn giản mà hiệu quả (đọc để thấy AI không phải lúc nào cũng phức tạp).

### Chặng 3 — Trái tim của dự án

- [ ] `src/uavcl/methods.py` — bắt đầu từ class `FineTune` (nó *"không làm gì cả"* → dễ nhất), rồi mới tới `EWC`, `Replay`, `LwF`. Mỗi class là một chiến lược chống quên khác nhau.
- [ ] `src/uavcl/engine.py` — vòng lặp huấn luyện nối tất cả lại. **Đọc file này sau cùng trong chặng** vì nó dùng mọi thứ ở trên. Để ý nó tạo ra "ma trận R" — bảng số ghi lại AI làm tốt/quên bao nhiêu sau mỗi đợt.

### Chặng 4 — Chấm điểm & chạy thử

- [ ] `src/uavcl/metrics/continual.py` — cách tính "học tốt không" và "quên bao nhiêu" từ ma trận R.
- [ ] `scripts/run_g1.py` — xem mọi mảnh được ráp lại để chạy một thí nghiệm thật.

### Chặng 5 — Nâng cao (để dành, đọc cuối cùng)

Chỉ đụng vào khi đã nắm chắc chặng 1–4:

- [ ] `src/uavcl/models/memory.py`, `titans_head.py`, `cms.py`, `hope.py`, `self_ref_memory.py`
- [ ] Cả thư mục `src/uavcl/optim/` (`m3.py`, `cms_optimizer.py`)

Đây là phần khó nhất (G2–G4). Nếu chưa hiểu, hoàn toàn bình thường — đó là nội dung trình độ nghiên cứu.

---

## Phần 5 — Ba mẹo đọc code cho người mới

1. **Đọc chú thích trước, code sau.** Mỗi file có khối `GIẢI THÍCH TỔNG QUAN` ở đầu — đọc nó là hiểu 70%. Mọi dòng bắt đầu bằng `#` hoặc dấu `↳` đều là **lời giải thích** cho người đọc, không phải lệnh máy chạy.

2. **Đừng cố hiểu từng dòng.** Mục tiêu là nắm *"dữ liệu đi từ đâu tới đâu"*, không phải dịch từng ký tự. Chỗ nào bí thì bỏ qua, đọc tiếp — thường đọc phần sau sẽ tự sáng ra phần trước.

3. **Đọc từ trên xuống trong mỗi file:** đầu file (phần mô tả trong `"""..."""`) → tên các hàm (dòng bắt đầu bằng `def`) → rồi mới vào chi tiết bên trong.

Mẹo phụ: gặp từ khóa lạ (`def`, `class`, `import`, `return`...) thì tra nhanh nghĩa một lần, lần sau sẽ quen. Xem bảng dưới.

---

## Phần 6 — Muốn thấy code chạy thật (tùy chọn)

Nhìn code chạy ra kết quả giúp hiểu nhanh hơn nhiều. Nếu máy đã cài đặt xong theo `README.md`, mở terminal trong thư mục `uav-continual-learning/` và gõ:

```bash
python scripts/check_env.py      # kiểm tra môi trường, mọi dòng nên là [OK]
python scripts/run_g1.py --config configs/g1_smoke.yaml   # chạy thử ~1 phút trên dữ liệu giả
```

Nếu chưa cài được cũng không sao — cứ đọc theo lộ trình ở Phần 4 trước.

---

## Phần 7 — Bảng thuật ngữ nhanh

| Từ trong code | Nghĩa dễ hiểu |
|---|---|
| `def ten_ham(...)` | Định nghĩa một **hàm** — một việc AI làm được, gọi tên là chạy. |
| `class TenClass` | Một **khuôn** gom dữ liệu + hành động lại (vd mỗi chiến lược chống quên là 1 class). |
| `import ...` | **Mượn** code từ file/thư viện khác để dùng. |
| `return` | Hàm **trả về** kết quả. |
| `#` hoặc `↳` | **Chú thích** cho người đọc — máy bỏ qua. |
| `"""..."""` | Đoạn **mô tả** ở đầu file/hàm (gọi là docstring). |
| `config / .yaml` | File **cài đặt** kiểu `khóa: giá_trị`. |
| `train` | **Huấn luyện** — cho AI học từ dữ liệu. |
| `eval` | **Đánh giá** — kiểm tra AI đoán đúng bao nhiêu. |
| `task` | Một **đợt học** (vài lớp mới). |
| `baseline` | Cách làm **cơ bản** để so sánh. |
| `forgetting` | Mức độ AI **quên** kiến thức cũ. |

---

*Bước tiếp theo gợi ý: sau khi đọc xong Chặng 0–1, quay lại nhờ giải thích chi tiết từng dòng của `data/stream.py` — đó là điểm khởi đầu tốt nhất để thật sự "đọc code".*
