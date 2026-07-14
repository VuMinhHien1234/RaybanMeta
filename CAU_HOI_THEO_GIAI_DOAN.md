# Câu hỏi cần chốt theo từng giai đoạn (đọc từ `Team_Plan_3nguoi.md`)

> Mục đích: trước khi bắt tay vào mỗi giai đoạn, team trả lời được các câu dưới đây
> thì làm sẽ trơn — không phải dừng giữa chừng để cãi nhau về thiết kế.
> Đánh dấu: 🔴 = chặn tiến độ, phải chốt trước khi code; 🟡 = chốt trong lúc làm.

---

## G0 — Nền tảng *(code xong — chỉ còn câu hỏi vận hành)*
1. 🔴 Cả 3 máy đã pass `check_env.py` + `pytest` (20 passed) chưa? Máy nào có GPU gì (CUDA/MPS), máy nào làm "máy chạy thí nghiệm chung"?
2. 🔴 Repo git chung đặt đâu (GitHub private?), quy ước làm việc: nhánh theo người hay theo giai đoạn, có cần review trước khi merge không?
3. 🟡 Log thí nghiệm dùng gì — JSON có sẵn trong `artifacts/` đủ chưa, hay muốn thêm W&B (plan cho phép cả hai)?
4. 🟡 Deadline tổng của dự án là khi nào? (Plan ước 10–19 tuần — có khớp lịch thật không? Nếu ngắn hơn, cắt G4 hay cắt ablation?)

## G1 — Baseline & khung đo *(code xong — câu hỏi nằm ở ĐỌC SỐ)*
1. 🔴 Bảng EuroSAT: `finetune` có quên rõ không (Forgetting ≫ 0, BWT âm)? Nếu KHÔNG → nghi ngờ lr/epochs/stream trước khi tin bất kỳ số nào khác.
2. 🔴 NCM đứng đâu so với finetune về Average Accuracy? (Câu này quyết định *luận điểm* của G2/G3: nếu NCM vừa không quên vừa accuracy cao → Titans/CMS phải nhắm vào chỗ NCM yếu, ví dụ khả năng thích nghi feature; nếu NCM accuracy thấp → NL có đất diễn rõ ràng.)
3. 🔴 RESISC45 full fine-tune mất bao lâu 1 method trên máy GPU của team? Có chấp nhận được ×5 method không, hay phải `backbone.freeze=true` / giảm `epochs_per_task`?
4. 🟡 Chốt bộ "official": seed nào (0?), `ewc_lambda` (100/1000/10000), `replay.buffer_per_class` (5/20/50), `lwf_lambda` (0.5/1/2) — giá trị nào vào bảng chính thức, ai chạy sweep?
5. 🟡 Ai viết 1 trang "đọc bảng G1" (kết luận: quên bao nhiêu, thuốc cũ đỡ bao nhiêu) làm mốc cho báo cáo?

## G2 — Titans *(câu hỏi THIẾT KẾ — chốt trước khi code)*
1. 🔴 **"Chuỗi" đưa vào NeuralMemory là gì?** — quyết định lớn nhất G2, plan gọi là "trục thời gian":
   - (a) chuỗi patch trong 1 ảnh (dễ nhất, memory reset mỗi ảnh),
   - (b) chuỗi ảnh liên tiếp trong 1 task (memory sống trong task),
   - (c) chuỗi xuyên task (memory sống cả stream — đúng "học liên tục" nhất, khó nhất).
   Đề xuất: bắt đầu (a) để chạy được, mục tiêu chính là (b)/(c).
2. 🔴 Bộ nhớ **reset khi nào**? (mỗi ảnh / mỗi task / không bao giờ) — phải chốt cùng câu 1, vì nó định nghĩa "trí nhớ" của hệ thống.
3. 🔴 G2 có **freeze backbone** không? (Freeze → cô lập được đóng góp của Titans, so thẳng với NCM; không freeze → lẫn hiệu ứng fine-tune. Đề xuất: freeze ở G2.)
4. 🟡 Adapter của N3 ra shape gì cho khớp titans-pytorch — (B, seq_len, dim) với dim = feat_dim của backbone hay chiếu về dim riêng? Chốt interface này sớm để N2/N3 làm song song.
5. 🟡 Titans đặt SAU backbone (feature-level) — có thử xen giữa block không, hay để G4?
6. 🟡 Tiêu chí "có tín hiệu" để qua cổng: thắng finetune về Forgetting? thắng EWC? (Plan chấp nhận cả kết quả âm miễn giải thích được — vậy ai viết phần giải thích?)

## G3 — CMS retrofit ⭐ *(mấu chốt của dự án)*
1. 🔴 Chia block ViT thành **mấy tầng tần số** và block nào nhanh/chậm? (Block đầu = đặc trưng thấp, block cuối = ngữ nghĩa — *chưa có đáp án sẵn cho vision*, phải ablate. Ai chạy ablation, mấy cấu hình? Plan yêu cầu ≥2.)
2. 🔴 Chu kỳ update mỗi tầng bao nhiêu (kiểu obekt `[[n,1],[n,4],[n,16]]`?) và **η per-tier** đặt thế nào (tầng chậm η→0)?
3. 🟡 Gradient giữa các lần update của tầng chậm: **gộp trung bình hay bỏ qua**? (obekt gộp trung bình — theo không?)
4. 🟡 Attention + LayerNorm: đóng băng hẳn hay cho vào tầng chậm nhất? (Đề xuất giai đoạn đầu: đóng băng, chỉ MLP đa tần số — đúng trọng tâm paper.)
5. 🟡 "Continual-adapt nhẹ" = bao nhiêu epoch, lr bao nhiêu so với G1? (để so công bằng: cùng tổng compute với baseline hay cùng số epoch?)
6. 🟡 Log gì để **chứng minh tầng chậm gần như bất động** (‖Δweight‖ theo thời gian?) — cần cho báo cáo, làm sẵn từ đầu rẻ hơn làm lại sau.

## G4 — HOPE
1. 🔴 Ghép Titans + CMS theo kiến trúc nào: Titans nằm *trong* khối CMS (bám §8 paper) hay chỉ nối tiếp Titans → backbone-CMS (đơn giản hoá)? Mức trung thành với paper là bao nhiêu thì đủ cho báo cáo?
2. 🔴 Có làm optimizer M3/Delta-Momentum không, hay giữ AdamW để tách bạch "đóng góp của kiến trúc" khỏi "đóng góp của optimizer"? (Đề xuất: AdamW trước, M3 là tùy chọn nếu dư thời gian.)
3. 🟡 Ablation bắt buộc: baseline / Titans-only / CMS-only / HOPE — có ép **cùng compute budget** không?
4. 🟡 Nếu HOPE *không* vượt baseline mạnh nhất (replay/NCM): kế hoạch phân tích "vì sao" gồm những thí nghiệm chẩn đoán nào?

## G5 — Đánh giá chặt
1. 🔴 Mấy seed cho bảng cuối (3 hay 5)? Tổng số run = #method × #dataset × #seed — máy chịu nổi trong bao lâu?
2. 🟡 Robustness test nào khả thi với RESISC45 (không có ngày/đêm thật): đổi **thứ tự task** theo seed? thêm corruption (mờ/nhiễu/sương)? train RESISC45 → test scene tương đương ở dataset khác?
3. 🟡 Case study định tính: chọn class nào để soi confusion trước/sau khi học task mới?

## G6 — Báo cáo
1. 🔴 Đầu ra cuối là gì: báo cáo đồ án, draft paper, hay cả hai? Deadline nộp?
2. 🟡 Khi nào **freeze code** (không sửa engine/metrics nữa để số không đổi)?
3. 🟡 Ai giữ script "one-click tái lập" và kiểm tra nó chạy trên máy sạch?

---

## 3 câu cần trả lời NGAY tuần này (vì đang ở ranh G1→G2)
1. Bảng EuroSAT + RESISC45 ra số thế nào — finetune có quên rõ, NCM đứng đâu? *(quyết định luận điểm G2/G3)*
2. G2: chuỗi cho Titans là patch-trong-ảnh, ảnh-trong-task, hay xuyên-task? *(quyết định toàn bộ thiết kế adapter của N3)*
3. G2: freeze backbone hay không? *(quyết định cách đọc kết quả Titans)*
