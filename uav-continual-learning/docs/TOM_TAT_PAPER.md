# Tóm tắt & giải thích chi tiết 13 nguồn trong DANH_SACH_PAPER.md

> **Bản mở rộng (v2)** — chi tiết hơn bản đầu: thêm số phương trình thật (đọc trực tiếp PDF gốc
> của A1 Titans và toàn văn NL.pdf qua `pypdf`, không phải bản HTML rút gọn), số liệu thực nghiệm
> thật (bảng kết quả, không chỉ mô tả suông), và các phát hiện lý thuyết sâu (Definition 2-4 của
> NL, suy diễn Adam/AdaGrad ở Appendix B, suy diễn Delta GD ở Appendix C, kết quả ablation của
> Titans). Mỗi mục vẫn giữ cấu trúc: **Nội dung chính** (công thức + số liệu thật) → **Map sang
> code dự án** → **Câu hỏi tự kiểm tra**. Đã đọc trực tiếp `memory.py`, `m3.py`, `cms.py`,
> `cms_optimizer.py`, `hope.py`, `titans_head.py`, `seq_adapter.py`, `state_utils.py`, `methods.py`,
> `metrics/continual.py` trước khi viết, không suy diễn chung chung.

---

## Nhóm A — BẮT BUỘC, phục vụ G2

### A1. Titans: Learning to Memorize at Test Time
Behrouz, Zhong, Mirrokni (Google Research) — arXiv:2501.00663 · https://arxiv.org/abs/2501.00663
(bản v1, 31/12/2024; nhóm nói rõ đây là bản đầu, "đang hoàn thiện kết quả với model lớn hơn").

**Bối cảnh & câu hỏi nghiên cứu.** Attention nhớ chính xác (không nén, giữ nguyên mọi token) nhưng
chi phí bậc hai theo độ dài chuỗi; linear attention/RNN hiện đại rẻ (tuyến tính) nhưng nén lịch sử
vào **một trạng thái cỡ cố định** — mâu thuẫn: dùng model tuyến tính để scale lên chuỗi cực dài,
nhưng chuỗi càng dài thì càng khó nén vừa vào một vector/ma trận cỡ cố định. Từ đó bài đặt 5 câu hỏi
nghiên cứu (Q1–Q5): cấu trúc bộ nhớ nên thế nào, luật cập nhật nên thế nào, cách truy xuất nên thế
nào, kiến trúc nên tích hợp nhiều bộ nhớ tương tác thế nào, và bộ nhớ có cần **sâu** (deep, nhiều
lớp) hay một ma trận là đủ. Toàn bộ phần còn lại của paper trả lời tuần tự 5 câu này.

**Cơ chế lõi (§3.1 Long-term Memory) — nguyên văn công thức, không phải diễn giải:**

Bắt đầu từ ý tưởng thô nhất — chỉ dùng surprise tức thời (Eq. 8):
```
M_t = M_{t-1} − θ_t·∇ℓ(M_{t-1}; x_t)                                    (8)
```
Nhược điểm: sau vài bước "quá sốc", gradient nhỏ dần → mắc kẹt ở vùng phẳng, bỏ lỡ thông tin đến
sau. Paper sửa bằng cách tách **surprise quá khứ** (past surprise) và **surprise tức thời**
(momentary surprise), giống hệt gradient descent với momentum (Eq. 9–10):
```
M_t = M_{t-1} + S_t
S_t = η_t·S_{t-1} − θ_t·∇ℓ(M_{t-1}; x_t)     [η_t: surprise decay | θ_t: tốc độ ghi tức thời]  (9-10)
```
η_t, θ_t là **hàm học từ chính x_t** (không phải hằng số) — đây là điểm khác biệt với momentum GD
thường: η_t→0 khi ngữ cảnh đổi (chủ động "quên" surprise cũ), η_t→1 khi token hiện tại còn liên
quan chặt đến các token gần trước. Đối tượng bị nhớ là cặp key-value, chiếu tuyến tính giống hệt
attention (Eq. 11) với loss L2 giữa giá trị truy hồi và giá trị thật (Eq. 12):
```
k_t = x_t·W_K,  v_t = x_t·W_V                                            (11)
ℓ(M_{t-1}; x_t) = ‖M_{t-1}(k_t) − v_t‖²                                  (12)
```
Cuối cùng thêm **forget gate** α_t ∈[0,1] (cũng học từ dữ liệu) để quản lý dung lượng hữu hạn khi
chuỗi cực dài (Eq. 13–14) — α_t→0: cập nhật không đụng tới phần đã nhớ; α_t→1: xoá sạch:
```
M_t = (1 − α_t)·M_{t-1} + S_t
S_t = η_t·S_{t-1} − θ_t·∇ℓ(M_{t-1}; x_t)                                (13-14)
```
Truy xuất — **forward không cập nhật trọng số** (ký hiệu `*`), q_t=x_t·W_Q (Eq. 15): `y_t = M*(q_t)`.
Paper nhấn mạnh: kiến trúc bộ nhớ là MLP ≥2 lớp ("deep memory") — vì bộ nhớ 1 ma trận tương đương
hồi quy tuyến tính online (giả định quan hệ dữ liệu tuyến tính), còn MLP 2 lớp biểu cảm hơn hẳn về
lý thuyết (Hornik et al. 1989) — và thực nghiệm ở §5.5 xác nhận đúng điều này.

**Song song hoá (§3.2, Eq. 16–18):** mini-batch GD + momentum + weight decay tưởng chừng phải chạy
tuần tự (mỗi bước phụ thuộc bước trước) nhưng paper chứng minh có thể viết lại thành các phép nhân
ma trận theo từng "chunk" (kỹ thuật thừa hưởng từ Sun et al. 2024/TTT — xem B4) — đây là lý do
NeuralMemory có tham số `chunk_size` mà dự án dùng trong `memory.py`.

**Persistent memory (§3.3, Eq. 19) — điểm hay bị bỏ qua:** ngoài neural memory (phụ thuộc ngữ cảnh),
Titans còn thêm N_p vector **học được nhưng KHÔNG phụ thuộc input**, gắn vào đầu chuỗi. Lý do thú
vị nhất (Eq. 20), nối thẳng sang paper A2: Sukhbaatar et al. (2019) chứng minh thay ReLU trong FFN
bằng Softmax cho ra `FFN(x) = W_V·Softmax(W_K·x)` — **giống hệt một phép attention nhưng với W_K,
W_V không phụ thuộc dữ liệu** — nên persistent memory được kỳ vọng hoạt động như "attention có sẵn
đáp án, không cần tính từ input". Đây chính là cầu nối lý thuyết giữa "MLP = key-value memory"
(A2) và "persistent memory = attention tĩnh" (A1).

**Ba kiến trúc tích hợp — công thức chính xác (§4), không chỉ mô tả:**

*MAC — Memory as Context (Eq. 21–25):* Trước tiên **đọc** (không ghi) bộ nhớ bằng chính đoạn hiện
tại làm query: `h_t = M*_{t-1}(q_t)`. Ghép `[persistent ‖ h_t ‖ đoạn hiện tại S^(t)]` làm input cho
attention: `y_t = Attn(S̃^(t))`. Sau đó dùng **output của attention** để tiếp tục huấn luyện bộ nhớ
cho đoạn kế tiếp: `M_t = M_{t-1}(y_t)` rồi lấy output cuối `o_t = y_t ⊗ M*_t(y_t)`. Nghĩa là attention
"quyết định" điều gì đáng nhớ dài hạn — không phải input thô mà là input ĐÃ qua attention.

*MAG — Gated Memory (Eq. 26–28):* Hai nhánh song song trên CÙNG input (không segment): sliding-window
attention `y = SW-Attn*(x̃)` và neural memory `M(x̃)` (ký hiệu không có chỉ số t nghĩa là "sau khi chạy
hết toàn bộ chuỗi"), trộn bằng gate phi tuyến: `o = y ⊗ M(x̃)`.

*MAL — Memory as a Layer (Eq. 29–31):* Xếp tuần tự như layer thường: `y = M(x̃)` rồi `o =
SW-Attn(y)`. Nhược điểm tự thừa nhận: bị giới hạn bởi khả năng của TỪNG layer riêng lẻ, không tận
dụng được tính bổ trợ giữa attention và memory.

*Memory-without-attention (LMM/Titans-LMM):* biến thể đặc biệt của MAL — bỏ hẳn attention, chỉ dùng
neural memory làm sequence model. Paper coi đây là phép thử độc lập bắt buộc, vì "học = có khả năng
nhớ hiệu quả ngay cả khi không có short-term memory".

**Theorem 4.1 (đáng chú ý, hay bị bỏ sót):** Transformer, linear RNN đường chéo, và DeltaNet đều bị
giới hạn trong lớp phức tạp TC⁰; Titans **vượt** giới hạn này — biểu cảm hơn về mặt lý thuyết cho
bài toán state-tracking, không chỉ là "nhanh hơn/rẻ hơn" mà còn "làm được việc Transformer không
làm được về nguyên tắc".

**Thực nghiệm — số liệu thật (§5), không chỉ "outperform baseline":**
- *Language modeling* (Bảng 1, 760M tham số/30B token): Titans (MAC) ppl **19.93** vs Transformer++
  **25.21** vs Gated DeltaNet **21.18** vs TTT **24.17** — Titans (LMM, chỉ riêng neural memory
  không attention) đã ppl **20.04**, tốt hơn MỌI baseline non-hybrid kể cả Mamba2/Gated DeltaNet.
- *Needle-in-Haystack* (Bảng 2, RULER S-NIAH, chuỗi 16K token): TTT rớt còn 0.0–4.4% ở dạng khó
  nhất (S-NIAH-W), Mamba2 rớt còn 0.0%, DeltaNet 0.0% — trong khi Titans (MAC) vẫn giữ **95.2%**.
  Đây là bằng chứng trực tiếp cho luận điểm "surprise + momentum + forget gate quản lý dung lượng
  bộ nhớ tốt hơn hẳn khi chuỗi cực dài" — không phải khác biệt nhỏ, mà là khác biệt giữa "còn nhớ"
  và "quên sạch".
  - **BABILong** (few-shot): Titans (MAC) vượt cả GPT-4, dù tham số ít hơn rất nhiều.
- *Ablation* (Bảng 5, đúng thứ tự đóng góp paper công bố): bỏ **weight decay** hại nhiều nhất (ppl
  27.01→29.04), rồi đến **momentum** (→28.98), **convolution** (→28.73), cuối cùng **persistent
  memory** (nhẹ nhất, 27.01→27.63 nhưng vẫn dương). Tức forget gate (weight decay) — chính là α_t
  — là thành phần quan trọng NHẤT trong 4 thành phần, quan trọng hơn cả momentum.
- *Độ sâu bộ nhớ* (§5.5): tăng L_M (số lớp MLP của memory) → ppl giảm đều, nhưng thông lượng train
  giảm tuyến tính theo độ sâu — trade-off hiệu quả/tốc độ rõ ràng, không "miễn phí".

**Map sang code dự án — đọc kỹ vì đây là chỗ hay bị hiểu lầm:** `TitansMemory`
(`src/uavcl/models/memory.py`) bọc thẳng `titans_pytorch.NeuralMemory` — module hiện thực đúng
Eq. 8–15 ở trên (log `[titans] norm(state)` trong `methods.py` theo dõi ‖M_t‖ để phát hiện memory
"nổ" nếu α_t/η_t học lệch). Nhưng `TitansClassifier` (`titans_head.py`) ghép: `frozen ViT →
SeqAdapter → TitansMemory → post_norm → head` — **không có attention nào chạy song song/nối tiếp**,
nên G2 hiện tại chính xác là biến thể **LMM/Titans-LMM** (memory không attention) trong bảng thực
nghiệm ở trên — nghĩa là số liệu tham khảo đúng nhất từ paper để so sánh kỳ vọng không phải hàng
MAC/MAG mà là hàng **"Titans (LMM)"**. Đến G4/HOPE, `attn: slow` trong `cms.py` đưa attention ViT
vào tier chậm nhất — ghép ở mức feature (sau ViT), không token-level như MAC/MAG/MAL thật.

**Câu hỏi tự kiểm tra:**
- Viết lại đúng thứ tự: Eq. 8 (surprise thô) → Eq. 9-10 (thêm momentum) → Eq. 13-14 (thêm forget
  gate) khác nhau ở đâu, và vì sao paper phải sửa dần từng bước thay vì viết thẳng công thức cuối?
- Bảng ablation cho thấy weight decay (α_t) quan trọng hơn momentum (η_t/θ_t) — điều này có ăn khớp
  với cách dự án log `norm(state)` để "phát hiện phình" không? Nếu α_t học kém, hiện tượng nào xảy
  ra trước: state phình (thiếu weight decay) hay state "cứng" không cập nhật (thiếu momentum)?
- G2 hiện tại tương ứng hàng nào trong Bảng 1/2 (MAC/MAG/MAL hay LMM)? Số ppl/accuracy trong bảng
  đó có so sánh được trực tiếp với bảng RESISC45/EuroSAT của dự án không, hay chỉ so sánh được về
  MẶT CƠ CHẾ (không so được về con số vì khác domain/task hoàn toàn)?
- Theorem 4.1 nói Titans vượt giới hạn TC⁰ — điều này có ý nghĩa gì cho lý do "tại sao chọn Titans
  thay vì chỉ tăng attention window" khi viết Related Work?
- Eq. 20 (FFN=Softmax attention tĩnh) nối persistent memory (A1) với "MLP=key-value" (A2) thế nào?

### A2. Transformer Feed-Forward Layers Are Key-Value Memories
Geva, Schuster, Berant, Levy — EMNLP 2021 · arXiv:2012.14913 · https://arxiv.org/abs/2012.14913

**Nội dung chính, chi tiết hơn.** FFN trong Transformer có dạng `FFN(x) = f(x·K)·V` (K là ma trận
lớp 1, V là ma trận lớp 2). Bài lập luận: nếu coi mỗi HÀNG của K là một "key" k_i và mỗi HÀNG tương
ứng của V là một "value" v_i, thì FFN chính là bộ nhớ liên kết y = Σ_i f(x·k_i)·v_i — giống hệt cấu
trúc attention (Q·Kᵀ rồi nhân V), chỉ khác K/V ở đây **không phụ thuộc input** (cố định sau khi
train — đây chính là điểm A1 dùng để biện minh cho persistent memory, xem Eq. 20 ở mục A1).

Ba phần thực nghiệm chính (không chỉ khẳng định suông):
1. **Key khớp pattern nào:** với mỗi key k_i, tìm trong tập huấn luyện những chuỗi token mà tích
   vô hướng x·k_i lớn nhất — quan sát: layer nông (gần input) khớp pattern **bề mặt** (n-gram/kết
   thúc câu bằng từ cụ thể); layer sâu khớp pptern **ngữ nghĩa** (cùng chủ đề, dù từ vựng khác hẳn).
2. **Value dự đoán token nào:** chiếu mỗi value v_i qua ma trận embedding đầu ra (giống cách suy ra
   logit) để xem token nào có xác suất cao nhất — token đó thường chính là token có khả năng xuất
   hiện NGAY SAU pattern mà key tương ứng bắt được, và ở các layer trên cùng, top-1 token này khớp
   với next-token thật khoảng một phần ba số trường hợp — tức nhiều "cell" nhớ trong FFN có thể tự
   nó "bỏ phiếu" trực tiếp cho token kế tiếp mà không cần layer nào khác.
3. **Ghép nhiều memory cell:** output FFN = TỔNG có trọng số của mọi value (không chỉ 1 cell chiếm
   ưu thế) — nên FFN hoạt động như bỏ phiếu từ hàng trăm/nghìn "chuyên gia nhỏ" cùng lúc; residual
   connection giữa các layer thì tinh chỉnh dần phân phối này để ra dự đoán cuối.

**Vì sao liên quan trực tiếp đến dự án:** đây là nền móng thực nghiệm khiến CMS (B1 §7) chọn đúng
khối MLP để chia tần số ghi nhớ, KHÔNG phải attention. Nếu mỗi "hàng" của MLP là 1 mục từ điển
key-value đã học, khoá MLP các block cũ vào tier chậm (η→0) = "khoá lại từ điển cũ", để MLP block
mới cập nhật tier nhanh = "ghi thêm mục mới" — đúng cơ chế continual learning không cần đụng attention.

**Map sang code dự án:** `build_cms_param_groups` (`models/cms.py`) chỉ gom `blocks[i].mlp.parameters()`
vào tier fast/mid/slow; `attn/norm1/norm2` bị tách riêng (tier chậm nhất hoặc đóng băng — quyết
định team, không phải MLP nên theo Geva et al. không "chứa tri thức" theo nghĩa key-value).

**Câu hỏi tự kiểm tra:**
- Vì sao Geva et al. gọi hàng của W1 là "keys" và hàng của W2 là "values" — phép loại suy này khác
  gì key/value trong self-attention (gợi ý: input-dependent hay không)?
- "Layer nông bắt pattern bề mặt, layer sâu bắt pattern ngữ nghĩa" — dữ liệu ảnh vệ tinh (RESISC45/
  EuroSAT) có tương tự không, hay khái niệm "pattern ngữ nghĩa" ở ảnh vs text khác nhau thế nào?
- Nếu ~1/3 value ở layer trên cùng "bỏ phiếu" trực tiếp đúng cho token kế tiếp, điều này gợi ý gì
  về việc chọn tier CHẬM (bảo toàn tri thức) nên ưu tiên block NÀO trong ViT — block cuối (gần
  head, semantic cao) hay block đầu?
- `attn` bị đưa vào tier chậm nhất (không có tier riêng) trong `cms.py` — theo lập luận Geva et al.,
  điều này có "bỏ sót" phần tri thức nào không (attention có lưu tri thức key-value theo nghĩa
  Geva hay không — trả lời dựa trên khác biệt input-dependent vs input-independent)?

---

## Nhóm B — NÊN ĐỌC, phục vụ G3/G4

### B1. NL.pdf — Nested Learning: The Illusion of Deep Learning Architecture
Behrouz, Razaviyayn, Zhong, Mirrokni (Google Research) — bản PDF trong folder, đề ngày 02/12/2025.
**Paper nguồn của toàn bộ kiến trúc HOPE/CMS/M3** — đã đọc trực tiếp §2, §3 (3.1–3.3), §4, §6, §7,
§8, và cả 3 phụ lục A/B/C (không bỏ sót phần nào có công thức quan trọng).

#### §2 — Nền tảng: 3 cách viết lại Gradient Descent
GD thường (Eq. 1) tương đương 3 cách viết khác nhau, cả 3 đều được paper dùng lại xuyên suốt:
- **Steepest descent** (Eq. 2): W_{t+1} = argmin_W {⟨∇L, W⟩ + (1/2η)‖W−W_t‖²} — GD = bước gần nhất
  (theo chuẩn L2) mà vẫn giảm loss tuyến tính hoá tại W_t. **Đây là dạng dùng để định nghĩa mọi
  associative memory trong bài** (so khớp Eq. 6 ở §3 dưới).
- **Follow-The-Regularized-Leader/FTRL** (Eq. 3): W_{t+1} = W_1 − η·Σ∇L — cộng dồn toàn bộ lịch sử
  gradient, không chỉ bước gần nhất.
- **Meta-learning framing** (Eq. 4): outer loop tối ưu Φ sao cho inner loop ℓ(θ,D;Φ) tốt trên phân
  phối task — đây chính là khuôn mà toàn bộ NL dùng để mô tả "level ngoài quyết định level trong
  học thế nào".

#### §3.1 — Associative Memory (Định nghĩa 1) + 3 ví dụ xây trực giác
**Định nghĩa 1 (Eq. 6):** M* = argmin_M 𝓛̃(M(K);V) — bất kỳ toán tử nào map key→value bằng cách tối
thiểu hoá một loss đều là associative memory. Luận điểm trung tâm nhắc lại nhiều lần: **"mọi thành
phần của một mô hình học — trọng số, optimizer, memory — đều là associative memory, chỉ khác nhau ở
tần số được cập nhật (ghi)."**

Ba ví dụ từng bước (§4 dùng lại y hệt lối lập luận này):
1. MLP 1 lớp + GD (Eq. 8): W_{t+1}=W_t−η∇L — ∇L gọi là **surprise**, đúng chữ mà A1 mượn lại.
2. MLP + momentum (Eq. 10-13): momentum GD là **quy trình tối ưu 2 tầng** — tầng trong học "nhớ"
   gradient, tầng ngoài dùng giá trị nhớ để cập nhật trọng số. **"Level" = một quá trình tối ưu với
   tần số cập nhật riêng, không phải "layer" kiến trúc.**
3. Linear attention (Eq. 14-18): M_t=M_{t-1}+v_t·k_tᵀ **chính là** một bước GD trên associative
   memory loss dạng dot-product `−⟨M·k,v⟩` — không phải phép ẩn dụ, là suy ra được bằng đại số.

#### §3.2 — Nested Optimization Processes: 3 định nghĩa hình thức + ví dụ AdaTransformer
**Definition 2 (Update Frequency):** với bất kỳ thành phần A (tham số hay khối phi tham số như
attention), tần số f_A = số lần cập nhật/đơn vị thời gian. Định nghĩa quan hệ thứ tự A≻B ("A nhanh
hơn B") nếu f_A>f_B, hoặc f_A=f_B nhưng tính B cần kết quả của A trước (còn nếu không ai phụ thuộc
ai và cùng tần số → A "cùng cấp" B, ví dụ chuẩn cho trường hợp này: 2 moment của AdamW cùng tần số,
tính độc lập nhau). Từ ≻, sắp các thành phần vào các "level" có thứ tự.

**Definition 3 (Nested System, Eq. 19)** và **Definition 4 (NSAM — Nested System of Associative
Memories, Eq. 20)**: hình thức hoá chặt chẽ ý tưởng ở trên — hệ K tầng, mỗi tầng là tập bài toán tối
ưu (L_i, C_i, Θ_i), tối ưu bằng GD; với NSAM, context C_i cụ thể là các cặp key-value.

**Ví dụ AdaTransformer (Eq. 21):** so Transformer thường với biến thể thay MLP bằng linear attention
có state khởi tạo meta-learned. Cả hai chỉ khác NHAU ở **level của trọng số** — W_MLP nằm level 1
(cố định sau train), còn W_LinAttn thích nghi trong-ngữ-cảnh bằng chính công thức linear attention
(M_t=M_{t-1}+v_tk_tᵀ). Kết luận quan trọng: **"hybrid architecture" (recurrent+attention) trong tài
liệu hiện tại thực ra chỉ là Transformer chuẩn với một số khối MLP được thêm 1 level tính toán** —
không phải hai thứ khác biệt bản chất bị ghép cơ học.

#### §3.3 — Knowledge Transfer Between Levels: 5 cơ chế chuyển giao tri thức
Đây là phần trả lời trực tiếp câu "(1)...(2)...(3)...(4) Learned optimizers..." nhắc ở đầu paper:
1. **Kết nối trực tiếp — tham số (Eq. 24-26):** forward của level chậm phụ thuộc thẳng vào tham số
   level nhanh, `M⁽⁰⁾(·):=M⁽⁰⁾(·;Θ⁽¹⁾)` — ví dụ linear attention/FWP, không backprop giữa 2 level.
2. **Kết nối trực tiếp — phi tham số (Eq. 27):** giống trên nhưng level nhanh không có tham số học
   được, chỉ có nghiệm non-parametric — ví dụ chính là softmax attention (Transformer gốc).
3. **Qua backpropagation:** cả 2 level cùng một luồng gradient, chỉ khác tần số cập nhật — ví dụ
   dùng cho CMS ở §7.
4. **Qua khởi tạo (MAML-style, Eq. 28):** level chậm học ra một ĐIỂM KHỞI ĐẦU tốt cho level nhanh
   (Θ⁽¹⁾_0 = argmin E[ℓ(...)]) — đây chính là biến thể "Nested" của CMS (Eq. 72 ở §7).
5. **Qua sinh tham số/context (hypernetwork, "learned optimizers"):** một level SINH RA trọng số
   hoặc dữ liệu cho level kia — ví dụ optimizer học (learned optimizer), hoặc chính optimizer
   thường (kiến trúc sinh gradient = context cho optimizer).

#### §4 — Optimizer cũng là Associative Memory (nguồn trực tiếp của `optim/m3.py`)
*(giữ nguyên nội dung đã xác nhận ở bản trước, tóm lại các mốc chính)*
- Backprop = associative memory (Eq. 29-31, δ_ℓ = "local output surprise").
- Momentum-based optimizer = associative memory (Eq. 32-37) → tổng quát hoá mọi optimizer.
- Muon suy ra từ associative memory (Eq. 38-44): chọn objective đo trực giao ‖P(g)ᵀP(g)−I‖²_F, giải
  1 bước GD ra đúng đa thức bậc 3 của Newton-Schulz.
- Vì sao cần "long context" trong optimizer (Eq. 45, §4.3): momentum β=0.9 chỉ thực nhớ ~43 bước —
  catastrophic forgetting có thể xảy ra ngay TRONG optimizer, không cần đợi trọng số model.
- Delta Momentum (Eq. 48-49) — công thức chính xác:
  ```
  m_{i+1} = m_i·(α_{i+1} − ∇L(W_i;x_i)ᵀ∇L(W_i;x_i)) − η_t·P_i·∇L(W_i;x_i)
  ```
  `m3.py` cài **bản rút gọn** dùng key hằng số thay P_i đầy đủ (tự nhận trong sai khác #1).
- Mở rộng Eq. 50-52: Deep Momentum GD (MLP thay ma trận), higher-order feature map, và
  "non-linear output σ(·)=NewtonSchulz(·) đúng bằng Muon" (Eq. 52).

#### §6 — Takeaways: những "aha" quan trọng nhất của cả bài (mới bổ sung ở bản v2)
- **"Illusion of deep learning architectures"** giải thích trực tiếp bằng câu của paper: sự đa dạng
  kiến trúc ta thấy (Transformer/Mamba/LSTM/...) chỉ là **NGHIỆM** của các bài toán tối ưu khác
  nhau — khi nhìn thẳng vào nghiệm (attention, gate...) mà không nhìn vào trục "level/tần số" đang
  sinh ra chúng, ta tưởng chúng là những kiến trúc khác biệt về bản chất — đây chính là tên bài báo.
- **"Model có nhiều tham số hơn ta tưởng":** momentum, hidden state của RNN... đều là "tham số" theo
  nghĩa NL (bị cập nhật bởi input) dù không nằm trong nhóm "learnable" truyền thống — chúng lưu tri
  thức thật (về loss landscape, về ngữ cảnh) và mất đi nếu không có cơ chế chuyển giao giữa các level.
- **"Nhiều tính toán hơn mỗi neuron":** không phải mọi việc "thêm level" đều để làm CMS — ví dụ Muon/
  NewtonSchulz_k(·) chính là "k bước tính toán NỘI BỘ cho mỗi lần cập nhật momentum" — thêm chiều sâu
  tính toán trên mỗi tham số mà không cần thêm tham số mới.
- **In-context learning KHÔNG phải là hiện tượng "nổi lên" (emergent) một cách bí ẩn** — nó là hệ quả
  TRỰC TIẾP của việc có nhiều level trong biểu diễn NL: mọi level đều đang "học trong ngữ cảnh của
  chính nó". ICL "nhìn có vẻ emergent" chỉ vì cần level tần số thấp đủ mạnh để level tần số cao thích
  nghi nhanh — model chưa train đủ thì ICL kém, không phải vì ICL "chưa xuất hiện".
- **Test-time training/memorization = một dạng in-context learning tham số hoá** (parametric ICL) —
  tri thức học được biến mất khi ngữ cảnh bị gỡ, vì không có cơ chế chuyển giao sang level bền hơn.
- **"Không có ranh giới train/test"** trong một neural learning module — chỉ có 2 trạng thái: đang
  nhận input, hoặc đang cô lập. Pre-training chỉ là một level cụ thể (context = toàn bộ dữ liệu
  pretrain) — về bản chất cũng là một dạng in-context learning, chỉ là với context "siêu lớn".
- **"Recurrent model thay thế khối MLP"**: Mamba/Titans/TTT không phải "loại kiến trúc khác biệt" so
  Transformer — chúng là Transformer với MỘT SỐ khối MLP được gắn thêm 1 level tính toán in-context.
- **"Kiến trúc sinh context cho optimizer"**: gradient (context của optimizer) do chính kiến trúc
  sinh ra → một optimizer tốt cho kiến trúc này chưa chắc tốt cho kiến trúc khác — gợi ý hướng
  nghiên cứu "optimizer đặc thù theo kiến trúc" thay vì optimizer vạn năng.

#### §7 — Continuum Memory System (CMS) — nguồn trực tiếp của `models/cms.py` + `optim/cms_optimizer.py`
*(giữ nguyên các công thức đã xác nhận, tóm lại)* Eq. 70 (chuỗi MLP lồng nhau), Eq. 71 (cập nhật
theo chu kỳ C^(ℓ), **tổng — không trung bình**), 3 biến thể Nested/Sequential/Independent (Eq. 72-74),
§7.3 Ad-hoc Level Stacking (khởi tạo bằng pretrained — khớp đúng cách `build_cms_param_groups` làm),
Algorithm 1 M3 (đối chiếu từng dòng với `optim/m3.py`, xem bảng ở cuối file).

#### §8 — HOPE: nguồn của `models/hope.py`
*(giữ nguyên)* Self-modifying Titans (Eq. 76-89: k,v,q,η,α đều tự học in-context + **tự sinh giá
trị của chính nó** v̂), chunk-wise training (Eq. 90-93), HOPE forward pass (Eq. 94-97). Đối chiếu
trung thực: `hope.py` dùng Titans **gốc** (thư viện `titans_pytorch`), CHƯA dùng self-modifying
Titans của §8.1.

#### Phụ lục B — Adam, AdaGrad là associative memory (mới bổ sung, suy diễn đầy đủ)
Xuất phát từ chính công thức momentum (Eq. 100), đặt câu hỏi: cách nào tốt nhất để momentum "nhớ"
gradient quá khứ? Định nghĩa objective L2 (Eq. 101): map gradient hiện tại vào một "thuộc tính toàn
cục" P_t của dữ liệu quá khứ (không chỉ map vào hằng số 1 như Hebbian thô):
```
L̃_t = Σ_{i=1}^t ‖m_ℓ,t⊙g_ℓ,i+1 − P_ℓ,t‖² + λ_ℓ‖m_ℓ,t‖²_F                (101)
```
Giải nghiệm tối ưu cho ra công thức có H (tích luỹ bình phương gradient, Eq. 102) đóng vai trò mẫu
số. Chọn **P_t = tổng bình phương gradient** (λ→0) → thu được đúng **GD với momentum thường**
(Eq. 104). Chọn **P_t = căn bậc hai của tổng bình phương gradient** (tức độ lệch chuẩn gần đúng) →
thu được ĐÚNG **công thức Adam** (Eq. 105):
```
W_{i+1} = W_i − (η_t/√β_2)·M̃_i / (H_i^(1/2) + ε)
```
**Kết luận đẹp:** Adam là "associative memory tối ưu cho objective L2 map gradient→phương sai của
chúng". Ghi chú tinh tế: hai moment bậc 1 và bậc 2 của Adam **cùng tần số cập nhật** (mỗi bước) và
**tính độc lập nhau** — một trong số ít ví dụ 2 thành phần "cùng level" mà không phụ thuộc nhau
(khớp định nghĩa "A ⁠f︀= B" ở Definition 2). Mở rộng sang dạng ma trận đầy đủ (outer-product, Eq.
106-111) cho ra **AdaGrad-with-momentum**, và khi β1=1 thì đúng AdaGrad gốc. Cùng lý luận này áp
dụng được cho RMSProp, SignSGD, NAdam, AMSGrad, RAdam, Lion, Shampoo, Soap — tất cả đều quy về
associative memory nén gradient.

#### Phụ lục C — Delta Gradient Descent, suy diễn đầy đủ bằng Sherman-Morrison
Từ công thức GD-là-associative-memory (Eq. 112), thay objective dot-product bằng **L2-regression**
(Eq. 113): `W_{t+1} = argmin_W (1/2)‖Wx_t−u_t‖² + (1/2η_t)‖W−W_t‖²` với u_t=−∇_{y_t}L. Lấy đạo hàm
= 0 (Eq. 114), rút gọn với giả định ‖x_t‖=λ (chuẩn hoá), áp bổ đề Sherman-Morrison để nghịch đảo
(x_t x_tᵀ+η_t I) mà KHÔNG cần nghịch đảo ma trận đầy đủ (Eq. 115-120), ra kết quả cuối gọn đẹp
(Eq. 121):
```
W_{t+1} = W_t·(I − α_t·x_t·x_tᵀ) − β·∇_{y_t}L(W_t,x_t)·x_tᵀ
```
Đây chính là **Delta Rule** — trọng số vừa "quên có chọn lọc theo hướng x_t" (số hạng α_t x_t x_tᵀ)
vừa "ghi phần lệch mới" — khác hẳn Hebbian/linear-attention thuần (chỉ có số hạng ghi, không có số
hạng quên). Đây là công thức TỔNG QUÁT mà cả Delta Momentum (§4, Eq. 48-49) và self-modifying
Titans (§8.1, Eq. 88) đều là trường hợp riêng.

**Câu hỏi tự kiểm tra (B1, quan trọng nhất vì dài nhất):**
- Định nghĩa 2 (Update Frequency) và quan hệ ≻ dùng để làm gì? Cho ví dụ 2 thành phần "cùng level"
  (f_A=f_B) nhưng KHÔNG phụ thuộc nhau trong chính optimizer M3 của dự án (gợi ý: m1 và v trong
  `m3.py` có tính độc lập nhau dù cùng cập nhật mỗi bước không?).
- Liệt kê đúng 5 cơ chế chuyển giao tri thức giữa level (§3.3) và xác định: CMS trong dự án
  (`cms_optimizer.py`) đang dùng cơ chế nào trong 5 cái đó để "tier chậm" và "tier nhanh" liên hệ
  với nhau (hay hoàn toàn KHÔNG có chuyển giao, mỗi tier độc lập)?
- Suy diễn lại Adam từ Eq. 101 bằng lời của bạn: vì sao chọn P_t = độ lệch chuẩn gradient (thay vì
  tổng bình phương thô) lại cho ra đúng công thức Adam quen thuộc?
- Delta Rule (Eq. 121, Phụ lục C) có 2 số hạng: "quên có hướng" và "ghi phần lệch". `m3.py` bản
  rút gọn (`M ← α·M + η·(g−M)`) giữ lại phần nào, bỏ phần nào so với công thức đầy đủ này?
- "Illusion of deep learning architectures" (§6) — dùng đúng luận điểm này để trả lời câu hỏi:
  Titans/CMS/HOPE có phải "3 kiến trúc khác nhau" hay chỉ là "cùng 1 ý tưởng NL, khác ở TẦNG nào
  của model được gắn thêm level"?
- HOPE dùng Titans gốc (A1) hay self-modifying Titans (§8.1)? Nêu 1 hệ quả thực nghiệm cụ thể.

### B2. Muon: An optimizer for hidden layers in neural networks
Keller Jordan et al., blog 12/2024 · https://kellerjordan.github.io/posts/muon/ ·
repo https://github.com/KellerJordan/Muon

**Nội dung chính, chi tiết hơn.** Muon = SGD-momentum + hậu xử lý Newton-Schulz (NS) để trực giao
hoá gần đúng update — đưa nó về gần UVᵀ (U,V từ SVD), bỏ độ lớn singular value, chỉ giữ "hướng":
```python
def newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + eps)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a*X + (b*A + c*A@A) @ X
    if G.size(0) > G.size(1): X = X.T
    return X
```
**Vì sao 3 hệ số đó, không phải số khác:** đặt φ(x)=ax+bx³+cx⁵ (đa thức bậc 5 áp lên từng singular
value), điều kiện cần: (1) a càng lớn càng tốt (a=φ'(0) quyết định tốc độ hội tụ ở singular value
nhỏ — thường bị "bỏ quên" nếu a nhỏ); (2) φᴺ(x)→[1−ε,1+ε] khi N→∞ với mọi x∈[0,1]. Thực nghiệm cho
thấy ε có thể tới ±30% mà không hại loss — nên bài toán trở thành "tối đa a với ràng buộc lỏng" →
giải bằng gradient descent thô sơ ra đúng (3.4445, −4.7750, 2.0315), hội tụ chỉ sau 5 bước lặp.

**Quan hệ với Shampoo (đáng chú ý, hay bị bỏ qua):** nếu bỏ phần tích luỹ preconditioner của Shampoo,
update trở thành `W_{t+1}=W_t−η(G_tGᵀ_t)^{-1/4}G_t(GᵀG_t)^{-1/4}` — khai triển SVD ra đúng
`W_t−η·UVᵀ`, tức **chính là gradient đã trực giao hoá**. Vậy Muon (bỏ momentum) ≈ "Shampoo tức thời,
không tích luỹ" — nhưng Muon rẻ hơn nhiều vì dùng Newton-Schulz thay vì nghịch đảo căn bậc 4 (đòi
hỏi ≥float32, chậm trên GPU hiện đại; NS chạy ổn định ở bfloat16).

**Quy ước quan trọng nhất cho dự án:** Muon CHỈ áp cho tham số 2D; vector/bias/LayerNorm VÀ layer
input/output (embedding, classifier head) phải dùng AdamW — tác giả nhấn mạnh dựa trên lý thuyết
modular norm (Large et al. 2024) rằng embedding/output layer có "động lực học" khác biệt cần chuẩn
hoá riêng.

**Map sang code dự án:** `m3.py` dòng `if p.ndim >= 2: update = newton_schulz(...) else: update =
st["m1"]` cài đúng quy ước — comment ghi rõ áp nguyên M3 lên tham số 1D gây limit-cycle (bug đã sửa,
`TIEN_DO_2026-07-17.md`). Có nguồn lý thuyết chính thức, không chỉ là vá bug.

**Câu hỏi tự kiểm tra:**
- Vì sao trực giao hoá lại giúp ích (condition number cao/gần low-rank của update ma trận)?
- Muon quan hệ với Shampoo thế nào — Muon KHÔNG có momentum thì tương đương gì với Shampoo?
- Hệ số Newton-Schulz được chọn theo tiêu chí gì? Tăng `ns_steps` (T trong M3) có cần đổi hệ số?

### B3. Linear Transformers Are Secretly Fast Weight Programmers
Schlag, Irie, Schmidhuber — ICML 2021 · arXiv:2102.11174 · https://arxiv.org/abs/2102.11174

**Nội dung chính, chi tiết hơn.** Chứng minh linear attention (softmax→kernel φ) tương đương hình
thức với **fast weight programmers** (Schmidhuber, đầu 1990s): một mạng "chậm" học bằng GD cách lập
trình trọng số "nhanh" của mạng khác qua chuỗi phép cộng tích ngoài (outer product) của các
activation pattern tự sinh (ngày nay gọi là key/value). Hai đóng góp kỹ thuật chính:
- **Delta rule cho fast weight:** thay vì cộng dồn thuần tuý M_t=M_{t-1}+v_tk_tᵀ (dễ tràn dung
  lượng khi chuỗi dài — key trùng nhau thì giá trị cũ/mới CỘNG LẠI thay vì thay thế), delta rule
  trước tiên "đọc" giá trị cũ ứng với key hiện tại, trừ nó ra, rồi mới ghi giá trị mới vào — tức
  **sửa/ghi đè** thay vì chỉ "insert" — giống thao tác UPDATE trong bảng băm thay vì chỉ INSERT.
  Cơ chế này tăng dung lượng nhớ hiệu quả đáng kể so với cộng dồn thuần khi số cặp key-value vượt
  quá dung lượng O(N) của ma trận.
- **Kernel DPFP (Deterministic Parameter-Free Projection):** một feature map φ(·) mở rộng chiều
  biểu diễn của key/query (tăng khả năng phân biệt các key gần nhau) mà KHÔNG cần thêm tham số học
  được và KHÔNG cần random features kiểu FAVOR+ (Performer) — rẻ và tất định (deterministic), khác
  hẳn cách random-feature-map phổ biến trước đó.

**Vì sao liên quan:** đây là **cầu nối lịch sử 30 năm** giữa fast weight programmers (Schmidhuber
1990s) → linear attention (Katharopoulos 2020) → delta rule (bài này, 2021) → DeltaNet có forget
gate (2024) → Titans/NL (2025). Đọc xong sẽ thấy "surprise + forget gate" của A1 và "Delta Momentum"
của B1 không phải ý tưởng cô lập mà là điểm mới nhất trên một trục liên tục: cộng dồn thuần tuý
(Hebbian) → có sửa/xoá (delta rule) → có tốc độ ghi học được (Titans) → tốc độ ghi VÀ giá trị đều
tự sinh (self-modifying Titans, §8.1 NL.pdf).

**Câu hỏi tự kiểm tra:**
- Delta rule khác Hebbian rule (cộng dồn thuần) ở điểm nào cụ thể — "đọc trước khi ghi" nghĩa là gì
  về mặt công thức?
- DPFP giải quyết vấn đề gì của linear attention (gợi ý: liên hệ đến khả năng PHÂN BIỆT các key gần
  nhau khi kernel tuyến tính hoá quá đơn giản)?
- "Fast weight" (Schmidhuber) và "state" trong `TitansMemory`/`state_utils.py` của dự án cùng một
  khái niệm chứ? Nêu 1 điểm giống và 1 điểm khác.

### B4. Learning to (Learn at Test Time): RNNs with Expressive Hidden States (TTT)
Sun, Li, Dalal, et al. — arXiv:2407.04620 · https://arxiv.org/abs/2407.04620 (tuỳ chọn)

**Nội dung chính, chi tiết hơn.** Biến hidden state của RNN thành CHÍNH một mô hình học máy (tuyến
tính cho TTT-Linear, MLP 2 lớp cho TTT-MLP). Luật cập nhật state = một bước **học tự giám sát**:
tại mỗi token, tạo ra "training view" (một phép chiếu hạng thấp/bị nhiễu của x_t qua θ_K) và "label
view" (một phép chiếu hạng thấp khác của x_t qua θ_V), rồi state (=trọng số của mô hình nhỏ bên
trong) học cách khử nhiễu training-view để ra label-view — đúng là một bài toán tái tạo
(reconstruction) tự giám sát, giải bằng 1 (hoặc vài) bước GD ngay trong forward pass, y hệt tinh
thần "test-time training". Điểm kỹ thuật quan trọng để chạy nhanh trên phần cứng thật: **dạng đối
ngẫu (dual form)** cho phép tính bước GD của toàn bộ mini-batch bằng phép nhân ma trận thay vì lặp
tuần tự — cùng ý tưởng song song hoá mà A1 Titans mượn lại ở §3.2 (paper A1 trích dẫn thẳng kỹ thuật
này của Sun et al. 2024).

**Vì sao liên quan:** TTT và Titans ra đời gần như song song (giữa 2024), cùng khai thác "test-time
training" nhưng khác khung nhìn (TTT nhấn "hidden state = chính một model", Titans nhấn "surprise +
forget gate"). Trong bảng thực nghiệm của chính A1 (Bảng 1 mục A1), TTT và Titans là 2 baseline
cạnh nhau — Titans thắng vì thêm momentum + weight decay mà TTT (bản gốc) không nhấn mạnh; NL.pdf
cũng trích Sun et al. 2024 cho kỹ thuật chunk-wise song song hoá ở cả §3.2 (A1 dùng) và §8.2 (HOPE
dùng). Đọc để Related Work không chỉ nói một mình Titans mà đặt đúng trong nhóm test-time-learning.

**Câu hỏi tự kiểm tra:**
- TTT và Titans giống nhau ở điểm cốt lõi nào (cả hai biến "cập nhật state" thành bài toán tối ưu,
  không phải công thức đóng)? Khác nhau ở chỗ nào (gợi ý: TTT có momentum/forget gate tường minh
  như Titans không)?
- "Dual form" của TTT dùng để làm gì — vì sao một bước GD tưởng tuần tự lại tính song song được?
- Nếu chỉ được trích 1 câu để so sánh TTT vs Titans trong Related Work, câu đó nên nói gì?

---

## Nhóm C — PHỤC VỤ VIẾT BÁO CÁO (cite đối thủ + metric chuẩn)

### C1. GEM: Gradient Episodic Memory for Continual Learning
Lopez-Paz, Ranzato — NeurIPS 2017 · arXiv:1706.08840

**Nội dung chính, chi tiết hơn.** Ma trận **R (T×T)**: R[i,j] = accuracy trên task j sau khi vừa học
xong task i — chính là `acc_matrix` mà `metrics/continual.py` dùng. Từ R: **Average Accuracy**
(trung bình hàng cuối), **BWT** (trung bình R[-1,j]−R[j,j], âm=quên), **FWT** (R[j-1,j]−đoán mò, đo
TRƯỚC khi học task j).

**Thuật toán GEM (không chỉ metric) — công thức chính xác:** tại mỗi bước, thay vì áp trực tiếp
gradient g của task hiện tại, GEM giải một **bài toán QP** (quadratic program) để tìm gradient
g̃ gần g nhất nhưng KHÔNG làm tăng loss trên các task cũ (ràng buộc góc ⟨g̃,g_k⟩≥0 với mọi task cũ
k, dùng gradient trên episodic memory làm đại diện cho "loss task cũ"):
```
minimize   (1/2)·vᵀ·G·Gᵀ·v − gᵀ·Gᵀ·v     subject to   v ≥ 0
g̃ = Gᵀ·v* + g
```
(G = ma trận xếp các gradient tham chiếu của task cũ theo hàng). Giải QP này cho v* (t−1 biến, t=số
task đã thấy) rồi suy ra g̃ — **cho phép BWT dương** (học task mới còn giúp nhớ task cũ tốt hơn),
khác hẳn các phương pháp chỉ NGĂN quên (BWT≤0). Bản nhẹ hơn sau này (A-GEM) thay ràng buộc riêng
từng task bằng MỘT ràng buộc trung bình, rẻ hơn nhiều nhưng mất khả năng BWT dương.

**Map sang code:** `average_accuracy`, `backward_transfer`, `forward_transfer` trong
`metrics/continual.py` cài đúng định nghĩa GEM. Lưu ý: dự án **không cài thuật toán GEM** (không có
QP nào trong `methods.py`), chỉ mượn 3 metric — nên trong báo cáo, GEM chỉ nên được cite cho phần
"nguồn gốc metric", không nên ngầm hiểu là "một trong các phương pháp được so sánh".

**Câu hỏi tự kiểm tra:**
- BWT âm và Forgetting dương đo cùng hiện tượng hay khác? Có thể BWT âm nhưng Forgetting=0 không?
- Ràng buộc QP của GEM (⟨g̃,g_k⟩≥0) đảm bảo điều gì về mặt hình học (góc giữa 2 vector gradient)?
- FWT trong dự án chỉ "có ý nghĩa" khi nào (`train.eval_future`)? Vì sao NCM/EWC/replay hiếm khi
  cải thiện FWT nhưng Titans/HOPE có cơ hội (liên hệ tới việc GEM cho phép BWT dương bằng ràng buộc,
  còn Titans/HOPE có thể cho FWT dương bằng CƠ CHẾ nhớ xuyên task, không cần ràng buộc tường minh)?

### C2. Three scenarios for continual learning
van de Ven, Tolias — arXiv:1904.07734 (2019)

**Nội dung chính, chi tiết hơn.** 3 kịch bản theo (a) task-ID có biết lúc test không, (b) nếu không,
có cần tự suy luận ra không: **Task-incremental** (biết task-ID, có thể dùng head riêng từng task),
**Domain-incremental** (không biết task-ID, không cần suy luận vì không gian nhãn giống nhau qua
các task), **Class-incremental** (không biết task-ID, PHẢI tự suy luận, mỗi task thêm class mới,
lúc test phân biệt giữa TẤT CẢ class đã học). Thực nghiệm trên Split/Permuted MNIST cho thấy nhiều
thuật toán tốt ở task-incremental sụp hoàn toàn ở class-incremental.

**Phát hiện đáng chú ý nhất (rất liên quan dự án):** khi task-ID phải tự suy luận (class-incremental),
các phương pháp **regularization-based** (điển hình: EWC) **THẤT BẠI** — gần như không hơn gì
finetune trần trụi; các phương pháp dựa trên **replay** (lưu hoặc sinh lại dữ liệu cũ) mới thực sự
cần thiết để có kết quả chấp nhận được. Đây là phát hiện được trích dẫn rất nhiều trong literature
continual learning sau này.

**Vì sao liên quan trực tiếp:** `RA_SOAT_NL_2026-07-17.md` mục A.1 xác nhận giao thức
class-incremental của dự án là đúng. Phát hiện "EWC thất bại ở class-incremental" ở trên **giải
thích trước** một kết quả nhiều khả năng sẽ thấy trong bảng G1 của dự án (EWC thua xa Replay) — nếu
đúng vậy thì đây KHÔNG phải bug hay cấu hình sai, mà là hiện tượng đã được ghi nhận trong literature.

**Câu hỏi tự kiểm tra:**
- Giải thích vì sao dự án đang làm class-incremental chứ không phải task-incremental, dùng đúng
  ngôn ngữ của bài (task-ID có được cho biết lúc test không?).
- Nếu bảng G1 của dự án cho thấy EWC thua xa Replay, đây có phải "kết quả xấu cần debug" không, hay
  là hiện tượng ĐÃ ĐƯỢC BÁO TRƯỚC bởi chính paper này? Nên viết gì trong báo cáo nếu gặp trường hợp
  này?
- Nếu đổi sang task-incremental, điều gì trong `engine.py`/`mask_logits` sẽ đổi? Baseline nào
  hưởng lợi nhiều nhất?

### C3. iCaRL: Incremental Classifier and Representation Learning
Rebuffi, Kolesnikov, Sperl, Lampert — CVPR 2017

**Nội dung chính, chi tiết hơn.** Hai ý tưởng ghép lại:

1. **Herding** — công thức chọn mẫu chính xác, chọn LẦN LƯỢT từng ảnh để thêm vào tập exemplar sao
   cho trung bình feature của tập đã chọn CÀNG GẦN trung bình feature của cả class càng tốt:
   ```
   p_k = argmin_x ‖μ_c − (1/k)·[φ(x) + Σ_{j=1}^{k-1} φ(p_j)]‖₂
   ```
   (μ_c = trung bình feature của TOÀN BỘ ảnh class c, φ(·) = feature extractor). Chọn tuần tự k=1..m
   — mỗi bước thêm đúng 1 ảnh làm cho trung bình của tập con hiện tại khớp μ_c nhất.
2. **Nearest-Mean-of-Exemplars (NCM):** phân loại bằng khoảng cách tới trung bình feature của các
   exemplar mỗi class (không dùng FC) — vì FC bị lệch (bias) về phía class mới học gần đây, còn so
   khoảng cách tới prototype thì ổn định hơn qua thời gian (không có tham số riêng bị train lệch).

Có thêm distillation loss (tinh thần giống LwF, C5) để giữ representation ổn định khi học class mới.

**Phát hiện thú vị đáng đưa vào báo cáo:** nghiên cứu sau này chỉ ra **không có khác biệt đáng kể**
giữa chọn exemplar bằng herding và chọn NGẪU NHIÊN — herding phức tạp hơn nhưng lợi ích thực nghiệm
khá mờ nhạt so với kỳ vọng ban đầu của chính iCaRL. Đây là bằng chứng khá vững cho việc dự án chọn
**random sampling** cho buffer của `Replay` — không chỉ là "đơn giản hoá cho nhanh" mà có cơ sở
thực nghiệm ủng hộ (đỡ phải bảo vệ lựa chọn này như một thiếu sót).

**Map sang code:** dự án có `Replay` (`methods.py`, buffer random theo quota mỗi class) VÀ `NCM`
(`models/ncm.py`, prototype trung bình mỗi class) như 2 baseline **tách biệt** — trong khi iCaRL là
nơi CẢ HAI ý tưởng xuất hiện cùng nhau lần đầu trong 1 hệ thống. Khi viết báo cáo, cite iCaRL cho cả
hai baseline.

**Câu hỏi tự kiểm tra:**
- Viết lại công thức herding bằng lời: vì sao chọn TUẦN TỰ (k=1,2,...) thay vì chọn một lần m ảnh
  tối ưu đồng thời?
- NCM trong dự án (`models/ncm.py`) dùng đúng feature space mà iCaRL đề xuất không?
- Vì sao FC-classifier "thiên vị" class mới học gần đây? Phát hiện "herding ≈ random" có ảnh hưởng
  gì đến việc có nên đầu tư thời gian cải tiến cách chọn buffer của `Replay` hay không?

### C4. Overcoming catastrophic forgetting in neural networks (EWC)
Kirkpatrick et al. — PNAS 2017 · https://www.pnas.org/doi/pdf/10.1073/pnas.1611835114

**Nội dung chính, chi tiết hơn.** Sau khi học xong task cũ, ước lượng **Fisher Information chéo**
F_i cho từng tham số θ_i — về bản chất là kỳ vọng bình phương gradient của log-likelihood, xấp xỉ
độ cong (đường chéo Hessian) của loss quanh điểm tối ưu θ*: F_i ≈ E[(∂log p(y|x,θ)/∂θ_i)²]. Khi học
task mới, thêm penalty: L = L_mới + (λ/2)·Σ F_i·(θ_i−θ*_i)². Về mặt Bayes: đây là xấp xỉ Laplace
(dùng đường chéo Hessian, bỏ off-diagonal) cho posterior của tham số sau khi thấy task cũ, dùng làm
prior khi học task mới — nghĩa là EWC ngầm giả định posterior là Gauss, và các chiều tham số độc
lập nhau (bỏ tương quan chéo) — 2 giả định khá mạnh, là nguồn gốc của nhiều phê bình sau này.

**Vấn đề "đếm hai lần" (double-counting) khi có >2 task — đáng đưa vào báo cáo:** cách xử lý nhiều
task cũ có 2 trường phái: (a) **cộng dồn riêng từng anchor** (mỗi task một cặp (θ*,F) riêng, cộng
tất cả penalty lại — đúng cách Kirkpatrick et al. mô tả ban đầu), hoặc (b) **gộp đệ quy** thành MỘT
penalty bậc hai duy nhất, neo tại điểm tối ưu GẦN NHẤT với độ cong TÍCH LUỸ (cách Huszár 2017 phê
bình và đề xuất sửa — tránh đếm hai lần thông tin prior khi các Fisher của nhiều task chồng lấn lên
cùng một hướng tham số).

**Map sang code:** lớp `EWC` trong `methods.py` cài theo đúng cách (a) — **cộng dồn riêng từng
anchor** (`self._anchors: List[...]`, mỗi task một Fisher+params riêng) — đây CHÍNH XÁC là biến thể
bị Huszár phê bình có nguy cơ "đếm hai lần" thông tin prior khi số task lớn, và cũng là nguyên nhân
trực tiếp khiến `footprint_floats` phình tuyến tính theo số task (P tham số × 2 × số task cũ).

**Câu hỏi tự kiểm tra:**
- Fisher Information xấp xỉ điều gì (liên hệ độ cong loss landscape/Hessian)? Vì sao dùng bình
  phương gradient thay vì tính Hessian thật (chi phí)?
- EWC ngầm giả định gì về posterior (phân phối Gauss, tham số độc lập)? Giả định nào trong 2 cái đó
  "nguy hiểm" hơn khi áp dụng cho ViT (hàng triệu tham số, chắc chắn có tương quan)?
- "Đếm hai lần" (Huszár) là vấn đề gì cụ thể? `methods.py::EWC` đang cài theo cách (a) hay (b)? Nếu
  chạy ≥3 task, nhược điểm này có ảnh hưởng đến số liệu Forgetting đo được không, hay chỉ ảnh hưởng
  chi phí bộ nhớ?

### C5. Learning without Forgetting (LwF)
Li, Hoiem — ECCV 2016 / TPAMI 2017 · arXiv:1606.09282

**Nội dung chính, chi tiết hơn.** KHÔNG lưu dữ liệu cũ. Ký hiệu gốc của paper: θ_s (tham số CNN
dùng chung mọi task), θ_o (tham số riêng học từ task cũ), θ_n (tham số riêng học ở task mới). Trước
khi học task mới: (1) ghi lại phản hồi (response) của mạng GỐC trên các đầu ra cũ cho dữ liệu task
mới (dùng làm "nhãn mềm"); (2) thêm node mới vào lớp đầu ra (khởi tạo ngẫu nhiên) cho class/task
mới. Loss: L = CE(task mới) + λ·T²·KL(student/T ‖ teacher/T) trên các cột class cũ.

**Chi tiết quy trình huấn luyện hay bị bỏ sót:** với kiến trúc multi-head, paper phát hiện **huấn
luyện RIÊNG cái đầu (head) mới trong vài epoch đầu** (trước khi mở khoá train toàn mạng) giúp giảm
đáng kể sụt giảm hiệu năng trên task cũ so với train toàn mạng ngay từ đầu — một dạng "warm-up" để
tránh đồng thích nghi (co-adaptation) có hại giữa head mới khởi tạo ngẫu nhiên và phần thân mạng đã
học tốt.

**Vì sao là "tổ tiên chung" với C3:** cùng dùng cơ chế distillation teacher/student để giữ ổn định
representation — khác iCaRL ở chỗ LwF hoàn toàn không cần lưu ảnh cũ.

**Map sang code:** lớp `LwF` trong `methods.py` khớp công thức chính — `begin_task` chụp teacher
(chỉ khi đã có class cũ), `extra_batch_loss` tính đúng KL có nhiệt độ. **Không thấy bước "warm-up
head" trong code** — đây có thể là một phần lý do khiến LwF "bất thường trên EuroSAT (F 0.609)" mà
`DANH_SACH_PAPER.md` gốc ghi nhận; nghi vấn hợp lý thứ hai vẫn là giá trị λ (paper gốc không cho λ
cố định, cần tune theo domain — domain ảnh vệ tinh khác xa domain ảnh vật thể LwF gốc thử nghiệm).

**Câu hỏi tự kiểm tra:**
- Vì sao LwF không cần lưu ảnh cũ mà vẫn "nhớ" được — vai trò của θ_s (dùng chung) so với θ_o/θ_n?
- "Warm-up head" giải quyết vấn đề gì (đồng thích nghi có hại)? `methods.py::LwF` có bước này
  không? Nếu thêm vào, dự đoán điều gì xảy ra với F=0.609 bất thường trên EuroSAT?
- LwF và EWC cùng regularization-based (không cần buffer) — khác nhau ở ĐỐI TƯỢNG bị ràng buộc
  (θ trực tiếp vs. đầu ra logits) — điều này liên hệ thế nào đến phát hiện của C2 rằng
  regularization-based thất bại ở class-incremental (LwF có bị đúng nhược điểm y hệt EWC không, hay
  khác vì ràng buộc ở logits thay vì tham số)?

---

## Nhóm D — "Paper bằng code" (đọc kèm B1)

### D1. github.com/obekt/HOPE-nested-learning
Repo Python từ-đầu (~154M tham số, huấn luyện trên Wikipedia + Q&A), không phải thư viện production
mà là bản demo dễ đọc. `CONFIG["cms_tiers"] = [[8,1],[5,4],[3,16]]` — đúng định dạng `[[số layer,
chu kỳ]]`. README mô tả CMS trực quan: "Fast Weights = layer tự sửa tham số real-time khi đọc text;
Slow Weights = layer sâu ít cập nhật, lưu tri thức dài hạn không bị quên thảm khốc". Credit đúng
nguồn `NL.pdf`.

**So với dự án:** `cms_tiers` của obekt áp cho **toàn bộ** layer (kể cả attention) của một LM tự
huấn luyện từ đầu; dự án chỉ áp cho **MLP block của ViT pretrained** (retrofit) — khác quy mô nhưng
cùng công thức Eq. 71.

### D2. github.com/kmccleary3301/nested_learning
Repo nghiêm túc hơn hẳn D1 — có CI, test suite, và `docs/PAPER_COMPLIANCE.md` (đã đọc toàn văn) là
bản đối chiếu phương trình→code giống hệt tinh thần `LOGIC_NESTED_LEARNING.md` của dự án, ở quy mô
LLM đầy đủ. Điểm đối chiếu được ngay:
- Phân biệt rõ "outer update" (optimizer.step() chuẩn) và "inner update" (cập nhật memory/CMS ngoài
  đồ thị gradient, dùng "teach signal" δ) — đúng khái niệm δ_ℓ ở Eq. 29-31 NL.pdf, và đúng tinh
  thần "TBPTT + detach mỗi batch" mà `state_utils.py` của dự án cài.
- Pin cứng nguồn tham chiếu bằng SHA-256 của file paper — thói quen đáng học.
- Có optimizer M3 y hệt tên gọi — xác nhận thêm "M3" đúng là tên paper dùng (Algorithm 1).
- "Surprise metric" của họ có 3 lựa chọn (`l2`/`loss`/`logit_entropy`) — dự án hiện không có khái
  niệm "surprise threshold" tường minh (memory luôn cập nhật, không gate theo surprise).
- README ghi "Self-modifying Titans (Eqs. 83–93; Eq. 91 residual MLP memories)" — **trùng khớp
  chính xác** với số phương trình đọc trực tiếp trong §8.1 NL.pdf (cross-check 2 nguồn độc lập).

**Câu hỏi tự kiểm tra (D1+D2):**
- `cms_tiers` của D1 áp cho toàn bộ layer, dự án chỉ áp MLP-block — ảnh hưởng gì đến tỷ lệ tham số
  "được bảo vệ" (tier chậm) so với tổng tham số model?
- "teach signal" (D2) và "surprise/δ_ℓ" (B1, A1) có phải cùng khái niệm dưới tên khác? Nêu bằng
  chứng từ chính văn bản đã đọc.
- Dự án hiện có gate cập nhật Titans theo ngưỡng surprise không? Nếu thêm vào, đổi gì trong log
  `[titans] norm(state)`?

---

## Bảng tổng hợp: paper/nguồn → phần code tương ứng

| # | Nguồn | Phần lý thuyết chính | File code dự án | Log/bằng chứng liên quan |
|---|---|---|---|---|
| A1 | Titans (2501.00663) | surprise Eq.8-10, forget gate Eq.13-14, MAC/MAG/MAL Eq.21-31 | `models/memory.py`, `models/titans_head.py` | `[titans] norm(state)` |
| A2 | Geva et al. 2021 | MLP = key-value memory, FFN=Softmax(W_K x)·W_V | `models/cms.py` (chọn đúng MLP để chia tier) | `[cms]` tier report |
| B1 §3 | NL.pdf | Def 1 Associative Memory, Def 2-4 frequency/NSAM, 5 cơ chế chuyển giao | nền tảng khái niệm cho toàn dự án | — |
| B1 §4 | NL.pdf | optimizer=associative memory, Delta Momentum Eq.48–49, Muon Eq.42–44 | `optim/m3.py` | log `‖Δw‖`, hội tụ M3 |
| B1 §6 | NL.pdf | "illusion of architectures", ICL không emergent | định hướng cách viết chương Related Work/Discussion | — |
| B1 §7 | NL.pdf | CMS Eq.70–71, 3 biến thể Eq.72–74, §7.3 pretrained init | `models/cms.py`, `optim/cms_optimizer.py` | `[cms] ‖Δw‖ per-tier` |
| B1 §8 | NL.pdf | self-modifying Titans Eq.83–93, HOPE Eq.94–97 | `models/hope.py` | `[hope] norm(state)` |
| B1 App.B | NL.pdf | Adam/AdaGrad suy từ associative memory Eq.100-111 | giải thích lý thuyết cho AdamW baseline | — |
| B1 App.C | NL.pdf | Delta GD suy diễn đầy đủ (Sherman-Morrison) Eq.112-121 | nền tảng toán cho Delta Momentum trong `m3.py` | — |
| B2 | Muon blog | Newton-Schulz, quan hệ Shampoo, quy ước 1D→Adam | `optim/m3.py::newton_schulz` | bug limit-cycle (đã sửa) |
| B3 | Schlag et al. 2021 | delta rule, DPFP kernel, fast weight programmers | nền tảng lịch sử cho A1/B1 | — |
| B4 | TTT (Sun et al. 2024) | hidden state = model, dual form song song hoá | song song A1 (A1 mượn kỹ thuật song song hoá) | — |
| C1 | GEM 2017 | ma trận R, Avg Acc/BWT/FWT, QP gradient projection | `metrics/continual.py` | bảng so sánh 5 baseline |
| C2 | van de Ven & Tolias 2019 | task/domain/class-incremental, EWC thất bại ở class-incr. | `models/classifier.py::mask_logits`, `engine.py` | giao thức eval |
| C3 | iCaRL 2017 | herding (nhưng ≈ random), NCM | `methods.py::Replay`, `models/ncm.py` | 2 baseline |
| C4 | EWC 2017 | Fisher information, double-counting (Huszár) | `methods.py::EWC` | `footprint_floats` phình theo task |
| C5 | LwF 2016 | distillation + warm-up head | `methods.py::LwF` | F=0.609 bất thường trên EuroSAT |
| D1/D2 | 2 repo GitHub | đối chiếu phương trình ↔ code ở quy mô LLM | tham khảo, không map trực tiếp | `docs/PAPER_COMPLIANCE.md` (D2) |

## Cách dùng file này

Đọc theo đúng thứ tự nhóm A→B→C→D. Sau mỗi mục, quay lại "Ghi chú sau khi đọc" ở cuối
`DANH_SACH_PAPER.md`, điền 3 dòng. Câu hỏi tự kiểm tra không cần trả lời bằng văn bản; nếu giải
thích miệng được cho người khác trong dưới 1-2 phút mỗi câu, coi như đã "hiểu rõ".
