# Đọc code — Bài 2: Trái tim (engine + methods + runner)

> Đọc sau khi xong **Bài 1**. Ba file trong bài này chiếm 1.326 dòng nhưng chỉ có **8 hàm** thật sự cần hiểu.

**Thứ tự đọc trong bài này:**

| # | Hàm | File | Vì sao đọc thứ tự này |
|---|---|---|---|
| 1 | `evaluate` | `engine.py` | ngắn nhất, dùng lại `mask_logits` vừa học ở Bài 1 |
| 2 | `train_one_task` | `engine.py` | vòng train một task — thấy 5 hook được gọi ở đâu |
| 3 | `run_continual` | `engine.py` | ⭐ vòng ngoài, sinh ma trận R |
| 4 | `FineTune` (5 hook) | `methods.py` | lớp cha, hiểu 5 hook là hiểu mọi method |
| 5 | `Replay` | `methods.py` | method dễ hiểu nhất, mạnh nhất |
| 6 | `LwF` | `methods.py` | dùng hook `begin_task` — bổ sung góc nhìn |
| 7 | `EWC` | `methods.py` | dùng hook `penalty` — hook cuối chưa gặp |
| 8 | `main()` | `scripts/run_g1.py` | nối tất cả lại |

---

# PHẦN A — `engine.py` (358 dòng)

**Nhiệm vụ:** vòng lặp học liên tục dùng chung cho **cả dự án**. G1 dựng ra, G2–G4 tái sử dụng không sửa.

**Giao thức** (ghi ngay docstring đầu file):

```
for t in tasks:
    train model trên task t   (logits mask về class của task t; + penalty của method)
    method.end_task(...)
    for j in 0..t:
        R[t, j] = accuracy trên test của task j  (logits mask về class ĐÃ THẤY)
```

---

## A1. `resolve_device(pref="auto")`

```python
pref = (pref or "auto").lower()
if pref != "auto":
    return torch.device(pref)               # người dùng ép sẵn -> tôn trọng
if torch.cuda.is_available():   return torch.device("cuda")
mps = getattr(torch.backends, "mps", None)
if mps is not None and mps.is_available(): return torch.device("mps")
return torch.device("cpu")
```

Giống `utils.get_device()` nhưng trả `torch.device` thay vì chuỗi, và **cho phép ép** bằng `device: cpu` trong yaml. Ép CPU hữu ích khi debug (lỗi CUDA hay báo ở chỗ sai).

---

## A2. `evaluate()` ⭐ HÀM ĐẦU TIÊN NÊN ĐỌC TRONG BÀI NÀY

```python
@torch.no_grad()
def evaluate(model, loader, device, allowed: Sequence[int]) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = mask_logits(model(x), allowed).argmax(dim=1)
        correct += int((pred == y).sum())
        total   += int(y.numel())
    return correct / max(total, 1)
```

**8 dòng, và là hàm sinh ra mọi con số trong dự án.** Mỗi ô của ma trận R là một lần gọi hàm này.

**Bốn chi tiết:**

1. **`@torch.no_grad()`** — decorator tắt việc dựng đồ thị tính đạo hàm. Tiết kiệm RAM và nhanh hơn nhiều. Quên nó thì eval trên RESISC45 có thể OOM.
2. **`model.eval()`** — tắt dropout, và quan trọng hơn với dự án này: **Titans không ghi ký ức ở chế độ eval**. Chấm điểm mà vô tình sửa bộ nhớ là kết quả vô nghĩa.
3. **`mask_logits(..., allowed)`** — chấm điểm chỉ trên lớp được phép. Không mask thì model có thể đoán ra lớp chưa từng học và accuracy bị bóp méo.
4. **`max(total, 1)`** — tránh chia 0 khi loader rỗng.

**Ai gọi:** `run_continual`, mỗi task gọi `t+1` lần.

---

## A3. `train_one_task()` — vòng train một task

```python
def train_one_task(model, method, loader, device, allowed, train_cfg, opt=None):
```

**Trả:** `(losses_từng_epoch, optimizer_đã_dùng)`.

**Phần 1 — chọn optimizer:**

```python
epochs = int(train_cfg.get("epochs_per_task", 3))
from .optim import build_optimizer
if opt is None:
    if (train_cfg.get("cms") or {}).get("enabled", False):
        from .optim.cms_optimizer import build_cms_optimizer
        opt = build_cms_optimizer(model, train_cfg)        # G3/G4: đa tần số
    else:
        opt = build_optimizer(model.parameters(), train_cfg)  # G1/G2: adamw | m3
```

**Tham số `opt` là một quyết định thiết kế đáng chú ý.** Docstring:

> `opt=None` → tạo optimizer **mới** cho task này (mặc định — không mang moment cũ sang).
> Truyền `opt` có sẵn → **ký ức gradient** (M1/M2/V của M3) và pha chu kỳ CMS sống **xuyên task** — đúng tinh thần NL "optimizer cũng là bộ nhớ dài hạn". Bật qua `train.optimizer_per_task: false`.

Nghĩa là: momentum của Adam **cũng là một dạng trí nhớ**. Mặc định dự án vứt nó đi giữa các task (chuẩn ngành). Cờ này cho phép giữ lại và đo xem có lợi không.

**Phần 2 — vòng train:**

```python
method.begin_task(model, device, allowed)      # ← HOOK 1 (LwF chụp teacher tại đây)
losses = []
model.train()
for ep in range(epochs):
    run, seen = 0.0, 0
    bar = tqdm(loader, desc=f"  epoch {ep+1}/{epochs}", leave=False)
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        logits_full = model(x)                              # điểm cho MỌI lớp
        logits = mask_logits(logits_full, allowed)          # che lớp ngoài task
        loss = F.cross_entropy(logits, y)                   # loss chính

        pen = method.penalty(model)                         # ← HOOK 2 (EWC)
        if pen is not None:   loss = loss + pen

        extra = method.extra_batch_loss(model, x, logits_full, device)   # ← HOOK 3
        if extra is not None: loss = loss + extra

        opt.zero_grad(set_to_none=True)
        loss.backward()

        grad_clip = train_cfg.get("grad_clip_norm")         # None -> bỏ qua, hành vi cũ
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=float(grad_clip), error_if_nonfinite=True)

        opt.step()
        run  += float(loss.detach()) * y.numel()
        seen += int(y.numel())
        bar.set_postfix(loss=f"{run/max(seen,1):.3f}")
    losses.append(run / max(seen, 1))
return losses, opt
```

**Đây là chỗ 3 trong 5 hook được gọi.** Đọc kỹ đoạn này là hiểu toàn bộ cơ chế method.

**Chi tiết đáng nhớ:**

- **`logits_full` được truyền vào `extra_batch_loss`, không phải `logits` đã mask.** Vì LwF cần logits trên **lớp cũ** — những lớp bị mask đi trong loss chính. Nếu truyền bản đã mask thì LwF sẽ distill từ toàn `-1e4`.
- **`run += loss * y.numel()`** rồi chia `seen` cuối epoch — cách tính loss trung bình đúng khi các batch **khác kích thước** (batch cuối thường thiếu vì `drop_last=False`).
- **`set_to_none=True`** trong `zero_grad` — nhanh hơn và tiết kiệm RAM hơn gán 0.
- **`error_if_nonfinite=True`** trong clip — báo lỗi **ngay** khi gradient thành NaN/inf, thay vì để NaN lan âm thầm khắp trọng số rồi mới phát hiện sau vài task.

---

## A4. `run_continual()` ⭐⭐ HÀM QUAN TRỌNG NHẤT DỰ ÁN

```python
def run_continual(model, method, stream, task_loaders, device,
                  train_cfg, verbose=True) -> tuple[np.ndarray, dict]
```

**Trả:** `(R, log)`. `R[i,j]` = acc task j sau khi học task i. `log` chứa loss từng task, ma trận NCM-head, trace prequential…

### Bộ khung — bỏ hết nhánh tuỳ chọn

```python
T = len(stream)
R = np.zeros((T, T), dtype=float)
seen: List[int] = []
for t, spec in enumerate(stream):
    allowed_train = spec.classes                    # ⭐ TRAIN: chỉ lớp của task t
    if getattr(method, "gradient_free", False):
        method.fit_task(model, task_loaders[t]["train"], device)     # NCM, SLDA
    else:
        losses, opt_used = train_one_task(model, method, task_loaders[t]["train"],
                                          device, allowed_train, train_cfg, opt=opt_carry)
    method.end_task(model, task_loaders[t]["train"], device, allowed_train)  # ← HOOK 4
    seen += list(allowed_train)
    allowed_eval = sorted(seen)                     # ⭐ EVAL: mọi lớp ĐÃ học
    for j in range(t + 1):
        R[t, j] = evaluate(model, task_loaders[j]["test"], device, allowed_eval)
return R, log
```

**Hai dòng có ⭐ là quy ước mask của cả dự án**, được cài đặt đúng ở đây và chỉ ở đây:

- train: `allowed = spec.classes` — chỉ lớp của task hiện tại
- eval: `allowed = sorted(seen)` — mọi lớp đã học từ task 0 tới t

**Vòng `for j in range(t+1)`** là lý do ma trận R có tam giác dưới: sau task t chỉ chấm được task 0..t.

**`getattr(method, "gradient_free", False)`** — cách engine phân biệt hai loại method mà không cần `isinstance`. NCM và SLDA không train bằng gradient, chúng chỉ "hấp thụ" dữ liệu.

### Sáu nhánh tuỳ chọn

Tất cả **mặc định TẮT**. Nguyên tắc: bật cờ mới thì run cũ **bất biến**.

#### (1) Optimizer sống xuyên task

```python
persist_opt = not bool(train_cfg.get("optimizer_per_task", True))
opt_carry = None
# ... sau mỗi task:
if persist_opt:
    opt_carry = opt_used
```

#### (2) Checkpoint / resume

```python
ckpt_path = train_cfg.get("checkpoint_path") or None
if ckpt_path and bool(train_cfg.get("resume", False)):
    if os.path.exists(ckpt_path):
        pay = load_run_checkpoint(ckpt_path, map_location=device)
        if int(pay["num_tasks"]) != T:
            raise ValueError(f"checkpoint có {pay['num_tasks']} task, stream hiện tại {T} — không khớp.")
        model.load_state_dict(pay["model"])
        if pay.get("memory_state") is not None:
            model._state = pay["memory_state"]        # ký ức Titans nằm NGOÀI state_dict
        method = pay["method"]                        # buffer/anchor/teacher sống lại nguyên vẹn
        R, log = pay["R"], pay["log"]
        torch.set_rng_state(pay["torch_rng"].cpu())   # ⭐ khôi phục cả RNG
        random.setstate(pay["py_rng"])
        start_task = int(pay["task_done"]) + 1
        seen = [c for s in stream[:start_task] for c in s.classes]
```

**Ba chi tiết quyết định tính đúng đắn:**

- **`method = pay["method"]`** — pickle **nguyên object method**, không chỉ state_dict. Vì buffer ảnh của Replay, anchor Fisher của EWC, teacher của LwF đều nằm trong object đó chứ không phải trong model.
- **`model._state`** phải khôi phục riêng vì ký ức Titans không nằm trong `state_dict()`.
- **Khôi phục RNG** — nhờ vậy chạy tiếp cho số **giống hệt** chạy liền một mạch. Test kiểm điều này: `tests/test_checkpoint.py`.

Lưu sau **mỗi** task, **atomic**:
```python
if ckpt_path:
    save_run_checkpoint(ckpt_path, model=model, opt=opt_carry, method=method,
                        R=R, log=log, task_done=t, num_tasks=T, proto=_proto)
```

#### (3) Dừng sớm

```python
if stop_after is not None and t >= stop_after:
    print(f"[engine] dừng sớm sau task {t} (train.stop_after_task={stop_after}).")
    break
```
Chạy theo ca trên VM có giới hạn giờ.

#### (4) Forward transfer

```python
if bool(train_cfg.get("eval_future", False)) and t + 1 < T:
    allowed_next = sorted(set(seen) | set(stream[t+1].classes))
    R[t, t+1] = evaluate(model, task_loaders[t+1]["test"], device, allowed_next)
```
Điền ô **tam giác trên** — accuracy trên task kế tiếp **trước khi** học nó. Mặc định tắt vì tốn thêm một lượt eval mỗi task.

#### (5) NCM-head shadow eval — "Đòn A"

Giả thuyết: head Linear train-liên-tục là **nút thắt** forgetting. Test rẻ: không train lại gì, chỉ dựng prototype từ feature-sau-memory rồi phân loại bằng cosine.

```python
@torch.no_grad()
def _memory_prototypes(model, task_loaders, device, seen_task_ids, num_classes, feat_dim):
    model.eval()
    psum = torch.zeros(num_classes, feat_dim, device=device)
    pcnt = torch.zeros(num_classes, device=device)
    for tid in seen_task_ids:
        for x, y in task_loaders[tid]["train"]:
            h = F.normalize(model.features(x.to(device)).float(), dim=1)
            psum.index_add_(0, y.to(device), h)             # cộng dồn theo lớp
            pcnt.index_add_(0, y.to(device), torch.ones_like(y, dtype=torch.float))
    return F.normalize(psum / pcnt.clamp(min=1.0).unsqueeze(1), dim=1)
```

**`index_add_`** là cách cộng dồn theo nhãn không cần vòng lặp Python — nhanh hơn nhiều lần.

```python
@torch.no_grad()
def _evaluate_ncm(model, loader, device, allowed, prototypes):
    feats = F.normalize(model.features(x).float(), dim=1)
    logits = feats @ prototypes.t()        # ⭐ cosine dùng THẲNG như logits
    pred = mask_logits(logits, allowed).argmax(dim=1)
```

Dòng ⭐: vì cả hai vế đã chuẩn hoá, tích vô hướng **chính là** cosine. Và vì shape `(B, C)` giống logits nên `mask_logits` dùng lại được không sửa gì.

**Kết quả của đòn này** (in ở cuối `run_g1.py`): Linear 0.583/F0.274 → NCM-head 0.758/F0.078. Chênh 17,5 điểm chỉ do cách **đọc** feature. Phát hiện này đẻ ra `CosineHead`.

#### (6) SDC — Semantic Drift Compensation

`_memory_prototypes` phải đọc lại **mọi** task đã thấy → đắt và vi phạm tinh thần "không đọc lại data cũ". SDC làm rẻ hơn:

```python
@torch.no_grad()
def _sdc_shift(proto, old_model, new_model, loader, device, seen_before, sigma=0.5):
    fo_list, d_list = [], []
    for x, _ in loader:                                  # CHỈ data task hiện tại
        fo = F.normalize(old_model.features(x).float(), dim=1)   # feature model CŨ
        fn = F.normalize(new_model.features(x).float(), dim=1)   # feature model MỚI
        fo_list.append(fo); d_list.append(fn - fo)               # δ = độ trôi
    fo = torch.cat(fo_list); dl = torch.cat(d_list)
    two_s2 = 2.0 * sigma * sigma
    for c in seen_before:
        p = proto[c]
        w = torch.exp(-((fo - p)**2).sum(dim=1) / two_s2)   # mẫu gần prototype -> trọng số lớn
        denom = w.sum()
        if float(denom) < 1e-8: continue
        drift = (w.unsqueeze(1) * dl).sum(dim=0) / denom
        proto[c] = F.normalize(p + drift, dim=0)
    return proto
```

**Ý tưởng:** backbone trôi khi học task mới → prototype lớp cũ trỏ sai chỗ. Đo độ trôi `δ` trên **data task hiện tại** (thứ duy nhất được phép đọc), rồi dịch prototype lớp cũ theo trung bình `δ` **có trọng số Gaussian** theo khoảng cách.

Mốc đo trôi được chụp ở đầu task:
```python
if want_ncm and ncm_mode == "sdc" and t > 0 and hasattr(model, "features"):
    old_model_sdc = copy.deepcopy(model).eval()
    for p in old_model_sdc.parameters(): p.requires_grad_(False)
# ... dùng xong:
del old_model_sdc          # giải phóng ngay, không giữ 2 model qua task sau
```

#### (7) Pha 2 không nhãn — mã hoá bài toán gốc

```python
_p2 = dict(train_cfg.get("pha2") or {})
_khong_nhan = bool(_p2.get("khong_nhan", False))
_chuyen_hc  = int(_p2.get("chuyen_hieu_chinh", 1))
if _khong_nhan and t >= _chuyen_hc:
    if hasattr(model, "hap_thu_khong_nhan"):
        _allowed_preq = sorted(set(seen) | set(allowed_train))
        log.setdefault("trace", {})[t] = model.hap_thu_khong_nhan(
            task_loaders[t]["train"], device, allowed=_allowed_preq)
    else:
        print(f"[pha2] chuyến {t}: KHÔNG nhãn, model không có tầng không nhãn "
              "-> đóng băng hoàn toàn (mốc U0)")
else:
    method.fit_task(model, task_loaders[t]["train"], device)     # pha 1, CÓ nhãn

if _khong_nhan and t == _chuyen_hc - 1 and hasattr(model, "chot_moc_pha1"):
    model.chot_moc_pha1()      # cuối pha hiệu chỉnh: đóng băng mốc (m0, v0)
```

**Comment trong code nói rõ vì sao vá:**

> Bài toán gốc (BAI_TOAN §2/§3): nhãn **chỉ có** ở pha hiệu chỉnh. Trước bản vá này, `fit_task` nhận `(x, y)` **có nhãn ở MỌI chuyến** — giải một bài **dễ hơn** bài thật.

Đây là lỗi nghiêm trọng nhất mà một dự án ML có thể mắc: **thí nghiệm chạy đúng, code không bug, nhưng đo nhầm bài toán.** Đáng ghi nhớ.

`log["trace"][t]` = chuỗi accuracy **prequential** của chuyến t, nuôi thước đo O2 và O3 (Bài 3).

---

# PHẦN B — `methods.py` (548 dòng)

**Nhiệm vụ:** 10 chiến lược chống quên, sau **một** giao diện 5 hook.

**Cách đọc file 548 dòng này:** đừng đọc tuần tự. Đọc `FineTune` (lớp cha) rồi nhảy tới từng method con, mỗi cái chỉ xem **phần ghi đè**.

---

## B1. `FineTune` — lớp cha, 5 hook

```python
class FineTune:
    name = "finetune"
    gradient_free = False          # True -> engine bỏ vòng train gradient, gọi fit_task

    def __init__(self, **_): pass                                   # ⭐ **_ nuốt config thừa
    def begin_task(self, model, device, allowed): pass              # HOOK 1
    def penalty(self, model): return None                           # HOOK 2
    def extra_batch_loss(self, model, x, logits_full, device): return None   # HOOK 3
    @torch.no_grad()
    def end_task(self, model, loader, device, allowed): pass        # HOOK 4
    def footprint_floats(self, model): return 0                     # HOOK 5
```

**Nó vừa là baseline vừa là lớp cha.** Là baseline: học task mới, không làm gì để giữ task cũ → quên nhiều nhất (mốc dưới, RESISC45 forgetting 0.61). Là lớp cha: định nghĩa đủ 5 hook rỗng.

**`**_` trong `__init__`** — nuốt mọi tham số thừa. Vì `build_method` truyền cả khối `cfg[tên_method]` vào, mà config có thể có key mà method không dùng.

**Bảng "method nào ghi đè hook nào"** — dùng làm bản đồ khi đọc:

| Method | `begin_task` | `penalty` | `extra_batch_loss` | `end_task` | khác |
|---|:---:|:---:|:---:|:---:|---|
| FineTune | — | — | — | — | |
| **EWC** | — | ✓ | — | ✓ | |
| **Replay** | — | — | ✓ | ✓ | |
| **LwF** | ✓ | — | ✓ | ✓ | |
| TitansCL | ✓ | — | — | ✓ | |
| CMS | ✓ | — | — | ✓ | |
| HOPE | ✓ | — | — | ✓ | kế thừa CMS |
| NCM | — | — | — | — | `fit_task`, `gradient_free` |
| SLDA | — | — | — | ✓ | `fit_task`, `gradient_free` |
| LatentReplay | ✓ | — | ✓ | ✓ | kế thừa TitansCL |

---

## B2. `Replay` — đọc method này trước

**Baseline mạnh nhất** (RESISC45 acc 0.794, forgetting 0.079). Ý tưởng đơn giản: giữ một ít ảnh cũ, mỗi bước train "ôn" thêm.

```python
def __init__(self, buffer_per_class=20, replay_batch=32, weight=1.0, seed=0, **_):
    self._buf  = []      # [(ảnh float16 CPU, nhãn)]
    self._seen = []      # lớp đã có trong buffer
    self._rng  = random.Random(seed)
```

**Hook `end_task` — nạp buffer:**

```python
@torch.no_grad()
def end_task(self, model, loader, device, allowed):
    quota = {int(c): self.buffer_per_class for c in allowed}   # hạn mức mỗi lớp
    for x, y in loader:
        for i in range(len(y)):
            c = int(y[i])
            if quota.get(c, 0) > 0:
                self._buf.append((x[i].detach().to(torch.float16).cpu(), c))
                quota[c] -= 1
        if all(v == 0 for v in quota.values()):
            break                                # ⭐ đủ hạn mức -> dừng sớm
    self._seen = sorted(set(self._seen) | {int(c) for c in allowed})
```

- **quota mỗi lớp** thay vì lấy N ảnh đầu → buffer cân bằng lớp
- **`float16` + `.cpu()`** → giảm nửa RAM, và không chiếm VRAM
- **`break` khi đủ** → không duyệt hết loader vô ích

**Hook `extra_batch_loss` — ôn bài:**

```python
def extra_batch_loss(self, model, x, logits_full, device):
    if not self._buf: return None                # task đầu: chưa có gì để ôn
    idx = [self._rng.randrange(len(self._buf))
           for _ in range(min(self.replay_batch, len(self._buf)))]
    xs = torch.stack([self._buf[i][0] for i in idx]).float().to(device)
    ys = torch.tensor([self._buf[i][1] for i in idx], dtype=torch.long, device=device)
    logits = mask_logits(model(xs), self._seen)   # ⭐ mask về lớp CÓ trong buffer
    return self.weight * F.cross_entropy(logits, ys)
```

**Dòng ⭐:** mask về `self._seen` (lớp có trong buffer), **không phải** `allowed` (lớp task hiện tại). Đúng logic — đang ôn lớp cũ.

**Chi phí:** RESISC45 45 lớp × 20 ảnh 224² float16 ≈ **270 MB RAM**, 135M float. Đây là lý do docstring nhắc *"điều UAV thực tế khó chấp nhận"* — cả về RAM lẫn quyền riêng tư (lưu ảnh thô).

---

## B3. `LwF` — không lưu ảnh, lưu model

```python
def __init__(self, lwf_lambda=1.0, temperature=2.0, **_):
    self._teacher = None
    self._old_classes = []
```

**Hook `begin_task` — chụp teacher:**

```python
def begin_task(self, model, device, allowed):
    if self._old_classes:                            # có lớp cũ -> mới cần teacher
        self._teacher = copy.deepcopy(model).to(device)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)
    else:
        self._teacher = None                         # task đầu: chưa có gì để giữ
```

**Đây là lý do hook `begin_task` tồn tại.** Phải chụp bản sao model **trước khi** trọng số bị task mới thay đổi. Chụp ở `end_task` là muộn — đã bị đổi rồi.

**Hook `extra_batch_loss` — distillation:**

```python
def extra_batch_loss(self, model, x, logits_full, device):
    if self._teacher is None: return None
    with torch.no_grad():
        t_logits = self._teacher(x)
    idx = torch.as_tensor(self._old_classes, dtype=torch.long, device=logits_full.device)
    T = self.temperature
    p_teacher     = F.softmax(t_logits[:, idx] / T, dim=1)
    log_p_student = F.log_softmax(logits_full[:, idx] / T, dim=1)
    return self.lwf_lambda * (T*T) * F.kl_div(log_p_student, p_teacher, reduction="batchmean")
```

Công thức: `L = CE(task mới) + λ·T²·KL( student/T ‖ teacher/T )` trên **cột lớp cũ**.

**Ba chi tiết:**
- **`logits_full[:, idx]`** — dùng logits **chưa mask**. Nếu dùng bản đã mask thì cột lớp cũ toàn `-1e4`, distill vô nghĩa. Đây là lý do `train_one_task` truyền `logits_full` chứ không phải `logits`.
- **`T²`** — chuẩn distillation (Hinton). Chia logits cho `T` làm gradient nhỏ đi `T²` lần, nhân lại để bù.
- **`F.kl_div(log_p, p)`** — PyTorch yêu cầu **log**-probability ở vế đầu, probability thường ở vế sau. Đảo là sai âm thầm.

**Cảnh báo thực tế** (từ `docs/KET_LUAN_G1.md`): LwF trên EuroSAT có forgetting **0.609 — tệ hơn cả finetune (0.257)**. Nghi vấn: task chỉ 2 lớp → distillation (λ=1.0, T=2) đè tín hiệu CE. Trên RESISC45 (5 lớp/task) thì chạy tốt (0.278). Bài học: **λ của regularizer phụ thuộc số lớp mỗi task**.

---

## B4. `EWC` — hook `penalty`

Sau mỗi task, ước lượng tham số nào **quan trọng** (Fisher) rồi phạt nếu task sau kéo chúng đi xa.

**Hook `end_task` — ước lượng Fisher:**

```python
# ⚠️ CHÚ Ý: đây là end_task DUY NHẤT trong file KHÔNG có @torch.no_grad().
#    Cố ý — nó cần .backward() để lấy gradient. Mọi end_task khác đều có decorator.
def end_task(self, model, loader, device, allowed):
    model.eval()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fisher = {n: torch.zeros_like(p) for n, p in params.items()}
    n_batches = 0
    for x, y in loader:
        if n_batches >= self.max_batches: break        # chỉ dùng tối đa 50 batch
        model.zero_grad(set_to_none=False)
        logits = mask_logits(model(x), allowed)
        F.cross_entropy(logits, y).backward()
        for n, p in params.items():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2      # ⭐ Fisher ≈ trung bình grad²
        n_batches += 1
    self._anchors.append({
        "params": {n: p.detach().clone() for n, p in params.items()},   # θ*
        "fisher": {n: f / n_batches for n, f in fisher.items()},        # F
    })
```

**Fisher ≈ trung bình bình phương gradient** — tham số nào gradient lớn thì loss nhạy với nó → nó quan trọng.

**Hook `penalty`:**

```python
def penalty(self, model):
    if not self._anchors: return None                  # task đầu chưa có mỏ neo
    loss = None
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for anchor in self._anchors:                       # ⭐ cộng phạt cho TỪNG task cũ
        for n, p in params.items():
            term = (anchor["fisher"][n] * (p - anchor["params"][n])**2).sum()
            loss = term if loss is None else loss + term
    return (self.ewc_lambda / 2.0) * loss
```

Công thức: `L = CE + (λ/2)·Σ_anchors Σ_i F_i·(θ_i − θ*_i)²`

**Vì sao EWC đắt kinh khủng** — nhìn thấy ngay từ code:

```python
def footprint_floats(self, model):
    return sum(t.numel() for anchor in self._anchors
               for group in ("params", "fisher") for t in anchor[group].values())
```

Mỗi task lưu **2 × P** float (bản sao tham số + Fisher). Với ViT-S 21,7M tham số × 9 task × 2 = **390M float ≈ 1,5 GB**. Và `penalty()` phải quét **toàn bộ** anchor **mỗi bước train** → RESISC45 mất **3 giờ**, gấp 5 lần finetune.

**Kết quả:** chỉ +0.01 acc so với finetune. Docstring giữ EWC trong bảng *"làm bằng chứng regularization cổ điển không đủ"*.

---

## B5. `NCM` và `SLDA` — nhánh `gradient_free`

```python
class NCM(FineTune):
    name = "ncm"
    gradient_free = True                  # ⭐ engine bỏ vòng train gradient

    @torch.no_grad()
    def fit_task(self, model, loader, device):
        if not hasattr(model, "update_prototypes"):
            raise TypeError("Method 'ncm' cần model là NCMClassifier (run_g1 tự chọn đúng).")
        model.eval()
        for x, y in loader:
            feats = model.backbone(x.to(device))
            model.update_prototypes(feats, y.to(device))
```

**Toàn bộ "học" của NCM là 4 dòng.** Không optimizer, không loss, không backward. Duyệt dữ liệu **đúng một lượt**, cộng feature vào prototype.

`SLDA.fit_task` giống hệt, chỉ gọi `model.update(feats, y)` thay vì `update_prototypes`.

**`SLDA.end_task`** thì dài — nhưng toàn bộ là **in bằng chứng**, không có logic học:
- `memory_report()` → chi phí byte thật
- `window_report()` → cửa sổ nhớ hiệu dụng, so với kỳ vọng lý thuyết `(1−λ^m)/(1−λ)`
- cảnh báo khi `n_c` lệch >5% so với lý thuyết → **nghi cài sai chỗ phân rã**

> Comment ở đó đáng đọc: so với tiệm cận `1/(1−λ)` là **so nhầm mốc** vì RESISC45 chỉ cho 420 lần gặp/lớp, chưa đủ hội tụ → phải so với công thức hữu hạn mẫu. Chi tiết kiểu này là thứ phân biệt "code chạy" với "kết quả tin được".

---

## B6. `TitansCL`, `CMS`, `HOPE` — method chỉ để quản lý vòng đời + log

**`TitansCL.begin_task`:**
```python
if getattr(model, "reset_mode", None) == "task":
    model.reset_state()                # chế độ "task": xoá ký ức mỗi task mới
if hasattr(model, "reset_eta_alpha"):
    model.reset_eta_alpha()            # bắt đầu đo η/α cho task này
```

**`TitansCL.end_task`** — dài ~60 dòng, **toàn bộ là chẩn đoán**:
- `norm(state)` sau task → phát hiện ký ức "phình/nổ"
- `self_mod_stats()` → độ "sống" của nhánh self-modifying
- `eta_alpha_stats()` → η/α trung bình, và **vị trí tương đối trong khoảng đã chặn**

```python
for gia_tri, ten, ky_hieu, dinh in ((eta_real, "eta", "η", "{:.4g}"),
                                    (alpha, "alpha", "α", "{:.4f}")):
    frac, r = _vi_tri(gia_tri, ten)
    print(f"[titans]   {ky_hieu}={dinh.format(gia_tri)} "
          f"({frac*100:.0f}% khoảng [{dinh.format(r[0])}, {dinh.format(r[1])}])")
    if frac > GATE_SAT_FRAC or frac < 1.0 - GATE_SAT_FRAC:
        bien = "TRẦN" if frac > 0.5 else "SÀN"
        print(f"[titans]   ⚠️ {ky_hieu} BÃO HOÀ Ở {bien} ...")
```

**Comment giải thích vì sao đổi cách in — bài học đắt:**

> log cũ in "η=0.896" mà **không in trần**, nên người đọc không có mốc để biết đó là sát trần hay giữa dải. Lỗi trần η lệch thang 100× vì thế **sống sót trọn một ngày**. Giờ in vị trí **tương đối** — máy tự nói ra thay vì bắt người suy luận.

**Nguyên tắc rút ra: log phải in kèm mốc so sánh, không chỉ in giá trị.**

**`CMS.begin_task` / `end_task`** — chụp trọng số đầu task, cuối task đo `‖Δw‖` từng tier:

```python
delta = sum(float((p.detach().float().cpu() - b).norm())**2
            for p, b in zip(g["params"], before)) ** 0.5
base  = sum(float(b.norm())**2 for b in before) ** 0.5
rel   = delta / base if base > 0 else 0.0
print(f"[cms] ‖Δw‖ {g['name']:<5s} (p={g['period']:<3d}): {delta:10.4f}  ({rel:.3%} của ‖w‖)")
```

**Kỳ vọng:** tier chậm ≈ 0% (giữ kiến thức), tier nhanh lớn (thích nghi). Docstring cảnh báo: *"Ngược lại nghĩa là mapping/chu kỳ cài sai — sửa trước khi tin bất kỳ số nào."*

**`HOPE`** kế thừa `CMS`, chỉ thêm phần Titans:
```python
def begin_task(self, model, device, allowed):
    if getattr(model, "reset_mode", None) == "task":
        model.reset_state()
    super().begin_task(model, device, allowed)      # snapshot Δw của CMS
```

---

## B7. `LatentReplay` — replay rẻ hơn 400 lần

Kế thừa `TitansCL` (để giữ vòng đời state), thay ảnh bằng **feature**:

```python
@staticmethod
def _backbone_feats(model, x):
    # Titans/HOPE có _extract; ContinualClassifier dùng thẳng backbone
    return model._extract(x) if hasattr(model, "_extract") else model.backbone(x)

def extra_batch_loss(self, model, x, logits_full, device):
    if not self._buf: return None
    if not hasattr(model, "forward_from_feats"):
        raise TypeError("latent_replay cần model có forward_from_feats ...")
    zs = torch.stack([...]).float().to(device)
    logits = mask_logits(model.forward_from_feats(zs), self._seen)
    return self.weight * F.cross_entropy(logits, ys)
```

**Lập luận chi phí** (từ docstring): backbone đóng băng → feature cũ không bao giờ "ôi" → chỉ cần lưu vector 384-d float16 ≈ **0,75 KB/mẫu**, so với ảnh 224² ≈ **300 KB**. Rẻ ~400 lần.

**Điều kiện:** model phải có `forward_from_feats`. Ném `TypeError` **rõ ràng** nếu thiếu — không để chết ở chỗ khác.

---

## B8. `build_method()` — factory

```python
_METHODS = {"finetune": FineTune, "ewc": EWC, "replay": Replay, "lwf": LwF,
            "ncm": NCM, "titans": TitansCL, "cms": CMS, "hope": HOPE,
            "slda": SLDA, "latent_replay": LatentReplay}

def build_method(name: str, cfg: dict) -> FineTune:
    name = name.lower()
    if name not in _METHODS:
        raise KeyError(f"Unknown method '{name}'. Available: {sorted(_METHODS)}")
    kwargs = dict(cfg.get(name, {}))     # cfg['ewc'] = {ewc_lambda:..., max_batches:...}
    return _METHODS[name](**kwargs)
```

**`cfg.get(name, {})`** — mỗi method đọc **khối config cùng tên** của nó. Đây là lý do yaml có khối `ewc:`, `replay:`, `lwf:` riêng. Thêm method mới = thêm class + một dòng registry + (tuỳ chọn) một khối yaml.

---

# PHẦN C — `scripts/run_g1.py` (420 dòng)

**Nhiệm vụ:** nối mọi mảnh lại. Dài nhưng **tuyến tính** — đọc từ trên xuống, không nhảy.

## C1. `run_dir_name(cfg, method_name)`

```python
name = f"{cfg['data']['name']}_{method_name}_seed{cfg.get('seed',0)}"
opt = str(cfg.get("train",{}).get("optimizer","adamw")).lower()
if opt != "adamw":              name += f"_{opt}"
if mem_cfg.get("enabled"):      name += f"_r{mem_cfg.get('reset','image')}"   # 3 bậc A/B/C
if not cfg["train"].get("optimizer_per_task", True): name += "_optkeep"
if cms_cfg.get("enabled"):
    periods = "-".join(str(t[1]) for t in cms_cfg.get("tiers", []))
    name += f"_{cms_cfg.get('order','late_slow')}_p{periods}"
return name
```

**Hàm 15 dòng nhưng sinh ra từ một bẫy thật:** hai cấu hình khác nhau (vd `reset=image` và `reset=never`) mà cùng tên thư mục → run sau **đè** kết quả run trước, và bạn không biết. Docstring gọi là *"nguồn duy nhất, dùng cho cả ghi lẫn resume"*.

## C2. `main()` — 8 khối

**Khối 1 — args + config:**
```python
cfg = apply_overrides(load_config(args.config), args.overrides)
method_name = (args.method or cfg.get("train",{}).get("method","finetune")).lower()
out = pathlib.Path(cfg["log"]["dir"]) / "results" / run_dir_name(cfg, method_name)
if args.skip_existing and (out/"metrics.json").exists():
    print(f"[SKIP] {out.name} — đã có kết quả."); return 0
seed_everything(seed)
```
`--skip-existing` là resume cấp campaign: chạy lại `run_all.py` sẽ bỏ qua run đã xong.

**Khối 2 — chọn stream** (4 nhánh, đây là chỗ dễ lạc nhất):

```python
if openset.enabled:                        stream, holdout = build_stream_with_holdout(...)
elif data.stream_type == "revisit":        stream = build_domain_stream(...)   # + lịch bay
elif data.stream_type == "domain":         stream = build_domain_stream(...)
else:                                      stream = build_stream(...)          # ← mặc định
```

Nhánh revisit có **cửa chặn**:
```python
if not bool((cfg["data"].get("drift") or {}).get("enabled", False)):
    raise ValueError("stream_type=revisit cần data.drift.enabled=true — không có trôi "
                     "thì mọi chuyến cùng điều kiện, thước đo O3 vô nghĩa")
```
Và lấy nhãn điều kiện thật **chỉ để chấm điểm**:
```python
_lich = lich_tu_cfg(len(stream), dict(cfg["data"].get("drift") or {}))
mode_that = [c.mode_that for c in _lich]      # ⭐ CHỈ để chấm điểm, model KHÔNG thấy
```

**Khối 3 — chọn model** (thứ tự if/elif quan trọng):

```python
if memory.enabled and cms.enabled:   model = HOPEClassifier(...)      # G4
elif memory.enabled:                 model = TitansClassifier(...)    # G2
elif method_name == "ncm":           model = NCMClassifier(...)
elif method_name == "slda":          model = SLDAClassifier(...)
else:                                model = ContinualClassifier(...) # G1
method = build_method(method_name, cfg)
```

**Model và method chọn ĐỘC LẬP.** Có thể cấu hình sai (vd `memory.enabled` + `--method ewc`), nên có cảnh báo:
```python
if method_name not in ("titans", "finetune"):
    print(f"[WARN] memory.enabled + method '{method_name}' — thường dùng --method titans")
```

**Khối 4 — chạy:**
```python
if cms_cfg.get("enabled"): cfg["train"]["cms"] = cms_cfg   # engine đọc từ train_cfg
R, log = run_continual(model, method, stream, loaders, device, cfg["train"])
```
Một dòng. Mọi thứ trước đó là chuẩn bị.

**Khối 5 — metrics + lưu:**
```python
metrics = {..., "average_accuracy": average_accuracy(R),
           "average_anytime_accuracy": average_anytime_accuracy(R),
           "average_forgetting": average_forgetting(R),
           "backward_transfer": backward_transfer(R),
           "forward_transfer": forward_transfer(R) if cfg["train"].get("eval_future") else None,
           "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
           "method_extra_floats": int(method.footprint_floats(model)),
           "runtime_sec": round(time.time()-t0, 1)}
pd.DataFrame(R, ...).to_csv(out/"acc_matrix.csv", float_format="%.4f")
(out/"metrics.json").write_text(json.dumps(metrics, indent=2))
save_config(cfg, out/"config.yaml")
```

**Luôn lưu kèm chi phí** (`trainable_params`, `method_extra_floats`, `runtime_sec`). Triết lý dự án: *"so cùng accuracy thì method rẻ hơn thắng"*.

**Khối 6 — open-set** (nếu có holdout), **Khối 7 — export memory state** (nếu model có), **Khối 8 — báo cáo revisit** (nếu `mode_that is not None`).

## C3. Khối revisit — nơi O3 được tính

Đáng đọc riêng vì chứa một phát hiện quan trọng:

```python
ts = sorted(trace); K = 10
m_dau = [sum(trace[t][:K]) / max(len(trace[t][:K]), 1) for t in ts]   # 10 batch ĐẦU
m_ca  = [sum(trace[t])     / max(len(trace[t]), 1)     for t in ts]   # cả chuyến
r10 = loi_ich_quay_lai(_R_gia(m_dau), modes_p2)
rv["o3_preq10_loi_ich"] = r10["loi_ich_quay_lai"]      # ⭐ số chính
```

**Comment giải thích:**

> đường chéo R chấm **sau** khi hấp thụ cả chuyến, lúc tầng nhanh **đã tự hội tụ** (~3-4 batch với λ=0,99) → `R[t][t]` của U1 và U2 gần trùng nhau và `loi_ich_quay_lai` trên R **bị mù** với ngân hàng chế độ. Lợi ích "nhận ra ngay" nằm ở **đầu chuyến** → đo trên trung bình prequential **10 batch đầu**.

Tức là: nếu chấm ở cuối chuyến thì cả hai arm đều đã kịp thích nghi → không thấy khác biệt. Lợi ích của ngân hàng chế độ nằm ở chỗ **không phải mất thời gian thích nghi**, và điều đó chỉ nhìn thấy ở đầu chuyến.

**Đây là bài học cùng loại với `metrics/revisit.py`: đo đúng chỗ, đúng lúc, mới thấy được hiệu ứng thật.**

---

# Checklist Bài 2

- [ ] Viết lại `run_continual` bằng trí nhớ, chỉ bộ khung 15 dòng
- [ ] Kể tên 5 hook và **thời điểm** engine gọi từng cái
- [ ] Giải thích vì sao `extra_batch_loss` nhận `logits_full` chứ không phải `logits` đã mask
- [ ] Giải thích vì sao LwF chụp teacher ở `begin_task` chứ không phải `end_task`
- [ ] Tính được footprint của EWC trên ViT-S 9 task, ra ~390M float
- [ ] Chạy `pytest tests/test_methods_g1.py tests/test_checkpoint.py -q` — xanh
- [ ] Mở một `metrics.json` bất kỳ trong `artifacts/`, đọc hiểu mọi trường

Xong → sang **Bài 3: models, optimizer, và phần bài toán UAV thật**.
