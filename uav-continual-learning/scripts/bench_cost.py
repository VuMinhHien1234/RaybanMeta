#!/usr/bin/env python3
"""B6 — Đo CHI PHÍ TRIỂN KHAI thật của SLDA vs Titans (bộ nhớ, độ trễ, thông lượng).

Vì sao cần: cả báo cáo đang chỉ có accuracy. Với UAV, ba con số dưới đây quyết định
phương pháp có dùng được không, và **chưa ai đo**:

  1. Bộ nhớ trạng thái phải mang theo (byte thật, đúng dtype — không phải "số float")
  2. Độ trễ suy luận / ảnh (ms)
  3. Thời gian cập nhật / ảnh (ms) — drone có kịp học giữa hai khung hình không

Đo trên feature NGẪU NHIÊN, không cần dataset — nên chạy được ở mọi máy, kể cả CPU.
Backbone bị loại khỏi phép đo có chủ đích: nó CHUNG cho mọi phương pháp, nên đưa vào
chỉ làm loãng khác biệt. Con số ở đây là "chi phí TĂNG THÊM so với ViT trần".

Dùng:
    python scripts/bench_cost.py                       # mặc định D=384 (ViT-S), C=45
    python scripts/bench_cost.py --dim 1024 --classes 45   # ViT-L, xem O(D²) phình ra sao
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def _timeit(fn, n_warm=3, n_rep=20):
    """Trả (trung bình ms, độ lệch chuẩn ms). Có sync CUDA/MPS để đo đúng."""
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.synchronize()

    for _ in range(n_warm):
        fn()
    _sync()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    m = sum(ts) / len(ts)
    sd = (sum((t - m) ** 2 for t in ts) / max(len(ts) - 1, 1)) ** 0.5
    return m, sd


class _Id(torch.nn.Module):
    """Backbone giả — feature đã có sẵn, ta chỉ đo phần THÊM VÀO."""

    def forward(self, x):
        return x


def ho_tro_fp64(device) -> bool:
    """MPS KHÔNG hỗ trợ float64 — và nhiều NPU biên cũng vậy. Đây là ràng buộc TRIỂN KHAI thật."""
    try:
        torch.zeros(2, dtype=torch.float64, device=device)
        return True
    except Exception:                                 # noqa: BLE001
        return False


def bench_slda(dim, classes, batch, device, cov_mode="streaming", stats_dtype="float64"):
    from uavcl.models.slda import SLDAClassifier

    m = SLDAClassifier(_Id(), dim, classes, cov_mode=cov_mode,
                       stats_dtype=stats_dtype).to(device)
    f = torch.randn(batch, dim, device=device)
    y = torch.randint(0, classes, (batch,), device=device)
    m.update(f, y)                                    # ↳ mồi để cache dựng được
    m(f)

    upd, upd_sd = _timeit(lambda: m.update(f, y))
    m(f)                                              # ↳ cache đã mới -> đo forward thuần
    fwd, fwd_sd = _timeit(lambda: m(f))

    def _inv():
        m._cache_version = -1                         # ↳ ép tính lại Λ (gồm nghịch đảo D×D)
        m._refresh_cache()
    inv, inv_sd = _timeit(_inv, n_rep=5)

    r = m.memory_report()
    nhan = f"SLDA ({cov_mode}" + (f", {stats_dtype}" if stats_dtype != "float64" else "") + ")"
    return {
        "ten": nhan,
        "thiet_bi": str(device),
        "bo_nho_MB": r["total_MB"],
        "suy_luan_ms_moi_anh": fwd / batch,
        "cap_nhat_ms_moi_anh": upd / batch,
        "nghich_dao_ms_moi_lan": inv,
        "ghi_chu": f"nghịch đảo O(D³) chạy LAZY — 1 lần/task, không phải mỗi ảnh"
                   if cov_mode != "identity" else "identity: không nghịch đảo",
        "_sd": {"fwd": fwd_sd, "upd": upd_sd, "inv": inv_sd},
    }


def bench_titans(dim, classes, batch, device):
    try:
        from uavcl.models.memory import TitansMemory
    except ImportError as e:
        return {"ten": "Titans", "loi": f"không import được: {e}"}
    # Dựng ĐÚNG như titans_head.py: `depth` KHÔNG phải kwarg trực tiếp của NeuralMemory,
    # nó đi qua `default_model_kwargs` (xem titans_head.py:_stability_kwargs + khối depth).
    stab = dict(gated_transition=True, spectral_norm_surprises=True,
                qk_rmsnorm=True, max_grad_norm=1.0,
                default_model_kwargs=dict(depth=3, expansion_factor=4.0))
    try:
        mem = TitansMemory(dim=dim, chunk_size=32, self_modifying=True, **stab).to(device)
    except Exception as e:                            # noqa: BLE001
        return {"ten": "Titans", "loi": f"không dựng được: {type(e).__name__}: {e}"}

    seq = torch.randn(1, batch, dim, device=device)
    out, state = mem(seq)
    fwd, fwd_sd = _timeit(lambda: mem(seq, state=state))

    n_param = sum(p.numel() * p.element_size() for p in mem.parameters())

    def _state_bytes(s, _da_tham=None):
        """Duyệt ĐỆ QUY mọi kiểu chứa tensor.

        Bản đầu chỉ xử lý tensor/list/tuple/dict nên trả về 0 — state của titans-pytorch là
        NamedTuple/dataclass, không rơi vào nhánh nào. Bản này bắt thêm `_asdict`, `__dict__`,
        `__slots__`, và chống đếm trùng bằng `id()` (state hay chia sẻ chung tensor).
        """
        _da_tham = set() if _da_tham is None else _da_tham
        if id(s) in _da_tham:
            return 0
        _da_tham.add(id(s))
        if torch.is_tensor(s):
            return s.numel() * s.element_size()
        if isinstance(s, (list, tuple, set)):
            return sum(_state_bytes(x, _da_tham) for x in s)
        if isinstance(s, dict):
            return sum(_state_bytes(x, _da_tham) for x in s.values())
        if hasattr(s, "_asdict"):                     # NamedTuple
            return _state_bytes(s._asdict(), _da_tham)
        if hasattr(s, "__dict__") and vars(s):        # dataclass / object thường
            return _state_bytes(vars(s), _da_tham)
        if hasattr(s, "__slots__"):
            return sum(_state_bytes(getattr(s, k), _da_tham)
                       for k in s.__slots__ if hasattr(s, k))
        return 0

    return {
        "ten": "Titans (self-modifying, depth=3)",
        "thiet_bi": str(device),
        "bo_nho_MB": (n_param + _state_bytes(state)) / 1e6,
        "suy_luan_ms_moi_anh": fwd / batch,
        "cap_nhat_ms_moi_anh": 0.0,
        "ghi_chu": "suy luận ĐÃ GỒM ghi bộ nhớ (test-time learning) — không tách được",
        "_sd": {"fwd": fwd_sd},
        "_tham_so_MB": n_param / 1e6,
        "_state_MB": _state_bytes(state) / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=384, help="384=ViT-S · 768=ViT-B · 1024=ViT-L")
    ap.add_argument("--classes", type=int, default=45)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None, help="ghi kết quả ra file JSON")
    a = ap.parse_args()

    dev = a.device
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
               else "cpu")

    print(f"\nthiết bị={dev} · D={a.dim} · C={a.classes} · batch={a.batch}")
    print("(loại backbone khỏi phép đo — nó chung cho mọi phương pháp; đây là chi phí TĂNG THÊM)\n")

    dev_slda, ghi_chu_dtype = dev, ""
    if not ho_tro_fp64(dev):
        dev_slda = "cpu"
        ghi_chu_dtype = (f"\n  ⚠️  {dev} KHÔNG hỗ trợ float64 -> SLDA (float64) đo trên CPU.\n"
                         f"      Đây là RÀNG BUỘC TRIỂN KHAI thật, không phải lỗi script:\n"
                         f"      · cách (a): stats_dtype=float32 -> chạy được trên {dev}, bộ nhớ giảm nửa\n"
                         f"      · cách (b): backbone trên {dev}, thống kê SLDA ở CPU-float64 (khuyến nghị)\n")
        print(ghi_chu_dtype)

    rows = [bench_slda(a.dim, a.classes, a.batch, dev_slda, m)
            for m in ("streaming", "identity", "frozen")]
    # luôn đo thêm bản float32: nó chạy được trên MỌI thiết bị và bộ nhớ chỉ bằng nửa
    rows.append(bench_slda(a.dim, a.classes, a.batch, dev, "streaming", stats_dtype="float32"))
    rows.append(bench_titans(a.dim, a.classes, a.batch, dev))

    w = 34
    print(f"{'phương pháp':{w}} {'thiết bị':>9} {'bộ nhớ (MB)':>12} "
          f"{'suy luận (ms/ảnh)':>18} {'cập nhật (ms/ảnh)':>18}")
    print("-" * (w + 62))
    for r in rows:
        if "loi" in r:
            print(f"{r['ten']:{w}} {'-':>9} {'-':>12} {'-':>18} {'-':>18}   [{r['loi']}]")
            continue
        print(f"{r['ten']:{w}} {r.get('thiet_bi', dev):>9} {r['bo_nho_MB']:12.3f} "
              f"{r['suy_luan_ms_moi_anh']:18.4f} {r['cap_nhat_ms_moi_anh']:18.4f}")
    if dev_slda != dev:
        print(f"\n  ⚠️  Các dòng float64 chạy trên {dev_slda}, dòng float32 trên {dev}.")
        print("      -> CHỈ so được cột bộ nhớ. Hai cột thời gian KHÁC THIẾT BỊ, không so chéo được.")

    ok = [r for r in rows if "loi" not in r]
    slda = next((r for r in ok if r["ten"].startswith("SLDA (streaming")), None)
    tit = next((r for r in ok if r["ten"].startswith("Titans")), None)
    if slda and tit and slda["bo_nho_MB"] > 0:
        print(f"\n  Titans / SLDA về bộ nhớ : {tit['bo_nho_MB'] / slda['bo_nho_MB']:.1f} lần")
        if slda["suy_luan_ms_moi_anh"] > 0:
            cung_may = slda.get("thiet_bi") == tit.get("thiet_bi")
            print(f"  Titans / SLDA về độ trễ : "
                  f"{tit['suy_luan_ms_moi_anh'] / slda['suy_luan_ms_moi_anh']:.1f} lần"
                  + ("" if cung_may else "   ⚠️ KHÁC THIẾT BỊ, chỉ tham khảo"))
        print(f"  (Titans tách riêng: tham số {tit.get('_tham_so_MB', 0):.2f} MB "
              f"+ state {tit.get('_state_MB', 0):.2f} MB)")

    for r in ok:
        if r.get("ghi_chu"):
            print(f"  · {r['ten']}: {r['ghi_chu']}")

    f64 = next((r for r in ok if r["ten"] == "SLDA (streaming)"), None)
    f32 = next((r for r in ok if "float32" in r["ten"]), None)
    if f64 and f32:
        print(f"\n  float32 so với float64 : bộ nhớ {f32['bo_nho_MB']:.3f} vs "
              f"{f64['bo_nho_MB']:.3f} MB  (giảm "
              f"{100 * (1 - f32['bo_nho_MB'] / f64['bo_nho_MB']):.0f}%)")
        print("  -> float32 chạy được trên MPS/NPU biên. Phải đo accuracy trước khi dùng chính thức.")

    print("\n  Bộ nhớ SLDA là O(D²) do `gram` chi phối, KHÔNG phải O(C·D).")
    print("  Chạy --dim 1024 để thấy chi phí khi đổi sang backbone lớn hơn.\n")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"device": dev, "dim": a.dim, "classes": a.classes,
                       "batch": a.batch, "rows": rows}, fh, indent=2, ensure_ascii=False)
        print(f"  đã ghi {a.json}\n")


if __name__ == "__main__":
    main()
