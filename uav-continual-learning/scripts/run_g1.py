#!/usr/bin/env python3
"""G1 runner — chạy MỘT thí nghiệm học liên tục từ 1 file config.

Ví dụ:
  python scripts/run_g1.py --config configs/g1_smoke.yaml                     # kiểm tra pipeline (CPU, ~1 phút)
  python scripts/run_g1.py --config configs/g1_eurosat.yaml --method finetune
  python scripts/run_g1.py --config configs/g1_eurosat.yaml --method ewc
  python scripts/run_g1.py --config configs/g1_resisc45.yaml --method ewc --set train.lr=5e-5

Kết quả lưu ở  <log.dir>/results/<dataset>_<method>_seed<seed>/ :
  acc_matrix.csv   ma trận R[i,j] = acc task j sau khi học task i
  metrics.json     average_accuracy / average_forgetting / backward_transfer
  config.yaml      config đã dùng (tái lập được)
Sau khi chạy >=2 method, gộp bảng so sánh:  python scripts/compare_g1.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def run_dir_name(cfg: dict, method_name: str) -> str:
    """Tên thư mục kết quả — nguồn duy nhất, dùng cho cả ghi lẫn resume (--skip-existing)."""
    name = f"{str(cfg['data']['name']).lower()}_{method_name}_seed{int(cfg.get('seed', 0))}"
    opt = str(cfg.get("train", {}).get("optimizer", "adamw")).lower()
    if opt != "adamw":
        name += f"_{opt}"
    mem_cfg = cfg.get("memory") or {}
    if mem_cfg.get("enabled", False):  # 3 bậc reset A/B/C KHÔNG được ghi đè/skip lẫn nhau
        name += f"_r{str(mem_cfg.get('reset', 'image')).lower()}"
    if not bool(cfg.get("train", {}).get("optimizer_per_task", True)):
        name += "_optkeep"  # ablation "ký ức optimizer xuyên task" không đè run thường
    cms_cfg = cfg.get("cms") or {}
    if cms_cfg.get("enabled", False):
        periods = "-".join(str(t[1]) for t in cms_cfg.get("tiers", []))
        name += f"_{cms_cfg.get('order', 'late_slow')}_p{periods}"
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="đường dẫn file yaml (vd configs/g1_eurosat.yaml)")
    ap.add_argument("--method", default=None, help="finetune | ewc (mặc định lấy từ config)")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="k.sub=v",
                    help="override config, vd: train.lr=1e-4 data.num_tasks=5")
    ap.add_argument("--resume", action="store_true",
                    help="#23: chạy tiếp từ resume_checkpoint.pt trong thư mục kết quả (tự bật lưu checkpoint)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="đã có metrics.json cho run này thì bỏ qua (resume cho run_all)")
    args = ap.parse_args()

    from uavcl.utils.config import apply_overrides, load_config, save_config
    from uavcl.utils.seed import seed_everything

    cfg = apply_overrides(load_config(args.config), args.overrides)
    method_name = (args.method or cfg.get("train", {}).get("method", "finetune")).lower()
    seed = int(cfg.get("seed", 0))

    out = pathlib.Path(cfg.get("log", {}).get("dir", "./artifacts")) / "results" / run_dir_name(cfg, method_name)
    if args.skip_existing and (out / "metrics.json").exists():
        print(f"[SKIP] {out.name} — đã có kết quả.")
        return 0
    # #23: bật checkpoint qua cờ CLI --resume hoặc config train.checkpoint=true (mặc định TẮT).
    if args.resume or bool(cfg.get("train", {}).get("checkpoint", False)):
        cfg["train"]["checkpoint_path"] = str(out / "resume_checkpoint.pt")
    if args.resume:
        cfg["train"]["resume"] = True
    seed_everything(seed)
    try:
        import torch  
    except ImportError:
        print("[ERR] Chưa có torch — xem README bước 3 để cài đúng nền tảng.")
        return 1
    from uavcl.data import build_stream, describe_stream, get_source
    from uavcl.data.loaders import build_task_loaders
    from uavcl.engine import resolve_device, run_continual
    from uavcl.methods import build_method
    from uavcl.metrics import (average_accuracy, average_anytime_accuracy, average_forgetting,
                               backward_transfer, forward_transfer)
    from uavcl.models import ContinualClassifier, build_backbone

    device = resolve_device(cfg.get("device", "auto"))
    print(f"== G1 run: dataset={cfg['data']['name']} method={method_name} seed={seed} device={device.type}")

    # --- data ---
    t0 = time.time()
    source = get_source(cfg["data"])
    # #27 open-set: openset.enabled=true -> giữ lại `holdout` class KHÔNG BAO GIỜ train làm mẫu lạ.
    os_cfg = cfg.get("openset") or {}
    holdout_classes: list = []
    # None = KHÔNG phải stream bay lặp lại -> bỏ qua toàn bộ phần thước đo revisit ở cuối.
    # Phải khởi tạo ở đây vì chỉ nhánh 'revisit' mới gán, mà chỗ dùng thì nằm ngoài if/elif.
    mode_that = None
    if bool(os_cfg.get("enabled", False)):
        from uavcl.data.stream import build_stream_with_holdout

        stream, holdout_classes = build_stream_with_holdout(
            source.splits["train"].labels,
            source.splits["val"].labels,
            source.splits["test"].labels,
            num_classes=source.num_classes,
            num_tasks=int(cfg["data"]["num_tasks"]),
            seed=seed,
            shuffle_classes=bool(cfg["data"].get("shuffle_classes", True)),
            holdout=int(os_cfg.get("holdout", 5)),
        )
        print(f"[openset] class GIỮ LẠI (không train, làm mẫu lạ): {holdout_classes}")
    elif str(cfg["data"].get("stream_type", "class")).lower() == "revisit":
        # BAY LẶP LẠI — chia mẫu y hệt domain-incremental (mọi chuyến đủ mọi lớp); khác biệt
        # nằm ở ĐIỀU KIỆN: loaders.py lấy mức trôi theo LỊCH BAY tuần hoàn, nên điều kiện cũ
        # QUAY LẠI. Đây mới là bài toán thật (xem BAI_TOAN_VA_MUC_TIEU §1).
        from uavcl.data.revisit import lich_tu_cfg
        from uavcl.data.stream import build_domain_stream

        # A2 (KE_HOACH_SUA): revisit mà KHÔNG trôi thì mọi chuyến cùng điều kiện — mode_that
        # vẫn được tính và báo cáo O3 in ra như thật nhưng đo một hiện tượng không tồn tại.
        # Chặn ngay tại đây, trước khi đốt giờ máy.
        if not bool((cfg["data"].get("drift") or {}).get("enabled", False)):
            raise ValueError("stream_type=revisit cần data.drift.enabled=true — không có trôi "
                             "thì mọi chuyến cùng điều kiện, thước đo O3 vô nghĩa")
        stream = build_domain_stream(
            source.splits["train"].labels,
            source.splits["val"].labels,
            source.splits["test"].labels,
            num_classes=source.num_classes,
            num_tasks=int(cfg["data"]["num_tasks"]),
            seed=seed,
        )
        # A3: lịch bay qua MỘT nguồn duy nhất — loaders.py dựng lại từ đúng hàm này.
        _lich = lich_tu_cfg(len(stream), dict(cfg["data"].get("drift") or {}))
        mode_that = [c.mode_that for c in _lich]      # CHỈ để chấm điểm, model không thấy
        print(f"[stream] REVISIT: {len(stream)} chuyến bay × {source.num_classes} lớp "
              f"(điều kiện QUAY LẠI — {sum(1 for c in _lich if c.lan_gap_mode > 1)} chuyến lặp)")
        # DL1 — `test_chung`: bài toán nói bay lại CÙNG KHU VỰC. Mặc định chia test rời từng
        # chuyến (~1/T tập test) -> mỗi chuyến chấm trên ẢNH KHÁC NHAU và nhiễu đường chéo
        # cỡ ±2 điểm — sát cỡ hiệu ứng O3. Bật cờ này: MỌI chuyến chấm trên TOÀN BỘ tập test,
        # chỉ khác drift của chuyến -> đúng nghĩa "cùng cảnh, khác điều kiện", σ giảm ~√T lần.
        # Giá: eval nặng hơn T lần — chạy GPU. Mặc định TẮT.
        _p2cfg = dict(cfg["data"].get("pha2") or {})
        if bool(_p2cfg.get("test_chung", False)):
            _all_test = list(range(len(source.splits["test"].labels)))
            for _spec in stream:
                _spec.test_idx = list(_all_test)
            print(f"[revisit] test_chung BẬT — mọi chuyến chấm trên CÙNG {len(_all_test)} ảnh test")
        # P1: engine cần biết pha 2 không nhãn bắt đầu từ chuyến nào.
        if _p2cfg:
            cfg["train"]["pha2"] = _p2cfg
    elif str(cfg["data"].get("stream_type", "class")).lower() == "domain":
        # D5 — DOMAIN-incremental: mọi task đủ MỌI lớp, chỉ khác ĐIỀU KIỆN (do drift transform).
        # Tách hẳn khỏi class-incremental để khi accuracy tụt còn biết là do quên lớp hay do trôi.
        from uavcl.data.stream import build_domain_stream

        stream = build_domain_stream(
            source.splits["train"].labels,
            source.splits["val"].labels,
            source.splits["test"].labels,
            num_classes=source.num_classes,
            num_tasks=int(cfg["data"]["num_tasks"]),
            seed=seed,
        )
        print(f"[stream] DOMAIN-incremental: {len(stream)} task × {source.num_classes} lớp "
              "(mọi task đủ mọi lớp; khác nhau ở ĐIỀU KIỆN quan sát)")
    else:
        stream = build_stream(
            source.splits["train"].labels,
            source.splits["val"].labels,
            source.splits["test"].labels,
            num_classes=source.num_classes,
            num_tasks=int(cfg["data"]["num_tasks"]),
            seed=seed,
            shuffle_classes=bool(cfg["data"].get("shuffle_classes", True)),
        )
    print(describe_stream(stream, source.class_names))
    cfg["data"].setdefault("seed", seed)           # ↳ drift dùng seed này để tái lập nhiễu
    loaders = build_task_loaders(source, stream, cfg["data"])

    # --- model + method ---
    backbone, feat_dim = build_backbone(cfg["backbone"])
    head_kind = str(cfg.get("head", "linear")).lower()  # ↳ 'linear' (mặc định) | 'cosine' (fix recency bias của head)
    mem_cfg = cfg.get("memory", {}) or {}
    cms_on = bool((cfg.get("cms") or {}).get("enabled", False))
    if mem_cfg.get("enabled", False) and cms_on:
        # G4 — HOPE: Titans (tầng nhanh) + backbone-CMS (tầng trung/chậm)
        from uavcl.models.hope import HOPEClassifier

        model = HOPEClassifier(backbone, feat_dim, source.num_classes, mem_cfg, head=head_kind).to(device)
        if method_name != "hope":
            print(f"[WARN] memory+cms cùng bật + method '{method_name}' — thường dùng --method hope")
        print(f"[hope] seq={model.seq_mode} reset={model.reset_mode} chunk={model.memory.chunk_size}")
    elif mem_cfg.get("enabled", False):
        from uavcl.models.titans_head import TitansClassifier

        model = TitansClassifier(backbone, feat_dim, source.num_classes, mem_cfg, head=head_kind).to(device)
        if method_name not in ("titans", "finetune"):
            print(f"[WARN] memory.enabled + method '{method_name}' — thường dùng --method titans")
        print(f"[titans] seq={model.seq_mode} reset={model.reset_mode} chunk={model.memory.chunk_size}")
    elif method_name == "ncm":
        from uavcl.models.ncm import NCMClassifier

        model = NCMClassifier(backbone, feat_dim, source.num_classes).to(device)
    elif method_name == "slda":
        # #21: SLDA trên backbone frozen. Nếu config đang bật memory (vd config g2), chạy với
        # --set memory.enabled=false để rơi vào nhánh này (SLDA định nghĩa trên feature cố định).
        from uavcl.models.slda import SLDAClassifier

        slda_cfg = cfg.get("slda", {}) or {}
        model = SLDAClassifier(
            backbone, feat_dim, source.num_classes,
            shrinkage=float(slda_cfg.get("shrinkage", 1e-4)),
            # B2 ablation: streaming (mặc định) | identity (= NCM) | frozen (Σ đóng băng)
            cov_mode=str(slda_cfg.get("cov_mode", "streaming")),
            cov_freeze_after=int(slda_cfg.get("cov_freeze_after", 1)),
            stats_dtype=str(slda_cfg.get("stats_dtype", "float64")),
            # D1 — hệ số QUÊN. 1.0 = không quên = hành vi cũ.
            decay_mean=float(slda_cfg.get("decay_mean", 1.0)),
            decay_cov=float(slda_cfg.get("decay_cov", 1.0)),
            # M1/M2 — BA TẦNG (mặc định TẮT; xem KE_HOACH_SUA nhóm D/E).
            tang_nhanh=slda_cfg.get("tang_nhanh"),
            ngan_hang=slda_cfg.get("ngan_hang"),
        ).to(device)
        print(f"[slda] shrinkage={model.shrinkage:g} cov_mode={model.cov_mode}"
              + (f" cov_freeze_after={model.cov_freeze_after}" if model.cov_mode == "frozen" else "")
              + f" dtype={model.stats_dtype}"
              + (f" λ_μ={model.decay_mean:g} λ_Σ={model.decay_cov:g}"
                 if (model.decay_mean < 1 or model.decay_cov < 1) else " λ=1 (không quên)"))
        if model.tang_nhanh is not None:
            print(f"[ba_tang] tầng NHANH bật: kieu={model.tang_nhanh.kieu} "
                  f"λ={model.tang_nhanh.decay:g}"
                  + (f" | tầng TRUNG bật: nguong={model.ngan_hang.nguong:g} "
                     f"k_max={model.ngan_hang.k_max}" if model.ngan_hang is not None else
                     " | tầng TRUNG tắt"))
    else:
        model = ContinualClassifier(backbone, feat_dim, source.num_classes, head=head_kind).to(device)
    method = build_method(method_name, cfg)

    # --- run ---
    cms_cfg = cfg.get("cms") or {}
    if cms_cfg.get("enabled", False):
        cfg["train"]["cms"] = cms_cfg  # engine đọc từ train_cfg -> dùng CMSOptimizer (G3)
    R, log = run_continual(model, method, stream, loaders, device, cfg["train"])
    _chi_cheo = bool(cfg["train"].get("eval_chi_duong_cheo", False))  # ↳ tam giác dưới không đo

    # --- metrics + save ---
    metrics = {
        "dataset": source.name,
        "method": method_name,
        "seed": seed,
        "num_tasks": len(stream),
        "backbone": cfg["backbone"]["name"],
        "head": head_kind,
        "optimizer": str(cfg["train"].get("optimizer", "adamw")).lower(),
        "optimizer_per_task": bool(cfg["train"].get("optimizer_per_task", True)),
        # train.eval_chi_duong_cheo=true -> tam giác dưới của R KHÔNG được đo (toàn 0), nên
        # 4 chỉ số dưới đây MẤT NGHĨA. Ghi null thay vì số rác, kẻo về sau đọc nhầm.
        # Track revisit dùng O1/O2/O3 trong metrics_revisit.json, không dùng 4 cái này.
        "average_accuracy": None if _chi_cheo else average_accuracy(R),
        "average_anytime_accuracy": None if _chi_cheo else average_anytime_accuracy(R),
        "average_forgetting": None if _chi_cheo else average_forgetting(R),
        "backward_transfer": None if _chi_cheo else backward_transfer(R),
        # FWT chỉ có nghĩa khi bật train.eval_future (đo acc task kế tiếp TRƯỚC khi học)
        "forward_transfer": forward_transfer(R) if cfg["train"].get("eval_future", False) else None,
        # chi phí — so cùng accuracy thì method rẻ hơn thắng:
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "method_extra_floats": int(method.footprint_floats(model)),
        "runtime_sec": round(time.time() - t0, 1),
    }
    if mem_cfg.get("enabled", False):
        metrics["memory_reset"] = str(mem_cfg.get("reset", "image")).lower()
    if cms_cfg.get("enabled", False):
        metrics["cms_order"] = cms_cfg.get("order", "late_slow")
        metrics["cms_periods"] = "-".join(str(t[1]) for t in cms_cfg.get("tiers", []))
    out.mkdir(parents=True, exist_ok=True)  # out đã tính từ run_dir_name ở đầu main
    cols = [f"task{j}" for j in range(len(stream))]
    pd.DataFrame(R, index=[f"after_task{i}" for i in range(len(stream))], columns=cols) \
        .to_csv(out / "acc_matrix.csv", float_format="%.4f")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_config(cfg, out / "config.yaml")

    # #27: chấm open-set sau khi học xong — điểm = max cosine tới prototype các class đã học.
    if holdout_classes:
        from uavcl.data.loaders import build_eval_loader
        from uavcl.metrics import open_set_summary
        from uavcl.openset_eval import collect_openset_scores

        held = set(int(c) for c in holdout_classes)
        unseen_idx = [i for i, yy in enumerate(source.splits["test"].labels) if int(yy) in held]
        unseen_loader = build_eval_loader(source, unseen_idx, cfg["data"])
        genuine, impostor = collect_openset_scores(model, loaders, unseen_loader,
                                                   source.num_classes, device)
        osum = open_set_summary(genuine, impostor)
        osum.update({"holdout_classes": sorted(held), "n_genuine": len(genuine),
                     "n_impostor": len(impostor)})
        (out / "metrics_openset.json").write_text(json.dumps(osum, indent=2), encoding="utf-8")
        print(f"== OPEN-SET (#27): AUC={osum['auc']:.4f}  EER={osum['eer']:.4f}  "
              f"TAR@FAR=10%={osum['tar@far=10%']:.4f}  (unseen={sorted(held)})")
    if hasattr(model, "export_state"):  # G2 (S10): lưu "cục ký ức" cuối stream
        st = model.export_state()
        if st is not None:
            import torch as _torch

            _torch.save(st, out / "memory_state.pt")
            print(f"  memory_state.pt đã lưu (norm={model.state_norm():.4f})")

    print("\n== Ma trận accuracy (hàng = sau khi học task i, cột = đánh giá task j)")
    with np.printoptions(precision=3, suppress=True):
        print(np.tril(R))
    # --- BAY LẶP LẠI: thước đo riêng cho O1/O3 ------------------------------------------
    # AAA/Acc/Forget KHÔNG đo được O3 ("gặp lại điều kiện cũ thì nhận ra ngay"). Bài học từ
    # D10: λ hoạt động tốt nhưng AAA che mất hoàn toàn. Nên in RIÊNG, không trộn vào bảng cũ.
    if mode_that is not None:
        from uavcl.metrics import (in_bao_cao, loi_ich_quay_lai, thoi_gian_hoi_phuc,
                                   tom_tat_revisit)

        rv = tom_tat_revisit(R, mode_that)
        rv["mode_that"] = list(map(int, mode_that))
        # --- O2 (A5/P2): thời gian hồi phục từ chuỗi PREQUENTIAL của pha 2 ---------------
        # log["trace"][t] = acc từng batch của chuyến t, CHẤM TRƯỚC KHI cập nhật. Chỉ những
        # chuyến mà chế độ ĐỔI so với chuyến trước mới có "sự kiện đổi điều kiện" để đo.
        trace = log.get("trace") or {}
        if trace:
            hoi_phuc = {}
            for t_idx in sorted(trace):
                if t_idx >= 1 and mode_that[t_idx] != mode_that[t_idx - 1]:
                    hp = thoi_gian_hoi_phuc(trace[t_idx])
                    hoi_phuc[str(t_idx)] = (hp if hp != float("inf") else "inf")
            huu_han = [v for v in hoi_phuc.values() if v != "inf"]
            rv["o2_hoi_phuc_theo_chuyen"] = hoi_phuc
            rv["o2_hoi_phuc_tb_batch"] = (sum(huu_han) / len(huu_han)) if huu_han else None
            rv["o2_so_chuyen_khong_hoi_phuc"] = sum(1 for v in hoi_phuc.values() if v == "inf")
            rv["prequential"] = {str(k): [round(float(a), 4) for a in v]
                                 for k, v in trace.items()}
            # --- ⭐ O3 trên PREQUENTIAL — con số chính khi có trace ------------------------
            # Phát hiện từ mô phỏng đầu-cuối (scripts/mo_phong_ba_tang.py, 2026-08-04):
            # đường chéo R chấm SAU khi hấp thụ cả chuyến, lúc tầng nhanh ĐÃ tự hội tụ
            # (~3-4 batch với λ=0,99) -> R[t][t] của U1 và U2 gần trùng nhau và
            # `loi_ich_quay_lai` trên R BỊ MÙ với ngân hàng chế độ. Lợi ích "nhận ra ngay"
            # nằm ở ĐẦU chuyến -> đo trên trung bình prequential 10 batch đầu. Báo cáo
            # O3 = preq10(U2) − preq10(U1) cùng seed (so một biến, tự khử carryover EMA).
            ts = sorted(trace)
            if ts:
                K = 10
                modes_p2 = [int(mode_that[t]) for t in ts]

                def _R_gia(vals):
                    n = len(vals)
                    return [[vals[j] if j <= i else 0.0 for j in range(n)] for i in range(n)]

                m_dau = [sum(trace[t][:K]) / max(len(trace[t][:K]), 1) for t in ts]
                m_ca = [sum(trace[t]) / max(len(trace[t]), 1) for t in ts]
                r10 = loi_ich_quay_lai(_R_gia(m_dau), modes_p2)
                rca = loi_ich_quay_lai(_R_gia(m_ca), modes_p2)
                rv["o3_preq10_loi_ich"] = r10["loi_ich_quay_lai"]     # ⭐ số chính
                rv["o3_preq10_so_lan"] = r10["so_lan_quay_lai"]
                rv["o3_preq_ca_chuyen_loi_ich"] = rca["loi_ich_quay_lai"]
                rv["o3_preq10_theo_lan"] = {k: v for k, v in r10.items()
                                            if k.startswith("loi_ich_lan_")}
        metrics.update({f"revisit_{k}": v for k, v in rv.items()
                        if k not in ("mode_that", "prequential", "o2_hoi_phuc_theo_chuyen")})
        print("\n== BAY LẶP LẠI — thước đo cho mục tiêu O1/O3")
        print(in_bao_cao(rv))
        if trace and rv.get("o2_hoi_phuc_tb_batch") is not None:
            print(f"  Thời gian hồi phục     : {rv['o2_hoi_phuc_tb_batch']:.1f} batch (O2) "
                  f"trên {len(rv['o2_hoi_phuc_theo_chuyen'])} lần đổi điều kiện"
                  + (f" · {rv['o2_so_chuyen_khong_hoi_phuc']} chuyến KHÔNG hồi phục ⚠️"
                     if rv["o2_so_chuyen_khong_hoi_phuc"] else ""))
            print("  ⚠️ O2 tính so với mức ổn định CỦA CHÍNH arm — arm đứng im ở mức thấp "
                  "cũng 'hồi phục' nhanh. Luôn đọc O2 KÈM acc.")
        if trace and "o3_preq10_loi_ich" in rv:
            print(f"  Lợi ích quay lại PREQ10: {rv['o3_preq10_loi_ich']:+.4f} (⭐ O3 số chính, "
                  f"{rv['o3_preq10_so_lan']} lần gặp lại) · cả chuyến {rv['o3_preq_ca_chuyen_loi_ich']:+.4f}")
            print("  ⚠️ O3 trên đường chéo R bị MÙ với ngân hàng chế độ (chấm cuối chuyến, "
                  "tầng nhanh đã tự hội tụ) — công bố bằng hiệu PREQ10 giữa U2 và U1.")
        # BA TẦNG: bằng chứng chạy đúng, in một dòng mỗi tầng (như window_report của SLDA).
        if getattr(model, "tang_nhanh", None) is not None:
            bc = model.tang_nhanh.bao_cao()
            print(f"  [ba_tang] tầng nhanh: {bc['so_mau']} mẫu, độ trôi hiện tại "
                  f"{bc['do_troi_hien_tai']:.4f}, {bc['bytes'] / 1024:.1f} KB")
        if getattr(model, "ngan_hang", None) is not None:
            bc = model.ngan_hang.bao_cao()
            rv["ngan_hang"] = bc
            print(f"  [ba_tang] tầng trung: {bc['so_che_do']} chế độ "
                  f"(số điều kiện THẬT: {len(set(mode_that))}) · nạp lại {bc['so_lan_nap']} lần "
                  f"· {bc['bytes'] / 1024:.1f} KB")
            if bc["so_che_do"] > 2 * len(set(mode_that)):
                print("  ⚠️ số chế độ NỔ so với số điều kiện thật — giảm nguong hoặc xem lại T1")
        # Ghi file SAU CÙNG để gồm cả o2_* lẫn báo cáo ngân hàng chế độ.
        (out / "metrics_revisit.json").write_text(json.dumps(rv, indent=2), encoding="utf-8")

    print("\n== Kết quả")
    if _chi_cheo:
        # eval_chi_duong_cheo: tam giác dưới KHÔNG đo -> 3 chỉ số này vô nghĩa, đã ghi null.
        print("  (train.eval_chi_duong_cheo=true — Average Accuracy / Forgetting / BWT không đo;")
        print("   dùng O1/O2/O3 trong metrics_revisit.json)")
    else:
        print(f"  Average Accuracy   : {metrics['average_accuracy']:.4f}  (cao = tốt)")
        print(f"  Average Forgetting : {metrics['average_forgetting']:.4f}  (thấp = tốt — con số dự án cần giảm)")
        print(f"  Backward Transfer  : {metrics['backward_transfer']:.4f}  (âm = quên)")

    # ĐÒN A: nếu bật train.eval_ncm_head, log có ma trận NCM-head -> tính + lưu + so sánh.
    if "ncm_R" in log:
        Rn = np.asarray(log["ncm_R"])
        ncm_metrics = {
            "readout": "ncm_head_posthoc",
            "average_accuracy": average_accuracy(Rn),
            "average_forgetting": average_forgetting(Rn),
            "backward_transfer": backward_transfer(Rn),
        }
        pd.DataFrame(Rn, index=[f"after_task{i}" for i in range(len(stream))], columns=cols) \
            .to_csv(out / "acc_matrix_ncm.csv", float_format="%.4f")
        (out / "metrics_ncm.json").write_text(json.dumps(ncm_metrics, indent=2), encoding="utf-8")
        print("\n== ĐÒN A — NCM-head (prototype trên feature sau memory, KHÔNG train lại)")
        print(f"  Linear-head : Acc {metrics['average_accuracy']:.4f} | Forget {metrics['average_forgetting']:.4f}")
        print(f"  NCM-head    : Acc {ncm_metrics['average_accuracy']:.4f} | Forget {ncm_metrics['average_forgetting']:.4f}")
        print(f"  Mốc NCM gốc : Acc 0.6933 | Forget 0.1000  (vượt được = head Linear là nút thắt)")

    print(f"  Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
