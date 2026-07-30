# Kết quả so sánh M3 practical và M3 paper

Báo cáo này được tạo tự động bởi `scripts/run_m3_paper_study.py summarize`.
LR được chọn bằng validation; test metrics chỉ dùng để báo cáo sau khi khóa cấu hình.

Trạng thái: **đã đủ EuroSAT + RESISC45**.

## Biến thể

- P0: `M3-delta-approx-clip`, baseline chính không đổi semantics.
- P1: `M3-paper-stabilized`, paper accumulation/timing + cơ chế ổn định của project.
- P2: `M3-paper-strict`, paper accumulation/timing không weight decay/gradient clip/update clip.
- P3: `M3-EMA-clip`, đối chứng chẩn đoán.

## Kết quả hiện có

| Dataset | Variant | LR | Seeds hữu hạn | Accuracy mean +/- sd | Forgetting mean +/- sd | BWT | FWT | Runtime (s) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| eurosat | M3-delta-approx-clip | 0.003 | 0,1,2 | 0.5897 +/- 0.1771 | 0.2267 +/- 0.3725 | -0.2022 | 0.0734 | 875.7 |
| eurosat | M3-delta-approx-clip | 0.005 | 0,1,2 | 0.6724 +/- 0.0703 | 0.1561 +/- 0.2213 | -0.1402 | 0.0473 | 904.4 |
| eurosat | M3-paper-stabilized | 0.0001 | 0 | 0.2128 +/- 0.0000 | 0.1588 +/- 0.0000 | -0.1548 | 0.2617 | 742.2 |
| eurosat | M3-paper-stabilized | 0.0003 | 0 | 0.1909 +/- 0.0000 | 0.1506 +/- 0.0000 | -0.1041 | 0.1695 | 742.2 |
| eurosat | M3-paper-stabilized | 0.001 | 0 | 0.2672 +/- 0.0000 | 0.1005 +/- 0.0000 | -0.0905 | 0.0984 | 757.4 |
| eurosat | M3-paper-stabilized | 0.003 | 0 | 0.1699 +/- 0.0000 | 0.1246 +/- 0.0000 | -0.1216 | 0.1339 | 748.5 |
| eurosat | M3-paper-stabilized | 0.005 | 0,1,2 | 0.3953 +/- 0.1121 | 0.0518 +/- 0.1764 | 0.0535 | 0.0501 | 764.1 |
| eurosat | M3-paper-strict | 1e-07 | 0 | 0.1927 +/- 0.0000 | 0.1193 +/- 0.0000 | -0.1183 | 0.2360 | 709.4 |
| eurosat | M3-paper-strict | 3e-07 | 0 | 0.1971 +/- 0.0000 | 0.1916 +/- 0.0000 | -0.1871 | 0.2470 | 715.8 |
| eurosat | M3-paper-strict | 1e-06 | 0 | 0.2485 +/- 0.0000 | 0.1930 +/- 0.0000 | -0.1680 | 0.2111 | 710.6 |
| eurosat | M3-paper-strict | 3e-06 | 0,1,2 | 0.2660 +/- 0.0630 | 0.2647 +/- 0.0971 | -0.2208 | 0.1473 | 755.2 |
| eurosat | M3-paper-strict | 1e-05 | 0 | 0.3157 +/- 0.0000 | 0.5677 +/- 0.0000 | -0.5659 | 0.0538 | 711.0 |
| eurosat | M3-paper-strict | 3e-05 | 0 | 0.3397 +/- 0.0000 | 0.6292 +/- 0.0000 | -0.6292 | 0.2336 | 709.7 |
| eurosat | M3-paper-strict | 0.0001 | 0 | 0.1882 +/- 0.0000 | 0.5907 +/- 0.0000 | -0.5907 | 0.0447 | 710.1 |
| eurosat | M3-EMA-clip | 0.005 | 0 | 0.7492 +/- 0.0000 | 0.0415 +/- 0.0000 | 0.0010 | 0.1010 | 733.0 |
| resisc45 | M3-delta-approx-clip | 0.005 | 0,1,2 | 0.7233 +/- 0.0164 | 0.0488 +/- 0.0210 | 0.0986 | 0.0007 | 2567.9 |
| resisc45 | M3-paper-stabilized | 0.005 | 0,1,2 | 0.1414 +/- 0.0928 | 0.2015 +/- 0.0634 | -0.0202 | 0.0000 | 2548.2 |
| synthetic | M3-delta-approx-clip | 0.003 | 0 | 0.7500 +/- 0.0000 | -0.5000 +/- 0.0000 | 0.5000 | n/a | 1.4 |
| synthetic | M3-paper-stabilized | 0.001 | 0 | 0.0000 +/- 0.0000 | 0.5000 +/- 0.0000 | -0.5000 | n/a | 1.3 |
| synthetic | M3-paper-strict | 1e-06 | 0 | 0.2500 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 | n/a | 1.4 |

## Kiểm chứng triển khai

- Unit/integration tests: công thức một bước, chunk timing, không bias-correct V, strict không clip, CMS clock và baseline regression.
- P1/P2 dùng `paper_timing=next_chunk`; mode `legacy_boundary` chỉ giữ để tái lập artifact cũ.
- Run identity dùng `experiment.tag`; P1/P2 không thể ghi đè nhau.
- Mỗi run lưu config, git/environment metadata, validation metrics, failure record và optimizer diagnostics.
- Diagnostics gồm gradient trước/sau clip, M1/M2/V/O1/O2, denominator, raw/post update, clip rate và drift theo CMS tier.

Run thất bại đã ghi nhận: **0**.

## Lưu ý

- `M3-paper-stabilized` giữ weight decay và hai lớp clipping của project.
- `M3-paper-strict` bỏ weight decay, global gradient clipping và update clipping.
- Mode paper là bản bám pseudocode với policy tensor 1D của project, không phải code gốc 100% của tác giả.
- Không kết luận variant thắng từ kết quả synthetic smoke; cần hoàn tất EuroSAT và stability gate trước RESISC45.

## Lệnh tiếp tục

```bash
python3 scripts/run_m3_paper_study.py eurosat-sweep
python3 scripts/run_m3_paper_study.py eurosat-replicate
python3 scripts/run_m3_paper_study.py resisc45
python3 scripts/run_m3_paper_study.py summarize
```

Runner mặc định `--skip-existing`, nên có thể chạy lại cùng lệnh để resume.

## Kết luận cuối

Giữ **P0 `M3-delta-approx-clip` (M3 practical của project)** làm kết quả chính.
Trong giao thức hiện tại, bản paper không cải thiện accuracy hay chống quên:

| Dataset | P0 practical | P1 paper-stabilized | Chênh lệch accuracy P0 - P1 |
|---|---:|---:|---:|
| EuroSAT | 0.6724 +/- 0.0703 | 0.3953 +/- 0.1121 | +0.2771 |
| RESISC45 | 0.7233 +/- 0.0164 | 0.1414 +/- 0.0928 | +0.5819 |

- P0 trên RESISC45 cũng quên ít hơn: `0.0488 +/- 0.0210`, so với P1 `0.2015 +/- 0.0634`.
- P2 strict không NaN ở LR rất nhỏ `3e-6`, nhưng EuroSAT chỉ đạt `0.2660 +/- 0.0630` với forgetting `0.2647 +/- 0.0971`; không đủ điều kiện để chạy RESISC45.
- Không có `failure.json` trong toàn bộ study. Các run đã hoàn thành hữu hạn; practical giữ norm state và CMS tier drift trong vùng ổn định ở RESISC45.

Điều này không chứng minh pseudocode Nested Learning sai. Nó cho thấy khi ghép trực tiếp paper accumulation/timing vào backbone, stream, lịch học và các policy tensor của project, hyperparameter và dynamics không còn phù hợp. Do đó paper mode nên được giữ làm implementation research baseline; không thay thế P0 cho các kết quả chính.

Hướng cải thiện hợp lý tiếp theo là tune paper mode độc lập (LR, frequency, alpha/beta và warm-up), rồi chỉ so sánh lại khi nó vượt qua EuroSAT 3 seed. Không nên diễn giải kết quả hiện tại là lợi thế của “paper chuẩn”.
