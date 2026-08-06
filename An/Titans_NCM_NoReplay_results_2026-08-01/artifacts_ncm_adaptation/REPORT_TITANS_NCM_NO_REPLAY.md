# Titans + NCM no-replay adaptation campaign

- Selected experimental gamma: `1`
- Transport: `{'type': 'diagonal', 'regularization': 10.0, 'beta': 0.5, 'validation_score': 0.8476190434561836, 'mean_gate_improvement': 0.00528194842627272, 'accepted_tasks': 8}`
- Fixed-anchor distillation weight: `0.1`
- Selection protocol: mean current-task validation; no old validation/test revisit

| Group | Valid seeds | Accuracy mean +/- std | Forgetting mean +/- std |
|---|---:|---:|---:|
| Blend | 3 | 0.6166 +/- 0.0185 | 0.2689 +/- 0.0269 |
| Full combination | 3 | 0.6898 +/- 0.0100 | 0.1711 +/- 0.0113 |

Baselines đã xác nhận trước campaign: Frozen NCM `0.7114`, Titans NCM Online `0.6190`, Post-hoc oracle `0.7459`.

Kết luận khoa học cuối chỉ được viết sau khi đủ 3 seed và hậu kiểm no-replay/NaN.
