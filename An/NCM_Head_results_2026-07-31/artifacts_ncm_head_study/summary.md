# NCM Head study

## Per-seed results

| Seed | Frozen NCM Acc/Fgt | Titans Linear Acc/Fgt | Titans NCM Online Acc/Fgt | Titans NCM Post-hoc Acc/Fgt | Online-Frozen | Oracle-Online |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7114/0.0846 | 0.6079/-0.0307 | 0.6143/0.2725 | 0.7454/0.0671 | -0.0971 | +0.1311 |
| 1 | 0.7114/0.0854 | 0.6006/-0.0068 | 0.6073/0.2825 | 0.7387/0.0679 | -0.1041 | +0.1314 |
| 2 | 0.7114/0.0921 | 0.6222/-0.0379 | 0.6356/0.2464 | 0.7537/0.0618 | -0.0759 | +0.1181 |

## Aggregate

- Valid seeds: 3/3
- Frozen NCM accuracy: 0.7114 +/- 0.0000
- Titans Linear accuracy: 0.6103 +/- 0.0110
- Titans NCM Online accuracy: 0.6190 +/- 0.0147
- Titans NCM Post-hoc accuracy: 0.7459 +/- 0.0075
- Titans Online gain over Frozen NCM: -0.0924 +/- 0.0147
- Post-hoc oracle minus Online gap: 0.1269 +/- 0.0076
- Final online/post-hoc prototype cosine: 0.9158 +/- 0.0063

Post-hoc được phép đọc lại toàn bộ train data đã thấy; Online và Frozen NCM không được phép.
Chỉ Titans NCM Online so với Frozen NCM là phép so chính cho head bounded/deployable.
