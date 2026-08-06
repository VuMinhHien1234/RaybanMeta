# Confirmatory evaluation: Anchored Blend gamma 0.25

- Gamma was fixed to `0.25` before these runs.
- Fresh seeds: `3, 4, 5`.
- Transport: disabled.
- Feature distillation: disabled.
- Old train replay: forbidden and audited.

| Seed | Accuracy | Forgetting | Frozen control | Gain over Frozen |
|---:|---:|---:|---:|---:|
| 3 | 0.7210 | 0.0979 | 0.7114 | +0.0095 |
| 4 | 0.7219 | 0.0968 | 0.7114 | +0.0105 |
| 5 | 0.7190 | 0.0857 | 0.7114 | +0.0076 |
| **Mean +/- std** | **0.7206 +/- 0.0015** | **0.0935 +/- 0.0067** | **0.7114 +/- 0.0000** | **+0.0092 +/- 0.0015** |

This report is confirmatory for the pre-registered gamma only; no gamma was re-selected from these test results.
