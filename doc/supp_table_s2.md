# Supplementary Table S2

Per-epoch GroupPooling ablation on seed 190223: max-pool (main method) vs. mean-pool. All R@1 values are percentages on the ConPR and ConSLAM test sets, evaluated from the per-epoch checkpoint with no test-set selection. Val R@1 is measured on our GSV-Cities validation split. ★ marks the val-best epoch used for reporting (max-pool: epoch 0; mean-pool: epoch 0).

| Epoch | Val R@1 (max) | Val R@1 (mean) | ConPR (max) | ConPR (mean) | ΔConPR | ConSLAM (max) | ConSLAM (mean) | ΔConSLAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 (max★/mean★) | 65.06 | 65.28 | 80.92 | 80.60 | -0.32 | 61.11 | 59.38 | -1.73 |
| 01 | 63.64 | 64.14 | 78.69 | 78.59 | -0.10 | 60.42 | 60.42 | +0.00 |
| 02 | 63.95 | 64.05 | 79.65 | 79.47 | -0.18 | 57.29 | 55.90 | -1.39 |
| 03 | 62.09 | 63.26 | 78.92 | 78.56 | -0.36 | 57.64 | 58.68 | +1.04 |
| 04 | 62.15 | 62.66 | 79.42 | 78.81 | -0.61 | 58.68 | 57.99 | -0.69 |
| 05 | 62.69 | 62.47 | 78.58 | 78.18 | -0.40 | 58.33 | 58.68 | +0.35 |
| 06 | 63.60 | 62.44 | 78.99 | 78.75 | -0.24 | 56.60 | 57.99 | +1.39 |
| 07 | 63.57 | 62.31 | 79.04 | 78.96 | -0.08 | 58.68 | 60.42 | +1.74 |
| 08 | 63.67 | 62.15 | 78.77 | 78.80 | +0.03 | 57.64 | 59.72 | +2.08 |
| 09 | 63.60 | 62.34 | 78.55 | 78.72 | +0.17 | 59.72 | 60.42 | +0.70 |

**Val-best summary** (seed 190223):
- max-pool @ epoch 0: ConPR 80.92% / ConSLAM 61.11%
- mean-pool @ epoch 0: ConPR 80.60% / ConSLAM 59.38%
- Δ(mean − max) at common val-best epoch 0: **ConPR -0.32, ConSLAM -1.73** R@1 (pp)
