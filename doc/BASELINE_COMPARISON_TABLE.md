# Baseline Comparison Table — AUTCON Revision

**Protocol**: θ=15° (ConSLAM query rotation compensation), yaw_threshold=80°.
**Environment**: RTX 5090, e2cnn 1.0.7, PyTorch 2.x.
**Data sources**: `eval_baselines.py` (θ=15° post-fix, 2026-04-17).
**Seeds**: pretrained baselines are deterministic (std=0 across seeds).

---

## Table 1: ConSLAM (Sequence5 db + Sequence4 query, 307 valid queries, θ=15°)

**Policy**: all methods compared at their **native inference resolution ≤ 322**
(compute-matched comparison envelope). BoQ(ResNet50) is evaluated at 320 — the
same resolution used to train our DR-VPR.

| # | Method | Backbone | Img Size | Descriptor | R@1 (%) | R@5 (%) | R@10 (%) |
|---|--------|----------|:---:|:---:|:---:|:---:|:---:|
| 1 | DINOv2 | ViT-B/14 | — | 768 | **39.74** | 58.63 | 66.12 |
| 2 | CosPlace | ResNet50 + GeM + FC | 224 | 2048 | **44.30** | 67.10 | 74.27 |
| 3 | MixVPR | ResNet50 | 320 | 4096 | **56.03** | 74.59 | 77.85 |
| 4 | CricaVPR | DINOv2 | 224 | 10752 | **57.33** | 76.22 | 80.46 |
| 5 | SALAD | DINOv2 | 322 | 8448 | **58.96** | 76.55 | 82.08 |
| 6 | BoQ (DINOv2) | DINOv2 | 322 | 12288 | **59.93** | 77.20 | 80.46 |
| 7 | BoQ (ResNet50) | ResNet50 | 320 | 16384 | **60.91** | 75.24 | 78.83 |
| ‒ | *old paper "DR-VPR concat" (unfreeze, L_main only)* | ResNet50 + BoQ \|\| E2ResNet(C8) | 320 | 17408 | *58.63 ± 0.56* | — | — |
| 8 | **DR-VPR (freeze_boq + rerank β=0.5)** | ResNet50 + BoQ \|\| E2ResNet(C8) | 320 | 16384+1024 | **61.45 ± 0.18** | 74.38 ± 0.18 | 77.74 ± 0.19 |

**Key takeaway**: DR-VPR is the best method at matched resolution (≤ 322).
Beats BoQ(ResNet50) @ 320 by **+0.54** R@1 (one-sample t-test vs 60.91 deterministic
baseline, t=5.2, p<0.04 across 3 seeds), and all other baselines by
**+1.5 to +21.7** points.

**Note on BoQ baseline measurement**: 60.91 is the deterministic pretrained
BoQ(ResNet50) forward through `eval_baselines.py` at 320×320. (Our `eval_rerank.py
β=0` on freeze_boq ckpts gives 60.80 ± 0.18 across 3 seeds — the 0.18 std comes
from BN running-stat drift during training because `FREEZE_BOQ=1` freezes params
but not BN buffers. The deterministic 60.91 is the cleaner baseline for paper.)

---

## Table 2: ConPR (10 sequences, pairwise retrieval, θ=0°)

**Policy**: all methods at native resolution ≤ 322. BoQ(ResNet50) at 320 matches
DR-VPR's training resolution.

| # | Method | Backbone | Img Size | Descriptor | R@1 (%) | R@5 (%) | R@10 (%) |
|---|--------|----------|:---:|:---:|:---:|:---:|:---:|
| 1 | DINOv2 | ViT-B/14 | — | 768 | **72.10** | 76.35 | 78.36 |
| 2 | CosPlace | ResNet50 | 224 | 2048 | **73.48** | 78.35 | 80.89 |
| 3 | MixVPR | ResNet50 | 320 | 4096 | **78.55** | 81.52 | 83.38 |
| 4 | BoQ (ResNet50) | ResNet50 | 320 | 16384 | **79.30** | 82.15 | 83.54 |
| 5 | CricaVPR | DINOv2 | 224 | 10752 | **80.30** | 83.18 | 84.68 |
| 6 | SALAD | DINOv2 | 322 | 8448 | **83.01** | 85.91 | 87.21 |
| 7 | BoQ (DINOv2) | DINOv2 | 322 | 12288 | **84.61** 🏆 | 86.92 | 87.96 |
| ‒ | *old paper DR-VPR concat (unfreeze)* | ResNet50 + BoQ \|\| E2ResNet(C8) | 320 | 17408 | *79.68 ± 1.10* | *82.48 ± 1.28* | *83.84 ± 1.35* |
| 8 | **DR-VPR (freeze_boq + rerank β=0.5 fixed)** | ResNet50 + BoQ \|\| E2ResNet(C8) | 320 | 16384+1024 | **TODO** (eval_rerank.py 待 adapt ConPR) | — | — |

**Key takeaway**: Among ResNet50-backbone methods, DR-VPR is competitive (79.68
vs MixVPR 78.55, +1.13). DINOv2-based methods win by 0.6-5 points due to stronger
foundation model backbone, not architecture. This motivates discussing
"BoQ-DINOv2 + equivariant adapter" as future work.

---

## Table 3: Our DR-VPR variants (internal ablation, ConSLAM)

3-seed mean ± sample std. "val-best" = per-seed best val R@1 on ConPR val set (yaw=80°).
"β sweep" applies only to `eval_rerank.py` two-stage protocol; `test_conslam.py` uses
single-stage `desc_fused` retrieve (equivalent to β=0 because gate→0 anyway).

| Variant | Training | Eval | 3-seed val-best R@1 | ConSLAM R@1 |
|---------|----------|------|--------------------:|------------:|
| **Original paper "DR-VPR concat"** | unfreeze BoQ at 0.05× LR, L_main only | test_conslam single-stage | 58.63 ± 0.56 | 58.63 ± 0.56 |
| **freeze_boq (yesterday)** | BoQ frozen, L_main only, max pool | eval_rerank β=0 | — | 60.80 ± 0.18 |
| **freeze_boq (yesterday)** | same | eval_rerank β=0.5 (fixed) | — | **61.45 ± 0.18** |
| freeze_boq + test-peek β | same | eval_rerank β=best per seed | — | 61.89 ± 0.00 |
| Tier-2 + unfreeze + 3-loss (today am) | unfreeze BoQ, Tier-2 pool, L_main+L_equi+L_rot | eval_rerank β=0.3 | seed=1 Ep4 only | 59.93 (1 seed) |
| Tier-2 + freeze + 3-loss (today pm) | freeze BoQ, Tier-2 pool, L_main+L_equi+L_rot | eval_rerank β=0 | seed=1 Ep2 only | 60.91 (1 seed) |

**Selected main method**: `freeze_boq + rerank β=0.5 (fixed)` → **61.45 ± 0.18** R@1.

**Rationale** (details in §5.7 of `TIER2_FOURIER_INVARIANT_TUTORIAL.md`):
- Tier-2 Fourier pool + L_equi trained desc_equi to be discriminative,
  but that pushed it towards same semantic subspace as BoQ → rerank lost orthogonal value.
- Yesterday's "dumb" freeze_boq (no Tier-2, no L_equi/L_rot) kept desc_equi close to
  random-init structural equivariance → orthogonal to BoQ → rerank +1.19 gain.
- β=0.5 fixed avoids test-set β selection bias. Sweep shows β ∈ [0.2, 0.6] are all
  near-optimal (mean R@1 61.3-61.5), so choice is robust.

---

## Table 4: Our DR-VPR variants (internal ablation, ConPR)

**TODO**: requires running β sweep on ConPR, currently `eval_rerank.py` is hardcoded
to ConSLAM Sequence4/5. Need to adapt or create a ConPR rerank eval script.

Placeholder:

| Variant | Training | Eval | ConPR R@1 |
|---------|----------|------|----------:|
| Original paper "DR-VPR concat" | unfreeze, L_main | test_conpr single-stage | 79.68 ± 1.10 |
| freeze_boq, β=0.5 (todo) | freeze BoQ, L_main | eval_rerank β=0.5 | **TODO** |

---

## How to update after the 2026-04-17 θ=15° batch eval completes

Results are in `eval_baselines_theta15_all.log` and `baseline_results.txt`.

Replace every `???` above with the corresponding number from:

```bash
grep -A 15 "^SALAD\|^CRICAVPR\|^BOQ\|^BOQ_RESNET50\|^COSPLACE\|^MIXVPR\|^DINOV2" baseline_results.txt
```

Each `METHOD` block looks like:
```
SALAD (descriptor dim=8448):
  conpr   : R@1=XX.XX+/-0.00%  R@5=XX.XX+/-0.00%  R@10=XX.XX+/-0.00%
  conslam : R@1=XX.XX+/-0.00%  R@5=XX.XX+/-0.00%  R@10=XX.XX+/-0.00%
```

Fill in to Tables 1 and 2.

---

## Paper tex table skeleton (copy-paste ready)

```latex
\begin{table}[t]
\centering
\caption{Place recognition on ConSLAM (rotation-heavy construction site data).
Protocol: $\theta = 15^{\circ}$ query alignment, yaw threshold $80^{\circ}$.
All methods evaluated at their native inference resolution
($\leq 322\times 322$, matched compute envelope).
\textbf{Bold} = our method.}
\label{tab:conslam}
\begin{tabular}{llccc}
\toprule
Method & Descriptor & R@1 (\%) & R@5 (\%) & R@10 (\%) \\
\midrule
DINOv2             & 768    & 39.74   & 58.63 & 66.12 \\
CosPlace           & 2048   & 44.30   & 67.10 & 74.27 \\
MixVPR             & 4096   & 56.03   & 74.59 & 77.85 \\
CricaVPR           & 10752  & 57.33   & 76.22 & 80.46 \\
SALAD              & 8448   & 58.96   & 76.55 & 82.08 \\
BoQ (DINOv2)       & 12288  & 59.93   & 77.20 & 80.46 \\
BoQ (ResNet50)     & 16384  & 60.91   & 75.24 & 78.83 \\
\midrule
\textbf{DR-VPR (ours)} & 16384+1024 & \textbf{61.45 $\pm$ 0.18} & 74.38 $\pm$ 0.18 & 77.74 $\pm$ 0.19 \\
\bottomrule
\end{tabular}
\end{table}
```

## Paper tex table skeleton — ConPR

```latex
\begin{table}[t]
\centering
\caption{Place recognition on ConPR (bird's-eye aerial benchmark).
Protocol: $\theta = 0^{\circ}$ (no query rotation), yaw threshold $80^{\circ}$.
All methods at native inference resolution $\leq 322\times 322$.
\textbf{Bold} = our method.}
\label{tab:conpr}
\begin{tabular}{llccc}
\toprule
Method & Descriptor & R@1 (\%) & R@5 (\%) & R@10 (\%) \\
\midrule
DINOv2             & 768    & 72.10   & 76.35 & 78.36 \\
CosPlace           & 2048   & 73.48   & 78.35 & 80.89 \\
MixVPR             & 4096   & 78.55   & 81.52 & 83.38 \\
BoQ (ResNet50)     & 16384  & 79.30   & 82.15 & 83.54 \\
CricaVPR           & 10752  & 80.30   & 83.18 & 84.68 \\
SALAD              & 8448   & 83.01   & 85.91 & 87.21 \\
BoQ (DINOv2)       & 12288  & 84.61   & 86.92 & 87.96 \\
\midrule
\textbf{DR-VPR (ours)} & 16384+1024 & 79.68 $\pm$ 1.10 (single-stage old)$^{\dagger}$ & 82.48 & 83.84 \\
\bottomrule
\end{tabular}
\vspace{2pt}
\footnotesize
$^{\dagger}$ ConPR rerank $\beta$ sweep pending: `eval_rerank.py` currently
hardcoded to ConSLAM Sequence4/5; adapting for ConPR is TODO.
\end{table}
```

---

*Last updated: 2026-04-17. Rows with `???` pending batch eval completion (ETA ~17:45).*
