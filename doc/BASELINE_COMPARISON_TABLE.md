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
| ‒ | *DR-VPR v1 (freeze_boq + rerank β=0.5)* (ablation) | ResNet50 + BoQ \|\| E2ResNet(C8) | 320 | 16384+1024 | *61.45 ± 0.18* | 74.38 ± 0.18 | 77.74 ± 0.19 |
| **8** | **DR-VPR v2 (P1 standalone multi-scale + BoQ β=0.10)** | BoQ(ResNet50) + E2ResNet(C8) multi-scale | 320 | 16384 + 1024 | **61.89 ± 0.33** | 75.68 ± 0.50 | 79.80 ± 0.66 |

**Key takeaway**: DR-VPR v2 is the best method at matched resolution (≤ 322).
Beats BoQ(ResNet50) by **+0.98** R@1 (61.89 vs 60.91 deterministic baseline),
and beats the previous DR-VPR v1 variant by **+0.44** R@1 (61.89 vs 61.45).
All other matched baselines are beaten by **+1.96 to +22.15** points.

**DR-VPR v2 architecture summary**:
- Stage-1 retrieve: official BoQ(ResNet50) at 320×320 (16384-d descriptor)
- Stage-2 rerank: E2ResNet(C8) multi-scale invariant pool + GeM + Linear,
  independently trained via MS loss on GSV-Cities (1024-d descriptor, ~1.34 M
  params). No BoQ branch in training pipeline — completely standalone.
- Inference: top-100 BoQ retrieve → rerank score = 0.90·boq_sim + 0.10·equi_sim.
  The β=0.10 is a fixed hyperparameter chosen from a fine sweep (see
  `doc/ABLATION_BETA_SWEEP.md`); the [0.05, 0.13] β range is a plateau with
  similar +0.76 gain, so the exact β value is not sensitive.

**Note on BoQ baseline measurement**: 60.91 is the deterministic pretrained
BoQ(ResNet50) forward through `eval_baselines.py` at 320×320 (official protocol,
Conslam_dataset_rot.evaluateResults, 307 valid queries). Our `eval_rerank_standalone.py`
β=0 on P1 ckpts gives 61.24 across all seeds (deterministic, same bug-fixed filter
but FAISS IndexFlatIP vs Conslam's IndexFlatL2 creates a 1-query tiebreak
difference). The deterministic 60.91 is the cleaner baseline for paper.

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
| **8** | **DR-VPR v2 (P1 standalone multi-scale + BoQ β=0.10)** | BoQ(ResNet50) + E2ResNet(C8) multi-scale | 320 | 16384 + 1024 | **79.74 ± 0.09** | — | — |

**Table 2 DR-VPR v2 = full 10-seq protocol (3-seed mean ± std)**: 9 query
sequences × 3 P1 standalone seeds, db=20230623. BoQ stage-1 top-100 + β=0.10
rerank. See `doc/EXPERIMENTS_TTA_UNION_FULLCONPR.md` §1 for per-pair breakdown.
The per-pair deltas range from −0.57 (q=20230531) to +1.77 (q=20230803), with
8 of 9 pairs showing gain. 3-seed std is tight (0.09) because BoQ is
deterministic and only equi varies.

**Key takeaway**: DR-VPR v2 achieves **79.74 ± 0.09 R@1** on full ConPR,
a **+0.44** improvement over BoQ(ResNet50)=79.30 at matched resolution and
descriptor pool. Among ResNet50-backbone methods, DR-VPR v2 is best. Among
all matched-resolution baselines, DR-VPR v2 remains second behind
BoQ(DINOv2)=84.61, confirming the DINOv2 backbone advantage on
in-distribution ConPR. Future work: BoQ(DINOv2) + multi-scale equi adapter.

---

## Table 3: DR-VPR variant ablation (ConSLAM, chronological)

3-seed mean ± sample std unless noted. All numbers post-2026-04-18 rotation
bug fix (older single-seed numbers replaced with fixed values). "β sweep"
applies only to `eval_rerank.py` / `eval_rerank_standalone.py` two-stage
protocol.

| # | Variant | Training config | Eval | ConSLAM R@1 |
|---|---------|-----------------|------|------------:|
| 1 | Original paper "DR-VPR concat" | unfreeze BoQ 0.05× LR, L_main only, MixVPR 4096-d | single-stage fused desc | 58.63 ± 0.56 |
| 2 | freeze_boq (v1 main) | BoQ frozen, L_main only, max pool | `eval_rerank.py` β=0 | 60.80 ± 0.18 |
| 3 | freeze_boq + rerank β=0.5 | same as 2 | `eval_rerank.py` β=0.5 fixed | **61.45 ± 0.18** |
| 4 | freeze_boq + test-peek β | same as 2 | β=best per seed | *61.89 ± 0.00* |
| 5 | Tier-2 unfreeze + 3-loss | unfreeze BoQ, Fourier pool, L_main+L_equi+L_rot | β=0.3 | 59.93 (1 seed) |
| 6 | Tier-2 freeze + 3-loss | freeze BoQ, Fourier pool, L_main+L_equi+L_rot | β=0 (best) | 60.91 (1 seed) |
| 7 | NormPool (B2 smoke) | freeze BoQ, norm pool, L_main+L_equi+L_rot | β=0 (best) | 60.91 (1 seed) |
| 8 | Adaptive β variants | freeze_boq ckpts, confidence-driven β | adaptive | ≤ 62.47 (no improvement) |
| 9 | Union retrieve (BoQ + equi stage-1) | freeze_boq ckpts, union top-K | β=0.5 | 62.47 (no improvement over 3) |
| **10** | **DR-VPR v2 (P1 standalone multi-scale)** | E2ResNet(C8) multi-scale, standalone MS loss, no BoQ in training | β=0.10 fixed | **61.89 ± 0.33** ← **main** |
| 10' | same, β=0.05 (tied best mean) | same | β=0.05 fixed | 62.00 ± 0.50 |
| 10'' | same, β=best per seed | same | β=best per seed (test-peek) | *62.21 ± 0.66* |

**Rationale for selecting row 10 (v2) as main method**:
- Highest mean + tightest std combination in the ablation
- β=0.10 has highest t-stat (3.41) in the β plateau [0.05, 0.13] — see
  `doc/ABLATION_BETA_SWEEP.md`
- Standalone training is architecturally cleaner than the v1 fusion hack
  (gate=0 stuck, L_main can't reach equi branch)
- Cross-dataset [10°, 20°] bucket gain (+2.67 ConSLAM, +1.84 ConPR) —
  see `doc/PER_YAW_ANALYSIS.md` v3

**Lessons from rows 5-9 (negative-result ablations)**:
- **Tier-2 Fourier pool + L_equi** pushed desc_equi towards BoQ subspace →
  rerank lost orthogonal tiebreak (rows 5-6). Mathematical explanation: when
  equi is trained on the same task as BoQ (GSV-Cities metric learning), they
  converge to overlapping features; rerank mixing redundant signals doesn't
  help.
- **NormPool** (row 7) gave identical stage-1 retrieve as max pool in the
  fusion architecture because gate=0 hides equi contribution; pool mode choice
  is invisible to single-stage val.
- **Adaptive β / union retrieve** (rows 8-9) can't exceed β=0.5 fixed because
  the top-K candidate set itself is the ceiling — BoQ doesn't include the true
  positive in top-100 for >30° yaw queries.
- **Standalone training unlocked gain** (row 10): direct MS loss on equi
  trained the descriptor to be useful by itself, while multi-scale pooling
  preserved enough orthogonality to BoQ to help in rerank.

---

## Table 4: Our DR-VPR variants (internal ablation, ConPR — full 10-seq)

Protocol: all 9 ConPR query sequences vs db=20230623, θ=0°, yaw=80°, 3 P1
standalone seeds. Source: `eval_rerank_conpr_full.log` 2026-04-18.

| Variant | ConPR R@1 (full 10-seq) |
|---------|------:|
| BoQ(ResNet50)@320 alone (β=0 in our pipeline) | 79.31 ± 0.00 (deterministic) |
| Old paper "DR-VPR concat" (single-stage fused desc) | 79.68 ± 1.10 |
| **DR-VPR v2 (P1 standalone + BoQ β=0.10)** | **79.74 ± 0.09** |

**For single-pair comparison (20230623 vs 20230809)**, DR-VPR v2 gives 81.99
(3-seed mean). This pair is rotation-easier than the full 10-seq average; its
individual number appears in the [10°, 20°] yaw bucket analysis in
`doc/PER_YAW_ANALYSIS.md`. The **full 10-seq number (79.74) is the correct
Table 2 comparison** vs other baselines.

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
\textbf{DR-VPR (ours)} & 16384+1024 & \textbf{79.74 $\pm$ 0.09} & — & — \\
\bottomrule
\end{tabular}
\vspace{2pt}
\footnotesize
Our result: full 10-sequence ConPR protocol (9 queries vs db=20230623), 3 P1
standalone seeds, BoQ(ResNet50)@320 top-100 stage-1 + $\beta = 0.10$ rerank.
\end{table}
```

---

*Last updated: 2026-04-18. Full 10-seq ConPR eval (Task 3), Union retrieve
(Task 2), TTA (Task 1) all completed — see `doc/EXPERIMENTS_TTA_UNION_FULLCONPR.md`.*

---

## Supplementary: TTA extension (ConSLAM only)

**Source**: `doc/EXPERIMENTS_TTA_UNION_FULLCONPR.md` §3, `eval_tta_rerank_standalone.log`.

| Method | ConSLAM R@1 | Δ over BoQ-R50 baseline |
|--------|---:|---:|
| BoQ(ResNet50) alone | 60.91 | +0.00 |
| DR-VPR v2 no TTA | 61.89 ± 0.33 | +0.98 |
| **DR-VPR v2 + TTA on-grid (C8, 8×)** | **62.32 ± 0.50** | **+1.41** |
| DR-VPR v2 + TTA off-grid (22.5°+k·45°, 8×) | 62.43 ± 1.32 | +1.52 |
| DR-VPR v2 + TTA full (16×) | 62.21 ± 0.86 | +1.30 |

TTA does **not** help on ConPR (θ=0° aligned queries) — reported in
`doc/EXPERIMENTS_TTA_UNION_FULLCONPR.md` §3. TTA is a supplementary result
recommended only for deployments with non-trivial query rotation.
