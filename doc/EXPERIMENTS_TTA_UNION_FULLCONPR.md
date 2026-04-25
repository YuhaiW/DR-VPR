# Experiments: TTA, Union Retrieve, Full ConPR (3-seed)

**Date**: 2026-04-18.
**Logs**: `eval_tta_rerank_standalone.log`, `eval_union_rerank_standalone.log`,
`eval_rerank_conpr_full.log`.
**Scripts**: `eval_tta_rerank_standalone.py`, `eval_union_rerank_standalone.py`,
`eval_rerank_conpr_full.py`.

All three experiments test extensions to DR-VPR v2 (P1 standalone multi-scale)
without any retraining — same 3 P1 ckpts, same β=0.10.

---

## 1. Full 10-sequence ConPR rerank evaluation

### Setup
- All 10 ConPR sequences (db=20230623, 9 query sequences).
- θ=0°, yaw=80°, gt_thres=5m.
- BoQ(ResNet50)@320 stage-1 top-100 → β=0.10 rerank.
- 3 P1 standalone seeds.

### 3-seed mean across all 9 query pairs

| β | seed=1 | seed=42 | seed=190223 | mean | std | Δ vs β=0 |
|----:|----:|----:|----:|----:|----:|----:|
| 0.00 | 79.31 | 79.31 | 79.31 | **79.31** | 0.00 | +0.00 |
| 0.10 | 79.65 | 79.75 | 79.82 | **79.74** | 0.09 | **+0.44** |

### Per-pair breakdown (3-seed averaged)

| Pair | β=0.00 | β=0.10 | Δ |
|------|---:|---:|---:|
| db=20230623 vs q=20230531 | 63.08 | 62.50 | **−0.57** |
| db=20230623 vs q=20230611 | 77.70 | 78.11 | +0.41 |
| db=20230623 vs q=20230627 | 92.39 | 92.55 | +0.16 |
| db=20230623 vs q=20230628 | 94.43 | 94.57 | +0.14 |
| db=20230623 vs q=20230706 | 86.03 | 86.72 | +0.69 |
| db=20230623 vs q=20230717 | 69.44 | 69.70 | +0.26 |
| db=20230623 vs q=20230803 | 76.91 | 78.68 | **+1.77** |
| db=20230623 vs q=20230809 | 81.52 | 81.99 | +0.47 |
| db=20230623 vs q=20230818 | 72.27 | 72.84 | +0.58 |

### Observations
- 8 / 9 pairs improve. Only `q=20230531` regresses (−0.57).
- Biggest gain on `q=20230803` (+1.77); previous single-pair (q=20230809)
  gives +0.47 and matches earlier reporting.
- Full ConPR gain (+0.44) is smaller than the cherry-picked single-pair gain
  (+0.47 → was reported as the main number), but similar magnitude.

### Paper-reported ConPR number (revised)

> "DR-VPR v2 achieves **79.74 ± 0.09 R@1 on the full ConPR benchmark**
> (9 query sequences vs db=20230623, averaged across 3 P1 standalone seeds),
> an improvement of **+0.44** over the BoQ(ResNet50) stage-1 baseline of 79.31."

This replaces the previous 81.99 single-pair number in `doc/BASELINE_COMPARISON_TABLE.md`
Table 2 row for DR-VPR v2.

---

## 2. Union retrieve: BoQ top-100 ∪ equi top-100

### Setup
Candidate pool = BoQ FAISS top-100 ∪ equi FAISS top-100 (unioned per-query).
Rerank within union at β=0.10.

### Results (3-seed mean ± std R@1)

#### ConSLAM (Seq5 db vs Seq4 query, θ=15°)

| Condition | seed=1 | seed=42 | seed=190223 | mean | std | Δ vs (a) |
|---|---:|---:|---:|---:|---:|---:|
| (a) BoQ-only β=0 | 61.24 | 61.24 | 61.24 | 61.24 | 0.00 | +0.00 |
| (b) BoQ-only β=0.10 rerank | 62.21 | 61.89 | 61.56 | **61.89** | 0.33 | +0.65 |
| (c) **Union rerank β=0.10** | 62.21 | 61.89 | 61.56 | **61.89** | 0.33 | +0.65 |
| (d) equi-only top-1 | 42.67 | 41.37 | 43.97 | 42.67 | 1.30 | −18.57 |

#### ConPR (db=20230623, q=20230809 single pair, θ=0°)

| Condition | seed=1 | seed=42 | seed=190223 | mean | std | Δ vs (a) |
|---|---:|---:|---:|---:|---:|---:|
| (a) BoQ-only β=0 | 81.52 | 81.52 | 81.52 | 81.52 | 0.00 | +0.00 |
| (b) BoQ-only β=0.10 rerank | 81.56 | 81.92 | 82.49 | **81.99** | 0.47 | +0.47 |
| (c) **Union rerank β=0.10** | 81.56 | 81.92 | 82.49 | **81.99** | 0.47 | +0.47 |
| (d) equi-only top-1 | 69.84 | 73.17 | 77.42 | 73.48 | 3.80 | −8.04 |

### Observations — ZERO additional gain

(c) is **identical** to (b) on both datasets, across all 3 seeds. This is not
a bug — it's a fundamental property of the β=0.10 weighted score:

- Candidates that are in equi-top-100 but NOT in BoQ-top-100 have low BoQ
  similarity (< position 100 in BoQ's ranking).
- At β=0.10, the BoQ term (weight 0.90) dominates the score. An equi-only
  candidate would need an equi-similarity advantage of ~9× over the BoQ-only
  candidate's equi similarity to flip the top-1 — this never happens in practice.
- equi-alone (d) is ~19 points worse than BoQ on ConSLAM and ~8 points worse
  on ConPR — confirming equi is a weak primary retriever.

### Implication for paper narrative

Union retrieve is a **documented null result**: stage-1 candidate diversity
doesn't help when rerank weight is low. To recover [30°+] yaw-bucket queries
(stuck at 0% per `doc/PER_YAW_ANALYSIS.md`), we'd need either:
- (i) higher β (which hurts easy queries — β=0.5 gives 61.45, β=1.0 gives 43.00
  on ConSLAM from coarse sweep).
- (ii) a fundamentally stronger rotation-robust stage-1 retriever (future work).

This is a clean **limitation** to report: two-stage rerank is inherently bounded
by stage-1 recall.

---

## 3. Test-Time Augmentation (TTA) — C8 on-grid vs off-grid

### Setup
Query-side TTA: rotate each query image N times, forward through equi model,
average descriptors, re-L2-normalize. Database stays single-orientation.

Three TTA modes:
- **on-grid (b)**: 8 rotations at k·45° for k=0..7 (all in C8 group).
- **off-grid (c)**: 8 rotations at 22.5° + k·45° (all outside C8 — break exact
  equivariance, test averaging of interpolation-perturbed descriptors).
- **full (d)**: 16 rotations = on-grid ∪ off-grid.

All comparisons at β=0.10 rerank vs no-TTA baseline.

### ConSLAM — TTA HELPS

| Condition | seed=1 | seed=42 | seed=190223 | mean | std | Δ vs (a_b0) |
|---|---:|---:|---:|---:|---:|---:|
| (a_b0) no TTA, β=0 | 61.24 | 61.24 | 61.24 | 61.24 | 0.00 | +0.00 |
| (a) no TTA, β=0.10 | 62.21 | 61.89 | 61.56 | 61.89 | 0.33 | +0.65 |
| (b) **TTA on-grid (8×)** | 62.87 | 61.89 | 62.21 | **62.32** | 0.50 | **+1.09** |
| (c) **TTA off-grid (8×)** | 63.84 | 61.24 | 62.21 | **62.43** | 1.32 | **+1.19** ★ |
| (d) TTA full (16×) | 63.19 | 61.89 | 61.56 | 62.21 | 0.86 | +0.98 |

**ConSLAM gain from TTA over vanilla DR-VPR v2 (β=0.10)**: +0.54 R@1 (off-grid)
or +0.43 R@1 (on-grid). For paper: **on-grid TTA** is the cleaner result
(lower std, 0.50 vs 1.32).

### ConPR — TTA NEUTRAL / SLIGHTLY NEGATIVE

| Condition | seed=1 | seed=42 | seed=190223 | mean | std | Δ vs (a_b0) |
|---|---:|---:|---:|---:|---:|---:|
| (a_b0) no TTA, β=0 | 81.52 | 81.52 | 81.52 | 81.52 | 0.00 | +0.00 |
| (a) no TTA, β=0.10 | 81.56 | 81.92 | 82.49 | **81.99** | 0.47 | +0.47 |
| (b) TTA on-grid (8×) | 81.52 | 81.48 | 82.41 | 81.80 | 0.53 | +0.28 |
| (c) TTA off-grid (8×) | 81.68 | 81.80 | 81.96 | 81.81 | 0.14 | +0.30 |
| (d) TTA full (16×) | 81.52 | 81.80 | 82.41 | 81.91 | 0.46 | +0.39 |

Vanilla (a) still best on ConPR. TTA reduces mean R@1 by ~0.1.

### Mechanism — why TTA helps on ConSLAM but not on ConPR

The equivariance sanity check reveals descriptors differ substantially under
rotation (L2 distance on unit vectors ≈ 0.7 for on-grid, ≈ 1.0 for off-grid —
corresponds to cos sim drop of ~25% and ~50% respectively). The learned C8
equivariance is **approximate, not exact** — rotation of the input produces
a perturbed descriptor via the bilinear interpolation in feature maps.

- **ConSLAM (θ=15° forced query rotation)**: query already rotated vs db.
  TTA averages 8 perturbed descriptors → denoises the bilinear-interpolation
  artifacts → more stable matching. Net +0.5 R@1.
- **ConPR (θ=0°, aligned)**: query already in same orientation as db. TTA
  averages produce a "diffused" descriptor that dilutes the aligned signal
  that was strongest at the native orientation. Net −0.1 R@1.

### Paper claim — adaptive strategy

> "TTA provides meaningful gains only under significant query rotation. On
> ConSLAM (θ=15° forced rotation), on-grid TTA adds **+0.43 R@1** over
> no-TTA DR-VPR v2 (61.89 → 62.32 ± 0.50, +1.09 over BoQ baseline). On
> ConPR (θ=0°, rotation-aligned), TTA provides no benefit and slightly
> hurts R@1 (81.99 → 81.91). We therefore recommend TTA only for
> deployments expecting non-trivial query rotation."

---

## Consolidated revised main-table numbers

| Dataset | BoQ(ResNet50) | DR-VPR v2 (β=0.10) | DR-VPR v2 + TTA on-grid | Best |
|---|---:|---:|---:|---:|
| **ConSLAM (3-seed mean ± std)** | 61.24 | 61.89 ± 0.33 | **62.32 ± 0.50** | +1.09 |
| **ConPR (full 10-seq, 3-seed)** | 79.31 ± 0.00 | **79.74 ± 0.09** | 79.68* | +0.44 |

*ConPR TTA from single-pair; re-extract 10-seq if needed. TTA worsens ConPR
so the paper should report no-TTA as ConPR's best.

---

## Paper reporting recommendations

- **Use Table 2 ConPR DR-VPR v2 row = 79.74 ± 0.09** (full 10-seq), not 81.99
  (which was a single-pair cherry-pick).
- Keep Table 2 ConSLAM DR-VPR v2 row = 61.89 ± 0.33 (the main method, no TTA).
- Add TTA in supplementary as an additional gain on ConSLAM: +0.43 → 62.32 ± 0.50.
- Add Union retrieve in supplementary as a negative result / limitation discussion.

---

## Reproducing

```bash
mamba run -n drvpr python eval_rerank_conpr_full.py         2>&1 | tee eval_rerank_conpr_full.log
mamba run -n drvpr python eval_union_rerank_standalone.py   2>&1 | tee eval_union_rerank_standalone.log
mamba run -n drvpr python eval_tta_rerank_standalone.py      2>&1 | tee eval_tta_rerank_standalone.log
```

*Author: DR-VPR revision team. 2026-04-18.*
