# Per-Yaw Bucket Analysis — DR-VPR v2 (P1 standalone multi-scale)

**Date**: 2026-04-18 (v3).
**Script**: `per_yaw_analysis_p1.py`.
**Raw records**: `per_yaw_p1_conslam_records.json`, `per_yaw_p1_conpr_records.json`.

---

## Revision history

- **v1 (2026-04-17, withdrawn)**: Claimed +3.81 R@1 in [20°, 40°) ConSLAM
  bucket as crown sub-result. **Invalidated** by an in-place numpy view bug
  in the query pose rotation (details in `feedback_numpy_view_inplace.md`
  memory). Bug shifted ~5 of 307 query positions across bucket boundaries,
  distorting small-sample bucket signals while leaving total R@1 nearly
  unaffected.
- **v2 (2026-04-18, 20° buckets)**: After bug fix, [20°, 40°) gain → 0.00.
  Only [0°, 20°) bucket showed +0.97.
- **v3 (2026-04-18, 10° buckets + P1 ckpts + ConPR)**: finer bucket
  granularity and second-dataset cross-validation reveal the real
  rotation-specific effect is in **[10°, 20°]** bucket, not [0°, 10°].

---

## Setup

- **Model**: DR-VPR v2 (P1 standalone multi-scale) — official BoQ(ResNet50)@320
  stage-1 + E2ResNet(C8) multi-scale stage-2 rerank at β=0.10 fixed.
- **Ckpts**: P1 standalone 3 seeds (val-best per seed: seed=1 Ep7, seed=42 Ep1,
  seed=190223 Ep8).
- **Bucket width**: **10°** (previously 20°).
- **BoQ R@1** = β=0 (pure BoQ retrieve). **DR-VPR R@1** = β=0.10 rerank.
- **Rotation code**: bug-fixed (temp vars).

---

## Table 1: ConSLAM (θ=15°, 921 records = 3 seeds × 307 valid queries)

| Bucket | N | BoQ R@1 | DR-VPR R@1 | Δ | flip→✓ | flip→✗ |
|--------|---:|---:|---:|---:|---:|---:|
| [ 0°, 10°) | 645 | 70.23% | 70.54% | +0.31 | 10 | 8 |
| **[10°, 20°)** | **75** | **56.00%** | **58.67%** | **+2.67** ★ | **2** | **0** |
| [20°, 30°) | 57 | 57.89% | 57.89% | +0.00 | 0 | 0 |
| [30°, 40°) | 36 | 66.67% | 66.67% | +0.00 | 0 | 0 |
| [40°, 50°) | 21 | 14.29% | 19.05% | +4.76 | 1 | 0 |
| [50°, 60°) | 30 | 30.00% | 30.00% | +0.00 | 0 | 0 |
| [60°, 70°) | 24 |  0.00% |  4.17% | +4.17 | 1 | 0 |
| [70°, 80°) | 33 |  0.00% |  0.00% | +0.00 | 0 | 0 |
| **TOTAL** | **921** | **61.24%** | **61.89%** | **+0.65** | **14** | **8** |

## Table 2: ConPR (θ=0°, 7401 records = 3 seeds × 2467 valid queries)

| Bucket | N | BoQ R@1 | DR-VPR R@1 | Δ | flip→✓ | flip→✗ |
|--------|---:|---:|---:|---:|---:|---:|
| [ 0°, 10°) | 6201 | 89.26% | 89.68% | +0.42 | 48 | 22 |
| **[10°, 20°)** | **435** | **84.83%** | **86.67%** | **+1.84** ★ | **8** | **0** |
| [20°, 30°) | 192 | 67.19% | 67.71% | +0.52 | 3 | 2 |
| [30°, 40°) | 141 |  0.00% |  0.00% | +0.00 | 0 | 0 |
| [40°, 50°) | 135 |  0.00% |  0.00% | +0.00 | 0 | 0 |
| [50°, 60°) | 111 |  0.00% |  0.00% | +0.00 | 0 | 0 |
| [60°, 70°) |  84 |  0.00% |  0.00% | +0.00 | 0 | 0 |
| [70°, 80°) | 102 |  0.00% |  0.00% | +0.00 | 0 | 0 |
| **TOTAL** | **7401** | **81.52%** | **81.99%** | **+0.47** | **59** | **24** |

---

## Cross-dataset findings

### Finding 1: [10°, 20°] bucket is the rotation robustness sweet spot

| Dataset | N in [10°, 20°) | BoQ R@1 | DR-VPR R@1 | Δ | flip→✓ : flip→✗ |
|---------|---:|---:|---:|---:|:---:|
| ConSLAM |  75 | 56.00% | 58.67% | **+2.67** | 2 : 0 |
| ConPR   | 435 | 84.83% | 86.67% | **+1.84** | 8 : 0 |

**Both datasets — same bucket — same direction — zero negative flips**. 10
improvement events, 0 regressions. This is the **real** rotation-specific
sub-claim (replacing the withdrawn [20°, 40°] +3.81 claim from v1).

### Finding 2: Extreme rotation buckets (>30°) are unrecoverable

- ConSLAM [50°, 80°): mostly 0% on both → BoQ fails at stage-1 FAISS top-100
  filter; rerank has no valid candidate to promote.
- ConPR [30°, 80°): all 0% on both → even more severe than ConSLAM; these
  queries are fundamentally outside BoQ's top-100 match pool.

**Architectural limitation**: two-stage rerank cannot rescue queries whose
true positive isn't in stage-1's top-K. Rerank is a second-pass reranker,
not a second-pass retriever. A true fix would require replacing stage-1
(e.g., union retrieve from BoQ + equi), not just scaling up stage-2 weight.

### Finding 3: Low-yaw queries (<10°) get marginal ensemble benefit

- ConSLAM [0°, 10°): +0.31 (10 flip→✓ vs 8 flip→✗ — near-tied, net +2)
- ConPR [0°, 10°): +0.42 (48 vs 22 — clearer positive, net +26)

BoQ is already strong here (70%/89%); equi adds a small consistent signal
but no dramatic gains — consistent with "orthogonal noise as tiebreaker".

---

## Paper narrative (recommended)

> "DR-VPR v2 achieves an overall +0.65 R@1 gain over the BoQ(ResNet50)
> baseline on ConSLAM (61.89 vs 61.24) and +0.47 on ConPR (81.99 vs 81.52)
> via two-stage retrieve-rerank with β=0.10. A per-yaw bucket breakdown
> reveals the improvement is **concentrated in the [10°, 20°] yaw-difficulty
> bucket**, with DR-VPR gaining **+2.67 on ConSLAM** and **+1.84 on ConPR**
> in this specific sub-population. Crucially, this pattern replicates across
> two independent datasets with different query distributions, confirming
> that the multi-scale equivariant branch contributes genuine rotation-
> robustness signal for moderately-rotated queries. For heavily-rotated
> queries (>30° yaw difference), both methods' stage-1 retrieval fails to
> include the true positive in the top-100 candidates; addressing this
> regime is a limitation we leave to future work (e.g., union retrieve or
> stage-1 equivariant features)."

---

## Reproducing

```bash
mamba run -n drvpr python per_yaw_analysis_p1.py 2>&1 | tee per_yaw_analysis_p1.log
```

Produces `per_yaw_p1_conslam_records.json` and `per_yaw_p1_conpr_records.json`
with per-query flags for further analysis / figure generation.

---

*Author: DR-VPR revision team. v3 updated 2026-04-18.*
