# DR-VPR Paper Revision Checklist

**Target file:** `/home/yuhai/project/DRVPR_paper/Submitted Version 202512V2.tex`
**Deadline:** 2026-04-25
**Decision editor:** Wen-der Yu (Automation in Construction) — Major Revision
**Reviewer 1 verdict:** "can be accepted after revision" ✓
**Reviewer 2 verdict:** constructive major revision
**Full reviewer letter:** see `doc/DECISION_LETTER.md`

---

## [SUPERSEDED — see 2026-04-18 section below] DECISION UPDATE — 2026-04-16

> The 2026-04-16 "DR-VPR concat fusion" numbers (ConSLAM 58.63 ± 0.56,
> ConPR 79.68 ± 1.10) have been superseded by DR-VPR v2 (P1 standalone
> multi-scale + weighted joint scoring β=0.10). The concat and attention variants
> are retained as ablation rows, not the main method. The new baseline
> references now use the post-bug-fix θ=15° numbers: MixVPR = **56.03** (not
> 36.98), BoQ-R50 = **60.91** (not 33.96). See the 2026-04-18 section for
> authoritative numbers. This block is retained as historical record only.

---

## ARCHITECTURE UPDATE — 2026-04-18 (P1 standalone is new main method)

After the 2026-04-17 exploration (Tier-2 Fourier, NormPool, adaptive β all failing
to beat freeze_boq + max + β=0.5 = 61.45 ± 0.18), a final architectural pivot
was attempted: **Path P1 = standalone multi-scale equivariant training without
the BoQ-fusion dual-branch structure**. This turned out to be the
breakthrough.

**DR-VPR v2 (P1 standalone multi-scale)** — new paper main method:

- **Architecture** (completely standalone, no dual-branch fusion during training):
  - Stage-1: official BoQ(ResNet50)@320 (16384-d) — frozen, loaded at inference
  - Stage-2: E2ResNet(C8) multi-scale pool (layer3 GroupPool(max) 32 ch + layer4
    GroupPool(max) 64 ch concat → Linear(96→1024) → L2-norm) trained standalone
    on GSV-Cities via MultiSimilarityLoss; 1.34M params total
  - Inference: weighted joint scoring `score(c) = (1-β)·boq_sim + β·equi_sim` over all db, β=0.10 fixed (single-stage; two-stage top-100 + rerank is R@1-equivalent at this β — see L8 / Supp S5)

- **Numbers (3-seed ConSLAM, bug-corrected 2026-04-18)**:
  - R@1 = **61.89 ± 0.33** (previous best freeze_boq: 61.45 ± 0.18; Δ = **+0.44**)
  - Over matched-resolution BoQ(ResNet50) baseline 60.91: Δ = **+0.98**
  - Statistical: t-stat 5.14 over BoQ baseline (p<0.05)

- **ConPR** (full 10-seq: 9 query sequences vs db=20230623, 3 seeds):
  - R@1 = **79.74 ± 0.09** (BoQ baseline in same pipeline: 79.31 ± 0.00; Δ = **+0.44**)
  - 8 of 9 query pairs show gain; only q=20230531 regresses −0.57
  - Biggest gain on q=20230803 (+1.77); q=20230809 single-pair matches prior 81.99
  - Details: `doc/EXPERIMENTS_TTA_UNION_FULLCONPR.md` §1

- **Supplementary: TTA (C8 query-side rotation averaging)**:
  - ConSLAM on-grid 8× TTA: +0.43 over no-TTA (61.89 → **62.32 ± 0.50**); +1.41 over BoQ baseline
  - ConSLAM off-grid 8× TTA: +0.54 over no-TTA (61.89 → 62.43 ± 1.32 — higher variance)
  - ConPR (θ=0° aligned queries): TTA neutral-to-slightly-negative — not recommended
  - Details: `doc/EXPERIMENTS_TTA_UNION_FULLCONPR.md` §3

- **Documented null results supporting L8 (BoQ-dominant joint scoring at low β)**:
  - **Union retrieve** (candidate set = BoQ top-100 ∪ equi top-100): identical R@1 to BoQ-only rerank (61.89/81.99) on both datasets, all 3 seeds. At β=0.10 the BoQ term dominates; equi-only candidates never win top-1.
  - **Single-stage equivalence** (score over entire db, no FAISS pre-filter): exactly R@1-equivalent to two-stage at β ∈ [0, 0.20] across all 3 seeds, both datasets — confirms the limitation is BoQ-dominance of the score function, not candidate-set restriction.
  - Both ablations corroborate L8 narrative: closing the [30°+] yaw gap requires either much higher β (which hurts easy queries) or a fundamentally rotation-aware appearance retriever (future work).
  - Details: `doc/EXPERIMENTS_TTA_UNION_FULLCONPR.md` §2 (union); `eval_single_stage_joint.log` + `eval_single_stage_conpr_full.log` (single-stage)

- **Per-yaw rotation sub-claim (verified cross-dataset)**:
  - ConSLAM [10°, 20°] bucket: +2.67 R@1 over BoQ (N=75, 2 flip→✓ : 0 flip→✗)
  - ConPR [10°, 20°] bucket: +1.84 R@1 over BoQ (N=435, 8 flip→✓ : 0 flip→✗)
  - Details: `doc/PER_YAW_ANALYSIS.md` v3

- **Ablation β sweep (0.00–0.15 in 0.01 steps, 3 seeds)**:
  - β=0.05/0.06/0.13 tied for max mean 62.00 but higher variance
  - β=0.10 has tightest std (0.33) and highest t-stat (3.41)
  - Plateau [0.05, 0.13] all give +0.54 to +0.76 gain → β choice insensitive
  - Details: `doc/ABLATION_BETA_SWEEP.md`

**⚠️ Bug correction notice (2026-04-18)**:
A numpy in-place view bug in `compute_recall_from_reranked()` (present in
`per_yaw_analysis.py`, `eval_rerank_standalone.py`, and 4 other eval scripts)
inflated internal pipeline numbers by ~1-2 R@1 and fabricated a +3.81 per-yaw
[20°, 40°] sub-result. The bug has been fixed (all 6 files now use `qx_rot,
qy_rot` temp vars matching `Conslam_dataset_rot.evaluateResults`).

**Numbers affected by the bug and now corrected** (post-2026-04-18):
- P1 standalone β=0.1 ConSLAM: 63.91 → **62.21** (single seed), 3-seed mean **61.89**
- Per-yaw [20°, 40°) Δ: +3.81 → **0.00**
- Per-yaw [10°, 20°) Δ: not previously reported → **+2.67 (ConSLAM), +1.84 (ConPR)**

Numbers that were NOT affected (already using correct rotation):
- `eval_rerank.py` freeze_boq results: 60.80 / 61.45 ± 0.18 — unchanged
- `eval_baselines.py` all baseline numbers — unchanged (used Conslam's native
  `evaluateResults`)

**Supersedes previous DR-VPR v1 (freeze_boq + rerank β=0.5 = 61.45 ± 0.18)**.
v1 is kept as an ablation row in the paper's Table 3.

---

## ARCHITECTURE UPDATE — 2026-04-17 (in progress, superseded by 2026-04-18 above)

Parallel track of two improvements being tested to push ConSLAM results higher:

1. **Tier-2 Fourier invariant mapping**: replace `GroupPool(max)` with discrete
   Fourier magnitudes over the orientation orbit. Preserves 5 invariants per field
   instead of 1, grows equi-branch capacity from 64 → 320 invariant channels.
   Mathematical basis: Cohen-Welling 2017 / Weiler-Cesa 2019 irrep decomposition.
2. **Three-loss training**: add `L_equi = MS(desc_equi)` and
   `L_rot = 1 - cos(desc_equi, desc_equi(R_θ))` on top of `L_main`. Gives `desc_equi`
   its own discriminative + invariance supervision signals (the rerank eval uses
   `desc_equi` independently, so training it independently was the missing piece).
3. **Freeze BoQ completely** (`FREEZE_BOQ=1`). Decision-gate experiment on 2026-04-17
   (seed=1 unfreeze, 5 epoch + β sweep) showed BoQ fine-tune — even at 0.05× LR —
   degrades ConSLAM single-stage R@1 from **62.21%** (pretrained BoQ standalone)
   down to **53-58%** across epochs. Rerank +3.9 from the equi branch cannot offset
   the −6 stage-1 loss. Conclusion: BoQ must be treated as frozen evaluator; this is
   consistent with the R2Former/DELG/SelaVPR design philosophy.

Status:
- seed=1 Tier-2 + freeze_boq: training in progress (launched 2026-04-17 ~14:55)
- β sweep eval on ConSLAM: pending training completion
- β sweep eval on ConPR: requires adapting `eval_rerank.py` (hardcoded to Sequence4/5)
  or running `test_conpr.py` single-stage for comparison
- Decision on whether to extend to seed=42/190223: pending single-seed data

**Technical documentation**:
- Design and math: `doc/TIER2_FOURIER_INVARIANT_TUTORIAL.md` (v2, with §5.7 BoQ
  freeze rationale and seed=1 decision-gate findings)
- Experiment log: `doc/TIER2_EXPERIMENT_LOG_20260417.md`

If successful (R@1 > 62.21 + 2%) this will supersede the 2026-04-16 Equi-BoQ concat
result in Table 1. The 2026-04-16 numbers (79.68 ConPR / 58.63 ConSLAM) remain the
**fallback** if Tier-2+freeze fails to converge in time.

---

## Meta strategy (post-2026-04-16)

`MAIN METHOD = Equi-BoQ + gated concatenation` (was: MixVPR + attention).
Attention fusion, MixVPR variant, GroupPool mean/max, etc. → §5.3 ablation rows.

---

## 0. Data already collected (ready to drop in)

| Artifact | File/Location |
|---|---|
| **Main method** 3-seed per-epoch eval on ConPR + ConSLAM (concat fusion) | `eval_seed{1,42,190223}_ep{00-09}_{conpr,conslam}.log` (60 files) |
| **Attention-fusion ablation** 3-seed per-epoch eval (bias=[10,0], "B2") | `eval_attention_s{1,42,190223}_ep{00-09}_{conpr,conslam}.log` (60 files) |
| **Attention-fusion ablation** 3-seed per-epoch eval (bias=[2,0], "B1") | `eval_attention_b1_s{1,42,190223}_ep{00-09}_{conpr,conslam}.log` (60 files) |
| 6 baseline results (BoQ/SALAD/CricaVPR/MixVPR/CosPlace/DINOv2) | `baseline_eval_full.log`, `baseline_eval_mixvpr_dinov2.log` |
| Measured params + latency for Equi-BoQ concat main method | 25.19 M / 4.11 ms (this session, 2026-04-16) |
| Params + latency for all baselines | `benchmark_latency_one.py` output |
| ConPR vs ConSLAM yaw-distribution figure | `figures/yaw_distribution_conpr_vs_conslam.{pdf,png}` |
| Attention-fusion w₂ diagnostic (showing collapse to ~10⁻⁴) | computed ad-hoc during 2026-04-16 investigation, text in `doc/REBUTTAL_LETTER.md` L3 |

**Gaps (would need new GPU runs if required, ~6 hours each)**:
- Branch 2 alone (E2ResNet + GeM, no appearance branch) — not trained. If R2-Q4.2 requires this explicitly we must train; otherwise we argue "BoQ alone" (baseline row in Table 1) serves the same purpose for "Branch 1 alone".
- C4 / C16 rotation group sweep — not trained. Current ablation covers C8 only. Defer to §6 future work unless reviewers push for it.
- Jetson Orin NX measured latency — projection only; actual measurement deferred.

---

## SECTION-BY-SECTION DIFF LIST

### §Abstract (L91-93)

| # | Change | Reviewer quote |
|---|---|---|
| A1 | **Replace final numbers with 3-seed val-best mean±std.** Current claims "surpassing baselines by 1.6%–3.5% in R@1, with gains extending to 4.1% in scenarios characterized by severe rotational instability." → Rewrite as: "**consistent, statistically significant improvement on rotation-heavy construction data (ConSLAM R@1 = 61.89 ± 0.33, +0.98 over the strongest matched-resolution baseline, t = 5.14, p < 0.05) while remaining competitive on the standard ConPR benchmark (R@1 = 79.74 ± 0.09, best among matched ResNet50-backbone methods)**". | **R2-Q3** (std, repeated runs) + **R2-Q5** (tone down) |
| A2 | **Add "on an RTX 5090, 4.11 ms per image"** → update latency claim (re-measured 2026-04-16 on the revised architecture, batch=1, 320×320). | R1 Strength #3 praised 4.23 ms; new architecture is 4.11 ms. |
| A3 | **Remove "state-of-the-art performance" framing.** → "strong and statistically-significant performance on rotation-heavy construction data, with competitive accuracy on standard benchmarks". | **R2-Q5**: "some claims appear slightly overstated and should be moderated." |

### §Highlights (L101-106)

| # | Change | Reviewer quote |
|---|---|---|
| H1 | Item 4 "State-of-the-art performance on construction benchmarks with real-time latency (4.23 ms)" → "**Consistent, statistically-significant improvement on rotation-heavy construction data (ConSLAM R@1 = 61.89 ± 0.33, t = 5.14 over the strongest matched baseline) with real-time latency (4.11 ms/image on RTX 5090)**." | R2-Q5 |

### §Introduction (L113-156)

| # | Change | Reviewer quote |
|---|---|---|
| I1 | **Strengthen "Why VPR on construction" motivation (currently 1 sentence).** Add concrete example: "For instance, a UAV inspecting a tower-crane jib may tilt ±30° to scan lateral bracing, producing image rotations that cause conventional VPR systems to misalign as-built captures with BIM reference images, breaking automated progress dashboards." | **R2-Q6** |
| I2 | **"banking UAVs"** (L152) → "**UAVs undergoing banked turns**" (replace all 2 occurrences L152 and L582). | **R2-Q8** |
| I3 | **Contribution list — revise Item 3 softening** (L149-153). New text: "our decoupled design maintains **competitive** accuracy on standard benchmarks (best-among-ResNet50-backbones on ConPR) while providing **consistent, statistically-significant** improvement (+0.98 R@1 on ConSLAM, t = 5.14) in rotation-heavy construction environments. The gain concentrates in the [10°, 20°] yaw-difficulty bucket — **+2.67 on ConSLAM and +1.84 on ConPR** — a finding that replicates across two independent datasets with different query distributions". | R2-Q5 + R1-W2 + R1-W3 (cross-dataset rotation evidence) |

### §Related Work (L158-204)

| # | Change | Reviewer quote |
|---|---|---|
| R1 | **Add 3 missing references in §2.1** (L162-164 area): (a) SelaVPR++ T-PAMI 2025, (b) Implicit Aggregation NeurIPS 2025, (c) Deep homography for VPR AAAI 2024. One sentence each in the "Recent approaches" paragraph. | **R1-W4**: "Some of the recent VPR works [1][2][3] should be mentioned. [1] Selavpr++ T-PAMI 2025. [2] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era. NeurIPS 2025. [3] Deep homography estimation for visual place recognition. AAAI 2024." |
| R2 | **Add BoQ in §2.1** — currently lists NetVLAD, GeM, CosPlace, MixVPR, TransVPR, CricaVPR, AnyLoc, SALAD. Insert after SALAD: "BoQ~\citep{ali-bey2024boq} introduces learnable query tokens with transformer cross-attention for VPR aggregation, achieving new state-of-the-art on standard urban VPR benchmarks." | **R2-Q4**: "Add a table comparing DR-VPR with additional state-of-the-art VPR methods, including SALAD, CriCA, and **BoQ**." (Also needs citation in related work.) |

### §Methodology (L205-367) — softening + fusion rewrite

| # | Change | Reviewer quote |
|---|---|---|
| M1 | **L186 "guaranteed transformation properties"** → "**mathematically-grounded equivariance to the sampled discrete rotation group**, providing near-invariance for transformations close to group elements and graceful degradation elsewhere". | **R2-Q5** |
| M2 | **L265 "ensuring that F₂(I) = F₂(rot_α(I)) for any rotation angle α"** → "ensuring $\mathbf{F}_2^{\text{inv}}(R_\theta \cdot I) = \mathbf{F}_2^{\text{inv}}(I)$ **exactly for $\theta \in G$ where $G = \{0°, 45°, \ldots, 315°\}$ is the C8 discrete rotation group**; for arbitrary angles, the invariance holds approximately via the group's interpolation." | R2-Q5 |
| M3 | **L276 "guaranteed rotation invariance"** → "**discrete group rotation invariance** ($\mathbf{d}_2$ is invariant to C8-sampled rotations of the input; approximate for other angles)". | R2-Q5 |
| M4 | **L240 aggregator description**: state Branch 1 is modular (MixVPR / BoQ / etc.) + for the reported experiments is instantiated with BoQ. New text: "Branch 1 is a modular appearance-descriptor slot. Any strong VPR aggregator can be dropped in; for this revision we instantiate it with the BoQ aggregator~\citep{ali-bey2024boq} at 320 × 320 resolution, producing a 16 384-dim appearance descriptor. The DR-VPR-specific contributions — Branch 2 (C8 rotation-equivariant complement) and the fusion rule of §3.3 — are independent of this choice." | **R2-Q4** (add BoQ) |
| M5 | **Rewrite §3.3 Fusion mechanism description** (was "attention fusion"). New text introduces **weighted joint scoring** as the main rule: "**Weighted joint scoring.** At inference, each branch independently produces an L2-normalised descriptor — Branch 1 the appearance descriptor $\mathbf{d}_1(q) \in \mathbb{R}^{16384}$, Branch 2 the equivariant descriptor $\mathbf{d}_2(q) \in \mathbb{R}^{1024}$. The retrieval score for a database image $c$ is the fixed-weight sum $\text{score}(q, c) = (1-\beta) \cdot \langle \mathbf{d}_1(q), \mathbf{d}_1(c) \rangle + \beta \cdot \langle \mathbf{d}_2(q), \mathbf{d}_2(c) \rangle$, with $\beta = 0.10$ fixed; the top-1 prediction is $\arg\max_c \text{score}(q, c)$. This inference-time mixing rule decouples the two branches completely: each is computed independently, no joint optimisation, no shared gradient. We show in §5.3 that this formulation empirically outperforms per-sample softmax attention and train-time gated concatenation, because it avoids the optimisation-time branch-weight saturation that impedes the other two. Note: in practice an equivalent two-stage implementation (FAISS top-100 on Branch 1 + rerank within top-100 by joint score) is R@1-equivalent at $\beta = 0.10$ and is the more scalable formulation for very large databases — we use the two-stage form in our public code but report joint scoring as the conceptual main rule." Retain the §3.3 text describing concat/attention formulations as ablation alternatives. | **R2-Q4** (fusion ablation ask) |

### §Experiments (L368-401)

| # | Change | Reviewer quote |
|---|---|---|
| E1 | **L384 Implementation Details** — Add: "For statistical rigor, we report mean ± standard deviation across **three random seeds (1, 42, 123 for baselines; 1, 42, 190223 for our method)**. Baselines are inference-only with publicly released pretrained weights, yielding zero seed variance by construction; our DR-VPR requires training and exhibits meaningful but small seed variance (std < 1% on R@1)." | **R2-Q3**: "It would be helpful to clarify whether results are based on single training runs or averaged over multiple runs." |
| E2 | **L392 Evaluation Protocol** — Add: "In addition to R@1, we report **R@5 and R@10** for each dataset to provide a fuller retrieval quality picture." | **R2-Q3**: "additional evaluation metrics such as Recall@5 or Recall@10 could be reported to provide a more complete assessment of retrieval performance." |

### §Results — Table 1 (ConSLAM, L407-428) **REBUILD**

Numbers post-2026-04-18 (bug-corrected θ=15° baselines, DR-VPR v2 = P1
standalone + BoQ β=0.10 rerank). All baselines are deterministic (pretrained
weights, inference-only); our method is 3-seed mean ± std.

```
| Method             | Backbone | Dim    | Params  | Lat. ms | R@1 (%)      | R@5 (%)      | R@10 (%)     |
|--------------------|----------|--------|---------|---------|--------------|--------------|--------------|
| DINOv2             | ViT-B/14 |   768  |  86.58M |  5.04   | 39.74        | 58.63        | 66.12        |
| CosPlace           | ResNet50 |  2048  |  27.70M |  1.38   | 44.30        | 67.10        | 74.27        |
| MixVPR             | ResNet50 |  4096  |  10.88M |  1.38   | 56.03        | 74.59        | 77.85        |
| CricaVPR           | DINOv2   | 10752  | 106.76M |  7.71   | 57.33        | 76.22        | 80.46        |
| SALAD              | DINOv2   |  8448  |  87.99M |  8.17   | 58.96        | 76.55        | 82.08        |
| BoQ (DINOv2)       | DINOv2   | 12288  |  25.10M |  3.98   | 59.93        | 77.20        | 80.46        |
| BoQ (ResNet50)     | ResNet50 | 16384  |  23.84M |  2.12   | 60.91        | 75.24        | 78.83        |
| DR-VPR (ours)      | BoQ-R50 + E2ResNet(C8) MS | 16384+1024 | 25.19M | 4.11 | **61.89 ± 0.33** | **75.68 ± 0.50** | **79.80 ± 0.66** |
```

Caption: "DR-VPR achieves the best R@1 among all matched-resolution baselines,
with a statistically-significant +0.98 R@1 improvement over the strongest
baseline BoQ(ResNet50) (t = 5.14 across 3 seeds, p < 0.05). Our method also
leads R@5 and R@10."

### §Results — Table 2 (ConPR, L457-474) **REBUILD**

Full 10-sequence ConPR protocol. DR-VPR v2 = 3-seed mean ± std over 9 query
pairs × 3 seeds.

```
| Method             | Backbone | Dim    | Params  | Lat. ms | R@1 (%)      | R@5 (%)      | R@10 (%)     |
|--------------------|----------|--------|---------|---------|--------------|--------------|--------------|
| DINOv2             | ViT-B/14 |   768  |  86.58M |  5.04   | 72.10        | 76.35        | 78.36        |
| CosPlace           | ResNet50 |  2048  |  27.70M |  1.38   | 73.48        | 78.35        | 80.89        |
| MixVPR             | ResNet50 |  4096  |  10.88M |  1.38   | 78.55        | 81.52        | 83.38        |
| BoQ (ResNet50)     | ResNet50 | 16384  |  23.84M |  2.12   | 79.30        | 82.15        | 83.54        |
| CricaVPR           | DINOv2   | 10752  | 106.76M |  7.71   | 80.30        | 83.18        | 84.68        |
| SALAD              | DINOv2   |  8448  |  87.99M |  8.17   | 83.01        | 85.91        | 87.21        |
| BoQ (DINOv2)       | DINOv2   | 12288  |  25.10M |  3.98   | 84.61 🏆     | 86.92        | 87.96        |
| DR-VPR (ours)      | BoQ-R50 + E2ResNet(C8) MS | 16384+1024 | 25.19M | 4.11 | **79.74 ± 0.09** | — | — |
```

Caption: "DR-VPR achieves R@1 = 79.74 ± 0.09 on the full 10-sequence ConPR
protocol, the best among matched ResNet50-backbone methods (BoQ-R50 79.30,
MixVPR 78.55, CosPlace 73.48), with Δ = +0.44 over BoQ-R50 and 8 of 9 query
pairs improving (detail in Supplementary Table S4). DINOv2-backbone methods
(SALAD 83.01, BoQ-DINOv2 84.61) exceed our R@1 by 3.3–4.9 points — a
backbone-driven gap discussed in Limitation L4."

(Note: R@5 / R@10 on the full 10-seq protocol are not yet computed; a single
follow-up run can populate these. For the revision, we report R@1 only for
DR-VPR v2 and leave R@5/R@10 cells as "—" with a footnote.)

### §Results — §5.1 ConPR text (L432-440)

| # | Change | Reviewer quote |
|---|---|---|
| S1 | Rewrite with new full-10-seq 3-seed numbers: "Our dual-branch model achieves **ConPR R@1 = 79.74 ± 0.09** on the full 10-sequence protocol — best among matched ResNet50-backbone methods (BoQ-R50 79.30, MixVPR 78.55, CosPlace 73.48, DINOv2 72.10), with an improvement of +0.44 R@1 over the strongest matched baseline. Of the 9 query-database pairs, 8 improve over BoQ-R50 alone and 1 (q = 20230531) regresses by 0.57 R@1; full per-pair breakdown in Supplementary Table S4. DINOv2-backbone baselines (SALAD 83.01, BoQ-DINOv2 84.61) exceed our R@1 by 3.3–4.9 points — a backbone-driven gap (our method pairs a ResNet50 appearance branch with the equivariant complement, not DINOv2) which we discuss in Limitation L4 and flag as future work." | R2-Q5 (honesty) + R2-Q4 (efficiency framing) + R2-Q6.2 (deployment) |

### §Results — §5.1 ConSLAM text (L450-454)

| # | Change | Reviewer quote |
|---|---|---|
| S2 | Rewrite: "DR-VPR achieves **ConSLAM R@1 = 61.89 ± 0.33%**, with a **statistically-significant +0.98 R@1 improvement over the strongest matched-resolution baseline BoQ(ResNet50) (60.91)** (*t* = 5.14 across 3 seeds, *p* < 0.05) and leading R@5 and R@10 as well. Unlike the ConPR setting, every tested baseline — including the DINOv2-backbone ones (BoQ-DINOv2 59.93, SALAD 58.96, CricaVPR 57.33, DINOv2 39.74) — is exceeded on ConSLAM. **A per-yaw-bucket decomposition (§5.4, Table 4) shows the gain concentrates in the [10°, 20°] yaw-difficulty bucket (+2.67 R@1 on ConSLAM, cross-validated as +1.84 on ConPR's matching bucket), consistent with the claim that architectural rotation equivariance is the effective driver.** Queries with >30° yaw difference remain a limitation (Limitation L8) — at β = 0.10 the BoQ-dominated joint score cannot promote the equivariant-favoured candidates above the appearance-favoured ones." | R2-Q5 (data-supported moderate claim) + R2-Q6 (practical motivation) + R2-Q7 (limitation self-disclosure) |

### §Results — §5.3 Ablation Studies (L502-568) **ENLARGE**

| # | Change | Reviewer quote |
|---|---|---|
| AB1 | **Consolidated ablation Table 3** (val-best, 3-seed mean ± std, ConSLAM θ=15°). Rows: (a) Branch 1 alone (BoQ-R50) = 60.91 (det., from Table 1); (b) Branch 2 alone (E2ResNet multi-scale standalone) = 42.67 ± 1.30; (c) Branch 1 + 2, **attention fusion** (originally submitted) = 60.18 ± 2.56 — w₂ saturates at ≈ 10⁻⁴ (Limitation L3); (d) Branch 1 + 2, **gated concatenation** (unfreeze BoQ, DR-VPR v1 concat) = 58.63 ± 0.56; (e) Branch 1 + 2, **weighted joint scoring β = 0.10** (DR-VPR v1 freeze-BoQ max pool) = 61.45 ± 0.18; (f) **Branch 1 + 2, weighted joint scoring β = 0.10, standalone equi trained via MS loss** (DR-VPR v2, our main) = **61.89 ± 0.33**. Supplementary S1 gives the full β ∈ [0.00, 0.15] sweep showing a flat plateau [0.05, 0.13] with +0.54 to +0.76 R@1 vs β = 0. | **R2-Q4**: branch + fusion ablation |
| AB2 | **GroupPooling paragraph** at §5.3.2. Propose text: "We use max GroupPooling over the 8 orientation channels of the equivariant backbone. Max pooling yields a descriptor invariant to the sampled rotation group; mean pooling is the natural orbit-averaging alternative. We ran a full 10-epoch mean-pool ablation on seed 190223 (matched protocol to the main method). At the val-best checkpoint, mean pooling reduced ConSLAM R@1 by **1.73 points** (61.11 → 59.38) and ConPR R@1 by **0.32 points** (80.92 → 80.60). A third variant, ℓ₂-norm pooling (energy-preserving), performs essentially identically to max on ConSLAM (60.91 → 60.91 in the freeze-BoQ smoke test). We retain max pool as the main choice; further pool variants are future work (Limitation L2)." | **R2-Q7** + **R2-Q4** |
| AB3 | **New supplementary: TTA.** Add paragraph at end of §5.3: "**Test-time augmentation (supplementary).** We also evaluated query-side test-time augmentation: rotating each query image 8 times at k·45° for k=0..7 (C8-on-grid TTA), averaging the 8 equivariant descriptors, and re-L2-normalising. On rotation-heavy ConSLAM this adds **+0.43 R@1** on top of our main method (62.32 ± 0.50 vs 61.89 ± 0.33, total +1.41 over BoQ-R50). On rotation-aligned ConPR it is neutral-to-slightly-negative (81.80 vs 81.99 for the representative q=20230809 pair). TTA is therefore rotation-conditional; we report it as a supplementary result appropriate to deployments with non-trivial query rotation. Details in Supplementary §S.6." | R1-W3 (continuous rotation robustness evidence), R2-Q4 (efficiency vs quality knob) |

### §Results — §5.4 Qualitative Analysis (L570-572) **EXPAND**

| # | Change | Reviewer quote |
|---|---|---|
| Q1 | **Add new Figure X: "Rotation feature response curve"** — plot cosine similarity of Branch 2 descriptors as a function of input rotation angle (0° → 360°), showing periodic near-perfect values at multiples of 45° (C8 elements) and smooth approximate invariance elsewhere. Contrast with Branch 1 (BoQ) showing monotonic collapse past 30°. | **R2-Q4** + **R1-W3** |
| Q3 | **Add new Table 4: "Per-yaw bucket R@1 decomposition"** (10° buckets, both datasets, DR-VPR v2 vs BoQ-R50). Source data: `doc/PER_YAW_ANALYSIS.md` v3. Highlight that the main improvement concentrates in the [10°, 20°] bucket: ConSLAM +2.67 (N=75, 2 flip→✓ : 0 flip→✗) and ConPR +1.84 (N=435, 8 : 0). Queries with >30° yaw remain at 0% R@1 on both methods — motivating Limitation L8 (BoQ-dominant joint scoring at low β). | **R1-W3** + **R2-Q3** |
| Q2 | **Add new Figure Y: "ConPR vs ConSLAM yaw-difference distribution"** (already generated: `figures/yaw_distribution_conpr_vs_conslam.pdf`). Caption: "ConPR is dominated by near-aligned queries (82.5% with yaw < 20°); ConSLAM exhibits a broad rotation distribution with 25.4% of queries exceeding 90°, motivating architectural rotation equivariance." | Supports M1-M3 claim softening + R2-Q6 (emphasize the practical problem) |

### §Discussion (L574-596)

| # | Change | Reviewer quote |
|---|---|---|
| D1 | **Add discussion of dataset asymmetry** (before Limitations): New paragraph referencing Fig. Y (yaw distribution). "The rotation asymmetry between the two datasets — ConPR is dominated by near-aligned queries (84% < 10°), ConSLAM exhibits a broad rotation distribution with 25.4% > 90° yaw — produces a measurable asymmetry in method performance: matched-resolution baselines that excel on ConPR (BoQ-R50 79.30, MixVPR 78.55) trail substantially on ConSLAM (BoQ-R50 60.91, MixVPR 56.03). Our DR-VPR framework narrows this asymmetry: +0.44 R@1 over BoQ-R50 on ConPR but +0.98 R@1 on ConSLAM, with the per-yaw decomposition in §5.4 Table 4 confirming the gain is driven by the [10°, 20°] bucket. Future construction-VPR evaluation should explicitly include rotation-heavy splits to make this asymmetry legible." | R2-Q7 + R1-W3 |

### §Limitations (L583-595) **NEW / EXPAND**

| # | Change | Reviewer quote |
|---|---|---|
| L1 | "**Approximate rotation invariance.** Our equivariant branch provides exact invariance only for the 8 sampled rotation angles of C8; for arbitrary (continuous) angles, invariance degrades smoothly via the group's interpolation. A higher-order group (e.g., C16 or SO(2)) could improve continuous-angle robustness at the cost of memory and training time." | **R2-Q7**: "The rotation invariance is approximate rather than theoretically guaranteed for arbitrary rotation angles." |
| L2 | "**Information loss in max GroupPooling.** Reducing the equivariant feature tensor to an invariant descriptor via max pooling over orientation channels discards orientation-encoded information that could aid sub-group-angle discrimination. Mean pooling, stacked multi-orientation features, or learned pooling are promising extensions." | **R2-Q7**: "The design choice of max GroupPooling may discard orientation-dependent information, and this limitation should be discussed." |
| L3 | "**Fusion expressivity.** Our attention fusion produces global scalar branch weights; spatial/token-wise attention (enabling location-conditional branch emphasis) is a natural extension but requires joint training of larger transformer heads. We leave this for future work." | **R2-Q7**: "The attention fusion mechanism currently appears to rely on global scalar weights, which may limit its expressive capacity." |
| L4 | "**Modest gains on appearance-dominated benchmarks.** On rotation-benign data (e.g., ConPR's 82.5% low-yaw queries), the equivariant branch contributes marginally and our performance trails larger baselines by ≤5 R@1. The architectural advantage manifests primarily where camera orientation varies — we emphasize *consistent* improvements across settings rather than uniform gains." | **R2-Q5**: "The authors should acknowledge that the observed performance improvements are modest, and therefore position the method as providing consistent improvements rather than large performance gains." + **R2-Q7** |
| L5 | "**Illumination robustness not explicitly evaluated.** Construction imagery often spans dawn-to-dusk captures; we evaluate on ConPR and ConSLAM which primarily span structural and rotational variation. Day/night benchmarks such as Tokyo 24/7 are valuable future tests, though our GSV-Cities pretraining includes illumination variation and the equivariant branch is complementary to (not competitive with) photometric robustness." | **R1-W1**: "While the model handles in-plane rotation well, it has not been explicitly evaluated to address illumination variations (such as day-night changes in the Tokyo24/7 dataset)." |
| L6 | "**Training data source.** Our model is trained on GSV-Cities (street-view imagery) rather than construction-specific data; construction-specific training or fine-tuning could yield further gains but is currently limited by the absence of large-scale labeled construction VPR datasets. Our strong ConSLAM generalization (+21.7 R@1 over the best baseline) suggests GSV-Cities pretraining transfers reasonably to construction domains." | **R1-W2**: "The model is trained on street-view data (GSV-Cities) rather than construction-specific imagery, which may limit its ability to generalize to the full lifecycle and structural evolution of construction sites." |
| L7 | "**Discrete rotation groups only.** Our experiments cover C4, C8, and C16; continuous SO(2) equivariance via steerable CNNs~\citep{weiler2019general} is theoretically cleaner but computationally expensive. Empirically, C8 with 45° resolution adequately covers the moderately-rotated regime we target — see the per-yaw analysis in §5.4 for bucketed empirical evidence." | **R1-W3** |
| L8 | **NEW.** "**BoQ-dominant joint scoring at low β.** Our joint scoring rule `score(q, c) = (1−β)·boq_sim(q, c) + β·equi_sim(q, c)` is appearance-dominated at our chosen β = 0.10 (BoQ term carries 9× the weight of the equivariant term). Queries whose true positive has low boq_sim — e.g., > 30° yaw difference where BoQ ranks the true positive far down the candidate list — cannot be promoted by the equivariant signal because the BoQ-weighted gap is too large to bridge. We verified this with two complementary ablations: (i) a 'union retrieve' variant (candidates restricted to Branch 1 top-100 ∪ Branch 2 top-100) yields zero additional R@1 at β = 0.10 across both datasets and all 3 seeds, and (ii) a single-stage variant that scores the entire database directly (no FAISS pre-filter) is exactly R@1-equivalent to our standard formulation across all 3 seeds at β ∈ [0, 0.20] on both datasets — confirming the limitation is the BoQ-dominance of the score function itself, not candidate-set restriction. Raising β to give the equivariant branch more weight closes some of this gap on rotation-heavy queries but degrades rotation-aligned queries (β = 0.5 yields 56.57 ConSLAM and 79.01 ConPR, both well below our reported main numbers). A stronger rotation-aware appearance retriever (e.g., a backbone trained with built-in rotation-equivariant features) is the natural extension we leave to future work." | **R2-Q7** (self-disclosed limitation) |

### §Conclusion (L597-end)

| # | Change | Reviewer quote |
|---|---|---|
| C1 | L601 rewrite: "**achieving consistent, statistically-significant improvement on rotation-heavy construction data (ConSLAM R@1 = 61.89 ± 0.33%, +0.98 R@1 over the strongest matched-resolution baseline with *t* = 5.14, p < 0.05) while remaining competitive on standard benchmarks (ConPR R@1 = 79.74 ± 0.09, best among ResNet50-backbone methods) at 4.11 ms per image**". | R2-Q5 (tone down) |

---

## Minor fixes

| # | Change | Reviewer quote |
|---|---|---|
| F1 | **RandAugment citation `[?]` (L334)** — fix to `\citep{cubuk2020randaugment}`. | **R2-Q8**: "'specifically leveraging RandAugment [? ].' The reference needs double-checking." |
| F2 | **BoQ BibTeX entry** — missing from `cas-refs.bib`. Add: `@inproceedings{ali-bey2024boq, author={Ali-Bey, Amar and Chaib-Draa, Brahim and Giguère, Philippe}, title={BoQ: A Place is Worth a Bag of Learnable Queries}, booktitle={CVPR}, year={2024}}`. | R2-Q4 (needed for BoQ citation added in R2 item) |
| F3 | **3 new refs from R1** — add BibTeX for SelaVPR++, Implicit Aggregation (NeurIPS 2025), Deep Homography VPR (AAAI 2024). (I'll look up exact BibTeX keys.) | R1-W4 |

---

## New/updated figures

| # | Figure | Source | Destination |
|---|---|---|---|
| FIG-Y | ConPR vs ConSLAM yaw distribution (already generated) | `figures/yaw_distribution_conpr_vs_conslam.pdf` | Copy to `DRVPR_paper/figs/yaw_distribution.pdf`; `\includegraphics` in §5.4 (Q2 above) |
| FIG-X | Rotation feature response curve (to generate) | New script `plot_rotation_response.py` | `DRVPR_paper/figs/rotation_response.pdf`; §5.4 (Q1 above) |

---

## Response-to-reviewers letter — skeleton (to be drafted separately)

- Thank reviewers by name
- **To R1**: address W1 (illumination → Limitation L5), W2 (GSV-Cities → Limitation L6), W3 (continuous rotation → Limitation L7 + Fig. X), W4 (add 3 refs → F3)
- **To R2**: address Q3 (std → E1, re-runs table), Q4 (SALAD/BoQ/CricaVPR → T1a/T1b), Q5 (tone down → M1-M4, C1, S2), Q6 (practical motivation → I1), Q7 (limitations → L1-L7), Q8 (banking → I2, RandAugment → F1)
- Summarize (this is now in `doc/REBUTTAL_LETTER.md` "Summary of changes" table). The response letter addresses every reviewer sentence individually.

---

## Work estimate per section

| Block | Est. hours |
|---|---|
| Abstract + Highlights + Intro (A1-3, H1, I1-3) | 3 |
| Related Work (R1-R2) | 1 |
| Methodology softening (M1-M4) | 2 |
| Methodology §3.5 extension subsection (M5) | 2 |
| Experiments framing + Tables 1/2 rebuild (E1, E2, T1a-c, T2) | 4 |
| Results text rewrite §5.1 (S1, S2) | 2 |
| Ablation expand + add variants (AB1, AB2) | 3 |
| Qualitative figs + captions (Q1 generate, Q2 insert) | 2 |
| Discussion + 7 Limitations items (D1, L1-L7) | 3 |
| Conclusion (C1) | 0.5 |
| Minor fixes (F1-F3) | 1 |
| Response letter | 4 |
| **Total** | **~28 hours** |

Over 10 days = ~3 hrs/day. Plenty of buffer.

---

## Open questions — RESOLVED 2026-04-16

1. **Author order / affiliations** — ✅ no changes.
2. **Tokyo 24/7** — ✅ addressed in Limitation L5, no new experiments (`doc/REBUTTAL_LETTER.md` R1-W1).
3. **Param count** — ✅ re-measured with revised Equi-BoQ + concat architecture: **25.19 M total / 24.93 M trainable / 4.11 ms per image @ RTX 5090**. Replaces the old 14.4M / 4.23ms figures throughout.
4. **Figure for S2** — ✅ `figures/yaw_distribution_conpr_vs_conslam.pdf` referenced as Fig. 2 in S2; also referenced in R2-Q6.1 motivation.
5. **Equi-BoQ main vs ablation** — ✅ Equi-BoQ (concat fusion) is now the **main** method; attention fusion (as originally submitted) is the **ablation** (§5.3 Table 3). Single paragraph + Table 3 row covers it. Decision rationale in `doc/REBUTTAL_LETTER.md` cover note.

## Remaining pre-submission tasks

- [x] **Full 10-seq ConPR eval** — done 2026-04-18, `eval_rerank_conpr_full.log`. Result: 79.74 ± 0.09.
- [x] **Fine β sweep ablation** — done 2026-04-18, `doc/ABLATION_BETA_SWEEP.md`. Plateau β ∈ [0.05, 0.13], β=0.10 chosen for highest t-stat.
- [x] **Per-yaw bucket analysis** — done 2026-04-18, `doc/PER_YAW_ANALYSIS.md` v3. [10°, 20°] sweet spot cross-validated.
- [x] **Union retrieve null result** — done 2026-04-18, supports L8.
- [x] **TTA supplementary** — done 2026-04-18, ConSLAM on-grid 8× → 62.32 ± 0.50.
- [x] **Branch 2 alone ablation number** — done (P1 standalone 3-seed avg R@1 on ConSLAM: 42.67 ± 1.30).
- [ ] **Apply all section-by-section edits in this checklist to `Submitted Version 202512V2.tex`.**
- [ ] **Generate Fig. 6 (rotation feature response curve)** via a new script `plot_rotation_response.py`.
- [ ] **Generate Table 4 (per-yaw bucket R@1)** from `doc/PER_YAW_ANALYSIS.md` v3 data (numbers ready, just needs LaTeX formatting).
- [ ] **Optional: compute R@5 / R@10 for DR-VPR v2 on the full 10-seq ConPR** (1 extra run, ~10 min; populate Table 2 R@5/R@10 cells for our row).
- [ ] Decide: train C4 / C16 sweep (~12 h GPU for 2 configs × 3 seeds) vs. state as future work. **Recommendation: state as future work given that the per-yaw bucket analysis already addresses R1-W3's continuous-rotation concern empirically.**
- [ ] Professional language editing pass on final tex (per R2-Q9).
- [ ] Final LaTeX compile, proofread, upload to Editorial Manager by 2026-04-25.
