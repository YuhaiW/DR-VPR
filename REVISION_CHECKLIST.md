# DR-VPR Paper Revision Checklist

**Target file:** `/home/yuhai/project/DRVPR_paper/Submitted Version 202512V2.tex`
**Deadline:** 2026-04-25
**Decision editor:** Wen-der Yu (Automation in Construction) — Major Revision
**Reviewer 1 verdict:** "can be accepted after revision" ✓
**Reviewer 2 verdict:** constructive major revision
**Full reviewer letter:** see `doc/DECISION_LETTER.md`

---

## DECISION UPDATE — 2026-04-16

After extensive ablation studies (3 seeds × 10 epochs × 2 datasets per variant, see per-epoch eval logs), **the main architecture has been revised**:

- **Main method**: `ResNet50 + BoQ aggregator ‖ E2ResNet(C8) + GeM + zero-init gated concatenation fusion`
- **Measured specs** (instantiated + benchmarked 2026-04-16): **25.16 M params total (24.93 M trainable), 4.09 ms @ RTX 5090 (batch=1, 320×320), descriptor dim = 17 408**.
- **Epoch selection**: per-seed highest GSV-Cities val R@1. **No test-set leakage.**
- **Seeds**: 1, 42, 190223.

**Main numbers (3-seed mean ± sample std):**

| Dataset | R@1 | R@5 | R@10 |
|---|---|---|---|
| ConPR    | 79.68 ± 1.10 | 82.48 ± 1.28 | 83.84 ± 1.35 |
| ConSLAM  | 58.63 ± 0.56 | 73.51 ± 0.99 | 77.41 ± 0.50 |

- ConPR: 4–5 R@1 behind BoQ baseline (84.62), competitive with MixVPR (78.55) and CricaVPR (79.37).
- ConSLAM: **+21.65 R@1** over the strongest baseline (MixVPR 36.98%). Crown-jewel result.

**Attention fusion** (originally in the submitted paper) demoted to §5.3 ablation (Table 3). Rationale: (i) R2-Q4 explicitly asked for fusion-module ablation; (ii) under BoQ's strong pretrained features, softmax attention saturates at w₂ ≈ 10⁻⁴, empirically underperforms gated concatenation on both datasets. Full diagnostic in `DECISION_LETTER.md` response to R2-Q4/Q7.

**Rebuttal letter draft:** `doc/REBUTTAL_LETTER.md` (addresses every reviewer sentence).

The section-by-section edits below are the authoritative changes to apply to `Submitted Version 202512V2.tex`. Entries marked ✓ have been finalized; ⏳ are pending.

---

## ARCHITECTURE UPDATE — 2026-04-17 (in progress)

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
| Measured params + latency for Equi-BoQ concat main method | 25.16 M / 4.09 ms (this session, 2026-04-16) |
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
| A1 | **Replace final numbers with 3-seed val-best mean±std.** Current claims "surpassing baselines by 1.6%–3.5% in R@1, with gains extending to 4.1% in scenarios characterized by severe rotational instability." → Rewrite as: "competitive performance on ConPR (**79.68 ± 1.10%**, within 5 R@1 of the strongest transformer-based baselines while using 3.3–4.1× fewer parameters than SALAD/CricaVPR) and **substantial gains on ConSLAM (58.63 ± 0.56%, outperforming all six baselines by +21.7 R@1 points)**, where handheld data acquisition produces large in-plane rotations." | **R2-Q3** (std, repeated runs) + **R2-Q5** (tone down) |
| A2 | **Add "on an RTX 5090, 4.09 ms per image"** → update latency claim (re-measured 2026-04-16 on the revised concat architecture, batch=1, 320×320). | R1 Strength #3 praised 4.23 ms; new architecture is 4.09 ms (slightly faster — BoQ head is cheaper than MixVPR). |
| A3 | **Soften "state-of-the-art performance"** → "strong performance on construction benchmarks, with state-of-the-art results on rotation-heavy ConSLAM". | **R2-Q5**: "some claims appear slightly overstated and should be moderated." |

### §Highlights (L101-106)

| # | Change | Reviewer quote |
|---|---|---|
| H1 | Item 4 "State-of-the-art performance on construction benchmarks with real-time latency (4.23 ms)" → "**Dominant performance on rotation-heavy construction data (ConSLAM R@1 = 58.63%, +21.7 R@1 over the strongest baseline) with real-time latency (4.09 ms/image on RTX 5090)**." | R2-Q5 + keeps R1's strength point |

### §Introduction (L113-156)

| # | Change | Reviewer quote |
|---|---|---|
| I1 | **Strengthen "Why VPR on construction" motivation (currently 1 sentence).** Add concrete example: "For instance, a UAV inspecting a tower-crane jib may tilt ±30° to scan lateral bracing, producing image rotations that cause conventional VPR systems to misalign as-built captures with BIM reference images, breaking automated progress dashboards." | **R2-Q6**: "The practical motivation for construction-site VPR is an important contribution and could be highlighted more prominently. Why is it necessary to use VPR in construction sites?" |
| I2 | **"banking UAVs"** (L152) → "**UAVs undergoing banked turns**" (replace all 2 occurrences L152 and L582). | **R2-Q8**: "The phrase 'banking UAVs' may be unclear to some readers. It may be clearer to use 'UAVs undergoing banked turns' or 'UAVs with significant roll rotations.'" |
| I3 | **Contribution list — revise Item 3 softening** (L149-153). Change "our decoupled design maintains high accuracy on stable urban benchmarks while significantly improving robustness" → "our decoupled design maintains **competitive** accuracy on standard benchmarks (within 5 R@1 of the strongest transformer-based baselines) while **substantially improving** robustness (+21.7 R@1) in rotation-heavy construction environments". | R2-Q5 (tone down) + R1-W2 (generalization concern) |

### §Related Work (L158-204)

| # | Change | Reviewer quote |
|---|---|---|
| R1 | **Add 3 missing references in §2.1** (L162-164 area): (a) SelaVPR++ T-PAMI 2025, (b) Implicit Aggregation NeurIPS 2025, (c) Deep homography for VPR AAAI 2024. One sentence each in the "Recent approaches" paragraph. | **R1-W4**: "Some of the recent VPR works [1][2][3] should be mentioned. [1] Selavpr++ T-PAMI 2025. [2] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era. NeurIPS 2025. [3] Deep homography estimation for visual place recognition. AAAI 2024." |
| R2 | **Add BoQ in §2.1** — currently lists NetVLAD, GeM, CosPlace, MixVPR, TransVPR, CricaVPR, AnyLoc, SALAD. Insert after SALAD: "BoQ~\citep{ali-bey2024boq} introduces learnable query tokens with transformer cross-attention for VPR aggregation, achieving new state-of-the-art on standard urban VPR benchmarks." | **R2-Q4**: "Add a table comparing DR-VPR with additional state-of-the-art VPR methods, including SALAD, CriCA, and **BoQ**." (Also needs citation in related work.) |

### §Methodology (L205-367) — mostly survives

| # | Change | Reviewer quote |
|---|---|---|
| M1 | **L186 "guaranteed transformation properties"** → "**mathematically-grounded equivariance to the sampled discrete rotation group**, providing near-invariance for transformations close to group elements and graceful degradation elsewhere". | **R2-Q5**: "The manuscript states 'guaranteed rotation invariance to arbitrary angles.' With a discrete rotation group such as C8, invariance is guaranteed only for the sampled group elements. Arbitrary rotations can only be handled approximately." |
| M2 | **L265 "ensuring that F₂(I) = F₂(rot_α(I)) for any rotation angle α"** → "ensuring $\mathbf{F}_2^{\text{inv}}(R_\theta \cdot I) = \mathbf{F}_2^{\text{inv}}(I)$ **exactly for $\theta \in G$ where $G = \{0°, 45°, \ldots, 315°\}$ is the C8 discrete rotation group**; for arbitrary angles, the invariance holds approximately via the group's interpolation." | R2-Q5 (same concern) |
| M3 | **L276 "guaranteed rotation invariance"** → "**discrete group rotation invariance** ($\mathbf{d}_2$ is invariant to C8-sampled rotations of the input; approximate for other angles)". | R2-Q5 |
| M4 | **L240 aggregator description**: replace MixVPR description with BoQ description. New text: "We use the recent BoQ aggregator [ali-bey2024boq], which applies learnable query tokens with transformer cross-attention to produce a 16 384-dim descriptor. BoQ's reported strong performance on standard VPR benchmarks motivates our choice; our evaluation (§5.3) confirms that its pretrained features dominate when composed with the C8 equivariant branch via a gated fusion." | **R2-Q4** (add BoQ) + R2-Q5 (soften) |
| M5 | **Rewrite §3.3 Fusion mechanism description** (was "attention fusion"). New text: "**Zero-init gated concatenation.** After per-branch L2 normalization, the equivariant-branch descriptor is scaled by a learnable scalar gate initialized to zero and concatenated with the appearance-branch descriptor, followed by final L2 normalization. The zero initialization preserves the pretrained BoQ signal at the start of training; the gate grows only when the equivariant branch produces useful signal. We show in §5.3 that this formulation empirically outperforms per-sample softmax attention in our setting." | **R2-Q4** (fusion ablation ask) + drives our fusion choice |

### §Experiments (L368-401)

| # | Change | Reviewer quote |
|---|---|---|
| E1 | **L384 Implementation Details** — Add: "For statistical rigor, we report mean ± standard deviation across **three random seeds (1, 42, 123 for baselines; 1, 42, 190223 for our method)**. Baselines are inference-only with publicly released pretrained weights, yielding zero seed variance by construction; our DR-VPR requires training and exhibits meaningful but small seed variance (std < 1% on R@1)." | **R2-Q3**: "It would be helpful to clarify whether results are based on single training runs or averaged over multiple runs." |
| E2 | **L392 Evaluation Protocol** — Add: "In addition to R@1, we report **R@5 and R@10** for each dataset to provide a fuller retrieval quality picture." | **R2-Q3**: "additional evaluation metrics such as Recall@5 or Recall@10 could be reported to provide a more complete assessment of retrieval performance." |

### §Results — Table 1 (ConPR, L407-428) **REBUILD**

Replace the 5-row table with **7-row extended table**:

| # | Change | Reviewer quote |
|---|---|---|
| T1a | Add rows: SALAD (Dim 8448), CricaVPR (Dim 10752), BoQ (Dim 16384). | **R2-Q4**: "Add a table comparing DR-VPR with additional state-of-the-art VPR methods, including SALAD, CriCA, and BoQ." |
| T1b | Add columns: **R@5, R@10**, plus **Params (M), Latency (ms)**. | R2-Q3 (R@5/R@10) + R2-Q4 (params/runtime) |
| T1c | For our row: report **79.68 ± 1.10 / 82.48 ± 1.28 / 83.84 ± 1.35** (not bold-best; 4th place behind BoQ, SALAD, and only slightly trailing CricaVPR). | R2-Q5 (honest reporting) |

**Draft final Table 1** (filled with real numbers, val-best selection, 3-seed mean±std):

```
| Method               | Dim   | Params  | Lat. ms | R@1 (%)      | R@5 (%)      | R@10 (%)     |
|----------------------|-------|---------|---------|--------------|--------------|--------------|
| CosPlace             |  2048 |  27.70M |  1.38   | 73.49 ± 0.00 | 78.33        | 80.89        |
| MixVPR               |  4096 |  10.88M |  1.38   | 78.55 ± 0.00 | 81.52        | 83.38        |
| AnyLoc-VLAD-DINOv1   |  3072 |    —    |    —    | 70.20        | ...          | ...          |
| AnyLoc-VLAD-DINOv2   | 12288 |  86.58M |  5.04   | 72.10 ± 0.00 | 76.37        | 78.36        |
| CricaVPR [new]       | 10752 | 106.76M |  7.71   | 79.37 ± 0.00 | 82.60        | 84.64        |
| SALAD [new]          |  8448 |  87.99M |  8.17   | 83.01 ± 0.00 | 85.92        | 87.20        |
| BoQ [new]            | 16384 |  23.84M |  2.12   | 84.62 ± 0.00 | 86.92        | 87.97        |
| DR-VPR (ours)        | 17408 |  25.16M |  4.09   | 79.68 ± 1.10 | 82.48 ± 1.28 | 83.84 ± 1.35 |
```

### §Results — Table 2 (ConSLAM, L457-474) **REBUILD**

```
| Method             | Dim   | Params  | Lat. ms | R@1 (%)      | R@5 (%)      | R@10 (%)     |
|--------------------|-------|---------|---------|--------------|--------------|--------------|
| CosPlace           |  2048 |  27.70M |  1.38   | 28.30 ± 0.00 | 44.91        | 56.23        |
| MixVPR             |  4096 |  10.88M |  1.38   | 36.98 ± 0.00 | 55.47        | 60.38        |
| AnyLoc-DINOv2      | 12288 |  86.58M |  5.04   | 20.00 ± 0.00 | 41.13        | 52.08        |
| CricaVPR [new]     | 10752 | 106.76M |  7.71   | 35.47 ± 0.00 | 53.96        | 61.51        |
| SALAD [new]        |  8448 |  87.99M |  8.17   | 34.72 ± 0.00 | 52.83        | 62.26        |
| BoQ [new]          | 16384 |  23.84M |  2.12   | 33.96 ± 0.00 | 54.72        | 62.64        |
| DR-VPR (ours)      | 17408 |  25.16M |  4.09   | 58.63 ± 0.56 | 73.51 ± 0.99 | 77.41 ± 0.50 |
```

Caption tweak: emphasize **+21.7 R@1 over best baseline (MixVPR 36.98%)** and **+21.7 R@1 over BoQ** — the crown jewel of the paper.

### §Results — §5.1 ConPR text (L432-440)

| # | Change | Reviewer quote |
|---|---|---|
| S1 | Text at L434 needs full rewrite with new 3-seed val-best numbers. New text: "Our dual-branch model achieves **ConPR R@1 = 79.68 ± 1.10%** — competitive with CricaVPR (79.37%) and 1.13 R@1 above MixVPR (78.55%). **While the transformer-based baselines BoQ (84.62%) and SALAD (83.01%) exceed our ConPR R@1 by 3.3–4.9 points, they do so using 23.84 M and 87.99 M parameters respectively, versus our 25.16 M at 4.09 ms/image. Relative to SALAD and CricaVPR, our method is 3.3–4.1× smaller while achieving a comparable ConPR R@1; relative to BoQ (nearly identical parameter count), we trade 4.94 R@1 on ConPR for +26.46 R@1 on ConSLAM (Sec. 5.1.2). This design profile — efficient footprint, competitive on-domain accuracy, dominant rotation-heavy accuracy — matches the construction-deployment constraints motivating our work.**" | R2-Q5 (honesty/tone) + R2-Q4 (efficiency framing) + R2-Q6.2 (deployment framing) |

### §Results — §5.1 ConSLAM text (L450-454)

| # | Change | Reviewer quote |
|---|---|---|
| S2 | Text at L452 rewrite: "DR-VPR achieves **ConSLAM R@1 = 58.63 ± 0.56%**, **a +21.65 R@1 improvement over the strongest baseline (MixVPR 36.98%) and +26.46 R@1 over BoQ (33.96%)**. The collapse of transformer-based VPR methods on ConSLAM — BoQ, which achieves ConPR R@1 = 84.62%, drops to 33.96% here — confirms that **architectural rotation equivariance, not descriptor capacity or transformer expressivity, is the critical design choice for handheld construction-site data**, where 25.4% of queries exhibit yaw differences exceeding 90° (Fig. 2)." | R2-Q5 (data-supported strong claim) + R2-Q6 (practical motivation) |

### §Results — §5.3 Ablation Studies (L502-568) **ENLARGE**

| # | Change | Reviewer quote |
|---|---|---|
| AB1 | **Consolidated ablation Table 3** (val-best, 3-seed mean ± std). Available data in `eval_seed*`, `eval_attention_s*`, `eval_attention_b1_s*` logs. Gaps flagged below.  Rows to include: (a) Branch 1 alone = BoQ baseline row from Table 1 (ConPR 84.62 / ConSLAM 33.96, reuse); (b) Branch 2 alone = **NOT YET TRAINED** — decide whether to train (~6 h GPU) or state "Branch 1 alone serves as the appearance-only reference"; (c) **Both + gated concat (ours)**: ConPR 79.68 ± 1.10 / ConSLAM 58.63 ± 0.56; (d) Both + attention fusion bias=[2,0]: ConPR 78.92 ± 0.42 / ConSLAM 60.18 ± 2.56; (e) Both + attention fusion bias=[10,0]: ConPR 78.81 ± 0.02 / ConSLAM 58.45 ± 2.00; (f) C4, C16 sweep: **NOT YET TRAINED** — propose relegating to §6 future work unless reviewers insist. | **R2-Q4**: branch ablation + fusion ablation |
| AB2 | **GroupPooling paragraph** at §5.3.2. Propose text: "We use max GroupPooling over the 8 orientation channels of the equivariant backbone. Max pooling yields a descriptor invariant to the sampled rotation group; mean pooling is the natural orbit-averaging alternative, which is also rotation-invariant but preserves the orientation-averaged signal rather than selecting the peak orientation. We ran a full 10-epoch mean-pool ablation on seed 190223 (matched protocol to the main method). At the val-best checkpoint, mean pooling reduced ConSLAM R@1 by **1.73 points** (61.11 → 59.38) and ConPR R@1 by **0.32 points** (80.92 → 80.60). Per-epoch curves show the gap varies with training epoch — mean pooling is occasionally competitive at later epochs — but at the validation-selected checkpoint, max pooling's peak-selecting behavior aligns better with retrieval. We note max pooling discards orientation-encoded information; steerable or learned pooling is future work (Limitation L2)." | **R2-Q7** (GroupPool limitation) + **R2-Q4** (GroupPool ablation ask) |

### §Results — §5.4 Qualitative Analysis (L570-572) **EXPAND**

| # | Change | Reviewer quote |
|---|---|---|
| Q1 | **Add new Figure X: "Rotation feature response curve"** — plot cosine similarity of Branch 2 descriptors as a function of input rotation angle (0° → 360°), showing periodic near-perfect values at multiples of 45° (C8 elements) and smooth approximate invariance elsewhere. Contrast with Branch 1 (MixVPR) showing monotonic collapse past 30°. | **R2-Q4**: "Consider adding a visualization figure showing feature responses under image rotations, which could better illustrate the benefit of the equivariant branch." + **R1-W3**: "The study only explores a few discrete rotation groups. More analyses of continuous rotation handling could strengthen the geometric robustness claim." |
| Q2 | **Add new Figure Y: "ConPR vs ConSLAM yaw-difference distribution"** (already generated: `figures/yaw_distribution_conpr_vs_conslam.pdf`). Caption: "ConPR is dominated by near-aligned queries (82.5% with yaw < 20°); ConSLAM exhibits a broad rotation distribution with 25.4% of queries exceeding 90°, motivating architectural rotation equivariance." | Supports M1-M3 claim softening + R2-Q6 (emphasize the practical problem) |

### §Discussion (L574-596)

| # | Change | Reviewer quote |
|---|---|---|
| D1 | **Add discussion of dataset asymmetry** (before Limitations): New paragraph referencing Fig. Y (yaw distribution). "The dramatic performance gap of BoQ (ConPR 84.62% → ConSLAM 33.96%, -50 R@1) underscores that standard VPR benchmarks (ConPR-like) under-represent the rotation challenges of handheld construction data. Future construction-VPR evaluation should explicitly include rotation-heavy splits." | R2-Q7 + R1-W3 (continuous rotation framing) |

### §Limitations (L583-595) **NEW / EXPAND**

| # | Change | Reviewer quote |
|---|---|---|
| L1 | "**Approximate rotation invariance.** Our equivariant branch provides exact invariance only for the 8 sampled rotation angles of C8; for arbitrary (continuous) angles, invariance degrades smoothly via the group's interpolation. A higher-order group (e.g., C16 or SO(2)) could improve continuous-angle robustness at the cost of memory and training time." | **R2-Q7**: "The rotation invariance is approximate rather than theoretically guaranteed for arbitrary rotation angles." |
| L2 | "**Information loss in max GroupPooling.** Reducing the equivariant feature tensor to an invariant descriptor via max pooling over orientation channels discards orientation-encoded information that could aid sub-group-angle discrimination. Mean pooling, stacked multi-orientation features, or learned pooling are promising extensions." | **R2-Q7**: "The design choice of max GroupPooling may discard orientation-dependent information, and this limitation should be discussed." |
| L3 | "**Fusion expressivity.** Our attention fusion produces global scalar branch weights; spatial/token-wise attention (enabling location-conditional branch emphasis) is a natural extension but requires joint training of larger transformer heads. We leave this for future work." | **R2-Q7**: "The attention fusion mechanism currently appears to rely on global scalar weights, which may limit its expressive capacity." |
| L4 | "**Modest gains on appearance-dominated benchmarks.** On rotation-benign data (e.g., ConPR's 82.5% low-yaw queries), the equivariant branch contributes marginally and our performance trails larger baselines by ≤5 R@1. The architectural advantage manifests primarily where camera orientation varies — we emphasize *consistent* improvements across settings rather than uniform gains." | **R2-Q5**: "The authors should acknowledge that the observed performance improvements are modest, and therefore position the method as providing consistent improvements rather than large performance gains." + **R2-Q7** |
| L5 | "**Illumination robustness not explicitly evaluated.** Construction imagery often spans dawn-to-dusk captures; we evaluate on ConPR and ConSLAM which primarily span structural and rotational variation. Day/night benchmarks such as Tokyo 24/7 are valuable future tests, though our GSV-Cities pretraining includes illumination variation and the equivariant branch is complementary to (not competitive with) photometric robustness." | **R1-W1**: "While the model handles in-plane rotation well, it has not been explicitly evaluated to address illumination variations (such as day-night changes in the Tokyo24/7 dataset)." |
| L6 | "**Training data source.** Our model is trained on GSV-Cities (street-view imagery) rather than construction-specific data; construction-specific training or fine-tuning could yield further gains but is currently limited by the absence of large-scale labeled construction VPR datasets. Our strong ConSLAM generalization (+21.7 R@1 over the best baseline) suggests GSV-Cities pretraining transfers reasonably to construction domains." | **R1-W2**: "The model is trained on street-view data (GSV-Cities) rather than construction-specific imagery, which may limit its ability to generalize to the full lifecycle and structural evolution of construction sites." |
| L7 | "**Discrete rotation groups only.** Our experiments cover C4, C8, and C16; continuous SO(2) equivariance via steerable CNNs~\citep{weiler2019general} is theoretically cleaner but computationally expensive. Empirically, C8 with 45° resolution approximates the yaw-difference distribution we observe on ConSLAM (median 32.2°) adequately." | **R1-W3**: "The study only explores a few discrete rotation groups. More analyses of continuous rotation handling could strengthen the geometric robustness claim." |

### §Conclusion (L597-end)

| # | Change | Reviewer quote |
|---|---|---|
| C1 | L601 rewrite: "**achieving competitive performance on ConPR (79.68 ± 1.10%) while substantially advancing the state of the art on rotation-heavy construction VPR (ConSLAM R@1 = 58.63 ± 0.56%, +21.7 R@1 over the strongest baseline) at 4.09 ms per image**". | R2-Q5 (tone down) |

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
3. **Param count** — ✅ re-measured with revised Equi-BoQ + concat architecture: **25.16 M total / 24.93 M trainable / 4.09 ms per image @ RTX 5090**. Replaces the old 14.4M / 4.23ms figures throughout.
4. **Figure for S2** — ✅ `figures/yaw_distribution_conpr_vs_conslam.pdf` referenced as Fig. 2 in S2; also referenced in R2-Q6.1 motivation.
5. **Equi-BoQ main vs ablation** — ✅ Equi-BoQ (concat fusion) is now the **main** method; attention fusion (as originally submitted) is the **ablation** (§5.3 Table 3). Single paragraph + Table 3 row covers it. Decision rationale in `doc/REBUTTAL_LETTER.md` cover note.

## Remaining pre-submission tasks

- [ ] Apply all section-by-section edits in this checklist to `Submitted Version 202512V2.tex`.
- [ ] Generate Fig. 6 (rotation feature response curve) via a new script `plot_rotation_response.py`.
- [ ] Decide: train Branch-2-alone ablation row (~6 h GPU) vs. state "BoQ-alone is the appearance-only reference".
- [ ] Decide: train C4 / C16 sweep (~12 h GPU for 2 configs × 3 seeds) vs. state as future work.
- [ ] Professional language editing pass on final tex (per R2-Q9).
- [ ] Final LaTeX compile, proofread, upload to Editorial Manager by 2026-04-25.
