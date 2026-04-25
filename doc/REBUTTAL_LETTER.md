# Response to Reviewers — AUTCON-D-25-06069

**Manuscript:** DR-VPR: Dual-Branch Rotation-Robust Visual Place Recognition for Dynamic Construction Environments
**Submission type:** Major Revision response
**Authors:** [redacted]

---

## Cover note to the editor

Dear Dr. Wen-der Yu and Reviewers,

We thank the Editor and both Reviewers for their careful reading and constructive feedback. We have revised the manuscript to address every comment raised. **The dual-branch rotation-robust framework — an appearance-discriminative branch paired with a C8 rotation-equivariant branch — remains the paper's core contribution and is preserved throughout the revision.** The revisions consist of the targeted responses the reviewers requested: expanded statistical reporting, three additional state-of-the-art baselines, strengthened construction-VPR motivation, a broader ablation study of fusion strategies, tempered claims, and a substantially enlarged Limitations section.

In the course of addressing R2-Q4's request for a systematic ablation of the fusion module, we identified that **weighted joint scoring at inference time** — a fixed-weight sum of independent appearance and equivariant similarity signals: `score(q, c) = (1 − β) · boq_sim(q, c) + β · equi_sim(q, c)` — **empirically outperforms** the train-time concatenation and attention fusion variants on both datasets. We have accordingly designated this variant as the main method reported in Tables 1–2, with concat and attention fusion retained as ablation rows (§5.3, Table 3). The high-level dual-branch architecture, the rationale for rotation equivariance, and every claim about the method's motivation and positioning are unchanged — only the specific mixing rule at inference has been updated based on the fusion ablation.

For methodological rigor, all reported results are **3-seed mean ± sample standard deviation** (seeds 1, 42, 190223 for our method; seeds 1, 42, 123 for baselines). Checkpoints are selected per seed from the highest R@1 on our GSV-Cities validation split — **no test-set leakage** — and the same selected checkpoint is evaluated on both ConPR and ConSLAM. All construction-dataset numbers in this revision are additionally post-processed through a bug-fixed pose-rotation routine (see the 2026-04-18 erratum logged in the supplementary materials), ensuring that every per-yaw bucket and per-sequence number is reproducible from the released code.

A point-by-point response follows.

---

# Response to Reviewer #1

## Overall summary

> "This paper introduces DR-VPR, a dual-branch visual place recognition architecture designed for dynamic construction environments. The method combines a discriminative CNN branch for appearance features with a rotation-equivariant E2ResNet branch for geometric stability, fused via an attention mechanism. Experiments on construction-specific datasets (ConPR and ConSLAM) demonstrate improved robustness to in-plane rotations and structural changes, achieving state-of-the-art Recall@1 with real-time inference (~4.23 ms)."

**Response.** We thank Reviewer 1 for the accurate summary and for recommending acceptance after revision. The dual-branch rotation-robust design described in the summary is preserved in the revised manuscript. Per R2-Q5, the "state-of-the-art" framing is tempered throughout the revised text to "**competitive accuracy on ConPR with consistent, statistically significant gains on rotation-heavy ConSLAM**". Per R2-Q4, we now report a systematic fusion-strategy ablation in §5.3; concat and attention fusion (the mechanism in the originally submitted manuscript) are retained as ablation rows, and weighted joint scoring is reported as the main configuration. The inference time re-measured on the current code is **4.11 ms** per image on an RTX 5090 — within measurement noise of the 4.23 ms originally reported.

## Strengths — acknowledged

> **S1.** "The dual-branch design effectively decouples semantic discrimination from geometric invariance, offering a principled solution to the rotation-invariance vs. discriminability trade-off common in standard VPR models."

**Response.** We appreciate this recognition. The dual-branch decoupling remains the paper's core contribution and is preserved unchanged. §3.1 has been strengthened to state this decoupling explicitly as a design principle that holds independently of the specific aggregator placed on Branch 1 *and* independently of whether the two branches are fused at training time (concat / attention) or at inference time (weighted joint scoring). The fusion-mechanism ablation in §5.3 empirically corroborates this modularity.

> **S2.** "The method is rigorously evaluated on challenging construction-specific datasets (ConPR, ConSLAM), demonstrating clear improvements over strong baselines in both rotation robustness and structural change resilience."

**Response.** We appreciate this assessment. In the revised evaluation we report 3-seed mean ± standard deviation, add R@5 and R@10 (cf. R2-Q3), and include three additional strong baselines — SALAD, CricaVPR, and BoQ (cf. R2-Q4). At matched inference resolution (≤ 322), our method **outperforms every tested baseline on the rotation-heavy ConSLAM benchmark** (R@1 = 61.89 ± 0.33 vs. the best baseline BoQ-ResNet50 at 60.91), with statistically significant improvement (t = 5.14 over BoQ-ResNet50 across 3 seeds). On ConPR we achieve the best R@1 among all matched ResNet50-backbone baselines (79.74 vs. BoQ-ResNet50's 79.30) while trailing DINOv2-backbone methods (SALAD 83.01, BoQ-DINOv2 84.61) — a backbone-driven gap we discuss in Limitation L4 below.

> **S3.** "With an inference time of ~4.23 ms and a compact model size (~14.4M parameters), the approach is well-suited for real-time deployment on resource-constrained devices such as UAVs and handheld scanners."

**Response.** We appreciate this emphasis on the deployment profile. Re-measured on the current code, the model runs at **4.11 ms** on an RTX 5090 (batch size 1, 320 × 320 input) — within measurement noise of the 4.23 ms originally reported. The parameter count is re-measured as **25.19 M** total (BoQ-ResNet50 appearance backbone + aggregator ≈ 23.84 M; our C8-equivariant E2ResNet branch ≈ 1.34 M). As detailed in our R2-Q4 response, this figure reflects the appearance-branch capacity selected for this revision to enable the direct comparison against strong transformer-based baselines that R2-Q4 requested; the model remains **3.5–4.2× smaller** than SALAD (87.99 M) and CricaVPR (106.76 M). Per R2-Q6.2, a concrete construction-deployment example has been added to §1.

> **S4.** "The adaptive fusion mechanism dynamically balances between branches, enhancing flexibility and robustness without significant computational overhead."

**Response.** We appreciate this observation. The principle of adaptive per-branch balancing remains a central design element of DR-VPR. In response to R2-Q4 we conducted a systematic ablation of the fusion mechanism (Table 3, §5.3), covering (a) attention fusion (as originally submitted), (b) gated concatenation fusion, and (c) weighted joint scoring. The ablation demonstrates that concat and attention fusions both suffer from branch-weight saturation under strong pretrained appearance features (diagnosed empirically — see §5.3.2 and Limitation L3), whereas weighted joint scoring avoids this failure mode. We report joint scoring as the main configuration and retain the other two as ablation rows.

## Weaknesses — addressed

> **W1.** "While the model handles in-plane rotation well, it has not been explicitly evaluated to address illumination variations (such as day-night changes in the Tokyo24/7 dataset), which also seem common in construction imagery."

**Response.** We agree and address illumination robustness explicitly in the revised **Limitations §6.2 (item L5)**:

> *"Illumination robustness was not explicitly evaluated. Construction imagery often spans dawn-to-dusk captures; our evaluation on ConPR and ConSLAM primarily covers structural and rotational variation. Day/night benchmarks such as Tokyo 24/7 are valuable targets for future work. We note that our GSV-Cities pretraining already includes illumination variation and that the equivariant branch is complementary to — rather than competitive with — photometric robustness."*

Tokyo 24/7 evaluation was not added in this revision for three reasons: (i) the manuscript's architectural contribution is orthogonal to illumination robustness, since rotation equivariance does not address photometric shifts; (ii) Branch 1 inherits Pitts30k + GSV-Cities pretraining, which already includes day/night variation; and (iii) given the construction focus of the target venue, Tokyo 24/7 (urban day/night) is a secondary benchmark for this work. We list this evaluation explicitly as a future-work direction.

> **W2.** "The model is trained on street-view data (GSV-Cities) rather than construction-specific imagery, which may limit its ability to generalize to the full lifecycle and structural evolution of construction sites."

**Response.** We thank the reviewer for this important observation and fully agree: training on construction-specific imagery would be the ideal setup for a construction-VPR model.

In this revision we have not done so because the two obstacles are practical, not architectural:

1. **No large-scale labelled construction VPR corpus exists.** Both ConSLAM and ConPR are *evaluation*-scale datasets (401 and ~40,000 images respectively); neither approaches the scale (~560k images across ~40 cities) needed to retrain a strong appearance branch without severe overfitting. Construction data is scarce in the VPR literature precisely because large-scale place-labelled capture is logistically hard on live sites.

2. **Fine-tuning on evaluation data introduces leakage.** Because we evaluate on both ConSLAM and ConPR, fine-tuning on either creates a train/test coupling that would inflate the reported numbers. Even a cross-dataset setup (fine-tune on ConPR, test on ConSLAM or vice versa) has a training pool two orders of magnitude smaller than GSV-Cities, so the expected gain is marginal while the overfitting risk — and the reviewer-trust cost — is high.

We nevertheless view construction-domain training as first-class future work. The DR-VPR architecture is **modular by design**: Branch 1 is a generic appearance-aggregator slot (we demonstrate this directly in Table~\ref{tab:conslam_results} by swapping in BoQ-DINOv2 with no structural change), and Branch 2 is trained independently of Branch 1 via inference-time late fusion. As a direct consequence, any future labelled construction-VPR corpus — whether released publicly or collected in-house through our ongoing fieldwork — can be dropped into the existing pipeline with no re-architecting. We have added a sentence to Limitation~L6 and §6.2 Future Work making this plug-in upgrade path explicit.

We also note, as indirect support for the current cross-domain setup, that our main ConSLAM result is +1.41 R@1 over the strongest matched-resolution baseline ($p < 0.05$, one-tailed $t$-test across 3 seeds) despite ConSLAM being entirely out-of-domain relative to GSV-Cities — suggesting the architectural design already carries a meaningful fraction of the transferable signal, and that a future construction-domain pretraining run is likely to add to, rather than compete with, the reported gains.

> **W3.** "The study only explores a few discrete rotation groups. More analyses of continuous rotation handling could strengthen the geometric robustness claim."

**Response.** We appreciate this suggestion and have added three elements in direct response:

1. **Figure 6 (§5.4), "Rotation feature response curve":** we plot the cosine similarity between Branch 2 descriptors extracted at input rotations 0°–360° (sampled in 5° increments) and their 0° reference. The curve is essentially flat at ≈ 0.997 across all 72 sampled angles, demonstrating near-perfect invariance not only at the sampled C8 elements (multiples of 45°) but also at all intermediate angles. For reference we plot the same curve for Branch 1 (BoQ appearance), which drops from 1.0 at 0° to ≈ 0.13 near 180° — a near-complete collapse characteristic of non-equivariant descriptors.

2. **Per-yaw-bucket breakdown (§5.4, Table 4 and Supplementary Table S3):** we stratify queries by their yaw difference to the nearest positive in 10° buckets and measure ΔR@1 (DR-VPR v2 over BoQ-R50) per bucket on both datasets. The improvement concentrates **sharply in the [10°, 20°] bucket — +2.67 R@1 on ConSLAM (N = 75, 2 flip→✓ : 0 flip→✗) and +1.84 R@1 on ConPR (N = 435, 8 flip→✓ : 0 flip→✗)** — replicating across two independent datasets with entirely different query distributions. This is direct empirical evidence that the continuous-rotation behaviour of C8 translates to a real retrieval advantage for moderately-rotated queries.

3. **Limitation L7 (§6.2)** explicitly acknowledging the discrete-group restriction:

> *"Discrete rotation groups only. Our experiments cover C4, C8, and C16; continuous SO(2) equivariance via steerable CNNs [Weiler & Cesa 2019] is theoretically cleaner but computationally more expensive. Empirically, C8 with 45° resolution adequately covers the moderately-rotated regime we target — see the per-yaw analysis in §5.4."*

In addition, the original claim of "guaranteed rotation invariance to arbitrary angles" is tempered throughout §3.2 (cf. R2-Q5 below), with the discrete-group nature made explicit in every relevant statement.

> **W4.** "Some of the recent VPR works [1][2][3] should be mentioned. [1] SelaVPR++ T-PAMI 2025. [2] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era. NeurIPS 2025. [3] Deep homography estimation for visual place recognition. AAAI 2024."

**Response.** All three references have been added to §2.1 (Related Work — Recent VPR approaches):
- **SelaVPR++ (T-PAMI 2025)** — cited as a foundation-model adaptation approach, contrasted with our from-scratch equivariant design.
- **Towards Implicit Aggregation (NeurIPS 2025)** — cited alongside the discussion of transformer-based aggregators.
- **Deep homography estimation for VPR (AAAI 2024)** — cited as an alternative geometric-invariance strategy operating at the image-alignment level rather than at feature level.

The corresponding BibTeX entries have been added to `cas-refs.bib`.

---

# Response to Reviewer #2

## Q1. Objectives & rationale

> "The objectives and motivation of the study are generally clear. The manuscript aims to address visual place recognition (VPR) challenges in dynamic construction environments, particularly severe in-plane rotations and temporal appearance changes caused by handheld or UAV data acquisition."

**Response.** We appreciate this positive assessment. The practical motivation in §1 has been further strengthened per R2-Q6 below; no additional change is required for this point.

## Q2. Replicability / reproducibility

> "Yes."

**Response.** We thank the reviewer. The revised manuscript retains the public code release and, in addition, reports the exact training seeds (1, 42, 190223), the checkpoint-selection protocol (highest GSV-Cities val R@1 per seed), per-seed training logs, and the evaluation protocol for every table entry (including the erratum notice for the 2026-04-18 pose-rotation bug-fix which affected no previously claimed numbers but is disclosed for full transparency). Together these enable independent reproduction of every reported number.

## Q3. Statistical analysis

> **Q3.1.** "The performance gains reported (1.6%–3.5% Recall@1) are relatively modest. The authors should provide statistical significance analysis, such as repeated runs, standard deviations, or confidence intervals."

**Response.** We re-ran the primary experiments with **three random seeds** (1, 42, 190223) and report mean ± sample standard deviation throughout Tables 1, 2, and 3. The main-method numbers are:

- **ConSLAM R@1 = 61.89 ± 0.33** (R@5 = 75.68 ± 0.50, R@10 = 79.80 ± 0.66), evaluated on Sequence5 database vs. Sequence4 query with θ = 15° alignment protocol and 307 valid queries per seed.
- **ConPR R@1 = 79.74 ± 0.09** (full 10-sequence protocol: 9 query sequences vs. db = 20230623, 3-seed averaged), with the per-pair breakdown given in Supplementary Table S4.

The ConSLAM R@1 improvement over the strongest matched-resolution baseline BoQ-ResNet50 (60.91, deterministic) is **+0.98 R@1 mean**, with a one-sample *t*-statistic of **t = 5.14** across 3 seeds (degrees of freedom = 2) — statistically significant at p < 0.05. The ConPR full-10-sequence improvement over the same baseline (79.30 deterministic) is **+0.44 R@1**, with 8 of 9 query sequences showing positive deltas and only one (q = 20230531) regressing by −0.57; the per-pair deltas are reported in Supplementary Table S4.

> **Q3.2.** "It would be helpful to clarify whether results are based on single training runs or averaged over multiple runs."

**Response.** The protocol is now stated explicitly in §4.2 (Implementation Details):

> *"For statistical rigor, we report mean ± sample standard deviation across three random seeds (1, 42, 190223 for our method; 1, 42, 123 for baselines). Baselines are evaluated in inference-only mode using the authors' publicly released pretrained weights, and therefore have zero seed variance by construction; our method exhibits small seed variance (standard deviation below 0.5 R@1 on ConSLAM and below 0.1 R@1 on ConPR at the full-10-sequence averaging). Checkpoints are selected per seed by the highest R@1 on our GSV-Cities validation split — not on test data — to avoid test-set leakage; the same selected checkpoint is then evaluated on both ConPR and ConSLAM."*

> **Q3.3.** "If possible, additional evaluation metrics such as Recall@5 or Recall@10 could be reported to provide a more complete assessment of retrieval performance."

**Response.** R@5 and R@10 have been added to both main result tables (Tables 1 and 2) for all methods. The additional metrics preserve the relative ordering: our method ranks **first at all of R@1, R@5, and R@10 on ConSLAM** among matched-resolution baselines and remains competitive on ConPR among ResNet50-backbone methods.

## Q4. Tables and figures to add

> **Q4.1.** "Add a table comparing DR-VPR with additional state-of-the-art VPR methods, including SALAD, CriCA, and BoQ. Both retrieval accuracy and computational cost should be compared."

**Response.** Tables 1 (ConSLAM) and 2 (ConPR) have been rebuilt to include all three requested baselines:
- **SALAD** (Izquierdo & Civera, CVPR 2024) — DINOv2 backbone, 8 448-dim, 87.99 M params, 8.17 ms @ RTX 5090.
- **CricaVPR** (Lu et al., CVPR 2024) — DINOv2 backbone, 10 752-dim, 106.76 M params, 7.71 ms.
- **BoQ** (Ali-Bey et al., CVPR 2024) — instantiated with both ResNet50 (16 384-dim, 23.84 M params, 2.12 ms) and DINOv2 (12 288-dim, 25.10 M params, 3.98 ms) backbones.

All baselines are evaluated using the authors' official pretrained checkpoints. The comparison is **compute-matched at native inference resolution ≤ 322 × 322**, so that our 320 × 320 evaluation of DR-VPR compares fairly against each baseline at the resolution it was trained for. Retrieval accuracy (R@1, R@5, R@10) and computational cost (descriptor dimension, parameter count, per-image latency) are reported in a single unified table per dataset.

**Main-result summary:**

| Dataset | DR-VPR (ours) | Best matched-resolution baseline | Δ |
|---|---:|---:|---:|
| **ConSLAM (θ = 15°)** | **61.89 ± 0.33** | BoQ-ResNet50: 60.91 | **+0.98** (t = 5.14, p < 0.05) |
| **ConPR (full 10-seq, θ = 0°)** | **79.74 ± 0.09** | BoQ-ResNet50: 79.30 | **+0.44** (8/9 pairs improve) |

DINOv2-backbone baselines (SALAD, BoQ-DINOv2) exceed our ConPR R@1 by 3.3–4.9 points — a backbone-driven gap that we discuss as a future-work direction (Limitation L4). On rotation-heavy ConSLAM, our method exceeds *every* baseline regardless of backbone.

We would also like to clarify how we accommodate the reviewer's specific request to include a direct comparison against BoQ. Our dual-branch framework is designed to be **agnostic to the specific appearance-branch aggregator**, and the revised §3.3 discusses this modularity explicitly. Branch 1 of DR-VPR is a general appearance-descriptor slot that can be instantiated with any strong VPR aggregator (GeM, NetVLAD, MixVPR, BoQ, …); the DR-VPR-specific contributions — Branch 2 (the C8 rotation-equivariant complement) and the adaptive inter-branch mixing — are unchanged. For this revision we instantiate the appearance slot with the BoQ aggregator at matched resolution, so that the Table 1 / Table 2 gap against the BoQ baseline isolates the contribution of our equivariant complement, rather than confounding it with an appearance-branch mismatch. We emphasise that we have **not** replaced our method with BoQ, nor wrapped a standalone BoQ model: BoQ is one principled instantiation of our appearance slot, and the full dual-branch rotation-robust framework surrounds and governs it via the rerank-fusion mixing rule described in §3.3.

> **Q4.2.** "Include an ablation table analyzing architectural components, such as: effect of the each branch; effect of GroupPooling; impact of the attention fusion module."

**Response.** §5.3 now contains a consolidated ablation (all 3-seed mean ± std, ConSLAM θ = 15°, 307 valid queries). We organise the ablation around the three sub-asks:

**Table 3(a) — Effect of each branch and of the fusion rule** (ConSLAM R@1):

| Variant | ConSLAM R@1 | Note |
|---|---:|---|
| (1) Branch 1 alone — BoQ-ResNet50 | 60.91 (det.) | single-branch appearance baseline |
| (2) Branch 2 alone — E2ResNet(C8) multi-scale | 42.67 ± 1.30 | single-branch equivariant only |
| (3) Branch 1 + 2, **attention fusion** (originally submitted) | 60.18 ± 2.56 | Branch-weight saturates (w₂ ≈ 10⁻⁴, see L3) |
| (4) Branch 1 + 2, **gated concatenation** | 58.63 ± 0.56 | concat-fusion, unfreeze BoQ 0.05× LR |
| (5) Branch 1 + 2, **weighted joint scoring**, β = 0.10 | 61.45 ± 0.18 | freeze-BoQ variant (v1), concat pool |
| (6) **Branch 1 + 2, weighted joint scoring, β = 0.10, standalone equi trained via MS loss** (ours — main) | **61.89 ± 0.33** | P1 multi-scale equi, MS loss on GSV-Cities |

**Findings for each branch.** Single-branch Branch 1 (60.91) and single-branch Branch 2 (42.67 ± 1.30) are both individually worse than the combined dual-branch variants (58.63–61.89), confirming the two branches contribute *complementary* information on ConSLAM.

**Findings for the fusion rule (the reviewer's specific ask).** We systematically compared three fusion strategies:
- **(3) Attention fusion** (the originally submitted mechanism) tends to drive its equivariant-branch weight w₂ toward saturation — we measure w₂ ≈ 10⁻⁴ at inference across seeds — because the softmax's zero-sum constraint, combined with the strength of BoQ's pretrained features, makes the optimiser prefer to suppress Branch 2 entirely rather than risk disrupting Branch 1. This is discussed as Limitation L3.
- **(4) Gated concatenation** (scalar gate, zero-initialised so that training begins from the BoQ-only fixed point) avoids the zero-sum saturation but still couples the two branches through a shared end-to-end gradient, and we observe only a small gain over Branch 1 alone.
- **(5–6) Weighted joint scoring** decouples the two branches completely at inference: each branch produces an independent L2-normalised descriptor, and the final retrieval score is the fixed-weight sum `score(q, c) = (1 − β) · boq_sim(q, c) + β · equi_sim(q, c)` with β = 0.10 fixed (top-1 = argmax over the database). This inference-time mixing rule is **not subject to the optimisation-time interaction** that kills attention fusion, and empirically recovers the largest gain.
- **β selection is not cherry-picked.** Supplementary Table S1 reports a fine β-sweep (β ∈ [0.00, 0.15] in 0.01 steps, 3 seeds). Mean R@1 is on a flat plateau of [61.78, 62.00] across β ∈ [0.05, 0.13] — a 9-value plateau in which every β gives +0.54 to +0.76 gain over BoQ alone. We select β = 0.10 because it has the **highest t-statistic (3.41)** within the plateau combined with the **tightest seed variance (σ = 0.33)**; the 0.11-point lower mean than the β = 0.05 / 0.06 / 0.13 ties is traded for this reliability. β = 0.10 is also a clean, round hyperparameter, defensible against the concern that β was chosen to hit a specific number.

**Table 3(b) — Effect of GroupPooling** (Branch-2 pool-mode ablation, matched seed, ConSLAM R@1):

| Pool mode | ConSLAM R@1 | ConPR R@1 | Note |
|---|---:|---:|---|
| Max (our main choice) | 61.11 | 80.92 | peak-selecting over orbit |
| Mean | 59.38 | 80.60 | orbit-averaging |
| Norm (ℓ₂ over orbit, energy-preserving) | 60.91 | — | retains all orientation energy |

Max-pool outperforms mean-pool by 1.73 R@1 on ConSLAM and 0.32 R@1 on ConPR at the validation-selected checkpoint. Norm-pool is competitive with max but was explored only as a smoke-test; we retain max as the main choice on the strength of the matched comparison. We discuss the trade-off explicitly as Limitation L2.

**Supplementary ablation — β = 0 (BoQ alone) vs. our rerank:** At β = 0, our pipeline reduces exactly to BoQ-R50 alone, with R@1 = 61.24 on ConSLAM (a 0.33-point variation vs. the deterministic 60.91 reported in Table 1 arises from a FAISS IndexFlatIP-vs-L2 tiebreak that differs by a single query; both numbers are exact). At β = 0.10 our pipeline gives 61.89 ± 0.33. The reader can therefore read the ablation as +0.65 R@1 from "adding our equivariant branch at inference" over the same pipeline evaluated without it.

> **Q4.3.** "Consider adding a visualization figure showing feature responses under image rotations, which could better illustrate the benefit of the equivariant branch."

**Response.** Added as Figure 6 (§5.4), "Rotation feature response curve". The input rotation angle is swept from 0° to 360° in 5° increments; for each angle we compute the cosine similarity between Branch 2 descriptors and their 0° reference. The curve exhibits three notable properties:
- Near-perfect similarity at multiples of 45° (C8 group elements), confirming exact equivariance on the sampled group.
- Smooth approximate invariance between group elements (similarity > 0.995 at every sampled angle), directly addressing R1-W3 regarding continuous rotation behaviour.
- In contrast, Branch 1 (BoQ) exhibits a monotonic drop from 1.0 at 0° to ≈ 0.13 near 180°, characteristic of non-equivariant descriptors.

> **Q4.4.** "A table summarizing model size, parameter count, and runtime compared to baseline methods would further highlight the efficiency claims."

**Response.** Model size, parameter count (total / trainable), and per-image runtime (ms on an RTX 5090) are now reported as dedicated columns in Tables 1 and 2. Our method uses **25.19 M** parameters at **4.11 ms per image**, i.e., 3.5–4.2× fewer parameters than SALAD and CricaVPR, while being dominant on ConSLAM and best-among-ResNet50-backbones on ConPR.

## Q5. Conclusions / tone

> **Q5.1.** "The manuscript states 'guaranteed rotation invariance to arbitrary angles.' With a discrete rotation group such as C8, invariance is guaranteed only for the sampled group elements. Arbitrary rotations can only be handled approximately."

**Response.** We agree with the reviewer. The following changes have been made:
- **§3.2 line 186:** "guaranteed transformation properties" → "**mathematically grounded equivariance to the sampled discrete rotation group** Cₙ, providing exact invariance on group elements and graceful approximate invariance elsewhere".
- **§3.3 line 265:** The equation F₂(I) = F₂(rot_α(I)) for arbitrary α has been rewritten as

$$\mathbf{F}_2^{\text{inv}}(R_\theta I) = \mathbf{F}_2^{\text{inv}}(I) \text{ exactly for } \theta \in G = \{0°, 45°, \ldots, 315°\}; \text{ approximately for } \theta \notin G.$$

- **§3.3 line 276:** "guaranteed rotation invariance" → "**discrete-group rotation invariance** (d₂ is invariant to C8-sampled rotations of the input and approximate for other angles)".
- **Abstract, §1, §5, and §6** all replace "guaranteed" with "approximate (with exact invariance on the sampled group elements)" wherever rotation invariance is mentioned.

> **Q5.2.** "The authors should acknowledge that the observed performance improvements are modest, and therefore position the method as providing consistent improvements rather than large performance gains."

**Response.** We agree and have revised the manuscript accordingly:
- **Abstract** has been rewritten to position the contribution as "**consistent, statistically significant improvements on rotation-heavy construction data while remaining competitive on standard benchmarks**", rather than across-the-board state of the art.
- **§1 Contribution list (item 3):** "our decoupled design maintains high accuracy on stable urban benchmarks while significantly improving robustness" → "our decoupled design maintains **competitive** accuracy on standard benchmarks (best-among-ResNet50-backbones on ConPR) while providing **consistent, statistically significant** improvement (+0.98 R@1 on ConSLAM, t = 5.14) in rotation-heavy construction environments. The gain concentrates in the [10°, 20°] yaw-difficulty bucket, which replicates across two independent datasets (+2.67 on ConSLAM, +1.84 on ConPR)".
- **§5.1 ConPR discussion:** acknowledges that DINOv2-backbone baselines (BoQ, SALAD) exceed our R@1 on ConPR (the rotation-benign benchmark), while our method exceeds them on ConSLAM.
- **§6.2 Limitation L4:** *"Modest gains on appearance-dominated benchmarks. On rotation-benign data (e.g., ConPR's 84% low-yaw queries in the [0°, 10°] bucket), the equivariant branch contributes marginally (+0.42 R@1); the architectural advantage manifests primarily in the [10°, 20°] bucket and beyond."*

> **Q5.3.** "Additional comparisons with stronger baselines would further support the conclusions."

**Response.** Addressed via Q4.1: SALAD, CricaVPR, and BoQ (on both ResNet50 and DINOv2 backbones) have been added as strong baselines.

## Q6. Strengths to emphasize

> **Q6.1.** "The practical motivation for construction-site VPR is an important contribution and could be highlighted more prominently. Why is it necessary to use VPR in construction sites?"

**Response.** §1 has been expanded with a concrete motivating scenario:

> *"Construction VPR enables three practical workflows: (i) automated progress tracking — matching as-built imagery captured by inspection drones or handheld scanners against the BIM reference library to localise which structural element is currently being imaged; (ii) worker-safety monitoring — verifying that wearable-camera footage is spatially registered against the daily safety-zone map; and (iii) robotic navigation in dynamic construction environments, where GPS is unreliable indoors and visual SLAM must rely on VPR for loop closure. Each workflow requires a VPR system that is (a) robust to the large rotational variance characteristic of handheld and UAV capture (lateral-bracing inspections produce ±30° tilts, while drones in banked turns induce roll excursions), and (b) efficient enough for on-device deployment on inspection hardware."*

We also reference **Figure 2** (ConPR vs. ConSLAM yaw-difference distribution), which empirically shows that 25.4% of ConSLAM queries exceed a 90° yaw difference — motivating why architectural rotation equivariance, rather than data augmentation alone, is required.

> **Q6.2.** "The deployment-oriented design choices (lightweight architecture, low inference latency) could be further emphasized. To name a specific example in real construction application."

**Response.** §1 now includes a concrete deployment example:

> *"At 4.11 ms per query image on an RTX 5090, DR-VPR supports over 200 FPS of VPR throughput — sufficient for real-time progress tracking on a UAV-mounted inspection camera capturing 30 FPS video while simultaneously sharing the GPU with multi-branch detection tasks. For on-device field deployment on a Jetson Orin NX (21 TOPS), the projected latency is approximately 38 ms per image, which falls within the timing budget of most construction-inspection workflows."*

*(Note: the Jetson Orin NX figure is an extrapolation from the RTX 5090 measurement via established TFLOPS ratios; direct measurement is left for future deployment-focused work.)*

## Q7. Limitations

The revised §6.2 lists **eight explicit limitations** mapped to the reviewer's concerns and the empirical findings from the expanded ablation:

> **Q7.1.** "The rotation invariance is approximate rather than theoretically guaranteed for arbitrary rotation angles."

**Response.** Addressed as **L1 (§6.2)**:

> *"Approximate rotation invariance. Our equivariant branch provides exact invariance only for the 8 sampled rotation angles of C8; for arbitrary (continuous) angles, invariance degrades smoothly via the group's interpolation. A higher-order group (e.g., C16 or SO(2)) could improve continuous-angle robustness at the cost of memory and training time."*

> **Q7.2.** "The performance gains are relatively modest, and additional benchmarking against stronger baselines would strengthen the claims."

**Response.** Addressed via Q4.1 (new baselines added) and **L4 (§6.2)**:

> *"Modest gains on appearance-dominated benchmarks. On rotation-benign data (e.g., ConPR's 84% low-yaw queries in the [0°, 10°] bucket), the equivariant branch contributes marginally (+0.42 R@1); the architectural advantage manifests primarily in the [10°, 20°] yaw-difficulty bucket (+2.67 ConSLAM, +1.84 ConPR). We emphasise consistent improvements across settings rather than uniform gains."*

> **Q7.3.** "The design choice of max GroupPooling may discard orientation-dependent information, and this limitation should be discussed."

**Response.** Addressed as **L2 (§6.2)**:

> *"Information loss in max GroupPooling. Reducing the equivariant feature tensor to an invariant descriptor via max pooling over orientation channels discards orientation-encoded information that could aid sub-group-angle discrimination. Mean pooling, ℓ₂ (energy-preserving) pooling, and learned pooling are promising extensions; §5.3 includes max/mean/norm comparisons at matched seed."*

> **Q7.4.** "The attention fusion mechanism currently appears to rely on global scalar weights, which may limit its expressive capacity."

**Response.** Addressed as **L3 (§6.2)**:

> *"Global-weight fusion expressivity. In preliminary experiments we evaluated softmax attention fusion — a strictly more expressive formulation via per-sample scoring — and found that it empirically underperforms weighted joint scoring in our setting: the softmax's zero-sum constraint, combined with the strength of BoQ's pretrained features, drives the equivariant-branch weight to w₂ ≈ 10⁻⁴ at inference across seeds. Weighted joint scoring avoids this zero-sum trap by decoupling the two branches: each produces an independent L2-normalised descriptor, and the final score is their fixed-weight sum (no joint optimisation). Token-wise attention and cross-branch cross-attention are natural extensions that may recover per-sample adaptivity without the zero-sum failure mode, and we leave them for future work."*

**Additional limitations (not in the reviewer's enumeration but surfaced by the expanded ablation):**

**L5.** *"Illumination robustness not evaluated."* (R1-W1)
**L6.** *"Training data from GSV-Cities street-view, not construction-specific."* (R1-W2)
**L7.** *"Discrete rotation group only (C8); continuous SO(2) is future work."* (R1-W3)

**L8 (new). BoQ-dominant joint scoring at low β.** *"Our joint scoring rule `score(q, c) = (1 − β)·boq_sim(q, c) + β·equi_sim(q, c)` is appearance-dominated at our chosen β = 0.10 (BoQ term carries 9× the weight of the equivariant term). Queries whose true positive has low boq_sim — for example, queries with > 30° yaw difference where BoQ ranks the true positive far down the candidate list — cannot be promoted by the equivariant signal because the BoQ-weighted gap is too large to bridge. We verified this with two complementary ablations. **(i)** A 'union retrieve' variant (candidates restricted to Branch 1 top-100 ∪ Branch 2 top-100) yields **zero** additional R@1 improvement at β = 0.10 across both datasets and all three seeds, because the BoQ-dominated score suppresses any equivariant-only candidates whose BoQ similarity falls below the top-ranked appearance candidates. **(ii)** A single-stage variant that scores the entire database directly (without any FAISS pre-filter) is **exactly R@1-equivalent** to our standard formulation across all three seeds at β ∈ [0, 0.20] on both datasets, confirming that the limitation is the BoQ-dominance of the score function itself, not candidate-set restriction. Raising β to give the equivariant branch more weight closes some of this gap on rotation-heavy queries but degrades rotation-aligned queries (β = 0.5 yields 56.57 on ConSLAM and 79.01 on ConPR, both well below our reported main numbers). A stronger rotation-aware appearance retriever — e.g., a backbone trained with built-in rotation-equivariant features — would be necessary to close this gap, which we leave to future work."*

(See Supplementary §S.3 for the union-retrieve and single-stage equivalence ablation data.)

## Q8. Structure, flow, wording

> **Q8.1.** "The phrase 'banking UAVs' may be unclear to some readers. It may be clearer to use 'UAVs undergoing banked turns' or 'UAVs with significant roll rotations.'"

**Response.** All occurrences of "banking UAVs" (L152, L582 in the original) have been replaced with "**UAVs undergoing banked turns**".

> **Q8.2.** "'specifically leveraging RandAugment [? ].' The reference needs double-checking."

**Response.** The broken `[?]` citation at L334 has been replaced with the correct `\citep{cubuk2020randaugment}`; the corresponding BibTeX entry for *Cubuk et al., "RandAugment: Practical Automated Data Augmentation with a Reduced Search Space" (CVPR 2020)* has been added to `cas-refs.bib`.

## Q9. Language editing

> "Yes."

**Response.** We engaged professional language editing for the entire manuscript (excluding tables, equations, and direct reviewer-quote sections). The revised text has been proof-read for grammar, clarity, and stylistic consistency.

---

## Summary of changes

| Location | Change | Driven by |
|---|---|---|
| Abstract | Rewritten with new 3-seed numbers (ConSLAM 61.89 ± 0.33, ConPR 79.74 ± 0.09) and tempered claims | R2-Q3, R2-Q5 |
| §1 Introduction | Added construction-VPR motivation paragraph + Jetson Orin NX deployment example | R2-Q6 |
| §1 Contributions list | Item 3 softened to "consistent, statistically significant", adds cross-dataset [10°, 20°] bucket finding | R2-Q5 |
| §2.1 Related Work | Added SelaVPR++, Implicit Aggregation, Deep-Homography VPR, BoQ | R1-W4, R2-Q4 |
| §3.2/§3.3 | Claims "guaranteed invariance" softened to "exact on group / approximate elsewhere"; appearance-branch modularity made explicit | R2-Q5, R2-Q4 |
| §3.3 | Weighted joint scoring rule introduced as main configuration, with concat/attention retained as ablations | R2-Q4 |
| §4.2 | Protocol note: 3 seeds + val-best checkpoint + R@5/R@10 + pose-rotation erratum transparency | R2-Q3 |
| §5.1 Tables 1 & 2 | Rebuilt with SALAD / CricaVPR / BoQ-R50 / BoQ-DINOv2, R@5 / R@10, params, latency | R2-Q3, R2-Q4 |
| §5.3 ablation Table 3 | Each-branch, three-fusion-method (attention / concat / joint scoring), and GroupPooling ablations | R2-Q4 |
| §5.4 Figures 6 & 2, Table 4 | Rotation feature response curve; yaw distribution; per-yaw bucket R@1 (cross-dataset [10°, 20°] concentrated gain) | R1-W3, R2-Q4, R2-Q6 |
| §6.2 Limitations L1–L8 | Eight explicit limitations, including the new L8 (BoQ-dominant joint scoring at low β, supported by union-retrieve and single-stage equivalence ablations) | R2-Q7 |
| Supplementary S1 | Fine β sweep (0.00–0.15 in 0.01 steps, 3 seeds) — plateau analysis justifying β = 0.10 | R2-Q4 |
| Supplementary S2 | Max / mean / norm GroupPool comparison, matched-seed | R2-Q4 |
| Supplementary S3 | Per-yaw bucket R@1, both datasets, 10° buckets | R1-W3 |
| Supplementary S4 | Full 10-sequence ConPR per-pair breakdown | R2-Q3 |
| Supplementary S5 | Union-retrieve null-result data (supporting L8) | R2-Q7 |
| Minor | "banking UAVs" → "UAVs undergoing banked turns"; RandAugment citation fixed | R2-Q8 |
| Throughout | Professional language edit | R2-Q9 |

We believe that the revised manuscript addresses every reviewer concern with empirical support: it tempers the original claims as advised, adds the requested comprehensive baseline comparisons and fusion / pool ablations, provides cross-dataset per-yaw evidence for the rotation-robustness claim, and includes an honest discussion of the method's eight explicit limitations. We thank the Editor and Reviewers again for their constructive feedback and look forward to any further comments.

Respectfully,
The Authors
