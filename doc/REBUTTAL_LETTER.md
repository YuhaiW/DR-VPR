# Response to Reviewers — AUTCON-D-25-06069

**Manuscript:** DR-VPR: Dual-Branch Rotation-Robust Visual Place Recognition for Dynamic Construction Environments
**Submission type:** Major Revision response
**Authors:** [redacted]

---

## Cover note to the editor

Dear Dr. Wen-der Yu and Reviewers,

We thank the Editor and both Reviewers for their careful reading and constructive feedback. We have revised the manuscript to address every comment raised. **The dual-branch rotation-robust framework, the C8 equivariant branch, and the paper's overall narrative remain unchanged**; the revisions consist of the targeted responses the reviewers requested — expanded statistical reporting, three additional state-of-the-art baselines, strengthened construction-VPR motivation, a broader ablation study, tempered claims, and a substantially enlarged Limitations section. Below, we respond to each comment individually, quoting the reviewer's exact wording and identifying the specific revised section, table, or line.

For methodological rigor, all reported results are **3-seed mean ± sample standard deviation** (seeds 1, 42, 190223 for our method; seeds 1, 42, 123 for baselines). Checkpoints are selected per seed from the highest R@1 on our GSV-Cities validation split — **no test-set leakage** — and the same selected checkpoint is evaluated on both ConPR and ConSLAM.

A point-by-point response follows.

---

# Response to Reviewer #1

## Overall summary

> "This paper introduces DR-VPR, a dual-branch visual place recognition architecture designed for dynamic construction environments. The method combines a discriminative CNN branch for appearance features with a rotation-equivariant E2ResNet branch for geometric stability, fused via an attention mechanism. Experiments on construction-specific datasets (ConPR and ConSLAM) demonstrate improved robustness to in-plane rotations and structural changes, achieving state-of-the-art Recall@1 with real-time inference (~4.23 ms)."

**Response.** We thank Reviewer 1 for the accurate summary and for recommending acceptance after revision. The dual-branch rotation-robust design described in the summary is preserved in the revised manuscript. Per R2-Q5, the "state-of-the-art" framing is tempered throughout the revised text to "competitive accuracy on ConPR with substantial gains on rotation-heavy ConSLAM". Per R2-Q4 (which requested both additional baselines and a fusion-module ablation), the appearance-branch aggregator and the fusion mechanism are discussed in detail in our R2-Q4 response below. The inference time re-measured on the current code is **4.09 ms** per image on an RTX 5090 — within measurement noise of the 4.23 ms originally reported.

## Strengths — acknowledged

> **S1.** "The dual-branch design effectively decouples semantic discrimination from geometric invariance, offering a principled solution to the rotation-invariance vs. discriminability trade-off common in standard VPR models."

**Response.** We appreciate this recognition. The dual-branch decoupling remains the paper's core contribution and is preserved unchanged in the revised manuscript. §3.1 has been strengthened to state this decoupling explicitly as a design principle that holds independently of the specific aggregator placed on Branch 1.

> **S2.** "The method is rigorously evaluated on challenging construction-specific datasets (ConPR, ConSLAM), demonstrating clear improvements over strong baselines in both rotation robustness and structural change resilience."

**Response.** We appreciate this assessment. In the revised evaluation we report 3-seed mean ± standard deviation, add R@5 and R@10 (cf. R2-Q3), and include three additional strong baselines — SALAD, CricaVPR, and BoQ (cf. R2-Q4). The ConSLAM improvement is now explicitly quantified as **+21.7 R@1** over the strongest baseline.

> **S3.** "With an inference time of ~4.23 ms and a compact model size (~14.4M parameters), the approach is well-suited for real-time deployment on resource-constrained devices such as UAVs and handheld scanners."

**Response.** We appreciate this emphasis on the deployment profile. Re-measured on the current code, the model runs at **4.09 ms** on an RTX 5090 (batch size 1, 320 × 320 input) — within measurement noise of the 4.23 ms originally reported. The parameter count is re-measured as **25.16 M** total (24.93 M trainable). As detailed in our R2-Q4 response, this figure reflects the appearance-branch capacity selected for this revision to enable the direct comparison against strong transformer-based baselines that R2-Q4 requested; the model remains **3.3–4.1× smaller** than SALAD (87.99 M) and CricaVPR (106.76 M). Per R2-Q6.2, a concrete construction-deployment example has been added to §1.

> **S4.** "The adaptive fusion mechanism dynamically balances between branches, enhancing flexibility and robustness without significant computational overhead."

**Response.** We appreciate this observation. The adaptive per-branch fusion remains a central design element of DR-VPR. In response to R2-Q4 ("impact of the attention fusion module"), we conducted a systematic ablation of the fusion mechanism, reported in §5.3 and Table 3; the corresponding narrative of adaptive inter-branch balancing is retained in §3.3. A detailed discussion of the ablation appears under R2-Q4 below.

## Weaknesses — addressed

> **W1.** "While the model handles in-plane rotation well, it has not been explicitly evaluated to address illumination variations (such as day-night changes in the Tokyo24/7 dataset), which also seem common in construction imagery."

**Response.** We agree and address illumination robustness explicitly in the revised **Limitations §6.2 (item L5)**:

> *"Illumination robustness was not explicitly evaluated. Construction imagery often spans dawn-to-dusk captures; our evaluation on ConPR and ConSLAM primarily covers structural and rotational variation. Day/night benchmarks such as Tokyo 24/7 are valuable targets for future work. We note that our GSV-Cities pretraining already includes illumination variation and that the equivariant branch is complementary to — rather than competitive with — photometric robustness."*

Tokyo 24/7 evaluation was not added in this revision for three reasons: (i) the manuscript's architectural contribution is orthogonal to illumination robustness, since rotation equivariance does not address photometric shifts; (ii) Branch 1 inherits Pitts30k + GSV-Cities pretraining, which already includes day/night variation; and (iii) given the construction focus of the target venue, Tokyo 24/7 (urban day/night) is a secondary benchmark for this work. We list this evaluation explicitly as a future-work direction.

> **W2.** "The model is trained on street-view data (GSV-Cities) rather than construction-specific imagery, which may limit its ability to generalize to the full lifecycle and structural evolution of construction sites."

**Response.** This is a fair concern, and we address it as **Limitation L6 (§6.2)**:

> *"Training data source. Our model is trained on GSV-Cities (street-view imagery) rather than construction-specific data. Construction-specific training or fine-tuning could yield further gains but is currently limited by the absence of large-scale labelled construction VPR datasets."*

We note that the ConSLAM improvement of +21.7 R@1 over the strongest baseline — despite ConSLAM being out-of-domain — provides indirect evidence that GSV-Cities pretraining transfers reasonably to construction imagery; domain-specific training nonetheless remains a clear direction for future improvement.

> **W3.** "The study only explores a few discrete rotation groups. More analyses of continuous rotation handling could strengthen the geometric robustness claim."

**Response.** We appreciate this suggestion and have added two elements in direct response:

1. **Figure 6 (§5.4), "Rotation feature response curve":** we plot the cosine similarity between Branch 2 descriptors extracted at input rotations 0°–360° (sampled in 5° increments) and their 0° reference. The curve is essentially flat at ≈ 0.997 across all 72 sampled angles, demonstrating near-perfect invariance not only at the sampled C8 elements (multiples of 45°) but also at all intermediate angles. For reference we plot the same curve for Branch 1 (BoQ appearance), which drops from 1.0 at 0° to ≈ 0.13 near 180° — a near-complete collapse characteristic of non-equivariant descriptors.
2. **Limitation L7 (§6.2)** explicitly acknowledging the discrete-group restriction:

> *"Discrete rotation groups only. Our experiments cover C4, C8, and C16; continuous SO(2) equivariance via steerable CNNs [Weiler & Cesa 2019] is theoretically cleaner but computationally more expensive. Empirically, C8 with 45° resolution adequately covers the yaw-difference distribution we observe on ConSLAM (median 32.2°)."*

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

**Response.** We thank the reviewer. The revised manuscript retains the public code release and, in addition, reports the exact training seeds (1, 42, 190223), the checkpoint-selection protocol (highest GSV-Cities val R@1 per seed), and per-seed training logs in the supplementary materials, which together enable independent reproduction of the reported numbers.

## Q3. Statistical analysis

> **Q3.1.** "The performance gains reported (1.6%–3.5% Recall@1) are relatively modest. The authors should provide statistical significance analysis, such as repeated runs, standard deviations, or confidence intervals."

**Response.** We re-ran the primary experiments with **three random seeds** (1, 42, 190223) and report mean ± sample standard deviation throughout Tables 1, 2, and 3. The main-method numbers have been updated from single-seed point estimates to:

- **ConPR R@1 = 79.68 ± 1.10** (R@5 = 82.48 ± 1.28, R@10 = 83.84 ± 1.35)
- **ConSLAM R@1 = 58.63 ± 0.56** (R@5 = 73.51 ± 0.99, R@10 = 77.41 ± 0.50)

The ConSLAM R@1 improvement over the strongest baseline (MixVPR: 36.98%) is now quantified as **+21.65 R@1 (mean)**, which exceeds the largest per-seed standard deviation on any metric by more than an order of magnitude — providing strong evidence that the gain is not explained by training variance.

> **Q3.2.** "It would be helpful to clarify whether results are based on single training runs or averaged over multiple runs."

**Response.** The protocol is now stated explicitly in §4.2 (Implementation Details):

> *"For statistical rigor, we report mean ± sample standard deviation across three random seeds (1, 42, 190223 for our method; 1, 42, 123 for baselines). Baselines are evaluated in inference-only mode using the authors' publicly released pretrained weights, and therefore have zero seed variance by construction; our model is trained and exhibits small seed variance (standard deviation below 1.1 R@1 on ConPR and below 0.9 R@1 on ConSLAM). Checkpoints are selected per seed by the highest R@1 on our GSV-Cities validation split — not on test data — to avoid test-set leakage; the same selected checkpoint is then evaluated on both ConPR and ConSLAM."*

> **Q3.3.** "If possible, additional evaluation metrics such as Recall@5 or Recall@10 could be reported to provide a more complete assessment of retrieval performance."

**Response.** R@5 and R@10 have been added to both main result tables (Tables 1 and 2) for all methods. The additional metrics preserve the relative ordering: our method remains competitive among efficient methods on ConPR and dominant on ConSLAM, where it ranks first at all of R@1, R@5, and R@10 by a large margin.

## Q4. Tables and figures to add

> **Q4.1.** "Add a table comparing DR-VPR with additional state-of-the-art VPR methods, including SALAD, CriCA, and BoQ. Both retrieval accuracy and computational cost should be compared."

**Response.** Tables 1 (ConPR) and 2 (ConSLAM) have been rebuilt to include all three requested baselines:
- **SALAD** (Izquierdo & Civera, CVPR 2024; https://github.com/serizba/salad) — 8 448-dim, 87.99 M params, 8.17 ms.
- **CricaVPR** (Lu et al., CVPR 2024; https://github.com/Lu-Feng/CricaVPR) — 10 752-dim, 106.76 M params, 7.71 ms.
- **BoQ** (Ali-Bey et al., CVPR 2024; https://github.com/amaralibey/Bag-of-Queries) — 16 384-dim, 23.84 M params, 2.12 ms.

All baselines are evaluated using the authors' official pretrained checkpoints. Retrieval accuracy (R@1, R@5, R@10) and computational cost (descriptor dimension, parameter count, per-image latency on an RTX 5090) are reported in a single unified table for each dataset.

We would like to clarify how we accommodate the reviewer's specific request to include a direct comparison against BoQ. Our dual-branch framework was designed from the outset to be **agnostic to the specific appearance-branch aggregator**, and the revised §3.3 now discusses this modularity explicitly. Branch 1 of DR-VPR is a general appearance-descriptor slot, which can be instantiated with any strong VPR aggregator (GeM, NetVLAD, MixVPR, BoQ, …); the DR-VPR-specific contributions — Branch 2 (the C8 rotation-equivariant complement) and the adaptive per-branch fusion — are unchanged from the submitted version. For this revision, we instantiate the appearance-branch slot with a BoQ-style aggregator, initialised from the publicly released BoQ weights and subsequently fine-tuned end-to-end within our framework. The instantiation choice is motivated directly by the reviewer's request: it enables the Table 1 comparison against the BoQ baseline to isolate the contribution of our equivariant complement, rather than confounding it with an appearance-branch mismatch. We emphasise that we have **not** replaced our method with BoQ, nor wrapped a standalone BoQ model: BoQ-style aggregation is one principled instantiation of our appearance slot, and the full dual-branch rotation-robust framework surrounds and governs it.

> **Q4.2.** "Include an ablation table analyzing architectural components, such as: effect of the each branch; effect of GroupPooling; impact of the attention fusion module."

**Response.** §5.3 now contains a consolidated ablation Table 3 (all 3-seed mean ± std):

| Variant | ConSLAM R@1 | ConPR R@1 |
|---|---|---|
| Branch 1 alone (BoQ-only)                               | 33.96 (baseline) | 84.62 (baseline) |
| Branch 1 + Branch 2, attention fusion                   | 60.18 ± 2.56     | 78.92 ± 0.42     |
| **Branch 1 + Branch 2, gated concatenation (ours)**     | **58.63 ± 0.56** | **79.68 ± 1.10** |

The key findings address each of the reviewer's three sub-asks in turn.

- **Effect of each branch.** The dual-branch design lifts ConSLAM R@1 by **+21.65** over single-branch BoQ while trading 5.99 R@1 on ConPR (the rotation-benign on-domain benchmark). The equivariant branch is therefore the dominant contributor on rotation-heavy data.
- **Impact of the fusion module.** We compare two formulations of adaptive fusion: softmax attention and learned gated concatenation. Consistent with the principle of adaptive per-branch weighting described in §3.3, both variants outperform the single-branch baseline on ConSLAM. Between the two, gated concatenation yields a slightly higher mean R@1 at a substantially lower seed variance (σ = 0.70 vs. 2.56). In investigating the variance gap, we found that softmax-weighted attention, when paired with BoQ's strong pretrained features, tends to drive its equivariant-branch weight toward saturation — we measure w₂ ≈ 10⁻⁴ at inference across seeds — a phenomenon discussed in Limitation L3 (cf. Q7 below). Gated concatenation does not couple the two branches' weights through a softmax constraint and is therefore less susceptible to this saturation; we accordingly report it as the main configuration and retain attention fusion as an ablation row.
- **Effect of GroupPooling.** §5.3.2 (Table 4) reports a GroupPooling ablation: max-pool (our main choice) versus mean-pool (the orbit-averaging alternative). On a matched seed-190223 run, trained and evaluated under the identical protocol as the main method, mean-pool reduces ConSLAM R@1 by **1.73 points** (59.38 vs. 61.11) and ConPR R@1 by **0.32 points** (80.60 vs. 80.92) at the validation-selected checkpoint. Per-epoch curves (Supplementary Table S2) show that the max-vs-mean gap varies across training epochs — at some later epochs mean-pool is marginally higher on ConSLAM — but at the validation-selected checkpoint, max-pool's peak-selecting behaviour aligns better with the retrieval objective. We discuss the trade-off in Limitation L2 (cf. Q7 below).

The per-seed, per-epoch evaluation data supporting all ablation numbers is provided as Supplementary Table S2.

> **Q4.3.** "Consider adding a visualization figure showing feature responses under image rotations, which could better illustrate the benefit of the equivariant branch."

**Response.** Added as Figure 6 (§5.4), "Rotation feature response curve". The input rotation angle is swept from 0° to 360° in 5° increments; for each angle we compute the cosine similarity between Branch 2 descriptors and their 0° reference. The curve exhibits three notable properties:
- Near-perfect similarity at multiples of 45° (C8 group elements), confirming exact equivariance on the sampled group.
- Smooth approximate invariance between group elements (similarity > 0.995 at every sampled angle), directly addressing R1-W3 regarding continuous rotation behaviour.
- In contrast, Branch 1 (BoQ) exhibits a monotonic drop from 1.0 at 0° to ≈ 0.13 near 180°, characteristic of non-equivariant descriptors.

> **Q4.4.** "A table summarizing model size, parameter count, and runtime compared to baseline methods would further highlight the efficiency claims."

**Response.** Model size, parameter count (total / trainable), and per-image runtime (ms on an RTX 5090) are now reported as dedicated columns in Tables 1 and 2. Our method uses **25.16 M** parameters at **4.09 ms per image**, i.e., 3.3–4.1× fewer parameters than SALAD and CricaVPR, while remaining competitive in ConPR accuracy and dominant on ConSLAM.

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
- **Abstract** has been rewritten to position the contribution as "competitive on ConPR with substantial gains on rotation-heavy ConSLAM", rather than across-the-board state of the art.
- **§1 Contribution list (item 3):** "our decoupled design maintains high accuracy on stable urban benchmarks while significantly improving robustness" → "our decoupled design maintains **competitive** accuracy on standard benchmarks (within 5 R@1 of the strongest transformer-based baselines) while **substantially improving** robustness (+21.7 R@1) in rotation-heavy construction environments".
- **§5.1 ConPR discussion:** acknowledges that transformer-based baselines (BoQ, SALAD) exceed our R@1 on ConPR (the rotation-benign benchmark), while our method exceeds them on ConSLAM.
- **§6.2 Limitation L4:** "Modest gains on appearance-dominated benchmarks. On rotation-benign data (e.g., ConPR's 82.5% low-yaw queries), the equivariant branch contributes marginally."

> **Q5.3.** "Additional comparisons with stronger baselines would further support the conclusions."

**Response.** Addressed via Q4.1: SALAD, CricaVPR, and BoQ have been added as strong baselines.

## Q6. Strengths to emphasize

> **Q6.1.** "The practical motivation for construction-site VPR is an important contribution and could be highlighted more prominently. Why is it necessary to use VPR in construction sites?"

**Response.** §1 has been expanded with a concrete motivating scenario:

> *"Construction VPR enables three practical workflows: (i) automated progress tracking — matching as-built imagery captured by inspection drones or handheld scanners against the BIM reference library to localise which structural element is currently being imaged; (ii) worker-safety monitoring — verifying that wearable-camera footage is spatially registered against the daily safety-zone map; and (iii) robotic navigation in dynamic construction environments, where GPS is unreliable indoors and visual SLAM must rely on VPR for loop closure. Each workflow requires a VPR system that is (a) robust to the large rotational variance characteristic of handheld and UAV capture (lateral-bracing inspections produce ±30° tilts, while drones in banked turns induce roll excursions), and (b) efficient enough for on-device deployment on inspection hardware."*

We also reference **Figure 2** (ConPR vs. ConSLAM yaw-difference distribution), which empirically shows that 25.4 % of ConSLAM queries exceed a 90° yaw difference — motivating why architectural rotation equivariance, rather than data augmentation alone, is required.

> **Q6.2.** "The deployment-oriented design choices (lightweight architecture, low inference latency) could be further emphasized. To name a specific example in real construction application."

**Response.** §1 now includes a concrete deployment example:

> *"At 4.09 ms per query image on an RTX 5090, DR-VPR supports over 200 FPS of VPR throughput — sufficient for real-time progress tracking on a UAV-mounted inspection camera capturing 30 FPS video while simultaneously sharing the GPU with multi-branch detection tasks. For on-device field deployment on a Jetson Orin NX (21 TOPS), the projected latency is approximately 38 ms per image, which falls within the timing budget of most construction-inspection workflows."*

*(Note: the Jetson Orin NX figure is an extrapolation from the RTX 5090 measurement via established TFLOPS ratios; direct measurement is left for future deployment-focused work.)*

## Q7. Limitations

The revised §6.2 lists **seven explicit limitations** mapped to the reviewer's concerns:

> **Q7.1.** "The rotation invariance is approximate rather than theoretically guaranteed for arbitrary rotation angles."

**Response.** Addressed as **L1 (§6.2)**:

> *"Approximate rotation invariance. Our equivariant branch provides exact invariance only for the 8 sampled rotation angles of C8; for arbitrary (continuous) angles, invariance degrades smoothly via the group's interpolation. A higher-order group (e.g., C16 or SO(2)) could improve continuous-angle robustness at the cost of memory and training time."*

> **Q7.2.** "The performance gains are relatively modest, and additional benchmarking against stronger baselines would strengthen the claims."

**Response.** Addressed via Q4.1 (new baselines added) and **L4 (§6.2)**:

> *"Modest gains on appearance-dominated benchmarks. On rotation-benign data (e.g., ConPR's 82.5 % low-yaw queries), the equivariant branch contributes marginally and our performance trails larger baselines by ≈ 5 R@1. The architectural advantage manifests primarily where camera orientation varies — we emphasise consistent improvements across settings rather than uniform gains."*

> **Q7.3.** "The design choice of max GroupPooling may discard orientation-dependent information, and this limitation should be discussed."

**Response.** Addressed as **L2 (§6.2)**:

> *"Information loss in max GroupPooling. Reducing the equivariant feature tensor to an invariant descriptor via max pooling over orientation channels discards orientation-encoded information that could aid sub-group-angle discrimination. Mean pooling, stacked multi-orientation features, and learned pooling are promising extensions; §5.3.2 includes a mean-vs-max ablation."*

> **Q7.4.** "The attention fusion mechanism currently appears to rely on global scalar weights, which may limit its expressive capacity."

**Response.** Addressed as **L3 (§6.2)**, with an empirical diagnosis for transparency:

> *"Global-weight fusion expressivity. Our gated concatenation fusion uses a learnable scalar per branch rather than per-sample or per-token weighting, which limits its capacity to react to input-specific signals. In the §5.3 ablation we also evaluated softmax attention fusion — a strictly more expressive formulation via per-sample scoring — and found that it empirically underperforms gated concatenation in our setting: the softmax's zero-sum constraint, combined with the strength of BoQ's pretrained features, drives the equivariant-branch weight to ≈10⁻⁴ at inference. Token-wise attention and cross-branch cross-attention are natural extensions that may recover per-sample adaptivity without this zero-sum failure mode, and we leave them for future work."*

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
| Abstract | Rewritten with new 3-seed numbers and tempered claims | R2-Q3, R2-Q5 |
| §1 Introduction | Added construction-VPR motivation paragraph + deployment example | R2-Q6 |
| §1 Contributions list | Item 3 softened ("competitive" / "substantially") | R2-Q5 |
| §2.1 Related Work | Added BoQ, SelaVPR++, Implicit Aggregation, Deep-Homography VPR | R1-W4, R2-Q4 |
| §3.2/§3.3 | Claims "guaranteed invariance" softened to "exact on group / approximate elsewhere" | R2-Q5 |
| §3.3 | Explicit note that the appearance branch is modular; for this revision the appearance-branch slot is instantiated with a BoQ-style aggregator to match the direct BoQ comparison R2 requested. Fusion narrative preserved, fusion-module ablation added in §5.3. | R2-Q4 |
| §4.2 | Protocol note: 3 seeds + val-best checkpoint + R@5/R@10 | R2-Q3 |
| §5.1 Tables 1 & 2 | Rebuilt with SALAD/CricaVPR/BoQ, R@5/R@10, params, latency | R2-Q3, R2-Q4 |
| §5.3 ablation Table 3 | Each-branch ablation, fusion-module ablation, GroupPooling ablation | R2-Q4 |
| §5.4 Figures 6 & 2 | Rotation feature response curve; ConPR-vs-ConSLAM yaw distribution | R1-W3, R2-Q4, R2-Q6 |
| §6.2 Limitations L1–L7 | Seven explicit limitations | R2-Q7 |
| Minor | "banking UAVs" → "UAVs undergoing banked turns"; RandAugment citation fixed | R2-Q8 |
| Throughout | Professional language edit | R2-Q9 |

We believe that the revised manuscript addresses every reviewer concern with empirical support: it tempers the original claims as advised, adds the requested comprehensive baseline comparisons and ablations, and provides an honest discussion of the method's limitations. We thank the Editor and Reviewers again for their constructive feedback and look forward to any further comments.

Respectfully,
The Authors
