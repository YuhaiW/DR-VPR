# AUTCON-D-25-06069 — Major Revision Decision Letter

**From:** Automation in Construction (Editorial Manager)
**To:** Gilbert Ye (y.ye@northeastern.edu)
**Date:** 2026-04-04
**Manuscript:** DR-VPR: Dual-Branch Rotation-Robust Visual Place Recognition for Dynamic Construction Environments
**Decision editor:** Wen-der Yu, Ph.D., Managing Editor
**Verdict:** Major Revision
**Deadline:** 2026-04-25

---

## Editor's Note

> The reviewers recommend reconsideration of your manuscript following major revision. I invite you to resubmit your manuscript after addressing the comments below. Please resubmit your revised manuscript by Apr 25, 2026.
>
> When revising your manuscript, please consider all issues mentioned in the reviewers' comments carefully: please outline every change made in response to their comments and provide suitable rebuttals for any comments not addressed. Please note that your revised submission may need to be re-reviewed.

---

## Reviewer #1 — "can be accepted after revision"

### Summary
> This paper introduces DR-VPR, a dual-branch visual place recognition architecture designed for dynamic construction environments. The method combines a discriminative CNN branch for appearance features with a rotation-equivariant E2ResNet branch for geometric stability, fused via an attention mechanism. Experiments on construction-specific datasets (ConPR and ConSLAM) demonstrate improved robustness to in-plane rotations and structural changes, achieving state-of-the-art Recall@1 with real-time inference (~4.23 ms). The authors argue that architectural equivariance outperforms data augmentation in handling rotation without sacrificing discriminability.

### Strengths
1. The dual-branch design effectively decouples semantic discrimination from geometric invariance, offering a principled solution to the rotation-invariance vs. discriminability trade-off common in standard VPR models.
2. The method is rigorously evaluated on challenging construction-specific datasets (ConPR, ConSLAM), demonstrating clear improvements over strong baselines in both rotation robustness and structural change resilience.
3. With an inference time of ~4.23 ms and a compact model size (~14.4M parameters), the approach is well-suited for real-time deployment on resource-constrained devices such as UAVs and handheld scanners.
4. The adaptive fusion mechanism dynamically balances between branches, enhancing flexibility and robustness without significant computational overhead.

### Weaknesses
1. While the model handles in-plane rotation well, it has not been explicitly evaluated to address illumination variations (such as day-night changes in the Tokyo24/7 dataset), which also seem common in construction imagery.
2. The model is trained on street-view data (GSV-Cities) rather than construction-specific imagery, which may limit its ability to generalize to the full lifecycle and structural evolution of construction sites.
3. The study only explores a few discrete rotation groups. More analyses of continuous rotation handling could strengthen the geometric robustness claim.
4. Some of the recent VPR works [1][2][3] should be mentioned.

### References suggested by R1
- [1] SelaVPR++: Towards seamless adaptation of foundation models for efficient place recognition. T-PAMI, 2025.
- [2] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era. NeurIPS 2025.
- [3] Deep homography estimation for visual place recognition. AAAI 2024.

### R1 verdict
> Overall, the paper presents a novel and timely contribution to VPR in construction, with clear strengths in architecture design and real-time performance. I think this paper can be accepted after revision.

---

## Reviewer #2 — constructive major revision

### Q1: Objectives & rationale
> The objectives and motivation of the study are generally clear. The manuscript aims to address visual place recognition (VPR) challenges in dynamic construction environments, particularly severe in-plane rotations and temporal appearance changes caused by handheld or UAV data acquisition.

### Q2: Replicability/reproducibility
> Yes.

### Q3: Statistical analysis
> The performance gains reported (1.6%–3.5% Recall@1) are relatively modest. The authors should provide statistical significance analysis, such as repeated runs, standard deviations, or confidence intervals.
>
> It would be helpful to clarify whether results are based on single training runs or averaged over multiple runs.
>
> If possible, additional evaluation metrics such as Recall@5 or Recall@10 could be reported to provide a more complete assessment of retrieval performance.
>
> A statistician review does not appear strictly necessary, but clearer statistical reporting would strengthen the experimental rigor.

### Q4: Tables and figures to add
> Add a table comparing DR-VPR with additional state-of-the-art VPR methods, including SALAD, CriCA, and BoQ. Both retrieval accuracy and computational cost should be compared.
>
> Include an ablation table analyzing architectural components, such as: effect of the each branch; effect of GroupPooling; impact of the attention fusion module.
>
> Consider adding a visualization figure showing feature responses under image rotations, which could better illustrate the benefit of the equivariant branch.
>
> A table summarizing model size, parameter count, and runtime compared to baseline methods would further highlight the efficiency claims.
>
> SALAD: https://github.com/serizba/salad
> CriCA: https://github.com/Lu-Feng/CricaVPR
> BoQ:   https://github.com/amaralibey/Bag-of-Queries

### Q5: Conclusions/tone
> Overall, the conclusions are generally supported by the experimental results. However, some claims appear slightly overstated and should be moderated.
>
> The manuscript states "guaranteed rotation invariance to arbitrary angles." With a discrete rotation group such as C8, invariance is guaranteed only for the sampled group elements. Arbitrary rotations can only be handled approximately.
>
> The authors should acknowledge that the observed performance improvements are modest, and therefore position the method as providing consistent improvements rather than large performance gains.
>
> Additional comparisons with stronger baselines would further support the conclusions.

### Q6: Strengths to emphasize more
> The practical motivation for construction-site VPR is an important contribution and could be highlighted more prominently. Why is it necessary to use VPR in construction sites?
>
> The deployment-oriented design choices (lightweight architecture, low inference latency) could be further emphasized. To name a specific example in real construction application.

### Q7: Limitations to acknowledge
> The manuscript currently does not sufficiently discuss its limitations. The following limitations should be acknowledged:
>
> - The rotation invariance is approximate rather than theoretically guaranteed for arbitrary rotation angles.
> - The performance gains are relatively modest, and additional benchmarking against stronger baselines would strengthen the claims.
> - The design choice of max GroupPooling may discard orientation-dependent information, and this limitation should be discussed.
> - The attention fusion mechanism currently appears to rely on global scalar weights, which may limit its expressive capacity.

### Q8: Structure, flow, and wording
> The overall structure is generally clear, minor language issues should be corrected.
>
> The phrase "banking UAVs" may be unclear to some readers. It may be clearer to use "UAVs undergoing banked turns" or "UAVs with significant roll rotations."
>
> "specifically leveraging RandAugment [? ]." The reference needs double-checking.

### Q9: Language editing
> Yes.

### R2 additional suggestions field
> (left empty)
