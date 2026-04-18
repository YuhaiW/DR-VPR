# Stale θ=0° baseline logs (pre-2026-04-16)

These files contain baseline VPR evaluation results measured with
`eval_baselines.py` **before commit db01de5** (2026-04-16 23:36), which fixed
`theta_degrees=0.0 → 15.0` for ConSLAM query trajectory.

**Why θ=15° matters**: ConSLAM query trajectory is slightly rotated relative to
the database reference frame. Before the fix, queries were matched with θ=0°
(no rotation compensation), producing unfairly low baseline R@1 values that
favor our rotation-robust DR-VPR.

**Do NOT cite numbers from these files in the revision paper.**

The correct θ=15° baseline measurements are in:
- `eval_baselines_theta15_all.log` (batch run 2026-04-17)
- `baseline_results.txt` (summary table)

File inventory:
- `baseline_eval.log` (2026-04-13)
- `baseline_eval_full.log` (2026-04-14): SALAD, CricaVPR, BoQ-DINOv2, CosPlace
- `baseline_eval_mixvpr_dinov2.log` (2026-04-14): MixVPR, DINOv2
- `baseline_conpr_diag.log` (2026-04-14): diagnostic per-sequence ConPR
- `baseline_conpr_mixvpr.log` (2026-04-14): MixVPR ConPR detail
- `baseline_conpr_dinov2.log` (2026-04-14): DINOv2 ConPR detail

Archived 2026-04-17 as part of baseline re-measurement for AUTCON revision.
