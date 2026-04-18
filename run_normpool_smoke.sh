#!/bin/bash
# B2 NormPool smoke test (Step 13 of plan).
# Single seed × 1-2 epochs to decide whether to commit to 3-seed full run.
#
# Decision gate: epoch-0 ConSLAM val R@1 vs sanity baseline (max pool)
#   ≥ baseline + 0.5  → continue to 3-seed full
#   <  baseline       → abort, NormPool not useful
#
# (val_set_names=['conpr','conslam'] hardcoded in train_fusion.py L508,
#  ckpt monitor = 'conslam/R1' L608)

set -e
cd /home/yuhai/project/DR-VPR

GROUP_POOL_MODE=norm \
FUSION_METHOD=concat \
FREEZE_BOQ=1 \
RUN_TAG=normpool_smoke \
    mamba run -n drvpr python train_fusion.py --seed 1 \
    2>&1 | tee train_normpool_smoke_s1.log
