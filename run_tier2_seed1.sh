#!/bin/bash
# Tier-2 Fourier invariant + three-loss training. SINGLE seed as a decision gate
# before committing to 3-seed overnight run (see .claude/plans/effervescent-meandering-dusk.md).
#
# Key settings:
#   GROUP_POOL_MODE=fourier  — 5 irrep magnitudes per field, 64 → 320 channels
#   FUSION_METHOD=concat     — keep original paper's concat+gate structure
#   FREEZE_BOQ=0             — BoQ continues to fine-tune at 0.05× LR
#   LAMBDA_EQUI=0.5          — weight on MS(desc_equi) auxiliary loss
#   LAMBDA_ROT=0.3           — weight on cosine rotation-consistency loss

set -e
cd /home/yuhai/project/DR-VPR

SEEDS=(1)                    # single seed decision gate; extend to (42 190223) after pass
TAG="tier2"

echo "=========================================="
echo "Tier-2 training (decision-gate): seeds ${SEEDS[*]}"
echo "  GROUP_POOL_MODE = fourier (320 invariant channels)"
echo "  FUSION_METHOD   = concat"
echo "  FREEZE_BOQ      = 0 (BoQ fine-tunes at 0.05x LR)"
echo "  LAMBDA_EQUI     = 0.5"
echo "  LAMBDA_ROT      = 0.3"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Training seed=$SEED ..."
    GROUP_POOL_MODE=fourier \
    FUSION_METHOD=concat \
    FREEZE_BOQ=0 \
    LAMBDA_EQUI=0.5 \
    LAMBDA_ROT=0.3 \
    RUN_TAG=${TAG} \
        mamba run -n drvpr python train_fusion.py --seed $SEED \
        2>&1 | tee train_${TAG}_s${SEED}.log
    echo ">>> Seed $SEED training done."
done

echo ""
echo "=========================================="
echo "Training complete. Next step: β sweep eval via eval_rerank.py"
echo "  GROUP_POOL_MODE=fourier \\"
echo "  DRVPR_CKPT=LOGS/resnet50_DualBranch_${TAG}_seed1/lightning_logs/version_0/checkpoints/ \\"
echo "    mamba run -n drvpr python eval_rerank.py 2>&1 | tee eval_${TAG}_s1.log"
echo "=========================================="
