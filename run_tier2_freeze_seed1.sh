#!/bin/bash
# Tier-2 Fourier invariant + three-loss training WITH BoQ FROZEN.
#
# Rationale (from seed=1 unfreeze run eval): fine-tuning BoQ (even at 0.05x LR)
# hurt ConSLAM performance — single-stage R@1 went from 62.21% (BoQ standalone)
# down to 53-58% across epochs 0-4. The equi branch DID learn rerank signal
# (+3.9 R@1 at best β), but couldn't overcome the BoQ degradation.
#
# Fix: freeze BoQ entirely → stage-1 retrieve retains 62.21% baseline.
# Three-loss supervision on desc_equi continues (it bypasses BoQ gradients).
# Expected: 62.21% + ~4% rerank ≈ 66% R@1 on ConSLAM (actual improvement).

set -e
cd /home/yuhai/project/DR-VPR

SEEDS=(1)
TAG="tier2_freeze"

echo "=========================================="
echo "Tier-2 + FREEZE_BOQ training: seeds ${SEEDS[*]}"
echo "  GROUP_POOL_MODE = fourier (320 invariant channels)"
echo "  FUSION_METHOD   = concat"
echo "  FREEZE_BOQ      = 1  <<< key change from run_tier2_seed1.sh"
echo "  LAMBDA_EQUI     = 0.5"
echo "  LAMBDA_ROT      = 0.3"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Training seed=$SEED ..."
    GROUP_POOL_MODE=fourier \
    FUSION_METHOD=concat \
    FREEZE_BOQ=1 \
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
