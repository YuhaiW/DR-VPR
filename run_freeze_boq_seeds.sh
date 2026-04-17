#!/bin/bash
# Train Equi-BoQ with Branch 1 (BoQ) FROZEN. Only equi branch + gate train.
# Then eval all 3 seeds on ConSLAM (theta=15, yaw=80) using two-stage rerank.

set -e
cd /home/yuhai/project/DR-VPR

SEEDS=(1 42 190223)
TAG="freeze_boq"

echo "=========================================="
echo "Freeze-BoQ training: seeds ${SEEDS[*]}"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Training seed=$SEED (BoQ frozen, equi trains) ..."
    FREEZE_BOQ=1 RUN_TAG=${TAG} FUSION_METHOD=concat GROUP_POOL_MODE=max \
        mamba run -n drvpr python train_fusion.py --seed $SEED \
        2>&1 | tee train_${TAG}_s${SEED}.log
    echo ">>> Seed $SEED training done."
done

echo ""
echo "=========================================="
echo "Training complete. Run eval_rerank.py next."
echo "=========================================="
