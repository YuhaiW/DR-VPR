#!/bin/bash
# Train Equi-BoQ (attention fusion) across 3 seeds, then eval each on full ConPR + ConSLAM.
# Attention-fusion runs are tagged `_attention_seed<N>` so they do NOT overwrite the
# prior concat-fusion runs in LOGS/resnet50_DualBranch_seed<N>/.

set -e
cd /home/yuhai/project/DR-VPR

SEEDS=(1 42 190223)
TAG="attention_b1"   # bias=(+2.0, 0.0) — softer softmax init (B1 fix)

echo "=========================================="
echo "Equi-BoQ (${TAG} fusion) multi-seed run: ${SEEDS[*]}"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Training Equi-BoQ (${TAG}) with seed=$SEED ..."
    RUN_TAG=${TAG} mamba run -n drvpr python train_fusion.py --seed $SEED \
        2>&1 | tee train_equiboq_${TAG}_s${SEED}.log
    echo ">>> Seed $SEED training done."
done

echo ""
echo "=========================================="
echo "Evaluating all seeds on ConPR + ConSLAM"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    CKPT_DIR="LOGS/resnet50_DualBranch_${TAG}_seed${SEED}/lightning_logs"
    BEST_CKPT=$(find "$CKPT_DIR" -name "*.ckpt" 2>/dev/null | sort -t'[' -k2 -r | head -1)
    if [ -z "$BEST_CKPT" ]; then
        echo "!!! Seed $SEED: no checkpoint found in $CKPT_DIR"
        continue
    fi
    echo ""
    echo ">>> Seed $SEED best checkpoint: $BEST_CKPT"

    echo ">>> Eval ConPR ..."
    DRVPR_CKPT="$BEST_CKPT" mamba run -n drvpr python test_conpr.py \
        2>&1 | tee eval_equiboq_${TAG}_s${SEED}_conpr.log | grep -E "平均 Recall|Average Recall" | head -4

    echo ">>> Eval ConSLAM ..."
    DRVPR_CKPT="$BEST_CKPT" mamba run -n drvpr python test_conslam.py \
        2>&1 | tee eval_equiboq_${TAG}_s${SEED}_conslam.log | grep -E "Average Recall" | head -3
done

echo ""
echo "=========================================="
echo "All done. Summary:"
echo "=========================================="
for SEED in "${SEEDS[@]}"; do
    echo "--- Seed $SEED ---"
    grep "Average Recall@1:" eval_equiboq_${TAG}_s${SEED}_conpr.log 2>/dev/null | head -1
    grep "Average Recall@1:" eval_equiboq_${TAG}_s${SEED}_conslam.log 2>/dev/null | head -1
done
