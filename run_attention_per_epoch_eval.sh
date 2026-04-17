#!/bin/bash
# Per-epoch eval for all attention-fusion checkpoints (B2 and B1) on ConPR + ConSLAM.
# Reads 30 ckpts per variant × 2 variants × 2 test scripts = 120 eval runs.
# All 60 checkpoints (3 seeds × 10 epochs × 2 variants) are already on disk; this script does NOT train.

set -e
cd /home/yuhai/project/DR-VPR

SEEDS=(1 42 190223)

# Map tag -> run directory prefix
declare -A DIR_PREFIX=(
    [attention]="resnet50_DualBranch_attention"
    [attention_b1]="resnet50_DualBranch_attention_b1"
)

TAGS=(attention attention_b1)

echo "=============================================="
echo "[$(date +%H:%M)] Per-epoch eval for attention variants"
echo "  tags:  ${TAGS[*]}"
echo "  seeds: ${SEEDS[*]}"
echo "=============================================="

for TAG in "${TAGS[@]}"; do
    PREFIX=${DIR_PREFIX[$TAG]}
    echo ""
    echo "=== Variant: ${TAG} ==="
    for SEED in "${SEEDS[@]}"; do
        CKPT_DIR="LOGS/${PREFIX}_seed${SEED}/lightning_logs/version_0/checkpoints"
        if [ ! -d "$CKPT_DIR" ]; then
            echo "!!! ${TAG} seed ${SEED}: no checkpoint dir at ${CKPT_DIR}"
            continue
        fi
        for CKPT in "$CKPT_DIR"/*.ckpt; do
            [ -f "$CKPT" ] || continue
            BN=$(basename "$CKPT")
            EPOCH=$(echo "$BN" | grep -oP 'epoch\(\K[0-9]+')
            if [ -z "$EPOCH" ]; then
                echo "  skip (bad filename): $BN"
                continue
            fi

            CP_LOG="eval_${TAG}_s${SEED}_ep${EPOCH}_conpr.log"
            CS_LOG="eval_${TAG}_s${SEED}_ep${EPOCH}_conslam.log"

            echo "[$(date +%H:%M)] ${TAG} seed=${SEED} epoch=${EPOCH}"

            DRVPR_CKPT="$CKPT" mamba run -n drvpr python test_conpr.py \
                > "$CP_LOG" 2>&1 || echo "  !! conpr eval failed"
            DRVPR_CKPT="$CKPT" mamba run -n drvpr python test_conslam.py \
                > "$CS_LOG" 2>&1 || echo "  !! conslam eval failed"

            CP_R1=$(grep "平均 Recall@1:" "$CP_LOG" | head -1 | awk '{print $NF}')
            CS_R1=$(grep "Average Recall@1:" "$CS_LOG" | head -1 | awk '{print $NF}')
            echo "  → ConPR R@1=${CP_R1}  ConSLAM R@1=${CS_R1}"
        done
    done
done

echo ""
echo "=============================================="
echo "[$(date +%H:%M)] All per-epoch evals done."
echo "=============================================="
