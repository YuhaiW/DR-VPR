#!/bin/bash
# GroupPool mean-pool ablation: train concat fusion + mean pool, seed 190223.
# Then per-epoch eval on ConPR + ConSLAM. Feeds into REBUTTAL_LETTER.md R2-Q4.2 GroupPool row.

set -e
cd /home/yuhai/project/DR-VPR

SEED=190223
TAG="concat_meanpool"

echo "=========================================="
echo "[$(date +%H:%M)] Mean-pool ablation: seed=${SEED}, concat fusion, GroupPool=mean"
echo "=========================================="

RUN_TAG=${TAG} FUSION_METHOD=concat GROUP_POOL_MODE=mean mamba run -n drvpr python train_fusion.py --seed ${SEED} \
    2>&1 | tee train_${TAG}_s${SEED}.log
echo "[$(date +%H:%M)] training done."

CKPT_DIR="LOGS/resnet50_DualBranch_${TAG}_seed${SEED}/lightning_logs/version_0/checkpoints"
echo "=========================================="
echo "[$(date +%H:%M)] Per-epoch eval"
echo "=========================================="
for CKPT in "$CKPT_DIR"/*.ckpt; do
    [ -f "$CKPT" ] || continue
    BN=$(basename "$CKPT")
    EPOCH=$(echo "$BN" | grep -oP 'epoch\(\K[0-9]+')
    echo "[$(date +%H:%M)] eval ep=${EPOCH}"
    # NOTE: test_*.py must also be run with GROUP_POOL_MODE=mean so the reloaded
    # model instantiates the same mean-pool backbone.
    GROUP_POOL_MODE=mean FUSION_METHOD=concat DRVPR_CKPT="$CKPT" mamba run -n drvpr python test_conpr.py \
        > "eval_${TAG}_s${SEED}_ep${EPOCH}_conpr.log" 2>&1 || echo "  !! conpr failed"
    GROUP_POOL_MODE=mean FUSION_METHOD=concat DRVPR_CKPT="$CKPT" mamba run -n drvpr python test_conslam.py \
        > "eval_${TAG}_s${SEED}_ep${EPOCH}_conslam.log" 2>&1 || echo "  !! conslam failed"
    CP=$(grep "平均 Recall@1:" "eval_${TAG}_s${SEED}_ep${EPOCH}_conpr.log" | head -1 | awk '{print $NF}')
    CS=$(grep "Average Recall@1:" "eval_${TAG}_s${SEED}_ep${EPOCH}_conslam.log" | head -1 | awk '{print $NF}')
    echo "  → ConPR=${CP}  ConSLAM=${CS}"
done

echo ""
echo "[$(date +%H:%M)] All done."
