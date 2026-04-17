#!/bin/bash
# Equi-BoQ multi-seed training with per-epoch full eval
# Train 3 seeds × 10 epochs, save every epoch, eval every epoch on ConPR + ConSLAM.

set -e
cd /home/yuhai/project/DR-VPR

SEEDS=(1 42 190223)

echo "=============================================="
echo "Equi-BoQ per-epoch eval: seeds ${SEEDS[*]}"
echo "=============================================="

# ---------- Training ----------
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> [$(date +%H:%M)] Training seed=$SEED (max_epochs=10, save all) ..."
    mamba run -n drvpr python train_fusion.py --seed "$SEED" \
        2>&1 | tee "train_equiboq_s${SEED}.log"
    echo ">>> [$(date +%H:%M)] seed $SEED done."
done

# ---------- Eval ----------
echo ""
echo "=============================================="
echo "[$(date +%H:%M)] Per-epoch full eval (30 ckpts)"
echo "=============================================="

for SEED in "${SEEDS[@]}"; do
    CKPT_DIR="LOGS/resnet50_DualBranch_seed${SEED}/lightning_logs/version_0/checkpoints"
    if [ ! -d "$CKPT_DIR" ]; then
        echo "!!! No checkpoint dir for seed $SEED at $CKPT_DIR"
        continue
    fi
    for CKPT in "$CKPT_DIR"/*.ckpt; do
        [ -f "$CKPT" ] || continue
        BN=$(basename "$CKPT")
        # Extract epoch number — filename like ..._epoch(00)_R1[0.6528].ckpt
        EPOCH=$(echo "$BN" | grep -oP 'epoch\(\K[0-9]+')
        echo ""
        echo "[$(date +%H:%M)] eval seed=$SEED epoch=$EPOCH ..."

        DRVPR_CKPT="$CKPT" mamba run -n drvpr python test_conpr.py \
            > "eval_seed${SEED}_ep${EPOCH}_conpr.log" 2>&1 || echo "  conpr eval failed"

        DRVPR_CKPT="$CKPT" mamba run -n drvpr python test_conslam.py \
            > "eval_seed${SEED}_ep${EPOCH}_conslam.log" 2>&1 || echo "  conslam eval failed"

        CP_R1=$(grep "平均 Recall@1:" "eval_seed${SEED}_ep${EPOCH}_conpr.log" | head -1 | awk '{print $NF}')
        CS_R1=$(grep "Average Recall@1:" "eval_seed${SEED}_ep${EPOCH}_conslam.log" | head -1 | awk '{print $NF}')
        echo "  → ConPR R@1=${CP_R1}  ConSLAM R@1=${CS_R1}"
    done
done

echo ""
echo "=============================================="
echo "[$(date +%H:%M)] All done."
echo "Run:  mamba run -n drvpr python summarize_per_epoch.py"
echo "=============================================="
