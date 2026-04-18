#!/bin/bash
# Option A: rerank β sweep on all 30 yesterday freeze_boq ckpts (3 seed × 10 epoch).
# These were trained last night with:
#   FREEZE_BOQ=1 FUSION_METHOD=concat GROUP_POOL_MODE=max, L_main only
# Never ran eval_rerank.py on them (power went out after training finished).
#
# Output: eval_yesterday_boq_interim/rerank_s{1,42,190223}_ep{00-09}.log

set +e
cd /home/yuhai/project/DR-VPR

OUTDIR="eval_yesterday_boq_interim"
mkdir -p "$OUTDIR"

# Architecture was trained with max pool → must eval with same
export GROUP_POOL_MODE=max

for SEED in 1 42 190223; do
    CKPT_DIR="LOGS/resnet50_DualBranch_freeze_boq_seed${SEED}/lightning_logs/version_0/checkpoints"
    if [ ! -d "$CKPT_DIR" ]; then
        echo "MISSING: $CKPT_DIR"; continue
    fi
    echo "========== seed=$SEED =========="
    for CKPT in $(ls -1 "$CKPT_DIR"/*.ckpt | sort); do
        EP=$(basename "$CKPT" | grep -oE 'epoch\([0-9]+\)' | grep -oE '[0-9]+')
        OUTLOG="$OUTDIR/rerank_s${SEED}_ep${EP}.log"
        echo ">>> seed=$SEED ep=$EP → $OUTLOG"
        DRVPR_CKPT="$CKPT" mamba run -n drvpr python eval_rerank.py > "$OUTLOG" 2>&1
        # quick-glance summary
        grep -E "Best β|β=0.0|β=0.6" "$OUTLOG" | tail -4 | sed 's/^/    /'
    done
done

echo ""
echo "=========================================="
echo "All 30 evaluations complete. Logs in $OUTDIR/"
echo "=========================================="
