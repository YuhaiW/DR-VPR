#!/bin/bash
# Wait for P1 standalone 3-seed training to finish, then auto-run eval + aggregate.
# Trigger this AFTER launching run_p1_2seeds_extend.sh in another background task.
#
# Output:
#   - eval_p1_3seed_interim/rerank_s{seed}_ep{best_ep}.log per seed
#   - eval_p1_3seed_interim/AGGREGATE_SUMMARY.log final 3-seed mean ± std

set +e
cd /home/yuhai/project/DR-VPR

OUTDIR="eval_p1_3seed_interim"
mkdir -p "$OUTDIR"

# --- Wait until no train_equi_standalone.py processes alive ---
echo "[$(date)] waiting for P1 training (train_equi_standalone.py) to finish..."
while pgrep -f "python train_equi_standalone.py" > /dev/null; do
    sleep 60
done
echo "[$(date)] training detected as done."

# Sanity: confirm all 3 seeds have ckpts saved
for SEED in 1 42 190223; do
    CKPT_DIR="LOGS/equi_standalone_seed${SEED}_ms/lightning_logs/version_0/checkpoints"
    if [ ! -d "$CKPT_DIR" ] || [ -z "$(ls -A $CKPT_DIR 2>/dev/null)" ]; then
        echo "MISSING ckpts for seed=$SEED at $CKPT_DIR"
        exit 1
    fi
done

# --- Eval each seed's val-best ckpt with β sweep on ConSLAM ---
export BOQ_IMG_SIZE=320

for SEED in 1 42 190223; do
    CKPT_DIR="LOGS/equi_standalone_seed${SEED}_ms/lightning_logs/version_0/checkpoints"
    # Pick highest val R1 ckpt (filename format ..._epoch(XX)_R1[X.XXXX].ckpt)
    BEST_CKPT=$(ls -1 "$CKPT_DIR"/*.ckpt | \
        awk -F'R1\\[|\\]' '{print $2 "\t" $0}' | sort -k1 -gr | head -1 | cut -f2)
    EP=$(basename "$BEST_CKPT" | grep -oE 'epoch\([0-9]+\)' | grep -oE '[0-9]+')
    R1=$(basename "$BEST_CKPT" | grep -oE 'R1\[[0-9.]+\]' | tr -d 'R1[]')

    OUTLOG="$OUTDIR/rerank_s${SEED}_ep${EP}.log"
    echo ""
    echo "[$(date)] Eval seed=$SEED, val-best ep=$EP (val R@1=$R1) → $OUTLOG"
    EQUI_CKPT="$BEST_CKPT" mamba run -n drvpr python eval_rerank_standalone.py > "$OUTLOG" 2>&1
    # Quick glance
    echo "  Best β line:"
    grep "Best β" "$OUTLOG" | head -1
done

# --- Aggregate via aggregate_rerank_sweep.py ---
echo ""
echo "[$(date)] Running aggregate_rerank_sweep.py on $OUTDIR ..."
mamba run -n drvpr python aggregate_rerank_sweep.py "$OUTDIR" 2>&1 | tee "$OUTDIR/AGGREGATE_SUMMARY.log"

echo ""
echo "=========================================="
echo "[$(date)] DONE."
echo "Per-seed best β R@1:"
grep -h "Best β" "$OUTDIR"/rerank_s*.log
echo ""
echo "3-seed mean ± std (val-best epoch × best β with selection bias warning):"
tail -10 "$OUTDIR/AGGREGATE_SUMMARY.log"
echo "=========================================="
