#!/bin/bash
# Evaluate tier2_freeze seed=1 checkpoints so far.
# Runs 3 passes serially (GPU shared with ongoing training):
#   1. eval_rerank.py (ConSLAM β sweep) — fastest, decision gate signal
#   2. test_conslam.py (single-stage fused on ConSLAM)
#   3. test_conpr.py (single-stage fused on ConPR, 10 sequences)

set +e  # don't exit on failures — test_conslam has known TypeError after printing Avg Recall
cd /home/yuhai/project/DR-VPR

CKPT_DIR="LOGS/resnet50_DualBranch_tier2_freeze_seed1/lightning_logs/version_0/checkpoints"
OUTDIR="eval_tier2_freeze_interim"
mkdir -p "$OUTDIR"

# CRITICAL: GROUP_POOL_MODE must match ckpt's training config (fourier)
export GROUP_POOL_MODE=fourier

CKPTS=$(ls -1 "$CKPT_DIR"/*.ckpt | sort)
echo "Found $(echo "$CKPTS" | wc -l) ckpts:"
for c in $CKPTS; do echo "  $c"; done
echo ""

# Pass 1: rerank β sweep on ConSLAM (fastest)
echo "=========================================="
echo "PASS 1: eval_rerank.py β sweep on ConSLAM"
echo "=========================================="
for CKPT in $CKPTS; do
    EP=$(basename "$CKPT" | grep -oE 'epoch\([0-9]+\)' | grep -oE '[0-9]+')
    OUTLOG="$OUTDIR/rerank_ep${EP}.log"
    echo ">>> rerank Ep$EP → $OUTLOG"
    DRVPR_CKPT="$CKPT" mamba run -n drvpr python eval_rerank.py > "$OUTLOG" 2>&1
    # quick glance at summary
    echo "  Best β:"
    grep -E "Best β|BoQ standalone" "$OUTLOG" | tail -2
    echo ""
done

# Pass 2: single-stage ConSLAM
echo "=========================================="
echo "PASS 2: test_conslam.py (single-stage)"
echo "=========================================="
for CKPT in $CKPTS; do
    EP=$(basename "$CKPT" | grep -oE 'epoch\([0-9]+\)' | grep -oE '[0-9]+')
    OUTLOG="$OUTDIR/test_conslam_ep${EP}.log"
    echo ">>> conslam Ep$EP → $OUTLOG"
    DRVPR_CKPT="$CKPT" mamba run -n drvpr python test_conslam.py > "$OUTLOG" 2>&1
    # extract recall (script crashes after printing, but values are in log)
    echo "  Avg Recall:"
    grep -E "Average Recall" "$OUTLOG" | head -3
    echo ""
done

# Pass 3: single-stage ConPR (10 sequences, slower)
echo "=========================================="
echo "PASS 3: test_conpr.py (single-stage, 10 seqs)"
echo "=========================================="
for CKPT in $CKPTS; do
    EP=$(basename "$CKPT" | grep -oE 'epoch\([0-9]+\)' | grep -oE '[0-9]+')
    OUTLOG="$OUTDIR/test_conpr_ep${EP}.log"
    echo ">>> conpr Ep$EP → $OUTLOG"
    DRVPR_CKPT="$CKPT" mamba run -n drvpr python test_conpr.py > "$OUTLOG" 2>&1
    echo "  Avg Recall:"
    grep -E "Average Recall" "$OUTLOG" | head -3
    echo ""
done

echo "=========================================="
echo "All evaluations complete. Results in $OUTDIR/"
echo "=========================================="
