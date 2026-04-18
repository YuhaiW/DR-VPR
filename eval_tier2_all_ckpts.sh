#!/bin/bash
# Evaluate all tier2 seed=1 ckpts across all three protocols.
# Runs serially to share GPU with ongoing training (low contention expected).
#
# Priority order:
#   1. eval_rerank.py on ConSLAM (fastest, ~3-5 min, decision-gate signal)
#   2. test_conslam.py single-stage (~5-10 min)
#   3. test_conpr.py single-stage (~20-30 min, 10 sequences)

set -e
cd /home/yuhai/project/DR-VPR

CKPT_DIR="LOGS/resnet50_DualBranch_tier2_seed1/lightning_logs/version_0/checkpoints"
OUTDIR="eval_tier2_interim"
mkdir -p "$OUTDIR"

# GROUP_POOL_MODE MUST be fourier to match tier2 ckpt weight shapes
# (branch2_aggregator Linear: (1024, 320) in fourier mode)
export GROUP_POOL_MODE=fourier

# Discover ckpts
CKPTS=$(ls -1 "$CKPT_DIR"/*.ckpt | sort)
echo "Found $(echo "$CKPTS" | wc -l) ckpts:"
echo "$CKPTS"
echo ""

# Pass 1: rerank (fast) on ConSLAM
echo "=========================================="
echo "PASS 1: eval_rerank.py (β sweep) on ConSLAM"
echo "=========================================="
for CKPT in $CKPTS; do
    EP=$(basename "$CKPT" | grep -oE 'epoch\([0-9]+\)' | grep -oE '[0-9]+')
    OUTLOG="$OUTDIR/rerank_ep${EP}.log"
    echo ">>> rerank Ep$EP → $OUTLOG"
    DRVPR_CKPT="$CKPT" mamba run -n drvpr python eval_rerank.py 2>&1 | tee "$OUTLOG" | tail -5
    echo ""
done

# Pass 2: single-stage ConSLAM
echo "=========================================="
echo "PASS 2: test_conslam.py (single-stage fused)"
echo "=========================================="
for CKPT in $CKPTS; do
    EP=$(basename "$CKPT" | grep -oE 'epoch\([0-9]+\)' | grep -oE '[0-9]+')
    OUTLOG="$OUTDIR/test_conslam_ep${EP}.log"
    echo ">>> conslam Ep$EP → $OUTLOG"
    DRVPR_CKPT="$CKPT" mamba run -n drvpr python test_conslam.py 2>&1 | tee "$OUTLOG" | tail -5
    echo ""
done

# Pass 3: single-stage ConPR (10 sequences, longest)
echo "=========================================="
echo "PASS 3: test_conpr.py (single-stage fused, 10 seqs)"
echo "=========================================="
for CKPT in $CKPTS; do
    EP=$(basename "$CKPT" | grep -oE 'epoch\([0-9]+\)' | grep -oE '[0-9]+')
    OUTLOG="$OUTDIR/test_conpr_ep${EP}.log"
    echo ">>> conpr Ep$EP → $OUTLOG"
    DRVPR_CKPT="$CKPT" mamba run -n drvpr python test_conpr.py 2>&1 | tee "$OUTLOG" | tail -5
    echo ""
done

echo "=========================================="
echo "All evaluations complete. Results in $OUTDIR/"
echo "=========================================="
