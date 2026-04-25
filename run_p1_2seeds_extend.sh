#!/bin/bash
# P1 standalone equi: extend to seed=42 + seed=190223 (we already have seed=1).
# Goal: validate +2.46 R@1 over freeze_boq baseline is stable across seeds.
#
# Each seed ~1.5h serial, ~3h total. Overnight-friendly.
# After training: eval_rerank_standalone.py β sweep on each best-val ckpt,
# aggregate via aggregate_rerank_sweep.py.

set -e
cd /home/yuhai/project/DR-VPR

SEEDS=(42 190223)

echo "=========================================="
echo "P1 standalone equi (multi-scale) — extending to ${SEEDS[*]}"
echo "  (seed=1 already trained: best val ConSLAM R@1 = 33.59 @ Ep7)"
echo "  (Ep7 + BoQ rerank β=0.1 → 63.91, +2.46 over freeze_boq baseline)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Training seed=$SEED ..."
    RUN_TAG=ms mamba run -n drvpr python train_equi_standalone.py \
        --seed $SEED --max_epochs 10 \
        2>&1 | tee train_equi_standalone_s${SEED}.log
    echo ">>> Seed $SEED done."
done

echo ""
echo "=========================================="
echo "All seeds done. Next: eval_rerank_standalone.py per seed:"
echo ""
for SEED in 1 "${SEEDS[@]}"; do
    echo "  EQUI_CKPT=LOGS/equi_standalone_seed${SEED}_ms/lightning_logs/version_0/checkpoints/ \\"
    echo "      mamba run -n drvpr python eval_rerank_standalone.py \\"
    echo "      | tee eval_p1_standalone_s${SEED}.log"
done
echo "=========================================="
