#!/bin/bash
# Train C4 + C16 P1 standalone equi for 3 seeds each (C8 already done).
# Sequential on a single GPU.

set -e
cd /home/yuhai/project/DR-VPR

SEEDS=(1 42 190223)

for ORI in 4 16; do
  for SEED in "${SEEDS[@]}"; do
    TAG="ms_C${ORI}"
    LOG="train_equi_standalone_C${ORI}_seed${SEED}.log"
    echo "[group-ablation] starting C${ORI} seed=${SEED}  → log ${LOG}"
    RUN_TAG="${TAG}" mamba run -n drvpr python train_equi_standalone.py \
      --seed "${SEED}" --max_epochs 10 --orientation "${ORI}" \
      2>&1 | tee "${LOG}"
    echo "[group-ablation] done C${ORI} seed=${SEED}"
  done
done

echo "[group-ablation] all 6 trainings finished"
