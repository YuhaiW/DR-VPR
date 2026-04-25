#!/bin/bash
# Train C32 P1 standalone equi for 3 seeds. Sequential on single GPU.
# Larger group order → slower training; estimate ~3 hr/seed × 3 seeds = ~9 hr total.

set -e
cd /home/yuhai/project/DR-VPR

SEEDS=(1 42 190223)
ORI=32

for SEED in "${SEEDS[@]}"; do
  TAG="ms_C${ORI}"
  LOG="train_equi_standalone_C${ORI}_seed${SEED}.log"
  echo "[C32-ablation] starting C${ORI} seed=${SEED}  → log ${LOG}  @ $(date +%T)"
  RUN_TAG="${TAG}" mamba run -n drvpr python train_equi_standalone.py \
    --seed "${SEED}" --max_epochs 10 --orientation "${ORI}" \
    2>&1 | tee "${LOG}"
  echo "[C32-ablation] done C${ORI} seed=${SEED}  @ $(date +%T)"
done

echo "[C32-ablation] all 3 trainings finished @ $(date +%T)"
