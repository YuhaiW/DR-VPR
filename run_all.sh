#!/bin/bash
# Sequential runner: 3-seed training, then all baselines
set -e
cd /home/yuhai/project/DR-VPR

echo "========================================"
echo "[$(date)] START: training seed 190223"
echo "========================================"
mamba run -n drvpr python train_fusion.py --seed 190223 2>&1 | tee train_seed190223.log

echo "========================================"
echo "[$(date)] START: training seed 42"
echo "========================================"
mamba run -n drvpr python train_fusion.py --seed 42 2>&1 | tee train_seed42.log

echo "========================================"
echo "[$(date)] START: training seed 12345"
echo "========================================"
mamba run -n drvpr python train_fusion.py --seed 12345 2>&1 | tee train_seed12345.log

echo "========================================"
echo "[$(date)] START: baselines (6 methods x 3 seeds)"
echo "========================================"
mamba run -n drvpr python eval_baselines.py --method all --dataset all --batch_size 8 --seeds 1 42 123 2>&1 | tee baseline_eval_full.log

echo "========================================"
echo "[$(date)] ALL DONE"
echo "========================================"
