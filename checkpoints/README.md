# Pretrained DR-VPR checkpoints

The 3-seed Branch-2 (E2ResNet C₁₆ multi-scale) checkpoints used in the paper will be released here upon paper acceptance.

Each checkpoint is small (≤ 5 MB) because Branch 1 (BoQ-ResNet50) is frozen and loaded from the [official Bag-of-Queries release](https://github.com/amaralibey/Bag-of-Queries) at runtime via `torch.hub`.

## Expected layout (after download)

```
checkpoints/
├── equi_seed1.ckpt        # ConSLAM val-best, seed 1
├── equi_seed42.ckpt       # ConSLAM val-best, seed 42  (closest to 3-seed mean)
├── equi_seed190223.ckpt   # ConSLAM val-best, seed 190223
└── README.md              # this file
```

## Reproducing paper results (after download)

```bash
# Single-seed evaluation
python eval_rerank_standalone.py --dataset conslam --ckpt checkpoints/equi_seed42.ckpt --beta 0.10
# → R@1 ≈ 62.65 (paper main: 62.65 ± 0.82 over 3 seeds)

# 3-seed mean (matches paper Table 1)
for seed in 1 42 190223; do
    python eval_rerank_standalone.py --dataset conslam --ckpt checkpoints/equi_seed${seed}.ckpt --beta 0.10
done
```

## Training from scratch (no checkpoints needed)

If you do not want to wait for the release, you can train Branch 2 from scratch in ~6 hours per seed on an RTX 5090:

```bash
for seed in 1 42 190223; do
    python ../train_equi_standalone.py --seed $seed
done
```
