# Pretrained checkpoints — code-only release

This is a **code-only release**. To reproduce the paper's main number (ConSLAM R@1 = 62.65 ± 0.82 over 3 seeds), train Branch 2 (the C₁₆-equivariant E2ResNet) from scratch using `train_equi_standalone.py`.

## Training cost

| Component | Trainable params | Time per seed (RTX 5090) | Total (3 seeds) |
| :--- | ---: | ---: | ---: |
| Branch 1 (BoQ-ResNet50) | 0 (frozen, loaded from torch.hub) | — | — |
| Branch 2 (E2ResNet C₁₆) | 0.67 M | ≈ 6 hours | ≈ 18 hours |

## Reproduce paper main number

```bash
# Train 3 seeds
for seed in 1 42 190223; do
    python ../train_equi_standalone.py --seed $seed
done

# Evaluate with joint scoring (β = 0.10)
python ../eval_rerank_standalone.py --dataset conslam --beta 0.10
python ../eval_rerank_standalone.py --dataset conpr   --beta 0.10
```

Branch 1 is **never trained** — it is loaded frozen from the official Bag-of-Queries release at runtime (no manual download needed). Each Branch-2 checkpoint is small (~2–5 MB), so 3 seeds together fit easily under any storage budget.
