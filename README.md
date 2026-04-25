# DR-VPR: Dual-Branch Rotation-Robust Visual Place Recognition

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Automation%20in%20Construction-blue)](https://github.com/YuhaiW/DR-VPR)
[![Framework](https://img.shields.io/badge/PyTorch-2.0.1%20%7C%202.8.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Visual Place Recognition for handheld and UAV capture in dynamic construction environments**

</div>

## 📖 Overview

DR-VPR is a dual-branch retrieval architecture for VPR in construction sites, where handheld scanners and UAVs introduce in-plane rotations and rapid scene evolution that defeat conventional VPR methods.

**Architecture (one line):**
A frozen pretrained **BoQ-ResNet50** discriminative branch is paired with a lightweight **C₁₆-equivariant E2ResNet** branch (0.67 M parameters); the two branches are trained independently and combined only at inference through a weighted joint-scoring rule, `score = (1−β)·⟨d₁,d₁⟩ + β·⟨d₂,d₂⟩` with β = 0.10.

**Why decoupled inference-time fusion?**
A systematic ablation in the paper (Table 4) shows that train-time attention or gated-concat fusion suffers from branch-weight saturation under strong pretrained features (the equivariant branch's weight collapses to ∼10⁻⁴). Inference-time joint scoring decouples the branches and recovers the full gain.

## 📊 Results

**ConSLAM (rotation-heavy handheld benchmark, 307 valid queries)**

| Method | Backbone | Params (M) | Lat. (ms) | R@1 | R@5 | R@10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| CosPlace        | ResNet50 | 27.70 | 1.38 | 44.30 | 67.10 | 74.27 |
| MixVPR          | ResNet50 | 10.88 | 1.38 | 56.03 | 74.59 | 77.85 |
| BoQ (ResNet50)  | ResNet50 | 23.84 | 2.12 | 60.91 | 75.24 | 78.83 |
| **DR-VPR (ours)** | R50 + E2ResNet(C16) | **24.51** | **4.24** | **62.65 ± 0.82** | **75.68 ± 0.50** | **79.80 ± 0.66** |

**ConPR (full 10-sequence cross-validation protocol)**

| Method | Params (M) | R@1 |
| :--- | ---: | ---: |
| CosPlace | 27.70 | 73.48 |
| MixVPR   | 10.88 | 78.55 |
| BoQ (ResNet50) | 23.84 | 79.30 |
| **DR-VPR (ours)** | **24.51** | **79.81 ± 0.21** |

DR-VPR is **3.5–4.2× smaller** than DINOv2-backbone baselines (SALAD 87.99 M, CricaVPR 106.76 M) at 4.24 ms per image on an RTX 5090, while delivering a statistically significant +1.74 R@1 improvement on rotation-heavy ConSLAM (one-sample _t_-test, _t_ = 3.67, _p_ < 0.05 across 3 seeds).

## 🛠️ Installation

1. **Clone**
   ```bash
   git clone https://github.com/YuhaiW/DR-VPR.git
   cd DR-VPR
   ```

2. **Conda environment**
   ```bash
   conda create -n drvpr python=3.9 -y
   conda activate drvpr
   ```

3. **PyTorch** (pick the path matching your GPU)

   <details>
   <summary><b>Ampere / Ada Lovelace (RTX 30xx, 40xx, A100) — CUDA 11.8</b></summary>

   ```bash
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
       --extra-index-url https://download.pytorch.org/whl/cu118
   ```
   </details>

   <details open>
   <summary><b>Blackwell (RTX 5090, B100/B200) — CUDA 12.8</b></summary>

   ```bash
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
       --index-url https://download.pytorch.org/whl/cu128
   pip install "numpy<2"   # required for faiss-gpu compatibility
   ```
   </details>

4. **Other dependencies**
   ```bash
   pip install -r requirements.txt
   ```

| Verified GPU | PyTorch | CUDA | Python |
| :--- | :--- | :--- | :--- |
| RTX 6000 (Ada) | 2.0.1 | 11.8 | 3.9 |
| RTX 5090 (Blackwell) | 2.8.0 | 12.8 | 3.9 |

## 📂 Data Preparation

Place datasets under `./datasets/`.

### 1. GSV-Cities (training)
[Download GSV-Cities](https://github.com/amaralibey/gsv-cities) and arrange as:
```
datasets/GSV-Cities/
├── Dataframes/    # .csv files (Bangkok.csv, ...)
└── Images/        # city folders (Bangkok/, ...)
```

### 2. Construction benchmarks (evaluation)

| Dataset | Ready-to-run | Original source |
| :--- | :--- | :--- |
| **ConPR** | [Download (Drive)](https://drive.google.com/file/d/1IwfYyKhdu8hsoLxXQ7TrZabkawMt-Qge/view?usp=sharing) | [dongjae0107/ConPR](https://github.com/dongjae0107/ConPR) |
| **ConSLAM** | [Download (Drive)](https://drive.google.com/file/d/1uudYN0WuWhkMYqg-6-LFDL6ueFNCoI3D/view?usp=sharing) | [mac137/ConSLAM](https://github.com/mac137/ConSLAM) |

Extract under `./datasets/ConPR/` and `./datasets/ConSLAM/`.

## 🚀 Quick Start

### Option A: Evaluate with released checkpoints

The 3-seed pretrained Branch-2 checkpoints used in the paper (each ≤ 5 MB; BoQ branch is loaded automatically from the official torch.hub release):

> **Pretrained weights:** _Will be released upon paper acceptance — see `checkpoints/README.md`._

```bash
# After downloading checkpoints into ./checkpoints/
python eval_rerank_standalone.py \
    --dataset conslam \
    --ckpt checkpoints/equi_seed42.ckpt \
    --beta 0.10
# → R@1 ≈ 62-63 (single-seed; paper reports 62.65 ± 0.82 over 3 seeds)
```

### Option B: Train from scratch (≈ 6 h per seed on RTX 5090)

```bash
# Train Branch 2 (the C16-equivariant standalone) for each seed
for seed in 1 42 190223; do
    python train_equi_standalone.py --seed $seed
done

# Evaluate the resulting checkpoints with joint scoring
python eval_rerank_standalone.py --dataset conslam --beta 0.10
python eval_rerank_standalone.py --dataset conpr   --beta 0.10
```

Branch 1 (BoQ-ResNet50) is **frozen** throughout and loaded from the official Bag-of-Queries release at runtime. Only the 0.67 M parameters of Branch 2 are trained.

## 🧪 Reproducing the baselines

```bash
python eval_baselines.py --method all --dataset all --seeds 1 42 123
# Evaluates CosPlace, MixVPR, BoQ-R50, BoQ-DINOv2, SALAD, CricaVPR, DINOv2 alone
# on both ConSLAM and ConPR, reporting R@1/5/10.
```

## ⏱️ Latency benchmarking

```bash
python benchmark_latency.py
# Reports forward-pass time on the available GPU at 320×320 resolution
```

## 📁 Repository layout

```
DR-VPR/
├── train_fusion.py              # joint train-time fusion entry (legacy / ablation)
├── train_equi_standalone.py     # canonical Branch-2 trainer (paper main)
├── eval_rerank_standalone.py    # canonical evaluator (paper main: BoQ + C16 standalone)
├── eval_rerank.py               # eval for full DualBranch checkpoints
├── eval_baselines.py            # 6 baseline evaluations
├── benchmark_latency.py         # inference-time measurement
├── test_conpr.py / test_conslam.py
├── conpr_eval_dataset_rot.py    # ConPR data interface
├── Conslam_dataset_rot.py       # ConSLAM data interface
├── models/                      # backbones, aggregators (BoQ, MixVPR, GeM, ...), E2ResNet
├── dataloaders/                 # GSV-Cities + construction val sets
├── utils/                       # losses, validation helpers
└── assets/                      # README figures
```

## 🎓 Citation

```bibtex
@article{wang2026drvpr,
  title   = {DR-VPR: Dual-Branch Rotation-Robust Visual Place Recognition for Dynamic Construction Environments},
  author  = {Wang, Yuhai and Hu, Xiao and Shi, Yangming and Ye, Yang},
  journal = {Automation in Construction},
  year    = {2026}
}
```

## 📄 License

MIT — see [LICENSE](LICENSE).
