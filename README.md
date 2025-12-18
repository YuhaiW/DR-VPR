# DR-VPR: Dual-Branch Rotation-Robust Visual Place Recognition

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Automation%20in%20Construction-blue)](https://github.com/YuhaiW/DR-VPR) 
[![Framework](https://img.shields.io/badge/PyTorch-2.0.1-red)](https://pytorch.org/) 
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**State-of-the-art Visual Place Recognition for Dynamic Construction Environments**

</div>

## 📖 Introduction

This repository contains the official implementation of **DR-VPR**, a dual-branch architecture designed for robust localization in dynamic construction sites using handheld devices or UAVs.

**Key Insight:** Instead of forcing a standard CNN to "memorize" rotation invariance through massive data augmentation (which we prove degrades discriminability), we explicitly encode geometric priors using a rotation-equivariant branch.

**Highlights:**
- 🏆 **SOTA Performance:** Outperforms MixVPR and CosPlace on **ConPR** (+1.6%) and **ConSLAM** (+3.5%) benchmarks.
- ⚡ **Real-Time Latency:** Extremely fast inference (**4.23 ms** on RTX 6000) suitable for embedded robotics.
- 🔄 **Equivariance > Augmentation:** Empirically resolves the *invariance-discriminability trade-off*, achieving superior robustness without explicit rotation augmentation training.

## 🏗️ Architecture

![Architecture](assets/architecture.png)
*Figure: Overview of DR-VPR. Branch 1 (ResNet-50 + MixVPR) captures discriminative semantic features, while Branch 2 (E2ResNet + GeM) ensures mathematical rotation invariance. An attention mechanism dynamically fuses these cues.*

## 🛠️ Installation

Our environment uses **PyTorch 2.0.1** with **CUDA 11.8**. Please follow the steps below to ensure compatibility.

1. **Clone the repository**
   ```bash
   git clone https://github.com/YuhaiW/DR-VPR.git
   cd DR-VPR


2.  **Create a Conda environment**

    ```bash
    conda create -n drvpr python=3.9
    conda activate drvpr
    ```

3.  **Install PyTorch (CUDA 11.8)**
    *It is critical to install this before the other requirements.*

    ```bash
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

4.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## 📂 Data Preparation

### 1\. GSV-Cities (Training)

Download the [GSV-Cities dataset](https://github.com/amaralibey/gsv-cities). The code expects the following structure in `./datasets/`:

```text
datasets/
└── GSV-Cities/
    ├── Dataframes/   # Contains .csv files (Bangkok.csv, etc.)
    └── Images/       # Contains city folders (Bangkok/, etc.)
```

### 2\. Evaluation Datasets

| Project | Ready-to-Run Version | Original Source |
| :--- | :--- | :--- |
| **ConPR** | [Download here](https://drive.google.com/file/d/1IwfYyKhdu8hsoLxXQ7TrZabkawMt-Qge/view?usp=sharing) | [Official Repository](https://github.com/dongjae0107/ConPR.git) |
| **ConSLAM** | [Download here](https://drive.google.com/file/d/1uudYN0WuWhkMYqg-6-LFDL6ueFNCoI3D/view?usp=sharing) | [Official Repository](https://github.com/mac137/ConSLAM.git) |
## 🚀 Training

```bash
python train_fusion.py \
    --backbone_arch resnet50 \
    --agg_arch MixVPR \
    --use_dual_branch \
    --equi_orientation 8 \
    --fusion_method attention \
    --batch_size 60 \
    --lr 0.04
```

## 📊 Evaluation & Results

To evaluate a trained checkpoint:

```bash
python test_conpr.py --checkpoint_path LOGS/resnet50_DualBranch/best_model.ckpt
```

### 🏆 Comprehensive Results (ConPR & ConSLAM)

Comparison of Recall@1 scores across dynamic construction sequences (ConPR) and handheld scanning (ConSLAM). 
**MixVPR + Rot. Aug.** denotes the baseline retrained with rotation augmentation ($p=0.5, \theta \in [0, 360^\circ)$).

| Method | Dim | 0531 | 0611 | 0627 | 0628 | 0706 | 0717 | 0803 | 0809 | 0818 | **ConPR Avg** | **ConSLAM** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CosPlace† | 2048 | *58.85* | 71.92 | 91.96 | 90.84 | 82.96 | 64.26 | 75.87 | 75.84 | 62.61 | 74.96 | 48.26 |
| AnyLoc-v1† | 3072 | 46.70 | 67.78 | 87.07 | 84.89 | 81.60 | 60.95 | 65.22 | 74.46 | 63.16 | 70.20 | 38.19 |
| AnyLoc-v2† | 12288 | 51.68 | 70.14 | 89.78 | 86.74 | 86.29 | **78.34** | **82.68** | **82.37** | 69.67 | 77.52 | 48.26 |
| MixVPR† | 4096 | 56.72 | 75.19 | 92.07 | *94.63* | *86.86* | 70.56 | *81.51* | 76.90 | *72.54* | *78.55* | 56.90 |
| MixVPR + Rot. Aug. | 4096 | 58.69 | **76.85** | **93.80** | 93.54 | 86.26 | 68.62 | 72.85 | 75.11 | 69.44 | 77.24 | *57.33* |
| **DR-VPR (Ours)** | 4096 | **59.85** | *76.16* | *93.04* | **94.74** | **89.10** | *71.24* | 81.15 | *81.03* | **75.09** | **80.15** | **60.40** |

*† Official pretrained models.*

*Inference time measured on NVIDIA RTX 6000 with batch size 1.*
## Qualitative Analysis

Visual comparison demonstrating robustness against extreme rotation and structural changes.
![Qualitative](assets/qualitative.png)


Figure: Query image (Left), Top-1 Match by DR-VPR (Middle), Top-1 Match by Baseline (Right).

## 🎓 Citation

If you use this code in your research, please cite our paper:

```bibtex
Coming Soon
```

## 📄 License

This project is licensed under the MIT License.