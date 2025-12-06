# DR-VPR: Dual-Branch Rotation-Robust Visual Place Recognition

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Automation%20in%20Construction-blue)](https://arxiv.org/abs/...) 
[![Framework](https://img.shields.io/badge/PyTorch-1.13%2B-red)](https://pytorch.org/) 
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**State-of-the-art Visual Place Recognition for Dynamic Construction Environments**

</div>

## 📖 Introduction

This repository contains the official implementation of **DR-VPR**, a dual-branch architecture designed for robust localization in dynamic construction sites using handheld devices or UAVs.

**Key Insight:** Instead of forcing a standard CNN to "memorize" rotation invariance through data augmentation (which degrades discriminability), we explicitly encode geometric priors using a rotation-equivariant branch.

**Highlights:**
- 🏆 **SOTA Performance:** Outperforms MixVPR and CosPlace on **ConPR** (+1.6%) and **ConSLAM** (+3.5%) benchmarks.
- ⚡ **Real-Time Latency:** Extremely fast inference (**4.23 ms**) suitable for embedded robotics.
- 🔄 **Equivariance > Augmentation:** Empirically resolves the *invariance-discriminability trade-off*, achieving superior robustness without explicit rotation augmentation training.

## 🏗️ Architecture

![Architecture](assets/architecture.png)
*Figure: Overview of DR-VPR. Branch 1 (ResNet-50 + MixVPR) captures discriminative semantic features, while Branch 2 (E2ResNet + GeM) ensures mathematical rotation invariance. An attention mechanism dynamically fuses these cues.*

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YourUsername/DR-VPR.git](https://github.com/YourUsername/DR-VPR.git)
   cd DR-VPR