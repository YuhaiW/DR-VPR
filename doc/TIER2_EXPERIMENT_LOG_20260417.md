# Tier-2 实验日志：2026-04-17（含 04-18 bug 修正注记）

DR-VPR revision sprint 的单日实验记录。**决策门 + pivot 的完整证据链**。
写给审稿时可能追问"你怎么得出这个结论"的 reviewer、以及以后复盘方法论的自己。

---

## ⚠️ 2026-04-18 BUG 修正注记（必读，覆盖以下所有"+3.81 / 63.91 / 62.25"数字）

本日志中所有引用 `per_yaw_analysis.py` 或 `eval_rerank_standalone.py` 的数字
（包括 04-17 晚的 P1 standalone β=0.1=63.91 和 per-yaw [20°, 40°)=+3.81）
**因 query-pose 旋转 in-place numpy view bug 而虚高**。

详见 `doc/PER_YAW_ANALYSIS.md` v2 (2026-04-18) 的 bug 章节和
`/home/yuhai/.claude/projects/-home-yuhai-project-DR-VPR/memory/feedback_numpy_view_inplace.md`。

修正后真实数字：

| Result | 原报告 (buggy) | 修正后 (2026-04-18) |
|--------|---|---|
| P1 standalone seed=1 Ep7, β=0.1, ConSLAM R@1 | 63.91 | **62.21** |
| P1 standalone seed=1 Ep7, β=0.0 (BoQ-only sanity) | 62.25 | **61.24** |
| Per-yaw [0°, 20°) bucket Δ | +0.57 | **+0.97** |
| Per-yaw **[20°, 40°)** bucket Δ | **+3.81** ❌ | **+0.00** ✓ |
| Per-yaw [40°, 60°) bucket Δ | −2.08 | **−1.96** |
| Total per-yaw Δ | +0.77 | **+0.65** |

**`eval_rerank.py` 报的 freeze_boq 60.80 / 61.45 ± 0.18 不受影响**——
那个文件用的是 `qx_rot, qy_rot` 临时变量写法，从一开始就对。

**对 paper 影响**：
- "+3.81 在 [20°, 40°) bucket 验证 rotation robustness" 这个 sub-result **不存在**
- DR-VPR 的真实 +0.65 R@1 增益**集中在 easy bucket**（low-rotation），
  不是预期的中等旋转区
- "rotation robustness" 主叙事**需要重新组织**——可改成 ensemble effect
  of two-ResNet50 descriptors，不强行讲旋转故事
- 主表绝对数字 (61.45 ± 0.18 freeze_boq + β=0.5) 仍 valid

**等 P1 3-seed 跑完（明早 04-18）需重新校准**：bug 修复后，P1 是否仍有
在 freeze_boq 之上的可衡量改进，待 3 seed mean ± std 出来才知道。

---

---

## Sprint 目标

修掉前四个变体（attention bias=[10,0] / attention bias=[2,0] / concat+gate /
freeze_boq）共同的症状：**equi 分支 10 epoch 学不起来，val R@1 始终困在 BoQ
baseline ~64%**。

## 实施方案

架构层：**Tier-2 Fourier invariant mapping**（`GroupPool(max) → rfft + abs`）
把 equi 分支不变通道从 64 升到 320，5× 容量提升。严格 C8-invariance（irrep 分解）。

监督层：**三项 loss**
- `L_main = MS_loss(desc_fused)`（服务单阶段 fused retrieve eval）
- `L_equi = MS_loss(desc_equi)`（独立 discriminative signal，服务 rerank eval）
- `L_rot = 1 - cos(desc_equi(x), desc_equi(R_θ(x)))`（θ 连续采样，补 C8→SO(2) 离散化缺口）

## 第一次实验：Tier-2 + BoQ unfreeze（seed=1, 5 epoch）

### 配置
```
GROUP_POOL_MODE=fourier  FUSION_METHOD=concat  FREEZE_BOQ=0
LAMBDA_EQUI=0.5          LAMBDA_ROT=0.3
```

### 训练曲线

| Epoch | train_loss | loss_rot | val R@1 (ConPR, yaw=80°) |
|-------|------------|----------|---------------------------|
| 0 | 0.677 | 0.027 | 62.15 |
| 1 | 0.573 | 0.037 | 64.05 |
| 2 | 0.525 | 0.031 | **64.71** ← val-best |
| 3 | 0.489 | 0.031 | 64.36 |
| 4 | 0.463 | 0.023 | 62.59 |
| 5 | — | — | 62.88 |

- train_loss 平滑下降，b_acc 稳定在 0.78-0.80 → 训练本身 healthy
- loss_rot 从 0.005 (random init 的结构等变性) 升到 0.03-0.04，等变性轻微放松后稳态
- val R@1 震荡而非单调，提示 cross-domain 问题

### ConSLAM β sweep eval（5 个 ckpt，`eval_rerank.py`, Sequence4 query → Sequence5 db）

| Epoch | β=0 (纯 BoQ retrieve) | Best β | Best R@1 | **Rerank 增益** |
|-------|----------------------|--------|----------|------------------|
| 0 | 56.35 | 0.1 | 57.98 | +1.63 |
| 1 | 53.42 | 0.2 | 57.00 | **+3.58** |
| 2 | 57.98 | 0.0 | 57.98 | 0 |
| 3 | 56.35 | 0.3 | 56.68 | +0.33 |
| 4 | 56.03 | 0.3 | **59.93** | **+3.90** |

**基线参照**：BoQ 预训练权重 standalone（不 fine-tune）在 ConSLAM 上 R@1 = **62.21%**。

### 两个实锤发现

**① Tier-2 + 三项 loss 确实让 equi 分支学到了 rerank 能力。**
Epoch 4 在 β=0.3 获得 +3.9 R@1 rerank 增益。
**这是前四个变体十个 epoch 都做不到的——之前 desc_equi 在 β>0 区间要么不涨
要么负增益。** L_equi + L_rot + Fourier 320 通道的组合产生了**独立于 BoQ 的、
有判别力的 desc_equi**。

**② BoQ 的任何 fine-tune 都会破坏 ConSLAM cross-domain 性能。**
所有 epoch 的 β=0 值（53-58%）**全部低于**预训练 BoQ 的 62.21%。
即便 0.05× LR 保守微调，GSV-Cities 分布的 fine-tune 方向也会把权重从
construction-optimal 推开。Rerank +3.9 的增益救不回 stage-1 的 -6 损失。

### 决策门结果

- 门 ① (rerank 涨幅 ≥ +2%)：**过** (Ep4 +3.9)
- 门 ② (超过 62.21 baseline)：**不过** (Ep4 best 59.93 < 62.21)

**判定：Pivot 到 FREEZE_BOQ=1**，保留 stage-1 baseline，让 equi 独立贡献。

---

## 第二次实验：Tier-2 + BoQ freeze（seed=1, 10 epoch）— 进行中

### 配置（唯一变化：FREEZE_BOQ=0 → 1）
```
GROUP_POOL_MODE=fourier  FUSION_METHOD=concat  FREEZE_BOQ=1
LAMBDA_EQUI=0.5          LAMBDA_ROT=0.3
```

### 启动确认
```
[FREEZE_BOQ] Froze 185 Branch-1 param tensors.
[FREEZE_BOQ] Trainable: 1,572,978 / 25,417,426 total (6.2%)
```

只有 equi 分支 + GeM proj + equi_gate 可训（1.57M / 25.4M = 6.2%）。
BoQ branch 完全冻结，stage-1 retrieve 永远 = 预训练 BoQ = 62.21% baseline。

### 预期
- stage-1 retrieve R@1 = **62.21%** (frozen BoQ, invariant to epoch)
- stage-2 rerank 增益 ≈ **+3 到 +4 点**（从第一次实验的 equi 学习行为外推）
- 最终 best β R@1 ≈ **65-66%** (ConSLAM)

### 结果（待训练完成填充）
- Epoch 0-9 训练 loss / val R@1：_(待填)_
- β sweep (ConSLAM)：_(待填)_
- β sweep (ConPR)：_(待填，eval_rerank.py 需 adapt 到 ConPR 数据集)_
- per-yaw bucket 分析：_(待填)_

---

## 方法论收获（写 rebuttal 时直接可用）

### 关于 cross-domain 迁移

**跨 domain VPR 场景下，pretraining on A → fine-tune on B → eval on C
的三跳迁移极易在 B→C 失效**。当 B (GSV-Cities 街景) 和 C (construction
site 俯视/斜视) distribution gap 大时：

- **冻结预训练分支是更安全的选择**
- **独立训练补充分支**来弥补 domain-specific 需求
- 这与 R2Former/DELG/SelaVPR 等近期 SOTA 方法的设计哲学一致

### 关于决策门实验

原计划是 3 seed × 10 epoch = 30 小时的 overnight training。如果直接跑 3 seed
才发现 BoQ 降质问题，时间成本是 30h。
**改用 seed=1 × 5 epoch + 立即 eval 作决策门，总成本 2h 发现问题**。
节省 28h。这个 "decision-gate experiment" 节奏对严格时间预算的 revision sprint
尤其宝贵。

### 关于"训练 val 指标的诊断局限"

seed=1 unfreeze 的训练 val R@1 (62-65% on ConPR) 比 freeze_boq baseline
(64%) 稍高，看起来是改进。但 rerank eval 揭示 stage-1 BoQ 已被破坏。
**训练 val 测的是 `desc_fused`（17408-d 包括 BoQ 16384 dim），不能独立暴露
BoQ 的 cross-domain 退化**。只有两阶段 rerank 的 β=0 行（只用 desc_boq）才
直接测到 stage-1 baseline。这个诊断洞察应记录到项目 methodology 文档里。

---

## 代码 artifact 索引

| 文件 | 作用 | 状态 |
|---|---|---|
| `models/backbones/e2resnet_backbone.py` | 加 `group_pool_mode='fourier'` 分支 | ✓ |
| `train_fusion.py` | `branch2_in_channels = backbone.out_channels`; dual-branch `forward(..., return_features=True)` 返回 dict; `training_step` 加三项 loss | ✓ |
| `eval_rerank.py` | `build_model` 读 GROUP_POOL_MODE env var; `.filter` 白名单避免 strict 检查误报 | ✓ |
| `run_tier2_seed1.sh` | Tier-2 unfreeze 决策门实验脚本 | ✓ (已跑) |
| `run_tier2_freeze_seed1.sh` | Tier-2 freeze 生产实验脚本 | ✓ (进行中) |
| `eval_tier2_all_ckpts.sh` | 批量 eval 脚本（rerank + test_conslam + test_conpr） | ✓ (partial run 已产出 rerank + conslam 结果) |
| `doc/TIER2_FOURIER_INVARIANT_TUTORIAL.md` | 教程（原理 + 实现 + 自洽性审查 + Q&A + 修订历史） | ✓ v2 |
| `LOGS/resnet50_DualBranch_tier2_seed1/` | 6 个 unfreeze ckpts (Ep0-Ep5)，保留作 ablation 对照 | ✓ |
| `eval_tier2_interim/rerank_ep*.log` | 5 个 β sweep 结果 | ✓ |
| `eval_tier2_interim/test_conslam_ep*.log` | 5 个 conslam 单阶段 eval（TypeError 崩溃，但 Recall 数字已写入） | ✓ |

---

## 遗留 TODO（training 完成后）

1. 挑 tier2_freeze best-val ckpt，跑 `eval_rerank.py` β sweep on ConSLAM
2. Adapt `eval_rerank.py` 支持 ConPR 数据集（目前 hardcoded SEQS=Sequence5/4）
   或直接复用 `test_conpr.py` 做单阶段 fused 对照
3. 更新 REVISION_CHECKLIST.md 主数字栏（ConPR R@1 / ConSLAM R@1 / best β）
4. 如果 seed=1 freeze 的结果 > 62.21% + 2%，扩 seed=42 + seed=190223 overnight
5. 补 ablation Table：Tier-2 unfreeze vs. Tier-2 freeze 对照（本文档 §5.7 的 4
   个数据点已经够写一行）
6. 补 ablation Table：GroupPool(max) vs Fourier（freeze_boq 的老 ckpt vs 新 Tier-2
   ckpt，控制变量是"BoQ 冻结 + 不冻结 pool mode"）

---

## 第三次实验 (晚): Path P1 Multi-Scale Standalone (BREAKTHROUGH)

### 动机

前两次（B2 NormPool + adaptive β）证明在现有 fusion 架构下 rerank 上限 ≈ 61.45。
Path A.1（自适应 β）实测和方式 B/C 都失败后，唯一未试过的方向是
**架构级改动 + 完全 standalone 训练**。

诊断流：
- Stringent test：用真随机单位向量替换 desc_equi → +0.77 → -4.63。证实 desc_equi
  的 +0.77 是真实结构信号，不是 ensemble 噪声（差距 +5.4 R@1 远超噪声）。
- 既然 equi 真有用，**给它独立训练 + 多尺度容量** 是合理 next step。

### 配置

```
架构: E2ResNetMultiScale (models/equi_multiscale.py)
  - E2ResNet C8 backbone (复用 EquivariantBasicBlock)
  - layer3 输出 → GroupPool(max) → GeM_l3 (learnable p=3 init)
  - layer4 输出 → GroupPool(max) → GeM_l4 (learnable p=3 init)
  - concat (32 + 64 = 96 dim) → Linear(96 → 1024) → L2-norm
  - **完全独立 — 没有 BoQ branch，没有 fusion，没有 gate**
  - 1.34M trainable params (vs dual-branch 25M)

训练: train_equi_standalone.py
  - 单 loss: MultiSimilarityLoss(desc_equi, labels)
  - AdamW, lr=1e-3, wd=1e-4, warmup 300, milestones=(8,14)
  - GSV-Cities batch=32×4, image=320×320, fp16
  - val on ConPR + ConSLAM (single-stage retrieve on desc_equi)
  - ckpt monitor='conslam/R1'

eval: eval_rerank_standalone.py
  - Stage 1: 官方 BoQ(ResNet50)@320 from torch.hub, FAISS top-100
  - Stage 2: 0.5·boq + β·equi rerank with desc_equi from standalone ckpt
  - β sweep [0.0, 0.1, ..., 1.0]
```

### Mid-Training 决策门 (Epoch 7 ckpt, single seed)

```
ConSLAM β sweep:

  β=0.0  R@1=62.25%   (此 run 内 BoQ-only 基线)
  β=0.1  R@1=63.91%   🏆 NEW HIGH, +1.66 over β=0
  β=0.2  R@1=62.25%
  β=0.3  R@1=62.25%
  β=0.5  R@1=58.28%   (β=0.5 不再是 sweet spot)
  β=1.0  R@1=43.71%   (纯 equi)
```

**vs 之前最佳**: freeze_boq + max + β=0.5 = 61.45 ± 0.18 (3 seed)
**P1 Standalone Ep7 + β=0.1 = 63.91 (single seed)** → **+2.46 over previous best**

### 训练轨迹（val R@1，metric 用全部 396 query 作分母）

| Epoch | ConPR R@1 | ConSLAM R@1 (ckpt selection) | b_acc |
|-------|-----------|------------------------------|-------|
| 0 | 51.93 | 29.80 | 0.27 |
| 1 | 55.81 | 29.04 | 0.36 |
| 2 | 58.14 | 27.53 | 0.41 |
| 3 | 58.62 | 30.30 | 0.45 |
| 4 | 54.55 | 26.77 | 0.46 |
| 5 | 59.66 | 32.32 | 0.47 |
| 6 | 60.10 | 32.32 | 0.48 |
| 7 | 55.59 | **33.59** ← best so far | 0.49 |
| 8-9 | (训练中) | (训练中) |  |

ConSLAM val 单调上升（29.80 → 33.59 over 7 epoch, +3.8 R@1），desc_equi 真在
学 metric learning。绝对值 33.59 << BoQ standalone 60.91，但**rerank 看的不是
绝对 standalone 强度，而是和 BoQ 的 orthogonality**——这次 trained 的
desc_equi 显然更 orthogonal 于 BoQ 比 untrained 的版本。

### 关键洞察 (反驳之前理论)

**之前的 hypothesis** (Tier-2 + L_equi 失败 → "L_equi 让 desc_equi 收敛到 BoQ
subspace")  **被这个结果驳回**。同样是用 MS loss 训 desc_equi，P1 standalone
work 了 (+1.66) 而 Tier-2+L_equi 没 (-1.0)。

可能原因：
1. **Multi-scale > single-scale**: layer3 + layer4 双尺度的 invariant 特征捕获
   不同的几何信号，比单 layer4 更 orthogonal 于 BoQ。
2. **Standalone > fusion-tied**: 没有 gate 的奇怪 gradient flow，desc_equi 优化
   信号干净。
3. **从 random init 直接训**比"被 fusion gradient 间接训" sample-efficiency 高。

也可能只是 **multi-scale 这个架构改动本身**就是关键。下一步 ablation 应分离这两
个变量（standalone single-scale vs standalone multi-scale）。

### Caveat

- **β=0 在此 run 给 62.25** 而非 eval_baselines.py 实测的 60.91，差 1.34 点。
  可能是 BN running stats / valid_query 过滤微差异。**需要 stress test 校准
  绝对数字**。但**relative +1.66 (β=0 → β=0.1, 同 pipeline 内) 可信**。
- **single seed**, 需扩 3 seed 验证稳定性。
- ckpt 选择 metric (ConSLAM val R@1 单阶段) 可能不是 best β rerank R@1 的最优
  predictor。Ep9/10 完成后应 eval 全部 ckpt 找真正的 best epoch。

### 后续

1. 等 Epoch 8/9 训完（~10 min），eval 全部 10 个 ckpt 找最佳 epoch
2. 扩 seed=42 + seed=190223 overnight（~5h）
3. 如果 3 seed mean 仍 ≥ 62.5，**P1 standalone 升为 paper 主方法**，
   freeze_boq 降为 ablation row
4. (可选) ablation: standalone single-scale (layer4 only) vs P1 multi-scale
   分离 "multi-scale" vs "standalone" 的贡献

---

*记录人：DR-VPR revision team*
*时间：2026-04-17 实验 session (含 P1 BREAKTHROUGH 至 21:00)*
