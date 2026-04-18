# Tier-2 实验日志：2026-04-17

DR-VPR revision sprint 的单日实验记录。**决策门 + pivot 的完整证据链**。
写给审稿时可能追问"你怎么得出这个结论"的 reviewer、以及以后复盘方法论的自己。

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

*记录人：DR-VPR revision team*
*时间：2026-04-17 实验 session*
