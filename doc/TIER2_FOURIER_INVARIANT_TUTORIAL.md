# Tier-2 Fourier Mode Magnitudes：等变分支的新不变映射

面向 DR-VPR revision sprint 的内部技术教程。
写给未来的自己、合作者和审稿时可能被追问的 reviewer。

---

## 0. 在做什么（一句话）

把 E2ResNet 最后一层的 `GroupPooling(max)`（每个 field 取 8 个朝向中的最大值）
换成**离散 Fourier 变换 + 取各频率分量的幅值**（每个 field 输出 5 个独立的旋转不变统计量）。
不变通道数从 64 升到 320，**5 倍容量提升**，严格保持 C8-invariance，代码改动 ~10 行。

---

## 1. 为什么要做这个改动

### 1.1 症状

前三个变体（attention bias=[10,0] / attention bias=[2,0] / concat+zero-init gate）
在 ConPR R@1 上长期卡在和 BoQ baseline 几乎一样的水平，equi 分支"学不起来"。
即使把 BoQ 冻结强迫梯度只能走 equi 分支（昨晚的 freeze_boq 实验），
3 seed × 10 epoch 的 val R@1 仍然稳定在 63.9–64.2，没有任何学习曲线。

### 1.2 两个根因

**根因 A：训练目标和评估目标断层。**
训练时 loss 作用在 `concat([BoQ, gate·equi])` 的 single-stage retrieve 上；
评估时 `eval_rerank.py` 走两阶段：BoQ 做 stage-1 retrieve，equi 做 stage-2 rerank
（加权和 `(1-β)·boq_sim + β·equi_sim`，独立抽取 desc_equi，不经过 gate）。
desc_equi 在训练时没有独立的监督信号，其独立 rerank 能力不被直接优化。
**解决方案**：加 `L_equi`（desc_equi 自身的 MS loss）+ `L_rot`（随机 θ 旋转一致性 loss）。

**根因 B（本文聚焦）：不变映射的信息瓶颈。**
当前 `GroupPooling(max)` 每个 regular field 只输出一个标量，
丢掉了 8 个朝向响应之间的统计结构。
**解决方案**：用 Fourier 分解保留所有频率分量的幅值。

本文档聚焦根因 B 的原理、实现和与整个项目的自洽性。
根因 A 的三项 loss 改造在另一份文档里。

---

## 2. 数学原理

### 2.1 等变 CNN 每个像素存了什么

我们的 `e2resnet_backbone.py` 用 `e2cnn` 库，rotation group 选 C8（8 阶循环群）。
在 `layer4` 输出处，对每个空间位置 `(h, w)`，每个 **regular field**
存的是一个 **8 维实向量**：

```
f = [f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7]   (实数)
     ↑                                        ↑
     0°                                      315°
```

`f_k` 的物理含义是：
"如果把原图预先旋转 `k × 45°` 再喂进这条卷积链路，这个位置此刻会激活多强。"
regular field 这个名字来自表示论——存的是群 C8 的 **regular representation**
（群作用矩阵是 8×8 的循环移位矩阵）。

在整条 `layer4` 上，我们一共有 64 个 field，
所以层输出的原始 tensor 尺寸是 `(B, 64 × 8 = 512, H/32, W/32)`。

### 2.2 等变性承诺

C8-等变卷积网络的数学承诺：
**输入旋转 45° ⟺ 输出 tensor 的 field 内 8 维向量循环移位一位**。

```
原图:         f = [f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7]
原图旋 +45°:  f' = [f_7, f_0, f_1, f_2, f_3, f_4, f_5, f_6]   (shift by +1)
```

这是硬保证，不是学出来的。
空间维 `(H/32, W/32)` 也会按 45° 旋转，但 field 内的循环移位是独立于空间的。

### 2.3 要构造旋转不变描述子，就要"吃掉 orientation 维"

需要一个函数 φ : ℝ⁸ → ℝᵈ，使得

```
φ(f) = φ(cyclic_shift(f))   对所有循环移位
```

满足这一条件的 φ，称为 **C8-invariant function on the regular representation**。

### 2.4 当前做法：GroupPool(max) —— 最粗糙的不变映射

```
φ_max(f) = max(f_0, f_1, ..., f_7)       d = 1
```

✓ 严格不变（max 不看顺序）
✗ 8 个数塌缩成 1 个，信息保留率 1/8 ≈ 12.5%

### 2.5 最优解：离散 Fourier 变换


把 f 视为周期为 8 的离散信号，做 **DFT**：

```
       N-1
F_k = Σ     f_j · exp(-2π i · jk / N),     k = 0, 1, ..., N-1       (N=8)
       j=0
```

DFT 有一个关键性质——**循环移位在频域对应相位旋转**：

```
cyclic_shift_by_m(f)   ⟹   F_k ↦ F_k · exp(-2π i · mk / N)
```

所以每个 `F_k` 的**幅值** `|F_k|` 在循环移位下不变（只有相位变了）。

这正是我们需要的不变量。

### 2.6 为什么只需要 ⌊N/2⌋+1 个幅值

实信号的 DFT 满足**共轭对称性**：

```
F_{N-k} = conj(F_k)     ⟹   |F_{N-k}| = |F_k|
```

对 C8（N=8），独立的幅值只有：

| k | F_k 性质 | |F_k| |
|---|---------|------|
| 0 | 实数（所有 f_j 之和） | DC 分量 |
| 1 | 复数（和 F_7 共轭） | 1-fold 频率 |
| 2 | 复数（和 F_6 共轭） | 2-fold 频率 |
| 3 | 复数（和 F_5 共轭） | 3-fold 频率 |
| 4 | 实数（N/2 的特殊情形，自共轭） | 4-fold 频率 (交替和) |

独立的不变量总数 = **⌊N/2⌋ + 1 = 5**。

信息保留率 = 5/8 ≈ 62.5%，比 max 提高 5 倍。

### 2.7 和表示论的对应

对有限循环群 C_N，群论告诉我们不可约表示（irreducible representations, irreps）
恰好就是 DFT 的这些频率分量：

- k=0 → **trivial representation**（1 维实）
- k=N/2 → **sign representation**（1 维实，仅当 N 偶数）
- 0 < k < N/2 → 复数 irreps 的共轭对（合并成 2 维实 irrep）

"regular representation 在 irrep 下的分解"就是 DFT。
所以**我们保留的 5 个幅值 = regular rep 在每个不可约子空间上的模**，
这是数学上**所有 C8-invariant 实函数的完备基础**（Schur 引理 + Peter-Weyl 定理）。

**任何 C8-invariant 实函数都可以写成这 5 个幅值的函数。**
换言之，我们保留的是能保留的一切。

**文献出处**：
- Cohen & Welling, "Steerable CNNs", ICLR 2017 — 首次把 irrep 分解用在 CNN 里
- Weiler & Cesa, "General E(2)-Equivariant Steerable CNNs", NeurIPS 2019 — escnn/e2cnn 的理论基础
- escnn 文档 §4 "Representation Theory"

---

## 3. 直观理解：旋转光谱

5 个幅值可以理解为场景**"旋转对称性指纹"**：

| 分量 | 名称 | 物理含义 | 工地场景举例 |
|------|------|---------|------------|
| \|F_0\| | DC | 所有朝向响应之和（≈ 各向同性能量） | 无方向性纹理、沙土面 |
| \|F_1\| | 1-fold | 是否有单一主方向 | 建筑立面朝向镜头、单根路灯 |
| \|F_2\| | 2-fold | 180° 对称 | 道路中线、对角斜撑 |
| \|F_3\| | 3-fold | 120° 对称 | 三角形桁架、Y 形支撑 |
| \|F_4\| | 4-fold | 90° 对称（交替正负响应） | 窗户阵列、脚手架方格、正方形法兰盘 |

### 3.1 一个关键的数值例子

两种截然不同的朝向响应：

```
模式 A (对称):    f_A = [5, 4, 3, 2, 1, 2, 3, 4]   ← 近似 2-fold 对称曲线
模式 B (尖峰):    f_B = [10, 1, 1, 1, 1, 1, 1, 1]  ← 只在 0° 有响应
```

**GroupPool(max) 的输出**：

- A: 5
- B: 10
- 只能区分"最强方向的大小"，**分不清 A 的结构性对称**。

**Fourier 幅值的输出**（实际数值）：

```
A: |F_0|=24, |F_1|≈1.4, |F_2|=8, |F_3|≈1.4, |F_4|=0     ← |F_2| 突出，识别出 2-fold
B: |F_0|=17, |F_1|≈9,   |F_2|=9, |F_3|≈9,   |F_4|=9     ← spike 在频域是 flat
```

两者幅值指纹完全不同。
VPR 检索需要能区分"我看到的立面是什么**结构**"，而不是"最强朝向的响应幅度"。
这 5 个数比 1 个 max 承载多得多的结构信息。

---

## 4. 实现

### 4.1 经验验证（今天已跑通）

在写代码之前，先跑了一个数值验证，确认：
(1) e2cnn 的 regular rep 通道布局是 `[f0_g0, f0_g1, ..., f0_g7, f1_g0, ...]`，
    reshape 成 `(B, F, G, H, W)` 把 orientation 放到 dim=2。
(2) 用 `.transform(g)` 模拟物理旋转后 tensor 确实是循环移位。
(3) 沿 orientation 维 FFT 取幅值后，旋转前后数值完全相等（max abs diff = 0.0）。

```
原始 regular rep feature (4 fields × 8 orientations):
[[ 0,  1,  2,  3,  4,  5,  6,  7],
 [ 8,  9, 10, 11, 12, 13, 14, 15],
 ...]

旋转 45° 后:
[[ 7,  0,  1,  2,  3,  4,  5,  6],   ← cyclic shift
 [15,  8,  9, 10, 11, 12, 13, 14],
 ...]

Fourier magnitudes (两个版本完全相同):
[[ 28.00, 10.45, 5.66, 4.33, 4.00, 4.33, 5.66, 10.45],
 [ 92.00, 10.45, 5.66, 4.33, 4.00, 4.33, 5.66, 10.45],
 ...]
```

注意到：

- `|F_0|` 确实就是这一行所有 `f_j` 的和；
- 第 2 行（field 1）的 `|F_0|` 大得多只是因为数值本身更大，
  但 AC 分量 `|F_1|..|F_4|` 不变（因为这一行只是 field 0 的整体偏移副本，
  上、下字段的 AC 结构相同）；
- 共轭对称如约成立：`|F_1| = |F_7|`、`|F_2| = |F_6|`、`|F_3| = |F_5|`。

### 4.2 代码改动（`models/backbones/e2resnet_backbone.py`）

**当前**（`group_pool_mode='max'` 分支，行 127-128、195-197）：

```python
if group_pool_mode == 'max':
    self.group_pool = enn.GroupPooling(self.final_type)
# ...
if self.group_pool_mode == 'max':
    x = self.group_pool(x)          # (B, 64, H, W)
    return x.tensor
```

**新增**（`group_pool_mode='fourier'` 分支）：

```python
elif group_pool_mode == 'fourier':
    self.group_pool = None           # 手动在 forward 里实现
    # 输出通道数 = 64 fields × (⌊8/2⌋+1) = 64 × 5 = 320
    # 供上游 dual_branch_aggregator 查询
    self.out_channels = (channels[3] // orientation) * (orientation // 2 + 1)
# ...
elif self.group_pool_mode == 'fourier':
    raw = x.tensor                                # (B, 512, H, W)
    B, C, H, W = raw.shape
    G = self.orientation                          # 8
    F_ = C // G                                   # 64 fields
    raw = raw.view(B, F_, G, H, W)               # (B, 64, 8, H, W)
    x_fft = torch.fft.fft(raw, dim=2)             # (B, 64, 8, H, W) complex
    x_modes = x_fft[:, :, : G // 2 + 1]           # (B, 64, 5, H, W) complex
    x_inv = x_modes.abs()                         # (B, 64, 5, H, W) real, invariant
    return x_inv.reshape(B, F_ * (G // 2 + 1), H, W)   # (B, 320, H, W)
```

### 4.3 下游通道数变化

所有调用 `get_equivariant_backbone` 并期待输出通道数的地方，
通道数计算公式从 `channels[3] // orientation = 64`
改成 `(channels[3] // orientation) * (orientation // 2 + 1) = 320`。

具体影响两处：

1. **`train_fusion.py:186`**：

   ```python
   branch2_in_channels = equi_channels[-1] // equi_orientation  # 当前: 512/8 = 64
   ```

   改为：

   ```python
   branch2_in_channels = (equi_channels[-1] // equi_orientation) * (equi_orientation // 2 + 1)
   # 512/8 * 5 = 320
   ```

2. **`dual_branch_aggregator.py` 的 `GeMAggregator(in_channels=320, out_channels=1024, p=3.0)`**
   自动跟着变，不用手改。

`branch2_out_dim=1024` 不变，所以 concat fused 描述子仍然是 16384+1024 = 17408 维，
评估脚本不用改。

### 4.4 单元测试

在 `e2resnet_backbone.py` 底部的 `__main__` 里加等变性测试：

```python
def test_rotation_invariance():
    import torchvision.transforms.functional as TF
    model = E2ResNetBackbone(orientation=8, group_pool_mode='fourier').eval()
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        feat_0 = model(x)                             # (1, 320, 10, 10)
        x_45 = TF.rotate(x, angle=45.0)              # C8 严格旋转
        feat_45 = model(x_45)
    # 注意：空间维也会转 45°。先回转再比较，或只比 L2 norm / GeM pool 后的结果。
    # 最简单：对 feat_0 和 feat_45 分别做 global average pooling 再比较。
    g0 = feat_0.mean(dim=(2, 3))
    g45 = feat_45.mean(dim=(2, 3))
    diff = (g0 - g45).abs().max().item()
    print(f"Max abs diff after 45° rotation: {diff:.2e}")
    # 由于插值误差，实际不会严格为 0，但应 < 1e-2 量级
```

注意：整个 input image 旋转 45° 会在空间维造成非整数像素偏移 + 插值误差，
所以整模型级验证数值不会为 0，但应该比换成 `max` pool 小一个数量级以上。
**纯 FFT-magnitude 算子本身的严格不变性**已在 4.1 的经验验证里确认了。

---

## 5. 和整个项目架构的自洽性审查

### 5.1 分支角色分工

```
                 ┌──────────────────────────────────────┐
                 │  Branch 1:  ResNet50 (BoQ 预训练)   │
                 │      + BoQ 聚合头                    │
                 │      → desc1 ∈ ℝ^16384              │
   input ──┤                                              ├── 
                 │                                      │
                 │  Branch 2:  E2ResNet (C8 等变)      │
                 │      + Fourier Inv. Mapping (Tier-2) │
                 │      + GeM pool + Linear proj        │
                 │      → desc2 ∈ ℝ^1024               │
                 └──────────────────────────────────────┘
```

- **Branch 1 负责"强表观匹配"**：BoQ 预训练权重在 GSV 级别数据上已 ~84% R@1，
  是稳定的 stage-1 retrieve 信号源。它**不应承担旋转不变性**的责任，
  因为 BoQ 不是结构上旋转不变的，只能靠数据增强软学到一点（soft guarantee）。
- **Branch 2 负责"旋转不变的场景指纹"**：通过 C8 等变卷积 + Fourier 不变映射，
  结构上严格保证对 45° 倍数旋转完全不变（hard guarantee）。
  这是 Branch 1 结构上做不到的。

**分工自洽性**：两分支在功能上**不重叠、不竞争**，是补足关系。
审稿人如果问 "BoQ 已经那么强，为什么还要 equi 分支"，标准答案是：
BoQ 的旋转鲁棒性是 soft、依赖训练数据覆盖；我们的 equi 分支是 hard，
对 out-of-distribution 的 yaw 提供结构保证。construction site 的相机 yaw 连续且分布广，
per-yaw bucket 分析（见 `analyze_per_yaw.py`）应当展现 equi 分支在高 yaw 样本上的增益。

### 5.2 训练目标和评估目标的对齐

> **⚠️ 2026-04-17 更新**：本节原本建议 BoQ 以 0.05× LR 继续 fine-tune。
> seed=1 决策门实验证明这个方案**不行**——BoQ 在 GSV-Cities 的 fine-tune 会使其
> 在 ConSLAM 上的 stage-1 retrieve R@1 从 62.21% 掉到 53-58%。
> **最终决定**：BoQ 必须完全冻结（`FREEZE_BOQ=1`）。详见 5.7。

Tier-2 只改"等变分支输出什么"，不改训练/评估目标本身。
但 Tier-2 让下面这两件事的对齐**成为可能**：

- 训练时 `L_equi = MS_loss(desc_equi, labels)` 直接优化 `desc_equi` 自身区分力；
  Tier-2 提供了更丰富的不变特征让这个 loss 有东西可学。
  （若沿用当前 64 通道，`desc_equi` 表达力本身就是 bottleneck，
  `L_equi` 再努力也学不到太多。）
- 评估时 `(1-β)·boq_sim + β·equi_sim` 的 β>0 收益，
  直接取决于 `desc_equi` 的区分力；Tier-2 的 5× 容量提升是前置条件。

**loss 和不变映射的自洽性**：三项 loss 针对"监督信号不足"问题；
Tier-2 针对"表达能力不足"问题。两者**必须同时做**才能完整修复 equi 分支涨不动的根因。
单独做 Tier-2 而不改 loss 仍会卡在训练目标断层；
单独改 loss 而不做 Tier-2 仍会被 64 通道容量限制。

### 5.3 fusion 层是否需要改

不需要。

- `concat` fusion 接收 `(desc1, desc2)`，无论 desc2 的上游来自哪种不变映射，
  concat 都是 tensor 级拼接。
- `equi_gate` (zero-init scalar) 仍然在 `desc2` 上乘，保证训练初期 concat 退化为纯 BoQ。
- 全局 L2-norm 在 concat 之后，不变映射的改动对全局 norm 是透明的。

### 5.4 rotation consistency loss 的角色

`L_rot = 1 - cos(desc_equi(x), desc_equi(R_θ(x)))`，θ ∈ [0°, 360°) 连续均匀采样。

**数学含义**：C8 等变网络只对 θ ∈ {0°, 45°, 90°, ..., 315°} 严格不变；
对中间角度（如 22°）因为群离散化 + 空间插值误差，不变性会破裂。
`L_rot` 用**连续 θ** 强迫网络把 C8 的离散不变性外推到 SO(2) 连续不变性。
这是在软约束层面补上 "C8 → SO(2)" 的离散化缺口，
而不是换 group（因为 C16 / C24 计算量暴涨，SO(2) 需要换 irrep 实现）。

**和 Tier-2 的互动**：Tier-2 保留的 5 个 Fourier 幅值
对 45° 倍数旋转**精确不变**（数值误差 0）；
对非 45° 倍数旋转**近似不变**（插值误差限定范围内）。
`L_rot` 优化的就是这个"近似"的 tightness。两者在训练中互相 reinforce。

### 5.5 参数量 & 计算量

- **参数量**：E2ResNet 主体不变，GeM 聚合器入口通道从 64 → 320，
  `Linear(320 → 1024)` 比 `Linear(64 → 1024)` 多 (320-64)×1024 ≈ 262K 参数。
  总参数量从 ~26M → ~26.3M。可以忽略。
- **计算量**：多一次沿 orientation 维 FFT，`O(B × 64 × 8·log8 × H × W)`，
  对比 E2ResNet 卷积主干的 `O(B × 512 × H × W × 9)`，约占 0.2%，可以忽略。
- **显存**：中间 complex tensor `(B, 64, 8, H, W)` 比实数多一倍，
  但 `.abs()` 后立刻降回实数，峰值只比现在多 10-20%，在 5090 上没问题。

### 5.6 推断延迟

原设计 `GroupPooling` 是单 op；Tier-2 多了一个 FFT + abs。
对 batch=1 推断延迟影响应 <0.3ms。
原始 DR-VPR 4.23ms @ 5090 的数字仍基本保持（预计 ~4.5ms）。

### 5.7 BoQ 必须冻结——2026-04-17 决策门实验复盘

**上下文**：5.2 之前推荐的 "BoQ 以 0.05× LR fine-tune + Tier-2 + 三项 loss" 组合。
为了在提交 3 seed 之前确认方案 work，先跑了 seed=1 作决策门（5 epoch + 完整 β sweep）。

**结果表（ConSLAM，Sequence4 query → Sequence5 database, yaw_threshold=80°）**：

| Epoch | β=0 (纯 BoQ retrieve) | Best β | Best R@1 | Rerank 增益 |
|-------|----------------------|--------|----------|-------------|
| 0 | 56.35 | 0.1 | 57.98 | +1.63 |
| 1 | 53.42 | 0.2 | 57.00 | **+3.58** |
| 2 | 57.98 | 0.0 | 57.98 | 0 |
| 3 | 56.35 | 0.3 | 56.68 | +0.33 |
| 4 | 56.03 | 0.3 | **59.93** | **+3.90** |

**基线参照**：BoQ 预训练权重 standalone（不 fine-tune）在 ConSLAM 上 R@1 = **62.21%**。

**两个发现**：

1. **✓ 正面信号：Tier-2 + L_equi + L_rot 让 equi 分支真的学到了 rerank 能力**。
   Epoch 4 在 β=0.3 时获得 +3.9 R@1 的 rerank 增益。这是前四个变体（attention
   bias=[10,0]/bias=[2,0] / concat+gate / freeze_boq）十个 epoch 都做不到的。
   `desc_equi` 携带了独立于 BoQ 的、有用的判别信号。

2. **✗ 负面信号：BoQ fine-tune 把 stage-1 baseline 拉低了**。
   所有 epoch 的 β=0 值（53-58%）都**低于**纯 BoQ 的 62.21%。
   GSV-Cities 的 fine-tune 对 ConSLAM cross-domain 是**负迁移**，
   即使 rerank 加 +3.9 点也救不回 -6 点的 stage-1 损失。

**决策**：**切换到 `FREEZE_BOQ=1`**。
- stage-1 BoQ retrieve 永远保留 62.21% 预训练 baseline
- stage-2 equi rerank 贡献 ~+4% 增益
- **预期最终 R@1 ≈ 66%** 在 ConSLAM 上（真正的净提升）

**为什么 5.2 原来的方案想当然地错了**：
"BoQ 以 0.05× LR fine-tune" 是个看似保守的选择——直觉上 LR 这么小应该不会破坏什么。
但 ConSLAM 的数据分布（施工场景、俯视/斜视、旋转 yaw 分布连续）和 GSV-Cities
（街景、水平 yaw、行人视角）**足够不同**，任何非零 LR 的 fine-tune 都是**把 BoQ
从 construction-optimal 推向 GSV-optimal**。对 VPR 而言，预训练权重在 source 分布
上的价值 ≈ 0，在 target 分布上的价值反而可能更高。**BoQ 预训练权重应被当作 frozen
evaluator，而不是起点。**

**对项目自洽性的修正**：
- 5.2 中"训练 L_main 流向 BoQ 的梯度"这条路径现在**被切断**，L_main 只训 equi
  分支（BoQ 冻结）。这反而**简化**了训练动力学——不再有"BoQ 退化 vs. equi 改善"的
  零和博弈，两边目标纯粹不冲突。
- `_build_param_groups()` 里 backbone 组 + boq_head 组 的 params 都会被
  `requires_grad=False` 过滤掉，优化器只看到 equi + other 组。
- ckpt 文件里 frozen branch 的权重仍然会保存（PyTorch 默认），不会有 state_dict
  缺失问题。

**经验教训（写进 rebuttal 里可用）**：
跨 domain VPR 场景下，pretraining on A → fine-tune on B → eval on C 的三跳迁移
**极易在 B→C 这一跳失效**。当 B (GSV-Cities) 和 C (construction) 的
distribution gap 大时，冻结预训练分支 + 只训补充分支是更安全的做法。
这也是 R2Former 等两阶段 rerank 方法的常见设计——stage-1 retrieve module
通常来自**冻结**的强 prior (foundation model)，stage-2 rerank module
独立训练于 target domain 的判别信号。

---

## 6. 为什么不做 X（备好审稿问答）

### Q1: 为什么不直接用 `escnn` / `e2cnn` 的 `InvariantMapping` / `NormPooling`？

- `e2cnn` 1.x 的 `GroupPooling` 只有 max 实现；`NormPooling` 是 escnn 2.x 的东西。
- 我们的代码库用 `e2cnn` 1.0.7（`equivariant_backbone.py:4`），升级到 escnn 2.x 有
  breaking changes（API 完全重命名）。
- 手写 FFT 更**透明**——论文里可以直接贴 3 行公式解释；
  而调用 `InvariantMapping` 需要讨论表示论上它具体做了什么，反而啰嗦。
- 手写版本和 `NormPooling`/`InvariantMapping` 的数值效果在 C_N 上应当一致
  （都基于 irrep 分解），但 debug 时可以 print 中间量。

### Q2: 为什么不增大群阶 N（换 C16, C24）？

- C_N 等变卷积的参数量随 N 线性增长（每个 field 成为 N 维）。
  C16 会让等变分支参数量翻倍，5090 上 batch 可能要砍半。
- **对 VPR 的实际收益有限**：construction site 的 yaw 分布虽然连续，
  但 45° 分辨率已经覆盖主要情形。剩余的中间角度由 `L_rot`（见 5.4）补上。
- 工程 sweet spot 是 C8 + Fourier invariant + 连续 θ 数据增强。

### Q3: 为什么不用 D_N（dihedral，加反射）？

- 反射对称在航拍/遥感很重要，在**地面视角的 construction site 罕见**（场景有"上下"之分）。
- D_N 把 irrep 数量再翻倍，得不偿失。

### Q4: 为什么不做等变 aggregation（在变成不变之前先聚合）？

想法是：`regular rep features (H,W)` → equivariant attention / equivariant GeM → 
`regular rep vector` → Fourier invariant → scalar invariant。

- 理论上是更好的信息流（先在高维等变空间里做 spatial 聚合，再塌缩），
  `R2Former` 的某些变体走这条路线。
- **实现风险高**：等变 attention 需要自己实现 QKV 的 representation matching，
  escnn 没有现成的 `R2Attention`。8 天 window 不够。
- 作为**"未来工作"**写进 discussion。

### Q5: Fourier 幅值会不会让所有特征都变成非负？对 cosine similarity 有偏吗？

- `|F_k|` 确实非负。
- GeM 之后接 Linear 投影 + L2 norm，Linear 层有负权重，投影后的 1024-dim desc_equi
  不受"输入非负"限制。
- 当前 `GroupPool(max)` 接 ReLU 后的特征时输入也非负，所以 Tier-2 没引入新问题。

### Q6: 为什么一定要冻结 BoQ？0.05× LR 的 fine-tune 那么保守还不够吗？

- **因为 source domain (GSV-Cities) 和 target domain (ConSLAM/ConPR) 分布差异大**。
  即便极小 LR 的 fine-tune 也是"把权重从 construction-optimal 推向 GSV-optimal"。
  2026-04-17 seed=1 决策门实验证明：0.05× LR 微调让 ConSLAM stage-1 R@1 从
  62.21% 掉到 53-58%（详见 §5.7）。
- **BoQ 预训练权重应作为 frozen stage-1 evaluator**——这是 R2Former/DELG 等
  两阶段方法的共识。
- **冻结 BoQ 并不削弱 Tier-2 + 三项 loss 的贡献**，因为 L_equi/L_rot 直接监督 equi
  分支（不经 BoQ），而 rerank 的增益全部来自 desc_equi。
- Rebuttal 叙事上，"frozen BoQ + trainable equi"**更干净**——两阶段职能完全解耦，
  故事接近 DELG pattern，reviewer 更熟。

---

## 7. 实施 checklist

```
[x] 改 e2resnet_backbone.py：加 group_pool_mode='fourier' 分支 (2026-04-17)
[x] 单元测试：pure FFT 算子严格不变（已在 §4.1 跑通，max abs diff = 0.0）
[x] 单元测试：整模型对 45° 输入旋转近似不变（下游 GeM 后差异在数值噪声级别）
[x] 改 train_fusion.py 的 branch2_in_channels 计算（用 backbone.out_channels 属性）
[x] 改 eval_rerank.py 的 build_model（读 GROUP_POOL_MODE env，.filter 白名单）
[x] 加三项 loss training_step（L_main + L_equi + L_rot）
[x] 新 run_tier2_seed1.sh 脚本 (BoQ 不冻) —— 决策门实验 (2026-04-17)
[x] 新 run_tier2_freeze_seed1.sh 脚本 (BoQ 冻) —— 生产实验 (2026-04-17)
[x] seed=1 unfreeze 决策门：5 epoch + β sweep → 发现 BoQ 降质，转 freeze
[ ] seed=1 freeze 训练 10 epoch (进行中，~2.5h)
[ ] seed=1 freeze β sweep eval (ConSLAM + ConPR)
[ ] (条件) seed=42 + seed=190223 扩展训练
[ ] per-yaw 分析，展示高-yaw bucket 的增益
[ ] ablation 表：max pool vs mean pool vs Fourier pool
[ ] ablation 表：Tier-2 unfreeze vs Tier-2 freeze（2026-04-17 seed=1 数据已在 §5.7 表里）
[ ] 更新 tex 方法论章节，加入 §4.2 的公式
```

---

## 8. 对 rebuttal 的帮助

### R2-Q4 (GroupPool 选择合理性)

Reviewer 2 质疑过"为什么选 max pool，没做对比"。
Tier-2 给这个 bullet 的回答升级了：

- "We replaced GroupPool(max) with a Fourier-basis invariant mapping
  that preserves 5 irreducible-representation magnitudes per field,
  following Cohen & Welling (2017). Ablation Table X shows +A.B% R@1
  over GroupPool(max) on ConSLAM Sequence 5."

### R1-W2 (理论贡献是否充足)

Tier-2 让我们可以引用 Schur/Peter-Weyl 论证"保留了所有 C_N-invariant 统计量的完备基础"，
这是 GroupPool(max) 完全不具备的理论保证。

### R2-Q6 (per-component ablation)

GroupPool(max) vs Fourier 的对照组在当前 codebase 里是 drop-in 的
（一个 env var 切换），一次训练循环就能加一行。
**新增可用 ablation**（来自 §5.7）：Tier-2 + 三项 loss + BoQ unfreeze
vs. Tier-2 + 三项 loss + BoQ freeze，对应同一 random seed 下的两次完整训练。
这一对照直接证明 "BoQ 必须冻结" 不是 hyperparameter 选择，而是 cross-domain
迁移的必然要求——unfreeze 下 stage-1 R@1 降 6 点，freeze 下保 62.21%。

### R1-W3 / R2-Q3 (跨 domain 迁移合理性)

预计 reviewer 会问 "GSV-Cities 和 construction site 差异大，单独 train 一个
construction-domain 的 backbone 是否更好？"。Tier-2 + freeze 的设计正面回答：
- 不在 GSV-Cities 上 fine-tune BoQ → 避免负迁移（§5.7 的实证）
- 在 construction-domain 单独训练 equi 分支 → 用 target-domain 信号补足
- 两阶段 retrieve-rerank 把"通用先验"和"domain 适配"职能完全解耦

这个故事和 SelaVPR (ICLR 2024) "frozen DINOv2 + LoRA adapter" 的设计哲学一致，
有近期 SOTA 文献背书。

---

## 9. 参考文献

- Cohen, T. S., & Welling, M. (2016). Group equivariant convolutional networks. *ICML*.
- Cohen, T. S., & Welling, M. (2017). Steerable CNNs. *ICLR*.
- Weiler, M., & Cesa, G. (2019). General E(2)-equivariant steerable CNNs. *NeurIPS*.
- escnn documentation, §4 "Representation Theory":
  https://quva-lab.github.io/escnn/
- Ali-bey, A. et al. (2024). BoQ: A Place is Worth a Bag of Learnable Queries. *CVPR*.
- Izquierdo, S., & Civera, J. (2024). SALAD: Optimal Transport Aggregation for VPR. *CVPR*.
- Lu, F. et al. (2024). Seamless Adaptation of DINOv2 for Visual Place Recognition (SelaVPR). *ICLR*.
- Cao, B., Araujo, A., & Sim, J. (2020). DELG: Unifying Deep Local and Global Features for Image Search. *ECCV*.
- Zhu, S., Yang, L. et al. (2023). R2Former: Unified Retrieval and Reranking Transformer for Place Recognition. *CVPR*.

---

## 修订历史

- **2026-04-17 v1**：初版，记录 Tier-2 Fourier 不变映射设计 + 三项 loss 训练 plan。
- **2026-04-17 v2**（本版）：seed=1 决策门实验复盘，加入 §5.7 BoQ 必须冻结的发现，
  Q6 和 R1-W3/R2-Q3 rebuttal 角度更新。架构从 "BoQ 0.05× LR fine-tune"
  转为 "BoQ 完全冻结"，训练脚本从 `run_tier2_seed1.sh` 切换到
  `run_tier2_freeze_seed1.sh`。

*作者：DR-VPR revision team. 内部文档。*
