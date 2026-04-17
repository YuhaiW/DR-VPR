# DR-VPR project — guide for Claude

## 协作原则（最高优先级）

请使用第一性原理思考。你不能总是假设我非常清楚自己想要什么和该怎么得到。请保持审慎，从原始需求和问题出发，如果动机和目标不清晰，停下来和我讨论。如果目标清晰但是路径不是最短，告诉我，并且建议更好的办法。

具体的落地方式：
- 用户给一个看似确定的指令时，先问自己"这个指令背后真正要解决的问题是什么"。如果指令路径和那个问题不对齐，先说出来。
- 当发现用户在承担假设的成本（比如花 6 小时重训你判断没必要的模型），要明确说出你的判断和依据，让用户可以否决。
- 不要把"执行用户字面要求"当成安全区。执行错误方向比建议换方向代价更大。
- 只有在用户确认路径后才动手；含糊或路径存疑时必须先对齐。

## 项目状态

- **目标**: Automation in Construction 期刊 major revision 返稿。
- **截稿**: 2026-04-25
- **期刊决定编辑**: Wen-der Yu, Ph.D.
- **Reviewer 1**: "can be accepted after revision"（4 条 weakness）
- **Reviewer 2**: constructive major revision（Q3–Q8 具体要求）
- **稿号**: AUTCON-D-25-06069
- **原始 decision letter 全文**: `doc/DECISION_LETTER.md`（两位 reviewer 逐题原话）
- **Rebuttal letter 草稿**: `doc/REBUTTAL_LETTER.md`（逐条回应所有 reviewer 语句）
- **节到节改动清单**: `REVISION_CHECKLIST.md`（每行关联一条 reviewer 意见）
- **原稿 tex**: `/home/yuhai/project/DRVPR_paper/Submitted Version 202512V2.tex`

## 方法栈

两套候选主方法（本分支 `equiboq-ablation` 同时存在）：

- **原 DR-VPR (MixVPR 变体)** — 对应原投稿: ResNet50 + MixVPR ‖ E2ResNet(C8) + GeM + attention fusion。descriptor 4608 dim。params ≈ 14.4M。Latency 4.23 ms @ RTX 5090。
- **Equi-BoQ 变体**（当前工作）: ResNet50 + BoQ ‖ E2ResNet(C8) + GeM + {concat | attention} fusion。
  - concat：descriptor 17408 dim；zero-init scalar gate on equi branch。
  - attention：descriptor 16384 dim；低秩 gate（proj1 16384→64、proj2 1024→64、score 128→2, softmax bias [2, 0]）。
  - 两者都 ≈ 26M params。

## 盘上已有实验数据

| 变体 | 3-seed 完整训练 | per-epoch ConPR eval | per-epoch ConSLAM eval | best-val 单点 eval |
|---|---|---|---|---|
| Equi-BoQ concat | ✓ `LOGS/resnet50_DualBranch_seed{1,42,190223}/` | ✓ `eval_seed*_ep*_conpr.log` | ✓ `eval_seed*_ep*_conslam.log` | ✓ |
| Equi-BoQ attention B2 bias=[10,0] | ✓ `LOGS/resnet50_DualBranch_attention_seed*/` | ⏳ `run_attention_per_epoch_eval.sh` 产出 `eval_attention_s*_ep*_*.log` | ⏳ 同上 | ✓ `eval_equiboq_attention_s*_con*.log` |
| Equi-BoQ attention B1 bias=[2,0] | ✓ `LOGS/resnet50_DualBranch_attention_b1_seed*/` | ⏳ `eval_attention_b1_s*_ep*_*.log` | ⏳ 同上 | ✓ `eval_equiboq_attention_b1_s*_con*.log` |
| 6 个 baseline（BoQ/SALAD/CricaVPR/MixVPR/CosPlace/DINOv2） | 单点推理 | — | — | ✓ `baseline_results.txt`、`baseline_eval_full.log` |

**checklist 里的"DR-VPR (ours) 79.92±0.87 / 60.30±0.72"的真实来源是 Equi-BoQ concat（按 ConPR-test best epoch 选点）**。这个命名/语义不一致要在正式 tex 里纠正。

## 经验教训（调试沉淀）

- **attention fusion 在这个模型里天然输给 concat**：softmax 是零和的，BoQ 预训练特征信号太强，优化器会把 w2 推向 0 来保护 BoQ。concat + zero-init scalar gate 则非零和，equi 分支可以在不牺牲 Branch 1 的前提下叠加。B2 和 B1 两次诊断都实锤了这个结论。
- **attention 的 softmax bias 不能太饱和**：bias=[10,0] 下 w2 初值 ≈ 4.5×10⁻⁵，训 10 epoch 也推不动；bias=[2,0] 下 w2 初值 0.119，但优化器仍会通过学 `score.weight` 把 logits 差推回 −10 以上，终究还是把 equi 关掉。机制层面这条路走不通。
- **BoQ 预训练权重在 ConSLAM 上已近最优**：per-epoch eval 显示 BoQ-based 模型在 ep0（几乎没有 GSV-Cities finetune）常常就是 ConSLAM R@1 最高点，后续 finetune 轻微损害。这是个值得进 paper 讨论的 insight。

## 关键命令 / 脚本

- 训练：`RUN_TAG=<tag> mamba run -n drvpr python train_fusion.py --seed <N>`（默认 `fusion_method='attention'`，要 concat 就改源码 L477）
- 评测：`DRVPR_CKPT=<path> mamba run -n drvpr python test_conpr.py`（或 `test_conslam.py`）
  - 已知 bug：`test_*.py` 在打印 Average Recall 之后会因 `recalls[i]` 被意外构造成 dict 而抛 TypeError。**平均 recall 已写入 log，结果仍可用**，不要被 "!! eval failed" 误报误导。
- 批跑：
  - `run_equiboq_seeds.sh` — 3 seed 训练 + best-val eval
  - `run_equiboq_per_epoch.sh` — 3 seed 训练 + 30 ckpts 全 eval（concat 的历史数据来源）
  - `run_attention_per_epoch_eval.sh` — 仅 eval，不训练，覆盖 B1+B2 共 60 ckpts

## 命名规范（强制）

任何新 checkpoint / eval log 必须带变体标签前缀，不得覆盖历史产物。
- 训练结果：`LOGS/resnet50_DualBranch_<TAG>_seed<N>/`（`<TAG>` 示例：`concat`、`attention`、`attention_b1`）
- 训练日志：`train_equiboq_<TAG>_s<N>.log`
- 评测日志：`eval_<TAG>_s<N>_ep<EE>_conpr.log` / `..._conslam.log`（per-epoch）或 `eval_equiboq_<TAG>_s<N>_conpr.log`（best-val）

## Paper 返稿 checklist 里的 open questions

见 `REVISION_CHECKLIST.md` 末尾"Open questions"——5 条已在对话里全部答过（见 `memory/project_revision_decisions.md`）。
