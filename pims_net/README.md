# PIMS-Net 研究目录

本目录独立承载后续 PIMS-Net 研究，复用当前 EfficientViT-SAM 源码、依赖和本地预训练权重。
当前 PIMS-Net 第一击使用冻结的官方 EfficientViT-SAM-L0，第二击起使用空间记忆、EditIntentDecoder 和 MaskEditor 迭代编辑。
`SpatialStateMemory` 融合图像、点击与上一轮 Mask，维护独立的 `[B,128,64,64]` 图像会话状态。
已提供可微训练步骤和缓存交互推理接口，见 [循环模型接口](docs/recurrent_pipeline.md)。
闭环训练脚本 `train_interactive.py` 支持最多 5 次在线点击与四项损失，见 [交互训练说明](docs/interactive_training.md)。
`train_ablations.py` 独立训练 A1/A2/A3，`evaluate_interactive.py` 统一对比 A0–A3，
输出 IoU@1/3/5、NoC@90、Boundary IoU 和正确区域破坏率，见 [消融与评估说明](docs/unified_evaluation.md)。
原 A1–A3 已完成 COCO 子集 seed 42 实验。独立 Directional 分支已完成16实例过拟合、
容量/损失消融及 PositiveCorrectionGate 校准；这些训练集探索不能作为COCO泛化性能。
公开版本的环境、数据及权重准备见 [GitHub使用说明](docs/github_setup.md)，
最新轻量统计见 [Gate校准结果](results/gate_calibration16_20261019/README.md)。

```text
pims_net/
├── models/       # Baseline 加载入口；后续研究网络
├── data/         # 数据适配、变换、采样及划分说明
├── training/     # 研究训练循环与损失；Baseline 训练说明
├── inference/    # 缓存图像特征、点击历史和循环状态的交互推理
├── evaluation/   # 指标、评估与结果分析
├── tests/        # 资源及 Baseline 完整性检查
├── configs/      # 独立实验配置
├── outputs/      # 研究权重、日志及结果
├── results/      # 可公开的轻量实验统计，不含数据与权重
├── docs/         # 工程检查记录、原始文件 SHA-256 清单
├── paths.py      # 共享资源定位，不依赖当前工作目录
└── requirements.txt  # 引用根目录依赖，不维护第二套版本
```

## 使用

复用已有 Python 环境，不需要新建环境或重复安装依赖。从工程根目录运行：

```bash
python -B -m pims_net.tests.smoke
python -B -m pims_net.tests.smoke --verify-baseline
# 依赖可用后，再执行实际权重加载检查：
python -B -m pims_net.tests.smoke --load-model
```

```python
from pims_net.models import build_baseline

model = build_baseline("efficientvit-sam-l0").eval()
# 按实验需求显式选择 model.to("cuda") 或 CPU。
```

默认 L0 仅作为较小的接入示例；完整权重支持 L0/L1/L2/XL0/XL1。
训练与评估操作见对应目录 README。
冻结编码器的输入约定、用法和真实权重测试见 [模型说明](models/README.md)。
`requirements.txt` 仅转引原始依赖；当前环境无需因此执行安装，且原依赖清单不代表完整可运行保证。

## 独立 DirectionalMaskEditor 实验

通过 `models.directional_pims_net.DirectionalPIMSNet` 使用新分支，原 `PIMSNet(variant="a3")`、
A3 损失、训练脚本和 checkpoint 保持不变。新分支仍用冻结 EV-SAM 首击与 `[B,128,64,64]` ConvGRU；
复用原 decoder 的结构与尺寸处理，但所有研究参数独立初始化，不加载 A3 研究权重。
新 head 输出两个 `[B,1,256,256]` 张量：`D=tanh(direction_logits)`、`S=sigmoid(strength_logits)`。

```text
P_new = P_old + S * where(D >= 0, D * (1 - P_old), D * P_old)
```

正方向增加前景概率，负方向减少，剩余到 0/1 的距离限制更新量，无需对 P_new 硬截断。
`D=0` 或 `S=0` 时严格保留旧概率及原始 logits，包括 ±100 的极端 logits；
隐藏状态和点击仍推进。新图像重新 initialize，记忆归零。非 KEEP 的概率转 logits 沿用 eps=1e-6。
严格 KEEP 的 logits 分支在恰好零方向处不向 D/S 传递 Mask Loss 梯度，方向监督仍能从零方向学习；
不使用直通梯度。head 权重小幅非零初始化，方向 bias=0、strength bias=-2。

方向标签来自 detached 旧 Mask：FN→+1、FP→−1、TP/TN→0。
为保持与 A3 的训练协议一致，继续使用旧概率 `>0.49`，不同时修改已知的训练/评估阈值差异。
方向损失为 SmoothL1(beta=1)，各个存在的 -1/0/+1 类别先求均值再等权平均；排除 padding。
总损失默认 `BCE + Dice + DirectionLoss + 0.1*Preserve`，通过 `--direction-weight` 调整方向监督。
Strength 不增加单独 GT 标签，通过 Mask/Preserve Loss 学习。完整保留跨轮 Mask/隐藏状态 BPTT。

同原 seed 42 的数据与训练预算，独立训练命令如下（本次没有启动该完整实验）：

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.train_directional `
  --manifest pims_net/outputs/coco_subset_20261017/data/train.jsonl `
  --output pims_net/outputs/directional_seed42_full `
  --epochs 5 --batch-size 2 --max-clicks 5 --seed 42 `
  --lr 0.0001 --weight-decay 0.0001 --grad-clip 1 --save-every 5 --device cuda
```

输出 config.json、metrics.jsonl（分项 direction 损失）、train.log、last.pt 和周期权重。
同原训练按每图 2–5 轮均值监督，再按批内实际训练图数平均梯度；RITM 种子、取点、AdamW 和裁剪相同。
checkpoint 标记 `variant=directional`，与 A3 双向加载会被拒绝；非空输出目录拒绝覆盖。
`--max-steps 1` 可只做一个批次的验证，结果会标为 `smoke_test_not_a_benchmark`。

缓存推理与现有评估协议兼容，不需要 GT 来执行新模型：

```python
import torch
from pims_net.models.directional_pims_net import DirectionalPIMSNet
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.evaluation.methods import PIMSSession

model = DirectionalPIMSNet().cuda().eval()
load_training_checkpoint("pims_net/outputs/directional_seed42_full/last.pt", model)
features, sizes = model.sam_initializer.encode(rgb_uint8_image)
session = PIMSSession(model, features, sizes)
probability = session.add_click(x, y, label)  # Original-image coordinates, labels 1/0.
# session.state.direction / strength are available after click 2.
# evaluate_session(session, gt_mask, EvaluationProtocol()) reuses the existing metric/click protocol.
```

现有 `evaluate_interactive.py` 仍只评估 A0–A3；不会把方向分支的未充分训练权重自动加入原正式表。
`PIMSSession.diagnostic_snapshot` 的旧三通道 intents 为 None；需要方向图时读取上述 direction/strength 字段。

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m unittest pims_net.tests.test_directional -v
```

13 项专项测试覆盖严格 KEEP、更新符号/边界、有限差分、方向标签/平衡权重/padding、首击一致、
五轮完整 BPTT、两个 head 通道与记忆梯度、独立图像重置及 checkpoint 隔离。
本次真实 COCO 训练验证只做 2 图各 5 击、1 次优化更新，输出位于
`outputs/directional_implementation_20261018/smoke_training/`，不能据此报告新分支性能。

## 固定 16 实例过拟合诊断（统一 0.5）

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.overfit_directional `
  --output pims_net/outputs/directional_overfit16_new_run --epochs 100 --eval-every 5
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluation.directional_overfit_report `
  pims_net/outputs/directional_overfit16_new_run
```

默认从已有 COCO 1024 训练清单用 seed 42 无放回抽取 16 个实例并保存 `fixed_train16.jsonl`；
从头初始化 Directional 研究参数，冻结官方 SAM，仅缓存图像特征及固定首击。
批量 2、学习率 1e-4、最大 100 epoch（800 次更新）；每 5 epoch 固定权重复测同一训练集。
每对象的 RNG 在各 rollout 重置以固定首击，后续点击仍由当前错误闭环产生。
这是训练集记忆能力诊断，不能作为泛化性能。

取点、方向监督、Preserve 正确区域及原图/原生 IoU 全部使用严格 `P>0.5`。
RITM 接收的是先按 0.5 二值化的输入，因此其内部 0.49 比较不会改变连续概率决策。
`directional_losses(..., threshold=.5)` 同时控制方向标签和 Preserve 区域；其默认仍为 .49，
保持原方向分支旧训练命令兼容，A3 的损失/取点函数不作修改。

`rounds.jsonl` 保存每对象每轮的 IoU、各项 Loss、Direction/Strength 在 all/TP/TN/FP/FN
区域的均值、标准差、分位数和直方图，以及 Mask/Direction 对双通道原始 logits 的梯度余弦。
`sequences.jsonl` 保存整段交互的 full-BPTT 参数梯度余弦与范数（all/memory/head），
`epochs.jsonl` 区分边训练边更新的统计与固定权重 eval 统计。所有梯度统计在裁剪/AdamW 前完成；
Mask=BCE+Dice，Direction 乘其损失权重，比较不包含 Preserve。零范数余弦为 null。
首击冻结，没有编辑损失或方向输出。无错误时不虚构点击，IoU@3/5 延用最后结果。

每个 epoch 保存独立 `last.pt`，按训练集 IoU@5 保存 `best.pt`；拒绝覆盖已有目录。
预设连续三次复测原图平均 IoU@5≥0.95 则提前停止，否则报告预算内是否未充分过拟合。
生成报告时独立核对完整 16 实例覆盖、均值、直方图计数，并导出逐轮 CSV 和图表。
原 A3 源码、已有 A3 权重及官方预训练权重在运行前后做哈希核对。

## 固定16实例的 Strength 倍率推理诊断

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluate_directional_strength `
  --reference pims_net/outputs/directional_overfit16_seed42_20261019 `
  --output pims_net/outputs/directional_strength_new_run
```

加载该次训练的 `last.pt`，使用最终 epoch 评估记录中的同一16对象五击序列，
比较 `S_effective=min(S_predicted*k,1)`，`k=1/2/4/8`。所有判断使用 `P>0.5`；
首击相同，每组独立反馈其修改后的 Mask/logits 和循环状态。无梯度、无优化器、无权重写入。
倍率1须精确复现原始80轮IoU；运行前后检查模型源码、原A3及现有权重哈希。

`comparison.csv` 保存 IoU@1/3/5 及首击到第五击的 FN修复/FP删除/正确区域破坏。
IoU为对象宏平均，后三者为原图像素汇总比率，分母分别是参考Mask的FN、FP和正确像素。
`summary.csv` 同时提供每轮相对首击、相对上一轮的统计；`per_object.jsonl` 保存原始计数。
`distributions.json/csv` 给出每轮原生有效区域 all/TP/TN/FP/FN 的 Direction、预测Strength、
有效Strength分布及饱和比例；分区根据各组上一轮Mask定义，排除padding。
该结果仅描述训练16实例的固定点击敏感性，不作为泛化性能。

## Decoder容量独立训练消融

`models/directional_wide.py` 保持 DirectionalPIMSNet 的首击、ConvGRU、点击编码、
Direction/Strength 激活及 MaskEditor 不变，仅把 Decoder 的隐藏通道
`64/32/16` 扩为 `128/64/32`，在64×64层增加两个64通道残差块。
每块为 `x + Conv3x3(GELU(Conv3x3(x)))`，不增加归一化或改变损失。
独立 checkpoint variant 为 `directional_wide`，与原权重不混用。

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.overfit_directional_wide `
  --output pims_net/outputs/directional_wide_new_run
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluation.directional_capacity_report `
  --reference pims_net/outputs/directional_overfit16_seed42_20261019 `
  --directory pims_net/outputs/directional_wide_new_run
```

复制原训练16清单，保持顺序与数据完全相同；复用原rollout和损失函数，seed42从头训练，
ConvGRU初始参数与原模型完全一致。保持100 epoch/800更新、batch2及原优化器超参数，
不提前结束预算。取点使用相同自适应RITM协议与对象RNG；后续坐标随模型预测变化。
比较固定权重eval结果的最终/最佳IoU、全部分项Loss、逐轮指标及梯度余弦，
报告区分相同预算最终值与训练集最佳值。仅衡量该16对象的训练集拟合能力。
原Directional/A3源码、损失和已有权重在训练前后做哈希核验；best/last均回载复现逐轮IoU。

### 第二级扩容：256/128/64与4个残差块

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.overfit_directional_large `
  --output pims_net/outputs/directional_large_new_run
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluation.directional_capacity_ladder `
  --directory pims_net/outputs/directional_large_new_run
```

新增独立 `LargeDirectionalPIMSNet`，variant=`directional_large`；Decoder隐藏通道为
256/128/64，在64×64层放置4个128通道Residual Block。保留原Direction/Strength激活、
MaskEditor、ConvGRU结构及全部损失，不叠加符号或Gaussian推理约束。
采用相同训练16清单、seed42、100epoch/800更新和优化器配置，从头训练。
新Decoder参数1,648,450，全部研究参数2,220,386；原两版权重不覆盖。
三级报告同时展示最终和最佳检查点，以及回载后的首击到第五击FN修复、FP删除、TP破坏；
各组遵循同一自适应RITM协议，后续点击坐标可不同。

## 扩容模型C0–C3约束推理消融

下述为初版定义；用户修订后的正式对比入口见后面的“修订版”小节，历史结果保留。

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluate_directional_constraints `
  --output pims_net/outputs/directional_constraints_new_run
```

加载扩容版已有 `last.pt`，固定其最终评估的16实例五击序列，阈值为 `P>0.5`。
C0原样推理；C1按最新点击符号截断Direction（正击D≥0、负击D≤0）；
C2仅以最新点击Gaussian乘Strength；C3同时施加两种约束。
Gaussian沿用64网格ClickEncoder的σ=2，双线性插值到256，排除padding，
不重新归一化峰值、不硬截断。模型内部仍接收累积点击，每组独立反馈其状态。
约束在独立评估脚本内实施，不修改模型源码或权重，不训练。

另报 `local_gt_oracle`：`P_new=P_old+G*(GT-P_old)`，与其余分组使用相同首击和点击。
这是GT辅助软Gaussian诊断，不是正式模型性能，也不是原图IoU的严格数学上界。
对于非C0分组，P数值未变化处保留原logits，避免端点裁剪造成非预期的远处编辑。

主表报告IoU@1/3/5，以及首击到第五击的FN修复率、FP删除率、TP破坏率。
分母分别为首击FN、FP、TP，TP不包含背景TN；三项按原图像素汇总。
`summary.csv`另存每轮相对上一轮及原生网格的计数与比率；`per_object.jsonl`可复核逐实例。
C0逐轮IoU须精确复现，Oracle原生正确像素不得破坏，符号约束及Gaussian外KEEP均有检查。

### 修订版：方向幅度保留、σ扫描与GT错误连通域

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluate_directional_constraints_v2 `
  --output pims_net/outputs/directional_constraints_v2_new_run
```

C1改用 `D_effective=(2*label-1)*abs(D_raw)`，翻转不一致符号并保留幅度，
不再将其裁剪到零。C2/C3分别测试外部Gaussian的σ=2、4、8（64网格）；
模型内部ClickEncoder仍为σ=2，所有组沿用初版固定16实例五击序列。
C0/C1各一组，C2/C3各三组，Oracle另三组，共880轮。

Oracle先按各自上一轮Mask（P>0.5）选择最新点击所在的同符号GT错误8连通域：
正击选择FN、负击选择FP。点击按官方1024变换/4并取最近原生像素，
不吸附到其他错误区域。仅执行 `P_new=P_old+G*component*(GT-P_old)`；
无对应错误时Mask及logits严格KEEP，点击与隐藏状态仍推进。Oracle仅作GT辅助诊断。
`per_object.jsonl`记录各轮连通域大小、点击原生坐标和是否跳过；
`verification.json`核对空域KEEP、原生正确区域零破坏、C0回放和C2 σ=2与旧结果一致。
各项指标分母及聚合方式沿用初版，原模型源码和所有已有权重均不修改。

## 更大版epoch95正负点击连通域诊断

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.diagnose_large_clicks `
  --output pims_net/outputs/large_epoch95_click_diagnostic_new_run
```

加载更大版 `best.pt`（epoch95/760更新），回放该检查点原有16实例五击序列。
不干预实际模型状态；每次编辑前选择当前点击的同符号GT错误8连通域（正FN、负FP）。
若原图点击映射到原生网格后已正确，则记录空域，不吸附到其他错误。
保存域内Direction、Strength、实际ΔP、阈值跨越率以及半径8邻域、其他错误/正确区域、
域内距离分层的分布与更新量。汇总按像素×事件，另提供非空事件宏平均。

单轮旁路反事实包括：纠正符号保留幅度、S=1保留D、|D|=1保留符号和S、
纠正符号且S=1、GT单位方向保留S；均不反馈下一轮。
域内互斥分类为已跨越、方向错误/零、正确方向但S不足、正确方向且即使S=1仍因D幅度
不足而不能跨越。最后一类可能同时存在S不足，不应解读为S已经充分。
空间分析区分域外其他错误与正确区域；累积点击模型的域外编辑不自动视为定位错误。
原生 `P_old/P_new/D/S/GT/valid/component` 存于 `native/`，可复核统计与代表案例。

## 更大版 L0/L1/L2 局部损失消融

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.overfit_directional_losses `
  --variant L1 --output pims_net/outputs/new_loss_experiment/L1
# 分别用 L0、L1、L2 运行；每组默认100epoch/800更新、seed42、同一16实例。
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluation.directional_loss_report `
  --directory pims_net/outputs/new_loss_experiment
```

保持 `LargeDirectionalPIMSNet` 原架构、完整前向与状态更新，三组从相同随机初始化独立训练。
L0完整复用原损失；L1仅在当前点击所在同符号GT错误8连通域内计算BCE+Dice，
在有效域内的连通域外计算Preserve，不计算Direction监督；L2增加该连通域内的
Direction SmoothL1监督（正击+1、负击−1）。权重沿用1/1/1/0.1，阈值统一0.5。
连通域基于编辑前的detach概率，坐标映射与epoch95诊断一致；空域不吸附，
此时局部Mask/Direction损失为0，Preserve覆盖整个有效域。

每组使用相同RITM自适应协议、数据顺序和训练预算；后续点击坐标可不同。
所有推理均无GT区域门控。独立保存best/last权重；主表比较800更新最终权重，
另报预算内训练集最佳检查点。报告记录逐击IoU、FN修复、FP删除、TP/TN破坏，
并提供相对首击、相对上一击两种编辑计数。
梯度诊断包括Mask–Direction、Mask–Preserve、Direction–Preserve的加权梯度余弦，
记录每轮raw head以及整条序列全BPTT参数（all/memory/head）；不存在的损失及零范数
余弦为null，排除冲突比例分母。各组模型源码及历史权重保持不变。

## Strength塌缩及梯度归因诊断

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.diagnose_strength_gradients `
  --output pims_net/outputs/new_strength_gradient_audit
```

主检查点为更大版原epoch95 best，另检查L0/L1/L2损失消融的800更新最终权重。
各自回放原16实例五击记录，按正负点击分别统计Strength的原始logits、Sigmoid导数、
当前错误连通域/FN/FP/正确区域/点击邻域中的Strength和各项损失的梯度。
同时核对概率logit裁剪的零梯度及`gz=gS*S*(1-S)`，避免把小Sigmoid导数误判为BCE梯度死亡。

模型始终eval，只使用autograd.grad，不构造优化器、不调用backward或更新参数。
记录当前轮损失和整条序列平均损失对每轮S/logits的梯度，后者包含未来损失反馈；
参数梯度按Strength Head第1行、Direction Head第0行、共享trunk/ConvGRU分组。
逐项BCE、Dice、Direction、加权Preserve的范数、总梯度投影及两两余弦均保存。
额外通过当前Strength通道VJP比较点击错误域与正确区域的Mask梯度，
该路径分解排除当前Direction和概率旁路，不等于完整Mask参数梯度。
所有原始图与梯度存于native/，表格见region_statistics、parameter_*、sequence_output_*。
检查点及整个模型state_dict哈希保持一致，逐轮IoU回放精确一致。

## S0/S1/S2 Strength辅助监督独立训练

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.overfit_strength_supervision `
  --variant S2 --output pims_net/outputs/new_strength_supervision/S2
# S0/S1/S2分别运行，默认同一16实例、seed42、100epoch/800次更新。
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluation.strength_supervision_report `
  --directory pims_net/outputs/new_strength_supervision
```

保持大版 `LargeDirectionalPIMSNet` 和原L0全部损失不变，从相同研究参数初始化独立训练。
S0为原L0；S1增加0.05权重的 `BCEWithLogits(z_strength,1)`，仅在当前正点击所在FN连通域内计算。
S2保留S1，并以0.01权重监督有害正确区域的Strength为0：旧预测TP且本轮Direction<0，
或旧预测TN且本轮Direction>0。筛选条件全部detach，不抑制其他未修复FN/FP。
激活、抑制各自在选中区域取均值，空区域跳过；使用原始Strength logits而非激活概率。

阈值0.5、冻结SAM首击、batch2、lr1e-4、weight decay1e-4、clip1和RITM协议均沿用。
训练时每对象完整BPTT，记录所有加权目标在Strength Head、共享参数、memory/head及全参数上
的梯度范数、总梯度投影及两两余弦；每轮记录输出梯度与损失，评估另存目标、FN/FP、
正确及有害区域的S/logit分布。未启用或零范数目标的余弦为null，不当作0冲突。

主表比较800更新最终权重，最佳检查点单列；记录IoU、FN修复、FP删除、TP/TN破坏，
各组使用同一自适应点击协议但后续坐标可不同。最终和最佳权重重新回放验证，
原始概率图、Strength、logits、Direction、GT和监督区域保存在报告目录native/。
`strength_supervision_smoke_S2_20261019` 是调整抑制定义前的旧冒烟结果，已标记EXCLUDED，
不纳入正式实验。正式定义仅使用用户指定的0.01有害正确区域抑制。

## PositiveCorrectionGate：冻结S0的独立正点击纠错

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m unittest pims_net.tests.test_positive_gate -v
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.overfit_positive_gate `
  --output pims_net/outputs/new_positive_gate
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluation.positive_gate_report `
  --output pims_net/outputs/new_positive_gate
```

`models/positive_correction_gate.py` 是独立分支，不改原DirectionalPIMSNet。
加载 `strength_supervision16_seed42_20261019/S0/best.pt`，整个基础网络eval、requires_grad=False，
每次基础转移在no_grad中运行。首击仍为冻结EV-SAM；第2–5击正点击执行
`P=Pbase+sigmoid(z_gate)*(1-Pbase)`，负点击逐位返回基础概率和Logits。
输入为256通道图像特征、最新正负点击Gaussian、128通道当前S0隐藏状态和基础Mask。
概率转Logits时保留正向单调性，避免接近1的概率裁剪造成Logits下降。

纠正后的Mask反馈给下一轮冻结S0，按新图重置状态；Gate梯度在冻结转移处截断。
监督使用当前点击在S0本轮输出后仍遗漏的FN连通域（原生256网格、8连通、无吸附），
其他有效像素为负；点击已正确则全负。只训练正点击，每个存在的正/负区域分别取
BCEWithLogits均值，再平均；G1仅Gate监督，G2额外全有效区域BCE和Dice，各项权重1。
两组加载同一Gate初值、固定16实例、seed42、batch2、100epoch/800更新，独立保存Gate专属
best/last检查点，不复制或保存冻结S0参数。主比较使用完整预算last，最佳快照单列并显示预算。

独立推理先创建 `LargeDirectionalPIMSNet` 并用 `load_training_checkpoint` 加载S0，
包装成 `PositiveCorrectionPIMSNet(base)` 后，以 `overfit_positive_gate.load_gate` 加载Gate专属权重；
接口为 `initialize(image, points, labels)` 和 `refine(state, points, labels, enabled=True)`。
关闭Gate的 `enabled=False` 用于G0对照。推理模块不读取GT；目标连通域仅在训练与诊断中计算。

报告包含S0固定五击Replay和各自RITM自适应五击（统一0.5阈值），逐击IoU、FN修复、FP删除、
TP破坏、TN误加、正负点击单独Gate效果及Gate/Logit/ΔP分布；native/保存480份逐轮原始数组。
额外记录正点击Gate≤0.01、≥0.99饱和率及Sigmoid导数<1e-4比例，以及每epoch的8个batch
裁剪前Gate总梯度L2均值。负点击强制0不计为学习塌缩。单元测试显式验证train/eval反馈一致、
冻结S0无梯度、未来轮损失不会跨过no_grad转移反传到早期Gate。
先通过真实16实例单轮梯度/冻结检查和G0的80个IoU回放，再训练。正式本次结果为
`positive_gate16_v2_seed42_20261019`；早期无Logits单调保护的中断短跑目录已标记EXCLUDED。
所有结果仅为16个训练实例的过拟合实验，不代表COCO泛化性能。

## G2 Gate单轮贡献与推理校准（不训练）

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m unittest pims_net.tests.test_gate_calibration -v
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluate_gate_calibration `
  --output pims_net/outputs/new_gate_calibration
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluation.gate_calibration_report `
  --output pims_net/outputs/new_gate_calibration
```

仅加载 `positive_gate16_v2_seed42_20261019/G2/last.pt`（100epoch/800步），
不使用epoch0的best。原模型及权重完全不变；新增独立推理脚本，整个过程使用inference_mode。
测试α={0,.1,.25,.5,.75,1}、τ={0,.3,.5,.7}的24种组合，采用
`g_eff=α*g*1[g>=τ]`，先按原始Gate阈值筛选，再缩放，保留通过阈值的幅度。
首击EV-SAM不变，负点击直通；每个校准条件反馈自身Mask及原S0记忆。
校准函数没有GT或GT空间支持参数。GT仅用于相同seed42的RITM点击模拟及事后指标。

每次正点击仅调用一次冻结S0，提取同一历史/点击/更新后记忆对应的P_base，
与Gate融合的P_fused比较。原图计数K=FN→TP、E=TN→FP、ΔIoU和K/E，
另存256有效网格统计。E=0,K>0为∞；K=E=0未定义，不添加epsilon。
按正点击轮次、实例池化K/E；事件统计可重复计入同一像素，区别于首击到第五击累计变化。

固定Replay沿用S0已有五击；自适应RITM各组使用自身Mask。α=0的每轮概率、原始Logits、
ConvGRU隐藏状态、点击历史必须与独立纯S0/G0逐位一致；α=1,τ=0逐位复现已有G2。
输出48组指标、3840条逐轮记录、768条实例汇总及每次正点击贡献；native/保存基础/融合
概率、原Gate及压缩的原图基础/融合二值Mask，instances/保存图像及GT，足以独立复核K/E。
报告检查完整更新公式、原图/原生网格计数、ΔIoU代数关系和旧权重哈希，
并包含完整表格、校准网格、倍率曲线、K/E散点和单轮最好/最差案例。
最优校准参数仅依据这16个训练实例的结果选出，不代表泛化性能。

## 隔离约定（原始工程文件）

所有研究代码与配置只在 `pims_net/` 内新增；原始 `efficientvit/`、`applications/`、
根目录依赖和 `assets/checkpoints/` 保持原样。直接引用现有权重，不复制、不下载、不覆盖。
后续网络修改通过本目录新模块实现，实验产物统一写入 `outputs/`。

当前目录没有 `.git`，因此使用 `docs/baseline_manifest.json` 记录建立研究目录时
213 个原始文件的 SHA-256（包括预训练权重，不含 Python 缓存），供检查原文件是否变化。
环境限制与发现的问题见 [工程检查记录](docs/inspection.md)。
