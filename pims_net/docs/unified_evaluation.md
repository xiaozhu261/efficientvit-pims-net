# A0–A3 独立训练消融与统一评估

复用现有 Conda `efficientvit` 环境：`E:/miniconda/envs/efficientvit/python.exe`，不是 uv。
源码、测试及实验产物均放在 `pims_net/`，不修改原始 EfficientViT-SAM 或共享权重。

## 四组正式定义

| 方法 | 第一击 | 第二击起 | 训练 |
| --- | --- | --- | --- |
| A0 原始 EV-SAM | 冻结官方 L0 | 累积正负点，输入上一轮选中候选的 256×256 logits 作为 mask prompt，每轮选择预测 IoU 最高候选 | 无 |
| A1 直接重预测 | 同一冻结官方 L0 | 保留完整 ConvGRU；DirectMaskDecoder 融合图像、隐藏状态和旧概率，直接输出完整 Mask logits | 独立训练 |
| A2 无历史记忆编辑 | 同一冻结官方 L0 | 每轮 ConvGRU 输入的历史隐藏状态清零；仍使用累积点击、旧 Mask 和当前计算出的状态，经 ADD/REMOVE/KEEP 编辑 | 独立训练，训练时同样清零历史 |
| A3 完整 PIMS-Net | 同一冻结官方 L0 | 保留历史隐藏状态，使用 ADD/REMOVE/KEEP 和 `P_new=KEEP*P_old+ADD` | 独立训练 |

A2 仅移除 ConvGRU 的历史状态连接，仍保留 Mask 反馈及点击历史，不能解释为完全无历史信息。
A1 的直接预测头与 intent decoder 复用相同融合主干，仅输出层为单通道 logits；不会调用旧 `InitialMaskDecoder`。
四组官方图像预处理、512 图像输入、1024 提示坐标系、原图后处理均一致，图像特征可以共用缓存。
每张新图像创建新的会话，清空点击、旧 Mask 和隐藏状态。

## 一次启动三组独立训练

JSONL 数据格式见 [交互训练说明](interactive_training.md)。从工程根目录执行，替换为真实训练清单：

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.train_ablations `
  --manifest 'G:/datasets/my_objects/train.jsonl' `
  --output pims_net/outputs/ablations_run_01 `
  --epochs 20 --batch-size 2 --max-clicks 5 --lr 0.0001 --device cuda --seed 42
```

脚本顺序启动 A1、A2、A3，各自重新初始化研究模块和 AdamW，各自重置相同随机种子，
使用相同数据顺序、RITM 训练采样规则及训练批次预算。点击采样根据各模型当前预测闭环生成，
后续坐标可以不同。不会继承其他变体已训练的研究参数。

默认 A1 损失为 `BCE + Dice + 0.1*Preserve`；A2/A3 为 `BCE + Dice + Edit + 0.1*Preserve`。
A1 没有编辑类别输出，Edit 项记为零，其余适用损失权重保持相同。
最多 5 击包含冻结首击，监督实际发生的第 2～5 击；首击完美的样本跳过训练。
相同预算指相同 epoch、样本批次和 `--max-steps`，实际更新次数也记录在 checkpoint 中。

产物分别位于 `ablations_run_01/a1/`、`a2/`、`a3/`，每组包含 `last.pt`、epoch 快照和训练日志。
也可单独调用 `pims_net.train_interactive --variant a1|a2|a3`，其余参数保持一致。
加载推理模型时必须选择相符变体，例如 `PIMSNet(variant="a1")` 后调用 `load_training_checkpoint`。

## 统一评估

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluate_interactive `
  --manifest 'G:/datasets/my_objects/val.jsonl' `
  --a1-checkpoint pims_net/outputs/ablations_run_01/a1/last.pt `
  --a2-checkpoint pims_net/outputs/ablations_run_01/a2/last.pt `
  --a3-checkpoint pims_net/outputs/ablations_run_01/a3/last.pt `
  --output pims_net/outputs/evaluation_run_01 --device cuda --max-clicks 20
```

三个 checkpoint 均为必填，不允许缺失权重时使用随机网络。
评估检查版本、变体、至少一次优化更新、相同训练数据清单路径与 SHA-256、样本数、
冻结 L0 权重 SHA-256、随机种子、训练点击协议、学习率、优化配置、损失权重、训练预算和已处理批次数。
不一致时立即报错；A2 与 A3 即使参数键完全相同也不允许混用。
清单哈希只覆盖 JSONL 内容，不包含图像文件内容；正式实验需固定不可变的数据版本和训练/验证划分。
旧版本 A3 checkpoint 可用于原有推理接口，但缺少完整消融来源信息，不能直接用于此正式对比。

### 相同交互协议

- 对同一清单中每个对象，四组按同一顺序评估；第一次从空 Mask 取点，因此首击相同。
- 在原图分辨率以 `probability > 0.5` 二值化。训练采样阈值保持 0.49，评估统一使用 0.5。
- 根据当前 FN/FP 计算外补一圈零的精确 L2 距离变换，选最大距离更大的错误类别及其最深像素。
  FN/FP 最大值相等时选负点，同类并列取行优先坐标；排除已经点击的像素。
- 各组下一击依据自己的当前误差产生。统一的是选点规则，不强制使用与自身误差无关的共同后续坐标。
- 最多点击 `--max-clicks` 次，默认 20、至少 5。达到 90% 后仍继续测到第 5 击；无剩余可点击错误时提前停止，
  后续固定轮次指标沿用最后结果，不伪造点击。如果第 5 击仍未曾达到 90%，继续到首次达到或预算耗尽。

### 指标精确定义

所有统计使用原图大小的二值 Mask，逐对象计算，再对同一对象清单做宏平均；比率均为 0–1。

| 输出列 | 定义 |
| --- | --- |
| `IoU@1` / `IoU@3` / `IoU@5` | 指定点击数的前景交并比；空并集记为 1 |
| `NoC@90` | 首次 IoU ≥ 0.90 的实际点击数；未达到者记为最大点击预算 |
| `failure_rate@90` | 预算内未达到 0.90 的对象比例，避免把失败与最后一击成功混淆 |
| `BoundaryIoU@5` | 第 5 击的边界区域 IoU，带宽为原图对角线的 2%，四舍五入且至少 1 像素 |
| `correct_region_damage_rate@2:5` | 第 2～5 击中“上轮正确、本轮错误”的像素数之和 / 上轮正确像素数之和；分母为零时记 0 |

边界带采用二值 Mask 内部腐蚀差集，图像边缘外补零，遵循
[Boundary IoU 官方实现](https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py)。
正确区域破坏率同时统计前景和背景；先按每对象实际交互的像素数加权，再跨对象宏平均。
没有发生新交互的提前停止轮次不添加虚构分母。该指标衡量二值正确性丢失，与训练中概率 L1 Preserve Loss 不同。

## 结果文件与验证

- `protocol.json`：方法定义、评估协议、训练配置、实际预算、数据清单及权重路径/哈希。
- `per_object.jsonl`：各对象各方法的汇总指标、每次点击坐标/正负标签/IoU/边界 IoU/破坏像素数和停止原因。
- `summary.csv`、`summary.json`：A0–A3 的统一指标对照表，包含失败率。

### 逐轮诊断

在上述评估命令末尾增加 `--diagnostics`，使用新的输出目录。仍使用同一组已训练权重、
相同选点规则和阈值，不增加模型前向、不修改 Mask 或隐藏状态。默认保存最多 20 击；
只诊断前五击时指定 `--max-clicks 5`。不同点击预算的 NoC 和失败率不可直接比较。

每个对象保存原图和 GT；每个实际发生的轮次保存：

- `mask.png`、`overlay.png`：原图分辨率二值 Mask，以及 TP/FN/FP、GT 边界和累积点击叠加图。
- `clicks.json`、`metrics.json`：点击原图坐标、正负标签、IoU、相邻轮次 ΔIoU、前景面积比例、Precision、Recall 和混淆矩阵计数。
- `ADD.png`、`REMOVE.png`、`KEEP.png`：A2/A3 第二击起的意图热力图，统一 VIRIDIS 色标 0–1。
  已归一化意图概率只沿用官方 resize/crop，不再 sigmoid。A0/A1 及首击没有这三个输出，不伪造热力图。
- `native.npz`：原生 256×256 float32 logits、概率、上一轮概率及实际存在的意图概率，供数值复查。
  `probability_u16.png` 为原图概率的量化预览，精确原生数据以 NPZ 为准。

`diagnostics/index.html` 是可离线打开的总览，实例按 A3 首击至第五击的退化程度排序，
进入实例后可切换 A0–A3 和点击轮次。GT 每个对象共用一份；仅保存真实交互轮次。
汇总表 `round_summary.csv` 沿用提前停止对象的最后 Mask，后续 ΔIoU 记零，不将其计入实际更新数。
`round_metrics.csv` 保存所有实际轮次，`intent_statistics.jsonl` 保存编辑条件统计。
`analysis.html`、`analysis.json` 和 `trends.png` 进一步汇总 A2/A3 的前景/背景破坏、
错误修复、正负点击满足率及 A2→A3 配对变化。已有诊断可独立重建报告，无需模型前向：

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluation.diagnostic_report `
  pims_net/outputs/evaluation_run_01
```

`native_confidence.jsonl` 记录原生 GT 前景/背景的概率均值及 [0.49,0.51] 附近比例，
排除 padding；报告对实际轮次且 GT 分组非空的对象宏平均，并记录对象数。
此区间用于识别置信度塌缩，不修改模型、损失或二值化阈值。
另在活跃编辑区（ADD+REMOVE>0.5）统计 |ADD−REMOVE| 均值及差值小于 0.02 的比例，
排除 padding，按具有活跃区的对象宏平均。ADD≈REMOVE 时，更新近似
`P_new=KEEP*P_old+(1-KEEP)/2`，可能将置信度推向 0.5；须结合真实统计判断。

前景面积比例为预测前景像素数 / 图像像素数，同时输出 GT 比例和预测 / GT 面积比。
Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，分母为零时记零；首击 ΔIoU 为 null。
逐轮汇总在同一对象全集上宏平均，比例采用 0–1。

为定位退化，诊断另计 `tp_lost`（正确前景被删除）、`tn_lost`（正确背景被添加为前景）、
`fn_fixed` 和 `fp_fixed`，以及最新点击是否满足。意图统计在原生 256 网格排除 padding，
按上轮 TP/TN/FN/FP 分组，报告 ADD/REMOVE/KEEP 和有效概率变化：
`effective_add=ADD*(1-P_old)`、`effective_remove=REMOVE*P_old`。
单看 KEEP 或全图正确区域破坏率容易遗漏小前景的损失，应结合 Recall、Precision 和这两项有效变化判断。
原生网格意图统计与原图指标不能直接逐像素等同；汇总的转换像素数包含多轮重复出现的像素。
这些是固定权重下的描述性证据，不能据此单独确定某项损失或 ConvGRU 的训练因果关系。

### 双向编辑竞争的离线诊断

已有 `--diagnostics` 输出可直接分析，无需加载模型或执行前向：

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluation.edit_competition `
  pims_net/outputs/coco_diagnostics_seed42_20261018 `
  --output pims_net/outputs/edit_competition_new_run
```

输出目录必须不存在。逐对象读取第 2–5 轮 `native.npz`，验证保存的上一轮概率与实际前一轮完全一致，
并核对 MaskEditor 更新公式和恒等式：
`P_new-0.5=q_keep*(P_old-0.5)+(q_add-q_remove)/2`。
分析前后校验研究模型源码、官方权重及三个独立训练 checkpoint 的 SHA-256。

`summary.csv/json` 包括每轮及第 2–5 轮合并的全图有效区、活跃区和旧 TP/TN/FP/FN 区域。
主表按像素次加权；JSON 另存非空对象区域宏平均。`per_object.jsonl` 保留逐对象原始统计，
`distributions.npz` 保留不采样的通道和概率差直方图。分位数输出宽度 0.005 的直方图区间，
均值与比例直接用原始 float32 数值、float64 累加。`report.html` 提供汇总与分布图。

- 活跃区：`q_add+q_remove>0.5`；双向共同激活：两者均大于 0.25。
- 近似平衡：`abs(q_add-q_remove)<0.02`，并提供 0.01、0.05 的敏感性统计。
- 竞争区：共同激活且近似平衡；以上是描述性诊断阈值，不是显著性检验。
- 向 0.5 收缩：`abs(P_new-0.5)<abs(P_old-0.5)-1e-6`，默认分母是所选区域全部像素。
- 有效抵消率：`sum(2*min(u,v))/sum(u+v)`，其中 `u=q_add*(1-P_old)`、`v=q_remove*P_old`。
  分母为零时返回 null。它与 q_add/q_remove 概率接近程度不同。
- `gt_error_delta` 为相对 GT 的概率绝对误差变化；正数表示变差。
  TP/TN 上收缩通常削弱正确置信度，FP/FN 上则可能帮助纠错，不能将收缩一概视为破坏。

### Edit Loss 监督与梯度审计（不更新参数）

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluation.edit_loss_audit `
  --reference pims_net/outputs/coco_diagnostics_seed42_20261018 `
  --output pims_net/outputs/edit_loss_audit_new_run --device cuda
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluation.edit_loss_report `
  pims_net/outputs/edit_loss_audit_new_run
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m unittest pims_net.tests.test_edit_loss_audit -v
```

读取已有 seed 42 A2/A3 checkpoint、实际训练损失配置及保存的各自评估点击，
全量重放第 2–5 轮，逐轮验证概率与原生诊断 NPZ 一致。
保持模型 eval，SAM 冻结，研究参数允许求导但仅使用 `autograd.grad`；
不构造优化器、不修改 `.grad`、不保存 checkpoint。校验模型内存 state_dict、
模型/训练源码及原始/研究权重前后哈希一致。

监督使用原损失的 0.49 前景阈值，另外统计改为评估阈值 0.5 时的标签分歧。
正/负最新点击附近取 SAM 1024 提示坐标除以 4 后、原生 256 网格上半径 8 的圆盘；
另存点击中心最近像素及历史同符号圆盘并集，排除 padding。
混淆矩阵行是真实 ADD/REMOVE/KEEP，列是预测 argmax；同时保存 q 均值，避免
微小概率差被 argmax 放大后误读。邻域不全是点击符号对应的编辑类别，GT 边界与旧预测共同决定标签。

Mask 梯度取实际加权 BCE+Dice；Edit 梯度取实际 edit 系数乘类别平衡 NLL。
局部梯度是本轮损失对本轮原始 intent logits 的导数；参数梯度是四轮平均损失的完整 BPTT，
按全部参数、memory、decoder_trunk、head 分组，报告每对象余弦/范数和全集平均梯度。
此处没有梯度裁剪或 AdamW，不能等同于优化器的实际更新方向。
当前 checkpoint 的 heldout 评估轨迹诊断也不能当作历史训练过程梯度：训练取点随机且阈值 0.49，
评估取点确定且阈值 0.5。A2/A3 使用同一对象集，但各自原评估点击序列不同。

输出 `report.html`、点击邻域混淆矩阵图、梯度冲突图，以及局部/损失/梯度 CSV、JSON 和逐对象 JSONL。
7 个专项测试验证实际标签与通道顺序、0.49 边界、类别平衡的解析梯度、缺失类别、padding、
有限差分、加权梯度线性关系、KEEP 可能与置信度提升冲突、点击几何和混淆矩阵汇总。

### GT Oracle 方向/位置分解（禁止作为正式模型性能）

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluate_oracle `
  --reference pims_net/outputs/coco_diagnostics_seed42_20261018 `
  --output pims_net/outputs/oracle_decomposition_new_run --device cuda
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m unittest pims_net.tests.test_oracle -v
```

使用现有 A3 权重，三组共用原 A3 保存的五击序列。各组第一击为同一个冻结 EV-SAM 输出；
后续独立维护 Mask 和 ConvGRU 状态，修改结果反馈下一轮。所有参数冻结，无优化器或训练。

- B1：定义总编辑强度 `m=q_ADD+q_REMOVE`，保留该连续空间强度图，用 GT 替换方向，
  `P_new=P_old+m*(GT-P_old)`。保留的是 m，而不是原始最终净增量 `abs(P_pred-P_old)`；
  也不是只交换 ADD/REMOVE 通道。没有对 m 阈值化，因此很小但非零的预测强度仍可利用 GT。
- B2：使用原生 256 网格旧概率 `P_old>0.5` 与 GT 的错误区域。正点击选择其所在 FN 八连通域，
  负点击选择其所在 FP 八连通域，仅在其中接受原始 A3 预测更新，其他位置逐位保留旧概率和原始 logits。
  原图点击经官方 1024 提示几何映射后除以 4、取最近原生像素。若该像素不再是同符号错误，
  Mask 不更新，但隐藏状态与累计点击仍推进。不寻找替代区域、不扩张、不重新取点。
- B3：原始 A3，不介入更新；逐轮核对旧诊断的概率与原图指标完全一致。

Oracle 不修改 padding。IoU@3/5 使用官方 logits 后处理回原图并以 sigmoid>0.5 二值化，
按相同对象集宏平均。原生区域保护不保证原图插值后绝对无损，另存原图正确区域破坏计数。
每轮 `edit_probabilities` 仍是网络原始输出，Oracle 实际应用的更新体现在存储的概率和 logits，
不应把该字段误当作经过 GT 介入的三通道概率。

输出独立目录下的 `comparison.csv/json`（IoU@3/5 及相对 B3 差异）、`summary.csv/json`、
`per_object.jsonl`、固定点击、协议、验证、HTML 报告与曲线图。所有表带
`run_type=oracle_diagnostic_not_model_performance`，明确标注 GT 推理介入，不写回正式评估表。
验证 B1 原生 GT 概率距离不增、B2 区域外概率/logits 不变、区域内等于原始更新、
原生正确像素无破坏，以及 B3 精确复现、模型/训练源码和原始/研究权重哈希不变。

B1/B2 分别提供方向替换与位置约束的敏感性证据；因反馈状态和后续预测随之改变，
差值不是可相加的纯因素贡献。固定原 A3 点击可能在 Oracle 分支变为陈旧点击，B2 会跳过相应更新；
报告该频率，不能把 Oracle 当作正式模型性能或可部署方法。

### 固定点击的 eta 推理对照

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.evaluate_eta `
  --reference pims_net/outputs/coco_diagnostics_seed42_20261018 `
  --output pims_net/outputs/eta_sweep_new_run --device cuda
```

读取该诊断评估的原 A3 权重、数据清单及前五击，固定 `eta={0,0.1,0.25,0.5,1.0}`。
第一击统一冻结 SAM，后续每组独立维护 ConvGRU 历史，实际执行
`P_scaled=P_old+eta*(P_edit-P_old)`，缩放后的概率和 logits 反馈给下一轮。
模型处于 eval/no_grad，参数全部禁用梯度，不构造优化器、不保存训练 checkpoint。
所有 eta 使用原 A3 的相同点击位置和正负标签，不重新根据各自误差取点。

eta=0 保留旧概率及原始 logits，避免纯“不更新”控制因再次 logit 截断而改变原图 Mask；
隐藏状态及点击历史仍继续更新。eta=1 原样使用现有 refine 输出，逐轮核对原评估指标和原生概率。
脚本还校验模型源码及权重哈希不变。需要 reference 含 `--diagnostics` 的原生 NPZ，且每对象至少有五击。

输出 `summary.csv/json`（各 eta、每轮的 IoU/Precision/Recall 等）、`per_object.jsonl`、
`fixed_clicks.json`、`protocol.json`、`verification.json` 以及 `report.html` 和趋势图。
0.5 聚集定义为概率落在闭区间 [0.49,0.51]，分别统计原生和原图的 GT 前景/背景/全图比例，
原生网格排除 padding，主结果采用同一对象全集的宏平均；GT 前景概率直方图另存 NPZ。
应将中间 eta 同时与 eta=0、eta=1 比较，区分减缓首击退化和实际纠错收益。
这是固定轨迹的敏感性实验，不等同于重新自适应选点后的方法性能，也不据此自动选择部署参数。

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m unittest `
  pims_net.tests.test_evaluation pims_net.tests.test_ablations pims_net.tests.test_diagnostics -v
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.tests.smoke --verify-baseline
```

功能冒烟使用 `outputs/interactive_smoke_data/train.jsonl` 的两张合成图，三组各训练一个批次后再评估。
评估增加 `--smoke-test --max-clicks 5`，结果标记为 `smoke_test_not_a_benchmark`。
`--smoke-test` 仅标记结果性质，仍严格要求三组独立训练权重及预算一致。
这些数据同时用于训练与测试，只能验证程序完整运行，不能作为模型性能结论。

最初的合成验证：43 项相关单元/集成测试通过；三组均对两张合成图执行 5 轮交互、各完成 1 次 AdamW 更新，
独立权重保存于 `outputs/ablation_smoke_training/{a1,a2,a3}/last.pt`。
四组评估保存于 `outputs/ablation_smoke_evaluation/`，共有 8 条对象/方法记录，首击坐标和首击 IoU 完全一致。
213 个原始文件 SHA-256 校验一致。随后已完成 COCO train2017 图像隔离子集的 seed 42 实验，
参见 [COCO 子集实验说明](coco_subset_experiment.md)；这不代表完整 COCO 训练或官方 val2017 评估。
上述命令退出码均为 0；环境仍会在退出时打印既有 wandb 临时目录清理权限错误，不影响已保存结果。
