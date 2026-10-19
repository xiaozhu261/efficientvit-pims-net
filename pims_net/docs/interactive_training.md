# RITM 闭环交互训练

入口：`pims_net/train_interactive.py`，支持 `python -m pims_net.train_interactive` 和直接脚本运行。
使用已有 Conda `efficientvit` 环境，无新增依赖。所有实验输出限定在 `pims_net/outputs/<run>/`。
默认 `--variant a3` 为完整 PIMS-Net；A1/A2 独立训练与统一评估见 [消融实验说明](unified_evaluation.md)。

## 数据清单

准备 JSONL 文件，每行一对 RGB 图像和二值实例掩码；相对路径按清单所在目录解析：

```json
{"id":"object_001","image":"images/001.jpg","mask":"masks/001.png"}
{"id":"object_002","image":"images/002.jpg","mask":"masks/002.png"}
```

掩码须为单通道 0/1 或 0/255，与原图同尺寸且前景非空。多实例 ID 图须先拆成单实例掩码。
当前不接受 ignore label、类别 ID 图或空目标。图像保留原分辨率，模型使用官方 SAM transform。
监督掩码按最长边 1024 缩放、右下补零，再最近邻缩小至 256×256；同时生成有效区域，所有损失排除补边。

## 训练命令

在工程根目录运行；将清单路径替换为实际文件，输出目录必须为空：

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.train_interactive `
  --manifest 'G:/datasets/my_objects/train.jsonl' `
  --output pims_net/outputs/interactive_run_01 `
  --epochs 20 --batch-size 2 --max-clicks 5 --lr 0.0001 --device cuda
```

`--batch-size` 表示每次优化更新累积的图像数。逐图执行可支持不同原图尺寸和提前停止，
每图损失先对其有效编辑轮次求均值，再对有编辑损失的图像平均梯度。
默认 AdamW、weight decay=1e-4、梯度范数裁剪=1；CPU 可用，未添加混合精度或分布式训练。
`--max-steps 1` 可限制为一个优化批次，用于检查链路；不代表完成一个数据集的训练。

## 点击闭环与冻结边界

“初始 Decoder”指当前首击冻结的官方 EfficientViT-SAM-L0 mask decoder，不是之前保留的 InitialMaskDecoder CNN。
第一次对全零预测模拟正点击，调用官方 SAM，按预测 IoU 选候选。后续每轮先将上一轮输出恢复到原图，
再与 GT 比较并采样新点击；由 SpatialStateMemory/ConvGRU、EditIntentDecoder、MaskEditor 更新 Mask。
图像特征只计算一次，各图像状态从零开始，总点击数包含第一击、最多 5 次。

采样依据 [RITM 原训练源码的保留镜像](https://github.com/ggsonic/ritm_interactive_segmentation/blob/master/isegm/engine/trainer.py)
中的 `get_next_points`：阈值 0.49，FN/FP 区域外补一圈零，用 L2 距离变换比较两类最大距离，
在选定类别中距离大于该最大值一半的区域随机取点。本实现另外排除已点击坐标，无可用错误点时提前停止。
原官方仓库地址检查时返回 404，因此核对的是保留源码镜像。

此处复用 RITM 的在线误差点击策略；不会复现 RITM 的完整训练配方。
我们对第 2～5 击的全部实际编辑轮次监督并保留跨轮梯度，而不是仅监督最后一轮。
点击采样与监督类别构造使用 detach；循环状态和 Mask 更新保留计算图。
整个官方 SAM（含 Backbone、prompt encoder、mask decoder）冻结且保持 eval。
首击无训练损失；如果首击后已无错误区域，该图跳过优化，不伪造额外点击。

## 损失定义

总损失默认：`L = BCE + Dice + Edit + 0.1 * Preserve`。
A1 使用直接完整 Mask 预测头，没有 Edit 类别监督，因此该项为零；其余损失与 A2/A3 相同。
权重可通过 `--bce-weight/--dice-weight/--edit-weight/--preserve-weight` 设置，须非负且不全为零。

- **BCE**：有效区域的 binary cross entropy with logits。
- **Dice**：有效区域的 soft Dice loss，平滑常数为 1。
- **类别平衡 Edit Loss**：依据停止梯度的上一轮概率和 GT，漏分像素标 ADD，多余像素标 REMOVE，
  原本正确像素标 KEEP。每个出现的类别分别计算平均负对数概率，再对类别均值等权平均。
  缺失类别不进入平均，不会产生除零。
- **Preserve Loss**：仅在上一轮已正确且非补边的像素上，计算新概率与停止梯度旧概率的平均绝对差。
  不惩罚对旧错误区域的必要修改；若没有已正确像素，返回可微的零。

旧的 `training_step` 仍支持人工提供点击序列及 BCE+Dice；四项损失和自动采样由新的
`training.interactive_rollout.interactive_rollout` 与训练脚本实现。

## 日志、权重与加载

每个运行目录保存：

- `config.json`：数据路径、训练超参数、设备、损失权重。
- `metrics.jsonl`：每张图的点击坐标/标签/逐轮 IoU 和各项平均损失；每个优化批次的聚合指标。
- `train.log`：运行进度与异常；`summary.json`：总批次数与实际优化更新次数。
- `last.pt`：最新处理的 epoch checkpoint；`epoch_0001.pt` 等按 `--save-every` 保存。

checkpoint 保存研究层权重、AdamW 状态、epoch/step 和配置，通过临时文件替换完成写入。
新 checkpoint 为版本 2，记录变体、训练清单和冻结权重哈希、点击协议及实际更新次数；加载时检查变体一致。
它不复制共享冻结 SAM 权重。由于不是完整 model.state_dict，请使用专用加载接口：

```python
from pims_net.models import PIMSNet
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.inference import PIMSPredictor

model = PIMSNet(variant="a3").to("cuda")  # A1/A2 权重须选择对应 variant。
checkpoint = load_training_checkpoint("pims_net/outputs/interactive_run_01/last.pt", model)
predictor = PIMSPredictor(model)
# predictor.set_image(rgb_image)
# result = predictor.add_click(x, y, label=1)
```

加载函数也接受已有 optimizer 以恢复保存的优化器状态；脚本当前不提供自动断点续训参数。
首击仍依赖工程内同一套官方 L0 权重。输出目录非空时拒绝覆盖，避免污染已有实验。

## 验证记录

单元测试入口：

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m unittest pims_net.tests.test_interactive_training -v
```

另使用两张随机图像及矩形 GT 跑通真实官方权重的 CUDA 训练：两张图各 5 次闭环点击、1 次 AdamW 更新，
全部损失有限，日志和两个 checkpoint 已保存到 `pims_net/outputs/interactive_smoke_run/`。
这些是合成数据上的功能测试产物，不是可用于实际分割的训练结果；尚未运行真实数据集训练。

9 项新增单元测试全部通过。另已重新加载实际保存的研究权重和优化器状态，确认研究参数发生更新、
SAM 仍冻结，并完成 5 击交互推理；213 个原始 Baseline 文件 SHA-256 校验一致。
实际训练与加载验证进程退出码为 0；退出时仍有既有 wandb 临时目录清理权限提示，未改动其依赖实现。
