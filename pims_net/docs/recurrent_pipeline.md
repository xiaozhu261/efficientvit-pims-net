# 首击 SAM + 循环编辑 PIMS-Net

## 路径与状态

1. 官方 L0 完整权重加载到 `FrozenSAMInitializer`。使用官方 transform 提取一次 `[B,256,64,64]` 图像特征。
2. 第一击调用冻结的官方 prompt encoder 和 mask decoder，生成三个候选，按预测 IoU 最大值选择一个。
   输出为官方 `[B,1,256,256]` 低分辨率 logits；此步骤不调用记忆更新或编辑模块。
3. 第二击起，累计正负点击的 `[B,2,64,64]` 高斯图与压缩后的图像特征、上一轮 sigmoid mask 下采样结果拼接。
   `SpatialStateMemory` 将图像特征投影到 32 通道，与两通道点击图和单通道上一轮 mask 概率拼接，
   通过 ConvGRU 更新固定 `[B,128,64,64]` 状态；EditIntentDecoder 再融合图像特征、新状态和旧 Mask 概率，
   预测 ADD/REMOVE/KEEP 概率。MaskEditor 按 `P_new = KEEP * P_old + ADD` 更新 Mask。
   第一轮隐藏状态为零。编辑模块为新初始化参数，需要训练后才能用于有效纠错。

SAM 全部参数冻结、始终 eval，不参与反向传播；新模块随 `model.train()/eval()` 切换。
旧 `EncoderAdapter`、`InitialMaskDecoder`、残差 `EditDecoder` 继续保留供独立实验使用，新流程不调用它们。
原始 `efficientvit/`、`applications/` 和共享预训练权重均未修改。

`InteractionState` 显式保存图像特征、原图尺寸、点击历史、logits、Mask 概率、编辑概率和隐藏状态。
`refine` 返回新的状态，不原地改写上一轮状态；训练时不 detach 循环状态，可做跨轮反向传播。
一个状态始终属于同一图像批次，不要在状态建立后更换模型权重、设备或图像。

`PIMSNet()` 现在固定使用 128 通道状态，不再接受旧的 `hidden_channels` 参数。
研究层参数名称和形状已改变，旧循环研究权重不能直接加载；官方 L0 权重继续复用。

## 独立 SpatialStateMemory 接口

```python
import torch
from pims_net.models import SpatialStateMemory

memory = SpatialStateMemory()
features = torch.randn(2, 256, 64, 64)
click_maps = torch.zeros(2, 2, 64, 64)  # 当前轮累计的正/负点击高斯图
previous_logits = torch.zeros(2, 1, 256, 256)
state = memory.reset_state(features)   # 新图像批次：全零 [2,128,64,64]
for step in range(5):
    state = memory(features, click_maps, previous_logits, state)
    # 实际交互中，每轮更新 click_maps，并把新的 Mask 概率转为 logits 后传入下一轮。
```

输入的上一轮 Mask 使用 logits，模块内部执行 sigmoid 并双线性缩小到 64×64。
省略 `state` 也表示从零开始；连续交互须显式传回上一轮状态。
状态不保存在模块属性中，多个图像会话可以交错调用同一套网络参数。
PIMS-Net 的 `initialize/initial_state` 每次生成全零状态；`PIMSPredictor.set_image` 和
`reset_image` 清除旧会话，下一张图像第一击重新使用 SAM 并建立零状态。

## EditIntentDecoder 与 MaskEditor

`EditIntentDecoder(features, hidden, previous_mask)` 输入为 `[B,256,64,64]` 图像特征、
`[B,128,64,64]` ConvGRU 状态、`[B,1,256,256]` 旧 Mask **概率**。
先将旧 Mask 双线性缩小到 64，再拼接三类信息，通过轻量 CNN 和两次上采样，输出
`[B,3,256,256]` softmax 概率。通道顺序固定为 **ADD=0、REMOVE=1、KEEP=2**，逐像素和为 1。
初始化时 KEEP 的输出层偏置为 4，其他偏置为 0，使新网络初始偏向保留；仍需训练才能有效纠错。

`MaskEditor(previous_mask, edit_probabilities)` 无可训练参数，直接计算：

```text
P_new = KEEP * P_old + ADD
```

纯 ADD 得到 1，纯 REMOVE 得到 0，纯 KEEP 保持 P_old；REMOVE 通过三类概率竞争抑制前景。
输入必须是归一化概率，不能把 logits 直接传给编辑器。该公式不做阈值化或截断，梯度可跨轮传播。
独立 MaskEditor 也支持其他匹配的 H×W 尺寸。

```python
from pims_net.models import EditIntentDecoder, MaskEditor

intent_decoder = EditIntentDecoder()
editor = MaskEditor()
features = torch.randn(2, 256, 64, 64)
hidden = torch.randn(2, 128, 64, 64)
previous_mask = torch.rand(2, 1, 256, 256)
intents = intent_decoder(features, hidden, previous_mask)
new_mask = editor(previous_mask, intents)  # [2,1,256,256] 概率
```

PIMS-Net 的 `state.probabilities` 保存精确公式结果；`state.edit_probabilities` 保存三通道编辑概率，
首击为 None。为兼容既有 BCEWithLogits 与 SAM 后处理接口，仅在转换 logits 时使用
`torch.logit(P_new, eps=1e-6)` 保证有限值，不改写保存的概率。原有 `forward` 仍返回 logits 序列。
新的研究解码器使用 `edit_intent_decoder` 参数名，旧残差解码器训练权重不能直接加载到新结构。

## 坐标约定

- 坐标为原图 `(x,y)` 像素，标签 `1` 正、`0` 负。每一步每张图必须有一个有效点击。
- 当前序列接口不接受 `-1` 填充或变长序列；批次内使用相同点击数，按点击数分组即可。
- 首击 SAM 使用官方 **1024 提示坐标系**；L0 图像输入虽为 512，提示编码器仍是 1024。
- 后续高斯点击图沿用 L0 的 512 图像缩放和右/下补边，再映射到 64 网格。
- 所有低分辨率 logits 为补边后的正方形画布，未经过 sigmoid。推理接口用官方后处理恢复原图尺寸。

## 训练接口

完整闭环训练请使用 [train_interactive.py](interactive_training.md)：自动采样最多 5 次点击，
计算 BCE、Dice、类别平衡 Edit 和 Preserve Loss，并保存日志和权重。以下是仍保留的人工点击序列接口。

`PIMSNet.forward(images, points, labels, image_sizes=None)` 返回 `[B,T,1,256,256]`，包含首击与所有后续编辑。
`points` 为 `[B,T,2]`，`labels` 为 `[B,T]`，`T>=1`。
图像可传 RGB uint8 HWC numpy 图像列表，尺寸自动推导；也可传已按官方预处理的
`[B,3,512,512]` 浮点张量，此时必须提供原图 `image_sizes=(H,W)` 或 `[B,2]`。

`training_step` 要求 `T>=2`，对第二击及之后的结果计算 BCE + soft Dice，各样本各轮等权平均。
它会调用 `model.train()` 并返回 `loss/bce/dice/logits`，不执行优化器操作。
首击不参与训练损失。没有自动点击模拟器，点击序列由调用方提供。
GT 为 `[B,1,256,256]`、值域 `[0,1]`，须提前对齐到官方补边画布：
按原图比例缩放至最长边 1024、右下补零，再最近邻缩放到 256。补边区域按背景参与当前损失。

```python
import numpy as np
import torch
from pims_net.models import PIMSNet
from pims_net.training import training_step

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PIMSNet().to(device)
optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-4)

# 仅展示接口；实际实验替换图像、点击与对齐后的 GT。
images = [np.zeros((256, 512, 3), dtype=np.uint8)]
points = torch.tensor([[[128., 64.], [300., 120.], [200., 80.]]])
labels = torch.tensor([[1, 0, 1]])
targets = torch.zeros(1, 1, 256, 256)
optimizer.zero_grad(set_to_none=True)
result = training_step(model, images, points, labels, targets)
result["loss"].backward()
optimizer.step()
```

此接口支持训练循环集成；没有启动真实数据集训练，也不包含数据加载、点击采样或训练调度。
权重与优化器状态可使用标准 PyTorch state_dict 保存到 `pims_net/outputs/<experiment>/`。

## 交互推理接口

```python
from pims_net.inference import PIMSPredictor

# model 是上文模型；正式推理前加载训练好的 PIMS-Net state_dict。
predictor = PIMSPredictor(model)
predictor.set_image(images[0])          # 提取并缓存图像特征，重置点击和隐藏状态
first = predictor.add_click(128, 64, label=1)   # 官方冻结 SAM
second = predictor.add_click(300, 120, label=0)  # SpatialStateMemory + 意图解码/Mask 编辑
third = predictor.add_click(200, 80, label=1)
mask = third["mask"]                  # bool [1,1,H,W]
original_logits = third["logits"]     # [1,1,H,W]
low_res_logits = third["low_res_logits"]  # [1,1,256,256]
low_res_probabilities = third["low_res_probabilities"]  # 精确更新后的 [1,1,256,256]
edit_probabilities = third["edit_probabilities"]  # [1,3,256,256]，首击为 None
predictor.reset_image()                # 释放当前交互状态
```

推理操作使用 no_grad，第二击起不会再次执行图像编码器或 SAM mask decoder。
`set_image` 建立新会话；一个 predictor 对应一张当前图像，可创建多个 predictor 隔离会话。
低层 `model.initialize(...)` / `model.refine(state, points, labels)` 也可直接调用，支持批量状态。

## 验证

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m unittest pims_net.tests.test_edit_intent pims_net.tests.test_spatial_state_memory pims_net.tests.test_recurrent_pims pims_net.tests.test_encoder_adapter pims_net.tests.test_click_decoder -v
python -B -m pims_net.tests.smoke --verify-baseline
```

循环模块单元测试与真实官方权重集成测试覆盖：首击与官方单图 Predictor、批量 forward 对齐，
第二击起的分支调用次数，B=1/2 的形状，编辑公式及通道语义，跨轮梯度，优化器只更新研究模块，
SAM 的冻结与 BatchNorm 不变，缓存推理与训练 forward 一致，以及新图像重置状态。
数值比较使用相同 batch size；CUDA 在不同 batch size 下选择的计算路径可能造成数值差异。

2026-10-17 首版循环模型验证：在已有 Conda `efficientvit` 环境运行，9 项新增循环测试与 14 项已有测试，
共 23 项通过；实际完成了合成数据上的反向传播及一步优化器更新。
进程退出码为 0，仍有原有 wandb 临时目录清理权限提示，不影响测试断言。

SpatialStateMemory 更新后验证：4 项独立记忆测试与 10 项循环集成测试全部通过。
包括 CPU/CUDA、B=1/2 连续 5 次记忆更新；第 5 轮向第 1 轮状态及三类输入的梯度传播；
交错图像会话隔离；真实 SAM 的连续 5 次点击；切换新图像后与全新 predictor 的结果一致。

EditIntentDecoder/MaskEditor 更新后验证：6 项新单元测试、11 项循环测试及 4 项空间记忆测试，
共 21 项通过；覆盖 CPU/CUDA 上的 B=1/2 意图归一化、三类输入梯度、ADD/REMOVE/KEEP 极端情况、
混合更新公式、输入校验、真实模型接线与连续 5 轮交互。
