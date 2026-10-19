# 模型

`build_baseline` 直接调用原始 `efficientvit.sam_model_zoo`，使用本地完整权重，默认 L0。
传入 `distilled=True` 可选择蒸馏初始化权重；当前没有 XL0 蒸馏文件，缺失时明确报错。
此入口不修改模块结构、训练模式或设备；推理时显式调用 `.eval()` 和 `.to(device)`。

后续 PIMS-Net 模块放在此目录，通过组合或子类复用 Baseline；不要改写 `efficientvit/`。
当前 PIMS-Net 采用首击冻结 SAM、后续空间记忆 + EditIntentDecoder + MaskEditor 的流程，见 [循环模型接口](../docs/recurrent_pipeline.md)。
循环融合已封装为 `SpatialStateMemory`，状态固定为 `[B,128,64,64]`，每张新图像从零开始。
`EditIntentDecoder` 输出 `[B,3,256,256]` ADD/REMOVE/KEEP softmax 概率；`MaskEditor` 执行
`P_new = KEEP * P_old + ADD`。旧残差 `EditDecoder` 仅保留作独立实验，不再参与当前流程。

## 冻结的 L0 编码器

`encoder_adapter.py` 中的 `EncoderAdapter` 严格加载本地官方 L0 完整权重，
随后仅保留整个 `image_encoder`（backbone、neck、最终 norm），不保留 prompt encoder 或 mask decoder。
所有参数冻结，调用适配器或上层模块的 `.train()` 也会保持编码器为 eval 模式，防止 BatchNorm 统计量改变。
前向使用 `no_grad`，所得特征可供后续可训练的研究模块使用。

```python
import numpy as np
from pims_net.models.encoder_adapter import EncoderAdapter

encoder = EncoderAdapter().to("cuda")  # 也支持 CPU
rgb = np.zeros((480, 640, 3), dtype=np.uint8)
features = encoder([rgb, rgb])          # [2, 256, 64, 64]
```

原始输入为 RGB、uint8、HWC numpy 图像或图像列表，列表允许不同分辨率。
直接复用官方 `sam.transform`：最长边缩放到 512 → ToTensor → 官方均值/标准差归一化 → 右/下补零至 512×512。
若使用 OpenCV 读取，调用前自行将 BGR 转为 RGB。
也可先调用 `encoder.preprocess(images)`，再把返回的 `[B,3,512,512]` 浮点张量交给编码器；
张量入口不再预处理，不能传入未归一化的原始张量。

从工程根目录使用已有环境执行真实权重集成测试（使用标准库 unittest，无需 pytest）：

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m unittest pims_net.tests.test_encoder_adapter -v
```

测试覆盖 B=1/2 输出形状、官方预处理与编码输出一致性、完整权重匹配、冻结与 BatchNorm 统计量、
后续训练头反向传播及错误输入。自动使用可用的 CUDA，否则使用 CPU。

## ClickEncoder 与 InitialMaskDecoder

`InitialMaskDecoder` 是保留的独立实验模块；新 PIMS-Net 首击由官方 SAM 解码，不再调用此 CNN。

`ClickEncoder(sigma=2.0)` 的独立 forward 接受：

- `points`：`[B,N,2]` 张量，坐标顺序为 `(x,y)`，单位是原图像素。
- `labels`：`[B,N]` 张量，`1` 为正点击、`0` 为负点击、`-1` 为批量填充，填充坐标被忽略。
- `image_size`：共享的 `(H,W)` 或逐图 `[B,2]` 张量。默认 `(512,512)` 表示坐标已在补边后的输入画布上。

输出为 `[B,2,64,64]` float32 高斯图，第 0 通道是正点击，第 1 通道是负点击。
`sigma` 单位为 64×64 网格像素；同类多个点击逐像素取最大值，不相加。
没有点击的通道输出零，支持 `N=0`。有效点击必须在图像范围内。
传入原图尺寸时，坐标按官方最长边 512 缩放（含尺寸舍入），再映射到 64 网格，保留右侧/底部补边位置。

`InitialMaskDecoder` 的独立 forward 接受 `[B,256,64,64]` 图像特征和 `[B,2,64,64]` 点击图。
拼接为 258 通道，以 1×1 卷积压缩到 64 通道，再用 3×3 卷积降到 32 通道；
两次双线性上采样配合 16 通道卷积与最终 1×1 输出层，生成 `[B,1,256,256]` mask logits。
中间使用 GroupNorm/GELU，参数量小于 5 万，所有解码器参数可训练，输出不做 sigmoid。
logits 对应补边后的正方形画布；原图尺寸恢复、去补边和阈值化由后续评估流程处理。

独立组合示例（不需要加载 SAM 权重）：

```python
import torch
from pims_net.models import ClickEncoder, InitialMaskDecoder

click_encoder = ClickEncoder()
decoder = InitialMaskDecoder()
features = torch.randn(2, 256, 64, 64)
points = torch.tensor([[[128., 128.], [384., 64.]],
                       [[64., 256.], [-1., -1.]]])
labels = torch.tensor([[1, 0], [1, -1]])
maps = click_encoder(points, labels, image_size=(512, 512))
logits = decoder(features, maps)  # [2, 1, 256, 256]
```

实际使用时将 `features` 替换为 `EncoderAdapter()(images)`，并向点击编码器传入每张原图的真实尺寸。
使用现有 Conda 环境运行测试：

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m unittest pims_net.tests.test_click_decoder -v
```

8 项测试覆盖独立形状、高斯中心与衰减、长宽比/舍入对齐、多点击最大值、空点击和填充、
输入校验、未经过 sigmoid 的 logits，以及 CPU/CUDA 上的组合前向、点击影响和梯度传播。
