# PIMS-Net / EfficientViT-SAM Interactive Segmentation Research

独立交互式分割研究工程：在冻结的官方 EfficientViT-SAM-L0 上，研究空间记忆、
方向/强度编辑、正点击纠错 Gate，以及逐击诊断和推理校准。

原始 EfficientViT 工程和 Apache-2.0 LICENSE 保持不变，研究实现集中在 `pims_net/`。

- [研究说明、模型及实验入口](../pims_net/README.md)
- [克隆后的环境、权重与数据准备](../pims_net/docs/github_setup.md)
- [最新 Gate 校准结果摘要](../pims_net/results/gate_calibration16_20261019/README.md)
- [原始 EfficientViT 项目说明](../README.md)

## 研究内容

- 冻结的官方 L0 编码器及 EV-SAM 首击初始化。
- ClickEncoder、ConvGRU 空间记忆和 ADD/REMOVE/KEEP 编辑。
- DirectionalPIMSNet 的 Direction/Strength 独立分支及容量、损失消融。
- 冻结 S0 的 PositiveCorrectionGate 分支。
- RITM 五击闭环训练、固定 Replay、自适应评估及逐轮归因诊断。

## 最新实验

固定16个COCO训练实例、seed42；仅调整已有 G2 Gate 的推理倍率和阈值，不重新训练。

| 方案 | 固定 Replay IoU@5 | 自适应 RITM IoU@5 |
|---|---:|---:|
| G0：关闭 Gate | 0.7586 | 0.7586 |
| 原 G2：α=1、τ=0 | 0.6690 | 0.7367 |
| α=0.25、τ=0.7 | 0.7811 | 0.7894 |

这些结果来自16个训练实例的探索，不能作为COCO泛化性能。
公开仓库包含代码、配置、测试和轻量统计结果；不包含COCO图像、预训练权重、
训练检查点或大体积原始诊断数组。

## 上游与许可

本工程基于 [MIT HAN Lab / EfficientViT](https://github.com/mit-han-lab/efficientvit)。
保留上游作者署名、说明与 [Apache-2.0 LICENSE](../LICENSE)。
