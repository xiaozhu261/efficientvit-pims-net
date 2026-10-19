# 训练

闭环训练入口已实现：`python -m pims_net.train_interactive --manifest <JSONL> --output pims_net/outputs/<run>`。
最多 5 次 RITM 风格在线点击，优化 BCE + Dice + 类别平衡 Edit + Preserve，保存日志和研究权重。
数据约定、完整命令及损失定义见 [交互训练说明](../docs/interactive_training.md)。
正式消融使用 `python -m pims_net.train_ablations`，相同预算分别训练并保存 A1/A2/A3，
具体定义与评估命令见 [消融与评估说明](../docs/unified_evaluation.md)。

新的 PIMS-Net 训练接口为 `pims_net.training.training_step`：对第二击起的循环编辑结果计算
BCE + soft Dice，保留跨轮梯度，只训练空间记忆中的图像特征投影、ConvGRU 和 EditIntentDecoder。
MaskEditor 无可训练参数；概率结果以数值稳定的 logit 转换连接既有训练损失。
首击的完整官方 SAM 始终冻结，具体接口与样例见 [循环模型接口](../docs/recurrent_pipeline.md)。
以下命令用于原始 Baseline 训练，不是新的 PIMS-Net 循环训练入口。

后续研究训练入口、损失函数和优化逻辑放在此目录。现有可复用实现：
`efficientvit.samcore.trainer.SAMTrainer`、`SAMRunConfig`。

`../configs/baseline_l0.yaml` 是原始 default + L0 配置的独立副本，修正了蒸馏文件名，
其余训练参数沿用原始值。从工程根目录运行以下命令可使用原始训练器：

```bash
torchrun --nproc_per_node=1 applications/efficientvit_sam/train_efficientvit_sam_model.py pims_net/configs/baseline_l0.yaml --data_provider.root /absolute/path/to/sam --path pims_net/outputs/baseline_l0
```

该命令要求已有 SA-1B 数据和支持 CUDA/NCCL 的环境；原始训练器不是 Windows 原生 CPU 训练入口。
默认 Python 环境的模型导入受缺失 `triton` 阻塞；已有 `efficientvit` Conda 环境已通过编码器测试，但尚未验证训练可运行。
新实验使用新的 `pims_net/outputs/<experiment>/`，仅在明确继续同一实验时使用 `--resume`。
后续研究配置另建文件，保留此 Baseline 配置作为对照。
