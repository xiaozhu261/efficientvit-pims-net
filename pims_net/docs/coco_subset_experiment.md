# 真实 COCO 子集实验

本次固定方案：1024 张训练图像、256 张 held-out 图像，每图随机选一个非 crowd 实例，
要求标注面积和解码后面积均至少 256 像素；划分种子 20261017。
原始数据位于 `G:/lab/VIMR-Click/dataset/coco`，图像实际位于 `train2017/train2017`。
该目录有 train2017 图像和 train/val 标注，但未发现官方 val2017 图像目录，
因此在 train2017 内按图像 ID 做固定、不重叠的训练/held-out 划分，不能称为官方 COCO val2017 结果。
同一图像的其他实例不会出现在另一划分。原始图像和标注只读，研究目录保存独立二值 Mask。

三个训练种子 42、43、44；每个种子独立训练 A1/A2/A3，均使用 5 epochs、batch size=2、
AdamW lr=1e-4、最多 5 击，默认损失权重不变。每个变体预算为 2560 个训练批次。
冻结 SAM、使用最后一个 epoch 权重，不依据 held-out 表现调参或挑选 checkpoint。
评估每个种子下的 A0–A3，统一最多 20 击，其余指标协议见 [统一评估说明](unified_evaluation.md)。

```powershell
& 'E:/miniconda/envs/efficientvit/python.exe' -B -m pims_net.run_coco_experiment `
  --coco-root G:/lab/VIMR-Click/dataset/coco `
  --output pims_net/outputs/coco_subset_20261017 `
  --train-count 1024 --test-count 256 --epochs 5 --seeds 42 43 44
```

驱动脚本顺序执行数据准备、每种子的训练和评估、最终汇总；单个阶段失败即记录状态并退出，
不把缺少种子的结果当成完整对比。Windows 后台启动时使用隐藏窗口并重定向标准输出和错误。
本次按用户要求启动后不主动汇报进度，待用户要求“查看结果”再读取状态和结果。

输出目录：`pims_net/outputs/coco_subset_20261017/`。

- `experiment.json`：预先固定的规模、种子、预算、解释器与驱动进程 PID。
- `status.json`：当前阶段、子进程 PID；完成时为 `completed`，失败时保存错误。
- `data/`：两份 JSONL、独立实例 Mask、选中源图像/Mask/标注文件 SHA-256。
- `seed_42/`、`seed_43/`、`seed_44/`：各自独立训练权重、日志及四组评估结果。
- `comparison.csv`、`comparison.md`：完整对比表，所有指标均值和种子间样本标准差。
- `per_seed_comparison.csv`：各随机种子的完整指标。
- `stability.json`：A3−A2、A3−A0、A3−A1 的逐种子差值与配对对象 bootstrap 区间。

“稳定提升”的本次操作性判据：A3−A2 的 IoU@5 在三个种子下均为正，
且固定这些种子后、配对 held-out 对象的 bootstrap 95% 区间下界大于 0。
这不是对任意数据集或任意训练随机种子的保证；若不满足，报告明确写未满足。
正确区域破坏率同时对照 IoU@5−IoU@1、首击到第五击的退化比例、Boundary IoU、NoC 和失败率，
避免把“不修改 Mask 导致的低破坏率”误解为有效纠错能力。
重采样以对象为单位，先平均同一对象在三个种子上的配对差值，不把重复对象当成独立样本。

正式运行前，用 4 张训练图像、2 张不重叠 held-out 图像执行 1 epoch 的独立预检，
产物位于 `coco_preflight_20261017/`；该预检不参与正式对比，不用于选超参数。
