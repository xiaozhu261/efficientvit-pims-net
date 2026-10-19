# GitHub 版本使用说明

## 仓库内容

`efficientvit/`、`applications/` 和根目录上游说明保持原样。
所有新增研究模型、训练、评估、诊断和测试均在 `pims_net/`。
`pims_net/results/` 保存可公开的轻量实验统计；`pims_net/outputs/` 中的本地运行产物不提交。

## 环境

本地真实训练使用已有 Conda 环境 `efficientvit`，不是 uv 虚拟环境。
已验证 Python 3.11.15、PyTorch 2.6.0+cu124、CUDA GPU，并具备 Triton 和 SAM 依赖。
根目录 `uv.lock` 是现有工程文件，不能据此认定研究实验在 uv 环境运行。

复现时按根目录 `requirements.txt`、`setup.py` 和 `pims_net/requirements.txt` 准备兼容环境；
不要把本机 `E:/miniconda/...` 解释器路径当作跨机器通用路径。
使用当前环境的 `python -B -m ...` 运行各入口。

## 预训练权重

权重不包含在Git仓库。按上游 [EfficientViT-SAM说明](../../applications/efficientvit_sam/README.md)
获取官方 EfficientViT-SAM-L0 完整权重，并放在：

```text
assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt
```

`pims_net/paths.py` 从当前检出目录解析此路径；不会自动下载或覆盖权重。
研究分支的 best/last 检查点也不包含在本仓库，需由本地训练生成或另行提供。
特别是 Gate 校准需要完整的 S0 和 G2 检查点，不能仅凭公开CSV恢复推理网络。

## 数据与实验路径

COCO数据不包含在Git仓库。原机器完整数据根目录为 `G:/lab/VIMR-Click/dataset/coco`；
其他机器请使用实际数据目录及相应 manifest，不要直接使用历史绝对路径。
各实验使用的16实例选择、点击协议、阈值和预算见研究README及对应协议。

部分后续实验默认引用此前本地 `pims_net/outputs/...` 中的 manifest、日志或检查点。
克隆后应先生成对应前置实验，或通过脚本的 `--reference`、`--manifest` 等现有参数指定资源。
参数是否可用以对应入口的 `--help` 为准。

## 快速检查

无需真实权重即可运行新的代数与校准合同测试：

```bash
python -B -m unittest pims_net.tests.test_gate_calibration -v
```

这仍要求环境中安装代码依赖，但不会创建训练优化器或修改检查点。
实际编码器、训练和全量推理测试需要准备上述权重及数据。

`docs/baseline_manifest.json` 记录原机器213项上游文件和本地预训练权重的SHA-256；
不含权重的全新检出无法直接通过全部本地资源哈希校验，这不代表代码检出损坏。
