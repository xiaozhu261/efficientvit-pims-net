# 工程检查记录（2026-10-17）

- 当前工程是无 `.git` 的源码目录；没有发现适用的 `AGENTS.md`。
- 模型工厂：`efficientvit/sam_model_zoo.py`；结构：`efficientvit/models/efficientvit/sam.py`。
- 数据及训练器：`efficientvit/samcore/data_provider/`、`efficientvit/samcore/trainer/`。
- 训练、评估、演示及部署入口：`applications/efficientvit_sam/`。
- 依赖来源：根目录 `requirements.txt` 和 `setup.py`。研究目录引用原 requirements，不新增依赖版本约束。
- 本地完整权重：L0、L1、L2、XL0、XL1；蒸馏权重：L0、L1、L2、XL1。未发现 XL0 蒸馏权重。
- 原始 L0 训练配置使用 `efficientvit-sam-l0.pt`，本地实际文件为 `efficientvit_sam_l0.pt`。
  仅研究目录独立配置使用正确文件名，原配置未修改。
- 未发现工程内训练数据集。原始训练和评估代码使用 CUDA/NCCL。

## 当前执行环境

解释器：`E:\miniconda\python.exe`，Python 3.13.5。
PyTorch 2.8.0+cu126，`torch.cuda.is_available()` 返回 True。
能定位 torch、torchvision、segment_anything、timm、omegaconf；这不代表完整依赖链可导入。

实际导入 Baseline 时在 `efficientvit/models/nn/triton_rms_norm.py` 遇到
`ModuleNotFoundError: No module named 'triton'`。原 requirements/setup 未声明该依赖。
导入失败退出时还出现 wandb 临时目录清理权限错误。
本次未改动或重装共享环境，也未修改 Baseline 绕过这些问题。

## 验证范围

资源检查与原始文件哈希校验可使用 Python 标准库完成。
资源检查通过仅代表路径存在及原文件内容未改变，不代表权重已成功加载。
上述默认环境尚不能加载模型；训练和数据集评估还需准备数据与兼容的执行环境。

## 编码器适配器验证（2026-10-17）

后续检查发现已有 `E:\miniconda\envs\efficientvit\python.exe` 环境：Python 3.11.15、
PyTorch 2.6.0+cu124，CUDA 可用，且已具备 Triton 和 SAM 所需依赖，无需安装新包。
已在该环境使用真实本地 L0 权重运行 `pims_net.tests.test_encoder_adapter`，6 项测试全部通过：

- B=1、B=2 的实际编码输出分别为 `[1,256,64,64]`、`[2,256,64,64]`，数值有限。
- 预处理逐元素等于官方 transform，编码输出与官方 image_encoder 一致。
- 仅保留完整 image_encoder，所有权重及缓冲区与加载的官方模型相同。
- 上层 `.train()` 不改变冻结状态或 BatchNorm 统计量，后续训练头可以正常反向传播。
- 非法输入明确报错。

213 个原始文件（含权重）的 SHA-256 再次校验一致。
测试进程退出码为 0；退出时仍存在 wandb 临时目录清理 PermissionError，与此前环境检查一致，未修改该依赖。
本次验证的是编码器推理及冻结行为，不涵盖完整训练或数据集评估。
