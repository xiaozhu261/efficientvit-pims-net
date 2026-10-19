# 数据

`InteractiveDataset` 已支持 JSONL 清单中的单实例图像/二值 Mask 对，供 `train_interactive.py` 使用。
具体格式与校验条件见 [交互训练说明](../docs/interactive_training.md)。

此目录存放研究数据集适配器、变换和采样代码。原始数据存放在外部数据目录，配置中引用路径，
避免复制大型数据集。可选本地 `raw/`、`processed/` 已加入本目录上级的忽略规则。

现有 SA-1B 数据实现位于 `efficientvit/samcore/data_provider/sam.py`，可复用 `SAMDataProvider`。
沿用其 `images/`、`masks/`、`sa_images_ids.txt` 数据约定。
COCO/LVIS 评估需要图像与对应标注 JSON；检测框评估还需要检测结果 JSON。

本次未发现工程内数据集目录，也未下载或生成训练数据。
后续实验须明确任务、标注格式及训练/验证/测试划分，并将划分清单随实验保存。
