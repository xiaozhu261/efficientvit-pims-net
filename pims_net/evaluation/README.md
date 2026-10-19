# 评估

统一交互评估入口为 `python -m pims_net.evaluate_interactive`，需要 A1/A2/A3 三组独立训练权重。
比较 A0（原始 EV-SAM + Mask 提示）、A1（ConvGRU + 直接重预测）、A2（历史状态清零的编辑器）、A3（完整 PIMS-Net）。
输出 CSV/JSON 指标对照和逐对象点击轨迹；命令及精确定义见 [统一评估说明](../docs/unified_evaluation.md)。

下面为原始框提示评估入口，保持不变：
原始评估入口是 `applications/efficientvit_sam/eval_efficientvit_sam_model.py`，
评估工具位于同目录的 `sam_eval_utils.py`。

从工程根目录执行 COCO 真值框提示 Baseline 评估示例：

```bash
torchrun --nproc_per_node=1 applications/efficientvit_sam/eval_efficientvit_sam_model.py --dataset coco --image_root /absolute/path/to/coco/val2017 --annotation_json_file /absolute/path/to/coco/annotations/instances_val2017.json --model efficientvit-sam-l0 --weight_url assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt --prompt_type box
```

需要数据、CUDA/NCCL 及可导入的 Baseline 环境；尚未运行数据集评估。
研究对比需固定数据划分、权重、提示类型、输入分辨率及指标定义。
日志与结果保存到 `pims_net/outputs/<experiment>/`，不要写回共享权重目录。
