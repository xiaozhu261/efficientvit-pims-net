# G2 Gate单轮贡献与推理校准结果

固定16个COCO训练实例、seed42，加载G2的100epoch/800步最终权重。
仅校准推理，不训练、不修改模型或原权重。

校准公式：`Gate_eff = alpha * Gate * 1[Gate >= tau]`，先阈值筛选，再乘倍率。
测试alpha={0,.1,.25,.5,.75,1}、tau={0,.3,.5,.7}的24组组合及两套点击协议。
首击保持EV-SAM，负点击直通；各条件反馈自身Mask；GT不参与Gate推理区域选择。

| 方案 | Replay IoU@3 / @5 | 自适应RITM IoU@3 / @5 |
|---|---|---|
| G0（alpha=0） | 0.7563 / 0.7586 | 0.7563 / 0.7586 |
| 原G2（alpha=1,tau=0） | 0.7042 / 0.6690 | 0.7370 / 0.7367 |
| alpha=0.25,tau=0.7 | 0.7802 / 0.7811 | 0.7810 / 0.7894 |

最优校准相对G0：Replay 9实例改善、5退化、2持平；自适应11改善、3退化、2持平。
自适应校准结果的TP破坏率仍为1.426%，高于G0的1.053%；不能把净IoU提升等同所有损伤下降。
这里只是在16个训练实例上选出的网格最优点，不是COCO泛化性能。

## 单轮Gate归因

在同一次S0前向后，对比同一历史、点击与隐藏状态对应的P_base和P_fused。
K为Gate单独造成的FN→TP像素事件数，E为TN→FP。

| 原G2协议 | 正点击次数 | K | E | 池化K/E | 平均单轮ΔIoU |
|---|---:|---:|---:|---:|---:|
| Replay | 51 | 40,513 | 41,892 | 0.9671 | -0.027105 |
| 自适应 | 22 | 49,075 | 26,625 | 1.8432 | -0.000623 |

池化K/E与逐点击等权平均ΔIoU的权重不同，不能互相替代。
同一个像素可能被多个轮次重复统计，单轮事件计数不能等同首击到第五击的累计变化。

## 验证与文件

7项校准测试通过；3840份逐轮数组的公式、原图/原生网格计数及ΔIoU关系均已核验。
alpha=0在640个逐轮状态上与G0的概率、Logits、隐藏状态和点击历史逐位一致。
alpha=1,tau=0在160个状态上精确复现原G2。原模型、权重及213个Baseline文件保持不变。

- `comparison.csv`：48组完整指标。
- `round_summary.csv`、`per_instance.csv`：逐轮与逐实例指标。
- `positive_click_contributions.csv`：每次正点击K/E及ΔIoU。
- `contribution_by_round.csv`、`contribution_by_instance.csv`：贡献汇总。
- `best_calibration_paired.csv`：最优点与G0的逐实例比较。
- `report.html`：移除原始COCO案例图后的公开报告。
- `protocol.json`、`statistics_checks.json`、`test_results.txt`：协议和验收记录。
- `fixed_instances.json`、`fixed_replay_clicks.json`：16实例的COCO标识和固定Replay点击，
  不包含本机数据路径、图像或GT Mask。

此目录只包含轻量导出结果，不包含COCO图像、GT Mask、检查点或3840份原始概率数组。
协议中的相对资源路径说明实验来源，不表示这些被忽略的本地运行文件已随仓库提供。

完整实验入口见 [`evaluate_gate_calibration.py`](../../evaluate_gate_calibration.py)，
报告入口见 [`evaluation/gate_calibration_report.py`](../../evaluation/gate_calibration_report.py)。
