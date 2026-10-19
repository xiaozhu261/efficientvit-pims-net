"""Matched-seed comparisons with paired object bootstrap and explicit tradeoffs."""

import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("IoU@1", "IoU@3", "IoU@5", "NoC@90", "BoundaryIoU@5",
           "correct_region_damage_rate@2:5", "failure_rate@90", "IoU_gain@5-1", "regression_rate@5_vs_1")
LOWER_BETTER = {"NoC@90", "correct_region_damage_rate@2:5", "failure_rate@90", "regression_rate@5_vs_1"}


def paired_delta(values, seed=20261017, repetitions=2000):
    """values: [training seeds, held-out objects], candidate minus reference.

    Bootstrap objects after averaging matched training seeds, avoiding treating
    repeated evaluations of the same object as independent observations.
    Seed variation is reported separately; this is not a population-seed CI.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected finite seed/object deltas")
    per_object = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    sampled = per_object[rng.integers(0, len(per_object), (repetitions, len(per_object)))].mean(axis=1)
    return {"delta": float(values.mean()), "per_seed_delta": values.mean(axis=1).tolist(),
            "paired_object_bootstrap_95ci": np.quantile(sampled, [.025, .975]).tolist(),
            "object_positive_fraction": float((per_object > 0).mean()),
            "object_negative_fraction": float((per_object < 0).mean())}


def build_report(root, seeds):
    root = Path(root)
    all_records, expected_ids = {}, None
    per_seed_rows = []
    for seed in seeds:
        folder = root / f"seed_{seed}/evaluation"
        records = [json.loads(line) for line in (folder / "per_object.jsonl").read_text().splitlines()]
        by_method = {method: {item["id"]: item for item in records if item["method"] == method}
                     for method in ("a0", "a1", "a2", "a3")}
        ids = sorted(by_method["a0"])
        if not ids or any(sorted(items) != ids for items in by_method.values()):
            raise ValueError("Incomplete or unmatched evaluation object sets")
        if expected_ids is not None and ids != expected_ids:
            raise ValueError("Held-out objects differ between seeds")
        expected_ids = ids
        for object_id in ids:
            initial = [by_method[method][object_id]["IoU@1"] for method in by_method]
            if not np.allclose(initial, initial[0], atol=1e-8, rtol=0):
                raise ValueError("First-click parity check failed")
        for method, items in by_method.items():
            for item in items.values():
                item["failure_rate@90"] = float(not item["reached_90"])
                item["IoU_gain@5-1"] = item["IoU@5"] - item["IoU@1"]
                item["regression_rate@5_vs_1"] = float(item["IoU@5"] < item["IoU@1"])
            per_seed_rows.append({"seed": seed, "method": method, "objects": len(ids),
                                  **{metric: float(np.mean([items[i][metric] for i in ids])) for metric in METRICS}})
        all_records[seed] = by_method
    table = []
    for method in ("a0", "a1", "a2", "a3"):
        rows = [row for row in per_seed_rows if row["method"] == method]
        row = {"method": method, "seeds": len(seeds), "objects_per_seed": len(expected_ids)}
        for metric in METRICS:
            values = [item[metric] for item in rows]
            row[metric] = float(np.mean(values))
            row[metric + "_seed_std"] = float(np.std(values, ddof=1)) if len(seeds) > 1 else 0.0
        table.append(row)
    comparisons = {}
    for reference in ("a0", "a1", "a2"):
        comparison = {}
        for metric in METRICS:
            deltas = [[all_records[seed]["a3"][i][metric] - all_records[seed][reference][i][metric]
                       for i in expected_ids] for seed in seeds]
            result = paired_delta(deltas)
            sign = -1 if metric in LOWER_BETTER else 1
            result["improved_seed_count"] = sum(sign * value > 0 for value in result["per_seed_delta"])
            comparison[metric] = result
        comparisons[f"a3_minus_{reference}"] = comparison
    for name, rows in (("comparison.csv", table), ("per_seed_comparison.csv", per_seed_rows)):
        with (root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    details = {"seeds": seeds, "objects": len(expected_ids), "comparisons": comparisons,
               "ci_scope": "paired objects, averaged over the specified seeds; training-seed uncertainty reported separately",
               "damage_caveat": "lower damage alone can reflect KEEP/no-op; jointly inspect IoU gain and regression rate"}
    (root / "stability.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    lines = ["# COCO 子集 A0–A3 对比", "", "固定 train2017 图像不重叠划分；不是官方 val2017 榜单。",
             f"训练随机种子：{seeds}；各组评估 {len(expected_ids)} 个相同对象。数值为均值 ± 种子间样本标准差，比例采用 0–1。", "",
             "| 方法 | " + " | ".join(METRICS) + " |", "| --- | " + " | ".join(["---"] * len(METRICS)) + " |"]
    for row in table:
        lines.append("| " + row["method"].upper() + " | " + " | ".join(
            f"{row[m]:.4f} ± {row[m + '_seed_std']:.4f}" for m in METRICS) + " |")
    lines += ["", "## A2 → A3 稳定性", "", "以下均为 A3 − 对照；IoU/Boundary 越大越好，NoC/破坏率越小越好。"]
    for reference in ("a2", "a0", "a1"):
        lines += ["", f"### A3 相对 {reference.upper()}", "",
                  "| 指标 | 平均差 | 逐种子差 | 配对对象 bootstrap 95% CI | 提升种子数 |",
                  "| --- | --- | --- | --- | --- |"]
        for metric in ("IoU@3", "IoU@5", "NoC@90", "BoundaryIoU@5", "correct_region_damage_rate@2:5", "IoU_gain@5-1"):
            item = comparisons[f"a3_minus_{reference}"][metric]
            low, high = item["paired_object_bootstrap_95ci"]
            lines.append(f"| {metric} | {item['delta']:+.5f} | " +
                         ", ".join(f"{value:+.5f}" for value in item["per_seed_delta"]) +
                         f" | [{low:+.5f}, {high:+.5f}] | {item['improved_seed_count']}/{len(seeds)} |")
    delta = comparisons["a3_minus_a2"]["IoU@5"]
    stable = len(seeds) >= 3 and delta["improved_seed_count"] == len(seeds) and delta["paired_object_bootstrap_95ci"][0] > 0
    lines += ["", "## 验收解读", "",
              "A2→A3 在本次种子及对象样本上表现出一致的 IoU@5 提升。" if stable else
              "本次实验未满足 A2→A3 稳定提升的判据（所有种子 IoU@5 差均为正，且配对对象区间下界 > 0）。",
              "破坏率降低需结合 IoU@5−IoU@1 和退化比例判断：仅保持原 Mask 不变也会得到很低的破坏率。",
              "置信区间重采样相同 held-out 对象，先对种子取平均；不把重复评估的同一对象视为独立样本，也不代表任意训练种子的总体置信区间。",
              "各模型采用相同闭环选点规则，但后续坐标随自身误差变化；本实验比较整体交互系统，不能单独归因于某一损失。",
              "该实验为固定子集、固定预算、最后一个 epoch 的预先约定对比，没有根据 held-out 结果选最佳 epoch 或调整参数。", ""]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return details
