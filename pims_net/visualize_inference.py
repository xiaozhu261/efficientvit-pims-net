"""Replay evaluated clicks and export auditable COCO inference comparison figures."""

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluation.experiment import file_identity, validate_experiment
from pims_net.evaluation.methods import create_sessions
from pims_net.evaluation.metrics import mask_iou, correct_region_damage
from pims_net.models import PIMSNet
from pims_net.paths import OUTPUT_ROOT, checkpoint_path
from pims_net.training.checkpoint import load_training_checkpoint


METHODS = ("a0", "a1", "a2", "a3")
NAMES = {"a0": "A0 原始 EV-SAM", "a1": "A1 直接重预测", "a2": "A2 无历史记忆", "a3": "A3 完整 PIMS-Net"}
COLORS = {"tp": np.array([42, 201, 127]), "fn": np.array([70, 137, 255]),
          "fp": np.array([250, 79, 91]), "damage": np.array([255, 205, 42])}


def select_cases(records):
    ordered = sorted(records, key=lambda key: (records[key]["a3"]["IoU@5"] - records[key]["a2"]["IoU@5"], key))
    return [("A3 相对 A2 改善最大", ordered[-1]),
            ("A3−A2 差异的中位样本", ordered[(len(ordered) - 1) // 2]),
            ("A3 相对 A2 退化最大", ordered[0]),
            ("A3 相对首击退化最大", min(ordered, key=lambda key: records[key]["a3"]["IoU@5"] - records[key]["a3"]["IoU@1"]))]


def blend(image, regions, opacity=.65):
    result = image.astype(float).copy()
    for mask, color in regions:
        result[mask] = (1 - opacity) * result[mask] + opacity * color
    return result.clip(0, 255).astype(np.uint8)


def display(ax, image, target=None, clicks=()):
    ax.imshow(image)
    if target is not None and target.any() and not target.all():
        ax.contour(target, levels=[.5], colors="black", linewidths=1.7)
        ax.contour(target, levels=[.5], colors="white", linewidths=.9)
    for index, click in enumerate(clicks, 1):
        positive = click["label"] == 1
        ax.scatter(click["x"], click["y"], marker="o" if positive else "X", s=36,
                   c="#00f1ff" if positive else "#ff9c24", edgecolors="black", linewidths=.8, zorder=4)
        ax.annotate(str(index), (click["x"], click["y"]), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="white", bbox=dict(facecolor="black", alpha=.65, pad=.2, edgecolor="none"))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def error_overlay(image, prediction, target):
    return blend(image, [(prediction & target, COLORS["tp"]), (target & ~prediction, COLORS["fn"]),
                         (prediction & ~target, COLORS["fp"])])


def replay(models, image, target, rows):
    features, sizes = models["a3"].sam_initializer.encode(image)
    sessions = create_sessions(models, features, sizes)
    result = {}
    for method, session in sessions.items():
        masks, damages = [], np.zeros_like(target)
        trace = rows[method]["trace"][:5]
        for click in trace:
            prediction = session.add_click(click["x"], click["y"], click["label"]) > .5
            actual_iou = mask_iou(prediction, target)
            if abs(actual_iou - click["iou"]) > 1e-7:
                raise RuntimeError(f"Replay does not match evaluated IoU: {method}, click {click['click']}: {actual_iou} vs {click['iou']}")
            if masks:
                damage = correct_region_damage(masks[-1], prediction, target)
                if damage["damaged_pixels"] != click["damaged_pixels"] or damage["previous_correct_pixels"] != click["previous_correct_pixels"]:
                    raise RuntimeError("Replay damage counts differ from the saved evaluation")
                damages |= (masks[-1] == target) & (prediction != target)
            masks.append(prediction)
        while len(masks) < 5:
            masks.append(masks[-1].copy())
        result[method] = {"masks": masks, "damage_union": damages, "trace": trace}
    return result


def save_detail(folder, label, sample, predictions, rows):
    image, target = sample["image"], sample["mask"]
    # Use one shared crop across methods/rounds, covering every GT/predicted
    # foreground pixel. Metrics remain those measured on the full image.
    foreground = target.copy()
    for info in predictions.values():
        for mask in info["masks"]:
            foreground |= mask
    ys, xs = np.where(foreground)
    padding = max(16, int(.15 * max(np.ptp(xs) + 1, np.ptp(ys) + 1)))
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half_w, half_h = max(64, np.ptp(xs) / 2 + padding), max(64, np.ptp(ys) / 2 + padding)
    bounds = (max(-.5, cx - half_w), min(image.shape[1] - .5, cx + half_w),
              max(-.5, cy - half_h), min(image.shape[0] - .5, cy + half_h))
    fig, axes = plt.subplots(4, 7, figsize=(21, 12), layout="constrained")
    for row, method in enumerate(METHODS):
        info = predictions[method]
        for step in range(5):
            pred = info["masks"][step]
            display(axes[row, step], blend(image, [(pred, COLORS["tp"])], .45), target, info["trace"][:step + 1])
            axes[row, step].set_title(f"第 {step + 1} 击 · IoU {100 * mask_iou(pred, target):.1f}%", fontsize=10)
            if step == 0:
                axes[row, step].set_ylabel(NAMES[method], fontsize=11)
            Image.fromarray(pred.astype(np.uint8) * 255).save(folder / f"{method}_click_{step + 1}.png")
        display(axes[row, 5], error_overlay(image, info["masks"][4], target), target)
        axes[row, 5].set_title("第 5 击误差", fontsize=10)
        display(axes[row, 6], blend(image, [(info["damage_union"], COLORS["damage"])], .8), target)
        axes[row, 6].set_title(f"2–5 击破坏率 {100 * rows[method]['correct_region_damage_rate@2:5']:.3f}%", fontsize=10)
    for ax in axes.flat:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[3], bounds[2])
    fig.suptitle(f"{label} | {sample['id']}\n统一局部放大 · 指标仍按完整原图计算；后续点击按各自误差闭环生成", fontsize=15)
    fig.supxlabel("白线：GT 边界；绿色：预测 Mask。误差图：绿=正确前景，蓝=漏分，红=多分。黄色=第 2–5 击任一轮正确→错误的像素并集。\n青色圆点=正点击，橙色叉=负点击，数字=点击顺序。破坏率仍按逐轮计数计算，黄色并集不能直接换算该比率。", fontsize=10)
    fig.savefig(folder / "detail.png", dpi=150, facecolor="white")
    plt.close(fig)


def save_summary(records, summary, output, seed=42):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    colors = ("#475569", "#bc6c25", "#386cb0", "#0a9678")
    for method, color in zip(METHODS, colors):
        trajectories = [[rows[method]["trace"][min(step, len(rows[method]["trace"])) - 1]["iou"]
                         for step in range(1, 6)] for rows in records.values()]
        axes[0].plot(range(1, 6), 100 * np.mean(trajectories, axis=0), marker="o", label=NAMES[method], color=color)
    axes[0].set(xlabel="点击次数", ylabel="平均 IoU (%)", xticks=range(1, 6), title=f"全部 {len(records)} 对象：逐点击效果")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    x = np.array([rows["a2"]["IoU@5"] for rows in records.values()]) * 100
    y = np.array([rows["a3"]["IoU@5"] for rows in records.values()]) * 100
    axes[1].scatter(x, y, s=16, alpha=.65, color="#0a9678")
    axes[1].plot([0, 100], [0, 100], color="#64748b", linestyle="--")
    axes[1].set(xlabel="A2 IoU@5 (%)", ylabel="A3 IoU@5 (%)", xlim=(-2, 102), ylim=(-2, 102), title="A2 → A3：每点一个对象\n虚线上方表示 A3 更好")
    damages = [100 * summary[m]["correct_region_damage_rate@2:5"] for m in METHODS]
    bars = axes[2].bar([m.upper() for m in METHODS], damages, color=colors)
    axes[2].bar_label(bars, fmt="%.3f%%", padding=5, fontsize=9)
    axes[2].set(ylabel="正确区域破坏率 (%) ↓", ylim=(0, max(damages) * 1.25), title="2–5 击：正确区域破坏\n需结合 IoU 判断，低破坏不等于有效纠错")
    fig.suptitle(f"COCO 留出子集 · 种子 {seed} · {len(records)} 个对象（仅一个训练种子）", fontsize=15)
    fig.savefig(output / "aggregate.png", dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root, output = Path(args.experiment).resolve(), Path(args.output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError("Use a dedicated output subdirectory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    evaluation = root / f"seed_{args.seed}/evaluation"
    protocol = json.loads((evaluation / "protocol.json").read_text())
    baseline = file_identity(checkpoint_path())
    if baseline != protocol["baseline"]:
        raise ValueError("Baseline differs from evaluated checkpoint")
    records = {}
    for line in (evaluation / "per_object.jsonl").read_text().splitlines():
        row = json.loads(line)
        records.setdefault(row["id"], {})[row["method"]] = row
    summary = {row["method"]: row for row in json.loads((evaluation / "summary.json").read_text())}
    manifest = root / "data/heldout.jsonl"
    if file_identity(manifest) != protocol["manifest"]:
        raise ValueError("Held-out manifest differs from evaluated data")
    dataset = InteractiveDataset(manifest)
    indices = {row[0]: index for index, row in enumerate(dataset.records)}
    models, checkpoints = {}, {}
    for variant in METHODS[1:]:
        path = root / f"seed_{args.seed}/training/{variant}/last.pt"
        if file_identity(path) != protocol["checkpoints"][variant]:
            raise ValueError("Research weights differ from the evaluated checkpoint")
        models[variant] = PIMSNet(variant=variant).to(args.device).eval()
        checkpoints[variant] = load_training_checkpoint(path, models[variant])
    validate_experiment(checkpoints, baseline["sha256"])
    del checkpoints
    cases = select_cases(records)
    fig, axes = plt.subplots(len(cases), 6, figsize=(18, 12), layout="constrained")
    selected, links = [], []
    with torch.no_grad():
        for row, (label, sample_id) in enumerate(cases):
            sample = dataset[indices[sample_id]]
            predictions = replay(models, sample["image"], sample["mask"], records[sample_id])
            folder = output / f"case_{row + 1}"
            folder.mkdir()
            Image.fromarray(sample["image"]).save(folder / "original.png")
            Image.fromarray(sample["mask"].astype(np.uint8) * 255).save(folder / "ground_truth.png")
            display(axes[row, 0], sample["image"])
            axes[row, 0].set_title(f"{label}\n{sample_id}", fontsize=10)
            display(axes[row, 1], blend(sample["image"], [(sample["mask"], COLORS["tp"])], .65), sample["mask"])
            axes[row, 1].set_title("GT 目标实例", fontsize=11)
            for column, method in enumerate(METHODS, 2):
                info = predictions[method]
                display(axes[row, column], error_overlay(sample["image"], info["masks"][4], sample["mask"]), sample["mask"], info["trace"])
                axes[row, column].set_title(f"{NAMES[method]}\nIoU@5 {100 * records[sample_id][method]['IoU@5']:.1f}%", fontsize=10)
            save_detail(folder, label, sample, predictions, records[sample_id])
            selected.append({"case": row + 1, "selection": label, "id": sample_id,
                             "records": records[sample_id], "replay_verified": True})
            links.append(f'<section id="case{row+1}"><h2>{html.escape(label)} · {sample_id}</h2><p>统一局部放大，指标仍按完整原图计算。点击图像可打开原尺寸；每行对应一种方法，第 1–5 列展示逐点击 Mask。</p><a href="case_{row+1}/detail.png"><img loading="lazy" src="case_{row+1}/detail.png"></a></section>')
            print(f"Verified and visualized {sample_id}", flush=True)
    legend = [Patch(color=COLORS[key] / 255, label=label) for key, label in (("tp", "正确前景"), ("fn", "漏分 FN"), ("fp", "多分 FP"))]
    legend += [Line2D([], [], color="black", label="白线：GT 边界"),
               Line2D([], [], marker="o", color="#00c4cc", linestyle="none", label="正点击"),
               Line2D([], [], marker="X", color="#ff9c24", linestyle="none", label="负点击")]
    fig.legend(handles=legend, loc="outside lower center", ncol=6, frameon=False, fontsize=11)
    fig.suptitle("真实 COCO 推理 · 第 5 击误差对比\n按完整评估结果选取的改善、中位与失败案例；极值案例不代表总体", fontsize=17)
    fig.savefig(output / "overview.png", dpi=160, facecolor="white")
    plt.close(fig)
    save_summary(records, summary, output, args.seed)
    (output / "selection_and_replay.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PIMS-Net · 真实 COCO 推理可视化</title>
<style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1600px;margin:40px auto;padding:0 24px;color:#18232e;background:#f5f7fa}h1,h2{line-height:1.35}section{margin:32px 0;padding:24px;background:white;border-radius:12px}img{display:block;width:100%;height:auto}a{color:#087c70}nav a{margin-right:22px}p{max-width:1100px}</style>
<h1>PIMS-Net · 真实 COCO 推理可视化</h1><p>种子 42，1024 张训练图像 / 256 张图像不重叠的留出图像。使用已完成评估的同一权重和保存点击，逐轮 IoU 与破坏像素数已核对一致。此页面不启动训练。</p>
<p>绿=正确前景，蓝=漏分，红=多分，白线=GT 边界。不同模型的后续点击可不同。案例按改善最大、中位差异、退化最大及相对首击退化最大选取，不能用极值案例推断总体。</p>
<nav><a href="#overall">总体统计</a><a href="#overview">案例总览</a><a href="#case1">改善案例</a><a href="#case2">中位案例</a><a href="#case3">退化案例</a><a href="#case4">首击退化案例</a></nav>
<section id="overall"><h2>全部 256 个对象的结果</h2><a href="aggregate.png"><img src="aggregate.png"></a><p>A3 的平均 IoU@5 低于首击，破坏率下降不能单独证明有效纠错。本次只完成一个种子，无法判断跨种子稳定性。</p></section>
<section id="overview"><h2>第 5 击案例总览</h2><a href="overview.png"><img src="overview.png"></a></section>'''
    document = document.replace("种子 42，1024 张训练图像 / 256 张图像不重叠的留出图像。",
                                f"种子 {args.seed}，{len(dataset)} 张留出图像（训练数据配置见原评估记录）。")
    document += "\n".join(links) + '<p><a href="selection_and_replay.json">案例选择、原始点击及指标记录</a>。每个 case 目录另存原图、GT 和各轮二值预测 PNG。</p></html>'
    (output / "index.html").write_text(document, encoding="utf-8")
    print(f"Saved visualization gallery: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
