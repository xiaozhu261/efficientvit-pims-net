"""Describe A2/A3 degradation from saved diagnostics; never rerun a model."""

import argparse
import json
from pathlib import Path

import numpy as np

from pims_net.paths import OUTPUT_ROOT


def transition_analysis(records):
    result = {}
    for method in ("a2", "a3"):
        objects = records[method]
        pairs = [(obj["trace"][i - 1], obj["trace"][i]) for obj in objects
                 for i in range(1, min(5, len(obj["trace"])))]
        transitions = {}
        for key, denominator in (("tp_lost", "tp"), ("tn_lost", "tn"),
                                 ("fn_fixed", "fn"), ("fp_fixed", "fp")):
            count = sum(current[key] for previous, current in pairs)
            eligible = sum(previous[denominator] for previous, current in pairs)
            transitions[key] = {"pixels": count, "eligible_pixels": eligible,
                                "rate": count / eligible if eligible else None}
        clicks = {}
        for label, name in ((1, "positive"), (0, "negative")):
            selected = [current for previous, current in pairs if current["label"] == label]
            clicks[name] = {"count": len(selected),
                            "satisfied_rate": float(np.mean([r["latest_click_satisfied"] for r in selected])) if selected else None,
                            "mean_iou_delta": float(np.mean([r["iou_delta"] for r in selected])) if selected else None}
        final_delta = [obj["trace"][min(5, len(obj["trace"])) - 1]["iou"] - obj["trace"][0]["iou"] for obj in objects]
        result[method] = {"objects": len(objects), "actual_edits_2_to_5": len(pairs),
                          "degrading_edits": sum(current["iou_delta"] < 0 for previous, current in pairs),
                          "objects_worse_than_first": sum(delta < 0 for delta in final_delta),
                          "mean_iou5_minus_iou1": float(np.mean(final_delta)),
                          "transitions": transitions, "clicks": clicks}
    identities = lambda rows: [(r.get("index"), r.get("id")) for r in rows]
    if identities(records["a2"]) != identities(records["a3"]):
        raise ValueError("Paired A2/A3 analysis requires identical object ordering")
    delta = np.asarray([b["IoU@5"] - a["IoU@5"] for a, b in zip(records["a2"], records["a3"])])
    result["a3_minus_a2"] = {"mean_iou5_delta": float(delta.mean()),
                              "improved_objects": int((delta > 0).sum()),
                              "worse_objects": int((delta < 0).sum()),
                              "tied_objects": int((delta == 0).sum()),
                              "scope": "Paired objects from one run; does not establish stability across training seeds."}
    return result


def probability_statistics(probabilities, target, valid):
    """Descriptive confidence band, not a change to the evaluation threshold."""
    probabilities = np.asarray(probabilities).squeeze(0)
    result = {}
    for name, region in (("foreground", target & valid), ("background", ~target & valid)):
        values = probabilities[region]
        result[name] = {"pixels": int(values.size),
                        "mean_probability": float(values.mean()) if values.size else None,
                        "near_05_fraction": float(((values >= .49) & (values <= .51)).mean()) if values.size else None}
    return result


def saved_confidence_analysis(root, objects):
    from PIL import Image
    from pims_net.training.interactive_losses import prepare_target

    rows = []
    for obj in objects:
        folder = root / obj["directory"]
        with Image.open(folder / "gt.png") as image:
            target, valid = prepare_target(np.asarray(image) > 0, "cpu")
        target, valid = target[0, 0].numpy().astype(bool), valid[0, 0].numpy().astype(bool)
        for method in ("a2", "a3"):
            for round_folder in sorted((folder / method).glob("round_*")):
                step = int(round_folder.name.removeprefix("round_"))
                if step > 5:
                    continue
                with np.load(round_folder / "native.npz") as native:
                    stats = probability_statistics(native["probabilities"], target, valid)
                    if "intents" in native:
                        add, remove, keep = native["intents"]
                        active = valid & ((add + remove) > .5)
                        gap = np.abs(add - remove)[active]
                        stats["intent_separation"] = {"active_pixels": int(active.sum()),
                                                       "active_fraction": float(active.sum() / valid.sum()),
                                                       "mean_abs_add_remove": float(gap.mean()) if gap.size else None,
                                                       "gap_below_002_fraction": float((gap < .02).mean()) if gap.size else None}
                    else:
                        stats["intent_separation"] = None
                rows.append({"index": obj["index"], "id": obj["id"], "method": method, "click": step, **stats})
    with (root / "native_confidence.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = []
    for method in ("a2", "a3"):
        for step in range(1, 6):
            selected = [r for r in rows if r["method"] == method and r["click"] == step]
            row = {"method": method, "click": step, "actual_objects": len(selected)}
            for group in ("foreground", "background"):
                populated = [r[group] for r in selected if r[group]["pixels"]]
                row[group] = {"objects_with_pixels": len(populated), **{
                    key: float(np.mean([r[key] for r in populated])) if populated else None
                    for key in ("mean_probability", "near_05_fraction")}}
            edits = [r["intent_separation"] for r in selected if r["intent_separation"] is not None]
            active_edits = [r for r in edits if r["active_pixels"]]
            row["intent_separation"] = {"actual_edits": len(edits), "edits_with_active_pixels": len(active_edits),
                                        "mean_abs_add_remove": float(np.mean([r["mean_abs_add_remove"] for r in active_edits])) if active_edits else None,
                                        "gap_below_002_fraction": float(np.mean([r["gap_below_002_fraction"] for r in active_edits])) if active_edits else None}
            summary.append(row)
    return {"band": [.49, .51], "grid": "native 256; exclude padding; macro mean over actual rounds with nonempty GT group",
            "intent_separation_definition": "Active pixels: ADD+REMOVE >0.5; report abs(ADD-REMOVE) and fraction <0.02; macro means over rounds with nonempty active region",
            "rounds": summary}


def write_failure_panel(root, records, objects):
    """Deterministically show the largest A3 drop; not a typical-case claim."""
    import matplotlib.pyplot as plt
    from PIL import Image

    index = min(range(len(objects)), key=lambda i: records['a3'][i]['IoU@5'] - records['a3'][i]['IoU@1'])
    obj = objects[index]
    folder = root / obj['directory']
    read = lambda name: np.asarray(Image.open(folder / name))
    region = read('gt.png') > 0
    rounds = {method: min(5, len(records[method][index]['trace'])) for method in ('a2', 'a3')}
    for method in ('a2', 'a3'):
        region |= read(f'{method}/round_{rounds[method]:02d}/mask.png') > 0
    region |= read('a3/round_01/mask.png') > 0
    yy, xx = np.nonzero(region)
    height, width = region.shape
    margin = max(12, int(max(xx.max() - xx.min(), yy.max() - yy.min()) * .15))
    x0, x1 = max(0, xx.min() - margin), min(width, xx.max() + margin + 1)
    y0, y1 = max(0, yy.min() - margin), min(height, yy.max() + margin + 1)
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    for axis in axes.flat:
        axis.axis('off')
    for ax, path, title in zip(axes[0, :3], ('image.png', 'gt.png', 'a3/round_01/overlay.png'),
                              ('Image (common ROI)', 'Ground truth', f'Frozen SAM, click 1: IoU {records["a3"][index]["IoU@1"]:.3f}')):
        ax.imshow(read(path)[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
    axes[0, 3].text(0, .9, 'Largest A3 IoU drop\nfrom click 1 to 5\n\nGreen: true positive\nBlue: missed foreground\nRed: false positive\nWhite outline: GT\n\nHeatmap scale: fixed 0-1\nPurple=0; yellow=1\n\nMetrics use full images.', va='top', fontsize=11)
    for row, method in enumerate(('a2', 'a3'), 1):
        step = rounds[method]
        for ax, name in zip(axes[row], ('overlay', 'ADD', 'REMOVE', 'KEEP')):
            path = f'{method}/round_{step:02d}/{name}.png'
            if (folder / path).is_file():
                ax.imshow(read(path)[y0:y1, x0:x1])
            else:
                ax.text(.1, .5, 'No edit intent at first click')
            ax.set_title(f'{method.upper()} click {step}: IoU {records[method][index]["IoU@5"]:.3f}' if name == 'overlay' else name, fontsize=11)
    fig.suptitle(str(obj['id']), fontsize=15)
    fig.savefig(root / 'failure_case.png', dpi=160)
    plt.close(fig)
    return {"index": obj['index'], "id": obj['id'], "selection": "largest A3 IoU drop from first to fifth click", "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)]}


def write_report(evaluation_directory):
    directory = Path(evaluation_directory).resolve()
    if not directory.is_relative_to(OUTPUT_ROOT.resolve()) or directory == OUTPUT_ROOT.resolve():
        raise ValueError("Diagnostic reports must be saved beneath pims_net/outputs")
    root = directory / "diagnostics"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    records = {method: [] for method in ("a0", "a1", "a2", "a3")}
    for line in (directory / "per_object.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        records[row["method"]].append(row)
    analysis = transition_analysis(records)
    analysis["native_confidence"] = saved_confidence_analysis(root, summary["objects"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis["failure_case"] = write_failure_panel(root, records, summary["objects"])
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fields = (("iou", "IoU (%)", 100), ("precision", "Precision (%)", 100),
              ("recall", "Recall (%)", 100), ("foreground_ratio", "Foreground / image (%)", 100),
              ("pred_to_gt_area_ratio", "Predicted / GT area (macro mean)", 1),
              ("degraded_from_first_fraction", "Objects worse than first click (%)", 100))
    colors = {"a0": "#326db3", "a1": "#8b64b2", "a2": "#de8040", "a3": "#278874"}
    for ax, (field, title, scale) in zip(axes.flat, fields):
        for method, color in colors.items():
            rows = [r for r in summary["rounds"] if r["method"] == method and r["click"] <= 5]
            ax.plot([r["click"] for r in rows], [r[field] * scale for r in rows], "o-", color=color, label=method.upper())
        if field == "foreground_ratio":
            ax.axhline(rows[0]["gt_foreground_ratio"] * 100, color="#777777", ls=":", label="GT")
        if field == "pred_to_gt_area_ratio":
            ax.axhline(1, color="#777777", ls=":")
        ax.set(title=title, xlabel="Click", xticks=range(1, 6))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Fixed checkpoints | {len(records['a3'])} objects | identical evaluation protocol", fontsize=15)
    fig.savefig(root / "trends.png", dpi=150)
    plt.close(fig)

    pct = lambda value: "N/A" if value is None else f"{100 * value:.2f}%"
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>A2/A3 退化诊断</title><style>body{font:16px/1.7 system-ui;margin:32px;max-width:1250px;color:#18232e}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #dde4ec;padding:8px}pre{white-space:pre-wrap}</style>'
    page += '<a href="index.html">逐轮表格与全部实例</a><h1>A2/A3 退化诊断</h1><img src="trends.png">'
    page += '<p>指标在原图计算，按对象宏平均。前景/GT 面积比也是逐对象比值的均值，小目标的过度分割可能使其明显升高。下表的区域转换率在第 2–5 击按符合条件的像素数加权，包含多轮重复出现的像素。</p>'
    for method in ("a2", "a3"):
        stats = analysis[method]
        rows = [r for r in summary["rounds"] if r["method"] == method and r["click"] <= 5]
        page += f'<h2>{method.upper()}</h2><p>{stats["objects_worse_than_first"]}/{stats["objects"]} 个对象的第五击 IoU 低于首击；实际编辑中 {stats["degrading_edits"]}/{stats["actual_edits_2_to_5"]} 次使 IoU 下降。</p>'
        page += '<table><tr><th>轮次</th><th>IoU</th><th>ΔIoU（百分点）</th><th>前景比例</th><th>Precision</th><th>Recall</th></tr>'
        for row in rows:
            delta_text = '首击' if row['iou_delta'] is None else f'{100 * row["iou_delta"]:+.2f}'
            page += f'<tr><td>{row["click"]}</td><td>{pct(row["iou"])}</td><td>{delta_text}</td><td>{pct(row["foreground_ratio"])}</td><td>{pct(row["precision"])}</td><td>{pct(row["recall"])}</td></tr>'
        page += '</table><ul>'
        labels = {"tp_lost": "旧正确前景被删除", "tn_lost": "旧正确背景被误加为前景", "fn_fixed": "旧漏分前景被补回", "fp_fixed": "旧误分前景被删除"}
        for key, label in labels.items():
            item = stats["transitions"][key]
            page += f'<li>{label}：{pct(item["rate"])}（{item["pixels"]:,} / {item["eligible_pixels"]:,} 像素次）。</li>'
        for name, label in (("positive", "正"), ("negative", "负")):
            item = stats["clicks"][name]
            page += f'<li>第 2–5 击的{label}点击满足率：{pct(item["satisfied_rate"])}，共 {item["count"]} 击。</li>'
        page += '</ul>'
        tp = summary['edit_analysis_clicks_2_to_5'][method]['groups']['tp']
        tn = summary['edit_analysis_clicks_2_to_5'][method]['groups']['tn']
        if tp['pixels'] and tn['pixels']:
            page += f'<p>原生网格上，旧 TP 的平均有效 ADD/REMOVE 为 {tp["effective_add"]:.4f}/{tp["effective_remove"]:.4f}；旧 TN 为 {tn["effective_add"]:.4f}/{tn["effective_remove"]:.4f}。它们描述概率上推/下拉，不等同于原图二值 Mask 的翻转率。</p>'
        confidence = [r for r in analysis["native_confidence"]["rounds"] if r["method"] == method]
        page += '<p>原生 GT 前景中概率处于 [0.49, 0.51] 的比例（实际轮次、非空 GT 前景对象宏平均）：' + ' → '.join(pct(r['foreground']['near_05_fraction']) for r in confidence) + '。此区间仅用于诊断置信度，不改变 0.5 评估阈值。</p>'
        separated = confidence[-1]['intent_separation']
        if separated['mean_abs_add_remove'] is not None:
            page += f'<p>第五击活跃编辑区（ADD+REMOVE&gt;0.5）平均 |ADD−REMOVE| 为 {separated["mean_abs_add_remove"]:.4f}，其中差值小于 0.02 的比例为 {pct(separated["gap_below_002_fraction"])}。按存在活跃区的对象宏平均，排除 padding。</p>'
    paired = analysis['a3_minus_a2']
    page += f'<h2>A2 → A3</h2><p>第五击平均 IoU 变化 {100 * paired["mean_iou5_delta"]:+.2f} 个百分点；{paired["improved_objects"]} 个对象改善，{paired["worse_objects"]} 个变差，{paired["tied_objects"]} 个相同。这是单次训练结果，不能证明跨种子稳定提升。</p>'
    page += '<p>当 ADD≈REMOVE 时，更新近似 P_new=KEEP×P_old+(1−KEEP)/2，会将概率推向 0.5。是否发生这种置信度收缩，应结合上面的概率区间与通道差值统计判断，不能单凭一张热力图下结论。</p>'
    page += '<h2>退化最严重的 A3 实例</h2><p>按首击到第五击 IoU 下降最多自动选择，属于极端失败案例，不能代替总体统计。统一 ROI 包含 GT、首击和两组最终 Mask；指标仍按全图计算。</p><img src="failure_case.png">'
    page += '<p>本报告提供固定权重下的描述性证据。Precision 与面积比用于识别多分，Recall 与旧 TP 丢失用于识别漏分；二者可能同时发生。各方法采用同一自适应选点协议，后续坐标随自身误差不同，不能把所有差异单独归因于历史记忆。训练损失、监督阈值或容量的因果影响仍需专门对照实验。</p></html>'
    (root / "analysis.html").write_text(page, encoding="utf-8")
    index = root / "index.html"
    index_text = index.read_text(encoding="utf-8")
    if 'href="analysis.html"' not in index_text:
        index.write_text(index_text.replace('<table>', '<p><a href="analysis.html">A2/A3 退化分析、趋势图和失败案例</a></p><table>', 1), encoding="utf-8")
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_directory")
    args = parser.parse_args()
    print(json.dumps(write_report(args.evaluation_directory), indent=2))
