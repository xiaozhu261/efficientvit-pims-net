"""Optional, observation-only per-click artifacts and edit degradation statistics."""

import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pims_net.training.interactive_losses import prepare_target


CHANNELS = ("ADD", "REMOVE", "KEEP")


def intent_statistics(previous, current, intents, target, valid):
    """Native 256-grid diagnostics; condition on OLD TP/TN/FN/FP at threshold .5.

    Effective probability additions/removals are A*(1-P_old) and R*P_old.
    Conditioning excludes padded pixels. Empty groups return null means.
    """
    previous, current = np.asarray(previous).squeeze(0), np.asarray(current).squeeze(0)
    target, valid = np.asarray(target, bool), np.asarray(valid, bool)
    if previous.shape != target.shape or current.shape != target.shape or valid.shape != target.shape or intents.shape != (3, *target.shape):
        raise ValueError("Intent statistics require aligned native probability/GT grids")
    old = previous > .5
    groups = {"tp": old & target, "tn": ~old & ~target, "fn": ~old & target, "fp": old & ~target}
    add_effect = intents[0] * (1 - previous)
    remove_effect = intents[1] * previous
    result = {"formula_max_error": float(np.max(np.abs(current - (previous + add_effect - remove_effect)))),
              "normalization_max_error": float(np.max(np.abs(intents.sum(0) - 1))), "groups": {}}
    for name, region in groups.items():
        selected = region & valid
        count = int(selected.sum())
        values = {"pixels": count}
        for channel, label in enumerate(CHANNELS):
            values[label] = float(intents[channel][selected].mean()) if count else None
        for label, data in (("effective_add", add_effect), ("effective_remove", remove_effect), ("probability_delta", current - previous)):
            values[label] = float(data[selected].mean()) if count else None
        values["foreground_crossings"] = int(np.count_nonzero(selected & ~old & (current > .5)))
        values["background_crossings"] = int(np.count_nonzero(selected & old & (current <= .5)))
        result["groups"][name] = values
    return result


def round_summary(records, max_clicks):
    """Macro object means; stopped objects carry their last mask, delta=0.

    Actual update counts and click satisfaction exclude carried-forward rows.
    """
    result = []
    for method, objects in records.items():
        for step in range(1, max_clicks + 1):
            rows = [obj["trace"][min(step, len(obj["trace"])) - 1] for obj in objects]
            updated = [obj["trace"][step - 1] for obj in objects if len(obj["trace"]) >= step]
            row = {"method": method, "click": step, "objects": len(objects), "objects_updated": len(updated)}
            for key in ("iou", "foreground_ratio", "gt_foreground_ratio", "pred_to_gt_area_ratio", "precision", "recall"):
                values = [item[key] for item in rows if item[key] is not None]
                row[key] = float(np.mean(values)) if values else None
            row["iou_delta"] = None if step == 1 else float(np.mean([
                obj["trace"][step - 1]["iou_delta"] if len(obj["trace"]) >= step else 0. for obj in objects]))
            row["degraded_from_first_fraction"] = float(np.mean([item["iou"] < obj["trace"][0]["iou"] for item, obj in zip(rows, objects)]))
            row["latest_click_satisfied_rate"] = float(np.mean([item["latest_click_satisfied"] for item in updated])) if updated else None
            result.append(row)
    return result


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class DiagnosticWriter:
    def __init__(self, output):
        self.root = Path(output) / "diagnostics"
        self.root.mkdir()
        self.rows = []
        self.objects = []
        metadata = {"mask_grid": "original image, threshold probability >0.5",
                    "intent_grid": "native 256x256; PNG heatmaps linearly resized/cropped to original image; NO sigmoid",
                    "heatmap_scale": "fixed [0,1] VIRIDIS, purple=0 yellow=1; no per-image normalization",
                    "precision_recall_zero_division": 0, "iou_delta_first_click": None,
                    "intent_statistics": "native grid; excludes padding; groups based on previous mask >0.5",
                    "missing_intents": "null for first click and for A0/A1; no fabricated heatmaps",
                    "native_npz": "float32 low-resolution logits, probabilities, previous probabilities and intent channels when available",
                    "transitions": "tp_lost=correct foreground removed; tn_lost=correct background added; fn_fixed/fp_fixed=errors corrected",
                    "early_stop": "save only real rounds; summary carries forward stopped masks with delta zero"}
        (self.root / "definitions.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        scale = cv2.applyColorMap(np.tile(np.arange(256, dtype=np.uint8), (20, 1)), cv2.COLORMAP_VIRIDIS)
        Image.fromarray(cv2.cvtColor(scale, cv2.COLOR_BGR2RGB)).save(self.root / "heatmap_scale.png")

    def begin_object(self, index, sample):
        self.sample = sample
        self.index = index
        self.folder = self.root / f"object_{index:04d}"
        self.folder.mkdir()
        Image.fromarray(sample["image"]).save(self.folder / "image.png")
        Image.fromarray(sample["mask"].astype(np.uint8) * 255).save(self.folder / "gt.png")
        target, valid = prepare_target(sample["mask"], "cpu")
        self.target, self.valid = target[0, 0].numpy(), valid[0, 0].numpy().astype(bool)
        self.object_rows = []

    def callback(self, method):
        if method not in ("a0", "a1", "a2", "a3"):
            raise ValueError("Unknown diagnostic method")

        def save(*, session, probabilities, current_prediction, previous_prediction, item, trace):
            folder = self.folder / method / f"round_{item['click']:02d}"
            folder.mkdir(parents=True)
            snapshot = session.diagnostic_snapshot()
            Image.fromarray(current_prediction.astype(np.uint8) * 255).save(folder / "mask.png")
            # This PNG is a display/inspection aid; NPZ retains exact native values.
            Image.fromarray(np.rint(probabilities * 65535).astype(np.uint16)).save(folder / "probability_u16.png")
            target = self.sample["mask"]
            overlay = self.sample["image"].astype(float).copy()
            for region, color in ((current_prediction & target, [40, 200, 125]),
                                  (~current_prediction & target, [60, 130, 255]),
                                  (current_prediction & ~target, [255, 65, 85])):
                overlay[region] = .4 * overlay[region] + .6 * np.asarray(color)
            overlay = overlay.astype(np.uint8)
            contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
            clicks = [{key: click[key] for key in ("click", "x", "y", "label")} for click in trace]
            for click in clicks:
                point = (click["x"], click["y"])
                color = (0, 240, 255) if click["label"] else (255, 155, 35)
                cv2.drawMarker(overlay, point, color, cv2.MARKER_CROSS if click["label"] else cv2.MARKER_TILTED_CROSS, 9, 2)
                cv2.putText(overlay, str(click["click"]), (point[0] + 5, point[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
            Image.fromarray(overlay).save(folder / "overlay.png")
            native = {key: value for key, value in snapshot.items() if value is not None and key != "original_intents"}
            np.savez_compressed(folder / "native.npz", **native)
            stats = None
            if snapshot["intents"] is not None:
                stats = intent_statistics(snapshot["previous_probabilities"], snapshot["probabilities"],
                                          snapshot["intents"], self.target, self.valid)
                for channel, label in enumerate(CHANNELS):
                    heatmap = np.rint(np.clip(snapshot["original_intents"][channel], 0, 1) * 255).astype(np.uint8)
                    colors = cv2.cvtColor(cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
                    Image.fromarray(colors).save(folder / f"{label}.png")
                stats["at_latest_click"] = {label: float(snapshot["original_intents"][channel, item["y"], item["x"]]) for channel, label in enumerate(CHANNELS)}
            record = {"index": self.index, "id": self.sample["id"], "method": method, **item,
                      "intents_available": stats is not None, "intent_statistics": stats,
                      "artifact_directory": str(folder.relative_to(self.root)).replace("\\", "/")}
            (folder / "clicks.json").write_text(json.dumps(clicks, indent=2), encoding="utf-8")
            (folder / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
            self.rows.append(record)
            self.object_rows.append(record)
        return save

    def end_object(self):
        # Embed metadata rather than fetch JSON so the gallery works from disk.
        data = json.dumps(self.object_rows, ensure_ascii=False).replace("<", "\\u003c")
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐轮诊断</title>
<style>body{font:16px/1.6 system-ui;margin:24px;background:#f5f7fa;color:#17232e}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}[hidden]{display:none!important}figure{margin:0;background:white;padding:10px}img{width:100%;max-height:480px;object-fit:contain}select{font-size:18px;margin:10px}pre{white-space:pre-wrap}</style>
<a href="../index.html">返回汇总</a><h1>OBJECT_TITLE</h1>
<label>方法 <select id="method"><option>a0</option><option>a1</option><option>a2</option><option selected>a3</option></select></label>
<label>点击轮次 <select id="step"></select></label><p id="metrics"></p>
<div class="grid"><figure><figcaption>原图</figcaption><img src="image.png"></figure><figure><figcaption>GT</figcaption><img src="gt.png"></figure><figure><figcaption>预测 Mask</figcaption><img id="mask"></figure>
<figure><figcaption>绿=正确前景，蓝=漏分，红=多分；白线=GT，青色+=正点击，橙色×=负点击</figcaption><img id="overlay"></figure>
<figure><figcaption>热力图固定 0–1：左侧 0，右侧 1</figcaption><img src="../heatmap_scale.png" style="height:40px"><p id="intentnote"></p><a id="native">下载原生数值 NPZ</a></figure></div>
<div class="grid" id="heats"><figure><figcaption>ADD</figcaption><img id="ADD"></figure><figure><figcaption>REMOVE</figcaption><img id="REMOVE"></figure><figure><figcaption>KEEP</figcaption><img id="KEEP"></figure></div>
<details><summary>点击、统计与数值诊断</summary><pre id="raw"></pre></details><script>
const rows=EMBEDDED_DATA, method=document.getElementById('method'), step=document.getElementById('step');
function update(){const r=rows.find(x=>x.method===method.value&&x.click===Number(step.value));if(!r)return;
const dir=r.method+'/round_'+String(r.click).padStart(2,'0')+'/';
for(const name of ['mask','overlay'])document.getElementById(name).src=dir+name+'.png';
document.getElementById('metrics').textContent=`IoU ${(r.iou*100).toFixed(2)}% | ΔIoU ${r.iou_delta===null?'首击':(r.iou_delta*100).toFixed(2)+' pp'} | 前景 ${(r.foreground_ratio*100).toFixed(2)}% (GT ${(r.gt_foreground_ratio*100).toFixed(2)}%) | Precision ${(r.precision*100).toFixed(2)}% | Recall ${(r.recall*100).toFixed(2)}%`;
document.getElementById('heats').hidden=!r.intents_available;
document.getElementById('intentnote').textContent=r.intents_available?'直接对概率 resize/crop，没有额外 sigmoid。':'此轮无编辑意图输出：首击或 A0/A1，不能视为全 KEEP。';
if(r.intents_available)for(const name of ['ADD','REMOVE','KEEP'])document.getElementById(name).src=dir+name+'.png';
document.getElementById('native').href=dir+'native.npz';document.getElementById('raw').textContent=JSON.stringify(r,null,2);}
function choose(){step.replaceChildren(...rows.filter(x=>x.method===method.value).map(r=>new Option(r.click,r.click)));update();}
method.onchange=choose;step.onchange=update;choose();</script></html>'''
        page = page.replace("OBJECT_TITLE", html.escape(str(self.sample["id"]))).replace("EMBEDDED_DATA", data)
        (self.folder / "index.html").write_text(page, encoding="utf-8")
        self.objects.append({"index": self.index, "id": self.sample["id"], "directory": self.folder.name})

    def finalize(self, records, max_clicks):
        rounds = round_summary(records, max_clicks)
        write_csv(self.root / "round_summary.csv", rounds)
        write_csv(self.root / "round_metrics.csv", [{key: value for key, value in row.items() if key != "intent_statistics"} for row in self.rows])
        with (self.root / "intent_statistics.jsonl").open("w", encoding="utf-8") as stream:
            for row in self.rows:
                if row["intents_available"]:
                    stream.write(json.dumps({key: row[key] for key in ("index", "id", "method", "click", "intent_statistics")}) + "\n")
        pooled = {}
        for method in ("a2", "a3"):
            edits = [row for row in self.rows if row["method"] == method and 2 <= row["click"] <= 5]
            pooled[method] = {"actual_edits": len(edits), **{key: sum(row[key] for row in edits) for key in ("tp_lost", "tn_lost", "fn_fixed", "fp_fixed")}, "groups": {}}
            for group in ("tp", "tn", "fn", "fp"):
                values = [row["intent_statistics"]["groups"][group] for row in edits if row["intents_available"]]
                count = sum(item["pixels"] for item in values)
                pooled[method]["groups"][group] = {"pixels": count,
                    **{key: sum(item[key] * item["pixels"] for item in values if item["pixels"]) / count if count else None
                       for key in (*CHANNELS, "effective_add", "effective_remove", "probability_delta")},
                    **{key: sum(item[key] for item in values) for key in ("foreground_crossings", "background_crossings")}}
        summary = {"rounds": rounds, "edit_analysis_clicks_2_to_5": pooled,
                   "objects": self.objects, "actual_rounds_saved": len(self.rows),
                   "interpretation": "Descriptive evidence from fixed weights and adaptive clicks, not causal attribution to a training loss."}
        (self.root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        columns = ("method", "click", "iou", "iou_delta", "foreground_ratio", "gt_foreground_ratio", "pred_to_gt_area_ratio", "precision", "recall", "degraded_from_first_fraction")
        table = '<table><tr>' + ''.join(f'<th>{key}</th>' for key in columns) + '</tr>'
        for row in rounds:
            table += '<tr>' + ''.join(f'<td>{row[key]:.5f}</td>' if isinstance(row[key], float) else f'<td>{row[key]}</td>' for key in columns) + '</tr>'
        table += '</table>'
        ranked = sorted(self.objects, key=lambda obj: next(row["iou"] for row in reversed(self.rows) if row["index"] == obj["index"] and row["method"] == "a3" and row["click"] <= 5)
                        - next(row["iou"] for row in self.rows if row["index"] == obj["index"] and row["method"] == "a3" and row["click"] == 1))
        links = ''.join(f'<li><a href="{obj["directory"]}/index.html">{html.escape(str(obj["id"]))}</a></li>' for obj in ranked)
        page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>交互评估诊断</title><style>body{font:15px/1.6 system-ui;margin:32px;color:#18232e}table{border-collapse:collapse}td,th{border:1px solid #dce3e8;padding:6px}th{background:#edf2f7}pre{white-space:pre-wrap}</style>'
        page += '<h1>A0–A3 逐轮诊断</h1><p>比率采用 0–1；首击 ΔIoU 为 null。每轮汇总包含同一对象全集，提前停止的 Mask 沿用最后结果，ΔIoU 记 0；不伪造图片或编辑意图。点击实例后可切换方法、轮次并查看热力图。</p>' + table
        page += '<h2>A2/A3 第 2–5 击编辑统计</h2><p>原图统计区分 TP 丢失与 TN 丢失。原生 256 网格意图均值排除补边，按旧 TP/TN/FN/FP 分组并按像素加权。有效添加=A×(1−旧概率)，有效删除=REMOVE×旧概率；不能仅根据 KEEP 的大小推断 Mask 是否被破坏。</p><pre>' + html.escape(json.dumps(pooled, ensure_ascii=False, indent=2)) + '</pre>'
        page += '<h2>实例列表：按 A3 首击到第五击退化程度排序</h2><ol>' + links + '</ol></html>'
        (self.root / "index.html").write_text(page, encoding="utf-8")
        return summary
