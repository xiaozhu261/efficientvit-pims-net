"""Inference-only strength sweep on the fixed 16 memorization objects."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluate_eta import state_metrics, write_csv
from pims_net.evaluation.experiment import file_identity
from pims_net.models.directional_pims_net import DirectionalPIMSNet
from pims_net.overfit_directional import describe, dump
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT, checkpoint_path
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.interactive_losses import prepare_target

MULTIPLIERS = (1, 2, 4, 8)


@torch.no_grad()
def refine_scaled(model, previous, points, labels, multiplier):
    if multiplier not in MULTIPLIERS:
        raise ValueError('Supported strength multipliers: 1/2/4/8')
    proposal = model.refine(previous, points, labels)
    effective = (proposal.strength * multiplier).clamp(max=1)
    if multiplier == 1:
        return proposal, effective
    probability, logits = model.directional_editor.update(
        previous.probabilities, previous.logits, proposal.direction, effective)
    # Retain raw predicted S for diagnostics; updated P/logits feed next round.
    return replace(proposal, probabilities=probability, logits=logits), effective


def edit_counts(reference, current, truth):
    reference, current, truth = [np.asarray(x, bool) for x in (reference, current, truth)]
    if reference.shape != current.shape or reference.shape != truth.shape:
        raise ValueError('Masks must align')
    fn, fp, correct = ~reference & truth, reference & ~truth, reference == truth
    return dict(fn_before=int(fn.sum()), fn_repaired=int((fn & current).sum()),
                fp_before=int(fp.sum()), fp_removed=int((fp & ~current).sum()),
                correct_before=int(correct.sum()), correct_damaged=int((correct & (current != truth)).sum()))


def rates(counts):
    return {name: counts[numerator] / counts[denominator] if counts[denominator] else None
            for name, numerator, denominator in (
                ('fn_repair_rate', 'fn_repaired', 'fn_before'),
                ('fp_removal_rate', 'fp_removed', 'fp_before'),
                ('damage_rate', 'correct_damaged', 'correct_before'))}


def add_distribution(pooled, key, values, low, high):
    stats = describe(values, low, high)
    if stats is None:
        return None
    bucket = pooled.setdefault(key, dict(count=0, total=0., squares=0., histogram=np.zeros(20, np.int64),
                                         low=low, high=high, near_zero=0, saturated=0))
    values = np.asarray(values, np.float64)
    bucket['count'] += values.size
    bucket['total'] += float(values.sum())
    bucket['squares'] += float(np.square(values).sum())
    bucket['histogram'] += np.asarray(stats['histogram'])
    bucket['near_zero'] += int((np.abs(values) <= .05).sum())
    bucket['saturated'] += int((values == 1).sum())
    return stats


def run(args):
    reference, output = Path(args.reference).resolve(), Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError('Use a new subdirectory under pims_net/outputs')
    checkpoint = reference / 'last.pt'
    manifest = reference / 'fixed_train16.jsonl'
    status = json.loads((reference / 'status.json').read_text())
    if status['status'] != 'completed':
        raise ValueError('Reference training must be completed')
    records = []
    with (reference / 'rounds.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['phase'] == 'eval' and row['epoch'] == status['epochs']:
                records.append(row)
    dataset = InteractiveDataset(manifest)
    assert len(dataset) == 16 and len(records) == 80
    watched = [checkpoint, reference / 'best.pt', checkpoint_path(), manifest, reference / 'rounds.jsonl']
    watched += sorted((RESEARCH_ROOT / 'models').glob('*.py'))
    original = json.loads((OUTPUT_ROOT / 'directional_implementation_20261018/original_hashes.json').read_text())
    before = {str(path): file_identity(path)['sha256'] for path in watched}
    for path, sha in original.items():
        assert file_identity(path)['sha256'] == sha
    before.update(original)
    torch.set_num_threads(4)
    model = DirectionalPIMSNet().to(args.device).eval()
    info = load_training_checkpoint(checkpoint, model)
    assert info['epoch'] == status['epochs'] and info['config']['threshold'] == .5
    model.requires_grad_(False)
    output.mkdir(parents=True)
    dump(output / 'protocol.json', dict(checkpoint=file_identity(checkpoint), epoch=info['epoch'],
        manifest=file_identity(manifest), clicks_source=file_identity(reference / 'rounds.jsonl'),
        threshold=.5, multipliers=MULTIPLIERS, strength='S_effective=min(multiplier*S_predicted,1)',
        feedback='Each multiplier uses its own updated P/logits and recurrent state; first state reset per image.',
        clicks='Recorded final-epoch multiplier-1 clicks, identical coordinates and labels for all groups.',
        metrics='Original resolution P>0.5; IoU/Precision/Recall object macro mean. Repair/removal/damage pooled pixel ratios.',
        reference='Main table: versus first-click mask. Also save each transition versus previous mask.',
        distributions='Native valid pixels, excludes padding; TP/TN/FP/FN from own previous P>0.5; predicted and effective strength separate.',
        interpretation='Training16 fixed-click sensitivity diagnostic, not generalization performance. No training or weight writes.',
        trained=False, device=args.device))
    del info
    rows, clicks_saved, pooled = [], [], {}
    replay_error = 0.
    with torch.no_grad(), (output / 'per_object.jsonl').open('w', encoding='utf-8') as stream:
        for index in range(16):
            sample = dataset[index]
            saved = sorted([r for r in records if r['index'] == index], key=lambda r: r['round'])
            assert [r['round'] for r in saved] == list(range(1, 6)) and all(r['id'] == sample['id'] for r in saved)
            clicks = [{k: r[k] for k in ('x', 'y', 'label')} for r in saved]
            clicks_saved.append(dict(index=index, id=sample['id'], clicks=clicks))
            c = clicks[0]
            initial = model.initialize(sample['image'], [[[c['x'], c['y']]]], [[c['label']]])
            target, valid_t = prepare_target(sample['mask'], 'cpu')
            truth, valid = target[0, 0].numpy().astype(bool), valid_t[0, 0].numpy().astype(bool)
            _, initial_mask, _ = state_metrics(model, initial, sample['mask'], truth, valid)
            for multiplier in MULTIPLIERS:
                state, previous_mask = initial, initial_mask
                for step, click in enumerate(clicks, 1):
                    distributions = None
                    if step > 1:
                        old = state.probabilities[0, 0].cpu().numpy() > .5
                        state, effective = refine_scaled(model, state, [[[click['x'], click['y']]]], [[click['label']]], multiplier)
                        regions = dict(all=valid, TP=valid & truth & old, TN=valid & ~truth & ~old,
                                       FP=valid & ~truth & old, FN=valid & truth & ~old)
                        fields = dict(direction=state.direction, strength_raw=state.strength, strength_effective=effective)
                        arrays = {k: v[0, 0].cpu().numpy() for k, v in fields.items()}
                        distributions = {region: {field: add_distribution(pooled, (multiplier, step, region, field),
                            array[selection], -1 if field == 'direction' else 0, 1)
                            for field, array in arrays.items()} for region, selection in regions.items()}
                    metrics, prediction, _ = state_metrics(model, state, sample['mask'], truth, valid)
                    assert state.points.shape[1] == step and torch.isfinite(state.hidden).all()
                    assert not any(p.grad is not None for p in model.parameters())
                    if multiplier == 1:
                        replay_error = max(replay_error, abs(metrics['iou'] - saved[step - 1]['iou']))
                        assert metrics['iou'] == saved[step - 1]['iou']
                    first = edit_counts(initial_mask, prediction, sample['mask'])
                    previous = edit_counts(previous_mask, prediction, sample['mask'])
                    row = dict(index=index, id=sample['id'], multiplier=multiplier, round=step, **click,
                               **metrics, versus_first=dict(**first, **rates(first)),
                               versus_previous=dict(**previous, **rates(previous)), distributions=distributions)
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    rows.append(row)
                    previous_mask = prediction
            print(f'Strength sweep {index + 1}/16', flush=True)
    summary = []
    for multiplier in MULTIPLIERS:
        for step in range(1, 6):
            selected = [r for r in rows if r['multiplier'] == multiplier and r['round'] == step]
            row = dict(multiplier=multiplier, round=step, objects=len(selected),
                       **{k: float(np.mean([r[k] for r in selected])) for k in ('iou', 'precision', 'recall', 'native_band_fg')})
            for reference_name in ('versus_first', 'versus_previous'):
                counts = {k: sum(r[reference_name][k] for r in selected) for k in first}
                row.update({reference_name + '_' + k: v for k, v in {**counts, **rates(counts)}.items()})
            summary.append(row)
    distribution_rows = []
    for (multiplier, step, region, field), bucket in sorted(pooled.items()):
        count = bucket['count']
        assert int(bucket['histogram'].sum()) == count
        mean = bucket['total'] / count
        distribution_rows.append(dict(multiplier=multiplier, round=step, region=region, field=field,
            count=count, mean=mean, std=float(np.sqrt(max(0, bucket['squares'] / count - mean**2))),
            near_zero_fraction=bucket['near_zero'] / count, saturated_at_one_fraction=bucket['saturated'] / count,
            histogram=bucket['histogram'].tolist(), bin_edges=np.linspace(bucket['low'], bucket['high'], 21).tolist()))
    after = {path: file_identity(path)['sha256'] for path in before}
    assert before == after
    assert len(rows) == 320 and all(r['objects'] == 16 for r in summary)
    dump(output / 'verification.json', dict(objects=16, rounds=320, multiplier_one_iou_replay_max_error=replay_error,
        sources_and_weights_unchanged=True, no_gradients=True, hashes=before))
    for name, data in (('fixed_clicks', clicks_saved), ('summary', summary), ('distributions', distribution_rows)):
        dump(output / (name + '.json'), data)
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'distributions.csv', [{**r, 'histogram': json.dumps(r['histogram']), 'bin_edges': json.dumps(r['bin_edges'])} for r in distribution_rows])
    comparison = []
    for multiplier in MULTIPLIERS:
        selected = {r['round']: r for r in summary if r['multiplier'] == multiplier}
        final = selected[5]
        paired = [r['iou'] - next(base['iou'] for base in rows if base['index'] == r['index'] and base['multiplier'] == 1 and base['round'] == 5)
                  for r in rows if r['multiplier'] == multiplier and r['round'] == 5]
        comparison.append(dict(multiplier=multiplier, **{f'IoU@{t}': selected[t]['iou'] for t in (1, 3, 5)},
            **{k: final['versus_first_' + k] for k in ('fn_repair_rate', 'fp_removal_rate', 'damage_rate')},
            precision=final['precision'], recall=final['recall'], better_than_x1=sum(d > 0 for d in paired),
            worse_than_x1=sum(d < 0 for d in paired), equal_to_x1=sum(d == 0 for d in paired)))
    write_csv(output / 'comparison.csv', comparison)
    dump(output / 'comparison.json', comparison)
    report(output, summary, comparison, distribution_rows)
    print(json.dumps(comparison, indent=2))


def report(output, summary, comparison, distributions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for multiplier in MULTIPLIERS:
        selected = [r for r in summary if r['multiplier'] == multiplier]
        for ax, key, title in zip(axes, ('iou', 'versus_first_fn_repair_rate', 'versus_first_damage_rate'),
                                  ('IoU (%)', 'Initial FN repaired (%)', 'Initial correct pixels damaged (%)')):
            ax.plot(range(1, 6), [100 * r[key] for r in selected], 'o-', label=f'x{multiplier}')
            ax.set(title=title, xlabel='Click', xticks=range(1, 6))
            ax.grid(alpha=.2)
            ax.legend()
    fig.savefig(output / 'comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, field in zip(axes, ('direction', 'strength_raw', 'strength_effective')):
        for multiplier in MULTIPLIERS:
            r = next(r for r in distributions if r['multiplier'] == multiplier and r['round'] == 5 and r['region'] == 'all' and r['field'] == field)
            edges = np.asarray(r['bin_edges'])
            ax.stairs(np.asarray(r['histogram']) / r['count'], edges, label=f'x{multiplier}')
        ax.set(title=f'Round 5: {field}', ylabel='Pixel fraction')
        ax.legend()
    fig.savefig(output / 'distributions.png', dpi=150)
    plt.close(fig)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Directional Strength 倍率诊断</title><style>body{font:16px/1.7 system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}img{max-width:100%}</style><h1>DirectionalPIMSNet：固定点击 Strength 倍率</h1>'
    page += '<p>使用已有第100 epoch权重和同一16个训练实例，统一阈值 P&gt;0.5。复用最终评估中倍率1的五击坐标与标签；每个倍率独立反馈修改后的Mask与ConvGRU状态。只做推理，S有效=min(倍率×S预测,1)。倍率1全部80轮IoU精确复现。原A3、模型源码及权重哈希不变。</p>'
    page += '<p>IoU为16对象宏平均；FN修复、FP删除、正确区域破坏为原图像素汇总比率，以下均比较首击与第五击。分母分别为首击FN、首击FP、首击正确像素。正确区域同时包含TP/TN，因此面积大的背景会影响破坏率；逐对象计数及逐轮增量另见per_object.jsonl、summary.csv。</p><table><tr><th>倍率</th><th>IoU@1</th><th>IoU@3</th><th>IoU@5</th><th>FN修复</th><th>FP删除</th><th>正确区域破坏</th><th>相对×1改善/退化/持平对象</th></tr>'
    for r in comparison:
        page += f'<tr><td>{r["multiplier"]}</td>' + ''.join(f'<td>{100*r[k]:.3f}%</td>' for k in ('IoU@1', 'IoU@3', 'IoU@5', 'fn_repair_rate', 'fp_removal_rate', 'damage_rate'))
        page += f'<td>{r["better_than_x1"]}/{r["worse_than_x1"]}/{r["equal_to_x1"]}</td></tr>'
    page += '</table><img src="comparison.png"><img src="distributions.png"><p>Direction及原始Strength在后续轮次可因不同Mask反馈而改变，并非冻结原始预测后反复放大。分布排除padding，另存每轮all/TP/TN/FP/FN的均值、标准差、直方图、有效Strength饱和比例；逐对象含精确分位数。该实验仅是16个训练实例上的固定点击敏感性诊断，不代表泛化性能或跨种子稳定提升。</p>'
    page += '<h2>第五轮分布及正确前景损伤</h2><table><tr><th>倍率</th><th>全图预测S均值</th><th>全图有效S均值</th><th>有效S=1比例</th><th>FN方向均值</th><th>FN有效S均值</th><th>首击TP被破坏比例</th></tr>'
    object_rows = [json.loads(line) for line in (output / 'per_object.jsonl').read_text(encoding='utf-8').splitlines()]
    for multiplier in MULTIPLIERS:
        def stat(region, field):
            return next(r for r in distributions if r['multiplier'] == multiplier and r['round'] == 5 and r['region'] == region and r['field'] == field)
        first = {r['index']: r for r in object_rows if r['multiplier'] == multiplier and r['round'] == 1}
        final = [r for r in object_rows if r['multiplier'] == multiplier and r['round'] == 5]
        damaged_tp = sum(r['fn'] - first[r['index']]['fn'] + r['versus_first']['fn_repaired'] for r in final)
        damage = damaged_tp / sum(r['tp'] for r in first.values())
        page += f'<tr><td>{multiplier}</td><td>{stat("all", "strength_raw")["mean"]:.4f}</td><td>{stat("all", "strength_effective")["mean"]:.4f}</td><td>{100*stat("all", "strength_effective")["saturated_at_one_fraction"]:.2f}%</td><td>{stat("FN", "direction")["mean"]:+.4f}</td><td>{stat("FN", "strength_effective")["mean"]:.4f}</td><td>{100*damage:.2f}%</td></tr>'
    page += '</table><p>在本次数据中，增大Strength同时提高FP删除和FN修复，但第五击IoU反而降低；损伤主要集中于首击已正确的前景。×2第三击略有改善，后续收益未维持。结果不支持“仅将Strength放大即可解决欠拟合”的解释。FN方向均值仍为正不代表每个FN都方向正确；各倍率的FN区域组成也会改变，不能将均值差直接解释为同一像素方向退化。</p></html>'
    (output / 'report.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default=str(OUTPUT_ROOT / 'directional_overfit16_seed42_20261019'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
