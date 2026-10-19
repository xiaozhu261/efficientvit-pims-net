"""A3 inference-only update damping with a shared, prerecorded click sequence."""

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluation.experiment import file_identity
from pims_net.evaluation.metrics import mask_iou, segmentation_statistics
from pims_net.models import PIMSNet
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT, checkpoint_path
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.interactive_losses import prepare_target

ETAS = (0., .1, .25, .5, 1.)


@torch.no_grad()
def refine_with_eta(model, previous, points, labels, eta):
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError('eta must be finite and in [0,1]')
    proposal = model.refine(previous, points, labels)
    if eta == 1:
        return proposal  # Bit-identical to the original implementation.
    if eta == 0:
        # Keep recurrent/click updates, but preserve raw logits as well as P.
        # A redundant logit(P, eps) would clip frozen-SAM endpoints.
        return replace(proposal, probabilities=previous.probabilities, logits=previous.logits)
    probability = previous.probabilities + eta * (proposal.probabilities - previous.probabilities)
    return replace(proposal, probabilities=probability, logits=torch.logit(probability, eps=1e-6))


def band_statistics(probability, target, valid=None):
    probability, target = np.asarray(probability), np.asarray(target, bool)
    valid = np.ones_like(target) if valid is None else np.asarray(valid, bool)
    if probability.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Probability, GT, and valid mask must align')
    band = (probability >= .49) & (probability <= .51)
    result = {}
    for name, selected in (('all', valid), ('fg', target & valid), ('bg', ~target & valid)):
        result[f'band_{name}'] = float(band[selected].mean()) if selected.any() else None
        result[f'mean_probability_{name}'] = float(probability[selected].mean()) if selected.any() else None
    return result


def state_metrics(model, state, original_gt, native_gt, valid):
    probability = model.sam_initializer.postprocess(state.logits, state.image_sizes)[0].sigmoid()[0, 0].cpu().numpy()
    prediction = probability > .5
    native = state.probabilities[0, 0].cpu().numpy()
    metrics = dict(iou=mask_iou(prediction, original_gt), **segmentation_statistics(prediction, original_gt))
    metrics.update({f'original_{key}': value for key, value in band_statistics(probability, original_gt).items()})
    metrics.update({f'native_{key}': value for key, value in band_statistics(native, native_gt, valid).items()})
    return metrics, prediction, native


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    reference, output = Path(args.reference).resolve(), Path(args.output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError('Output must be a run subdirectory of pims_net/outputs')
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((reference / 'protocol.json').read_text(encoding='utf-8'))
    dataset = InteractiveDataset(protocol['manifest']['path'])
    checkpoint = protocol['checkpoints']['a3']['path']
    for path, expected in ((dataset.manifest, protocol['manifest']), (checkpoint_path(), protocol['baseline']),
                           (checkpoint, protocol['checkpoints']['a3'])):
        if file_identity(path) != expected:
            raise ValueError('Inputs differ from saved A3 evaluation')
    records = [json.loads(line) for line in (reference / 'per_object.jsonl').read_text(encoding='utf-8').splitlines()]
    records = [r for r in records if r['method'] == 'a3']
    if len(records) != len(dataset) or any(len(r['trace']) < 5 for r in records):
        raise ValueError('This fixed-five-click experiment requires five recorded A3 clicks per object')
    watched = [Path(identity['path']) for identity in (protocol['baseline'], *protocol['checkpoints'].values())]
    watched += sorted((RESEARCH_ROOT / 'models').glob('*.py'))
    before = {str(path): file_identity(path)['sha256'] for path in watched}
    torch.set_num_threads(4)
    torch.manual_seed(protocol['seed'])
    model = PIMSNet(variant='a3').to(args.device).eval()
    info = load_training_checkpoint(checkpoint, model)
    del info
    model.requires_grad_(False)
    output.mkdir(parents=True)
    metadata = dict(etas=ETAS, update='P_scaled=P_old+eta*(P_edit-P_old)',
                    feedback='Each eta has its own recurrent state; scaled P/logits feed the next refinement.',
                    click_source=file_identity(reference / 'per_object.jsonl'),
                    fixed_click_policy='First five ORIGINAL A3 clicks, identical across eta; no adaptive reselection.',
                    eta_zero='Preserve previous P and raw logits exactly, while updating hidden state and click history.',
                    eta_one='Return original model.refine result exactly.',
                    probability_band=[.49, .51], aggregation='Macro means over identical objects; native excludes padding.',
                    comparison='Fixed-click sensitivity experiment, not a new adaptive benchmark or a validation-selected eta.',
                    checkpoint=protocol['checkpoints']['a3'], baseline=protocol['baseline'], manifest=protocol['manifest'],
                    device=args.device, seed=protocol['seed'], trained=False)
    (output / 'protocol.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    all_rows, clicks_saved = [], []
    hist_edges = np.linspace(0, 1, 201)
    histograms = {(eta, step): np.zeros(200, np.int64) for eta in ETAS for step in range(1, 6)}
    max_native_replay_error = max_metric_replay_error = 0.
    zero_mask_changes = 0
    with torch.no_grad(), (output / 'per_object.jsonl').open('w', encoding='utf-8', buffering=1) as stream:
        for index in range(len(dataset)):
            sample, ref = dataset[index], records[index]
            if ref['id'] != sample['id'] or ref['index'] != index:
                raise ValueError('Reference object order differs from manifest')
            clicks = [{key: step[key] for key in ('x', 'y', 'label')} for step in ref['trace'][:5]]
            clicks_saved.append(dict(index=index, id=sample['id'], clicks=clicks))
            first = clicks[0]
            initial = model.initialize(sample['image'], [[[first['x'], first['y']]]], [[first['label']]])
            target, valid_tensor = prepare_target(sample['mask'], 'cpu')
            gt, valid = target[0, 0].numpy().astype(bool), valid_tensor[0, 0].numpy().astype(bool)
            initial_metrics, initial_mask, _ = state_metrics(model, initial, sample['mask'], gt, valid)
            for eta in ETAS:
                state = initial
                for step, click in enumerate(clicks, 1):
                    if step > 1:
                        state = refine_with_eta(model, state, [[[click['x'], click['y']]]], [[click['label']]], eta)
                    metrics, prediction, native = state_metrics(model, state, sample['mask'], gt, valid)
                    if not torch.isfinite(state.hidden).all():
                        raise AssertionError('Nonfinite recurrent state')
                    if state.points.shape[1] != step:
                        raise AssertionError('Click history not advanced')
                    if eta == 0:
                        zero_mask_changes += int(np.count_nonzero(prediction != initial_mask))
                        assert torch.equal(state.logits, initial.logits) and torch.equal(state.probabilities, initial.probabilities)
                    if eta == 1:
                        prior = ref['trace'][step - 1]
                        for key in ('iou', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn'):
                            error = abs(metrics[key] - prior[key])
                            max_metric_replay_error = max(max_metric_replay_error, error)
                            if error != 0:
                                raise AssertionError(f'eta=1 metric replay mismatch: {index}, {step}, {key}')
                        with np.load(reference / 'diagnostics' / f'object_{index:04d}' / 'a3' / f'round_{step:02d}' / 'native.npz') as prior_native:
                            error = float(np.abs(native - prior_native['probabilities'][0]).max())
                            max_native_replay_error = max(max_native_replay_error, error)
                            if error != 0:
                                raise AssertionError(f'eta=1 native replay mismatch: {index}, {step}')
                    row = dict(index=index, id=sample['id'], eta=eta, click=step, **click, **metrics,
                               iou_minus_first=metrics['iou'] - initial_metrics['iou'])
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    all_rows.append(row)
                    histograms[eta, step] += np.histogram(native[gt & valid], bins=200, range=(0, 1))[0]
            if (index + 1) % 16 == 0:
                print(f'Eta sweep {index + 1}/{len(dataset)} objects', flush=True)
    after = {str(path): file_identity(path)['sha256'] for path in watched}
    assert before == after
    assert zero_mask_changes == 0
    summary = []
    for eta in ETAS:
        for step in range(1, 6):
            selected = [r for r in all_rows if r['eta'] == eta and r['click'] == step]
            keys = list(initial_metrics) + ['iou_minus_first']
            means = {key: float(np.mean([r[key] for r in selected if r[key] is not None]))
                     if any(r[key] is not None for r in selected) else None for key in keys}
            summary.append(dict(eta=eta, click=step, objects=len(selected), **means))
    verification = dict(objects=len(dataset), eta_groups=len(ETAS), recorded_rounds=len(all_rows),
                        eta_zero_mask_changed_pixels=zero_mask_changes, eta_one_metric_max_error=max_metric_replay_error,
                        eta_one_native_probability_max_error=max_native_replay_error,
                        model_sources_and_weights_unchanged=True, hashes=before)
    for name, data in (('summary', summary), ('verification', verification), ('fixed_clicks', clicks_saved)):
        (output / f'{name}.json').write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    write_csv(output / 'summary.csv', summary)
    np.savez_compressed(output / 'native_fg_histograms.npz', edges=hist_edges,
                        **{f'eta_{eta}_click_{step}': values for (eta, step), values in histograms.items()})
    report(output, summary)
    print(json.dumps({key: value for key, value in verification.items() if key != 'hashes'}, indent=2))


def report(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for eta in ETAS:
        selected = [r for r in rows if r['eta'] == eta]
        for ax, key, title in zip(axes, ('iou', 'precision', 'native_band_fg'),
                                  ('IoU (%)', 'Precision (%)', 'Native GT foreground in [0.49,0.51] (%)')):
            ax.plot([r['click'] for r in selected], [100 * r[key] for r in selected], 'o-', label=f'eta={eta:g}')
            ax.set(title=title, xlabel='Click', xticks=range(1, 6))
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('A3 fixed-click inference: same weights, scaled update fed back each round')
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>A3 eta 敏感性实验</title><style>body{font:16px/1.7 system-ui;margin:32px;color:#17232e}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}img{max-width:100%}</style><h1>A3 固定点击 eta 实验</h1>'
    page += '<p>使用原 A3 的五击序列，五组坐标和标签完全相同。每轮先重新计算编辑与 ConvGRU，再缩放概率更新，缩放结果反馈至下一轮。没有重新训练。eta=0 保留首击概率和原始 logits；eta=1 原样复现已有 A3。该实验不等同于各 eta 独立自适应选点的性能比较。</p>'
    fifth = {row['eta']: row for row in rows if row['click'] == 5}
    best = max((row for row in fifth.values() if row['eta'] > 0), key=lambda row: row['iou'])
    difference = 100 * (best['iou'] - fifth[0.]['iou'])
    page += f'<p><strong>本次结果：</strong>最高的非零 eta 第五击 IoU 为 {100 * best["iou"]:.4f}%（eta={best["eta"]:g}），相对不更新 Mask 的 eta=0 变化 {difference:+.4f} 个百分点。eta=1 的第五击 IoU 为 {100 * fifth[1.]["iou"]:.4f}%。这比较的是固定权重、固定点击下的更新步长效应。</p>'
    page += '<img src="comparison.png"><p>下表为对象宏平均，指标以百分数表示。0.5 聚集定义为概率落在 [0.49,0.51]；同时给出原生 GT 前景（排除 padding）与原图 GT 前景结果。完整逐轮表另见 summary.csv。</p><table><tr><th>eta</th><th>轮次</th><th>IoU</th><th>Precision</th><th>Recall</th><th>原生前景聚集</th><th>原图前景聚集</th></tr>'
    for row in rows:
        if row['click'] in (3, 5):
            page += '<tr>' + ''.join(f'<td>{row[key]}</td>' for key in ('eta', 'click')) + ''.join(f'<td>{100 * row[key]:.2f}%</td>' for key in ('iou', 'precision', 'recall', 'native_band_fg', 'original_band_fg')) + '</tr>'
    page += '</table><p>较小 eta 即使优于 eta=1，也应与 eta=0 比较，区分“减缓对首击的破坏”和“带来新的纠错收益”。所有值来自同一个训练种子，不构成跨种子结论。</p></html>'
    (output / 'report.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, help='Completed --diagnostics evaluation containing A3 native arrays')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
