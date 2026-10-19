"""Read saved probabilities to diagnose competing edits; no model forward."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT
from pims_net.training.interactive_losses import prepare_target

EPS = 1e-6
GROUPS = ('all', 'active', 'tp', 'tn', 'fp', 'fn')
Q_EDGES = np.linspace(0, 1, 201)
GAP_EDGES = np.linspace(-1, 1, 401)


def pixel_fields(old, new, q, target, valid):
    old, new = np.asarray(old).squeeze(0), np.asarray(new).squeeze(0)
    target, valid = np.asarray(target, bool), np.asarray(valid, bool)
    if old.shape != target.shape or new.shape != target.shape or valid.shape != target.shape or q.shape != (3, *target.shape):
        raise ValueError('Native arrays and GT must have matching shapes')
    if not all(np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all() for x in (old, new, q)):
        raise ValueError('Expected finite probabilities in [0,1]')
    a, r, k = q
    normalization = float(np.abs(a + r + k - 1).max())
    formula = float(np.abs(new - (k * old + a)).max())
    centered = float(np.abs((new - .5) - (k * (old - .5) + .5 * (a - r))).max())
    if max(normalization, formula, centered) > 1e-5:
        raise ValueError('Saved probabilities violate edit identities')
    u, v = a * (1 - old), r * old
    delta = new - old
    gap = a - r
    active = (a + r) > .5
    coactive = (a > .25) & (r > .25)
    # Measure distances in float64 so the +/- epsilon comparisons form an
    # exact partition even for float32 probabilities near the tolerance.
    before, after = np.abs(old.astype(np.float64) - .5), np.abs(new.astype(np.float64) - .5)
    distance_delta = after - before
    shrink = distance_delta < -EPS
    fields = dict(q_add=a, q_remove=r, q_keep=k, gap=gap, abs_gap=np.abs(gap),
                  old_probability=old, new_probability=new, delta=delta,
                  effective_add=u, effective_remove=v, gross_edit=u + v,
                  canceled_edit=2 * np.minimum(u, v), distance_before=before, distance_after=after,
                  distance_delta=distance_delta, gt_error_delta=np.where(target, -delta, delta),
                  shrink=shrink, expand=distance_delta > EPS,
                  unchanged_distance=np.abs(distance_delta) <= EPS,
                  off_center=before > EPS, active=active, coactive=coactive,
                  balanced_001=active & (np.abs(gap) < .01),
                  balanced_002=active & (np.abs(gap) < .02),
                  balanced_005=active & (np.abs(gap) < .05),
                  competing=coactive & (np.abs(gap) < .02),
                  competing_shrink=coactive & (np.abs(gap) < .02) & shrink,
                  add_advantage=gap > .02, remove_advantage=gap < -.02,
                  keep_center_component=np.abs(k * (old - .5)),
                  directional_center_component=.5 * np.abs(gap),
                  binary_flip=(old > .5) != (new > .5))
    foreground = old > .5
    regions = dict(all=valid, active=valid & active, tp=valid & foreground & target,
                   tn=valid & ~foreground & ~target, fp=valid & foreground & ~target,
                   fn=valid & ~foreground & target)
    return fields, regions, {'normalization': normalization, 'formula': formula, 'centered_identity': centered}


class Accumulator:
    def __init__(self):
        self.pixels = self.snapshots = 0
        self.sums, self.macro_sums = {}, {}
        self.histograms = {name: np.zeros(200 if name != 'gap' else 400, np.int64)
                           for name in ('q_add', 'q_remove', 'q_keep', 'gap')}
        self.joint = np.zeros((50, 50), np.int64)

    def add(self, fields, selected):
        count = int(selected.sum())
        if not count:
            return None
        self.pixels += count
        self.snapshots += 1
        means = {}
        for name, value in fields.items():
            total = float(np.sum(value[selected], dtype=np.float64))
            self.sums[name] = self.sums.get(name, 0.) + total
            means[name] = total / count
            self.macro_sums[name] = self.macro_sums.get(name, 0.) + means[name]
        for name, hist in self.histograms.items():
            hist += np.histogram(fields[name][selected], bins=400 if name == 'gap' else 200,
                                 range=(-1, 1) if name == 'gap' else (0, 1))[0]
        self.joint += np.histogram2d(fields['q_add'][selected], fields['q_remove'][selected],
                                     bins=50, range=((0, 1), (0, 1)))[0].astype(np.int64)
        return dict(pixels=count, **means)

    def result(self):
        if not self.pixels:
            return dict(pixels=0, snapshots=0)
        means = {key: value / self.pixels for key, value in self.sums.items()}
        ratio = lambda x, y: self.sums[x] / self.sums[y] if self.sums[y] else None
        quantiles = {}
        for name, hist in self.histograms.items():
            edges = GAP_EDGES if name == 'gap' else Q_EDGES
            cumulative = hist.cumsum()
            quantiles[name] = {}
            for percentile in (5, 25, 50, 75, 95):
                i = min(int(np.searchsorted(cumulative, self.pixels * percentile / 100)), len(hist) - 1)
                quantiles[name][str(percentile)] = [float(edges[i]), float(edges[i + 1])]
        return dict(pixels=self.pixels, snapshots=self.snapshots, **means,
                    macro={key: value / self.snapshots for key, value in self.macro_sums.items()},
                    cancellation_ratio=ratio('canceled_edit', 'gross_edit'),
                    shrink_off_center=ratio('shrink', 'off_center'),
                    coactive_given_active=ratio('coactive', 'active'),
                    balanced_001_given_active=ratio('balanced_001', 'active'),
                    balanced_002_given_active=ratio('balanced_002', 'active'),
                    balanced_005_given_active=ratio('balanced_005', 'active'),
                    competing_given_active=ratio('competing', 'active'),
                    shrink_given_competing=ratio('competing_shrink', 'competing'),
                    quantile_intervals=quantiles)


def hashes(paths):
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        result[str(path)] = digest.hexdigest()
    return result


def diagnose(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError('Output must be a new run directory beneath pims_net/outputs')
    if output.exists():
        raise FileExistsError(output)
    root = directory / 'diagnostics'
    source = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    identities = [protocol['baseline'], *protocol['checkpoints'].values()]
    watched = [Path(x['path']) for x in identities] + sorted((RESEARCH_ROOT / 'models').glob('*.py'))
    before = hashes(watched)
    for identity in identities:
        assert before[identity['path']] == identity['sha256'], 'Checkpoint changed since evaluation'
    output.mkdir(parents=True)
    bins = {(method, step, group): Accumulator() for method in ('a2', 'a3')
            for step in (2, 3, 4, 5, '2:5') for group in GROUPS}
    maximum = dict(normalization=0., formula=0., centered_identity=0.)
    rounds = 0
    with (output / 'per_object.jsonl').open('w', encoding='utf-8') as stream:
        for obj in source['objects']:
            folder = root / obj['directory']
            with Image.open(folder / 'gt.png') as image:
                target, valid = prepare_target(np.asarray(image) > 0, 'cpu')
            target, valid = target[0, 0].numpy().astype(bool), valid[0, 0].numpy().astype(bool)
            for method in ('a2', 'a3'):
                with np.load(folder / method / 'round_01/native.npz') as initial:
                    previous = initial['probabilities']
                for step in range(2, 6):
                    path = folder / method / f'round_{step:02d}/native.npz'
                    if not path.is_file():
                        break  # No fabricated updates for early-stopped objects.
                    with np.load(path) as native:
                        old, new, q = native['previous_probabilities'], native['probabilities'], native['intents']
                    if not np.array_equal(old, previous):
                        raise ValueError(f'Stale previous probability: {path}')
                    fields, regions, errors = pixel_fields(old, new, q, target, valid)
                    record = dict(index=obj['index'], id=obj['id'], method=method, click=step, groups={})
                    for group, selected in regions.items():
                        record['groups'][group] = bins[method, step, group].add(fields, selected)
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    for key in maximum:
                        maximum[key] = max(maximum[key], errors[key])
                    rounds += 1
                    previous = new
            if (obj['index'] + 1) % 32 == 0:
                print(f"Analyzed {obj['index'] + 1}/{len(source['objects'])} objects", flush=True)
    for method in ('a2', 'a3'):
        for group in GROUPS:
            pooled = bins[method, '2:5', group]
            for step in range(2, 6):
                acc = bins[method, step, group]
                pooled.pixels += acc.pixels
                pooled.snapshots += acc.snapshots
                for field in acc.sums:
                    pooled.sums[field] = pooled.sums.get(field, 0.) + acc.sums[field]
                    pooled.macro_sums[field] = pooled.macro_sums.get(field, 0.) + acc.macro_sums[field]
                for field in pooled.histograms:
                    pooled.histograms[field] += acc.histograms[field]
                pooled.joint += acc.joint
    summary = [dict(method=method, click=step, region=group, **value.result())
               for (method, step, group), value in bins.items()]
    definitions = dict(source=str(directory), grid='native 256x256, padding excluded; old P>0.5 defines TP/TN/FP/FN',
                       aggregation='Main means/rates pooled by pixel occurrences; macro means over nonempty object-round regions also saved. 2:5 repeats pixels across four updates.',
                       active='q_add+q_remove>0.5', coactive='q_add>0.25 AND q_remove>0.25',
                       competing='coactive AND abs(q_add-q_remove)<0.02; descriptive cutoff, not a statistical hypothesis test',
                       balance_sensitivity='Also report abs gap<0.01 and <0.05 among active pixels',
                       shrink='abs(P_new-0.5)<abs(P_old-0.5)-1e-6; all selected pixels in denominator unless specified',
                       cancellation='sum(2*min(u,v))/sum(u+v), u=q_add*(1-P_old), v=q_remove*P_old; null for zero gross edit',
                       quantiles='Histogram quantile INTERVALS, bin width 0.005; no sampling. Means and rates use original float32 values with float64 accumulation.',
                       centered_identity='P_new-0.5=q_keep*(P_old-0.5)+0.5*(q_add-q_remove)',
                       interpretation='Shrinkage is not necessarily harmful: TP/TN confidence loss and FP/FN correction must be distinguished. Descriptive fixed-weight evidence; no training-cause or cross-seed claim.')
    after = hashes(watched)
    assert before == after, 'Model source or weight changed during analysis'
    verification = dict(objects=len(source['objects']), actual_edit_rounds=rounds, previous_probability_exact=True,
                        max_errors=maximum, watched_hashes_unchanged=True, hashes=before)
    for name, data in (('summary', summary), ('definitions', definitions), ('verification', verification)):
        (output / f'{name}.json').write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    flat = [{k: v for k, v in row.items() if k not in ('macro', 'quantile_intervals')} for row in summary]
    columns = list(dict.fromkeys(key for row in flat for key in row))
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    histogram_data = {f'{m}_{s}_{g}_{field}': counts for (m, s, g), acc in bins.items()
                      for field, counts in acc.histograms.items()}
    histogram_data.update({f'{m}_{s}_{g}_joint_add_remove': acc.joint for (m, s, g), acc in bins.items()})
    np.savez_compressed(output / 'distributions.npz', q_edges=Q_EDGES, gap_edges=GAP_EDGES, **histogram_data)
    render_report(output, summary, bins, definitions, verification)
    return verification


def render_report(output, rows, bins, definitions, verification):
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    centers = (Q_EDGES[:-1] + Q_EDGES[1:]) / 2
    for i, method in enumerate(('a2', 'a3')):
        for j, channel in enumerate(('q_add', 'q_remove', 'q_keep')):
            for step in range(2, 6):
                acc = bins[method, step, 'active']
                axes[i, j].plot(centers, acc.histograms[channel] / max(1, acc.pixels), label=f'Click {step}')
            axes[i, j].set(title=f'{method.upper()} {channel} | active pixels', xlabel='Probability', ylabel='Fraction per 0.005 bin', xlim=(0, 1))
            axes[i, j].legend()
            axes[i, j].grid(alpha=.2)
    fig.savefig(output / 'q_distributions.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    for i, method in enumerate(('a2', 'a3')):
        for j, step in enumerate(range(2, 6)):
            acc = bins[method, step, 'active']
            density = acc.joint.T / max(1, acc.pixels)
            im = axes[i, j].imshow(np.ma.masked_where(density == 0, density), origin='lower', extent=(0, 1, 0, 1), norm=LogNorm(1e-5, 1), cmap='viridis')
            axes[i, j].plot([0, 1], [0, 1], '--', color='red', lw=.8)
            axes[i, j].set(title=f'{method.upper()} click {step}', xlabel='q_add', ylabel='q_remove')
    fig.colorbar(im, ax=axes, label='Fraction of active pixels / 0.02 x 0.02 bin')
    fig.suptitle('Same-pixel ADD / REMOVE joint distributions (active regions)')
    fig.savefig(output / 'joint_competition.png', dpi=150)
    plt.close(fig)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双向编辑竞争诊断</title><style>body{font:16px/1.7 system-ui;margin:32px;color:#18232e}img{max-width:1400px;width:100%}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}pre{white-space:pre-wrap}</style><h1>A2/A3 双向编辑竞争诊断</h1>'
    page += '<p>只读取已有 256×256 概率，无模型前向。区域按上一轮预测划分，排除 padding；下表按像素次加权，不能与对象宏平均混淆。</p>'
    page += '<p>P_new−0.5 = q_keep×(P_old−0.5) + (q_add−q_remove)/2。当两种编辑均活跃而概率接近时，定向项很小，更新以向 0.5 收缩为主。有效增删的抵消程度另行统计；概率对称不代表有效增删在远离 0.5 的像素上恰好相等。</p>'
    lookup = {(r['method'], str(r['click']), r['region']): r for r in rows}
    page += '<h2>本次数据结论</h2><ul>'
    for method in ('a2', 'a3'):
        pooled = lookup[method, '2:5', 'active']
        start, end = lookup[method, '2', 'active'], lookup[method, '5', 'active']
        tp, tn = lookup[method, '2:5', 'tp'], lookup[method, '2:5', 'tn']
        fp, fn = lookup[method, '2:5', 'fp'], lookup[method, '2:5', 'fn']
        page += f'<li>{method.upper()}：第 2–5 轮活跃区中 {pooled["coactive_given_active"]:.2%} 的像素同时 q_add/q_remove&gt;0.25；{pooled["competing_given_active"]:.2%} 同时满足两者差值小于 0.02。有效增删抵消率由第 2 轮 {start["cancellation_ratio"]:.2%} 升至第 5 轮 {end["cancellation_ratio"]:.2%}，直接支持双向编辑竞争。</li>'
        page += f'<li>{method.upper()}：旧 TP 概率平均变化 {tp["delta"]:+.5f}，旧 TN {tn["delta"]:+.5f}，削弱正确区域置信度；旧 FP {fp["delta"]:+.5f}、旧 FN {fn["delta"]:+.5f}，则在均值上向正确方向移动。旧 FP 的 q_add−q_remove 仍为 {fp["gap"]:+.5f}，未形成稳定的 REMOVE 优势。</li>'
    page += '<li>竞争表现为概率通道对称、定向项弱，以及后期实际增删相互抵消。它不是算术或状态传递错误：保存的上一轮概率逐像素一致，更新恒等式误差不超过 float32 舍入量级。当前结果不能确定训练成因，亦不能据此证明某种替代架构有效。</li></ul>'
    columns = ('method', 'click', 'region', 'q_add', 'q_remove', 'q_keep', 'gap', 'abs_gap', 'shrink', 'cancellation_ratio')
    page += '<h2>总体及活跃区分布与收缩</h2>' + table_html([r for r in rows if r['region'] in ('all', 'active')], columns)
    columns = ('method', 'region', 'q_add', 'q_remove', 'q_keep', 'gap', 'old_probability', 'new_probability', 'delta', 'effective_add', 'effective_remove', 'shrink', 'gt_error_delta', 'binary_flip')
    page += '<h2>第 2–5 轮区域汇总</h2><p>shrink 为收缩比例；gt_error_delta 为相对 GT 的概率绝对误差变化，正数表示变差。binary_flip 在 TP/TN 上是破坏，在 FP/FN 上是修复。完整逐轮区域表见 summary.csv。</p>' + table_html([r for r in rows if r['click'] == '2:5' and r['region'] in ('tp', 'tn', 'fp', 'fn')], columns)
    columns = ('method', 'click', 'active', 'coactive_given_active', 'balanced_001_given_active', 'balanced_002_given_active', 'balanced_005_given_active', 'competing_given_active', 'shrink_given_competing')
    page += '<h2>同一像素的双向竞争与阈值敏感性</h2>' + table_html([r for r in rows if r['region'] == 'all'], columns)
    page += '<img src="q_distributions.png"><img src="joint_competition.png"><h2>定义与验证</h2><pre>' + html.escape(json.dumps(definitions, ensure_ascii=False, indent=2)) + '</pre>'
    page += '<pre>' + html.escape(json.dumps({k: v for k, v in verification.items() if k != 'hashes'}, indent=2)) + '</pre></html>'
    (output / 'report.html').write_text(page, encoding='utf-8')


def table_html(rows, columns):
    import html
    page = '<table><tr>' + ''.join(f'<th>{column}</th>' for column in columns) + '</tr>'
    for row in rows:
        page += '<tr>' + ''.join('<td>' + (f'{row[column]:.5f}' if isinstance(row.get(column), float) else html.escape(str(row.get(column)))) + '</td>' for column in columns) + '</tr>'
    return page + '</table>'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation_directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = diagnose(args.evaluation_directory, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'hashes'}, indent=2))
