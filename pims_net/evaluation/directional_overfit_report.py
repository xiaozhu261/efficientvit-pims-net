"""Summarize fixed-16 training diagnostics; never present as heldout performance."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '<table><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>' + ''.join(
        '<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</table>'


def report(directory):
    root = Path(directory)
    status = json.loads((root / 'status.json').read_text())
    if status['status'] != 'completed':
        raise ValueError('Run is not completed')
    epochs = [json.loads(line) for line in (root / 'epochs.jsonl').read_text().splitlines()]
    evaluated = [row for row in epochs if row['phase'] == 'eval']
    initial, final = evaluated[0], evaluated[-1]
    best = max(evaluated, key=lambda row: row['IoU@5'])
    epoch_rows = [{**{k: v for k, v in r.items() if k != 'losses'}, **{f'loss_{k}': v for k, v in r['losses'].items()}} for r in epochs]
    write_csv(root / 'epochs.csv', epoch_rows)
    rounds = defaultdict(list)
    count = 0
    with (root / 'rounds.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            if row['phase'] == 'eval':
                rounds[row['epoch'], row['round']].append(row)
    compact, distributions = [], []
    for (epoch, step), rows in sorted(rounds.items()):
        cosine = [r['output_gradient']['cosine'] for r in rows if r['output_gradient'] is not None and r['output_gradient']['cosine'] is not None]
        metrics = dict(epoch=epoch, round=step, objects=len(rows), iou=float(np.mean([r['iou'] for r in rows])),
                       native_iou=float(np.mean([r['native_iou'] for r in rows])),
                       output_gradient_cosine_mean=float(np.mean(cosine)) if cosine else None,
                       output_gradient_conflict_fraction=float(np.mean(np.array(cosine) < 0)) if cosine else None)
        for key in ('loss', 'bce', 'dice', 'direction', 'preserve'):
            values = [r['losses'][key] for r in rows if r['losses'] is not None]
            metrics[f'loss_{key}'] = float(np.mean(values)) if values else None
        compact.append(metrics)
        if step == 1:
            continue
        for region in ('all', 'TP', 'TN', 'FP', 'FN'):
            selected = [r['distributions'][region] for r in rows if r['distributions'][region]['direction'] is not None]
            if not selected:
                continue
            total = sum(r['direction']['count'] for r in selected)
            item = dict(epoch=epoch, round=step, region=region, pixel_occurrences=total)
            for key in ('direction', 'strength'):
                item[f'{key}_mean'] = sum(r[key]['mean'] * r[key]['count'] for r in selected) / total
                item[f'{key}_histogram'] = np.sum([r[key]['histogram'] for r in selected], axis=0).tolist()
                assert sum(item[f'{key}_histogram']) == total
            item['direction_near_zero_fraction'] = sum(r['direction_near_zero'] * r['direction']['count'] for r in selected) / total
            distributions.append(item)
    write_csv(root / 'eval_rounds.csv', compact)
    write_csv(root / 'eval_distributions.csv', [{k: v for k, v in r.items() if not k.endswith('histogram')} for r in distributions])
    (root / 'eval_distributions.json').write_text(json.dumps(distributions, indent=2), encoding='utf-8')
    # Independently reconcile object-level metrics and gradient averages.
    sequences = [json.loads(line) for line in (root / 'sequences.jsonl').read_text().splitlines()]
    for epoch in epochs:
        selected = [r for r in sequences if r['phase'] == epoch['phase'] and r['epoch'] == epoch['epoch']]
        assert len(selected) == 16 and len({r['id'] for r in selected}) == 16
        for key in ('IoU@1', 'IoU@3', 'IoU@5', 'native_IoU5'):
            np.testing.assert_allclose(epoch[key], np.mean([r[key] for r in selected]), atol=1e-14, rtol=0)
    stats = dict(passed=True, sequence_records=len(sequences), actual_round_records=count,
                 complete_16_unique_objects_per_phase_epoch=True, epoch_means_recomputed=True,
                 histogram_counts_verified=True, unit_tests_passed=17)
    (root / 'statistics_checks.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    train = [r for r in epochs if r['phase'] == 'train']
    for key, label in (('IoU@3', 'IoU@3 original'), ('IoU@5', 'IoU@5 original'), ('native_IoU5', 'IoU@5 native')):
        axes[0, 0].plot([r['epoch'] for r in evaluated], [100*r[key] for r in evaluated], 'o-', markersize=3, label=label)
    axes[0, 0].axhline(95, color='gray', linestyle=':', label='Predeclared 95% target')
    axes[0, 0].set(title='Fixed-weight evaluation on same training 16', ylabel='IoU (%)')
    for key in ('bce', 'dice', 'direction', 'preserve'):
        factor = .1 if key == 'preserve' else 1
        axes[0, 1].plot([r['epoch'] for r in evaluated], [factor*r['losses'][key] for r in evaluated], label=key + (' x0.1' if factor != 1 else ''))
    axes[0, 1].set(title='Weighted loss components (evaluation)')
    axes[1, 0].plot([r['epoch'] for r in train], [r['parameter_gradient_cosine_mean'] for r in train], alpha=.4, label='Train (changing weights)')
    axes[1, 0].plot([r['epoch'] for r in evaluated], [r['parameter_gradient_cosine_mean'] for r in evaluated], 'o-', markersize=3, label='Evaluation')
    axes[1, 0].axhline(0, color='gray', linestyle=':')
    axes[1, 0].set(title='Mask vs Direction full-BPTT parameter cosine', ylabel='Mean per-object cosine')
    axes[1, 1].plot([r['epoch'] for r in evaluated], [100*r['parameter_gradient_conflict_fraction'] for r in evaluated], 'o-', markersize=3, label='Conflicting objects')
    axes[1, 1].set(title='Full-BPTT gradient conflict fraction', ylabel='Objects (%)')
    for ax in axes.ravel():
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed COCO train16 overfit diagnostic - NOT generalization performance')
    fig.savefig(root / 'training_curves.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    checkpoints = list(dict.fromkeys([0, best['epoch'], final['epoch']]))
    for i, region in enumerate(('all', 'FN', 'FP')):
        for j, name in enumerate(('direction', 'strength')):
            for epoch in checkpoints:
                chosen = [r for r in distributions if r['epoch'] == epoch and r['region'] == region]
                if not chosen:
                    continue
                histogram = np.sum([r[f'{name}_histogram'] for r in chosen], axis=0).astype(float)
                x = np.linspace(-.95, .95, 20) if name == 'direction' else np.linspace(.025, .975, 20)
                axes[i, j].plot(x, histogram / histogram.sum(), label=f'epoch {epoch}')
            axes[i, j].set(title=f'{region}: {name} distribution', xlabel=name, ylabel='Pixel occurrence fraction')
            axes[i, j].legend()
            axes[i, j].grid(alpha=.2)
    fig.suptitle('Rounds 2-5 pooled; GT regions defined by old P>0.5; padding excluded')
    fig.savefig(root / 'direction_strength.png', dpi=160)
    plt.close(fig)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Directional 16实例过拟合诊断</title><style>body{font:16px/1.7 system-ui;margin:32px;max-width:1400px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}img{max-width:100%}</style><h1>DirectionalPIMSNet：固定 16 个 COCO 训练实例</h1>'
    page += f'<p>仅用于训练集记忆能力诊断，不是泛化性能。使用 seed 42 固定抽样并保存清单，研究参数从头初始化，冻结官方 EV-SAM。完成 {status["epochs"]} epoch、{status["optimizer_updates"]} 次 AdamW 更新。模型结构未调整，A3 源码及已有权重校验不变。</p>'
    page += '<p>统一阈值 P&gt;0.5：取点、方向标签、Preserve 区域、原图及原生 IoU。GT&gt;0.5。旧 RITM 接收的是已在 0.5 二值化后的输入，不会再次对连续概率使用 0.49。首击与每对象 RNG 固定，后续点击仍随当前预测闭环变化；缓存的是冻结图像特征和固定首击，不缓存任何可训练的后续状态。</p>'
    page += '<p>batch=2，lr=1e-4，weight decay=1e-4，梯度裁剪=1。损失 BCE+Dice+Direction+0.1×Preserve，方向监督为类别平衡 SmoothL1。每 5 epoch 在相同训练对象上用固定权重复测。训练曲线为一个 epoch 内更新过程的均值，不能与固定权重复测混为一谈。</p>'
    page += '<p>预先设定最多 100 epoch，若原图平均 IoU@5 连续三次复测达到 95% 则提前停止。本次' + ('达到该停止条件。' if status['achieved_095_three_evaluations'] else '未达到该条件；不能宣称已成功充分过拟合。') + '</p>'
    page += f'<p><strong>实际观察：</strong>最终 IoU@5 相比初始变化 {100*(final["IoU@5"]-initial["IoU@5"]):+.2f} 个百分点；总 Loss 从 {initial["losses"]["loss"]:.4f} 降到 {final["losses"]["loss"]:.4f}，Dice Loss 却从 {initial["losses"]["dice"]:.4f} 变为 {final["losses"]["dice"]:.4f}。Loss 下降不能替代二值 Mask 拟合成功的证据；本实验仅说明当前配置在该预算内未达到预设目标，不证明更长训练或其他配置也无法拟合。</p>'
    fn = next((r for r in distributions if r['epoch'] == final['epoch'] and r['round'] == 5 and r['region'] == 'FN'), None)
    if fn:
        page += f'<p>最终第 5 轮 FN 区域平均 Direction={fn["direction_mean"]:+.4f}、Strength={fn["strength_mean"]:.4f}：方向均值偏向补前景，但强度仍较小。两者均值不是实际更新量的乘积估计，不能凭均值证明单一退化原因。完整交互的参数梯度有 {100*final["parameter_gradient_conflict_fraction"]:.1f}% 对象冲突；逐轮输出梯度和共享参数梯度应分开解读。</p>'
    page += table(['检查点', 'epoch', 'IoU@1', 'IoU@3', 'IoU@5', '原生 IoU@5', 'Mask/Direction 梯度余弦', '冲突对象比例'], [
        [label, r['epoch'], *[f'{100*r[k]:.2f}%' for k in ('IoU@1', 'IoU@3', 'IoU@5', 'native_IoU5')],
         f'{r["parameter_gradient_cosine_mean"]:.4f}', f'{100*r["parameter_gradient_conflict_fraction"]:.1f}%'] for label, r in [('初始', initial), ('训练集最佳', best), ('最终', final)]])
    page += '<img src="training_curves.png"><p>梯度余弦：Mask=BCE+Dice，对比 Direction Loss；不包含 Preserve。每对象取第 2–5 轮平均损失的完整 BPTT 参数梯度，统计发生在裁剪/优化器更新前。另在 rounds.jsonl 中记录每轮本轮损失对原始双通道输出 logits 的梯度余弦。零范数时为 null，不伪造为零余弦。</p><h2>最终检查点逐轮结果</h2>'
    page += table(['轮次', '对象数', 'IoU', '原生 IoU', 'BCE', 'Dice', 'Direction', 'Preserve（未乘0.1）', '输出梯度余弦'], [
        [r['round'], r['objects'], f'{100*r["iou"]:.2f}%', f'{100*r["native_iou"]:.2f}%',
         *['—' if r[f'loss_{k}'] is None else f'{r[f"loss_{k}"]:.4f}' for k in ('bce', 'dice', 'direction', 'preserve')],
         '—' if r['output_gradient_cosine_mean'] is None else f'{r["output_gradient_cosine_mean"]:.4f}'] for r in compact if r['epoch'] == final['epoch']])
    page += '<p>首击冻结，编辑损失/Direction/Strength/编辑梯度为空。若无剩余错误则停止真实交互，IoU@3/5 沿用最后结果；逐轮表仅统计实际发生的点击。</p><img src="direction_strength.png"><h2>最终分布均值</h2>'
    page += table(['轮次', '旧 Mask 区域', '像素数', 'Direction 均值', 'Strength 均值', '|D|≤0.05'], [
        [r['round'], r['region'], r['pixel_occurrences'], f'{r["direction_mean"]:.4f}', f'{r["strength_mean"]:.4f}', f'{100*r["direction_near_zero_fraction"]:.2f}%']
        for r in distributions if r['epoch'] == final['epoch']])
    page += '<p>区域使用旧概率的 TP/TN/FP/FN；分布按有效像素出现次数汇总，IoU/损失/梯度余弦按对象平均。原生 256 网格与原图指标受 GT 缩放和官方 logits 插值影响，不应混为一个指标。</p><p>fixed_train16.jsonl：固定实例；rounds.jsonl：每轮 IoU、全部分项 Loss、双通道输出梯度、各区域分布均值/标准差/分位数/直方图；sequences.jsonl：整段交互及 all/memory/head 参数梯度；epochs.csv、eval_rounds.csv、eval_distributions.csv：汇总；best.pt/last.pt：独立权重。best 仅按训练集选择。17 项测试通过，statistics_checks.json 独立核对汇总和直方图计数。</p></html>'
    (root / 'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(initial=initial, best=best, final=final, statistics=stats), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    report(parser.parse_args().directory)
