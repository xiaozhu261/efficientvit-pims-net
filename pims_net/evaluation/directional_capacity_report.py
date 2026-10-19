"""Verify and compare the original and wider decoder train16 experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluation.directional_overfit_report import table, write_csv
from pims_net.evaluation.experiment import file_identity
from pims_net.evaluation.metrics import mask_iou
from pims_net.models.directional_pims_net import DirectionalEditDecoder
from pims_net.models.directional_wide import WideDirectionalPIMSNet
from pims_net.overfit_directional import dump
from pims_net.training.checkpoint import load_training_checkpoint


def read_lines(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


@torch.no_grad()
def replay(root, device):
    model = WideDirectionalPIMSNet().to(device).eval().requires_grad_(False)
    dataset = InteractiveDataset(root / 'fixed_train16.jsonl')
    records = [r for r in read_lines(root / 'rounds.jsonl') if r['phase'] == 'eval']
    outcomes = []
    for filename in ('last.pt', 'best.pt'):
        info = load_training_checkpoint(root / filename, model)
        selected = [r for r in records if r['epoch'] == info['epoch']]
        errors = []
        for index in range(16):
            sample = dataset[index]
            steps = sorted([r for r in selected if r['index'] == index], key=lambda r: r['round'])
            for step, r in enumerate(steps):
                points, labels = [[[r['x'], r['y']]]], [[r['label']]]
                state = model.initialize(sample['image'], points, labels) if step == 0 else model.refine(state, points, labels)
                p = model.sam_initializer.postprocess(state.logits, state.image_sizes)[0].sigmoid()[0, 0].cpu().numpy()
                errors.append(abs(mask_iou(p > .5, sample['mask']) - r['iou']))
        assert max(errors) == 0
        outcomes.append(dict(checkpoint=filename, epoch=info['epoch'], optimizer_updates=info['global_step'],
                             rounds=len(errors), max_iou_error=max(errors)))
        del info
    dump(root / 'checkpoint_replay.json', outcomes)
    return outcomes


def report(reference, directory, device):
    original, wide = Path(reference).resolve(), Path(directory).resolve()
    roots = {'original': original, 'wide_residual': wide}
    configs = {name: json.loads((root / 'config.json').read_text()) for name, root in roots.items()}
    for key in ('seed', 'epochs', 'batch_size', 'max_clicks', 'lr', 'weight_decay', 'grad_clip', 'threshold', 'loss_weights', 'eval_every'):
        assert configs['original'][key] == configs['wide_residual'][key], key
    assert (original / 'fixed_train16.jsonl').read_bytes() == (wide / 'fixed_train16.jsonl').read_bytes()
    curves, all_rounds, sequence_records, comparison, round_summary, distribution_rows = {}, {}, {}, [], [], []
    for name, root in roots.items():
        status = json.loads((root / 'status.json').read_text())
        assert status['status'] == 'completed' and status['epochs'] == 100 and status['optimizer_updates'] == 800
        curves[name] = [r for r in read_lines(root / 'epochs.jsonl') if r['phase'] == 'eval']
        sequence_records[name] = [r for r in read_lines(root / 'sequences.jsonl') if r['phase'] == 'eval']
        all_rounds[name] = [r for r in read_lines(root / 'rounds.jsonl') if r['phase'] == 'eval']
        for epoch in curves[name]:
            sequences = [r for r in sequence_records[name] if r['epoch'] == epoch['epoch']]
            assert len(sequences) == 16 and len({r['index'] for r in sequences}) == 16
            for key in ('IoU@1', 'IoU@3', 'IoU@5', 'native_IoU5'):
                assert abs(epoch[key] - np.mean([r[key] for r in sequences])) < 1e-14
        best = max(curves[name], key=lambda r: r['IoU@5'])
        for label, row in (('initial', curves[name][0]), ('best', best), ('final', curves[name][-1])):
            comparison.append(dict(model=name, checkpoint=label, epoch=row['epoch'],
                **{k: row[k] for k in ('IoU@1', 'IoU@3', 'IoU@5', 'native_IoU5', 'parameter_gradient_cosine_mean', 'parameter_gradient_conflict_fraction')},
                **{'loss_' + k: v for k, v in row['losses'].items()}))
        for step in range(1, 6):
            rows = [r for r in all_rounds[name] if r['epoch'] == 100 and r['round'] == step]
            round_summary.append(dict(model=name, round=step, objects=len(rows),
                                      iou=float(np.mean([r['iou'] for r in rows])),
                                      native_iou=float(np.mean([r['native_iou'] for r in rows]))))
            if step == 1:
                continue
            for region in ('all', 'TP', 'TN', 'FP', 'FN'):
                selected = [r['distributions'][region] for r in rows if r['distributions'][region]['direction'] is not None]
                count = sum(r['direction']['count'] for r in selected)
                if count:
                    item = dict(model=name, round=step, region=region, pixels=count)
                    for field in ('direction', 'strength'):
                        assert all(sum(r[field]['histogram']) == r[field]['count'] for r in selected)
                        item[field + '_mean'] = sum(r[field]['mean'] * r[field]['count'] for r in selected) / count
                    distribution_rows.append(item)
    firsts = {name: [(r['index'], r['x'], r['y'], r['label'], r['iou']) for r in rows if r['epoch'] == 100 and r['round'] == 1] for name, rows in all_rounds.items()}
    assert firsts['original'] == firsts['wide_residual']
    counts = dict(original_decoder=sum(p.numel() for p in DirectionalEditDecoder().parameters()),
                  wide_decoder=configs['wide_residual']['parameter_counts']['decoder'],
                  wide_trainable=configs['wide_residual']['parameter_counts']['trainable'])
    counts['original_trainable'] = counts['wide_trainable'] - counts['wide_decoder'] + counts['original_decoder']
    pairs = []
    for index in range(16):
        a, b = [next(r for r in sequence_records[name] if r['epoch'] == 100 and r['index'] == index) for name in roots]
        assert a['id'] == b['id']
        pairs.append(dict(index=index, id=a['id'], original_iou5=a['IoU@5'], wide_iou5=b['IoU@5'], delta=b['IoU@5'] - a['IoU@5']))
    for filename, rows in (('capacity_comparison', comparison), ('capacity_rounds', round_summary),
                           ('capacity_distributions', distribution_rows), ('capacity_pairs', pairs)):
        dump(wide / (filename + '.json'), rows)
        write_csv(wide / (filename + '.csv'), rows)
    replays = replay(wide, device)
    for path, sha in json.loads((wide / 'verification.json').read_text())['original_hashes'].items():
        assert file_identity(path)['sha256'] == sha
    checks = dict(passed=True, same_manifest_bytes=True, same_hyperparameters=True,
                  same_first_clicks_and_masks=True, updates_each=800, original_files_and_weights_unchanged=True,
                  means_recomputed=True, parameter_counts=counts, checkpoint_replays=replays,
                  final_better=sum(r['delta'] > 0 for r in pairs), final_worse=sum(r['delta'] < 0 for r in pairs),
                  final_equal=sum(r['delta'] == 0 for r in pairs))
    dump(wide / 'capacity_verification.json', checks)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, rows in curves.items():
        epochs = [r['epoch'] for r in rows]
        axes[0, 0].plot(epochs, [100*r['IoU@5'] for r in rows], 'o-', markersize=3, label=name)
        axes[0, 1].plot(epochs, [r['losses']['loss'] for r in rows], 'o-', markersize=3, label=name)
        axes[1, 0].plot(epochs, [r['losses']['dice'] for r in rows], 'o-', markersize=3, label=name)
        axes[1, 1].plot(epochs, [r['parameter_gradient_cosine_mean'] for r in rows], 'o-', markersize=3, label=name)
    for ax, title in zip(axes.flat, ('Train16 evaluation IoU@5 (%)', 'Total loss', 'Dice loss', 'Mask vs Direction parameter cosine')):
        ax.set(title=title, xlabel='Epoch')
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(wide / 'capacity_comparison.png', dpi=160)
    plt.close(fig)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Decoder容量过拟合对比</title><style>body{font:16px/1.7 system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}img{max-width:100%}</style><h1>DirectionalPIMSNet：Decoder容量消融</h1>'
    page += '<p>仅改变Decoder：隐藏通道64/32/16扩大到128/64/32，在64×64层添加两个64通道残差块，每块为x+Conv3×3(GELU(Conv3×3(x)))，不加归一化。ConvGRU、点击编码、冻结SAM首击、tanh方向、sigmoid强度及MaskEditor全部沿用原实现。</p>'
    page += '<p>同一16实例及顺序，seed42，研究参数从头训练，无权重热启动。相同ConvGRU初始化已单元测试核对。batch2，lr/weight decay均1e-4，clip1，完整100epoch/800更新，每5epoch复测。损失仍BCE+Dice+Direction+0.1×Preserve，统一P&gt;0.5。使用相同自适应RITM协议和对象RNG；后续点击随各自模型变化，不固定为同一坐标。本表使用固定权重eval结果，不混用训练中均值。</p>'
    page += f'<p>Decoder参数：{counts["original_decoder"]:,} → {counts["wide_decoder"]:,}；全部可训练参数：{counts["original_trainable"]:,} → {counts["wide_trainable"]:,}。训练预算按优化步数匹配，不是等计算量。</p>'
    page += table(['模型', '检查点', 'epoch', 'IoU@1', 'IoU@3', 'IoU@5', '总Loss', 'BCE', 'Dice', 'Direction', 'Preserve(未乘0.1)'], [
        [r['model'], r['checkpoint'], r['epoch'], *[f'{100*r[k]:.2f}%' for k in ('IoU@1','IoU@3','IoU@5')],
         *[f'{r["loss_"+k]:.4f}' for k in ('loss','bce','dice','direction','preserve')]] for r in comparison])
    finals = {r['model']:r for r in comparison if r['checkpoint'] == 'final'}
    delta = 100 * (finals['wide_residual']['IoU@5'] - finals['original']['IoU@5'])
    page += f'<p>同预算最终IoU@5变化 {delta:+.2f} 个百分点；16对象中改善 {checks["final_better"]}，退化 {checks["final_worse"]}，持平 {checks["final_equal"]}。该单种子训练集诊断不证明泛化提升或跨种子稳定性。最佳检查点单独展示，避免以不同epoch替代相同预算比较。</p>'
    page += '<img src="capacity_comparison.png"><h2>最终逐轮IoU</h2>'
    page += table(['模型','轮次','对象数','IoU','原生IoU'], [[r['model'],r['round'],r['objects'],f'{100*r["iou"]:.2f}%',f'{100*r["native_iou"]:.2f}%'] for r in round_summary])
    page += '<h2>完整交互梯度与方向强度</h2>'
    page += table(['模型','Mask/Direction梯度余弦','冲突对象比例'], [[name,f'{r["parameter_gradient_cosine_mean"]:.4f}',f'{100*r["parameter_gradient_conflict_fraction"]:.1f}%'] for name,r in finals.items()])
    page += table(['模型','第5轮区域','Direction均值','Strength均值'], [[r['model'],r['region'],f'{r["direction_mean"]:.4f}',f'{r["strength_mean"]:.4f}'] for r in distribution_rows if r['round']==5])
    page += '<p>完整逐轮IoU、Loss、梯度及分布在rounds.jsonl和sequences.jsonl；各epoch汇总在epochs.jsonl。新last.pt/best.pt独立保存，加载回放IoU逐轮精确一致；模型/损失源码、原Directional权重及A3权重哈希检查通过。容量差异同时含通道和深度，不能由本次实验单独区分两者贡献。</p></html>'
    (wide / 'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(comparison=comparison, verification=checks), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    report(args.reference, args.directory, args.device)
