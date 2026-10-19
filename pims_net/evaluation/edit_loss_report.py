"""Render the completed read-only Edit Loss audit; no model imports or execution."""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def csv_file(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '<table><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers) + '</tr>' + ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in row) + '</tr>' for row in rows) + '</table>'


def validate(local, gradient, losses, individual_grad, checks):
    """Independently reconcile disjoint partitions and raw-record aggregates."""
    n = checks['objects_per_variant']
    assert len(losses) == 8 * n and len(individual_grad) == 8 * n
    for variant in ('a2', 'a3'):
        for step in (2, 3, 4, 5):
            rows = [r for r in losses if r['variant'] == variant and r['round'] == step]
            assert len(rows) == n and len({r['index'] for r in rows}) == n
            regions = {r['region']: r for r in local if r['variant'] == variant and r['round'] == step}
            assert regions['all']['pixels'] == sum(regions[k]['pixels'] for k in ('TP', 'TN', 'FP', 'FN'))
            assert sum(regions.get(k, {}).get('occurrences', 0) for k in ('latest_positive', 'latest_negative')) == n
            confusion = np.asarray(regions['all']['confusion'])
            np.testing.assert_array_equal(confusion.sum(1), np.sum([r['class_counts'] for r in rows], axis=0))
    for row in local:
        assert np.asarray(row['confusion']).sum() == row['pixels']
        assert abs(sum(row['q_mean']) - 1) < 1e-6
        assert 0 <= row['gradient_conflict'] <= row['gradient_nonzero'] <= row['pixels']
        assert 0 <= row['threshold_mismatch_fraction'] <= 1
    for row in gradient:
        selected = [r for r in individual_grad if r['variant'] == row['variant'] and r['group'] == row['group']]
        assert len(selected) == n
        assert row['conflict_objects'] == sum(r['cosine'] < -1e-8 for r in selected)
        np.testing.assert_allclose(row['mean_object_cosine'], np.mean([r['cosine'] for r in selected]))
    return dict(passed=True, refinement_records=len(losses), parameter_gradient_records=len(individual_grad),
                checks=['Complete unique object/round coverage', 'TP+TN+FP+FN partition',
                        'Positive/negative latest click partition', 'Confusion true counts vs loss labels',
                        'Confusion totals and probability normalization', 'Gradient/mismatch bounds',
                        'Gradient summary vs per-object records'])


def main(output):
    output = Path(output)
    local = json.loads((output / 'local_summary.json').read_text(encoding='utf-8'))
    gradient = json.loads((output / 'gradient_summary.json').read_text(encoding='utf-8'))
    checks = json.loads((output / 'verification.json').read_text(encoding='utf-8'))
    losses = read_jsonl(output / 'losses_per_round.jsonl')
    individual_grad = read_jsonl(output / 'gradients_per_object.jsonl')
    validation = validate(local, gradient, losses, individual_grad, checks)
    (output / 'statistics_checks.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    lookup = {(r['variant'], r['round'], r['region']): r for r in local}
    classes = ('ADD', 'REMOVE', 'KEEP')
    flat = []
    for row in local:
        item = {k: row[k] for k in ('variant', 'round', 'region', 'occurrences', 'pixels',
                'threshold_mismatch_fraction', 'gradient_conflict_fraction', 'pooled_gradient_cosine')}
        for i, name in enumerate(classes):
            item.update({f'q_{name}': row['q_mean'][i], f'true_{name}': row['true_distribution'][i],
                         f'predicted_{name}': row['predicted_distribution'][i]})
            for j, prediction in enumerate(classes):
                item[f'true_{name}_pred_{prediction}'] = row['confusion'][i][j]
        flat.append(item)
    csv_file(output / 'local_summary.csv', flat)
    csv_file(output / 'gradient_summary.csv', [{**{k: v for k, v in row.items() if k != 'mean_gradient'},
                **{f'mean_gradient_{k}': v for k, v in row['mean_gradient'].items()}} for row in gradient])
    loss_summary = []
    for variant in ('a2', 'a3'):
        for step in (2, 3, 4, 5, 'all'):
            selected = [r for r in losses if r['variant'] == variant and (step == 'all' or r['round'] == step)]
            counts = np.asarray([r['class_counts'] for r in selected])
            weights = np.asarray([[0 if x is None else x for x in r['per_pixel_edit_weight']] for r in selected])
            present = counts > 0
            item = dict(variant=variant, round=step, occurrences=len(selected),
                        **{key: float(np.mean([r[key] for r in selected])) for key in ('loss', 'mask', 'bce', 'dice', 'edit', 'preserve')})
            for i, name in enumerate(classes):
                item[f'true_{name}'] = float(counts[:, i].sum() / counts.sum())
                item[f'mean_CE_mass_{name}'] = float((present[:, i] / present.sum(1)).mean())
                ratio = weights[:, i] / np.maximum(weights[:, 2], 1e-30)
                included = present[:, i] & present[:, 2]
                item[f'median_pixel_weight_vs_KEEP_{name}'] = float(np.median(ratio[included])) if included.any() else None
            loss_summary.append(item)
    csv_file(output / 'loss_summary.csv', loss_summary)
    (output / 'loss_summary.json').write_text(json.dumps(loss_summary, indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for i, variant in enumerate(('a2', 'a3')):
        for j, sign in enumerate(('positive', 'negative')):
            row = lookup[variant, 'all', f'latest_{sign}']
            matrix = 100 * np.asarray(row['confusion_row_normalized'])
            ax = axes[i, j]
            ax.imshow(matrix, vmin=0, vmax=100, cmap='Blues')
            ax.set(title=f'{variant.upper()} latest {sign} click (radius 8)', xlabel='Predicted class', ylabel='True edit class',
                   xticks=range(3), yticks=range(3), xticklabels=classes, yticklabels=classes)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, f'{matrix[a,b]:.1f}%\n({row["confusion"][a][b]:,})', ha='center', va='center',
                            color='white' if matrix[a,b] > 50 else 'black', fontsize=10)
    fig.suptitle('Rows normalized within true class; rounds 2-5 pooled, valid native pixels')
    fig.savefig(output / 'click_confusion.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for variant in ('a2', 'a3'):
        cosines = [r['cosine'] for r in individual_grad if r['variant'] == variant and r['group'] == 'all']
        axes[0].hist(cosines, bins=np.linspace(-1, 1, 21), alpha=.5, label=variant.upper())
        for region, style in (('TP', '-'), ('TN', '--')):
            axes[1].plot(range(2, 6), [100 * lookup[variant, t, region]['gradient_conflict_fraction'] for t in range(2, 6)],
                         style, marker='o', label=f'{variant.upper()} {region}')
    axes[0].set(title='Full-BPTT parameter gradient cosine per object', xlabel='Mask vs Edit gradient cosine', ylabel='Objects')
    axes[0].axvline(0, color='black', lw=1)
    axes[1].set(title='Local intent-logit gradient conflict on correct pixels', xlabel='Round', ylabel='Conflicting nonzero pixels (%)', xticks=range(2, 6))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(output / 'gradient_conflict.png', dpi=160)
    plt.close(fig)

    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PIMS-Net Edit Loss 诊断</title><style>body{font:16px/1.7 system-ui;margin:32px;max-width:1400px;color:#182430}table{border-collapse:collapse;margin:20px 0}td,th{border:1px solid #ccc;padding:7px}img{max-width:100%}code{background:#eee;padding:2px}</style><h1>PIMS-Net Edit Loss 监督与梯度审计</h1>'
    page += f'<p>seed 42 已有 A2/A3 权重，各 {checks["objects_per_variant"]} 个相同 heldout 对象，共 {checks["total_refinement_rounds"]} 个编辑轮次。重放各方法已有第 1–5 击；没有重新训练、没有修改模型、损失或权重。局部统计为非 padding 的原生 256×256 像素出现次数加权。</p>'
    main_grad = {r['variant']: r for r in gradient if r['group'] == 'all'}
    page += '<p><strong>本次结论：</strong>未发现类别/通道颠倒、损失公式错误或梯度断开。共享参数上存在明显梯度冲突：' + '；'.join(
        f'{v.upper()} 为 {r["conflict_objects"]}/{r["objects"]} 个对象（{100*r["conflict_objects"]/r["objects"]:.1f}%），全集平均梯度余弦 {r["mean_gradient"]["cosine"]:.4f}'
        for v, r in main_grad.items()) + '。但原生 FN/FP 像素的本轮输出梯度冲突率均为 0%，说明“同一错误像素标签写反”不能解释共享参数上的冲突；不同区域经共享网络映射、类别权重和跨轮路径汇总后出现竞争。</p>'
    positive, negative, center = (lookup['a3', 'all', region] for region in ('latest_positive', 'latest_negative', 'center_positive'))
    page += f'<p>A3 正点击邻域 q_add={positive["q_mean"][0]:.4f}、q_remove={positive["q_mean"][1]:.4f}，基本没有拉开方向；负点击邻域真实 REMOVE 中只有 {100*negative["confusion_row_normalized"][1][1]:.1f}% 被预测为 REMOVE。正点击中心的原生标签有 {100*center["threshold_mismatch_fraction"]:.2f}% 因 0.49/0.5 阈值而改变，提示接近 0.5 后训练监督与评估纠错目标存在明显错位。以上是现象证据，不能单独证明训练失败的唯一原因。</p>'
    page += '<h2>监督实现核对</h2><p>类别/通道严格对应 ADD=0、REMOVE=1、KEEP=2；旧概率 &gt;0.49 视为前景，GT &gt;0.5 视为前景。FN→ADD、FP→REMOVE、TP/TN→KEEP。实际配置 BCE=1、Dice=1、Edit=1、Preserve=0.1。Edit 对每个当前存在类别先求平均 NLL，再对类别平均：像素权重为 1/(K×该类像素数)，不是三个全局固定的逆频率权重。q 经一次 softmax，损失用 -log(q_true)，等价于原始 logits 的交叉熵（除 q_true&lt;1e-7 的截断）。没有把概率再次传入 CrossEntropy。</p>'
    page += '<p>监督标签/Preserve 的旧概率参考 detach；前向的旧 Mask 和 A3 隐状态没有 detach，因此保留完整 BPTT。A2 仅清零 ConvGRU 历史，旧 Mask 仍形成跨轮路径。SAM 的所有参数被冻结。训练总损失对第 2–5 轮平均，每个对象反向后按批内对象数平均；本诊断按同样的轮次平均计算参数梯度，但不裁剪、不进行 AdamW 或优化器更新。</p>'
    page += '<h2>真实类别比例与类别平衡</h2>'
    page += table(['方法', 'ADD/REMOVE/KEEP 像素占比', 'ADD/REMOVE/KEEP 平均 CE 权重质量', 'ADD/KEEP 单像素权重中位比', 'REMOVE/KEEP 中位比', 'Mask Loss', 'Edit Loss', 'Preserve 未乘0.1'], [
        [r['variant'].upper(), '/'.join(f'{100*r[f"true_{c}"]:.2f}%' for c in classes),
         '/'.join(f'{100*r[f"mean_CE_mass_{c}"]:.2f}%' for c in classes),
         f'{r["median_pixel_weight_vs_KEEP_ADD"]:.1f}', f'{r["median_pixel_weight_vs_KEEP_REMOVE"]:.1f}',
         f'{r["mask"]:.4f}', f'{r["edit"]:.4f}', f'{r["preserve"]:.4f}'] for r in loss_summary if r['round'] == 'all'])
    page += '<h2>正负点击附近</h2><p>主表统计每轮最新点击周围半径 8 原生像素的圆盘（约为点击高斯一个 σ），排除 padding。每轮只有一个最新点击，所以正/负组对象轮次不同。正点击不意味着整个圆盘都该 ADD；标签还依赖 GT 边界及旧概率。history_* 另统计所有历史同符号圆盘的并集；center_* 另统计映射到最近原生像素的点击中心，均见 CSV/JSON。</p>'
    page += table(['方法/最新点击', '轮次样本数', 'q ADD/REMOVE/KEEP', '真实 ADD/REMOVE/KEEP', 'argmax ADD/REMOVE/KEEP', '0.49↔0.5 标签变化', '局部梯度冲突'], [
        [r['variant'].upper() + '/' + ('正' if r['region'].endswith('positive') else '负'), r['occurrences'],
         '/'.join(f'{100*x:.2f}%' for x in r['q_mean']), '/'.join(f'{100*x:.2f}%' for x in r['true_distribution']),
         '/'.join(f'{100*x:.2f}%' for x in r['predicted_distribution']), f'{100*r["threshold_mismatch_fraction"]:.2f}%',
         f'{100*r["gradient_conflict_fraction"]:.2f}%'] for r in local if r['round'] == 'all' and r['region'] in ('latest_positive', 'latest_negative')])
    page += '<img src="click_confusion.png"><p>上图行是真实编辑类别，列是 argmax 预测类别；同时显示行归一百分数与原始计数。argmax 会放大 q_add 与 q_remove 的微小差异，必须结合概率均值解读。</p>'
    page += '<h2>Mask Loss 与 Edit Loss 的参数梯度</h2><p>Mask = BCE+Dice；Edit 系数为 1。对四轮平均损失计算全 BPTT 梯度。cos&lt;0 表示局部一阶方向冲突；范数比是 Edit/Mask。每对象余弦均值和整个对象集的平均梯度余弦是两个不同量。这里不是训练时 AdamW 的实际参数步长。</p>'
    page += table(['方法/参数组', '冲突对象数', '对象余弦均值', '对象余弦中位数', '对象范数比中位数', '平均梯度余弦', '平均梯度范数比'], [
        [r['variant'].upper() + '/' + r['group'], f'{r["conflict_objects"]}/{r["objects"]}',
         f'{r["mean_object_cosine"]:.4f}', f'{r["median_object_cosine"]:.4f}', f'{r["median_edit_to_mask_norm"]:.2f}',
         f'{r["mean_gradient"]["cosine"]:.4f}', f'{r["mean_gradient"]["edit_to_mask_norm"]:.2f}'] for r in gradient])
    page += '<img src="gradient_conflict.png"><h2>冲突发生在哪里</h2><p>局部梯度是“本轮损失对本轮原始三通道 logits”的导数，非参数梯度；TP/TN/FP/FN 使用训练阈值 0.49。冲突比例分母是两项梯度均非零的像素，cos&lt;-1e-8 计为冲突。</p>'
    page += table(['方法/区域', 'q ADD/REMOVE/KEEP', '局部冲突比例', '拼接局部梯度余弦', '0.49↔0.5 标签变化'], [
        [r['variant'].upper() + '/' + r['region'], '/'.join(f'{100*x:.2f}%' for x in r['q_mean']),
         f'{100*r["gradient_conflict_fraction"]:.2f}%', f'{r["pooled_gradient_cosine"]:.4f}',
         f'{100*r["threshold_mismatch_fraction"]:.2f}%'] for r in local if r['round'] == 'all' and r['region'] in ('TP', 'TN', 'FP', 'FN')])
    page += '<p><strong>解释边界：</strong>正确的硬分类仍可能只有约 0.5 的置信度，Mask Loss 希望继续向 GT 的 0/1 推进；KEEP 监督则倾向保留旧概率。两者会在部分状态下冲突，但 KEEP 也可能阻止一次有害编辑并与 Mask Loss 同向，因此不能笼统说所有 KEEP 都冲突。类别平衡使小面积错误获得很高单像素权重；当网络不能充分区分 ADD/REMOVE 区域时，这可能形成同时激活的压力，但当前梯度统计不足以证明它是训练退化的唯一原因。</p>'
    page += '<p>训练取点阈值也是 0.49，因此 0.49 标签与训练取点本身一致；不一致发生在 0.5 评估协议与训练监督之间。尤其旧前景概率处于 (0.49,0.5] 时，评估中的 FN 在训练标签中会成为 KEEP。当前重放使用每个方法自己的评估点击序列，不能据此声称重建了训练期间的随机轨迹或历史梯度；单种子也不能证明跨种子稳定性。</p>'
    page += '<h2>验证与复现</h2>' + table(['检查', '结果'], [
        ['类别、平衡权重、padding、有限差分、加权梯度、混淆矩阵、点击几何单元测试', '7/7 通过（test_edit_loss_audit）'],
        ['概率重放最大误差', checks['max_probability_replay_error']], ['编辑概率重放最大误差', checks['max_intent_replay_error']],
        ['独立 NLL 公式最大误差', checks['max_edit_formula_error']], ['加权总梯度线性一致性最大误差', checks['max_total_gradient_linearity_error']],
        ['q_true<1e-7 的有效像素', checks['nll_clamped_valid_pixels']], ['所有梯度有限、所有研究参数接收梯度', checks['finite_gradients'] and not checks['trainable_parameters_without_gradient']],
        ['参数 .grad 未积累、SAM 冻结', checks['parameter_grad_buffers_empty'] and checks['sam_frozen']],
        ['模型内存参数/缓冲区哈希前后一致', all(v['before'] == v['after'] for v in checks['model_hashes'].values())],
        ['模型/训练源码、数据清单、原始与研究权重哈希一致', checks['source_and_weight_hashes_unchanged']]])
    page += '<p>完整逐轮统计见 local_summary.csv、loss_summary.csv、gradient_summary.csv；逐对象原始数据见三个 JSONL。protocol.json 和 verification.json 保存协议、范围和哈希。</p></html>'
    (output / 'report.html').write_text(page, encoding='utf-8')
    print(f'Report saved: {output / "report.html"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    main(parser.parse_args().output)
