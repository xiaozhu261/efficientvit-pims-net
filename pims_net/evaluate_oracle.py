"""GT-assisted Oracle decomposition of existing A3, NEVER a model benchmark."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluate_eta import state_metrics, write_csv
from pims_net.evaluation.experiment import file_identity
from pims_net.models import PIMSNet
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.interactive_losses import prepare_target


RUN_TYPE = 'oracle_diagnostic_not_model_performance'
BRANCHES = ('b1', 'b2', 'b3')


def clicked_error_component(old, truth, valid, click, image_shape):
    """8-connected same-sign error at the latest click, on native 256 canvas.

    Map original xy through rounded SAM 1024 prompt geometry / 4, then nearest
    native pixel. If that pixel is no longer the requested error, return empty;
    never snap to another error or silently choose the largest component.
    """
    old, truth, valid = np.asarray(old), np.asarray(truth, bool), np.asarray(valid, bool)
    if old.shape != (256, 256) or truth.shape != old.shape or valid.shape != old.shape:
        raise ValueError('B2 requires aligned native 256x256 maps')
    h, w = image_shape
    rh, rw = int(h * 1024 / max(h, w) + .5), int(w * 1024 / max(h, w) + .5)
    x = min(255, int(click['x'] * rw / w / 4 + .5))
    y = min(255, int(click['y'] * rh / h / 4 + .5))
    error = (truth & (old <= .5)) if click['label'] else (~truth & (old > .5))
    error &= valid
    count, labels = cv2.connectedComponents(error.astype(np.uint8), connectivity=8)
    selected = labels[y, x]
    gate = (labels == selected) if selected else np.zeros_like(error)
    return gate, dict(error_components=count - 1, selected_component_pixels=int(gate.sum()),
                      click_native_x=x, click_native_y=y, selected_error=bool(selected),
                      same_sign_error_pixels=int(error.sum()))


def oracle_update(previous, proposal, branch, truth, valid, gate=None):
    """Apply intervention outside model code; retain proposal hidden/click history.

    B1: retain total edit strength m=A+R, replace d=A/m by GT in
        P_new=(1-m)*P_old+m*d. This preserves m, NOT net |delta P|.
    B2: retain original proposal exactly inside the supplied gate, preserve old
        P and raw logits exactly outside. The gate is selected before editing.
    B3: return the original proposal object, with no conversion or intervention.
    edit_probabilities remain the raw network prediction for all branches.
    """
    if branch == 'b3':
        return proposal
    if branch not in ('b1', 'b2'):
        raise ValueError('Unknown Oracle branch')
    valid = valid.bool()
    if branch == 'b1':
        strength = proposal.edit_probabilities[:, :2].sum(1, keepdim=True).clamp(0, 1)
        probability = previous.probabilities + strength * (truth - previous.probabilities)
        selected = valid
        # Preserve unchanged raw logits too: logit(sigmoid(x)) clips endpoints.
        changed = selected & (probability != previous.probabilities)
        probability = torch.where(selected, probability, previous.probabilities)
        logits = torch.where(changed, torch.logit(probability, eps=1e-6), previous.logits)
    else:
        if gate is None or gate.shape != valid.shape:
            raise ValueError('B2 needs a gate aligned with the native GT')
        selected = gate.bool() & valid
        probability = torch.where(selected, proposal.probabilities, previous.probabilities)
        logits = torch.where(selected, proposal.logits, previous.logits)
    return replace(proposal, probabilities=probability, logits=logits)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def run(args):
    reference, output = Path(args.reference).resolve(), Path(args.output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError('Output must be a new run directory under pims_net/outputs')
    if output.exists():
        raise FileExistsError(output)
    source = json.loads((reference / 'protocol.json').read_text(encoding='utf-8'))
    dataset = InteractiveDataset(source['manifest']['path'])
    checkpoint = source['checkpoints']['a3']['path']
    for identity in (source['manifest'], source['baseline'], source['checkpoints']['a3']):
        assert file_identity(identity['path']) == identity, 'Inputs changed since reference evaluation'
    records = [json.loads(line) for line in (reference / 'per_object.jsonl').read_text(encoding='utf-8').splitlines()]
    records = [row for row in records if row['method'] == 'a3']
    assert len(records) == len(dataset)
    watched = [Path(x['path']) for x in (source['baseline'], *source['checkpoints'].values())]
    watched += sorted((RESEARCH_ROOT / 'models').glob('*.py')) + sorted((RESEARCH_ROOT / 'training').glob('*.py'))
    before = {str(path): file_identity(path)['sha256'] for path in watched}
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    assert count > 0
    torch.set_num_threads(4)
    torch.manual_seed(source['seed'])
    model = PIMSNet(variant='a3').to(args.device).eval()
    info = load_training_checkpoint(checkpoint, model)
    del info
    model.requires_grad_(False)
    output.mkdir(parents=True)
    dump(output / 'protocol.json', dict(run_type=RUN_TYPE, formal_model_performance=False, gt_at_inference=['b1', 'b2'],
        objects=count, manifest=source['manifest'], checkpoint=source['checkpoints']['a3'], baseline=source['baseline'],
        click_source=file_identity(reference / 'per_object.jsonl'),
        clicks='Original A3 first five clicks, same coordinates/labels in all three branches; no adaptive reselection.',
        feedback='Each branch has its own probabilities/logits/ConvGRU history; modified state feeds subsequent rounds.',
        b1='m=q_ADD+q_REMOVE; P_new=P_old+m*(GT-P_old) on valid pixels. Retains edit strength m, not net update magnitude.',
        b2='Preserve original predicted update only in current-click same-sign native GT error 8-connected component; no update elsewhere.',
        b2_geometry=clicked_error_component.__doc__, b2_threshold=.5,
        b2_empty='If latest fixed click is no longer a same-sign native error, skip mask update; hidden/clicks still advance.',
        b3='Unmodified model.refine; bitwise replay checked against original A3.',
        mask_metrics='Original resolution official logit postprocessing; sigmoid > 0.5; macro mean across objects.',
        padding='Excluded from Oracle changes and native invariants. Original-size metrics can differ through interpolation.',
        caveat='GT-assisted fixed-click diagnostic only. No formal benchmark claims, and no additive causal attribution across branches.',
        trained=False, device=args.device, seed=source['seed']))
    rows, fixed_clicks = [], []
    verification = dict(b3_probability_max_error=0., b3_metric_max_error=0., b1_gt_distance_increase_max=0.,
                        b1_strength_equation_max_error=0., b2_changed_outside_gate=0, b2_inside_proposal_mismatch=0,
                        b2_native_correct_pixels_damaged=0, parameters_frozen=True, finite_states=True)
    with torch.no_grad(), (output / 'per_object.jsonl').open('w', encoding='utf-8', buffering=1) as stream:
        for index in range(count):
            sample, ref = dataset[index], records[index]
            assert sample['id'] == ref['id'] and ref['index'] == index
            clicks = [{k: row[k] for k in ('x', 'y', 'label')} for row in ref['trace'][:5]]
            assert len(clicks) == 5
            fixed_clicks.append(dict(index=index, id=sample['id'], clicks=clicks))
            first = clicks[0]
            initial = model.initialize(sample['image'], [[[first['x'], first['y']]]], [[first['label']]])
            target, valid_tensor = prepare_target(sample['mask'], args.device)
            truth, valid = target[0, 0].cpu().numpy().astype(bool), valid_tensor[0, 0].cpu().numpy().astype(bool)
            for branch in BRANCHES:
                state = initial
                old_original_prediction = None
                for step, click in enumerate(clicks, 1):
                    details = dict(selected_error=None, selected_component_pixels=None, error_components=None,
                                   same_sign_error_pixels=None, click_native_x=None, click_native_y=None,
                                   mean_predicted_strength=None, mean_abs_applied_update=0.)
                    if step > 1:
                        previous = state
                        old = previous.probabilities[0, 0].cpu().numpy()
                        gate_tensor = None
                        if branch == 'b2':
                            gate, cc = clicked_error_component(old, truth, valid, click, sample['mask'].shape)
                            gate_tensor = torch.as_tensor(gate, device=args.device)[None, None]
                            details.update(cc)
                        proposal = model.refine(previous, [[[click['x'], click['y']]]], [[click['label']]])
                        state = oracle_update(previous, proposal, branch, target, valid_tensor, gate_tensor)
                        strength = proposal.edit_probabilities[:, :2].sum(1, keepdim=True)
                        details['mean_predicted_strength'] = float(strength[valid_tensor.bool()].mean())
                        details['mean_abs_applied_update'] = float((state.probabilities - previous.probabilities).abs()[valid_tensor.bool()].mean())
                        if branch == 'b1':
                            change = (state.probabilities - target).abs() - (previous.probabilities - target).abs()
                            verification['b1_gt_distance_increase_max'] = max(verification['b1_gt_distance_increase_max'], float(change[valid_tensor.bool()].max()))
                            expected_delta = strength * (target - previous.probabilities)
                            error = float(((state.probabilities - previous.probabilities) - expected_delta)[valid_tensor.bool()].abs().max())
                            verification['b1_strength_equation_max_error'] = max(verification['b1_strength_equation_max_error'], error)
                            assert change[valid_tensor.bool()].max() <= 1e-7 and error < 2e-7
                        elif branch == 'b2':
                            outside = ~gate_tensor
                            verification['b2_changed_outside_gate'] += int((state.probabilities[outside] != previous.probabilities[outside]).sum())
                            verification['b2_changed_outside_gate'] += int((state.logits[outside] != previous.logits[outside]).sum())
                            verification['b2_inside_proposal_mismatch'] += int((state.probabilities[gate_tensor] != proposal.probabilities[gate_tensor]).sum())
                            correct = ((old > .5) == truth) & valid
                            new_prediction = state.probabilities[0, 0].cpu().numpy() > .5
                            verification['b2_native_correct_pixels_damaged'] += int((correct & (new_prediction != truth)).sum())
                    metrics, prediction, native = state_metrics(model, state, sample['mask'], truth, valid)
                    assert torch.isfinite(state.hidden).all() and torch.isfinite(state.logits).all()
                    assert np.isfinite(native).all() and native.min() >= 0 and native.max() <= 1
                    assert state.points.shape[1] == step
                    if branch == 'b3':
                        with np.load(reference / 'diagnostics' / f'object_{index:04d}' / 'a3' / f'round_{step:02d}' / 'native.npz') as saved:
                            error = float(np.abs(native - saved['probabilities'][0]).max())
                            verification['b3_probability_max_error'] = max(verification['b3_probability_max_error'], error)
                            assert error == 0
                        for key in ('iou', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn'):
                            error = abs(metrics[key] - ref['trace'][step - 1][key])
                            verification['b3_metric_max_error'] = max(verification['b3_metric_max_error'], error)
                            assert error == 0
                    correct = (old_original_prediction == sample['mask']) if old_original_prediction is not None else None
                    metrics['original_correct_pixels_damaged'] = int((correct & (prediction != sample['mask'])).sum()) if correct is not None else 0
                    metrics['original_previously_correct_pixels'] = int(correct.sum()) if correct is not None else 0
                    row = dict(run_type=RUN_TYPE, index=index, id=sample['id'], branch=branch, round=step,
                               **click, **metrics, **details)
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    old_original_prediction = prediction
            if (index + 1) % 16 == 0 or index + 1 == count:
                print(f'Oracle diagnostic {index + 1}/{count} objects (B1/B2/B3)', flush=True)
    after = {str(path): file_identity(path)['sha256'] for path in watched}
    assert before == after
    assert verification['b2_changed_outside_gate'] == verification['b2_inside_proposal_mismatch'] == verification['b2_native_correct_pixels_damaged'] == 0
    summary = []
    for branch in BRANCHES:
        for step in range(1, 6):
            selected = [r for r in rows if r['branch'] == branch and r['round'] == step]
            means = {key: float(np.mean([r[key] for r in selected])) for key in
                     ('iou', 'precision', 'recall', 'native_band_fg', 'mean_abs_applied_update')}
            original_correct = sum(r['original_previously_correct_pixels'] for r in selected)
            summary.append(dict(run_type=RUN_TYPE, branch=branch, round=step, objects=len(selected), **means,
                b2_no_selected_error_rounds=sum(r['selected_error'] is False for r in selected),
                original_correct_damage_rate=sum(r['original_correct_pixels_damaged'] for r in selected) / original_correct if original_correct else 0.))
    assert len(rows) == count * 15
    comparison = []
    for branch in BRANCHES:
        result = dict(run_type=RUN_TYPE, branch=branch, gt_at_inference=branch != 'b3', formal_model_performance=False, objects=count)
        for step in (3, 5):
            selected = [r for r in rows if r['branch'] == branch and r['round'] == step]
            control = [r for r in rows if r['branch'] == 'b3' and r['round'] == step]
            assert [r['id'] for r in selected] == [r['id'] for r in control]
            deltas = np.array([r['iou'] - c['iou'] for r, c in zip(selected, control)])
            result.update({f'IoU@{step}': float(np.mean([r['iou'] for r in selected])),
                           f'delta_vs_B3_pp@{step}': float(deltas.mean() * 100),
                           f'better_than_B3@{step}': int((deltas > 0).sum()),
                           f'worse_than_B3@{step}': int((deltas < 0).sum())})
        comparison.append(result)
    verification.update(objects=count, recorded_rounds=len(rows), model_sources_and_weights_unchanged=True, hashes=before)
    for name, data in (('summary', summary), ('comparison', comparison), ('fixed_clicks', fixed_clicks), ('verification', verification)):
        dump(output / f'{name}.json', data)
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'comparison.csv', comparison)
    report(output, summary, comparison, rows)
    print(json.dumps(comparison, indent=2), flush=True)


def report(output, summary, comparison, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for branch, label in zip(BRANCHES, ('B1 GT direction (ORACLE)', 'B2 GT error component (ORACLE)', 'B3 original A3')):
        chosen = [r for r in summary if r['branch'] == branch]
        ax.plot([r['round'] for r in chosen], [r['iou'] * 100 for r in chosen], 'o-', label=label)
    ax.set(title='GT-assisted diagnostic only - NOT model performance', xlabel='Click (same original A3 sequence)', ylabel='IoU (%)', xticks=range(1, 6))
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(output / 'oracle_comparison.png', dpi=160)
    plt.close(fig)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Oracle 分解诊断：非正式模型性能</title><style>body{font:16px/1.7 system-ui;margin:32px;max-width:1200px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}img{max-width:100%}.notice{background:#fff0cf;padding:16px}</style><h1>Oracle 分解诊断</h1><p class="notice"><strong>B1/B2 在推理时使用 GT，仅供诊断。禁止作为正式模型性能、部署方法或公平 benchmark 结果。没有修改 A3 架构或权重，没有重新训练。</strong></p>'
    page += '<p>同一 seed 42 A3 checkpoint、同一 heldout 对象集，三组使用原 A3 的相同五击坐标和标签。第一击统一冻结 EV-SAM；随后各组维护自己的 Mask 和 ConvGRU 状态，并反馈修改结果。B3 逐轮与已有 A3 概率/指标精确核对。</p>'
    page += '<p><strong>B1：</strong>将原更新写为 P_new=(1−m)P_old+m·d，其中 m=q_ADD+q_REMOVE，d=q_ADD/m。保留预测编辑强度 m，用 GT 替换 d：P_new=P_old+m·(GT−P_old)。这里保留的是总编辑强度，不是最终净概率增量的绝对值；没有对 m 阈值化，极小但非零的 m 也会保留。Oracle 可以利用预测支撑范围内的 GT，因此不是纯粹的可部署方向纠错。</p>'
    page += '<p><strong>B2：</strong>在原生 256×256 网格，以旧 P&gt;0.5 与 GT 构造 FN/FP；正点击选择其所在 FN 八连通区域，负点击选择 FP 区域。原图坐标按 SAM 提示几何映射后取最近像素。只在选中区域接受原模型完整更新，其他位置保留旧概率和原始 logits。固定点击若已不在对应错误区域，该轮 Mask 不变，隐藏状态和点击历史仍推进。不扩张、不跳转其他连通区域、不改变预测方向。</p>'
    page += '<p>两种 Oracle 均排除 padding。IoU 使用官方 logits 后处理恢复原图，再 sigmoid&gt;0.5，按对象宏平均。原生网格的保护不保证原图插值后完全无损，逐轮结果另记录原图正确区域破坏率。</p><table><tr><th>分支</th><th>性质</th><th>IoU@3</th><th>IoU@5</th><th>相对 B3 @3</th><th>相对 B3 @5</th></tr>'
    for r in comparison:
        page += f'<tr><td>{r["branch"].upper()}</td><td>{"GT Oracle，非模型性能" if r["gt_at_inference"] else "原 A3 对照"}</td><td>{100*r["IoU@3"]:.2f}%</td><td>{100*r["IoU@5"]:.2f}%</td><td>{r["delta_vs_B3_pp@3"]:+.2f} pp</td><td>{r["delta_vs_B3_pp@5"]:+.2f} pp</td></tr>'
    b2 = [r for r in rows if r['branch'] == 'b2' and r['round'] > 1]
    empty = sum(r['selected_error'] is False for r in b2)
    page += f'</table><img src="oracle_comparison.png"><p>B2 共 {len(b2)} 个编辑轮次，其中 {empty} 轮（{100*empty/len(b2):.2f}%）当前固定点击未选中同符号错误连通域。这既可能是该组此前已修正对应位置，也可能与原生网格映射有关，不能将跳过的轮次当作模型主动学会停止。</p>'
    page += '<p><strong>解释范围：</strong>B1 衡量在保留总编辑强度的条件下，GT 方向替换能改善多少；B2 衡量 GT 错误区域限制能减少多少错误位置上的更新。两组改变后的 Mask 会影响后续预测，因此差值不是可相加的纯方向/位置贡献，也不能据此认定某一个因素是训练失败的唯一原因。固定原 A3 点击会在 Oracle 分支变成陈旧点击，这不是各分支独立自适应点击的性能评估。</p><p>comparison.csv 为 IoU@3/5 对比；summary.csv 为完整逐轮统计；per_object.jsonl 为逐对象结果；fixed_clicks.json、protocol.json、verification.json 保存点击、定义与验证记录。正式评估目录保持不变。</p></html>'
    (output / 'report.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    run(parser.parse_args())
