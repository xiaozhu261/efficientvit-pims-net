"""Fixed-click C0-C3 inference constraints and local GT diagnostic, no training."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluate_eta import state_metrics, write_csv
from pims_net.evaluation.directional_overfit_report import table
from pims_net.evaluation.experiment import file_identity
from pims_net.models.directional_wide import WideDirectionalPIMSNet
from pims_net.overfit_directional import dump
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT, checkpoint_path
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.interactive_losses import prepare_target

BRANCHES = ('C0', 'C1', 'C2', 'C3', 'local_gt_oracle')


def transition_counts(old, new, truth):
    old, new, truth = [np.asarray(v, bool) for v in (old, new, truth)]
    if old.shape != new.shape or old.shape != truth.shape:
        raise ValueError('All masks must align')
    fn, fp, tp, tn = ~old & truth, old & ~truth, old & truth, ~old & ~truth
    return dict(fn_before=int(fn.sum()), fn_repaired=int((fn & new).sum()),
                fp_before=int(fp.sum()), fp_removed=int((fp & ~new).sum()),
                tp_before=int(tp.sum()), tp_damaged=int((tp & ~new).sum()),
                tn_before=int(tn.sum()), tn_damaged=int((tn & new).sum()))


def transition_rates(counts):
    return {name: counts[num] / counts[den] if counts[den] else None for name, num, den in (
        ('fn_repair_rate', 'fn_repaired', 'fn_before'), ('fp_removal_rate', 'fp_removed', 'fp_before'),
        ('tp_damage_rate', 'tp_damaged', 'tp_before'), ('tn_damage_rate', 'tn_damaged', 'tn_before'))}


def latest_gaussian(model, proposal, valid):
    """Exactly reuse the single latest click's 64-grid map, bilinear to 256."""
    maps = model.click_encoder(proposal.points[:, -1:], proposal.labels[:, -1:], proposal.image_sizes)
    return F.interpolate(maps.sum(1, keepdim=True), size=(256, 256), mode='bilinear', align_corners=False) * valid


def intervene(model, previous, proposal, branch, gaussian, truth=None):
    if branch == 'C0':
        return proposal
    if branch not in BRANCHES:
        raise ValueError('Unknown branch')
    if branch == 'local_gt_oracle':
        if truth is None:
            raise ValueError('Oracle requires GT')
        probability = previous.probabilities + gaussian * (truth - previous.probabilities)
        logits = torch.logit(probability, eps=1e-6)
    else:
        direction, strength = proposal.direction, proposal.strength
        if branch in ('C1', 'C3'):
            positive = proposal.labels[:, -1:].view(-1, 1, 1, 1) == 1
            direction = torch.where(positive, direction.clamp(min=0), direction.clamp(max=0))
        if branch in ('C2', 'C3'):
            strength = strength * gaussian
        probability, logits = model.directional_editor.update(previous.probabilities, previous.logits, direction, strength)
    # Exact KEEP even when tiny Gaussian updates round to zero. Otherwise a
    # logit(eps) conversion alone can clip SAM endpoints far from the click.
    logits = torch.where(probability == previous.probabilities, previous.logits, logits)
    return replace(proposal, probabilities=probability, logits=logits)


def run(args):
    reference, output = Path(args.reference).resolve(), Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError('Use a new run directory under pims_net/outputs')
    source_status = json.loads((reference / 'status.json').read_text())
    assert source_status['status'] == 'completed'
    records = []
    with (reference / 'rounds.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['phase'] == 'eval' and row['epoch'] == source_status['epochs']:
                records.append(row)
    manifest, checkpoint = reference / 'fixed_train16.jsonl', reference / 'last.pt'
    dataset = InteractiveDataset(manifest)
    assert len(dataset) == 16 and len(records) == 80
    watched = [checkpoint, reference / 'best.pt', manifest, reference / 'rounds.jsonl', checkpoint_path(),
               OUTPUT_ROOT / 'directional_overfit16_seed42_20261019/last.pt',
               OUTPUT_ROOT / 'directional_overfit16_seed42_20261019/best.pt']
    watched += list((RESEARCH_ROOT / 'models').glob('*.py')) + list((RESEARCH_ROOT / 'training').glob('*.py'))
    before = {str(p): file_identity(p)['sha256'] for p in watched}
    original = json.loads((OUTPUT_ROOT / 'directional_implementation_20261018/original_hashes.json').read_text())
    for path, sha in original.items():
        assert file_identity(path)['sha256'] == sha
    before.update(original)
    torch.set_num_threads(4)
    model = WideDirectionalPIMSNet().to(args.device).eval().requires_grad_(False)
    info = load_training_checkpoint(checkpoint, model)
    assert info['epoch'] == source_status['epochs'] and info['config']['threshold'] == .5
    output.mkdir(parents=True)
    dump(output / 'protocol.json', dict(checkpoint=file_identity(checkpoint), manifest=file_identity(manifest),
        clicks_source=file_identity(reference / 'rounds.jsonl'), epoch=info['epoch'], threshold=.5, branches=BRANCHES,
        C0='Unmodified wide model.refine, exact saved IoU replay.',
        C1='Latest positive click: D=max(D,0); latest negative click: D=min(D,0). Global sign constraint; S unchanged.',
        C2='D unchanged; S_effective=S*G_latest.', C3='Both C1 and C2.',
        gaussian='Original ClickEncoder sigma=2 on 64 grid, latest click only, bilinear 256 align_corners=False, no peak renormalization or hard truncation; zero padding.',
        oracle='GT-assisted diagnostic only: P_new=P_old+G_latest*(GT-P_old). Same initial mask, own feedback. Not a strict upper bound on original-resolution IoU.',
        keep='C1/C2/C3/Oracle preserve raw previous logits wherever P is numerically unchanged to avoid logit(eps)-only edits.',
        clicks='Same recorded 16x5 final-epoch wide baseline clicks for every branch. No re-selection.',
        feedback='Each branch has its own updated mask/logits/ConvGRU history. No changes to model code or weights.',
        metrics='Original-size official logits postprocessing, P>0.5. IoU/precision/recall: object macro. FN/FP/TP rates: pooled pixels.',
        references='Main comparison against first-click mask. Also record each transition against previous mask.',
        trained=False, formal_oracle_performance=False, device=args.device))
    del info
    rows, saved_clicks = [], []
    checks = dict(C0_iou_max_error=0., oracle_native_gt_distance_increase=0.,
                  oracle_native_correct_pixels_damaged=0, sign_violation_pixels=0, outside_gaussian_changes=0,
                  no_gradients=True, unchanged_probability_logit_changes=0)
    with torch.no_grad(), (output / 'per_object.jsonl').open('w', encoding='utf-8') as stream:
        for index in range(16):
            sample = dataset[index]
            saved = sorted([r for r in records if r['index'] == index], key=lambda r: r['round'])
            assert [r['round'] for r in saved] == list(range(1, 6)) and all(r['id'] == sample['id'] for r in saved)
            clicks = [{k:r[k] for k in ('x','y','label')} for r in saved]
            saved_clicks.append(dict(index=index, id=sample['id'], clicks=clicks))
            first = clicks[0]
            initial = model.initialize(sample['image'], [[[first['x'], first['y']]]], [[first['label']]])
            target, valid_t = prepare_target(sample['mask'], args.device)
            truth, valid = target[0,0].cpu().numpy().astype(bool), valid_t[0,0].cpu().numpy().astype(bool)
            _, first_mask, first_native = state_metrics(model, initial, sample['mask'], truth, valid)
            for branch in BRANCHES:
                state, previous_mask = initial, first_mask
                previous_native = first_native
                for step, click in enumerate(clicks, 1):
                    details = None
                    if step > 1:
                        previous = state
                        proposal = model.refine(previous, [[[click['x'],click['y']]]], [[click['label']]])
                        gaussian = latest_gaussian(model, proposal, valid_t)
                        state = intervene(model, previous, proposal, branch, gaussian, target if branch == 'local_gt_oracle' else None)
                        delta = state.probabilities - previous.probabilities
                        unchanged = delta == 0
                        if branch != 'C0':
                            checks['unchanged_probability_logit_changes'] += int((state.logits[unchanged] != previous.logits[unchanged]).sum())
                        if branch in ('C1','C3'):
                            violations = delta < 0 if click['label'] else delta > 0
                            checks['sign_violation_pixels'] += int(violations.sum())
                        if branch in ('C2','C3','local_gt_oracle'):
                            outside = gaussian == 0
                            checks['outside_gaussian_changes'] += int((delta[outside] != 0).sum())
                            checks['outside_gaussian_changes'] += int((state.logits[outside] != previous.logits[outside]).sum())
                        if branch == 'local_gt_oracle':
                            change = (state.probabilities-target).abs() - (previous.probabilities-target).abs()
                            checks['oracle_native_gt_distance_increase'] = max(checks['oracle_native_gt_distance_increase'], float(change[valid_t.bool()].max()))
                            correct = ((previous.probabilities > .5) == (target > .5)) & valid_t.bool()
                            checks['oracle_native_correct_pixels_damaged'] += int((correct & ((state.probabilities > .5) != (target > .5))).sum())
                        details = dict(gaussian_max=float(gaussian.max()), gaussian_mass=float(gaussian.sum()),
                                       gaussian_ge_half_pixels=int((gaussian>=.5).sum()),
                                       mean_abs_native_update=float(delta[valid_t.bool()].abs().mean()),
                                       direction_mean=float(proposal.direction[valid_t.bool()].mean()),
                                       strength_mean=float(proposal.strength[valid_t.bool()].mean()))
                    metrics, prediction, native = state_metrics(model, state, sample['mask'], truth, valid)
                    assert torch.isfinite(state.hidden).all() and torch.isfinite(state.logits).all()
                    assert state.points.shape[1] == step and all(p.grad is None for p in model.parameters())
                    if branch == 'C0':
                        error = abs(metrics['iou'] - saved[step-1]['iou'])
                        checks['C0_iou_max_error'] = max(checks['C0_iou_max_error'], error)
                        assert error == 0
                    first_counts = transition_counts(first_mask, prediction, sample['mask'])
                    previous_counts = transition_counts(previous_mask, prediction, sample['mask'])
                    native_counts = transition_counts((first_native>.5)[valid], (native>.5)[valid], truth[valid])
                    row = dict(index=index, id=sample['id'], branch=branch, round=step, **click, **metrics,
                        versus_first={**first_counts, **transition_rates(first_counts)},
                        versus_previous={**previous_counts, **transition_rates(previous_counts)},
                        native_versus_first={**native_counts, **transition_rates(native_counts)}, intervention=details)
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    previous_mask, previous_native = prediction, native
            print(f'C0-C3 + local GT {index+1}/16', flush=True)
    assert checks['oracle_native_gt_distance_increase'] <= 1e-7
    for key in ('oracle_native_correct_pixels_damaged','sign_violation_pixels','outside_gaussian_changes','unchanged_probability_logit_changes'):
        assert checks[key] == 0, key
    assert before == {path:file_identity(path)['sha256'] for path in before}
    summary = []
    for branch in BRANCHES:
        for step in range(1,6):
            selected = [r for r in rows if r['branch']==branch and r['round']==step]
            assert len(selected)==16 and len({r['index'] for r in selected})==16
            row = dict(branch=branch, round=step, objects=16,
                **{k:float(np.mean([r[k] for r in selected])) for k in ('iou','precision','recall')})
            for reference_name in ('versus_first','versus_previous','native_versus_first'):
                counts = {k:sum(r[reference_name][k] for r in selected) for k in first_counts}
                row.update({reference_name+'_'+k:v for k,v in {**counts,**transition_rates(counts)}.items()})
            summary.append(row)
    comparison = []
    for branch in BRANCHES:
        selected = {r['round']:r for r in summary if r['branch']==branch}
        comparison.append(dict(branch=branch, oracle=branch=='local_gt_oracle',
            **{f'IoU@{step}':selected[step]['iou'] for step in (1,3,5)},
            **{k:selected[5]['versus_first_'+k] for k in ('fn_repair_rate','fp_removal_rate','tp_damage_rate','tn_damage_rate')},
            precision=selected[5]['precision'],recall=selected[5]['recall']))
    checks.update(objects=16, total_rounds=len(rows), models_and_weights_unchanged=True, hashes=before)
    for filename, value in (('summary',summary),('comparison',comparison),('fixed_clicks',saved_clicks),('verification',checks)):
        dump(output/(filename+'.json'),value)
    write_csv(output/'summary.csv',summary)
    write_csv(output/'comparison.csv',comparison)
    report(output,summary,comparison)
    print(json.dumps(comparison,indent=2))


def report(output, summary, comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for branch in BRANCHES:
        rows=[r for r in summary if r['branch']==branch]
        for ax,key,title in zip(axes,('iou','versus_first_fn_repair_rate','versus_first_tp_damage_rate'),('IoU (%)','Initial FN repaired (%)','Initial TP damaged (%)')):
            ax.plot(range(1,6),[100*r[key] for r in rows], 'o--' if branch=='local_gt_oracle' else 'o-',label=branch)
            ax.set(title=title,xlabel='Click',xticks=range(1,6))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.savefig(output/'comparison.png',dpi=160)
    plt.close(fig)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>C0-C3约束消融</title><style>body{font:16px/1.7 system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}img{max-width:100%}</style><h1>扩容DirectionalPIMSNet：C0-C3固定点击消融</h1>'
    page+='<p>使用已有第100 epoch权重，16实例各五击，所有组坐标和标签一致。统一P&gt;0.5。无训练、无权重或模型源码修改；C0精确复现原有80轮IoU。每组独立反馈修改后的Mask/logits及记忆。</p>'
    page+='<p>C0：原模型；C1：当前正击只允许D≥0，负击只允许D≤0；C2：仅S乘当前点击Gaussian；C3：两种约束同时使用。Gaussian直接复用ClickEncoder的σ=2（64网格）后双线性插值到256，约σ=8原生像素，不截断、不重新归一化峰值，仅排除padding。仅以最新点击约束更新，网络仍接收累积点击。</p>'
    page+='<p>局部GT Oracle：P_new=P_old+G·(GT−P_old)，使用相同Gaussian。它使用GT，是诊断参考，不能作为正式模型性能，也不是原图IoU的严格数学上界。高斯为软权重，因此不等于把整个邻域一次替换成GT。原生GT缩放及logits插值可能导致原图结果与原生单调纠错不同。</p>'
    headers=['分组','IoU@1','IoU@3','IoU@5','FN修复率','FP删除率','TP破坏率']
    keys=('IoU@1','IoU@3','IoU@5','fn_repair_rate','fp_removal_rate','tp_damage_rate')
    page+='<h2>模型推理消融</h2>'+table(headers,[[r['branch'],*[f'{100*r[k]:.3f}%' for k in keys]] for r in comparison if not r['oracle']])
    page+='<h2>GT辅助诊断（单独报告）</h2>'+table(headers,[[r['branch'],*[f'{100*r[k]:.3f}%' for k in keys]] for r in comparison if r['oracle']])
    page+='<p>IoU按对象宏平均；三种编辑率按原图像素汇总，以上比较首击与第五击。分母分别是首击FN、FP、TP，TP破坏率不含背景TN。summary.csv同时给出各轮相对首击、相对上一轮及原生有效网格的统计；per_object.jsonl保留所有逐对象计数。</p><img src="comparison.png">'
    by_name={r['branch']:r for r in comparison}
    c0,c1,c2,c3,oracle=[by_name[b] for b in BRANCHES]
    page+=f'<p><strong>本次观察：</strong>C1相比C0，第五击IoU变化{100*(c1["IoU@5"]-c0["IoU@5"]):+.3f}个百分点，TP破坏率由{100*c0["tp_damage_rate"]:.3f}%降至{100*c1["tp_damage_rate"]:.3f}%。Gaussian约束进一步减少TP破坏，但同时降低FP删除，C2/C3的第五击IoU分别为{100*c2["IoU@5"]:.3f}%/{100*c3["IoU@5"]:.3f}%。这表明在当前σ和软权重下，保护与纠错幅度一起受到抑制，不能仅凭破坏率低判断模型更好。</p>'
    oracle_native=next(r for r in summary if r['branch']=='local_gt_oracle' and r['round']==5)
    page+=f'<p>相同Gaussian下，GT辅助更新达到IoU@5={100*oracle["IoU@5"]:.3f}%，FN修复率{100*oracle["fn_repair_rate"]:.3f}%；模型局部编辑尚未利用这部分纠错空间。但Oracle同时替换了方向与强度，不能把差距单独归因于某一个分支。Oracle在原生256网格破坏TP={oracle_native["native_versus_first_tp_damaged"]}像素、TN={oracle_native["native_versus_first_tn_damaged"]}像素；原图TP破坏率{100*oracle["tp_damage_rate"]:.3f}%来自原生GT及原图重建口径差异，报告没有将其强行记为零。</p>'
    page+='<p>对C1/C2/C3与Oracle，更新后P数值未变化的像素保留旧logits，避免仅logit(eps)转换造成远离点击的端点裁剪。C0原样返回，无任何转换。所有分组使用相同固定点击，因此不能视为各模型重新自适应取点的性能。该结果仅来自16个训练实例，不是泛化性能。</p></html>'
    (output/'report.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default=str(OUTPUT_ROOT/'directional_wide_overfit16_seed42_20261019'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())
