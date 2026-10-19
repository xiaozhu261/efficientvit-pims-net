"""Revised sign-magnitude/Gaussian sweep with clicked-error-component GT Oracle."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluate_directional_constraints import transition_counts, transition_rates
from pims_net.evaluate_eta import state_metrics, write_csv
from pims_net.evaluate_oracle import clicked_error_component
from pims_net.evaluation.directional_overfit_report import table
from pims_net.evaluation.experiment import file_identity
from pims_net.models.click_encoder import ClickEncoder
from pims_net.models.directional_wide import WideDirectionalPIMSNet
from pims_net.overfit_directional import dump
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT, checkpoint_path
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.interactive_losses import prepare_target

GROUPS = [('C0', 'C0', None), ('C1', 'C1', None)] + [
    (f'{branch}_sigma{sigma}', branch, sigma) for branch in ('C2', 'C3', 'Oracle') for sigma in (2, 4, 8)]


def signed_direction(direction, labels):
    signs = labels[:, -1:].reshape(-1, 1, 1, 1) * 2 - 1
    return signs.to(direction.dtype) * direction.abs()


def gaussian_map(encoder, state, valid):
    maps = encoder(state.points[:, -1:], state.labels[:, -1:], state.image_sizes)
    return F.interpolate(maps.sum(1, keepdim=True), size=(256, 256), mode='bilinear', align_corners=False) * valid


def intervene(model, previous, proposal, branch, gaussian=None, truth=None, component=None):
    if branch == 'C0':
        return proposal
    if branch == 'Oracle':
        if truth is None or component is None or gaussian is None:
            raise ValueError('Oracle requires GT, Gaussian and clicked component')
        probability = previous.probabilities + gaussian * component * (truth - previous.probabilities)
        logits = torch.logit(probability, eps=1e-6)
    elif branch in ('C1', 'C2', 'C3'):
        direction, strength = proposal.direction, proposal.strength
        if branch in ('C1', 'C3'):
            direction = signed_direction(direction, proposal.labels)
        if branch in ('C2', 'C3'):
            if gaussian is None:
                raise ValueError('Spatial constraint requires Gaussian')
            strength = strength * gaussian
        probability, logits = model.directional_editor.update(previous.probabilities, previous.logits, direction, strength)
    else:
        raise ValueError('Unknown branch')
    logits = torch.where(probability == previous.probabilities, previous.logits, logits)
    return replace(proposal, probabilities=probability, logits=logits)


def run(args):
    reference, output = Path(args.reference).resolve(), Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError('Use a new directory under pims_net/outputs')
    status = json.loads((reference / 'status.json').read_text())
    assert status['status'] == 'completed'
    saved_rounds = []
    with (reference / 'rounds.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['phase'] == 'eval' and row['epoch'] == status['epochs']:
                saved_rounds.append(row)
    manifest, checkpoint = reference / 'fixed_train16.jsonl', reference / 'last.pt'
    dataset = InteractiveDataset(manifest)
    assert len(dataset) == 16 and len(saved_rounds) == 80
    legacy = OUTPUT_ROOT / 'directional_constraints16_seed42_20261019'
    legacy_rows = [json.loads(line) for line in (legacy / 'per_object.jsonl').read_text().splitlines()]
    watched = [checkpoint, reference / 'best.pt', manifest, reference / 'rounds.jsonl', checkpoint_path(),
               legacy / 'per_object.jsonl', legacy / 'fixed_clicks.json',
               OUTPUT_ROOT / 'directional_overfit16_seed42_20261019/last.pt',
               OUTPUT_ROOT / 'directional_overfit16_seed42_20261019/best.pt']
    watched += list((RESEARCH_ROOT / 'models').glob('*.py')) + list((RESEARCH_ROOT / 'training').glob('*.py'))
    before = {str(path): file_identity(path)['sha256'] for path in watched}
    original = json.loads((OUTPUT_ROOT / 'directional_implementation_20261018/original_hashes.json').read_text())
    for path, sha in original.items():
        assert file_identity(path)['sha256'] == sha
    before.update(original)
    torch.set_num_threads(4)
    model = WideDirectionalPIMSNet().to(args.device).eval().requires_grad_(False)
    info = load_training_checkpoint(checkpoint, model)
    assert info['epoch'] == status['epochs'] and info['config']['threshold'] == .5
    # External spatial gates only. The model's internal click encoding stays sigma=2.
    encoders = {sigma: ClickEncoder(sigma).to(args.device) for sigma in (2, 4, 8)}
    output.mkdir(parents=True)
    dump(output / 'protocol.json', dict(version=2, checkpoint=file_identity(checkpoint),
        manifest=file_identity(manifest), epoch=info['epoch'], groups=GROUPS, threshold=.5,
        click_source=file_identity(reference / 'rounds.jsonl'), fixed_clicks_reference=file_identity(legacy / 'fixed_clicks.json'),
        C0='Original wide model.refine, unchanged.',
        C1='D_effective=(2*latest_click_label-1)*abs(D_raw). Preserve direction magnitude, flip inconsistent signs; global constraint.',
        C2='S_effective=S_raw*Gaussian_latest; D unchanged.', C3='Combine C1 and C2.',
        gaussian='External sigma=2/4/8 on 64x64 grid; original 512-resize click transform; bilinear to256 align_corners=False. No peak renormalization or hard truncation; padding zero.',
        internal_click_encoder='Unchanged sigma=2, cumulative clicks; external spatial gate uses latest click only.',
        oracle='Select current-click same-sign GT error 8-connected component using own previous P>0.5. P_new=P_old+G*component*(GT-P_old). Empty component => exact mask/logit KEEP.',
        component_geometry=clicked_error_component.__doc__,
        oracle_scope='GT-assisted diagnostic only, not model performance or a rigorous original-resolution IoU upper bound.',
        feedback='Own updated probabilities/logits/ConvGRU history per group. Hidden state and click history advance even if Oracle mask update is skipped.',
        keep='Non-C0 branches retain previous raw logits wherever probability is numerically unchanged.',
        metrics='Original resolution P>0.5. IoU macro mean. FN/FP/TP rates pooled pixels; main table references first-click mask. Also save per-transition and native-grid counts.',
        trained=False, device=args.device))
    del info
    checks = dict(C0_replay_iou_error=0., C2_sigma2_legacy_metric_error=0., sign_violations=0,
                  oracle_native_correct_damage=0, oracle_native_gt_distance_increase=0.,
                  oracle_outside_component_changes=0, exact_keep_logit_changes=0,
                  empty_oracle_rounds={str(s):0 for s in encoders})
    rows, fixed_clicks = [], []
    with torch.no_grad(), (output / 'per_object.jsonl').open('w', encoding='utf-8') as stream:
        for index in range(16):
            sample = dataset[index]
            saved = sorted([r for r in saved_rounds if r['index']==index], key=lambda r:r['round'])
            assert [r['round'] for r in saved] == list(range(1,6)) and all(r['id']==sample['id'] for r in saved)
            clicks = [{k:r[k] for k in ('x','y','label')} for r in saved]
            fixed_clicks.append(dict(index=index,id=sample['id'],clicks=clicks))
            first = clicks[0]
            initial = model.initialize(sample['image'], [[[first['x'], first['y']]]], [[first['label']]])
            target, valid_t = prepare_target(sample['mask'], args.device)
            truth, valid = target[0,0].cpu().numpy().astype(bool), valid_t[0,0].cpu().numpy().astype(bool)
            _, first_mask, first_native = state_metrics(model, initial, sample['mask'], truth, valid)
            for group, branch, sigma in GROUPS:
                state, previous_mask = initial, first_mask
                for step, click in enumerate(clicks, 1):
                    details = None
                    if step > 1:
                        previous = state
                        component = None
                        if branch == 'Oracle':
                            gate, details = clicked_error_component(previous.probabilities[0,0].cpu().numpy(), truth, valid, click, sample['mask'].shape)
                            component = torch.as_tensor(gate, device=args.device)[None,None]
                            checks['empty_oracle_rounds'][str(sigma)] += int(not gate.any())
                        proposal = model.refine(previous, [[[click['x'],click['y']]]], [[click['label']]])
                        gaussian = gaussian_map(encoders[sigma], proposal, valid_t) if sigma is not None else None
                        state = intervene(model, previous, proposal, branch, gaussian, target if branch=='Oracle' else None, component)
                        delta = state.probabilities - previous.probabilities
                        if branch != 'C0':
                            checks['exact_keep_logit_changes'] += int((state.logits[delta==0] != previous.logits[delta==0]).sum())
                        if branch in ('C1','C3'):
                            direction = signed_direction(proposal.direction, proposal.labels)
                            assert torch.equal(direction.abs(), proposal.direction.abs())
                            checks['sign_violations'] += int((delta<0 if click['label'] else delta>0).sum())
                        if gaussian is not None:
                            assert torch.equal(state.probabilities[gaussian==0], previous.probabilities[gaussian==0])
                        if branch == 'Oracle':
                            outside = ~component
                            checks['oracle_outside_component_changes'] += int((delta[outside]!=0).sum())
                            checks['oracle_outside_component_changes'] += int((state.logits[outside]!=previous.logits[outside]).sum())
                            correct = ((previous.probabilities>.5)==target.bool()) & valid_t.bool()
                            checks['oracle_native_correct_damage'] += int((correct & ((state.probabilities>.5)!=target.bool())).sum())
                            distance = (state.probabilities-target).abs()-(previous.probabilities-target).abs()
                            checks['oracle_native_gt_distance_increase'] = max(checks['oracle_native_gt_distance_increase'],float(distance[valid_t.bool()].max()))
                            if not component.any():
                                assert torch.equal(state.probabilities, previous.probabilities) and torch.equal(state.logits, previous.logits)
                    metrics, prediction, native = state_metrics(model, state, sample['mask'], truth, valid)
                    assert state.points.shape[1]==step and torch.isfinite(state.hidden).all() and torch.isfinite(state.logits).all()
                    if branch=='C0':
                        checks['C0_replay_iou_error'] = max(checks['C0_replay_iou_error'], abs(metrics['iou']-saved[step-1]['iou']))
                    if group=='C2_sigma2':
                        old=next(r for r in legacy_rows if r['index']==index and r['round']==step and r['branch']=='C2')
                        for key in ('iou','tp','tn','fp','fn','precision','recall'):
                            checks['C2_sigma2_legacy_metric_error']=max(checks['C2_sigma2_legacy_metric_error'],abs(metrics[key]-old[key]))
                    references = dict(versus_first=transition_counts(first_mask,prediction,sample['mask']),
                                      versus_previous=transition_counts(previous_mask,prediction,sample['mask']),
                                      native_versus_first=transition_counts((first_native>.5)[valid],(native>.5)[valid],truth[valid]))
                    row = dict(index=index,id=sample['id'],group=group,branch=branch,sigma=sigma,round=step,**click,**metrics,
                               **{k:{**v,**transition_rates(v)} for k,v in references.items()},component=details)
                    rows.append(row)
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                    previous_mask=prediction
            print(f'Revised C0-C3 / component Oracle {index+1}/16',flush=True)
    assert all(p.grad is None for p in model.parameters()) and model.click_encoder.sigma==2
    for key in ('C0_replay_iou_error','C2_sigma2_legacy_metric_error','sign_violations','oracle_native_correct_damage','oracle_outside_component_changes','exact_keep_logit_changes'):
        assert checks[key]==0, key
    assert checks['oracle_native_gt_distance_increase'] <= 1e-7
    assert fixed_clicks==json.loads((legacy/'fixed_clicks.json').read_text())
    assert before=={path:file_identity(path)['sha256'] for path in before}
    summary=[]
    for group,branch,sigma in GROUPS:
        for step in range(1,6):
            selected=[r for r in rows if r['group']==group and r['round']==step]
            assert len(selected)==16 and len({r['index'] for r in selected})==16
            row=dict(group=group,branch=branch,sigma=sigma,round=step,objects=16,
                     **{k:float(np.mean([r[k] for r in selected])) for k in ('iou','precision','recall')})
            for prefix in references:
                counts={k:sum(r[prefix][k] for r in selected) for k in references[prefix]}
                row.update({prefix+'_'+k:v for k,v in {**counts,**transition_rates(counts)}.items()})
            summary.append(row)
    comparison=[]
    for group,branch,sigma in GROUPS:
        selected={r['round']:r for r in summary if r['group']==group}
        comparison.append(dict(group=group,branch=branch,sigma=sigma,oracle=branch=='Oracle',
            **{f'IoU@{step}':selected[step]['iou'] for step in (1,3,5)},
            **{k:selected[5]['versus_first_'+k] for k in ('fn_repair_rate','fp_removal_rate','tp_damage_rate','tn_damage_rate')},
            precision=selected[5]['precision'],recall=selected[5]['recall']))
    checks.update(objects=16,groups=len(GROUPS),rounds=len(rows),weights_and_model_sources_unchanged=True,
                  original_internal_sigma=2,identical_previous_click_sequences=True,hashes=before)
    for name,value in (('summary',summary),('comparison',comparison),('fixed_clicks',fixed_clicks),('verification',checks)):
        dump(output/(name+'.json'),value)
    write_csv(output/'summary.csv',summary)
    write_csv(output/'comparison.csv',comparison)
    report(output,summary,comparison,checks)
    print(json.dumps(comparison,indent=2))


def report(output, summary, comparison, checks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,sigma in zip(axes,(2,4,8)):
        for group in ('C0','C1',f'C2_sigma{sigma}',f'C3_sigma{sigma}',f'Oracle_sigma{sigma}'):
            rows=[r for r in summary if r['group']==group]
            ax.plot(range(1,6),[100*r['iou'] for r in rows],'o--' if group.startswith('Oracle') else 'o-',label=group)
        ax.set(title=f'External Gaussian sigma={sigma}',xlabel='Click',ylabel='IoU (%)',xticks=range(1,6))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output/'comparison.png',dpi=160)
    plt.close(fig)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>C0-C3修订消融</title><style>body{font:16px/1.7 system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}img{max-width:100%}</style><h1>C0–C3修订版：保留方向幅度与Gaussian半径消融</h1>'
    page+='<p>扩容版第100 epoch权重；固定原16实例五击坐标和标签。全部阈值P&gt;0.5。无训练，模型源码和已有权重哈希不变，旧实验结果保留。</p><p>C0原模型；C1为点击符号×|原Direction|（翻转不一致方向，保留幅度，不再裁剪为零）；C2仅Strength乘Gaussian；C3同时施加两者。外部Gaussian分别σ=2/4/8，约对应256网格σ=8/16/32像素。模型内部ClickEncoder始终σ=2；外部Gaussian只取最新点击，不归一化峰值、不硬截断，双线性插值后排除padding。</p>'
    page+='<p>Oracle：在各自上一轮Mask上，寻找当前点击所在的同符号GT错误8连通域（正击FN，负击FP），只在该域内按P_new=P_old+G·(GT−P_old)更新；没有对应错误则Mask和logits严格KEEP，隐藏状态与点击历史继续推进。不寻找最近错误、不扩大连通域。点击按官方1024坐标变换/4后四舍五入映射到原生网格；Gaussian沿用512图像特征变换/8再插值。Oracle单独作为GT辅助诊断，不是正式模型性能或严格原图IoU上界。</p>'
    headers=['分组','σ(64网格)','IoU@1','IoU@3','IoU@5','FN修复','FP删除','TP破坏']
    keys=('IoU@1','IoU@3','IoU@5','fn_repair_rate','fp_removal_rate','tp_damage_rate')
    for oracle,title in ((False,'模型推理消融'),(True,'GT错误连通域Oracle（诊断参考）')):
        page+='<h2>'+title+'</h2>'+table(headers,[[r['branch'],r['sigma'] if r['sigma'] is not None else '—',*[f'{100*r[k]:.3f}%' for k in keys]] for r in comparison if r['oracle']==oracle])
    page+='<p>IoU为16对象宏平均；编辑率为原图像素汇总。主表比较首击与第五击，FN/FP/TP分母分别为首击FN/FP/TP，TP不含背景。summary.csv另存每轮相对上一轮和原生网格的结果；per_object.jsonl保留像素计数与每次Oracle连通域选择。</p><img src="comparison.png">'
    by_name={r['group']:r for r in comparison}
    c0,c1=by_name['C0'],by_name['C1']
    page+=f'<h2>结果解释</h2><p>C1相比C0的IoU@5变化{100*(c1["IoU@5"]-c0["IoU@5"]):+.2f}个百分点。虽然TP破坏率由{100*c0["tp_damage_rate"]:.3f}%降至{100*c1["tp_damage_rate"]:.3f}%，但首击正确背景被误加为前景的比例由{100*c0["tn_damage_rate"]:.6f}%升至{100*c1["tn_damage_rate"]:.3f}%；对象平均Precision由{100*c0["precision"]:.2f}%降至{100*c1["precision"]:.2f}%。全图翻转方向在正击时也会把背景上的REMOVE变为ADD，因此仅看TP破坏会漏掉这部分错误。</p>'
    page+='<p>C2/C3随着σ增大，纠错范围及有效更新权重增加，第五击IoU提高，但本次均未超过C0。相同Gaussian下的错误连通域Oracle明显更高，说明点击邻域仍有GT辅助可修复空间；Oracle同时使用GT方向和GT错误区域，不能将差距单独归因于Direction或Strength。</p>'
    page+='<p>Oracle第2–5轮共64次编辑尝试，各σ因固定点击已不在同符号错误域而KEEP的次数：'+str(checks['empty_oracle_rounds'])+'。Oracle原生正确像素破坏为0，原图TP破坏可因GT缩放和logits插值产生差异，实际计数保留。C0与原训练记录精确一致；C2 σ=2与上一版实验精确一致。所有非C0分组在P数值未变化处保留旧logits，避免仅端点裁剪造成编辑。</p><p>这是16个训练对象的固定点击推理敏感性实验，不代表泛化性能或各组自适应取点性能。不同组反馈各自的新Mask与记忆，后续Direction/Strength会随之变化。</p></html>'
    (output/'report.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default=str(OUTPUT_ROOT/'directional_wide_overfit16_seed42_20261019'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())
