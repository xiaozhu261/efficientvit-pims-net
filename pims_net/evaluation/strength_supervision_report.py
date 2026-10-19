"""Verified S0/S1/S2 metrics, Strength distributions and gradient comparison."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.diagnose_large_clicks import describe
from pims_net.evaluate_directional_constraints import transition_counts,transition_rates
from pims_net.evaluate_eta import state_metrics,write_csv
from pims_net.evaluate_oracle import clicked_error_component
from pims_net.evaluation.directional_loss_report import lines,pooled_metrics,pair_summary
from pims_net.evaluation.directional_overfit_report import table
from pims_net.evaluation.experiment import file_identity
from pims_net.models.directional_large import LargeDirectionalPIMSNet
from pims_net.overfit_directional import dump
from pims_net.overfit_strength_supervision import OBJECTIVES,PAIRS
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.interactive_losses import prepare_target
from pims_net.training.strength_supervised_losses import harmful_strength_region

VARIANTS=('S0','S1','S2')


@torch.no_grad()
def replay(root,variant,checkpoint,records,device,out):
    model=LargeDirectionalPIMSNet().to(device).eval().requires_grad_(False)
    info=load_training_checkpoint(root/checkpoint,model)
    dataset=InteractiveDataset(root/'fixed_train16.jsonl')
    selected=[r for r in records if r['epoch']==info['epoch']]
    assert len(selected)==80
    captured=[]
    handle=model.directional_decoder.head.register_forward_hook(lambda m,inputs,result:captured.append(result))
    rows,stats,pools=[],[],defaultdict(list)
    max_error=loss_error=0.
    for index in range(16):
        sample=dataset[index]
        target,valid=prepare_target(sample['mask'],device)
        truth,valid_np=target[0,0].cpu().numpy().astype(bool),valid[0,0].cpu().numpy().astype(bool)
        first=None
        steps=sorted([r for r in selected if r['index']==index],key=lambda r:r['round'])
        for step in steps:
            points,labels=[[[step['x'],step['y']]]],[[step['label']]]
            if step['round']==1:
                state=model.initialize(sample['image'],points,labels)
            else:
                previous=state.probabilities
                old=previous[0,0].cpu().numpy()
                component,cc=clicked_error_component(old,truth,valid_np,step,sample['mask'].shape)
                state=model.refine(state,points,labels)
                raw=captured.pop()
                assert torch.equal(state.strength,raw[:,1:2].sigmoid())
                harmful=harmful_strength_region(previous,target,valid,state.direction)
                assert not torch.any(harmful&(((previous>.5)!=(target>.5))|~valid.bool()))
                expected_activation=int(component.sum()) if variant!='S0' and step['label']==1 else 0
                expected_suppression=int(harmful.sum()) if variant=='S2' else 0
                assert expected_activation==step['support']['activation_pixels']
                assert expected_suppression==step['support']['suppression_pixels']
                gate=torch.as_tensor(component,device=device)[None,None]
                activate=F.softplus(-raw[:,1:2][gate]).mean() if expected_activation else torch.tensor(0.)
                suppress=F.softplus(raw[:,1:2][harmful]).mean() if expected_suppression else torch.tensor(0.)
                loss_error=max(loss_error,abs(float(activate)-step['losses']['strength_activate']),
                               abs(float(suppress)-step['losses']['strength_suppress']))
                assert loss_error<2e-5
                if checkpoint=='last.pt':
                    s=state.strength[0,0].cpu().numpy()
                    z=raw[0,1].cpu().numpy()
                    d=state.direction[0,0].cpu().numpy()
                    new=state.probabilities[0,0].cpu().numpy()
                    h=harmful[0,0].cpu().numpy()
                    fields=dict(strength=s,logit=z,direction=d,delta_p=new-old,
                        low_strength=(s<.01).astype(np.float32),high_strength=(s>.95).astype(np.float32),
                        exact_endpoint=((s==0)|(s==1)).astype(np.float32),
                        effective_strength=s*np.abs(d),near_zero_direction=(np.abs(d)<=.05).astype(np.float32),
                        direction_aligned=(d>0 if step['label'] else d<0).astype(np.float32),
                        crossed=(new>.5 if step['label'] else new<=.5).astype(np.float32))
                    regions=dict(target=component,correct=valid_np&((old>.5)==truth),harmful=h,
                        FN=valid_np&truth&(old<=.5),FP=valid_np&~truth&(old>.5),outside_target=valid_np&~component)
                    for region,mask in regions.items():
                        for field,array in fields.items():
                            if mask.any():
                                pools[step['label'],region,field].append(array[mask].copy())
                    filename=f'{variant}_object_{index:02d}_round_{step["round"]}.npz'
                    np.savez_compressed(out/'native'/filename,previous=old,probabilities=new,strength=s,logit=z,
                        direction=d,truth=truth,valid=valid_np,component=component,harmful=h)
            metrics,prediction,_=state_metrics(model,state,sample['mask'],truth,valid_np)
            max_error=max(max_error,abs(metrics['iou']-step['iou']))
            assert max_error==0
            if first is None:
                first=prediction.copy()
            counts=transition_counts(first,prediction,sample['mask'])
            assert counts==step['cumulative_counts']
            rows.append(dict(variant=variant,checkpoint=checkpoint,epoch=info['epoch'],index=index,
                round=step['round'],label=step['label'],**metrics,**counts,**transition_rates(counts)))
    handle.remove()
    assert not captured
    for (label,region,field),arrays in sorted(pools.items()):
        stats.append(dict(variant=variant,label=label,region=region,field=field,**describe(np.concatenate(arrays))))
    return dict(variant=variant,checkpoint=checkpoint,epoch=info['epoch'],updates=info['global_step'],
                replayed_rounds=80,iou_max_error=max_error,transition_counts_exact=True,
                auxiliary_loss_max_error=loss_error,harmful_filter_excludes_all_errors=True),rows,stats


def report(args):
    out=Path(args.directory).resolve()
    (out/'native').mkdir(exist_ok=True)
    configs={v:json.loads((out/v/'config.json').read_text()) for v in VARIANTS}
    assert len({c['initial_research_sha256'] for c in configs.values()})==1
    assert len({(out/v/'fixed_train16.jsonl').read_bytes() for v in VARIANTS})==1
    curves,records={},{}
    comparison,rounds,epochs,gradients,norms,coverage=[],[],[],[],[],[]
    firsts={}
    max_loss_sum_error=0.
    for variant in VARIANTS:
        root=out/variant
        c=configs[variant]
        status=json.loads((root/'status.json').read_text())
        assert (status['status'],status['epochs'],status['optimizer_updates'])==('completed',100,800)
        assert (c['seed'],c['batch_size'],c['threshold'],c['lr'],c['weight_decay'])==(42,2,.5,1e-4,1e-4)
        assert c['parameter_counts']==dict(decoder=1648450,trainable=2220386)
        assert c['strength_loss_weights']==dict(activate=.05 if variant!='S0' else 0.,suppress=.01 if variant=='S2' else 0.)
        logs=lines(root/'epochs.jsonl')
        sequences=lines(root/'sequences.jsonl')
        evaluation=[]
        with (root/'rounds.jsonl').open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['losses'] is not None:
                    t=r['losses']
                    expected=t['bce']+t['dice']+t['direction']+.1*t['preserve']+.05*t['strength_activate']+.01*t['strength_suppress']
                    max_loss_sum_error=max(max_loss_sum_error,abs(expected-t['loss']))
                if r['phase']=='eval':
                    evaluation.append(r)
        assert max_loss_sum_error<2e-5
        records[variant]=evaluation
        curves[variant]=[r for r in logs if r['phase']=='eval']
        assert len(logs)==121 and len(sequences)==1936
        for log in logs:
            selected=[r for r in sequences if (r['phase'],r['epoch'])==(log['phase'],log['epoch'])]
            assert len(selected)==16
            for key in ('IoU@1','IoU@3','IoU@5','native_IoU5'):
                assert abs(log[key]-np.mean([r[key] for r in selected]))<1e-12
            for group in ('all','shared','strength_head','memory','head'):
                for key in (*OBJECTIVES,'total'):
                    values=[r['parameter_gradient_details'][group]['norms'][key] for r in selected if r['parameter_gradient_details']]
                    assert abs(log['gradient_norms'][group][key]-np.mean(values))<1e-12
        firsts[variant]=[(r['index'],r['x'],r['y'],r['label'],r['iou']) for r in evaluation if r['epoch']==100 and r['round']==1]
        for stage,log in (('final',curves[variant][-1]),('best',max(curves[variant],key=lambda r:r['IoU@5']))):
            selected=[r for r in evaluation if r['epoch']==log['epoch']]
            final=[r for r in selected if r['round']==5]
            assert len(final)==16
            comparison.append(dict(variant=variant,stage=stage,epoch=log['epoch'],updates=log['optimizer_updates'],
                **{key:log[key] for key in ('IoU@1','IoU@3','IoU@5')},**pooled_metrics(final),losses=log['losses']))
            for step in range(1,6):
                selected_step=[r for r in selected if r['round']==step]
                for scope,key in (('from_first','cumulative_counts'),('from_previous','incremental_counts')):
                    rounds.append(dict(variant=variant,stage=stage,epoch=log['epoch'],round=step,scope=scope,
                                       **pooled_metrics(selected_step,key)))
        for log in curves[variant]:
            selected=[r for r in evaluation if r['epoch']==log['epoch'] and r['round']==5]
            editing=[r for r in evaluation if r['epoch']==log['epoch'] and r['round']>1]
            extra={}
            for name,label in (('positive_target',1),('negative_target',0),('harmful',None)):
                region='harmful' if name=='harmful' else 'target'
                data=[r['strength_regions'][region]['strength'] for r in editing if (label is None or r['label']==label)
                      and r['strength_regions'][region]['strength'] is not None]
                count=sum(d['count'] for d in data)
                extra[name+'_strength']=sum(d['count']*d['mean'] for d in data)/count if count else None
            epochs.append(dict(variant=variant,epoch=log['epoch'],**pooled_metrics(selected),**extra))
        for window,seqs in (('train_all',[r for r in sequences if r['phase']=='train']),
                            ('train_last20',[r for r in sequences if r['phase']=='train' and r['epoch']>80]),
                            ('final_eval',[r for r in sequences if r['phase']=='eval' and r['epoch']==100])):
            for group in ('all','shared','strength_head','memory','head'):
                details=[r['parameter_gradient_details'][group] for r in seqs if r['parameter_gradient_details']]
                for key in (*OBJECTIVES,'total'):
                    projections=[r['projections'][key] for r in details if r['projections'][key] is not None]
                    norms.append(dict(variant=variant,window=window,group=group,loss=key,observations=len(details),
                        mean_norm=float(np.mean([r['norms'][key] for r in details])),
                        zero_fraction=float(np.mean([r['norms'][key]==0 for r in details])),
                        mean_projection=float(np.mean(projections)) if projections else None))
                for name,_,_ in PAIRS:
                    gradients.append(dict(variant=variant,window=window,group=group,pair=name,
                                          **pair_summary([r['pairs'][name] for r in details])))
        for label in (1,0):
            selected=[r for r in evaluation if r['epoch']==100 and r['round']>1 and r['label']==label]
            coverage.append(dict(variant=variant,label=label,events=len(selected),
                nonempty_target=sum(r['clicked_component']['selected_error'] for r in selected),
                activation_pixels=sum(r['support']['activation_pixels'] for r in selected),
                suppression_pixels=sum(r['support']['suppression_pixels'] for r in selected),
                harmful_pixels=sum(r['support']['harmful_pixels'] for r in selected)))
    assert firsts['S0']==firsts['S1']==firsts['S2']
    paired=[]
    for index in range(16):
        selected={v:next(r for r in records[v] if r['epoch']==100 and r['round']==5 and r['index']==index) for v in VARIANTS}
        assert len({r['id'] for r in selected.values()})==1
        paired.append(dict(index=index,id=selected['S0']['id'],**{v:r['iou'] for v,r in selected.items()},
            delta_S1=selected['S1']['iou']-selected['S0']['iou'],delta_S2=selected['S2']['iou']-selected['S0']['iou']))
    verifications,replay_rows,stats=[],[],[]
    for variant in VARIANTS:
        for checkpoint in ('last.pt','best.pt'):
            checked,rows,distributions=replay(out/variant,variant,checkpoint,records[variant],args.device,out)
            verifications.append(checked)
            replay_rows.extend(rows)
            stats.extend(distributions)
            print(json.dumps(checked),flush=True)
    watched={}
    for variant in VARIANTS:
        watched.update(json.loads((out/variant/'verification.json').read_text())['original_hashes'])
    for path,sha in watched.items():
        assert file_identity(path)['sha256']==sha,path
    verification=dict(architecture_unchanged=True,identical_initial_parameters=True,same_instances=True,same_first_clicks=True,
        updates_each=800,source_and_old_weights_unchanged=True,epoch_metrics_and_gradient_norms_recomputed=True,
        loss_sum_max_error=max_loss_sum_error,activation_weight=.05,harmful_suppression_weight=.01,replays=verifications,
        paired_final={v:dict(improved=sum(r['delta_'+v]>0 for r in paired),degraded=sum(r['delta_'+v]<0 for r in paired),
            equal=sum(r['delta_'+v]==0 for r in paired)) for v in ('S1','S2')},
        checkpoint_hashes={v+'/'+f:file_identity(out/v/f)['sha256'] for v in VARIANTS for f in ('last.pt','best.pt')})
    for name,data in (('comparison',comparison),('round_comparison',rounds),('epoch_metrics',epochs),('gradient_norms',norms),
                      ('gradient_pairs',gradients),('strength_distributions',stats),('supervision_coverage',coverage),
                      ('paired_objects',paired),('checkpoint_replay',replay_rows)):
        dump(out/(name+'.json'),data)
        write_csv(out/(name+'.csv'),[{k:v for k,v in r.items() if k!='losses'} for r in data])
    dump(out/'verification.json',verification)
    render(out,comparison,rounds,epochs,norms,gradients,stats,coverage,verification)


def render(out,comparison,rounds,epochs,norms,pairs,stats,coverage,verification):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def pct(v): return '—' if v is None else f'{100*v:.2f}%'
    def num(v): return '—' if v is None else f'{v:.4g}'
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for variant in VARIANTS:
        data=[r for r in epochs if r['variant']==variant]
        for ax,field,title in zip(axes.flat,('iou','fn_repair_rate','fp_removal_rate','tp_damage_rate','positive_target_strength','harmful_strength'),
            ('Train16 IoU@5','FN repaired vs first','FP removed vs first','TP damaged vs first','Positive target mean Strength','Harmful correct-region mean Strength')):
            ax.plot([r['epoch'] for r in data],[None if r[field] is None else 100*r[field] for r in data],label=variant)
            ax.set(title=title,xlabel='Epoch',ylabel='%')
            ax.grid(alpha=.2)
            ax.legend()
    fig.savefig(out/'strength_supervision_comparison.png',dpi=160)
    plt.close(fig)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Strength监督消融</title><style>body{font:16px/1.7 system-ui;margin:32px}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:7px}img{max-width:100%}</style><h1>大版DirectionalPIMSNet：S0/S1/S2独立训练</h1>'
    page+='<p>同一16训练实例、seed42、完全相同研究参数初始化、100epoch/800次AdamW更新；batch2、lr1e-4、weight decay1e-4、clip1、统一P&gt;0.5。冻结官方SAM首击，原大版架构及完整推理不变。各组使用相同RITM闭环协议，后续点击坐标随自身预测变化。主表比较800更新最终权重，另报预算内最佳；仅说明单种子训练集拟合能力。</p>'
    page+=table(['组别','保留原L0全部损失','Strength激活','Strength抑制'],[
        ['S0','BCE+Dice+Direction+0.1 Preserve','无','无'],
        ['S1','全部保留','0.05 BCEWithLogits(z_S,1)，仅正点击所在FN连通域','无'],
        ['S2','全部保留','同S1','0.01 BCEWithLogits(z_S,0)，仅TP且D<0或TN且D>0']])
    page+='<p>FN连通域：编辑前原生256有效域中、当前点击所在的同符号错误8连通域；坐标采用官方1024变换/4取最近像素，不吸附其他错误。TP/TN依据旧P&gt;0.5和GT，Direction使用本轮预测值，全部筛选条件detach。S2对正负点击均适用，不抑制其他未修复FN/FP；激活与抑制区域独立取均值，空区域跳过。首击不训练新模块，不使用初始随机Decoder作为首击。</p>'
    def comparison_table(rows):
        return table(['组别','epoch','更新','IoU@1','IoU@3','IoU@5','FN修复','FP删除','TP破坏','TN破坏'],[
            [r['variant'],r['epoch'],r['updates'],*[pct(r[k]) for k in ('IoU@1','IoU@3','IoU@5')],
             f'{100*r["fn_repair_rate"]:.3f}% ({r["fn_repaired"]}/{r["fn_before"]})',pct(r['fp_removal_rate']),
             f'{100*r["tp_damage_rate"]:.3f}%',f'{100*r["tn_damage_rate"]:.4f}%'] for r in rows])
    page+='<h2>800步最终权重</h2>'+comparison_table([r for r in comparison if r['stage']=='final'])
    final={r['variant']:r for r in comparison if r['stage']=='final'}
    page+='<p>'+''.join(f'{v}相对S0：IoU@5 {100*(final[v]["IoU@5"]-final["S0"]["IoU@5"]):+.2f}个百分点，FN修复率 {100*(final[v]["fn_repair_rate"]-final["S0"]["fn_repair_rate"]):+.2f}个百分点，TP破坏率 {100*(final[v]["tp_damage_rate"]-final["S0"]["tp_damage_rate"]):+.2f}个百分点。' for v in ('S1','S2'))+'</p>'
    page+='<p>'+''.join(f'{v}相对S0的16对象IoU@5：改善{verification["paired_final"][v]["improved"]}、退化{verification["paired_final"][v]["degraded"]}、持平{verification["paired_final"][v]["equal"]}。' for v in ('S1','S2'))+'逐对象结果见paired_objects.csv。</p>'
    page+='<p>IoU为原图尺度对象宏平均；编辑率相对首击、汇总原图像素计数。FN分母=首击FN，TP破坏分母=首击TP，另列FP删除与TN破坏。round_comparison另含相对上一击口径。</p>'
    page+='<h2>预算内最佳检查点</h2>'+comparison_table([r for r in comparison if r['stage']=='best'])
    page+='<img src="strength_supervision_comparison.png">'
    page+='<h2>最终逐击结果</h2>'+table(['组别','点击','IoU','FN修复','FP删除','TP破坏'],[
        [r['variant'],r['round'],*[pct(r[k]) for k in ('iou','fn_repair_rate','fp_removal_rate','tp_damage_rate')]] for r in rounds if r['stage']=='final' and r['scope']=='from_first'])
    def distribution(v,label,region,field,key='mean'):
        row=next((r for r in stats if (r['variant'],r['label'],r['region'],r['field'])==(v,label,region,field)),None)
        return row[key] if row else None
    page+='<h2>最终固定权重2–5击Strength分布</h2><p>按最新点击符号、原生有效区域像素×事件精确合并；分位数由重新回放保存的原始数组计算，不平均各对象分位数。目标域为正FN/负FP，correct与harmful均在编辑前确定。</p>'
    page+=table(['组别','点击','目标像素次数','目标S均值','目标S中位数','目标logit中位数','目标S<0.01','目标S>0.95','正确区域S均值','有害区域S均值'],[
        [v,'正FN' if l else '负FP',distribution(v,l,'target','strength','count'),num(distribution(v,l,'target','strength')),
         num(distribution(v,l,'target','strength','median')),num(distribution(v,l,'target','logit','median')),
         pct(distribution(v,l,'target','low_strength')),pct(distribution(v,l,'target','high_strength')),
         num(distribution(v,l,'correct','strength')),num(distribution(v,l,'harmful','strength'))] for v in VARIANTS for l in (1,0)])
    page+='<h2>目标域Direction与实际编辑幅度</h2><p>辅助诊断：S高并不保证有效更新，需同时检查Direction。有效幅度m=S×|D|，实际ΔP还受旧P到0/1距离约束。跨阈值率按每轮编辑前点击错误域内像素计算，属于原生256单步诊断，不同于前文相对首击的原图FN修复率。</p>'
    page+=table(['组别','点击','Direction均值','Direction中位数','方向正确比例','|D|≤0.05','S×|D|均值','实际ΔP均值','原生目标域跨阈值'],[
        [v,'正FN' if l else '负FP',num(distribution(v,l,'target','direction')),num(distribution(v,l,'target','direction','median')),
         pct(distribution(v,l,'target','direction_aligned')),pct(distribution(v,l,'target','near_zero_direction')),num(distribution(v,l,'target','effective_strength')),
         num(distribution(v,l,'target','delta_p')),pct(distribution(v,l,'target','crossed'))] for v in VARIANTS for l in (1,0)])
    page+='<h2>本次实验的主要发现</h2>'
    if distribution('S1',1,'target','strength')>.95 and distribution('S1',1,'target','direction_aligned')<.01:
        page+=f'<p>S1已将正点击目标FN域Strength均值提高到{num(distribution("S1",1,"target","strength"))}，但该域Direction正确比例仅{pct(distribution("S1",1,"target","direction_aligned"))}，Direction均值为{num(distribution("S1",1,"target","direction"))}。门控打开后方向仍错误，原生目标域跨阈值率为{pct(distribution("S1",1,"target","crossed"))}；整图相对首击仅修复{final["S1"]["fn_repaired"]}个FN像素。Strength的目标值学对不等于学会有效ADD。</p>'
    if final['S2']['fn_repair_rate']>final['S0']['fn_repair_rate'] and final['S2']['tp_damage_rate']<final['S0']['tp_damage_rate']:
        page+=f'<p>S2提高了FN修复（{pct(final["S2"]["fn_repair_rate"])}）并降低TP破坏（{100*final["S2"]["tp_damage_rate"]:.3f}%），但FP删除从S0的{pct(final["S0"]["fp_removal_rate"])}降到{pct(final["S2"]["fp_removal_rate"])}。S2负点击FP域Strength均值仍有{num(distribution("S2",0,"target","strength"))}，Direction正确率却仅{pct(distribution("S2",0,"target","direction_aligned"))}，并非简单的低Strength门控。它没有同时稳定学会ADD和REMOVE。</p>'
    late={(r['variant'],r['group'],r['pair']):r for r in pairs if r['window']=='train_last20'}
    q=late['S2','strength_head','strength_activate__strength_suppress']
    shared=late['S2','shared','strength_activate__strength_suppress']
    page+=f'<p>末20epoch，S2激活与抑制在Strength Head的{q["valid_cosines"]}个有效对象序列中，梯度冲突率为{pct(q["conflict_fraction"])}，平均余弦{num(q["mean_cosine"])}；共享参数平均余弦为{num(shared["mean_cosine"])}。区域不重叠并不消除共享参数上的竞争。结合Direction分布和FP删除下降，新增监督解决了低门控的一部分问题，但未消除编辑方向与区域保护的权衡。以上为当前16实例和预算的观察，不能断言唯一训练成因。</p>'
    page+='<h2>各辅助损失的完整BPTT参数梯度范数</h2><p>每对象4轮平均加权损失，clip/AdamW前统计；Strength Head只包含输出层第1行，shared=Decoder trunk+ConvGRU。范数为事件宏平均；总梯度投影允许为负、可能存在抵消。损失权重已计入：激活0.05，抑制0.01。未启用的损失范数为0。</p>'
    page+=table(['组别','窗口','参数组','Loss','梯度范数均值','零范数比例','总梯度投影均值'],[
        [r['variant'],r['window'],r['group'],r['loss'],num(r['mean_norm']),pct(r['zero_fraction']),num(r['mean_projection'])]
        for r in norms if r['group'] in ('strength_head','shared') and r['window'] in ('train_last20','final_eval')])
    chosen={'mask__direction','mask__preserve','mask__strength_activate','mask__strength_suppress','direction__strength_activate','preserve__strength_activate','strength_activate__strength_suppress'}
    page+='<h2>参数梯度冲突</h2><p>零范数或未启用损失的余弦不可定义，排除冲突率分母；不能视为0冲突。完整CSV包含所有训练、末20epoch、最终评估及全部参数组/损失对。</p>'
    page+=table(['组别','窗口','参数组','Loss对','有效/观察数','余弦均值','冲突率'],[
        [r['variant'],r['window'],r['group'],r['pair'],f'{r["valid_cosines"]}/{r["observations"]}',num(r['mean_cosine']),pct(r['conflict_fraction'])]
        for r in pairs if r['group'] in ('strength_head','shared') and r['window']=='train_last20' and r['pair'] in chosen])
    page+='<h2>监督覆盖与核验</h2>'+table(['组别','点击','事件','非空目标','激活像素次数','抑制像素次数'],[
        [r['variant'],'正' if r['label'] else '负',r['events'],r['nonempty_target'],r['activation_pixels'],r['suppression_pixels']] for r in coverage])
    page+='<p>三组初始化SHA256及16实例清单一致；各800更新。6个最终/最佳权重重新推理回放共480轮，IoU和原图编辑计数精确一致；独立核对辅助损失、监督像素数和有害区域排除所有未修复错误。原模型源码及历史权重哈希不变。comparison.csv、strength_distributions.csv、gradient_norms.csv、gradient_pairs.csv为主要表格；native/保存最终权重所有编辑轮的原始P、S、z、D、GT和区域。各组目录分别保留best.pt、last.pt和全训练日志。</p></html>'
    (out/'report.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4)
    report(args)
