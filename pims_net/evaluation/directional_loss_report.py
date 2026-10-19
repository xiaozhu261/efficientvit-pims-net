"""Verified train16 loss ablation report; no model changes or GT inference gate."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluate_directional_constraints import transition_counts, transition_rates
from pims_net.evaluate_eta import state_metrics, write_csv
from pims_net.evaluation.directional_overfit_report import table
from pims_net.evaluation.experiment import file_identity
from pims_net.models.directional_large import LargeDirectionalPIMSNet
from pims_net.overfit_directional import dump
from pims_net.overfit_directional_losses import PAIRS
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.interactive_losses import prepare_target


def lines(path, predicate=lambda r: True):
    with path.open(encoding='utf-8') as stream:
        return [r for line in stream if predicate(r := json.loads(line))]


def pooled_metrics(rows, count_key='cumulative_counts'):
    counts = {key: sum(r[count_key][key] for r in rows) for key in rows[0][count_key]}
    return dict(objects=len(rows), iou=float(np.mean([r['iou'] for r in rows])),
                native_iou=float(np.mean([r['native_iou'] for r in rows])), **counts, **transition_rates(counts))


def pair_summary(values):
    present=[v for v in values if v is not None and v['cosine'] is not None]
    return dict(observations=len(values), valid_cosines=len(present),
        mean_cosine=float(np.mean([v['cosine'] for v in present])) if present else None,
        conflict_fraction=float(np.mean([v['cosine']<0 for v in present])) if present else None,
        left_norm_mean=float(np.mean([v['left_norm'] for v in present])) if present else None,
        right_norm_mean=float(np.mean([v['right_norm'] for v in present])) if present else None)


@torch.no_grad()
def replay(root, variant, checkpoint, records, device):
    model=LargeDirectionalPIMSNet().to(device).eval().requires_grad_(False)
    info=load_training_checkpoint(root/checkpoint, model)
    assert info['global_step']==info['epoch']*8
    dataset=InteractiveDataset(root/'fixed_train16.jsonl')
    selected=[r for r in records if r['epoch']==info['epoch']]
    replay_rows=[]
    error=0.
    for index in range(16):
        sample=dataset[index]
        target,valid=prepare_target(sample['mask'],device)
        truth,valid_np=target[0,0].cpu().numpy().astype(bool),valid[0,0].cpu().numpy().astype(bool)
        steps=sorted([r for r in selected if r['index']==index],key=lambda r:r['round'])
        first=None
        for saved in steps:
            points,labels=[[[saved['x'],saved['y']]]],[[saved['label']]]
            state=model.initialize(sample['image'],points,labels) if saved['round']==1 else model.refine(state,points,labels)
            metrics,prediction,_=state_metrics(model,state,sample['mask'],truth,valid_np)
            error=max(error,abs(metrics['iou']-saved['iou']))
            assert error==0, (variant,checkpoint,index,saved['round'])
            if first is None:
                first=prediction.copy()
            counts=transition_counts(first,prediction,sample['mask'])
            assert counts==saved['cumulative_counts']
            replay_rows.append(dict(variant=variant,checkpoint=checkpoint,epoch=info['epoch'],index=index,
                round=saved['round'],**metrics,**counts,**transition_rates(counts)))
    assert all(p.grad is None for p in model.parameters())
    return dict(variant=variant,checkpoint=checkpoint,epoch=info['epoch'],updates=info['global_step'],
                replayed_rounds=len(replay_rows),max_iou_error=error,transition_counts_exact=True),replay_rows


def report(args):
    out=Path(args.directory).resolve()
    variants=('L0','L1','L2')
    configs={v:json.loads((out/v/'config.json').read_text()) for v in variants}
    assert len({c['initial_research_sha256'] for c in configs.values()})==1
    assert len({(out/v/'fixed_train16.jsonl').read_bytes() for v in variants})==1
    for v,c in configs.items():
        status=json.loads((out/v/'status.json').read_text())
        assert status['status']=='completed' and status['optimizer_updates']==800 and status['epochs']==100
        assert c['parameter_counts']==dict(decoder=1648450,trainable=2220386)
        assert (c['seed'],c['epochs'],c['lr'],c['weight_decay'],c['batch_size'],c['threshold'])==(42,100,1e-4,1e-4,2,.5)
    curves,eval_rounds,summary,round_rows,gradient_rows,distributions={}, {}, [], [], [], []
    epoch_rows=[]
    component_coverage=[]
    collapse=[]
    first_clicks={}
    for variant in variants:
        root=out/variant
        epochs=lines(root/'epochs.jsonl')
        curves[variant]=[r for r in epochs if r['phase']=='eval']
        sequences=lines(root/'sequences.jsonl')
        rounds=lines(root/'rounds.jsonl',lambda r:r['phase']=='eval')
        eval_rounds[variant]=rounds
        assert len(epochs)==121 and len(sequences)==1936
        for epoch in epochs:
            selected=[r for r in sequences if r['epoch']==epoch['epoch'] and r['phase']==epoch['phase']]
            assert len(selected)==16
            for key in ('IoU@1','IoU@3','IoU@5','native_IoU5'):
                assert abs(epoch[key]-np.mean([r[key] for r in selected]))<1e-12
            for name,_,_ in PAIRS:
                values=[r['parameter_gradient_pairs']['all'][name] for r in selected if r['parameter_gradient_pairs']]
                calc=pair_summary(values)
                assert calc['valid_cosines']==epoch['gradient_pairs'][name]['valid_cosines']
                assert calc['mean_cosine']==epoch['gradient_pairs'][name]['mean_cosine']
                assert calc['conflict_fraction']==epoch['gradient_pairs'][name]['conflict_fraction']
        first_clicks[variant]=[(r['index'],r['x'],r['y'],r['label'],r['iou']) for r in rounds if r['epoch']==100 and r['round']==1]
        diagnostic=dict(variant=variant)
        for field in ('direction','strength'):
            data=[r['distributions']['all'][field] for r in rounds if r['epoch']==100 and r['round']==5]
            total=sum(r['count'] for r in data)
            diagnostic[field+'_pooled_mean']=sum(r['count']*r['mean'] for r in data)/total
            diagnostic[field+'_object_medians']=[r['quantiles'][3] for r in data]
            diagnostic[field+'_edge_histogram_fraction']=sum(
                (r['histogram'][0]+r['histogram'][-1]) if field=='direction' else r['histogram'][-1] for r in data)/total
        collapse.append(diagnostic)
        best=max(curves[variant],key=lambda r:r['IoU@5'])
        for stage,epoch in (('final',curves[variant][-1]),('best',best)):
            selected=[r for r in rounds if r['epoch']==epoch['epoch']]
            last=[r for r in selected if r['round']==5]
            assert len(last)==16
            summary.append(dict(variant=variant,stage=stage,epoch=epoch['epoch'],updates=epoch['optimizer_updates'],
                **{k:epoch[k] for k in ('IoU@1','IoU@3','IoU@5')},**pooled_metrics(last),
                losses=epoch['losses'],gradient_pairs=epoch['gradient_pairs']))
            for step in range(1,6):
                rows=[r for r in selected if r['round']==step]
                for scope,key in (('from_first','cumulative_counts'),('from_previous','incremental_counts')):
                    round_rows.append(dict(variant=variant,stage=stage,epoch=epoch['epoch'],round=step,scope=scope,
                                           **pooled_metrics(rows,key)))
        for epoch in curves[variant]:
            rows=[r for r in rounds if r['epoch']==epoch['epoch'] and r['round']==5]
            epoch_rows.append(dict(variant=variant,epoch=epoch['epoch'],**pooled_metrics(rows)))
        for window,selected in (
            ('train_all',[r for r in sequences if r['phase']=='train']),
            ('train_last20',[r for r in sequences if r['phase']=='train' and r['epoch']>80]),
            ('final_eval',[r for r in sequences if r['phase']=='eval' and r['epoch']==100])):
            for group in ('all','memory','head'):
                for name,_,_ in PAIRS:
                    values=[r['parameter_gradient_pairs'][group][name] for r in selected if r['parameter_gradient_pairs']]
                    gradient_rows.append(dict(variant=variant,window=window,level='parameter',group=group,pair=name,
                                              **pair_summary(values)))
        for step in range(2,6):
            selected=[r for r in rounds if r['epoch']==100 and r['round']==step]
            for label in (1,0):
                clicks=[r['clicked_component'] for r in selected if r['label']==label]
                if clicks:
                    component_coverage.append(dict(variant=variant,round=step,label=label,clicks=len(clicks),
                        nonempty=sum(c['selected_error'] for c in clicks),
                        mean_component_pixels=float(np.mean([c['selected_component_pixels'] for c in clicks]))))
            for name,_,_ in PAIRS:
                gradient_rows.append(dict(variant=variant,window='final_eval',level='output',group=f'round{step}',pair=name,
                    **pair_summary([r['output_gradient_pairs'][name] for r in selected])))
            for region in ('FN','FP','TP','TN'):
                for field in ('direction','strength'):
                    data=[r['distributions'][region][field] for r in selected if r['distributions'][region][field]]
                    count=sum(r['count'] for r in data)
                    distributions.append(dict(variant=variant,round=step,region=region,field=field,count=count,
                        mean=sum(r['mean']*r['count'] for r in data)/count if count else None))
    assert first_clicks['L0']==first_clicks['L1']==first_clicks['L2']
    # Re-run every saved best/final checkpoint using its own recorded clicks.
    replays,replay_rows=[],[]
    for variant in variants:
        for checkpoint in ('last.pt','best.pt'):
            verification,rows=replay(out/variant,variant,checkpoint,eval_rounds[variant],args.device)
            replays.append(verification)
            replay_rows.extend(rows)
            print(json.dumps(verification),flush=True)
    unchanged={}
    for variant in variants:
        for path,sha in json.loads((out/variant/'verification.json').read_text())['original_hashes'].items():
            unchanged[path]=sha
    for path,sha in unchanged.items():
        assert file_identity(path)['sha256']==sha,path
    old_root=Path(configs['L0']['experiment']['reference'])
    old_epochs=lines(old_root/'epochs.jsonl',lambda r:r['phase']=='eval')
    l0_reproduction=max(abs(a[k]-b[k]) for a,b in zip(curves['L0'],old_epochs) for k in ('IoU@1','IoU@3','IoU@5'))
    verification=dict(same_initial_parameters=True,initial_research_sha256=configs['L0']['initial_research_sha256'],
        same_instances=True,same_first_clicks=True,updates_each=800,architecture_unchanged=True,
        original_files_and_weights_unchanged=True,epoch_means_and_gradients_recomputed=True,
        legacy_L0_max_iou_difference=l0_reproduction,
        legacy_L0_final_iou5_difference=curves['L0'][-1]['IoU@5']-old_epochs[-1]['IoU@5'],replays=replays,
        checkpoint_hashes={f'{v}/{f}':file_identity(out/v/f)['sha256'] for v in variants for f in ('last.pt','best.pt')})
    for name,rows in (('comparison',summary),('round_comparison',round_rows),('gradients',gradient_rows),
                      ('epoch_metrics',epoch_rows),('distributions',distributions),('checkpoint_replay',replay_rows),
                      ('component_coverage',component_coverage)):
        dump(out/(name+'.json'),rows)
        if name=='comparison':
            flat=[{k:v for k,v in r.items() if k not in ('losses','gradient_pairs')} for r in rows]
            write_csv(out/(name+'.csv'),flat)
        else:
            write_csv(out/(name+'.csv'),rows)
    dump(out/'collapse_diagnostics.json',collapse)
    dump(out/'verification.json',verification)
    render(out,curves,summary,round_rows,gradient_rows,epoch_rows,verification,component_coverage,collapse)
    print(json.dumps(summary,indent=2),flush=True)


def render(out,curves,summary,rounds,gradients,epochs,verification,component_coverage,collapse):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for variant,rows in curves.items():
        data=[r for r in epochs if r['variant']==variant]
        for ax,values,title in zip(axes.flat,([r['IoU@5'] for r in rows],[r['fn_repair_rate'] for r in data],
            [r['tp_damage_rate'] for r in data],[r['gradient_pairs']['mask_preserve']['conflict_fraction'] for r in rows]),
            ('Train16 IoU@5','FN repaired vs first mask','TP damaged vs first mask','Mask-Preserve parameter conflict')):
            ax.plot([r['epoch'] for r in rows],[None if v is None else 100*v for v in values],label=variant)
            ax.set(title=title,xlabel='Epoch',ylabel='%')
            ax.grid(alpha=.2)
            ax.legend()
    fig.savefig(out/'loss_comparison.png',dpi=160)
    plt.close(fig)
    def pct(value):
        return '—' if value is None else f'{100*value:.2f}%'
    def num(value):
        return '—' if value is None else f'{value:+.4f}'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>更大版Directional损失消融</title><style>body{font:16px/1.7 system-ui;margin:32px}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:8px}img{max-width:100%}</style><h1>更大版 DirectionalPIMSNet：L0/L1/L2 损失消融</h1>'
    page+='<p>固定同一16个训练实例、seed42、完全相同随机初始化、100epoch/800次AdamW更新，batch2、lr1e-4、weight decay1e-4、clip1；冻结SAM首击，五击RITM闭环，统一P&gt;0.5。架构及原有权重不变，无热启动。各模型自适应选择后续点击，坐标可能不同；推理完全使用原模型，不使用GT连通域限制更新。该实验仅反映单种子16实例拟合能力。</p>'
    page+=table(['组别','Mask监督','Direction监督','Preserve监督'],[
        ['L0','全有效域BCE+Dice','全域、FN/FP/正确三类平衡SmoothL1','GT正确区域'],
        ['L1','当前点击同符号GT错误8连通域内BCE+Dice','无','有效域中连通域以外全部像素'],
        ['L2','同L1','同一连通域内SmoothL1，正击+1/负击−1','同L1']])
    page+='<p>权重为BCE=1、Dice=1、Direction=1（L1禁用）、Preserve=0.1；各区域内取均值，Dice平滑项为1。旧Mask与连通域选择均停止梯度，模型状态仍全程BPTT。映射采用官方1024提示坐标/4取最近原生像素；若对应点已不是该符号错误，不吸附其他区域，局部监督为0，只保留域外Preserve。Preserve参考为旧概率的绝对差。不同损失覆盖区域不同，标量Loss大小不能直接用作模型优劣指标。</p>'
    page+='<h2>同预算最终权重（800更新）</h2>'
    def comparison_table(rows):
        return table(['组别','epoch','更新','IoU@1','IoU@3','IoU@5','FN修复','FP删除','TP破坏','TN破坏'],[
            [r['variant'],r['epoch'],r['updates'],*[pct(r[k]) for k in ('IoU@1','IoU@3','IoU@5','fn_repair_rate','fp_removal_rate','tp_damage_rate','tn_damage_rate')]] for r in rows])
    page+=comparison_table([r for r in summary if r['stage']=='final'])
    final={r['variant']:r for r in summary if r['stage']=='final'}
    page+='<p>'+''.join(f'{v}相对L0：IoU@5变化{100*(final[v]["IoU@5"]-final["L0"]["IoU@5"]):+.2f}个百分点，FN修复率变化{100*(final[v]["fn_repair_rate"]-final["L0"]["fn_repair_rate"]):+.2f}个百分点，TP破坏率变化{100*(final[v]["tp_damage_rate"]-final["L0"]["tp_damage_rate"]):+.2f}个百分点。' for v in ('L1','L2'))+'</p>'
    page+='<p>IoU为原图尺度对象宏平均；编辑率以首击为基准、汇总原图像素计数：FN修复/首击FN，TP破坏/首击TP（不混入TN）。每组相同首击及分母。另输出FP删除与TN破坏，防止单纯扩张Mask掩盖错误。</p>'
    page+='<h2>本次实验的主要发现</h2>'
    if all(final[v]['IoU@5']<final['L0']['IoU@5'] and final[v]['fn_repair_rate']>final['L0']['fn_repair_rate']
           and final[v]['tp_damage_rate']>final['L0']['tp_damage_rate'] for v in ('L1','L2')):
        page+='<p>L1/L2提升了FN修复，但同时大幅破坏正确前景并新增背景误检，整体IoU显著退化。</p>'
    best={r['variant']:r for r in summary if r['stage']=='best'}
    if best['L1']['epoch']==best['L2']['epoch']==0:
        page+='<p>L1/L2预算内最佳IoU@5均出现在epoch0，训练过程中没有超过初始水平。</p>'
    page+='<p>最终局部损失：'+''.join(f'{v} BCE/Dice={final[v]["losses"]["bce"]:.4f}/{final[v]["losses"]["dice"]:.4f}，域外Preserve={final[v]["losses"]["preserve"]:.4f}（未加权）；' for v in ('L1','L2'))+'局部目标的低Loss应结合整图编辑指标判断。</p>'
    page+='<p>最终第5击全有效域Strength均值：'+ '、'.join(f'{r["variant"]}={r["strength_pooled_mean"]:.4f}' for r in collapse)+'。Direction落入首尾直方图区间（约|D|≥0.9）的比例：'+'、'.join(f'{r["variant"]}={pct(r["direction_edge_histogram_fraction"])}' for r in collapse)+'。结合TP/TN破坏可检查编辑是否强烈且范围过大；分布细节见collapse_diagnostics.json及各组原始日志。这些观察不能独立证明唯一训练成因。</p>'
    late={(r['variant'],r['pair']):r for r in gradients if r['level']=='parameter' and r['group']=='all' and r['window']=='train_last20'}
    page+='<p>末20epoch全参数Mask–Direction冲突率：'+ '、'.join(f'{v}={pct(late[v,"mask_direction"]["conflict_fraction"])}' for v in ('L0','L2'))+'（L1无该损失）；Mask–Preserve冲突率：'+'、'.join(f'{v}={pct(late[v,"mask_preserve"]["conflict_fraction"])}' for v in ('L0','L1','L2'))+'。局部修复目标之间更一致，并不保证与区域保护一致；不能仅据某一对冲突比例降低判定训练改善。</p>'
    page+='<h2>预算内最佳检查点（按训练16实例IoU@5选择）</h2>'+comparison_table([r for r in summary if r['stage']=='best'])
    page+='<p>最佳检查点更新数可能少于800，单独列示；没有验证集或泛化结论。</p><img src="loss_comparison.png">'
    page+='<h2>最终逐击结果</h2>'+table(['组别','点击','IoU','相对首击FN修复','相对首击TP破坏'],[
        [r['variant'],r['round'],pct(r['iou']),pct(r['fn_repair_rate']),pct(r['tp_damage_rate'])]
        for r in rounds if r['stage']=='final' and r['scope']=='from_first'])
    page+='<h2>全参数梯度冲突</h2><p>每个对象使用4轮平均加权损失的完整BPTT梯度，统计clip/AdamW前的余弦；负余弦计为冲突。缺失Direction或零梯度的余弦为不可定义，排除分母，不能记成无冲突。局部Mask与域外Preserve在本轮输出上的直接支持互斥，但共享参数及时间路径上仍可能冲突；表中报告全参数结果，CSV另含memory/head和逐轮输出梯度。</p>'
    page+=table(['组别','窗口','损失对','有效/观察数','平均余弦','冲突率'],[
        [r['variant'],r['window'],r['pair'],f'{r["valid_cosines"]}/{r["observations"]}',num(r['mean_cosine']),pct(r['conflict_fraction'])]
        for r in gradients if r['level']=='parameter' and r['group']=='all'])
    page+='<h2>最终评估点击与局部监督区域</h2><p>这是评估轨迹上可构造的局部监督区域，推理不使用该区域门控。非空数低可能源于原图点击映射到原生网格后已正确；不吸附其他错误。</p>'+table(['组别','轮次','符号','非空/点击数','平均连通域像素'],[
        [r['variant'],r['round'],'正' if r['label'] else '负',f'{r["nonempty"]}/{r["clicks"]}',f'{r["mean_component_pixels"]:.1f}'] for r in component_coverage])
    page+='<h2>核验与文件</h2><p>三组初始可训练参数SHA256一致；每组800更新。6个最佳/最终检查点均重新推理回放，逐轮IoU及编辑计数精确一致；模型源码及旧权重哈希不变。本次L0相对旧800更新实验，评估曲线IoU最大差为'+f'{100*verification["legacy_L0_max_iou_difference"]:.3f}'+'个百分点，最终IoU@5差为'+f'{100*verification["legacy_L0_final_iou5_difference"]:+.3f}'+'个百分点；训练轨迹并非逐位复现，主表统一使用本次三组重训结果。</p>'
    page+='<p>comparison.csv/json为总表，round_comparison.csv含相对首击及相对上一击两种口径；gradients.csv含全训练、末20epoch与最终评估窗口；distributions.csv含最终逐击FN/FP/TP/TN的D/S分布均值。L0/L1/L2子目录分别保存逐轮/逐对象/逐epoch日志及best.pt、last.pt。</p></html>'
    (out/'report.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4)
    report(args)
