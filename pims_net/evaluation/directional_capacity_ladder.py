"""Three-capacity train16 comparison, with independent checkpoint replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluate_directional_constraints import transition_counts, transition_rates
from pims_net.evaluate_eta import state_metrics
from pims_net.evaluation.directional_capacity_report import read_lines
from pims_net.evaluation.directional_overfit_report import table, write_csv
from pims_net.evaluation.experiment import file_identity
from pims_net.models.directional_large import LargeDirectionalPIMSNet
from pims_net.models.directional_pims_net import DirectionalPIMSNet
from pims_net.models.directional_wide import WideDirectionalPIMSNet
from pims_net.overfit_directional import dump
from pims_net.paths import OUTPUT_ROOT
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.interactive_losses import prepare_target


@torch.no_grad()
def replay(root, label, factory, checkpoint_name, device, records):
    model=factory().to(device).eval().requires_grad_(False)
    info=load_training_checkpoint(root/checkpoint_name,model)
    dataset=InteractiveDataset(root/'fixed_train16.jsonl')
    selected=[r for r in records if r['epoch']==info['epoch']]
    metrics_rows=[]
    max_error=0.
    for index in range(16):
        sample=dataset[index]
        target,valid_t=prepare_target(sample['mask'],'cpu')
        truth,valid=target[0,0].numpy().astype(bool),valid_t[0,0].numpy().astype(bool)
        steps=sorted([r for r in selected if r['index']==index],key=lambda r:r['round'])
        for step,r in enumerate(steps):
            points,labels=[[[r['x'],r['y']]]],[[r['label']]]
            state=model.initialize(sample['image'],points,labels) if step==0 else model.refine(state,points,labels)
            metrics,prediction,_=state_metrics(model,state,sample['mask'],truth,valid)
            if step==0:
                first_mask=prediction
            error=abs(metrics['iou']-r['iou'])
            max_error=max(max_error,error)
            assert error==0
            counts=transition_counts(first_mask,prediction,sample['mask'])
            metrics_rows.append(dict(model=label,checkpoint=checkpoint_name,epoch=info['epoch'],index=index,
                id=sample['id'],round=step+1,x=r['x'],y=r['y'],label=r['label'],**metrics,**counts,**transition_rates(counts)))
    counts=dict(decoder=sum(p.numel() for p in model.directional_decoder.parameters()),
                research=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('sam_initializer.')))
    verification=dict(model=label,checkpoint=checkpoint_name,epoch=info['epoch'],updates=info['global_step'],
                      replayed_rounds=len(metrics_rows),max_iou_error=max_error,parameters=counts)
    return verification,metrics_rows


def report(args):
    out=Path(args.directory).resolve()
    roots=dict(original=Path(args.original).resolve(),wide2_res2=Path(args.wide).resolve(),wide4_res4=out)
    factories=dict(original=DirectionalPIMSNet,wide2_res2=WideDirectionalPIMSNet,wide4_res4=LargeDirectionalPIMSNet)
    configs={name:json.loads((root/'config.json').read_text()) for name,root in roots.items()}
    original=configs['original']
    curves,rounds,sequences={}, {}, {}
    comparison,round_table,distributions=[],[],[]
    for name,root in roots.items():
        status=json.loads((root/'status.json').read_text())
        assert status['status']=='completed' and status['epochs']==100 and status['optimizer_updates']==800
        assert (root/'fixed_train16.jsonl').read_bytes()==(roots['original']/'fixed_train16.jsonl').read_bytes()
        for key in ('seed','epochs','batch_size','max_clicks','lr','weight_decay','grad_clip','threshold','loss_weights','eval_every'):
            assert configs[name][key]==original[key],key
        curves[name]=[r for r in read_lines(root/'epochs.jsonl') if r['phase']=='eval']
        rounds[name]=[r for r in read_lines(root/'rounds.jsonl') if r['phase']=='eval']
        sequences[name]=[r for r in read_lines(root/'sequences.jsonl') if r['phase']=='eval']
        for epoch in curves[name]:
            selected=[r for r in sequences[name] if r['epoch']==epoch['epoch']]
            assert len(selected)==16 and len({r['id'] for r in selected})==16
            for key in ('IoU@1','IoU@3','IoU@5','native_IoU5'):
                assert abs(epoch[key]-np.mean([r[key] for r in selected]))<1e-14
            for key in epoch['losses']:
                assert abs(epoch['losses'][key]-np.mean([r['losses'][key] for r in selected]))<1e-14
        for stage,row in (('initial',curves[name][0]),('best',max(curves[name],key=lambda r:r['IoU@5'])),('final',curves[name][-1])):
            comparison.append(dict(model=name,checkpoint=stage,epoch=row['epoch'],
                **{k:row[k] for k in ('IoU@1','IoU@3','IoU@5','native_IoU5','parameter_gradient_cosine_mean','parameter_gradient_conflict_fraction')},
                **{'loss_'+k:v for k,v in row['losses'].items()}))
        for step in range(1,6):
            selected=[r for r in rounds[name] if r['epoch']==100 and r['round']==step]
            round_table.append(dict(model=name,round=step,objects=len(selected),iou=float(np.mean([r['iou'] for r in selected]))))
            if step>1:
                for region in ('all','TP','TN','FP','FN'):
                    data=[r['distributions'][region] for r in selected if r['distributions'][region]['direction'] is not None]
                    total=sum(r['direction']['count'] for r in data)
                    if total:
                        distributions.append(dict(model=name,round=step,region=region,pixels=total,
                            **{field:sum(r[field]['mean']*r[field]['count'] for r in data)/total for field in ('direction','strength')}))
    firsts={name:[(r['index'],r['x'],r['y'],r['label'],r['iou']) for r in rows if r['epoch']==100 and r['round']==1] for name,rows in rounds.items()}
    assert firsts['original']==firsts['wide2_res2']==firsts['wide4_res4']
    verifications,metrics_rows=[],[]
    for name,root in roots.items():
        verified,metrics=replay(root,name,factories[name],'last.pt',args.device,rounds[name])
        verifications.append(verified)
        metrics_rows.extend(metrics)
    verified,best_metrics=replay(out,'wide4_res4',LargeDirectionalPIMSNet,'best.pt',args.device,rounds['wide4_res4'])
    verifications.append(verified)
    with (out/'checkpoint_metrics.jsonl').open('w',encoding='utf-8') as stream:
        for row in metrics_rows+best_metrics:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    repair_rows=[]
    count_keys=('fn_before','fn_repaired','fp_before','fp_removed','tp_before','tp_damaged','tn_before','tn_damaged')
    for name in roots:
        selected=[r for r in metrics_rows if r['model']==name and r['round']==5]
        counts={k:sum(r[k] for r in selected) for k in count_keys}
        repair_rows.append(dict(model=name,**counts,**transition_rates(counts),
                               precision=float(np.mean([r['precision'] for r in selected])),recall=float(np.mean([r['recall'] for r in selected]))))
    pairs=[]
    for index in range(16):
        rows={name:next(r for r in sequences[name] if r['epoch']==100 and r['index']==index) for name in roots}
        assert len({r['id'] for r in rows.values()})==1
        pairs.append(dict(index=index,id=rows['original']['id'],**{name:r['IoU@5'] for name,r in rows.items()},
                          delta_vs_original=rows['wide4_res4']['IoU@5']-rows['original']['IoU@5'],
                          delta_vs_wide2=rows['wide4_res4']['IoU@5']-rows['wide2_res2']['IoU@5']))
    for name,rows in (('capacity_comparison',comparison),('capacity_rounds',round_table),('capacity_distributions',distributions),('capacity_pairs',pairs),('repair_comparison',repair_rows)):
        dump(out/(name+'.json'),rows)
        write_csv(out/(name+'.csv'),rows)
    for path,sha in json.loads((out/'verification.json').read_text())['original_hashes'].items():
        assert file_identity(path)['sha256']==sha,path
    verification=dict(passed=True,unchanged_models_and_weights=True,identical_data_and_hyperparameters=True,
        same_first_clicks=True,updates_each=800,means_recomputed=True,replays=verifications,
        better_than_wide2=sum(r['delta_vs_wide2']>0 for r in pairs),worse_than_wide2=sum(r['delta_vs_wide2']<0 for r in pairs),
        equal_to_wide2=sum(r['delta_vs_wide2']==0 for r in pairs))
    dump(out/'capacity_verification.json',verification)
    render(out,curves,comparison,repair_rows,round_table,distributions,verification)
    print(json.dumps(dict(final=[r for r in comparison if r['checkpoint']=='final'],verification=verification,repair=repair_rows),indent=2))


def render(out,curves,comparison,repairs,rounds,distributions,verification):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name,rows in curves.items():
        x=[r['epoch'] for r in rows]
        for ax,values,title in zip(axes.flat,([100*r['IoU@5'] for r in rows],[r['losses']['loss'] for r in rows],
            [r['losses']['dice'] for r in rows],[r['parameter_gradient_cosine_mean'] for r in rows]),
            ('Train16 IoU@5 (%)','Total loss','Dice loss','Mask vs Direction parameter cosine')):
            ax.plot(x,values,'o-',markersize=3,label=name)
            ax.set(title=title,xlabel='Epoch')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.savefig(out/'capacity_comparison.png',dpi=160)
    plt.close(fig)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Directional容量递增实验</title><style>body{font:16px/1.7 system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}img{max-width:100%}</style><h1>DirectionalPIMSNet：三级Decoder容量对比</h1>'
    page+='<p>原版64/32/16、无残差块；第一扩容版128/64/32、2个64通道残差块；本次256/128/64、4个128通道残差块。残差块位于64×64网格，均为x+Conv3×3(GELU(Conv3×3(x)))。本次仅新增独立模型和权重，前两版及A3不变。</p>'
    page+='<p>同一16训练实例、seed42、100epoch/800更新，batch2、lr=1e-4、weight decay=1e-4、clip1。无热启动，非Decoder初始参数完全一致。冻结SAM首击与ConvGRU结构不变；损失仍BCE+Dice+Direction+0.1×Preserve，阈值P&gt;0.5；不叠加C1–C3推理约束。各组按同一RITM协议自适应点击，后续坐标可不同。本表使用固定权重eval，预算按更新次数相同，计算量不相同。</p>'
    page+=table(['模型','Decoder参数','全部研究参数'],[[r['model'],f'{r["parameters"]["decoder"]:,}',f'{r["parameters"]["research"]:,}'] for r in verification['replays'] if r['checkpoint']=='last.pt'])
    page+=table(['模型','检查点','epoch','IoU@1','IoU@3','IoU@5','总Loss','BCE','Dice','Direction','Preserve(未加权)'],[
        [r['model'],r['checkpoint'],r['epoch'],*[f'{100*r[k]:.2f}%' for k in ('IoU@1','IoU@3','IoU@5')],*[f'{r["loss_"+k]:.4f}' for k in ('loss','bce','dice','direction','preserve')]] for r in comparison])
    finals={r['model']:r for r in comparison if r['checkpoint']=='final'}
    new=finals['wide4_res4']
    page+=f'<p>同预算最终IoU@5相比前一扩容版变化{100*(new["IoU@5"]-finals["wide2_res2"]["IoU@5"]):+.2f}个百分点；16对象改善{verification["better_than_wide2"]}、退化{verification["worse_than_wide2"]}、持平{verification["equal_to_wide2"]}。第100epoch训练集平均IoU@5为{100*new["IoU@5"]:.2f}%，应与预设95%拟合目标区分。</p><img src="capacity_comparison.png">'
    page+='<h2>最终五击编辑效果</h2><p>以下回载各自最终权重、回放其最终评估点击；编辑率比较首击与第五击，按原图像素汇总，TP分母仅包含首击正确前景，IoU为对象宏平均。不同模型后续点击不同，属于同协议比较。</p>'
    page+=table(['模型','FN修复','FP删除','TP破坏','TN破坏','Precision','Recall'],[[r['model'],*[f'{100*r[k]:.3f}%' for k in ('fn_repair_rate','fp_removal_rate','tp_damage_rate','tn_damage_rate','precision','recall')]] for r in repairs])
    page+='<h2>最终逐轮IoU</h2>'+table(['模型','轮次','实际对象数','IoU'],[[r['model'],r['round'],r['objects'],f'{100*r["iou"]:.2f}%'] for r in rounds])
    page+='<h2>第五击FN/FP方向与强度</h2>'+table(['模型','区域','Direction均值','Strength均值'],[[r['model'],r['region'],f'{r["direction"]:+.4f}',f'{r["strength"]:.4f}'] for r in distributions if r['round']==5 and r['region'] in ('FN','FP')])
    page+='<p>三组最终权重及本次best均回放核验；IoU误差为零，旧权重与模型源码哈希不变。完整日志保留各轮IoU、所有损失及输出/参数梯度余弦。本实验仅说明单种子训练16实例的拟合能力，不代表泛化或跨种子稳定性。通道与残差深度同时增加，不能单独分离两者贡献。</p></html>'
    (out/'report.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',default=str(OUTPUT_ROOT/'directional_overfit16_seed42_20261019'))
    parser.add_argument('--wide',default=str(OUTPUT_ROOT/'directional_wide_overfit16_seed42_20261019'))
    parser.add_argument('--directory',required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4)
    report(args)
