"""Epoch95 click-component editing diagnostics; actual state never intervened."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluate_eta import state_metrics, write_csv
from pims_net.evaluate_oracle import clicked_error_component
from pims_net.evaluation.directional_overfit_report import table
from pims_net.evaluation.experiment import file_identity
from pims_net.models.directional_large import LargeDirectionalPIMSNet
from pims_net.overfit_directional import dump
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT, checkpoint_path
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.interactive_losses import prepare_target


def describe(values):
    values=np.asarray(values,np.float64)
    if not values.size:
        return None
    assert np.isfinite(values).all()
    return dict(count=int(values.size),mean=float(values.mean()),std=float(values.std()),
                minimum=float(values.min()),q10=float(np.quantile(values,.1)),median=float(np.median(values)),
                q90=float(np.quantile(values,.9)),q99=float(np.quantile(values,.99)),maximum=float(values.max()))


def probability_update(p,d,s):
    return p+s*torch.where(d>=0,d*(1-p),d*p)


def counterfactuals(p,d,s,positive):
    """One-step alternatives, never fed into the actual model/state."""
    desired=1. if positive else -1.
    updates=dict(actual=probability_update(p,d,s),
        fix_sign=probability_update(p,desired*d.abs(),s),
        strength_one=probability_update(p,d,torch.ones_like(s)),
        unit_magnitude=probability_update(p,d.sign(),s),
        fix_sign_strength_one=probability_update(p,desired*d.abs(),torch.ones_like(s)),
        gt_unit_direction=probability_update(p,torch.full_like(d,desired),s))
    return {name:(value>.5 if positive else value<=.5) for name,value in updates.items()}


def classify_component(d,crossings,positive):
    aligned=d if positive else -d
    crossed=crossings['actual']
    remaining=~crossed
    categories=dict(crossed=crossed,
        wrong_or_zero_direction=remaining & (aligned<=0),
        strength_limited=remaining & (aligned>0) & crossings['strength_one'],
        direction_magnitude_limited=remaining & (aligned>0) & ~crossings['strength_one'])
    assert torch.all(sum(value.to(torch.int32) for value in categories.values())==1)
    return categories


def run(args):
    root,out=Path(args.reference).resolve(),Path(args.output).resolve()
    if out.exists() or not out.is_relative_to(OUTPUT_ROOT.resolve()) or out==OUTPUT_ROOT.resolve():
        raise ValueError('Use a new subdirectory under pims_net/outputs')
    checkpoint,manifest=root/'best.pt',root/'fixed_train16.jsonl'
    watched=[checkpoint,root/'last.pt',manifest,root/'rounds.jsonl',checkpoint_path()]
    watched+=list((RESEARCH_ROOT/'models').glob('*.py'))+list((RESEARCH_ROOT/'training').glob('*.py'))
    original=json.loads((OUTPUT_ROOT/'directional_implementation_20261018/original_hashes.json').read_text())
    for path,sha in original.items():
        assert file_identity(path)['sha256']==sha
    before={str(p):file_identity(p)['sha256'] for p in watched}
    before.update(original)
    torch.set_num_threads(4)
    model=LargeDirectionalPIMSNet().to(args.device).eval().requires_grad_(False)
    info=load_training_checkpoint(checkpoint,model)
    assert info['epoch']==95 and info['global_step']==760 and info['config']['threshold']==.5
    dataset=InteractiveDataset(manifest)
    assert len(dataset)==16
    records=[]
    with (root/'rounds.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['phase']=='eval' and r['epoch']==95:
                records.append(r)
    assert len(records)==80
    out.mkdir(parents=True)
    (out/'native').mkdir()
    dump(out/'protocol.json',dict(checkpoint=file_identity(checkpoint),epoch=95,optimizer_updates=760,
        manifest=file_identity(manifest),click_source=file_identity(root/'rounds.jsonl'),threshold=.5,
        clicks='Same epoch95 adaptive-evaluation recorded five clicks per training object. No click regeneration.',
        component='Before each actual refinement: current-click same-sign GT error 8-connected component on native256. Positive=FN, negative=FP. Empty stays empty; no snapping.',
        component_geometry=clicked_error_component.__doc__,
        distributions='Native valid pixels, per click event; pooled pixel occurrences and separate event macro. Regions: clicked component, radius8 click disk, other same-sign errors, opposite errors, GT-correct, all.',
        actual='Original model.refine only. Counterfactual arrays are immediate one-step diagnostics and never feed back.',
        counterfactuals=dict(fix_sign='click_sign*abs(D), S unchanged',strength_one='D unchanged, S=1',
            unit_magnitude='sign(D), S unchanged (D=0 stays0)',fix_sign_strength_one='click_sign*abs(D), S=1',
            gt_unit_direction='D=click_sign, S unchanged'),
        categories='Disjoint component categories: actual crossed; wrong/zero direction; correct sign not crossed but S=1 would cross; correct sign still not crossed at S=1 (D magnitude limits attainable update).',
        crossing='Positive FN requires P_new>0.5; negative FP requires P_new<=0.5. Exact float32 update is used.',
        spatial='Abs(deltaP) allocation plus density enrichment versus area. Updates outside component are partitioned into other errors and GT-correct; outside alone is not localization failure with cumulative clicks.',
        radius_native=8,radial_bins=[0,8,16,32,'infinity'],trained=False,model_unchanged=True,device=args.device))
    del info
    pools=defaultdict(list)
    events,metrics_rows,fixed_clicks=[],[],[]
    error=0.
    with torch.no_grad(),(out/'events.jsonl').open('w',encoding='utf-8') as stream:
        for index in range(16):
            sample=dataset[index]
            saved=sorted([r for r in records if r['index']==index],key=lambda r:r['round'])
            assert [r['round'] for r in saved]==list(range(1,6)) and all(r['id']==sample['id'] for r in saved)
            clicks=[{k:r[k] for k in ('x','y','label')} for r in saved]
            fixed_clicks.append(dict(index=index,id=sample['id'],clicks=clicks))
            target,valid_t=prepare_target(sample['mask'],args.device)
            truth,valid=target[0,0].cpu().numpy().astype(bool),valid_t[0,0].cpu().numpy().astype(bool)
            for step,click in enumerate(clicks,1):
                points,labels=[[[click['x'],click['y']]]],[[click['label']]]
                if step==1:
                    state=model.initialize(sample['image'],points,labels)
                else:
                    previous=state
                    old=previous.probabilities[0,0].cpu().numpy()
                    component,cc=clicked_error_component(old,truth,valid,click,sample['mask'].shape)
                    state=model.refine(previous,points,labels)
                    p,d,s=[v[0,0] for v in (previous.probabilities,state.direction,state.strength)]
                    expected=probability_update(p,d,s)
                    assert torch.equal(expected,state.probabilities[0,0])
                    new=state.probabilities[0,0].cpu().numpy()
                    delta=new-old
                    positive=bool(click['label'])
                    crossed=counterfactuals(p,d,s,positive)
                    classes=classify_component(d,crossed,positive)
                    yy,xx=np.mgrid[:256,:256]
                    distance=np.hypot(xx-cc['click_native_x'],yy-cc['click_native_y'])
                    same=(truth & (old<=.5)) if positive else (~truth & (old>.5))
                    opposite=(~truth & (old>.5)) if positive else (truth & (old<=.5))
                    correct=(old>.5)==truth
                    regions=dict(component=component,click_disk=(distance<=8)&valid,
                        other_same_errors=same&valid&~component,opposite_errors=opposite&valid,correct=correct&valid,all=valid)
                    for name,lo,hi in (('component_r0_8',0,8),('component_r8_16',8,16),('component_r16_32',16,32),('component_r32_plus',32,np.inf)):
                        regions[name]=component&(distance>=lo)&(distance<hi)
                    direction,strength=d.cpu().numpy(),s.cpu().numpy()
                    fields=dict(direction=direction,strength=strength,delta_p=delta,absolute_delta=np.abs(delta),
                        old_p=old,new_p=new,effective_m=strength*np.abs(direction),
                        required_m=(.5-old)/(1-old).clip(1e-30) if positive else (old-.5)/old.clip(1e-30))
                    fields['direction_aligned']=(direction>0 if positive else direction<0).astype(np.float32)
                    fields['direction_near_zero']=(np.abs(direction)<=.05).astype(np.float32)
                    fields['changed_ge_001']=(np.abs(delta)>=.01).astype(np.float32)
                    distributions={}
                    for region,selection in regions.items():
                        distributions[region]={}
                        for field,array in fields.items():
                            if field=='required_m' and not region.startswith('component'):
                                continue
                            values=array[selection]
                            distributions[region][field]=describe(values)
                            if values.size:
                                pools[click['label'],region,field].append(values.copy())
                    gate=torch.as_tensor(component,device=args.device)
                    counts={name:int(value[gate].sum()) for name,value in classes.items()}
                    cf={name:int(value[gate].sum()) for name,value in crossed.items()}
                    count=int(component.sum())
                    assert sum(counts.values())==count
                    mass={region:float(np.abs(delta)[selection].astype(np.float64).sum()) for region,selection in regions.items()}
                    area_fraction=count/int(valid.sum())
                    mass_fraction=mass['component']/mass['all'] if mass['all'] else None
                    file=f'object_{index:02d}_round_{step}.npz'
                    np.savez_compressed(out/'native'/file,previous=old,probabilities=new,direction=direction,strength=strength,
                                        truth=truth,valid=valid,component=component,click=np.array([cc['click_native_x'],cc['click_native_y'],click['label']]))
                    event=dict(index=index,id=sample['id'],round=step,**click,**cc,component_pixels=count,
                        classes=counts,crossings=cf,distributions=distributions,
                        area_fraction=area_fraction,absolute_update_mass=mass,component_update_mass_fraction=mass_fraction,
                        localization_enrichment=mass_fraction/area_fraction if mass_fraction is not None and area_fraction else None,
                        correct_pixels_damaged=int((correct&valid&((new>.5)!=truth)).sum()),
                        component_crossing_rate=cf['actual']/count if count else None,native_file='native/'+file)
                    events.append(event)
                    stream.write(json.dumps(event,allow_nan=False)+'\n')
                metrics,_,_=state_metrics(model,state,sample['mask'],truth,valid)
                error=max(error,abs(metrics['iou']-saved[step-1]['iou']))
                assert error==0 and all(param.grad is None for param in model.parameters())
                metrics_rows.append(dict(index=index,id=sample['id'],round=step,**click,**metrics))
            print(f'Epoch95 click diagnostic {index+1}/16',flush=True)
    stats=[]
    for (label,region,field),arrays in sorted(pools.items()):
        stats.append(dict(label=label,region=region,field=field,**describe(np.concatenate(arrays))))
    summary=[]
    for label in (1,0):
        for step in (0,2,3,4,5):
            selected=[r for r in events if r['label']==label and (step==0 or r['round']==step)]
            if not selected:
                continue
            count=sum(r['component_pixels'] for r in selected)
            actual=sum(r['crossings']['actual'] for r in selected)
            nonempty=[r for r in selected if r['component_pixels']]
            all_mass=sum(r['absolute_update_mass']['all'] for r in selected)
            correct_mass=sum(r['absolute_update_mass']['correct'] for r in selected)
            inside_mass=sum(r['absolute_update_mass']['component'] for r in selected)
            valid_count=sum(r['distributions']['all']['direction']['count'] for r in selected)
            area_fraction=count/valid_count
            mass_fraction=inside_mass/all_mass if all_mass else None
            summary.append(dict(label=label,round=step,events=len(selected),nonempty_components=len(nonempty),
                empty_components=len(selected)-len(nonempty),component_pixel_occurrences=count,
                crossed_pixels=actual,crossing_rate=actual/count if count else None,
                event_macro_crossing_rate=float(np.mean([r['component_crossing_rate'] for r in nonempty])) if nonempty else None,
                **{name+'_count':sum(r['classes'][name] for r in selected) for name in selected[0]['classes']},
                **{name+'_rate':sum(r['crossings'][name] for r in selected)/count if count else None for name in selected[0]['crossings']},
                component_area_fraction=area_fraction,component_mass_fraction=mass_fraction,
                component_density_enrichment=mass_fraction/area_fraction if mass_fraction is not None and area_fraction else None,
                correct_region_mass_fraction=correct_mass/all_mass if all_mass else None,
                native_correct_pixels_damaged=sum(r['correct_pixels_damaged'] for r in selected)))
    assert len(events)==64 and len(metrics_rows)==80
    assert before=={path:file_identity(path)['sha256'] for path in before}
    checks=dict(epoch=95,optimizer_updates=760,objects=16,replayed_rounds=80,edit_events=64,
        iou_max_error=error,original_model_and_weights_unchanged=True,counterfactuals_not_fed_back=True,
        formula_bit_exact=True,no_gradients=True,hashes=before)
    for name,value in (('summary',summary),('region_distributions',stats),('fixed_clicks',fixed_clicks),('verification',checks)):
        dump(out/(name+'.json'),value)
    write_csv(out/'summary.csv',summary)
    write_csv(out/'region_distributions.csv',stats)
    write_csv(out/'round_metrics.csv',metrics_rows)
    render(out,events,summary,stats)
    print(json.dumps([r for r in summary if r['round']==0],indent=2))


def render(out,events,summary,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pooled=[r for r in summary if r['round']==0]
    def value(label,region,field,key='mean'):
        row=next((r for r in stats if r['label']==label and r['region']==region and r['field']==field),None)
        return row[key] if row else None
    def fmt(x,percent=False):
        return '—' if x is None else (f'{100*x:.3f}%' if percent else f'{x:.5f}')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Epoch95正负点击编辑诊断</title><style>body{font:16px/1.7 system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}img{max-width:100%}</style><h1>更大版DirectionalPIMSNet epoch95：点击错误连通域诊断</h1>'
    page+='<p>加载best.pt（epoch95/760更新），回放该检查点16实例的原五击记录；80轮IoU精确一致。统计第2–5击64次编辑，阈值P&gt;0.5。所有数据来自原模型前向，不训练、不修改模型或权重；单轮反事实只在旁路计算，不反馈状态。</p>'
    page+='<p>正击定位当前点击所在FN的8连通域，负击定位FP域。点击按官方1024变换/4后取最近原生像素；若映射后已正确，则记录空域，不吸附其他错误。指标按原生256网格有效像素计算，排除padding。汇总单位是“像素×点击事件”，同一像素可多轮出现；另报每个非空事件的宏平均跨阈值率。</p>'
    page+=table(['点击','编辑次数','非空/空域','域内像素次数','Direction均值','Strength均值','实际ΔP均值','实际跨0.5比例','事件宏平均'],[
        ['正(FN)' if r['label'] else '负(FP)',r['events'],f'{r["nonempty_components"]}/{r["empty_components"]}',r['component_pixel_occurrences'],
         fmt(value(r['label'],'component','direction')),fmt(value(r['label'],'component','strength')),fmt(value(r['label'],'component','delta_p')),
         fmt(r['crossing_rate'],True),fmt(r['event_macro_crossing_rate'],True)] for r in pooled])
    pos=next(r for r in pooled if r['label']==1)
    neg=next(r for r in pooled if r['label']==0)
    page+='<h2>诊断结论</h2>'
    page+=f'<p><b>正点击的主要单步瓶颈是Strength不足。</b>FN域内方向正确比例为{fmt(value(1,"component","direction_aligned"),True)}，但Strength中位数仅{value(1,"component","strength","median"):.3g}，实际跨阈值率为{fmt(pos["crossing_rate"],True)}。保持原Direction、旁路将Strength设为1，跨阈值率达到{fmt(pos["strength_one_rate"],True)}；只纠正方向符号且保留Strength，跨越率仅{fmt(pos["fix_sign_rate"],True)}。这些数值说明当前输出中的瓶颈，不证明其训练成因。</p>'
    page+=f'<p><b>空间表现是正点击附近更新被压低，尚不能归因为没有找到点击。</b>点击所在FN域半径8像素内，方向正确比例{fmt(value(1,"component_r0_8","direction_aligned"),True)}，Direction均值{fmt(value(1,"component_r0_8","direction"))}，但Strength均值仅{value(1,"component_r0_8","strength"):.3g}。整个FN域的更新密度为全图均值的{pos["component_density_enrichment"]:.2f}倍；远离点击的域内也有方向错误。累积点击和Strength门控使定位、幅度相互影响，不能仅凭域外更新量认定独立的空间定位失败。</p>'
    page+=f'<p><b>负点击明显更有效，但效果并非每个事件一致。</b>FP域像素合并跨越率为{fmt(neg["crossing_rate"],True)}，事件宏平均为{fmt(neg["event_macro_crossing_rate"],True)}；前者受大连通域影响更大。FP域更新密度为全图均值的{neg["component_density_enrichment"]:.2f}倍，表明负点击修改集中在目标错误区域。结论仅适用于当前16个训练实例及这组固定点击。</p>'
    page+='<h2>方向与幅度：互斥分类</h2><p>已跨阈值；方向错误/为0；方向正确但原更新未跨阈值、将S设为1可跨越（Strength不足）；方向正确但即使S=1仍不能跨越（Direction幅度也不足）。最后一类不代表Strength已经足够，两者可能同时不足。</p>'
    categories=('crossed','wrong_or_zero_direction','strength_limited','direction_magnitude_limited')
    page+=table(['点击','已跨越','方向错误/零','Strength限制','D幅度限制（S=1仍不足）'],[
        ['正(FN)' if r['label'] else '负(FP)',*[fmt(r[k+'_count']/r['component_pixel_occurrences'] if r['component_pixel_occurrences'] else None,True) for k in categories]] for r in pooled])
    page+='<h2>同一旧Mask上的单轮反事实跨阈值率</h2><p>纠正符号保留|D|与S；S=1保留原D；|D|=1保留原符号与S（D=0仍为0）；纠正符号且S=1仍保留|D|；GT单位方向令D为点击符号但保留S。均不是重新运行多轮模型的性能。</p>'
    page+=table(['点击','实际','纠正符号','S=1','|D|=1','纠正符号+S=1','GT单位方向、原S'],[
        ['正(FN)' if r['label'] else '负(FP)',*[fmt(r[k+'_rate'],True) for k in ('actual','fix_sign','strength_one','unit_magnitude','fix_sign_strength_one','gt_unit_direction')]] for r in pooled])
    page+='<h2>空间分布</h2><p>只用“编辑在域外”不能证明定位失败：模型接收累积点击，域外其他错误也可能值得修复。这里同时提供目标域更新覆盖、各区域更新密度、正确区域所占更新量，以及域内随距离的分布。</p>'
    page+=table(['点击','区域','D方向正确比例','|D|≤0.05','S均值','|ΔP|均值','|ΔP|≥0.01覆盖'],[
        ['正' if label else '负',region,fmt(value(label,region,'direction_aligned'),True),fmt(value(label,region,'direction_near_zero'),True),
         fmt(value(label,region,'strength')),fmt(value(label,region,'absolute_delta')),fmt(value(label,region,'changed_ge_001'),True)]
        for label in (1,0) for region in ('component','click_disk','other_same_errors','opposite_errors','correct','component_r0_8','component_r8_16','component_r16_32','component_r32_plus')])
    page+=table(['点击','目标域面积占比','目标域占总|ΔP|','更新密度/全图均值','正确区域占总|ΔP|','原生正确像素破坏次数'],[
        ['正' if r['label'] else '负',fmt(r['component_area_fraction'],True),fmt(r['component_mass_fraction'],True),fmt(r['component_density_enrichment']),fmt(r['correct_region_mass_fraction'],True),r['native_correct_pixels_damaged']] for r in pooled])
    for label in (1,0):
        eligible=[r for r in events if r['label']==label and r['component_pixels']]
        if not eligible:
            continue
        case=max(eligible,key=lambda r:r['component_pixels'])
        with np.load(out/case['native_file']) as arrays:
            fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
            for ax,key,title,low,high in zip(axes.flat,('previous','probabilities','direction','strength','delta','truth'),
                ('Previous P','New P','Direction','Strength','Actual delta P','GT + clicked component'),(0,0,-1,0,-.5,0),(1,1,1,1,.5,1)):
                arr=arrays['probabilities']-arrays['previous'] if key=='delta' else arrays[key]
                shown=np.ma.masked_where(~arrays['valid'],arr)
                im=ax.imshow(shown,cmap='coolwarm' if key in ('direction','delta') else 'viridis',vmin=low,vmax=high)
                ax.contour(arrays['component'],levels=[.5],colors='orange',linewidths=.7)
                ax.scatter([case['click_native_x']],[case['click_native_y']],c='red',s=20,marker='x')
                ax.set(title=title,xticks=[],yticks=[])
                fig.colorbar(im,ax=ax,shrink=.7)
            fig.suptitle(f'{"Positive FN" if label else "Negative FP"} largest component: object {case["index"]}, round {case["round"]}')
            filename=f'case_{"positive" if label else "negative"}.png'
            fig.savefig(out/filename,dpi=150)
            plt.close(fig)
        page+=f'<h2>{"正" if label else "负"}击最大错误连通域示例</h2><p>对象{case["index"]}，第{case["round"]}击；橙线为点击错误连通域，红叉为点击。</p><img src="{filename}">'
    page+='<p>events.jsonl保存每次点击的连通域选择、所有分布、反事实计数及空间更新量；native/保存完整P_old/P_new/D/S/GT/有效域/连通域以供复核。region_distributions.csv给出精确合并像素均值、标准差及分位数。该分析区分单步数值瓶颈与空间现象，不把它们当作互不相关的训练原因，也不代表泛化性能。</p></html>'
    (out/'report.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default=str(OUTPUT_ROOT/'directional_large_overfit16_seed42_20261019'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())
