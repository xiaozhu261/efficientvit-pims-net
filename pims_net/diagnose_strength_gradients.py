"""Read-only Strength saturation and loss-gradient attribution on saved clicks."""

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.diagnose_large_clicks import describe
from pims_net.evaluate_eta import state_metrics, write_csv
from pims_net.evaluate_oracle import clicked_error_component
from pims_net.evaluation.directional_overfit_report import table
from pims_net.evaluation.edit_loss_audit import model_digest
from pims_net.evaluation.experiment import file_identity
from pims_net.models.directional_large import LargeDirectionalPIMSNet
from pims_net.overfit_directional import dump
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.directional_losses import directional_losses
from pims_net.training.interactive_losses import prepare_target
from pims_net.training.local_directional_losses import local_directional_losses

TERMS=('bce','dice','direction','preserve')
WEIGHTS=dict(bce=1.,dice=1.,direction=1.,preserve=.1)


def gradients(loss, inputs):
    values=torch.autograd.grad(loss,inputs,retain_graph=True,allow_unused=True)
    return [torch.zeros_like(t) if g is None else g for t,g in zip(inputs,values)]


def vector_summary(vectors):
    """Per-loss magnitude, signed total projection and all pairwise cosines."""
    vectors={k:v.detach().reshape(-1).double() for k,v in vectors.items()}
    lengths={k:float(v.norm()) for k,v in vectors.items()}
    total=vectors['total']
    total_sq=float(total@total)
    contributions=[]
    for key,value in vectors.items():
        contributions.append(dict(loss=key,norm=lengths[key],mean=float(value.mean()),
            norm_to_total=lengths[key]/lengths['total'] if lengths['total'] else None,
            projection_on_total=float(value@total)/total_sq if total_sq else None))
    pairs=[]
    for a,b in itertools.combinations(vectors,2):
        if a=='total' or b=='total':
            continue
        dot=float(vectors[a]@vectors[b])
        denominator=lengths[a]*lengths[b]
        pairs.append(dict(left=a,right=b,dot=dot,cosine=dot/denominator if denominator else None))
    assert all(np.isfinite(r['norm']) for r in contributions)
    return contributions,pairs


def parameter_vectors(grads,names,group):
    pieces=[]
    for name,g in zip(names,grads):
        if group=='strength_head' and name.startswith('directional_decoder.head.'):
            pieces.append(g[1].reshape(-1))
        elif group=='direction_head' and name.startswith('directional_decoder.head.'):
            pieces.append(g[0].reshape(-1))
        elif group=='shared' and (name.startswith('spatial_memory.') or
                                (name.startswith('directional_decoder.') and '.head.' not in name)):
            pieces.append(g.reshape(-1))
        elif group=='decoder_trunk' and name.startswith('directional_decoder.') and '.head.' not in name:
            pieces.append(g.reshape(-1))
        elif group=='memory' and name.startswith('spatial_memory.'):
            pieces.append(g.reshape(-1))
    return torch.cat(pieces)


def combinations(grads):
    result=dict(grads)
    result['mask']=[a+b for a,b in zip(grads['bce'],grads['dice'])]
    result['total']=[sum(parts) for parts in zip(*(grads[k] for k in TERMS))]
    return result


def attribute(grads,names,metadata):
    contributions,pairs=[],[]
    expanded=combinations(grads)
    for group in ('strength_head','direction_head','shared','decoder_trunk','memory'):
        vectors={k:parameter_vectors(v,names,group) for k,v in expanded.items()}
        c,p=vector_summary(vectors)
        contributions += [dict(**metadata,group=group,**r) for r in c]
        pairs += [dict(**metadata,group=group,**r) for r in p]
    return contributions,pairs


def regional_strength_gradients(raw, parameters, grad_z, masks):
    result={}
    for key,mask in masks.items():
        upstream=torch.zeros_like(raw)
        upstream[:,1:2]=grad_z*mask
        result[key]=list(torch.autograd.grad(
            raw,parameters,grad_outputs=upstream,retain_graph=True,allow_unused=True))
        result[key]=[torch.zeros_like(p) if g is None else g for p,g in zip(parameters,result[key])]
    return result


def statistic(values):
    result=describe(values)
    if result is not None:
        values=np.asarray(values,np.float64)
        result.update(mean_abs=float(np.abs(values).mean()),rms=float(np.sqrt(np.mean(values**2))),
            zero_fraction=float((values==0).mean()),positive_fraction=float((values>0).mean()),negative_fraction=float((values<0).mean()))
    return result


def select_sources():
    original=OUTPUT_ROOT/'directional_large_overfit16_seed42_20261019'
    ablation=OUTPUT_ROOT/'large_loss_ablation16_seed42_20261019'
    return [('large95',original,'best.pt','L0')]+[(v,ablation/v,'last.pt',v) for v in ('L0','L1','L2')]


def run(args):
    out=Path(args.output).resolve()
    if out.exists() or not out.is_relative_to(OUTPUT_ROOT.resolve()) or out==OUTPUT_ROOT.resolve():
        raise ValueError('Use a new output subdirectory')
    sources=select_sources()
    watched=list((RESEARCH_ROOT/'models').glob('*.py'))+list((RESEARCH_ROOT/'training').glob('*.py'))
    watched += [root/name for _,root,checkpoint,_ in sources for name in (checkpoint,'fixed_train16.jsonl','rounds.jsonl')]
    before={str(p):file_identity(p)['sha256'] for p in watched}
    out.mkdir(parents=True)
    (out/'native').mkdir()
    dump(out/'protocol.json',dict(
        checkpoints=[dict(name=v,checkpoint=file_identity(root/file),loss_recipe=recipe) for v,root,file,recipe in sources],
        threshold=.5,instances=16,clicks='Each checkpoint recorded eval five-click sequence, replayed unchanged; sequences differ across checkpoints.',
        mode='eval with research autograd; autograd.grad only; no optimizer, backward, train mode or weight updates.',
        loss_weights=WEIGHTS,own_output='Own round loss dL/dS and dL/dz_s; no future losses. Positive gradient descent decreases S, negative increases S.',
        parameters='Own round and sequence-mean full-BPTT loss gradients on Strength Head row1, Direction Head row0, shared=decoder trunk+ConvGRU, and each shared subgroup.',
        terms='BCE, Dice, Direction, weighted0.1 Preserve; mask=BCE+Dice; total=all four. L1 Direction inactive.',
        recurrent='Direction has no own-round direct S path, but may influence Strength Head through previous masks and recurrent state in full BPTT.',
        region_vjp='Current Strength-logit channel path only: Mask gradient restricted to clicked error component vs GT-correct region; includes earlier parameter occurrences through current head features. Does not equal full Mask gradient including current D and bypass paths.',
        saturation='Sigmoid derivative S(1-S)<1e-4 is weak gate, not proof of loss gradient disappearance; measure dL/dS, dL/dz_s, clipping and exact zero rates.',
        clipping='Current MaskEditor torch.logit(P_new,eps=1e-6); BCE may have zero P gradient when clipped. Dice acts directly on P_new.',
        regions='Native256 valid; clicked same-sign error component, all FN, FP, old GT-correct, radius8 disk, aligned/wrong-direction clicked error. Same nearest-pixel 8-connectivity as earlier audits.',
        statistics='Pooled pixel occurrences and distributions by latest click sign; parameters per-event macro norms/cosines with zero norm excluded. No causal history claim.'))
    all_stats,contributions,pairs,regional_pairs,events,sequences,verification=[],[],[],[],[],[],[]
    torch.set_num_threads(4)
    for variant,root,file,recipe in sources:
        model=LargeDirectionalPIMSNet().to(args.device).eval()
        info=load_training_checkpoint(root/file,model)
        dataset=InteractiveDataset(root/'fixed_train16.jsonl')
        records=[]
        with (root/'rounds.jsonl').open() as f:
            for line in f:
                r=json.loads(line)
                if r['phase']=='eval' and r['epoch']==info['epoch']:
                    records.append(r)
        assert len(dataset)==16 and len(records)==80
        digest=model_digest(model)
        parameters=list(model.trainable_parameters())
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        captured=[]
        handle=model.directional_decoder.head.register_forward_hook(lambda m,inputs,result: captured.append((result,inputs[0])))
        pools=defaultdict(list)
        max_error=chain_error=0.
        variant_events=0
        for index in range(16):
            sample=dataset[index]
            target,valid_t=prepare_target(sample['mask'],args.device)
            truth,valid=target[0,0].cpu().numpy().astype(bool),valid_t[0,0].cpu().numpy().astype(bool)
            saved=sorted([r for r in records if r['index']==index],key=lambda r:r['round'])
            sequence_losses=[]
            sign_trace=[]
            for step,click in enumerate(saved,1):
                points,labels=[[[click['x'],click['y']]]],[[click['label']]]
                if step==1:
                    state=model.initialize(sample['image'],points,labels)
                else:
                    previous=state.probabilities
                    old=previous.detach()[0,0].cpu().numpy()
                    component,cc=clicked_error_component(old,truth,valid,click,sample['mask'].shape)
                    gate=torch.as_tensor(component,device=args.device)[None,None]
                    state=model.refine(state,points,labels)
                    raw,features=captured.pop()
                    assert torch.equal(raw[:,1:2].sigmoid(),state.strength)
                    if recipe=='L0':
                        terms=directional_losses(state,previous,target,valid_t,threshold=.5)
                    else:
                        terms=local_directional_losses(state,previous,target,valid_t,gate,click['label'],recipe)
                    weighted={k:WEIGHTS[k]*terms[k] for k in TERMS}
                    sequence_losses.append(weighted)
                    sign_trace.append(click['label'])
                    metadata=dict(variant=variant,index=index,round=step,label=click['label'],scope='own_round_full_BPTT')
                    term_grads,output_grads={},{}
                    for key,loss in weighted.items():
                        g=gradients(loss,[state.strength,raw,state.probabilities,*parameters])
                        output_grads[key]=g[:3]
                        term_grads[key]=g[3:]
                    c,p=attribute(term_grads,names,metadata)
                    contributions.extend(c)
                    pairs.extend(p)
                    expanded=combinations(output_grads)
                    masks=dict(clicked=gate,correct=torch.as_tensor(((old>.5)==truth)&valid,device=args.device)[None,None])
                    regional=regional_strength_gradients(raw,parameters,expanded['mask'][1][:,1:2].detach(),masks)
                    for group in ('strength_head','shared','decoder_trunk','memory'):
                        a,b=[parameter_vectors(regional[k],names,group).double() for k in ('clicked','correct')]
                        na,nb=float(a.norm()),float(b.norm())
                        regional_pairs.append(dict(**metadata,group=group,clicked_norm=na,correct_norm=nb,
                            cosine=float(a@b)/(na*nb) if na*nb else None,
                            correct_to_clicked_norm=nb/na if na else None))
                    direction=state.direction.detach()[0,0].cpu().numpy()
                    strength=state.strength.detach()[0,0].cpu().numpy()
                    new=state.probabilities.detach()[0,0].cpu().numpy()
                    z=raw.detach()[0,1].cpu().numpy()
                    derivative=strength*(1-strength)
                    yy,xx=np.mgrid[:256,:256]
                    disk=(xx-cc['click_native_x'])**2+(yy-cc['click_native_y'])**2<=64
                    regions=dict(component=component,FN=truth&(old<=.5)&valid,FP=~truth&(old>.5)&valid,
                        correct=((old>.5)==truth)&valid,click_disk=disk&valid,
                        component_aligned=component&((direction>0) if click['label'] else (direction<0)),
                        component_wrong=component&((direction<=0) if click['label'] else (direction>=0)))
                    maps=dict(logit_s=z,strength=strength,sigmoid_derivative=derivative,direction=direction,
                        delta_p=new-old,probability_new=new,available_step=np.where(direction>=0,direction*(1-old),direction*old),
                        sigmoid_gate_small=(derivative<1e-4).astype(np.float32),
                        logit_s_below_minus8=(z<-8).astype(np.float32),sigmoid_exact_endpoint=((strength==0)|(strength==1)).astype(np.float32),
                        bce_probability_clipped=((new<1e-6)|(new>1-1e-6)).astype(np.float32))
                    bce_support=int(valid.sum()) if recipe=='L0' else int(component.sum())
                    for key,g in expanded.items():
                        gs,gz=g[0].detach()[0,0].cpu().numpy(),g[1].detach()[0,1].cpu().numpy()
                        maps['gS_'+key]=gs
                        maps['gz_'+key]=gz
                        expected=gs*derivative
                        residual=float(np.max(np.abs(gz-expected)))
                        scale=float(np.max(np.abs(gz)))
                        assert residual<=1e-6*max(scale,1e-12)+1e-12,(variant,key,residual,scale)
                        chain_error=max(chain_error,residual)
                    maps['bce_unreduced_gz']=maps['gz_bce']*bce_support
                    maps['bce_zero_gz']=(maps['gz_bce']==0).astype(np.float32)
                    maps['mask_zero_gz']=(maps['gz_mask']==0).astype(np.float32)
                    event_stats={}
                    for region,selection in regions.items():
                        event_stats[region]={}
                        for field,array in maps.items():
                            values=array[selection]
                            event_stats[region][field]=statistic(values)
                            if values.size:
                                pools[click['label'],region,field].append(values.copy())
                    filename=f'{variant}_object_{index:02d}_round_{step}.npz'
                    np.savez_compressed(out/'native'/filename,**maps,previous=old,truth=truth,valid=valid,component=component)
                    event=dict(**metadata,epoch=info['epoch'],component_pixels=int(component.sum()),**cc,
                        x=click['x'],y=click['y'],losses={k:float(v.detach()) for k,v in weighted.items()},
                        distributions=event_stats,native_file='native/'+filename)
                    events.append(event)
                    variant_events+=1
                    del term_grads,regional,output_grads,expanded
                with torch.no_grad():
                    metrics,_,_=state_metrics(model,state,sample['mask'],truth,valid)
                    max_error=max(max_error,abs(metrics['iou']-click['iou']))
                    assert max_error==0
            sequence_grads={k:gradients(torch.stack([t[k] for t in sequence_losses]).mean(),parameters) for k in TERMS}
            metadata=dict(variant=variant,index=index,round=0,label=-1,scope='sequence_mean_full_BPTT')
            c,p=attribute(sequence_grads,names,metadata)
            contributions.extend(c)
            pairs.extend(p)
            sequences.append(dict(variant=variant,index=index,epoch=info['epoch'],labels=sign_trace,
                losses={k:float(torch.stack([t[k] for t in sequence_losses]).mean().detach()) for k in TERMS}))
            del sequence_grads,sequence_losses,state,raw,features,previous
            assert all(p.grad is None for p in model.parameters())
            print(f'{variant} Strength gradient audit {index+1}/16',flush=True)
        for (label,region,field),arrays in sorted(pools.items()):
            all_stats.append(dict(variant=variant,label=label,region=region,field=field,**statistic(np.concatenate(arrays))))
        del pools
        handle.remove()
        assert not captured and digest==model_digest(model)
        verification.append(dict(variant=variant,epoch=info['epoch'],updates=info['global_step'],replayed_rounds=80,
            edit_events=variant_events,iou_max_error=max_error,sigmoid_chain_max_absolute_error=chain_error,
            model_digest_unchanged=True,parameter_grad_fields_empty=True))
        del model,parameters
        for name,value in (('region_statistics',all_stats),('parameter_contributions',contributions),('parameter_pairs',pairs),
                           ('regional_strength_pairs',regional_pairs),('events',events),('sequences',sequences),('verification',verification)):
            dump(out/(name+'.json'),value)
    assert before=={p:file_identity(p)['sha256'] for p in before}
    dump(out/'unchanged_hashes.json',before)
    for name,rows in (('region_statistics',all_stats),('parameter_contributions',contributions),('parameter_pairs',pairs),('regional_strength_pairs',regional_pairs)):
        write_csv(out/(name+'.csv'),rows)
    sequence_output_audit(out,args.device)
    summarize(out,all_stats,contributions,pairs,regional_pairs,verification)


def sequence_output_audit(out,device):
    """Actual mean-round training loss gradients at every Strength occurrence."""
    all_stats,checks=[],[]
    for variant,root,file,recipe in select_sources():
        model=LargeDirectionalPIMSNet().to(device).eval()
        info=load_training_checkpoint(root/file,model)
        dataset=InteractiveDataset(root/'fixed_train16.jsonl')
        with (root/'rounds.jsonl').open() as stream:
            records=[r for line in stream if (r:=json.loads(line))['phase']=='eval' and r['epoch']==info['epoch']]
        before=model_digest(model)
        captured=[]
        handle=model.directional_decoder.head.register_forward_hook(lambda m,inputs,result:captured.append(result))
        pools=defaultdict(list)
        chain_error=0.
        for index in range(16):
            sample=dataset[index]
            target,valid=prepare_target(sample['mask'],device)
            truth=target[0,0].cpu().numpy().astype(bool)
            valid_np=valid[0,0].cpu().numpy().astype(bool)
            nodes,losses,metadata=[],[],[]
            clicks=sorted([r for r in records if r['index']==index],key=lambda r:r['round'])
            for step,click in enumerate(clicks,1):
                points,labels=[[[click['x'],click['y']]]],[[click['label']]]
                if step==1:
                    state=model.initialize(sample['image'],points,labels)
                    continue
                previous=state.probabilities
                old=previous.detach()[0,0].cpu().numpy()
                component,_=clicked_error_component(old,truth,valid_np,click,sample['mask'].shape)
                state=model.refine(state,points,labels)
                raw=captured.pop()
                terms=directional_losses(state,previous,target,valid,threshold=.5) if recipe=='L0' else local_directional_losses(
                    state,previous,target,valid,torch.as_tensor(component,device=device)[None,None],click['label'],recipe)
                losses.append({k:WEIGHTS[k]*terms[k] for k in TERMS})
                nodes.extend((state.strength,raw))
                metadata.append(dict(round=step,label=click['label'],component=component,
                    FN=truth&(old<=.5)&valid_np,correct=((old>.5)==truth)&valid_np))
                with torch.no_grad():
                    actual,_,_=state_metrics(model,state,sample['mask'],truth,valid_np)
                    assert actual['iou']==click['iou']
            grads={k:gradients(torch.stack([t[k] for t in losses]).mean(),nodes) for k in TERMS}
            expanded=combinations(grads)
            for event_index,meta in enumerate(metadata):
                s=nodes[2*event_index].detach()[0,0].cpu().numpy()
                derivative=s*(1-s)
                maps={}
                file_name=f'{variant}_object_{index:02d}_round_{meta["round"]}.npz'
                with np.load(out/'native'/file_name) as own:
                    for key,g in expanded.items():
                        gs=g[2*event_index].detach()[0,0].cpu().numpy()
                        gz=g[2*event_index+1].detach()[0,1].cpu().numpy()
                        residual=float(np.max(np.abs(gz-gs*derivative)))
                        assert residual<=1e-6*max(float(np.max(np.abs(gz))),1e-12)+1e-12
                        chain_error=max(chain_error,residual)
                        maps['gS_'+key]=gs
                        maps['gz_'+key]=gz
                        maps['gz_future_'+key]=gz-own['gz_'+key]/len(losses)
                    own_total=own['gz_total']/len(losses)
                    maps['future_reverses_upward_total']=((own_total<0)&(maps['gz_total']>0)).astype(np.float32)
                    maps['future_reverses_downward_total']=((own_total>0)&(maps['gz_total']<0)).astype(np.float32)
                for region in ('component','FN','correct'):
                    for field,array in maps.items():
                        values=array[meta[region]]
                        if values.size:
                            pools[meta['label'],region,field].append(values.copy())
                np.savez_compressed(out/'native'/('sequence_'+file_name),**maps)
            del nodes,losses,grads,expanded,state,raw,previous,terms
            assert all(p.grad is None for p in model.parameters())
        handle.remove()
        assert before==model_digest(model) and not captured
        for (label,region,field),values in sorted(pools.items()):
            all_stats.append(dict(variant=variant,label=label,region=region,field=field,**statistic(np.concatenate(values))))
        checks.append(dict(variant=variant,objects=16,edited_outputs=64,model_unchanged=True,
                           sigmoid_chain_max_absolute_error=chain_error,all_iou_exact=True))
        del pools,model
        print(f'{variant} full-sequence output gradients verified',flush=True)
    dump(out/'sequence_output_statistics.json',all_stats)
    write_csv(out/'sequence_output_statistics.csv',all_stats)
    dump(out/'sequence_output_verification.json',checks)


def summarize(out,stats,contributions,pairs,regional,verification):
    summaries=[]
    for variant in ('large95','L0','L1','L2'):
        for label in (1,0,-1):
            scope='sequence_mean_full_BPTT' if label==-1 else 'own_round_full_BPTT'
            for group in ('strength_head','shared','decoder_trunk','memory'):
                base=[r for r in contributions if r['variant']==variant and r['label']==label and r['scope']==scope and r['group']==group]
                for loss in (*TERMS,'mask','total'):
                    selected=[r for r in base if r['loss']==loss]
                    summaries.append(dict(variant=variant,label=label,scope=scope,group=group,loss=loss,events=len(selected),
                        mean_norm=float(np.mean([r['norm'] for r in selected])),
                        zero_norm_fraction=float(np.mean([r['norm']==0 for r in selected])),
                        mean_projection_on_total=float(np.mean([r['projection_on_total'] for r in selected if r['projection_on_total'] is not None])) if any(r['projection_on_total'] is not None for r in selected) else None))
    pair_summaries=[]
    keys={(r['variant'],r['label'],r['group'],r['left'],r['right'],r['scope']) for r in pairs}
    for variant,label,group,left,right,scope in sorted(keys):
        selected=[r['cosine'] for r in pairs if (r['variant'],r['label'],r['group'],r['left'],r['right'],r['scope'])==(variant,label,group,left,right,scope)]
        valid=[x for x in selected if x is not None]
        pair_summaries.append(dict(variant=variant,label=label,group=group,left=left,right=right,scope=scope,
            events=len(selected),valid_cosines=len(valid),mean_cosine=float(np.mean(valid)) if valid else None,
            conflict_fraction=float(np.mean(np.asarray(valid)<0)) if valid else None))
    dump(out/'contribution_summary.json',summaries)
    dump(out/'pair_summary.json',pair_summaries)
    write_csv(out/'contribution_summary.csv',summaries)
    write_csv(out/'pair_summary.csv',pair_summaries)
    render(out,stats,summaries,pair_summaries,regional,verification)


def render(out,stats,contributions,pairs,regional,verification):
    def get(v,l,f,key='mean',region='component'):
        r=next((r for r in stats if (r['variant'],r['label'],r['region'],r['field'])==(v,l,region,f)),None)
        return r[key] if r else None
    def number(v): return '—' if v is None else f'{v:.4g}'
    def pct(v): return '—' if v is None else f'{100*v:.2f}%'
    variants=('large95','L0','L1','L2')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Strength梯度诊断</title><style>body{font:16px/1.7 system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}</style><h1>DirectionalPIMSNet Strength塌缩：只读梯度诊断</h1>'
    page+='<p>主诊断large95为更大版原epoch95最佳权重；L0/L1/L2为损失消融800更新最终权重。各自回放16实例原有五击，按当前点击符号分组。原模型eval，使用autograd.grad，无优化器和权重更新。各组后续点击不同，且L1/L2损失区域不同，不作严格数值因果比较。</p>'
    page+='<h2>点击所在错误连通域：Sigmoid门控与Mask梯度</h2><p>正点击对应FN，负点击对应FP。统计按像素×事件合并；clip指P_new超出torch.logit(eps=1e-6)的内部区间。小Sigmoid导数仅表示门控小，须结合实际梯度判断。</p>'
    page+=table(['权重','点击','z_S中位数','S均值','S中位数','S(1−S)中位数','导数<1e−4','S恰为0/1','BCE裁剪比例','BCE gz=0','Mask gz=0','BCE逐像素等效gz均值'],[
        [v,'正FN' if l else '负FP',number(get(v,l,'logit_s','median')),number(get(v,l,'strength')),number(get(v,l,'strength','median')),
         number(get(v,l,'sigmoid_derivative','median')),pct(get(v,l,'sigmoid_gate_small')),pct(get(v,l,'sigmoid_exact_endpoint')),pct(get(v,l,'bce_probability_clipped')),
         pct(get(v,l,'bce_zero_gz')),pct(get(v,l,'mask_zero_gz')),number(get(v,l,'bce_unreduced_gz'))] for v in variants for l in (1,0)])
    page+='<h2>主检查点结论：衰减与竞争并存，不能等同梯度死亡</h2>'
    page+=f'<p>large95正点击FN连通域的Strength-logit中位数为{number(get("large95",1,"logit_s","median"))}，S中位数为{number(get("large95",1,"strength","median"))}。Sigmoid导数小于1e−4的像素占{pct(get("large95",1,"sigmoid_gate_small"))}，但BCE的Strength-logit梯度为0仅占{pct(get("large95",1,"bce_zero_gz"))}；BCE负梯度（梯度下降会提高S）占{pct(get("large95",1,"gz_bce","negative_fraction"))}。Mask=BCE+Dice的gz为0比例为{pct(get("large95",1,"mask_zero_gz"))}。这支持低门控和梯度衰减，但不支持整个FN域被Sigmoid或logit裁剪完全阻断。</p>'
    cc=[r for r in regional if r['variant']=='large95' and r['label']==1 and r['group']=='strength_head']
    valid_cc=[r for r in cc if r['cosine'] is not None]
    if valid_cc:
        mean_cc=float(np.mean([r['clicked_norm'] for r in cc]))
        mean_correct=float(np.mean([r['correct_norm'] for r in cc]))
        page+=f'<p>同一个全域Mask Loss内，FN域修复与原本正确区域，经当前Strength输出路径回传到Strength Head后，{len(valid_cc)}个有效正点击事件的冲突率为{pct(float(np.mean([r["cosine"]<0 for r in valid_cc])))}，平均余弦{number(float(np.mean([r["cosine"] for r in valid_cc])))}。正确区域与目标域的平均梯度范数分别为{number(mean_correct)}、{number(mean_cc)}。这是空间监督竞争的直接证据，不能仅比较Mask与Direction两个损失的整体余弦。</p>'
    def contribution(loss,group='strength_head'):
        return next(r['mean_norm'] for r in contributions if r['variant']=='large95' and r['label']==1 and r['group']==group and r['loss']==loss)
    page+=f'<p>正点击事件的Strength Head完整历史梯度范数宏平均：Mask={number(contribution("mask"))}、Direction={number(contribution("direction"))}、加权Preserve={number(contribution("preserve"))}。因此不能将当前检查点归因于Preserve梯度范数主导。Direction对本轮S没有直接监督；其Strength Head梯度来自历史反馈，而共享参数上的Direction梯度也可影响后续S。</p>'
    page+='<p>对照L1/L2移除正确区域的Mask监督后，该路径的正确区域Mask梯度为0，同时出现接近1的Strength和严重整图破坏。这与“修复和保护之间缺少合适的空间分离”相符，但这些消融同时改变了监督区域与配方，不能单独证明历史训练的唯一原因。</p>'
    page+='<p>L1/L2还出现float32 Sigmoid精确输出1的数值饱和，正点击域约有一半以上像素、负点击域几乎全部像素处于该端点。但有些端点已经成功纠正错误，零梯度本身并不等于失败；需结合正确区域破坏判断。large95正点击域没有S精确为0或1的像素，应与这种端点饱和区分。</p>'
    plot_diagnostics(out,contributions,regional)
    page+='<img style="max-width:100%" src="strength_gradient_diagnostics.png">'
    page+='<p>BCE逐像素等效gz=实际gz×BCE监督像素数，解除mean归一化影响。负梯度意味着梯度下降增加Strength；正梯度意味着降低Strength。Dice的归一化不同，原始梯度单独列示。</p>'
    page+=table(['权重','点击','Loss','∂L/∂S均值','|∂L/∂S|均值','∂L/∂z均值','|∂L/∂z|均值','gz负比例','gz正比例'],[
        [v,'正' if l else '负',loss,number(get(v,l,'gS_'+loss)),number(get(v,l,'gS_'+loss,'mean_abs')),
         number(get(v,l,'gz_'+loss)),number(get(v,l,'gz_'+loss,'mean_abs')),pct(get(v,l,'gz_'+loss,'negative_fraction')),pct(get(v,l,'gz_'+loss,'positive_fraction'))]
        for v in variants for l in (1,0) for loss in (*TERMS,'mask','total')])
    page+='<h2>各项损失的参数梯度贡献</h2><p>表为各事件自身损失的完整历史BPTT梯度；Strength Head仅输出层第1行权重/偏置，共65参数；shared为Decoder trunk与ConvGRU，不包含两个输出head。Preserve已乘0.1。范数是事件宏平均，投影贡献dot(g_i,g_total)/||g_total||²允许负值，四项投影之和为1。Mask是BCE+Dice，不能与四项再次相加。</p>'
    page+=table(['权重','点击','参数组','Loss','平均梯度范数','零范数事件','平均总梯度投影贡献'],[
        [r['variant'],'正' if r['label']==1 else ('负' if r['label']==0 else '整序列平均'),r['group'],r['loss'],number(r['mean_norm']),pct(r['zero_norm_fraction']),number(r['mean_projection_on_total'])]
        for r in contributions if r['group'] in ('strength_head','shared')])
    page+='<h2>参数梯度冲突（事件宏平均）</h2><p>零范数不计为无冲突，排除分母。Direction对当前S输出无直接梯度；对Strength参数的非零梯度可经过去轮Mask与记忆反馈形成。</p>'
    page+=table(['权重','点击','参数组','Loss对','有效/事件','余弦均值','冲突率'],[
        [r['variant'],'正' if r['label']==1 else ('负' if r['label']==0 else '整序列'),r['group'],r['left']+' / '+r['right'],f'{r["valid_cosines"]}/{r["events"]}',number(r['mean_cosine']),pct(r['conflict_fraction'])]
        for r in pairs if r['group'] in ('strength_head','shared') and (r['left'],r['right']) in (('direction','mask'),('preserve','mask'))])
    page+='<h2>当前Strength路径：错误连通域与正确区域的Mask梯度</h2><p>只将当前Strength-logit通道的Mask梯度按空间区域分开再做VJP，排除了当前Direction与概率旁路；可检查同一个Strength输出路径上的空间监督竞争。该分解不等于完整Mask参数梯度。</p>'
    rows=[]
    for v in variants:
        for l in (1,0):
            for group in ('strength_head','shared'):
                data=[r for r in regional if r['variant']==v and r['label']==l and r['group']==group]
                valid=[r['cosine'] for r in data if r['cosine'] is not None]
                rows.append([v,'正' if l else '负',group,len(valid),number(float(np.mean(valid)) if valid else None),pct(float(np.mean(np.asarray(valid)<0)) if valid else None),
                    number(float(np.mean([r['clicked_norm'] for r in data]))),number(float(np.mean([r['correct_norm'] for r in data])))])
    page+=table(['权重','点击','参数组','有效事件','平均余弦','冲突率','错误域梯度范数','正确域梯度范数'],rows)
    if (out/'sequence_output_statistics.json').exists():
        seq=json.loads((out/'sequence_output_statistics.json').read_text())
        def seq_get(v,l,f,key='mean'):
            row=next((r for r in seq if (r['variant'],r['label'],r['region'],r['field'])==(v,l,'component',f)),None)
            return row[key] if row else None
        page+='<h2>实际整序列平均损失对每轮Strength输出的梯度</h2><p>这里的梯度包含后续各轮损失，且按4次编辑平均，区别于前文单轮自身损失。future项=整序列梯度−本轮自身梯度/4，因此Direction也可能对早期S产生非零梯度。future翻转比例只统计“自身总梯度要求增大S，但整序列总梯度反而要求降低S”的像素占当前错误域的比例。</p>'
        page+=table(['权重','点击','Loss','整序列gS均值','整序列gz均值','整序列|gz|均值','仅future gz均值'],[
            [v,'正' if l else '负',loss,number(seq_get(v,l,'gS_'+loss)),number(seq_get(v,l,'gz_'+loss)),number(seq_get(v,l,'gz_'+loss,'mean_abs')),number(seq_get(v,l,'gz_future_'+loss))]
            for v in variants for l in (1,0) for loss in (*TERMS,'mask','total')])
        page+=table(['权重','点击','future将增S改为降S比例'],[[v,'正' if l else '负',pct(seq_get(v,l,'future_reverses_upward_total'))] for v in variants for l in (1,0)])
        page+=f'<p>large95正点击FN域的future反向翻转仅占{100*seq_get("large95",1,"future_reverses_upward_total"):.3f}%，整序列总gz均值为{number(seq_get("large95",1,"gz_total"))}，总体仍要求提高Strength；不支持未来损失普遍逆转FN修复梯度。</p>'
    page+='<h2>核验与局限</h2><p>每个检查点80轮IoU精确回放；所有模型状态字典、已有权重和模型源码哈希不变；参数.grad始终为空；逐像素检验gz=gS×S(1−S)。完整分布、梯度和区域保存在region_statistics.csv与native/；逐事件参数范数/投影/两两余弦保存在parameter_*.csv，scope区分本轮损失与整序列平均损失。当前检查点的梯度只能说明局部优化倾向，不能重建历史训练因果。</p></html>'
    (out/'report.html').write_text(page,encoding='utf-8')


def plot_diagnostics(out,contributions,regional):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pools={l:defaultdict(list) for l in (1,0)}
    events=json.loads((out/'events.json').read_text())
    for event in events:
        if event['variant']!='large95' or not event['component_pixels']:
            continue
        with np.load(out/event['native_file']) as data:
            gate=data['component']
            for field in ('logit_s','sigmoid_derivative','bce_unreduced_gz'):
                pools[event['label']][field].append(data[field][gate])
    values={l:{k:np.concatenate(a) for k,a in fields.items()} for l,fields in pools.items()}
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axes[0,0].boxplot([values[l]['logit_s'] for l in (1,0)],tick_labels=['Positive FN','Negative FP'],showfliers=False)
    axes[0,0].set(title='Strength logits: large95 (outliers hidden)',ylabel='Pre-sigmoid logit')
    for label,color in ((1,'tab:blue'),(0,'tab:orange')):
        derivative=values[label]['sigmoid_derivative']
        gradient=np.abs(values[label]['bce_unreduced_gz'])
        valid=(derivative>0)&(gradient>0)
        ids=np.flatnonzero(valid)
        if ids.size>2000:
            ids=np.random.default_rng(42+label).choice(ids,2000,replace=False)
        axes[0,1].scatter(derivative[ids],gradient[ids],s=3,alpha=.3,color=color,label='Positive FN' if label else 'Negative FP')
    axes[0,1].set(xscale='log',yscale='log',xlabel='S(1-S)',ylabel='|BCE gz| x supervised pixels',title='Strength gate vs BCE gradient\n(nonzero pixels only)')
    axes[0,1].legend()
    x=np.arange(len(TERMS))
    for label,offset in ((1,-.18),(0,.18)):
        norms=[next(r['mean_norm'] for r in contributions if r['variant']=='large95' and r['label']==label and r['group']=='strength_head' and r['loss']==loss) for loss in TERMS]
        axes[1,0].bar(x+offset,norms,width=.35,label='Positive click' if label else 'Negative click')
    axes[1,0].set(xticks=x,xticklabels=TERMS,yscale='log',ylabel='Mean gradient norm',title='Strength Head: weighted loss gradients, full history')
    axes[1,0].legend()
    arrays=[]
    for label,group in ((1,'strength_head'),(1,'shared'),(0,'strength_head'),(0,'shared')):
        arrays.append([r['cosine'] for r in regional if r['variant']=='large95' and r['label']==label and r['group']==group and r['cosine'] is not None])
    axes[1,1].boxplot(arrays,tick_labels=['Pos / S head','Pos / shared','Neg / S head','Neg / shared'],showfliers=True)
    axes[1,1].axhline(0,color='black',linewidth=.7)
    axes[1,1].set(ylim=(-1.05,1.05),ylabel='Cosine',title='Clicked error vs correct-region Mask gradients\n(current Strength path)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig(out/'strength_gradient_diagnostics.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())
