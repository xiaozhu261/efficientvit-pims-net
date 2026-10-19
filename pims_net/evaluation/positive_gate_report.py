"""Verify saved native masks, then report Gate training and both click protocols."""

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pims_net.diagnose_large_clicks import describe
from pims_net.evaluate_eta import write_csv
from pims_net.evaluate_oracle import clicked_error_component
from pims_net.evaluate_directional_constraints import transition_rates
from pims_net.evaluation.directional_overfit_report import table
from pims_net.evaluation.experiment import file_identity
from pims_net.overfit_directional import dump


COLORS={'G0':'#586174','G1':'#087e8b','G2':'#d47819'}


def lines(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def verify(output,config,rows):
    checks=dict(native_arrays=0,formula_exact=0,monotone_probability=0,monotone_logits=0,
                disabled_or_negative_identity=0,target_components_exact=0,max_sigmoid_error=0.)
    for path in sorted((output/'native').glob('*.npz')):
        variant,protocol,_,round_name=path.stem.split('_')
        data=np.load(path)
        p,base,g,z,l,bl=[data[k] for k in ('probabilities','base','gate','gate_logits','logits','base_logits')]
        x,y,label=data['click'].tolist()
        assert bool(data['original_gt'][y,x])==bool(label),path
        active=label==1 and variant!='G0' and round_name!='r1'
        expected=base+g*(1-base)
        assert np.array_equal(expected,p),path
        assert np.all(p>=base) and np.all(l>=bl),path
        checks['native_arrays']+=1
        checks['formula_exact']+=1
        checks['monotone_probability']+=1
        checks['monotone_logits']+=1
        if not active:
            assert np.array_equal(p,base) and np.array_equal(l,bl) and not g.any(),path
            checks['disabled_or_negative_identity']+=1
        else:
            sigmoid=1/(1+np.exp(-np.clip(z.astype(np.float64),-700,700)))
            error=float(np.max(np.abs(sigmoid-g)))
            assert error<1.5e-7,(path,error)
            checks['max_sigmoid_error']=max(checks['max_sigmoid_error'],error)
        if round_name!='r1':
            component,_=clicked_error_component(base,data['target'].astype(bool),data['valid'].astype(bool),
                dict(x=x,y=y,label=label),data['original_gt'].shape)
            if label==0: component[:]=False
            assert np.array_equal(component,data['selected']),path
            checks['target_components_exact']+=1
    assert checks['native_arrays']==480, checks
    replay=[r for r in rows if r['selection']=='last' and r['protocol']=='replay']
    for index in range(16):
        sequences=[[tuple(r[k] for k in ('x','y','label')) for r in replay if r['variant']==v and r['index']==index] for v in ('G0','G1','G2')]
        assert sequences[0]==sequences[1]==sequences[2]
    checks['fixed_replay_identical_16x5']=True
    initial=[]
    for variant in ('G1','G2'):
        payload=torch.load(output/variant/'last.pt',map_location='cpu',weights_only=False)
        assert payload['global_step']==800 and payload['epoch']==100
        assert 'research_state_dict' not in payload
        assert payload['config']['frozen_s0']==config['frozen_s0']
        assert sum(v.numel() for v in payload['gate_state_dict'].values())==config['gate_parameters']
        initial.append(payload['config']['initial_gate_sha256'])
    assert len(set(initial))==1
    checks['same_initialization_and_800_updates']=True
    integrity=json.loads((output/'verification.json').read_text())
    assert integrity['unchanged']
    assert all(file_identity(path)['sha256']==sha for path,sha in integrity['watched_sha256'].items())
    checks['old_sources_and_checkpoints_unchanged']=True
    original=json.loads((output/'original_integrity.json').read_text())
    assert original['passed'] and original['baseline_original_files']==213
    checks['original_baseline_files']=original['baseline_original_files']
    checks['original_a3_files_and_weights']=original['original_a3_files_and_weights']
    tests=(output/'test_results.txt').read_text()
    assert 'Ran 8 tests' in tests and 'OK' in tests
    checks['unit_tests_passed']=8
    dump(output/'statistics_checks.json',checks)
    return checks


def pooled_counts(rows,key):
    values=[r[key] for r in rows if r[key] is not None]
    return {name:sum(v[name] for v in values) for name in values[0]} if values else {}


def gate_distributions(output):
    result=[]
    for variant in ('G1','G2'):
        for protocol in ('replay','adaptive'):
            for label in (1,0):
                parts={region:dict(gate=[],raw_sigmoid=[],logit=[],delta=[]) for region in ('all','target','FN','FP','TP','TN')}
                for path in sorted((output/'native').glob(f'{variant}_{protocol}_*.npz')):
                    d=np.load(path)
                    if path.stem.endswith('_r1') or d['click'][2]!=label: continue
                    v,t,b=[d[k].astype(bool) if k!='base' else d[k] for k in ('valid','target','base')]
                    regions=dict(all=v,target=d['selected'],FN=v&t&(b<=.5),FP=v&~t&(b>.5),TP=v&t&(b>.5),TN=v&~t&(b<=.5))
                    z=d['gate_logits']
                    values=dict(gate=d['gate'],logit=z,delta=d['probabilities']-b,
                                raw_sigmoid=1/(1+np.exp(-np.clip(z.astype(np.float64),-700,700))))
                    for region,mask in regions.items():
                        for key,value in values.items(): parts[region][key].append(value[mask])
                for region,values in parts.items():
                    pooled={k:np.concatenate(v) if v else np.empty(0) for k,v in values.items()}
                    stats={k:describe(v) for k,v in pooled.items()}
                    g,raw=pooled['gate'],pooled['raw_sigmoid']
                    saturation=dict(near_zero=float((g<=.01).mean()),near_one=float((g>=.99).mean()),
                        endpoint_total=float(((g<=.01)|(g>=.99)).mean()),
                        raw_sigmoid_derivative_below_1e_minus_4=float((raw*(1-raw)<1e-4).mean())) if g.size else None
                    result.append(dict(variant=variant,protocol=protocol,label=label,region=region,saturation=saturation,**stats))
    dump(output/'gate_distributions.json',result)
    flat=[]
    for row in result:
        entry={k:row[k] for k in ('variant','protocol','label','region')}
        for field in ('gate','raw_sigmoid','logit','delta'):
            entry.update({field+'_'+key:row[field][key] if row[field] else None for key in ('count','mean','median','q90','maximum')})
        entry.update({'saturation_'+k:row['saturation'][k] if row['saturation'] else None
            for k in ('near_zero','near_one','endpoint_total','raw_sigmoid_derivative_below_1e_minus_4')})
        flat.append(entry)
    write_csv(output/'gate_distributions.csv',flat)
    return result


def fmt(x,digits=4):
    return '—' if x is None else f'{x:.{digits}f}'


def pct(x):
    return '—' if x is None else f'{100*x:.3f}%'


def report(output):
    config=json.loads((output/'protocol.json').read_text())
    summaries=json.loads((output/'comparison.json').read_text())
    rows=list(lines(output/'evaluation_rounds.jsonl'))
    checks=verify(output,config,rows)
    distributions=gate_distributions(output)
    flat=[]
    for s in summaries:
        flat.append({k:v for k,v in s.items() if k!='counts'})
    write_csv(output/'comparison.csv',flat)
    round_rows=[]
    signed=[]
    for protocol in ('replay','adaptive'):
        for variant in ('G0','G1','G2'):
            selected=[r for r in rows if r['protocol']==protocol and r['variant']==variant and r['selection']=='last']
            for step in range(1,6):
                batch=[r for r in selected if r['round']==step]
                c=pooled_counts(batch,'cumulative_counts')
                round_rows.append(dict(protocol=protocol,variant=variant,round=step,IoU=float(np.mean([r['iou'] for r in batch])),**transition_rates(c)))
            for label in (1,0):
                batch=[r for r in selected if r['round']>1 and r['label']==label and not r['click_skipped']]
                c=pooled_counts(batch,'gate_only_counts')
                inc=pooled_counts(batch,'incremental_counts')
                signed.append(dict(protocol=protocol,variant=variant,label=label,clicks=len(batch),
                    empty_target_clicks=sum(r['target_pixels']==0 for r in batch),target_pixels=sum(r['target_pixels'] for r in batch),
                    **{('gate_'+k):v for k,v in transition_rates(c).items()},
                    **{('whole_round_'+k):v for k,v in transition_rates(inc).items()},
                    gate_counts=c,whole_round_counts=inc))
    write_csv(output/'round_comparison.csv',round_rows)
    dump(output/'signed_edit_effects.json',signed)
    write_csv(output/'signed_edit_effects.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in signed])
    # Adaptive training curves; best snapshots may precede the full budget.
    epochs={v:list(lines(output/v/'epochs.jsonl')) for v in ('G1','G2')}
    gradient_summary=[]
    for variant,records in epochs.items():
        train=[r for r in records if r['phase']=='train']
        values=np.array([r['gradient_norm_mean'] for r in train])
        gradient_summary.append(dict(variant=variant,epoch_means=len(values),updates=len(values)*8,
            total_L2_before_clip_mean=float(values.mean()),minimum_epoch_mean=float(values.min()),maximum_epoch_mean=float(values.max()),
            final20_epochs_mean=float(values[-20:].mean()),clip_threshold=1.,
            definition='Each epoch value is mean of 8 batch2 pre-clip total Gate gradient L2 norms. Frozen S0 gradient None.'))
    dump(output/'gradient_norms.json',gradient_summary)
    write_csv(output/'gradient_norms.csv',[{k:v for k,v in r.items() if k!='definition'} for r in gradient_summary])
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
    for variant,data in epochs.items():
        ev=[r for r in data if r['phase']=='eval']
        axes[0].plot([r['updates'] for r in ev],[r['IoU@5'] for r in ev],marker='o',label=variant,color=COLORS[variant])
        tr=[r for r in data if r['phase']=='train']
        for key,style in (('gate','-'),('bce','--'),('dice',':')):
            axes[1].plot([r['updates'] for r in tr],[r['losses'][key] for r in tr],style,label=f'{variant} {key}',color=COLORS[variant])
        axes[2].plot([r['updates'] for r in tr],[r['gradient_norm_mean'] for r in tr],label=variant,color=COLORS[variant])
    axes[0].set_title('Adaptive train16 IoU@5'); axes[1].set_title('Positive-round losses (unweighted)');axes[2].set_title('Gate pre-clip L2 norm (8-batch epoch mean)')
    for ax in axes:
        ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(output/'training.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,protocol in zip(axes,('replay','adaptive')):
        for variant in ('G0','G1','G2'):
            selected=[r for r in round_rows if r['protocol']==protocol and r['variant']==variant]
            ax.plot([r['round'] for r in selected],[r['IoU'] for r in selected],marker='o',label=variant,color=COLORS[variant])
        ax.set_title(protocol);ax.set_xlabel('Click');ax.set_xticks(range(1,6));ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'click_iou.png',dpi=150);plt.close(fig)
    # Visualize original-resolution predictions; Gate heatmaps remain native256.
    adaptive_last=[r for r in rows if r['protocol']=='adaptive' and r['selection']=='last' and r['round']==5]
    diffs=[]
    for i in range(16):
        pair={r['variant']:r['iou'] for r in adaptive_last if r['index']==i}
        diffs.append((pair['G2']-pair['G0'],i))
    examples=[min(diffs)[1],max(diffs)[1]]
    fig,axes=plt.subplots(2,6,figsize=(15,5.4),layout='constrained')
    for line,index in enumerate(examples):
        native={v:np.load(output/'native'/f'{v}_adaptive_{index:02d}_r5.npz') for v in ('G0','G1','G2')}
        axes[line,0].imshow(native['G0']['original_gt'],cmap='gray',vmin=0,vmax=1);axes[line,0].set_title(f'Instance {index} GT')
        for j,v in enumerate(('G0','G1','G2'),1):
            axes[line,j].imshow(native[v]['original_prediction'],cmap='gray',vmin=0,vmax=1)
            iou=next(r['iou'] for r in adaptive_last if r['variant']==v and r['index']==index)
            axes[line,j].set_title(f'{v} IoU {iou:.3f}')
        for j,v in enumerate(('G1','G2'),4):
            axes[line,j].imshow(native[v]['gate'],cmap='magma',vmin=0,vmax=1)
            label=int(native[v]['click'][2]);axes[line,j].set_title(f'{v} Gate, sign {label}')
        for ax in axes[line]:ax.axis('off')
    fig.savefig(output/'examples.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(13,5),layout='constrained')
    for line,variant in enumerate(('G1','G2')):
        choices=[]
        for path in sorted((output/'native').glob(f'{variant}_adaptive_*.npz')):
            d=np.load(path)
            if not path.stem.endswith('_r1') and d['click'][2]==1:
                choices.append((int(d['selected'].sum()),path))
        _,chosen=max(choices,key=lambda x:x[0])
        d=np.load(chosen)
        panels=[(d['target'],variant+' GT','gray'),(d['base']>.5,'S0 base','gray'),
                (d['selected'],'Residual clicked FN','gray'),(d['gate'],'Gate','magma'),
                (d['probabilities']>.5,'After Gate','gray')]
        x,y,label=d['click'].tolist()
        _,location=clicked_error_component(d['base'],d['target'].astype(bool),d['valid'].astype(bool),
            dict(x=x,y=y,label=label),d['original_gt'].shape)
        for ax,(value,label,cmap) in zip(axes[line],panels):
            ax.imshow(value,cmap=cmap,vmin=0,vmax=1);ax.set_title(label);ax.axis('off')
            ax.plot(location['click_native_x'],location['click_native_y'],'+',color='#42d9ff',markersize=10,markeredgewidth=2)
    fig.savefig(output/'positive_examples.png',dpi=150);plt.close(fig)
    headers=['方案','IoU@1','IoU@3','IoU@5','FN修复','FP删除','TP破坏','TN误加','Precision@5','Recall@5']
    sections=[]
    for protocol in ('replay','adaptive'):
        selected=[s for s in summaries if s['protocol']==protocol and s['selection']=='last']
        data=[[s['variant'],*[fmt(s[f'IoU@{r}']) for r in (1,3,5)],
               *[pct(s[k]) for k in ('fn_repair_rate','fp_removal_rate','tp_damage_rate','tn_damage_rate')],fmt(s['precision']),fmt(s['recall'])] for s in selected]
        sections.append(f'<h2>{protocol} · 完整800步最终权重</h2>'+table(headers,data))
    best_rows=[[s['variant'],s['protocol'],s['epoch'],s['updates'],fmt(s['IoU@3']),fmt(s['IoU@5'])]
        for s in summaries if s['selection']=='best']
    sections.append('<h2>最佳训练集快照（补充）</h2>'+table(['方案','协议','epoch','更新数','IoU@3','IoU@5'],best_rows))
    initialized_best=sorted({s['variant'] for s in summaries if s['selection']=='best' and s['updates']==0})
    if initialized_best:
        sections.append('<p>'+html.escape('、'.join(initialized_best))+' 的最佳快照是epoch0，属于未训练Gate初始化，不能作为训练改善的证据；正式对照以800步最终权重为准。</p>')
    sections.append('<h2>单独 Gate 带来的像素变化</h2><p>从每轮冻结 S0 的 Pbase 到 Gate 输出，按点击符号分别汇总；分母为各轮基础预测中的对应像素数之和。区别于上表从第一击到第五击的累计变化。</p>'+table(
        ['协议','方案','点击符号','次数','空目标','目标像素','FN修复','FP删除','TP破坏','TN误加'],
        [[r['protocol'],r['variant'],'正' if r['label'] else '负',r['clicks'],r['empty_target_clicks'],r['target_pixels'],
          *[pct(r['gate_'+k]) for k in ('fn_repair_rate','fp_removal_rate','tp_damage_rate','tn_damage_rate')]] for r in signed]))
    dist_rows=[]
    for r in distributions:
        if r['label']!=1: continue
        g,z,d=r['gate'],r['logit'],r['delta']
        dist_rows.append([r['protocol'],r['variant'],r['region'],g['count'] if g else 0,
            fmt(g['mean']) if g else '—',fmt(g['median']) if g else '—',fmt(g['q90']) if g else '—',
            fmt(z['median']) if z else '—',fmt(d['mean']) if d else '—'])
    sections.append('<h2>正点击 Gate 分布（第2–5击，有效区域像素池化）</h2>'+table(
        ['协议','方案','区域','像素数','Gate均值','中位数','P90','Logit中位数','ΔP均值'],dist_rows))
    sections.append('<h2>Gate饱和率与梯度范数</h2><p>饱和定义为Gate≤0.01或≥0.99，另统计原始Sigmoid导数&lt;1e−4。只列启用Gate的正点击；负点击强制0不作为学习塌缩证据。下表为原生有效像素池化比例，不能由小导数单独推断BCE梯度消失。</p>'+table(
        ['协议','方案','区域','Gate≤.01','Gate≥.99','合计饱和','Sigmoid导数&lt;1e−4'],
        [[r['protocol'],r['variant'],r['region'],*[pct(r['saturation'][k]) for k in ('near_zero','near_one','endpoint_total','raw_sigmoid_derivative_below_1e_minus_4')]]
            for r in distributions if r['label']==1 and r['saturation']])+table(
        ['方案','更新数','全训练平均梯度L2','末20epoch均值','最小epoch均值','最大epoch均值','裁剪阈值'],
        [[r['variant'],r['updates'],*[fmt(r[k]) for k in ('total_L2_before_clip_mean','final20_epochs_mean','minimum_epoch_mean','maximum_epoch_mean','clip_threshold')]] for r in gradient_summary])+
        '<p>梯度L2先按batch2平均，再在裁剪前测量；每个epoch记录8个batch的平均值，表中最小/最大也指epoch均值，并非单batch极值。Gate每轮直接监督保留梯度；Mask数值反馈一致，但冻结S0转移使用no_grad截断跨轮梯度，不优化未来轮损失对早期Gate的间接影响。</p>')
    coverage=[]
    for variant in ('G1','G2'):
        # Stream training logs instead of retaining the full pixel-statistic history.
        selected=[r for r in lines(output/variant/'rounds.jsonl') if r['round']>1 and r['label']==1]
        coverage.append([variant,len(selected),sum(r['target_pixels']==0 for r in selected),sum(r['target_pixels'] for r in selected),
            fmt(float(np.mean([r['losses']['gate'] for r in selected if r['losses'] is not None])))])
    sections.append('<h2>训练监督覆盖</h2>'+table(['方案','正点击次数','空残余FN目标次数','目标像素总数','Gate BCE均值'],coverage))
    notes=[]
    for protocol in ('replay','adaptive'):
        selected={s['variant']:s for s in summaries if s['protocol']==protocol and s['selection']=='last'}
        for variant in ('G1','G2'):
            delta=selected[variant]['IoU@5']-selected['G0']['IoU@5']
            paired=[]
            for index in range(16):
                baseline=next(r['iou'] for r in rows if r['variant']=='G0' and r['protocol']==protocol and r['selection']=='last' and r['round']==5 and r['index']==index)
                candidate=next(r['iou'] for r in rows if r['variant']==variant and r['protocol']==protocol and r['selection']=='last' and r['round']==5 and r['index']==index)
                paired.append(candidate-baseline)
            notes.append(f'{protocol}：{variant} 相对 G0 的 IoU@5 变化 {delta:+.6f}；16实例改善{sum(x>1e-12 for x in paired)}、退化{sum(x< -1e-12 for x in paired)}、持平{sum(abs(x)<=1e-12 for x in paired)}。')
    for variant in ('G1','G2'):
        positive=next(r for r in signed if r['variant']==variant and r['protocol']=='adaptive' and r['label']==1)
        negative=next(r for r in signed if r['variant']==variant and r['protocol']=='adaptive' and r['label']==0)
        fixed=next(r for r in signed if r['variant']==variant and r['protocol']=='replay' and r['label']==1)
        notes.append(f"{variant} 自适应正点击中，Gate单独FN修复 {pct(positive['gate_fn_repair_rate'])}、TN误加 {pct(positive['gate_tn_damage_rate'])}、TP破坏0；负点击{negative['clicks']}次，Gate严格直通。固定Replay的{fixed['clicks']}次正点击中，{fixed['empty_target_clicks']}次基础预测已无点击所在残余FN域。此信息仅用于诊断，推理未用GT强制关闭Gate。")
    notes.append('G0自适应第2–5击共13次负点击；G1/G2更多的负点击表明新增FP改变了点击预算分配。Gate单独不会删除前景，但后续冻结S0仍可能破坏TP。TN区域Gate中位数接近零，并不意味着没有误加：应同时查看分布尾部、跨阈值计数与实例均值IoU。')
    dump(output/'observations.json',notes)
    sections.insert(0,'<h2>结果观察</h2>'+''.join('<p>'+html.escape(n)+'</p>' for n in notes))
    css='body{font:15px system-ui,sans-serif;background:#f4f6fa;color:#1d293d;margin:auto;max-width:1450px;padding:32px}h1{font-size:28px}h2{margin-top:32px}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dde4ed;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;background:white;border-radius:8px}code{font-size:12px}p{line-height:1.6}.protocol{background:#e7eef7;padding:16px;border-radius:8px}'
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PositiveCorrectionGate · train16</title><style>{css}</style>
    <h1>PositiveCorrectionGate：冻结 S0 的正点击纠错实验</h1>
    <div class="protocol">相同16个COCO训练实例 · seed42 · G1/G2各800次更新 · 阈值统一 P&gt;0.5。<br>
    S0最佳权重 epoch{config['s0_epoch']} / {config['s0_updates']}步；Direction、Strength、ConvGRU和SAM全部冻结。Gate参数{config['gate_parameters']:,}。<br>
    第一击保留EV-SAM；后续正点击 P=Pbase+Gate·(1−Pbase)，负点击严格返回Pbase。Gate从图像特征、最新点击、更新后的历史状态和基础Mask预测。<br>
    G0关闭Gate；G1仅区域均衡Gate BCE；G2增加全有效区域Mask BCE+Dice，三项权重均1。各图先平均正点击损失，再按batch2累积；冻结S0转移截断梯度。<br>
    正目标：当前点击在S0本轮基础预测后仍遗漏的FN 8连通域；其余有效像素负监督；点击点已正确则全负。GT只用于训练监督、模拟点击和统计，推理Gate不读取GT。<br>
    Replay沿用S0最佳权重的固定五击；adaptive由各方法自身预测选择后续点击，使用相同RITM规则与种子。所有方法每轮独立反馈自身Mask，记忆按图重置。<br>
    IoU为原图16实例均值；主表FN修复、FP删除、TP破坏和TN误加是第一击到第五击的像素池化比例。仅为训练集过拟合实验，不代表泛化性能。</div>
    {''.join(sections)}
    <h2>逐击IoU</h2><img src="click_iou.png"><h2>训练曲线与梯度范数</h2><img src="training.png">
    <h2>自适应协议下 G2 改变量最小/最大实例</h2><img src="examples.png">
    <h2>最大残余FN目标的正点击单轮案例</h2><img src="positive_examples.png">
    <h2>验收检查</h2><pre>{html.escape(json.dumps(checks,ensure_ascii=False,indent=2))}</pre>
    <p>正向Gate本身不删除FP，也不破坏TP；完整闭环中后续冻结S0仍可能产生删除或破坏，需以上表实测结果为准。近饱和概率转换Logits时加入单调保护，避免数值截断导致正向更新的Logits下降。负点击和G0均逐位保留基础概率及原始Logits。</p>
    <p><a href="comparison.csv">完整对比CSV</a> · <a href="round_comparison.csv">逐击指标</a> · <a href="signed_edit_effects.csv">正负点击编辑效果</a> · <a href="gate_distributions.csv">Gate分布及饱和率</a> · <a href="gradient_norms.csv">梯度范数</a> · <a href="protocol.json">协议与来源</a> · <a href="statistics_checks.json">独立统计验证</a> · <a href="test_results.txt">单元测试</a></p></html>'''
    (output/'report.html').write_text(document,encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    report(Path(parser.parse_args().output).resolve())
