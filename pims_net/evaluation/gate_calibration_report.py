"""Independent saved-array checks and complete G2 inference calibration report."""

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pims_net.evaluate_eta import write_csv
from pims_net.evaluation.directional_overfit_report import table
from pims_net.evaluation.experiment import file_identity
from pims_net.overfit_directional import dump
from pims_net.paths import REPO_ROOT,RESEARCH_ROOT


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def unpack(data,key,shape):
    return np.unpackbits(data[key],count=int(np.prod(shape))).reshape(shape).astype(bool)


def verify_arrays(output,rows,config):
    assets={i:dict(np.load(output/'instances'/f'{i:02d}.npz')) for i in range(16)}
    checks=dict(arrays=0,native_formula_exact=0,original_contribution_exact=0,native_contribution_exact=0,
                monotone_original_masks=0,max_delta_iou_identity_error=0.,calibration_uses_GT=False)
    for row in rows:
        with np.load(output/row['array']) as d:
            a=assets[row['index']]
            base,fused,g=d['base'],d['fused'],d['raw_gate']
            effective=np.float32(row['alpha'])*np.where(g>=row['tau'],g,np.zeros_like(g))
            expected=base+effective*(1-base)
            assert np.array_equal(expected,fused),row['array']
            assert np.all(fused>=base)
            shape=a['original_gt'].shape
            old,new=unpack(d,'base_prediction_bits',shape),unpack(d,'prediction_bits',shape)
            gt=a['original_gt'].astype(bool)
            assert np.all(~old|new),row['array']
            assert d['click'].tolist()==[row['x'],row['y'],row['label']]
            if not row['click_skipped']: assert bool(gt[row['y'],row['x']])==bool(row['label'])
            # Recount directly from saved masks, independently of runtime helpers.
            k=int((~old&new&gt).sum()); e=int((~old&new&~gt).sum())
            tp=int((old&gt).sum()); union=int((old|gt).sum())
            iou_before=tp/union if union else 1.
            union_after=int((new|gt).sum())
            iou_after=int((new&gt).sum())/union_after if union_after else 1.
            c=row['contribution']
            assert (k,e)==(c['K'],c['E'])
            assert abs(c['base_iou']-iou_before)<1e-12 and abs(c['fused_iou']-iou_after)<1e-12
            assert abs(row['iou']-iou_after)<1e-12
            error=abs((iou_after-iou_before)-c['delta_iou_from_counts'])
            assert error<1e-12
            checks['max_delta_iou_identity_error']=max(checks['max_delta_iou_identity_error'],error)
            if e:
                assert c['ratio_status']=='finite' and c['K_over_E']==k/e
                assert ((k/e>iou_before)==(iou_after>iou_before)) or abs(iou_after-iou_before)<1e-12
            else:
                assert c['K_over_E'] is None
                assert c['ratio_status']==('infinite' if k else 'undefined')
            valid=a['valid'].astype(bool); truth=a['target'].astype(bool)&valid
            nb=(base>.5)&valid; nf=(fused>.5)&valid
            nk=int((~nb&nf&truth).sum()); ne=int((~nb&nf&~truth).sum())
            assert (nk,ne)==(row['native_contribution']['K'],row['native_contribution']['E'])
            if row['label']!=1 or row['round']==1:
                assert np.array_equal(base,fused) and np.array_equal(old,new)
            checks['arrays']+=1
            checks['native_formula_exact']+=1
            checks['original_contribution_exact']+=1
            checks['native_contribution_exact']+=1
            checks['monotone_original_masks']+=1
    assert checks['arrays']==3840
    for i in range(16):
        seqs=[]
        for setting in config['settings']:
            selected=[r for r in rows if r['protocol']=='replay' and r['index']==i and r['setting']==setting['key']]
            seqs.append([(r['x'],r['y'],r['label']) for r in selected])
        assert all(s==seqs[0] for s in seqs)
    checks['fixed_replay_same_24x16x5']=True
    provenance=json.loads((output/'verification.json').read_text())
    assert provenance['alpha0_exact_states']==640
    assert provenance['original_G2_exact_states']==160
    assert provenance['G0_reference_exact_states']==160
    assert provenance['no_optimizer'] and provenance['no_backward'] and provenance['model_state_dict_unchanged']
    assert all(file_identity(p)['sha256']==sha for p,sha in provenance['watched_sha256'].items())
    baseline=json.loads((RESEARCH_ROOT/'docs/baseline_manifest.json').read_text())
    assert all(file_identity(REPO_ROOT/p)['sha256']==sha for p,sha in baseline.items())
    tests=(output/'test_results.txt').read_text()
    assert 'Ran 7 tests' in tests and 'OK' in tests
    checks.update(alpha0_bitexact_states=640,original_G2_bitexact_states=160,unit_tests=7,
        model_and_weights_unchanged=True,baseline_files_unchanged=len(baseline))
    dump(output/'statistics_checks.json',checks)
    return assets,checks


def fmt(v,digits=4): return '—' if v is None else f'{v:.{digits}f}'
def pct(v): return '—' if v is None else f'{100*v:.3f}%'
def rfmt(row,prefix=''):
    state=row[prefix+'ratio_status']
    return '∞' if state=='infinite' else ('—' if state=='undefined' else fmt(row[prefix+'K_over_E'],3))


def report(output):
    config=json.loads((output/'protocol.json').read_text())
    status=json.loads((output/'status.json').read_text())
    assert status['status']=='completed' and not config['smoke']
    rows=read_lines(output/'rounds.jsonl')
    comparison=json.loads((output/'comparison.json').read_text())
    assets,checks=verify_arrays(output,rows,config)
    primary=[r for r in rows if r['alpha']==1 and r['tau']==0 and r['positive_click']]
    contribution_round=list(csv.DictReader((output/'contribution_by_round.csv').open(encoding='utf-8')))
    contribution_instance=list(csv.DictReader((output/'contribution_by_instance.csv').open(encoding='utf-8')))
    best=[]
    paired=[]
    instances=list(csv.DictReader((output/'per_instance.csv').open(encoding='utf-8')))
    observations=[]
    for method in ('replay','adaptive'):
        candidates=[r for r in comparison if r['protocol']==method]
        g0=next(r for r in candidates if r['alpha']==0 and r['tau']==0)
        g2=next(r for r in candidates if r['alpha']==1 and r['tau']==0)
        optimum=max(candidates,key=lambda r:r['IoU@5'])
        best.append(optimum)
        deltas=[]
        for index in range(16):
            baseline=next(r for r in instances if r['protocol']==method and r['setting']==g0['setting'] and int(r['index'])==index)
            calibrated=next(r for r in instances if r['protocol']==method and r['setting']==optimum['setting'] and int(r['index'])==index)
            delta=float(calibrated['IoU@5'])-float(baseline['IoU@5'])
            deltas.append(delta)
            paired.append(dict(protocol=method,index=index,setting=optimum['setting'],G0_IoU5=float(baseline['IoU@5']),
                calibrated_IoU5=float(calibrated['IoU@5']),delta_IoU5=delta))
        zero=[r for r in candidates if r['alpha']==0]
        keys=[k for k in g0 if k.startswith('IoU@') or k.endswith('_rate')]
        assert all(all(r[k]==g0[k] for k in keys) for r in zero)
        observations.append(f"{method}：最佳网格点α={optimum['alpha']:g}、τ={optimum['tau']:g}，IoU@5={optimum['IoU@5']:.6f}；相对G0 {optimum['IoU@5']-g0['IoU@5']:+.6f}，相对原G2 {optimum['IoU@5']-g2['IoU@5']:+.6f}。")
        observations.append(f"{method} 最佳校准相对G0：16实例改善{sum(d>1e-12 for d in deltas)}、退化{sum(d< -1e-12 for d in deltas)}、持平{sum(abs(d)<=1e-12 for d in deltas)}；TP破坏 {pct(g0['tp_damage_rate'])}→{pct(optimum['tp_damage_rate'])}，TN误加 {pct(g0['tn_damage_rate'])}→{pct(optimum['tn_damage_rate'])}。")
        originals=[r for r in primary if r['protocol']==method]
        k=sum(r['contribution']['K'] for r in originals); e=sum(r['contribution']['E'] for r in originals)
        observations.append(f"{method} 原G2的{len(originals)}次正点击：Gate单独K={k:,}、E={e:,}，池化K/E={k/e:.4f}；ΔIoU均值={np.mean([r['contribution']['delta_iou'] for r in originals]):+.6f}；改善{sum(r['contribution']['delta_iou']>1e-12 for r in originals)}次、退化{sum(r['contribution']['delta_iou']< -1e-12 for r in originals)}次。")
    observations.append('ΣK/ΣE是像素事件池化比值，平均ΔIoU则对正点击逐次等权平均，二者不能相互替代。K/E>基础IoU的改善判据应逐次使用本轮K、E和基础IoU，不能拿全体池化K/E与平均IoU比较。原G2在小目标上的少量误加可产生较大IoU损失，案例图给出实测反例。')
    write_csv(output/'best_calibration_paired.csv',paired)
    dump(output/'best_calibration.json',best)
    dump(output/'observations.json',observations)
    # All calibration metrics include alpha0 repeated across thresholds.
    alpha_values=config['alphas']; tau_values=config['taus']
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for line,method in enumerate(('replay','adaptive')):
        for col,metric in enumerate(('IoU@3','IoU@5','tn_damage_rate')):
            grid=np.array([[next(r[metric] for r in comparison if r['protocol']==method and r['alpha']==a and r['tau']==t) for t in tau_values] for a in alpha_values])
            ax=axes[line,col]
            display_grid=grid if metric.startswith('IoU') else grid*100
            image=ax.imshow(display_grid,cmap='viridis' if metric.startswith('IoU') else 'magma')
            ax.set_xticks(range(4),[str(t) for t in tau_values]);ax.set_yticks(range(6),[str(a) for a in alpha_values])
            ax.set_xlabel('tau (threshold before alpha)');ax.set_ylabel('alpha');ax.set_title(method+' / '+metric+(' (%)' if not metric.startswith('IoU') else ''))
            for y in range(6):
                for x in range(4): ax.text(x,y,f'{display_grid[y,x]:.3f}',ha='center',va='center',fontsize=9,color='white' if image.norm(display_grid[y,x])<.55 else 'black')
            fig.colorbar(image,ax=ax,fraction=.045)
    fig.savefig(output/'calibration_grid.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for ax,method in zip(axes,('replay','adaptive')):
        g0=next(r['IoU@5'] for r in comparison if r['protocol']==method and r['alpha']==0 and r['tau']==0)
        for tau in tau_values:
            values=[next(r['IoU@5'] for r in comparison if r['protocol']==method and r['alpha']==a and r['tau']==tau) for a in alpha_values]
            ax.plot(alpha_values,values,marker='o',label=f'tau {tau:g}')
        ax.axhline(g0,color='#555',ls='--',label='G0');ax.set_title(method);ax.set_xlabel('alpha');ax.set_ylabel('IoU@5');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(output/'alpha_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,method in zip(axes,('replay','adaptive')):
        selected=[r for r in primary if r['protocol']==method]
        for positive,color,label in ((True,'#087f83','delta IoU > 0'),(False,'#cc5037','delta IoU <= 0')):
            subset=[r for r in selected if (r['contribution']['delta_iou']>1e-12)==positive]
            ax.scatter([r['contribution']['E'] for r in subset],[r['contribution']['K'] for r in subset],color=color,label=label,alpha=.75)
        ax.set_xscale('symlog',linthresh=1);ax.set_yscale('symlog',linthresh=1);ax.set_xlabel('E: TN -> FP');ax.set_ylabel('K: FN -> TP');ax.set_title('Original G2 / '+method);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(output/'single_round_contribution.png',dpi=150);plt.close(fig)
    # Best and worst local effects, each selected from the original G2 trajectory.
    cases=[min(primary,key=lambda r:r['contribution']['delta_iou']),max(primary,key=lambda r:r['contribution']['delta_iou'])]
    fig,axes=plt.subplots(2,6,figsize=(16,5.8),layout='constrained')
    for line,row in enumerate(cases):
        a=assets[row['index']]
        d=np.load(output/row['array'])
        shape=a['original_gt'].shape
        old,new=unpack(d,'base_prediction_bits',shape),unpack(d,'prediction_bits',shape)
        gt=a['original_gt'].astype(bool)
        additions=np.zeros((*shape,3),np.float32)
        additions[~old&new&gt]=[.05,.85,.35]
        additions[~old&new&~gt]=[1.,.2,.1]
        panels=[(a['image'],'Instance '+str(row['index']),None),(gt,'GT','gray'),
                (old,f'Pbase IoU {row["contribution"]["base_iou"]:.3f}','gray'),
                (new,f'Pfused IoU {row["iou"]:.3f}','gray'),(d['raw_gate'],'Raw Gate','magma'),
                (additions,f'K {row["contribution"]["K"]} / E {row["contribution"]["E"]}',None)]
        for col,(value,label,cmap) in enumerate(panels):
            ax=axes[line,col];ax.imshow(value,cmap=cmap,vmin=0 if cmap else None,vmax=1 if cmap else None);ax.set_title(label,fontsize=10);ax.axis('off')
            if col==4:
                h,w=shape
                rh,rw=int(h*1024/max(h,w)+.5),int(w*1024/max(h,w)+.5)
                px,py=row['x']*rw/w/4,row['y']*rh/h/4
            else:
                px,py=row['x'],row['y']
            ax.plot(px,py,'+',color='#00d5ff',markersize=10,markeredgewidth=2)
    fig.savefig(output/'local_examples.png',dpi=150);plt.close(fig)
    sections=[]
    sections.append('<h2>结果观察</h2>'+''.join('<p>'+html.escape(s)+'</p>' for s in observations))
    highlight=[]
    for method in ('replay','adaptive'):
        optimum=next(r for r in best if r['protocol']==method)
        for name,alpha,tau in (('G0',0.,0.),('原G2',1.,0.),('最佳校准',optimum['alpha'],optimum['tau'])):
            r=next(r for r in comparison if r['protocol']==method and r['alpha']==alpha and r['tau']==tau)
            highlight.append([method,name,fmt(alpha,2),fmt(tau,1),*[fmt(r[f'IoU@{i}']) for i in (1,3,5)],
                *[pct(r[k]) for k in ('fn_repair_rate','fp_removal_rate','tp_damage_rate','tn_damage_rate')]])
    sections.append('<h2>关键对比</h2>'+table(['协议','方案','α','τ','IoU@1','IoU@3','IoU@5','FN修复','FP删除','TP破坏','TN误加'],highlight))
    for method in ('replay','adaptive'):
        selected=[r for r in comparison if r['protocol']==method]
        g0=next(r for r in selected if r['alpha']==0 and r['tau']==0)
        grid=[]
        for r in selected:
            grid.append([fmt(r['alpha'],2),fmt(r['tau'],1),*[fmt(r[f'IoU@{i}']) for i in (1,3,5)],
                *[pct(r[k]) for k in ('fn_repair_rate','fp_removal_rate','tp_damage_rate','tn_damage_rate')],
                f'{r["IoU@5"]-g0["IoU@5"]:+.5f}'])
        sections.append('<h2>'+method+' · 全部24组</h2>'+table(
            ['α','τ','IoU@1','IoU@3','IoU@5','FN修复','FP删除','TP破坏','TN误加','ΔIoU@5 vs G0'],grid))
    # Only alpha1/tau0 is primary original-G2 single-round attribution.
    for title,records,group in (('按点击轮次',contribution_round,'round'),('按实例',contribution_instance,'index')):
        selected=[r for r in records if float(r['alpha'])==1 and float(r['tau'])==0]
        data=[]
        for r in selected:
            if group=='round' and int(r['round'])==1: continue
            status=r['ratio_status'];ratio='∞' if status=='infinite' else ('—' if status=='undefined' else fmt(float(r['K_over_E']),3))
            data.append([r['protocol'],r[group],r['positive_clicks'],r['K'],r['E'],ratio,
                fmt(float(r['delta_iou_mean'])) if r['delta_iou_mean'] else '—',r['improved_clicks'],r['damaged_clicks']])
        sections.append('<h2>原G2单轮贡献 · '+title+'</h2>'+table(
            ['协议',group,'正点击数','K：FN→TP','E：TN→FP','ΣK/ΣE','平均ΔIoU','改善次数','退化次数'],data))
    checks_text=html.escape(json.dumps(checks,ensure_ascii=False,indent=2))
    css='body{font:15px system-ui,sans-serif;background:#f4f6fa;color:#1e293b;max-width:1480px;margin:auto;padding:30px}h1{font-size:28px}h2{margin-top:30px}table{background:white;border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:right;border-bottom:1px solid #dce3ed}th:first-child,td:first-child{text-align:left}img{max-width:100%;background:white;border-radius:8px}p{line-height:1.65}.protocol{background:#e7edf7;padding:17px;border-radius:9px}'
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>G2 Gate 单轮贡献与推理校准</title><style>{css}</style>
    <h1>G2 Gate：单轮贡献与推理校准</h1><div class="protocol">
    原G2的800步最终权重；16个固定COCO训练实例，seed42。模型全程eval + inference_mode，不训练、不保存新模型权重。<br>
    正点击第2–5轮：Gate′ = α × Gate × 1[Gate≥τ]；P_fused = P_base + Gate′ × (1−P_base)。阈值作用于倍率前原始Gate，保留通过阈值的幅度。<br>
    第一击保留冻结EV-SAM，负点击返回P_base；α=0精确保留P_base及原始Logits，隐藏状态仍由原冻结S0正常推进。<br>
    单轮对照：同一次S0前向、同一输入历史与当前点击、同一更新后的ConvGRU隐藏状态，仅分支Gate更新；P_base不会再次推进记忆。<br>
    GT仅用于RITM模拟点击与事后统计，绝不用于推理区域或Gate截断。各校准条件反馈自身Mask，训练时与推理时的数值反馈定义一致。<br>
    Replay沿用原实验S0固定五击；adaptive每组使用相同RITM规则、相同种子但从自身Mask选择后续点击。<br>
    K/E及ΔIoU主结果在原图网格、官方SAM Logits后处理后以P&gt;0.5计算。原生256有效网格另存，不混合统计。<br>
    K：本轮Gate独自把基础FN修为TP；E：基础TN被误加为FP。按轮/实例汇总ΣK、ΣE及正点击ΔIoU均值，重复编辑像素可重复计数。<br>
    若基础TP为T、并集为U，ΔIoU=(T+K)/(U+E)−T/U；E&gt;0时，Gate改善IoU的条件为K/E&gt;基础IoU。Gate本身TP破坏与FP删除均为0。<br>
    完整五击FN修复/FP删除/TP破坏/TN误加均以首击为参照，按原图像素池化；IoU为16实例平均。不能将单轮Gate计数等同完整闭环变化。<br>
    所有最优点仅从这16个训练实例选出，属于推理校准探索，不代表泛化结果。</div>
    {''.join(sections)}
    <h2>校准网格</h2><img src="calibration_grid.png"><h2>倍率扫描</h2><img src="alpha_curves.png">
    <h2>原G2单轮K/E与ΔIoU</h2><img src="single_round_contribution.png">
    <h2>原G2最差/最好单轮案例</h2><p>绿=新增TP，红=新增FP；青色十字为当前点击。</p><img src="local_examples.png">
    <h2>独立验收检查</h2><pre>{checks_text}</pre>
    <p><a href="comparison.csv">完整对比CSV</a> · <a href="round_summary.csv">逐轮指标</a> · <a href="positive_click_contributions.csv">每次正点击贡献</a> · <a href="contribution_by_round.csv">按轮贡献</a> · <a href="contribution_by_instance.csv">按实例贡献</a> · <a href="per_instance.csv">实例完整指标</a> · <a href="best_calibration_paired.csv">最佳校准逐实例对比</a> · <a href="protocol.json">实验协议</a> · <a href="statistics_checks.json">独立验证</a> · <a href="test_results.txt">单元测试</a></p></html>'''
    (output/'report.html').write_text(doc,encoding='utf-8')
    print(json.dumps(dict(checks=checks,best=best,observations=observations),indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    report(Path(parser.parse_args().output).resolve())
