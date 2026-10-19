"""Inference-only G2 Gate attribution and alpha x tau calibration, train16."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluate_directional_constraints import transition_counts, transition_rates
from pims_net.evaluate_eta import write_csv
from pims_net.evaluation.experiment import file_identity
from pims_net.evaluation.metrics import mask_iou, segmentation_statistics
from pims_net.models.directional_large import LargeDirectionalPIMSNet
from pims_net.models.positive_correction_gate import PositiveCorrectionPIMSNet
from pims_net.overfit_directional import dump
from pims_net.overfit_positive_gate import gate_digest, load_gate, original_probability
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT, checkpoint_path
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.click_simulator import RITMClickSimulator
from pims_net.training.interactive_losses import prepare_target


ALPHAS=(0.,.1,.25,.5,.75,1.)
TAUS=(0.,.3,.5,.7)


def setting_key(alpha,tau):
    return f'a{alpha:g}_t{tau:g}'.replace('.','p')


def calibrated_update(base,base_logits,raw_gate,labels,alpha,tau):
    """Threshold the original Gate, retain its magnitude, THEN scale by alpha.

    This function has no GT or spatial-region argument. Labels only enable
    positive clicks. Its Logit monotonic guard is the original G2 convention.
    """
    if not all(math.isfinite(v) and 0<=v<=1 for v in (alpha,tau)):
        raise ValueError('alpha and tau must be finite in [0,1]')
    positive=labels.reshape(-1,1,1,1)==1
    effective=alpha*torch.where(raw_gate>=tau,raw_gate,torch.zeros_like(raw_gate))*positive
    if alpha==0:
        return base,base_logits,effective
    candidate=base+effective*(1-base)
    changed=positive&(candidate!=base)
    probability=torch.where(changed,candidate,base)
    logits=torch.where(changed,torch.maximum(torch.logit(probability,eps=1e-6),base_logits),base_logits)
    return probability,logits,effective


@torch.no_grad()
def calibrated_refine(model,previous,points,labels,alpha,tau):
    # A SINGLE frozen S0 transition gives both counterfactuals the same hidden
    # state, current click and prior corrected Mask. No second memory update.
    proposal=model.refine(previous,points,labels)
    if alpha==1 and tau==0:
        return proposal
    p,l,g=calibrated_update(proposal.base_probabilities,proposal.base_logits,proposal.gate,
        proposal.labels[:,-1],alpha,tau)
    return replace(proposal,probabilities=p,logits=l,gate=g)


def ratio(k,e):
    return dict(K_over_E=k/e if e else None,
        ratio_status='finite' if e else ('infinite' if k else 'undefined'))


def contribution(base,fused,truth):
    """Original/native-grid one-step attribution; zero denominators stay explicit."""
    c=transition_counts(base,fused,truth)
    k,e=c['fn_repaired'],c['tn_damaged']
    old,new=mask_iou(base,truth),mask_iou(fused,truth)
    assert c['fp_removed']==0 and c['tp_damaged']==0, 'ADD calibration must be monotone'
    union=int((base|truth).sum())
    tp=int((base&truth).sum())
    analytic=(tp+k)/(union+e)-tp/union if union else new-old
    assert abs(analytic-(new-old))<1e-12
    return dict(K=k,E=e,base_iou=old,fused_iou=new,delta_iou=new-old,
                delta_iou_from_counts=analytic,**ratio(k,e))


def clicker_for(sample,index):
    return RITMClickSimulator(sample['mask'],np.random.default_rng(np.random.SeedSequence([42,index])))


def pooled(rows,key):
    return {k:sum(r[key][k] for r in rows) for k in rows[0][key]}


def metric_summary(rows):
    result={f'IoU@{step}':float(np.mean([r['iou'] for r in rows if r['round']==step])) for step in (1,3,5)}
    last=[r for r in rows if r['round']==5]
    counts=pooled(last,'cumulative_counts')
    return {**result,**transition_rates(counts),**{'cumulative_'+k:v for k,v in counts.items()},
        'precision_at5':float(np.mean([r['statistics']['precision'] for r in last])),
        'recall_at5':float(np.mean([r['statistics']['recall'] for r in last]))}


def contribution_summary(rows):
    positive=[r for r in rows if r['round']>1 and r['label']==1 and not r['click_skipped']]
    k=sum(r['contribution']['K'] for r in positive)
    e=sum(r['contribution']['E'] for r in positive)
    return dict(positive_clicks=len(positive),K=k,E=e,**ratio(k,e),
        delta_iou_mean=float(np.mean([r['contribution']['delta_iou'] for r in positive])) if positive else None,
        delta_iou_sum=float(sum(r['contribution']['delta_iou'] for r in positive)),
        improved_clicks=sum(r['contribution']['delta_iou']>1e-12 for r in positive),
        damaged_clicks=sum(r['contribution']['delta_iou']< -1e-12 for r in positive),
        unchanged_clicks=sum(abs(r['contribution']['delta_iou'])<=1e-12 for r in positive))


@torch.inference_mode()
def run(args):
    output=Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT.resolve()) or output==OUTPUT_ROOT.resolve():
        raise ValueError('Use a new run directory below pims_net/outputs')
    reference=Path(args.reference).resolve()
    source=json.loads((reference/'protocol.json').read_text())
    frozen=source['frozen_s0']
    assert file_identity(frozen['path'])==frozen
    assert file_identity(source['manifest']['path'])==source['manifest']
    gate_path=reference/'G2/last.pt'  # best.pt is epoch0 and is deliberately NOT used.
    watched=list((RESEARCH_ROOT/'models').glob('*.py'))+list((RESEARCH_ROOT/'training').glob('*.py'))
    watched += [checkpoint_path(),Path(frozen['path']),reference/'G2/best.pt',gate_path,
                reference/'protocol.json',reference/'fixed_clicks.json',reference/'comparison.json',
                RESEARCH_ROOT/'overfit_positive_gate.py']
    before={str(p):file_identity(p)['sha256'] for p in watched}
    output.mkdir(parents=True)
    (output/'native').mkdir()
    (output/'instances').mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    base=LargeDirectionalPIMSNet().to(args.device).eval()
    del_info=load_training_checkpoint(frozen['path'],base)
    del del_info
    model=PositiveCorrectionPIMSNet(base).to(args.device).eval().requires_grad_(False)
    info=load_gate(gate_path,model,frozen)
    assert info['epoch']==100 and info['global_step']==800 and info['config']['variant']=='G2'
    del info
    model_before=gate_digest(model)
    configs=[(a,t) for a in ALPHAS for t in TAUS]
    if args.smoke: configs=[(0.,0.),(.5,.5),(1.,0.)]
    manifest=output/'fixed_train16.jsonl'
    manifest.write_bytes(Path(source['manifest']['path']).read_bytes())
    replay=json.loads((reference/'fixed_clicks.json').read_text())
    protocol=dict(run_type='inference_calibration_on_train16_not_generalization',seed=42,
        manifest=file_identity(manifest),frozen_s0=frozen,G2_checkpoint=file_identity(gate_path),epoch=100,updates=800,
        alphas=ALPHAS,taus=TAUS,settings=[dict(key=setting_key(a,t),alpha=a,tau=t) for a,t in configs],
        formula='g_eff=alpha*g*1[g>=tau], threshold BEFORE alpha; retain g magnitude, no binarization/rescaling.',
        update='P_fused=P_base+g_eff*(1-P_base), positive clicks rounds2-5 only; other outputs exact passthrough.',
        feedback='Each condition feeds its calibrated Mask/logits and frozen S0 updated hidden to its next round, identically to G2 training.',
        counterfactual='P_base and P_fused share one call to frozen S0, identical incoming history/click and updated hidden; branch ONLY at Gate update.',
        GT_usage='Only RITM click simulation and post-hoc metrics; NEVER used for Gate calibration or spatial support.',
        probability_threshold=.5,gate_threshold='raw Gate >= tau',
        replay='Identical S0 epoch100 five-click sequence reused from previous Gate experiment.',
        adaptive='RITM on each calibrated condition own original-resolution P>0.5; same seed42/object RNG.',
        metric_grid='Original image after official SAM Logit postprocessing; native valid256 attribution also saved separately.',
        rates='First-click to fifth-click pooled original pixels. IoU is instance macro mean.',
        primary_attribution='Original G2 alpha1/tau0 trajectories, both protocols; per-round and per-instance positive-click event counts.',
        ratio='K=sum(FN->TP), E=sum(TN->FP); K/E pooled. E=0,K>0 infinite; E=K=0 undefined (null), no epsilon.',
        gradients='torch.inference_mode, all parameters require_grad=False, no optimizer or backward.',
        smoke=args.smoke)
    dump(output/'protocol.json',protocol)
    dataset=InteractiveDataset(manifest)
    indices=range(1 if args.smoke else 16)
    checks=dict(alpha0_exact_states=0,original_G2_exact_states=0,G0_reference_exact_states=0,
        monotone_states=0,negative_passthrough_states=0,formula_exact_states=0,reference_fields=['probabilities','logits','hidden','points','labels'])
    all_rows=[]
    start=time.monotonic()
    with (output/'rounds.jsonl').open('w',encoding='utf-8',buffering=1) as stream:
        for index in indices:
            sample=dataset[index]
            first=clicker_for(sample,index).next_click(np.zeros_like(sample['mask'],dtype=bool))
            x,y,label=first
            initial=model.initialize(sample['image'],[[[x,y]]],[[label]])
            target,valid=prepare_target(sample['mask'],args.device)
            native_gt=target[0,0].cpu().numpy().astype(bool)
            native_valid=valid[0,0].cpu().numpy().astype(bool)
            np.savez_compressed(output/'instances'/f'{index:02d}.npz',original_gt=sample['mask'],
                target=native_gt,valid=native_valid,image=sample['image'])
            for method in ('replay','adaptive'):
                # Generate independent pure-S0/G0 state references before any calibrated run.
                refs=[]
                reference_clicker=clicker_for(sample,index)
                assert reference_clicker.next_click(np.zeros_like(sample['mask'],dtype=bool))==first
                reference_state=initial
                for step in range(1,6):
                    if step>1:
                        click=tuple(replay[str(index)][step-1][key] for key in ('x','y','label')) if method=='replay' else reference_clicker.next_click(reference_p>.5)
                        assert click is not None
                        rx,ry,rl=click
                        reference_state=model.base.refine(reference_state,[[[rx,ry]]],[[rl]])
                    reference_p=original_probability(model,reference_state)
                    saved=np.load(reference/'native'/f'G0_{method}_{index:02d}_r{step}.npz')
                    assert np.array_equal(saved['probabilities'],reference_state.probabilities[0,0].cpu().numpy())
                    assert np.array_equal(saved['logits'],reference_state.logits[0,0].cpu().numpy())
                    assert np.array_equal(saved['original_prediction'],reference_p>.5)
                    checks['G0_reference_exact_states']+=1
                    refs.append((reference_state,reference_p>.5))
                for alpha,tau in configs:
                    key=setting_key(alpha,tau)
                    state=initial
                    clicker=clicker_for(sample,index)
                    assert clicker.next_click(np.zeros_like(sample['mask'],dtype=bool))==first
                    first_prediction=previous_prediction=None
                    for step in range(1,6):
                        skipped=False
                        if step==1:
                            x,y,label=first
                            base_p,base_logits=state.probabilities,state.logits
                            raw_gate=torch.zeros_like(base_p)
                            effective=torch.zeros_like(base_p)
                        else:
                            click=tuple(replay[str(index)][step-1][k] for k in ('x','y','label')) if method=='replay' else clicker.next_click(probability>.5)
                            if click is None:
                                skipped=True
                                x=y=label=-1
                                base_p,base_logits=state.probabilities,state.logits
                                raw_gate=torch.zeros_like(base_p)
                                effective=torch.zeros_like(base_p)
                            else:
                                x,y,label=click
                                state=calibrated_refine(model,state,[[[x,y]]],[[label]],alpha,tau)
                                base_p,base_logits=state.base_probabilities,state.base_logits
                                raw_gate=state.gate_logits.sigmoid() if label==1 else torch.zeros_like(base_p)
                                effective=alpha*torch.where(raw_gate>=tau,raw_gate,torch.zeros_like(raw_gate))
                        assert torch.equal(base_p+effective*(1-base_p),state.probabilities)
                        assert (state.probabilities>=base_p).all() and (state.logits>=base_logits).all()
                        checks['formula_exact_states']+=1
                        checks['monotone_states']+=1
                        if label==0:
                            assert torch.equal(state.probabilities,base_p) and torch.equal(state.logits,base_logits)
                            checks['negative_passthrough_states']+=1
                        if alpha==0:
                            rs,rp=refs[step-1]
                            for field in checks['reference_fields']: assert torch.equal(getattr(rs,field),getattr(state,field)),(method,index,key,step,field)
                            checks['alpha0_exact_states']+=1
                        probability=original_probability(model,state)
                        prediction=probability>.5
                        if alpha==0: assert np.array_equal(prediction,refs[step-1][1])
                        if alpha==1 and tau==0:
                            saved=np.load(reference/'native'/f'G2_{method}_{index:02d}_r{step}.npz')
                            assert np.array_equal(saved['probabilities'],state.probabilities[0,0].cpu().numpy())
                            assert np.array_equal(saved['logits'],state.logits[0,0].cpu().numpy())
                            assert np.array_equal(saved['original_prediction'],prediction)
                            if step>1: assert saved['click'].tolist()==[x,y,label]
                            checks['original_G2_exact_states']+=1
                        base_prediction=original_probability(model,replace(state,probabilities=base_p,logits=base_logits))>.5
                        native_p=state.probabilities[0,0].cpu().numpy()
                        native_base=base_p[0,0].cpu().numpy()
                        attribution=contribution(base_prediction,prediction,sample['mask'])
                        native_attribution=contribution((native_base>.5)&native_valid,(native_p>.5)&native_valid,native_gt&native_valid)
                        if first_prediction is None: first_prediction=prediction.copy()
                        cumulative=transition_counts(first_prediction,prediction,sample['mask'])
                        incremental=transition_counts(previous_prediction,prediction,sample['mask']) if previous_prediction is not None else cumulative
                        path=f'native/{method}_{key}_{index:02d}_r{step}.npz'
                        if not args.smoke:
                            np.savez_compressed(output/path,base=native_base,fused=native_p,raw_gate=raw_gate[0,0].cpu().numpy(),
                                base_prediction_bits=np.packbits(base_prediction),prediction_bits=np.packbits(prediction),
                                click=np.array([x,y,label]),original_shape=np.array(sample['mask'].shape))
                        row=dict(protocol=method,setting=key,alpha=alpha,tau=tau,index=index,id=sample['id'],round=step,
                            x=int(x),y=int(y),label=int(label),click_skipped=skipped,iou=mask_iou(prediction,sample['mask']),
                            statistics=segmentation_statistics(prediction,sample['mask']),contribution=attribution,
                            native_contribution=native_attribution,cumulative_counts=cumulative,incremental_counts=incremental,
                            positive_click=step>1 and label==1 and not skipped,array=path if not args.smoke else None)
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                        all_rows.append(row)
                        previous_prediction=prediction
                    # Each loop starts from identical initial state. No trajectory is shared.
                elapsed=time.monotonic()-start
                print(f'instance={index+1}/{len(indices)} protocol={method} rows={len(all_rows)} elapsed={elapsed:.1f}s',flush=True)
                dump(output/'status.json',dict(status='running',instance=index+1,protocol=method,rows=len(all_rows),elapsed_seconds=elapsed))
    comparison=[]
    round_summary=[]
    per_instance=[]
    attribution_round=[]
    attribution_instance=[]
    for method in ('replay','adaptive'):
        for alpha,tau in configs:
            key=setting_key(alpha,tau)
            selected=[r for r in all_rows if r['protocol']==method and r['setting']==key]
            context=dict(protocol=method,setting=key,alpha=alpha,tau=tau)
            comparison.append(dict(**context,**metric_summary(selected),**{'gate_'+k:v for k,v in contribution_summary(selected).items()}))
            for step in range(1,6):
                batch=[r for r in selected if r['round']==step]
                round_summary.append(dict(**context,round=step,IoU=float(np.mean([r['iou'] for r in batch])),
                    **transition_rates(pooled(batch,'cumulative_counts'))))
                attribution_round.append(dict(**context,round=step,**contribution_summary(batch)))
            for index in indices:
                batch=[r for r in selected if r['index']==index]
                per_instance.append(dict(**context,index=index,id=batch[0]['id'],**metric_summary(batch)))
                attribution_instance.append(dict(**context,index=index,id=batch[0]['id'],**contribution_summary(batch)))
    dump(output/'comparison.json',comparison)
    write_csv(output/'comparison.csv',comparison)
    write_csv(output/'round_summary.csv',round_summary)
    write_csv(output/'per_instance.csv',per_instance)
    write_csv(output/'contribution_by_round.csv',attribution_round)
    write_csv(output/'contribution_by_instance.csv',attribution_instance)
    write_csv(output/'positive_click_contributions.csv',[
        {**{k:r[k] for k in ('protocol','setting','alpha','tau','index','id','round','x','y')},**r['contribution'],
         **{'native_'+k:v for k,v in r['native_contribution'].items()}}
        for r in all_rows if r['positive_click']])
    original_hashes=json.loads((OUTPUT_ROOT/'directional_implementation_20261018/original_hashes.json').read_text())
    assert all(file_identity(p)['sha256']==sha for p,sha in original_hashes.items())
    assert all(file_identity(p)['sha256']==sha for p,sha in before.items())
    assert model_before==gate_digest(model)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    dump(output/'verification.json',dict(**checks,source_files_unchanged=True,watched_sha256=before,
        model_state_dict_sha256=model_before,model_state_dict_unchanged=True,no_optimizer=True,no_backward=True,
        old_a3_unchanged=len(original_hashes),thresholds_applied_without_GT=True))
    dump(output/'status.json',dict(status='smoke_passed' if args.smoke else 'completed',
        objects=len(indices),settings=len(configs),protocols=2,rows=len(all_rows),elapsed_seconds=time.monotonic()-start))
    print(json.dumps(dict(status='completed',rows=len(all_rows),checks=checks),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default=str(OUTPUT_ROOT/'positive_gate16_v2_seed42_20261019'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
