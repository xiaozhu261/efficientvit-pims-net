"""Frozen-S0 positive correction: train16, Gate-only checkpoints, replay/RITM."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.diagnose_large_clicks import describe
from pims_net.evaluate_directional_constraints import transition_counts, transition_rates
from pims_net.evaluation.experiment import file_identity
from pims_net.evaluation.metrics import mask_iou, segmentation_statistics
from pims_net.models.directional_large import LargeDirectionalPIMSNet
from pims_net.models.positive_correction_gate import PositiveCorrectionPIMSNet
from pims_net.overfit_directional import dump
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT, checkpoint_path
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.click_simulator import RITMClickSimulator
from pims_net.training.interactive_losses import prepare_target
from pims_net.training.positive_gate_losses import gate_target, positive_gate_losses


def gate_digest(gate):
    h = hashlib.sha256()
    for name, value in gate.state_dict().items():
        h.update(name.encode())
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def original_probability(model, state):
    return model.sam_initializer.postprocess(state.logits, state.image_sizes)[0].sigmoid()[0,0].detach().cpu().numpy()


def save_gate(path, model, optimizer, epoch, updates, config):
    payload = dict(format_version=1, variant='positive_correction_gate', epoch=epoch, global_step=updates,
                   gate_state_dict=model.gate.state_dict(), optimizer=optimizer.state_dict(), config=config)
    temp=path.with_suffix('.tmp')
    torch.save(payload,temp)
    temp.replace(path)


def load_gate(path, model, base_identity):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['variant']=='positive_correction_gate'
    assert payload['config']['frozen_s0']==base_identity
    model.gate.load_state_dict(payload['gate_state_dict'],strict=True)
    return payload


def region_distribution(state, target, valid, selected):
    if getattr(state,'gate',None) is None:
        return None
    base=state.base_probabilities.detach()[0,0].cpu().numpy()
    truth=target[0,0].cpu().numpy().astype(bool)
    v=valid[0,0].cpu().numpy().astype(bool)
    t=selected[0,0].cpu().numpy()
    g=state.gate.detach()[0,0].cpu().numpy()
    z=state.gate_logits.detach()[0,0].cpu().numpy()
    delta=(state.probabilities-state.base_probabilities).detach()[0,0].cpu().numpy()
    regions=dict(all=v,target=t,FN=v&truth&(base<=.5),FP=v&~truth&(base>.5),
                 TP=v&truth&(base>.5),TN=v&~truth&(base<=.5))
    return {name:dict(pixels=int(r.sum()),gate=describe(g[r]),logit=describe(z[r]),delta=describe(delta[r]),
        gate_above_05=float((g[r]>.5).mean()) if r.any() else None) for name,r in regions.items()}


def rollout(model, cached, index, variant, protocol='adaptive', replay=None, training=False,
            epoch=0, stream=None, arrays=None):
    sample,initial,target,valid,first=cached
    clicker=RITMClickSimulator(sample['mask'],np.random.default_rng(np.random.SeedSequence([42,index])))
    assert clicker.next_click(np.zeros_like(sample['mask'],dtype=bool))==first
    state=initial
    previous_prediction=first_prediction=None
    rows,terms=[],[]
    for step in range(1,6):
        losses=selected=metadata=None
        old=state
        if step==1:
            x,y,label=first
        else:
            click=tuple(replay[step-1][key] for key in ('x','y','label')) if protocol=='replay' else clicker.next_click(probability>.5)
            if click is None:
                # Keep a complete five-click metric trajectory if prediction is perfect.
                last=dict(rows[-1],round=step,click_skipped=True,losses=None)
                rows.append(last)
                if stream: stream.write(json.dumps(last,allow_nan=False)+'\n')
                continue
            x,y,label=click
            state=model.refine(state,[[[x,y]]],[[label]],enabled=variant!='G0')
            selected,metadata=gate_target(state.base_probabilities,target,valid,dict(x=x,y=y,label=label),sample['mask'].shape)
            if training and label==1:
                losses=positive_gate_losses(state,target,valid,selected,variant)
                terms.append(losses['loss'])
        with torch.no_grad():
            probability=original_probability(model,state)
            prediction=probability>.5
            if first_prediction is None: first_prediction=prediction.copy()
            counts=transition_counts(first_prediction,prediction,sample['mask'])
            incremental=transition_counts(previous_prediction,prediction,sample['mask']) if previous_prediction is not None else counts
            base_counts=None
            if step>1:
                base_state=replace(state,probabilities=state.base_probabilities,logits=state.base_logits)
                base_prediction=original_probability(model,base_state)>.5
                base_counts=transition_counts(base_prediction,prediction,sample['mask'])
            row=dict(index=index,id=sample['id'],variant=variant,protocol=protocol,epoch=epoch,round=step,
                x=int(x),y=int(y),label=int(label),click_skipped=False,iou=mask_iou(prediction,sample['mask']),
                statistics=segmentation_statistics(prediction,sample['mask']),
                iou_delta=mask_iou(prediction,sample['mask'])-rows[-1]['iou'] if rows else None,
                losses={k:float(v.detach()) for k,v in losses.items()} if losses else None,
                target_pixels=int(selected.sum()) if selected is not None else None,target_component=metadata,
                cumulative_counts=counts,incremental_counts=incremental,gate_only_counts=base_counts,
                gate_regions=region_distribution(state,target,valid,selected) if selected is not None else None)
            if stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
            if arrays is not None:
                np.savez_compressed(arrays/f'{variant}_{protocol}_{index:02d}_r{step}.npz',
                    probabilities=state.probabilities[0,0].cpu().numpy(),logits=state.logits[0,0].cpu().numpy(),
                    previous=old.probabilities[0,0].cpu().numpy(),target=target[0,0].cpu().numpy(),valid=valid[0,0].cpu().numpy(),
                    base=state.base_probabilities[0,0].cpu().numpy() if step>1 else state.probabilities[0,0].cpu().numpy(),
                    base_logits=state.base_logits[0,0].cpu().numpy() if step>1 else state.logits[0,0].cpu().numpy(),
                    gate=state.gate[0,0].cpu().numpy() if step>1 else np.zeros((256,256),np.float32),
                    gate_logits=state.gate_logits[0,0].cpu().numpy() if step>1 else np.zeros((256,256),np.float32),
                    selected=selected[0,0].cpu().numpy() if selected is not None else np.zeros((256,256),bool),
                    original_prediction=prediction,original_gt=sample['mask'],click=np.array([x,y,label]))
            rows.append(row)
            previous_prediction=prediction
    if training:
        loss=torch.stack(terms).mean() if terms else sum(p.sum()*0 for p in model.gate.parameters())
        loss.backward()
    return rows


def summarize(rows):
    result={f'IoU@{r}':float(np.mean([x['iou'] for x in rows if x['round']==r])) for r in (1,3,5)}
    final=[r for r in rows if r['round']==5]
    counts={k:sum(r['cumulative_counts'][k] for r in final) for k in final[0]['cumulative_counts']}
    result.update(transition_rates(counts))
    result['counts']=counts
    result['precision']=float(np.mean([r['statistics']['precision'] for r in final]))
    result['recall']=float(np.mean([r['statistics']['recall'] for r in final]))
    return result


def train_variant(model,cache,replay,output,variant,config,epochs):
    torch.manual_seed(42)
    model.gate.load_state_dict(config.pop('_initial_gate'),strict=True)
    directory=output/variant
    directory.mkdir()
    variant_config={**config,'variant':variant,'loss_weights':dict(gate=1.,bce=1. if variant=='G2' else 0.,dice=1. if variant=='G2' else 0.)}
    dump(directory/'config.json',variant_config)
    optimizer=torch.optim.AdamW(model.gate.parameters(),lr=1e-4,weight_decay=1e-4)
    parameters=list(model.gate.parameters())
    best=-1.
    updates=0
    start=time.monotonic()
    with (directory/'rounds.jsonl').open('w',buffering=1) as stream, (directory/'epochs.jsonl').open('w',buffering=1) as epochs_stream:
        for epoch in range(epochs+1):
            if epoch:
                model.train()
                order=np.random.default_rng(np.random.SeedSequence([42,epoch])).permutation(16)
                train_rows=[]
                norms=[]
                for batch in range(0,16,2):
                    optimizer.zero_grad(set_to_none=True)
                    for idx in order[batch:batch+2]:
                        train_rows.extend(rollout(model,cache[int(idx)],int(idx),variant,training=True,epoch=epoch,stream=stream))
                    for p in parameters:
                        if p.grad is not None: p.grad.div_(2)
                    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                    norms.append(float(norm))
                    optimizer.step()
                    updates+=1
                    assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
                train_summary=dict(phase='train',epoch=epoch,updates=updates,gradient_norm_mean=float(np.mean(norms)),**summarize(train_rows))
                losses=[r['losses'] for r in train_rows if r['losses'] is not None]
                train_summary['losses']={k:float(np.mean([l[k] for l in losses])) for k in losses[0]} if losses else None
                epochs_stream.write(json.dumps(train_summary)+'\n')
            if epoch%10==0 or epoch==epochs:
                model.eval()
                with torch.no_grad():
                    rows=[r for i,c in enumerate(cache) for r in rollout(model,c,i,variant,epoch=epoch)]
                summary=dict(phase='eval',epoch=epoch,updates=updates,**summarize(rows))
                epochs_stream.write(json.dumps(summary)+'\n')
                if summary['IoU@5']>best:
                    best=summary['IoU@5']
                    save_gate(directory/'best.pt',model,optimizer,epoch,updates,variant_config)
                print(f'{variant} epoch={epoch} updates={updates} IoU3={summary["IoU@3"]:.6f} IoU5={summary["IoU@5"]:.6f}',flush=True)
            save_gate(directory/'last.pt',model,optimizer,epoch,updates,variant_config)
            dump(directory/'status.json',dict(status='running',epoch=epoch,updates=updates,best_IoU5=best,elapsed_seconds=time.monotonic()-start))
        dump(directory/'status.json',dict(status='completed',epoch=epoch,updates=updates,best_IoU5=best,elapsed_seconds=time.monotonic()-start))


def run(args):
    output=Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Use a new output directory below pims_net/outputs')
    reference=Path(args.reference).resolve()
    watched=list((RESEARCH_ROOT/'models').glob('*.py'))+list((RESEARCH_ROOT/'training').glob('*.py'))
    watched+=[checkpoint_path()]+list(reference.parent.glob('S*/*.pt'))
    before={str(p):file_identity(p)['sha256'] for p in watched}
    output.mkdir(parents=True)
    manifest=output/'fixed_train16.jsonl'
    manifest.write_bytes((reference/'fixed_train16.jsonl').read_bytes())
    torch.set_num_threads(4)
    torch.manual_seed(42)
    base=LargeDirectionalPIMSNet().to(args.device).eval()
    source=load_training_checkpoint(reference/'best.pt',base)
    model=PositiveCorrectionPIMSNet(base).to(args.device)
    frozen_digest=gate_digest(model.base)
    initial={k:v.clone() for k,v in model.gate.state_dict().items()}
    base_identity=file_identity(reference/'best.pt')
    config=dict(run_type='train16_overfit_not_generalization',seed=42,manifest=file_identity(manifest),
        frozen_s0=base_identity,s0_epoch=source['epoch'],s0_updates=source['global_step'],
        epochs=args.epochs,optimizer_updates=args.epochs*8,batch_size=2,lr=1e-4,weight_decay=1e-4,grad_clip=1.,threshold=.5,
        gate_parameters=sum(p.numel() for p in model.gate.parameters()),initial_gate_sha256=gate_digest(model.gate),
        first_click='Frozen SAM unchanged; gate only rounds 2-5.',feedback='Corrected mask and S0 updated hidden feed next frozen transition.',
        gradient='S0 no_grad/eval/requires_grad=False; truncate through frozen transitions; mean losses over positive rounds per object; accumulate batch2.',
        input='Frozen image features256, latest positive/negative click Gaussian2 (sigma2), current S0 hidden128, S0 proposal probability1; all detached.',
        target='Native256 residual FN component containing current click AFTER S0 proposal; 8-connected, no snapping; all other valid pixels negative; empty component all negative.',
        gate_loss='Positive and negative regions separately mean BCEWithLogits, then mean present regions; positive clicks only.',
        G1='Gate BCE only',G2='Gate BCE + full valid image Mask BCE(logits) + soft Dice, weights1 each.',
        selection='best by adaptive training-set IoU5, eval every10; primary comparison LAST at full800 budget, best reported separately.',
        fixed_replay=file_identity(reference/'rounds.jsonl'),s0_frozen=True)
    dump(output/'protocol.json',config)
    del source
    records=[json.loads(x) for x in (reference/'rounds.jsonl').read_text().splitlines()]
    replay={i:sorted([r for r in records if r['phase']=='eval' and r['epoch']==config['s0_epoch'] and r['index']==i],key=lambda r:r['round']) for i in range(16)}
    assert all(len(v)==5 for v in replay.values())
    dump(output/'fixed_clicks.json',{i:[{k:r[k] for k in ('round','x','y','label','iou','id')} for r in rows] for i,rows in replay.items()})
    dataset=InteractiveDataset(manifest)
    cache=[]
    for index in range(16):
        sample=dataset[index]
        first=RITMClickSimulator(sample['mask'],np.random.default_rng(np.random.SeedSequence([42,index]))).next_click(np.zeros_like(sample['mask'],dtype=bool))
        x,y,label=first
        state=model.initialize(sample['image'],[[[x,y]]],[[label]])
        target,valid=prepare_target(sample['mask'],args.device)
        cache.append((sample,state,target,valid,first))
    checks=[]
    model.eval()
    for i,cached in enumerate(cache):
        with torch.no_grad():
            g0=rollout(model,cached,i,'G0',protocol='replay',replay=replay[i])
        for row,expected in zip(g0,replay[i]):
            assert abs(row['iou']-expected['iou'])<1e-12, (i,row['round'],row['iou'],expected['iou'])
        # Validate a real positive second-round transition even if S0 replay starts negative.
        sample,initial_state,target,valid,first=cached
        x,y,_=replay[i][1]['x'],replay[i][1]['y'],replay[i][1]['label']
        proposal=model.refine(initial_state,[[[x,y]]],[[1]])
        component,_=gate_target(proposal.base_probabilities,target,valid,dict(x=x,y=y,label=1),sample['mask'].shape)
        loss=positive_gate_losses(proposal,target,valid,component,'G2')['loss']
        model.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is None for p in model.base.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.gate.parameters())
        assert (proposal.probabilities>=proposal.base_probabilities).all()
        checks.append(dict(index=i,g0_exact_replay=True,frozen_base_no_gradient=True,single_positive_round_loss=float(loss),residual_fn_pixels=int(component.sum())))
    dump(output/'single_round_validation.json',dict(passed=True,objects=checks))
    print('Single-round real16 validation PASS; G0 replay matches all80 S0 IoUs exactly.',flush=True)
    if args.smoke:
        assert gate_digest(model.base)==frozen_digest
        dump(output/'status.json',dict(status='smoke_passed'))
        return
    for variant in ('G1','G2'):
        train_variant(model,cache,replay,output,variant,{**config,'_initial_gate':initial},args.epochs)
    evaluations=[]
    arrays=output/'native'
    arrays.mkdir()
    with (output/'evaluation_rounds.jsonl').open('w',buffering=1) as stream:
        for variant in ('G0','G1','G2'):
            for selection in (('last',) if variant=='G0' else ('last','best')):
                payload=None
                if variant!='G0': payload=load_gate(output/variant/(selection+'.pt'),model,base_identity)
                model.eval()
                for protocol in ('replay','adaptive'):
                    with torch.no_grad():
                        rows=[r for i,c in enumerate(cache) for r in rollout(model,c,i,variant,protocol=protocol,replay=replay[i],
                            stream=None,arrays=arrays if selection=='last' else None)]
                    for row in rows:
                        row['selection']=selection
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                    summary=dict(variant=variant,selection=selection,protocol=protocol,
                        epoch=payload['epoch'] if payload else config['s0_epoch'],updates=payload['global_step'] if payload else 0,**summarize(rows))
                    evaluations.append(summary)
                    print(json.dumps(summary),flush=True)
    dump(output/'comparison.json',evaluations)
    assert all(file_identity(p)['sha256']==sha for p,sha in before.items())
    assert gate_digest(model.base)==frozen_digest
    dump(output/'verification.json',dict(unchanged=True,watched_sha256=before,base_never_trained=True,
        frozen_base_state_dict_sha256=frozen_digest,gate_only_checkpoints=True))
    dump(output/'status.json',dict(status='completed',G1_updates=args.epochs*8,G2_updates=args.epochs*8))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default=str(OUTPUT_ROOT/'strength_supervision16_seed42_20261019/S0'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
