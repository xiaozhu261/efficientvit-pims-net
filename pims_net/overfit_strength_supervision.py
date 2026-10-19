"""Independent S0/S1/S2 train16 experiment on unchanged LargeDirectionalPIMSNet."""

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluate_oracle import clicked_error_component
from pims_net.evaluate_directional_constraints import transition_counts, transition_rates
from pims_net.evaluation.experiment import file_identity
from pims_net.evaluation.metrics import mask_iou
from pims_net.models.directional_large import LargeDirectionalPIMSNet
from pims_net.overfit_directional import (binary_for_clicker, distribution, dump, epoch_summary, prepare_fixed_manifest)
from pims_net.diagnose_large_clicks import describe
from pims_net.diagnose_strength_gradients import parameter_vectors
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT, checkpoint_path
from pims_net.training.checkpoint import save_training_checkpoint
from pims_net.training.click_simulator import RITMClickSimulator
from pims_net.training.directional_losses import DirectionalLossWeights
from pims_net.training.interactive_losses import prepare_target
from pims_net.training.strength_supervised_losses import strength_supervised_losses, harmful_strength_region

OBJECTIVES=('mask','direction','preserve','strength_activate','strength_suppress')
PAIRS=[(a+'__'+b,a,b) for a,b in itertools.combinations(OBJECTIVES,2)]
LOSS_KEYS=('loss','bce','dice','direction','preserve','strength_activate','strength_suppress')


def objectives(terms,weights):
    return dict(mask=weights.bce*terms['bce']+weights.dice*terms['dice'],
        direction=weights.direction*terms['direction'],preserve=weights.preserve*terms['preserve'],
        strength_activate=.05*terms['strength_activate'],strength_suppress=.01*terms['strength_suppress'])


def get_gradients(losses,parameters,active):
    result={}
    for key,loss in losses.items():
        grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) if active[key] else [None]*len(parameters)
        result[key]=[torch.zeros_like(p) if g is None else g for p,g in zip(parameters,grads)]
    return result


def vector_report(vectors):
    vectors=dict(vectors,total=sum(vectors.values()))
    keys=list(vectors)
    matrix=torch.stack([vectors[k].detach().reshape(-1).double() for k in keys])
    gram=(matrix@matrix.T).cpu().numpy()
    assert np.isfinite(gram).all()
    norms={key:float(np.sqrt(max(0.,gram[i,i]))) for i,key in enumerate(keys)}
    pairs={}
    for i,a in enumerate(keys[:-1]):
        for j in range(i+1,len(keys)-1):
            b=keys[j]
            denominator=norms[a]*norms[b]
            pairs[a+'__'+b]=dict(cosine=float(gram[i,j]/denominator) if denominator else None,
                left_norm=norms[a],right_norm=norms[b])
    return dict(norms=norms,
        projections={key:float(gram[i,-1]/gram[-1,-1]) if gram[-1,-1] else None for i,key in enumerate(keys)},pairs=pairs)


def parameter_report(grads,names):
    report={}
    for group in ('all','shared','strength_head','memory','head'):
        if group=='all':
            vectors={k:torch.cat([g.reshape(-1) for g in v]) for k,v in grads.items()}
        elif group=='head':
            vectors={k:torch.cat([g.reshape(-1) for n,g in zip(names,v) if n.startswith('directional_decoder.head.')]) for k,v in grads.items()}
        else:
            vectors={k:parameter_vectors(v,names,group) for k,v in grads.items()}
        report[group]=vector_report(vectors)
    return report


def summarize_pairs(sequences):
    result={}
    for name,_,_ in PAIRS:
        values=[r['parameter_gradient_details']['all']['pairs'][name]['cosine'] for r in sequences if r['parameter_gradient_details']]
        good=[v for v in values if v is not None]
        result[name]=dict(valid_cosines=len(good),mean_cosine=float(np.mean(good)) if good else None,
            conflict_fraction=float(np.mean(np.asarray(good)<0)) if good else None)
    return result


def summarize_norms(sequences):
    result={}
    for group in ('all','shared','strength_head','memory','head'):
        data=[r['parameter_gradient_details'][group] for r in sequences if r['parameter_gradient_details']]
        result[group]={key:float(np.mean([r['norms'][key] for r in data])) if data else None for key in (*OBJECTIVES,'total')}
    return result


def diagnostic_regions(state,previous,raw,truth,valid,component,cc,output_grads):
    s=state.strength.detach()[0,0].cpu().numpy()
    d=state.direction.detach()[0,0].cpu().numpy()
    z=raw.detach()[0,1].cpu().numpy()
    old=previous.detach()[0,0].cpu().numpy()
    yy,xx=np.mgrid[:256,:256]
    regions=dict(target=component,outside_target=valid&~component,correct=valid&((old>.5)==truth),
        FN=valid&truth&(old<=.5),FP=valid&~truth&(old>.5),all=valid,
        harmful=valid&((truth&(old>.5)&(d<0))|(~truth&(old<=.5)&(d>0))),
        click_disk=valid&((xx-cc['click_native_x'])**2+(yy-cc['click_native_y'])**2<=64))
    data={}
    for name,mask in regions.items():
        data[name]=dict(pixels=int(mask.sum()),strength=describe(s[mask]),logit=describe(z[mask]),direction=describe(d[mask]),
            sigmoid_gate_small=float((s[mask]*(1-s[mask])<1e-4).mean()) if mask.any() else None,
            exact_endpoint=float(((s[mask]==0)|(s[mask]==1)).mean()) if mask.any() else None,
            gz={key:dict(mean=float(g[0].detach()[0,1].cpu().numpy()[mask].mean(dtype=np.float64)),
                         mean_abs=float(np.abs(g[0].detach()[0,1].cpu().numpy()[mask]).mean(dtype=np.float64))) if mask.any() else None
                for key,g in output_grads.items()})
    return data


def rollout(model,cached,index,epoch,phase,seed,weights,parameters,names,captured,round_stream,sequence_stream,variant):
    sample,initial,target,valid_tensor,first=cached
    truth,valid=target[0,0].cpu().numpy().astype(bool),valid_tensor[0,0].cpu().numpy().astype(bool)
    clicker=RITMClickSimulator(sample['mask'],np.random.default_rng(np.random.SeedSequence([seed,index])))
    assert clicker.next_click(np.zeros(sample['mask'].shape,bool))==first
    state,terms,trace,activity=initial,[],[],[]
    first_prediction=previous_prediction=None
    for step in range(1,6):
        term=stats=extra=outputs=cc=None
        support=None
        if step==1:
            x,y,label=first
        else:
            click=clicker.next_click(binary_for_clicker(probability))
            if click is None:
                break
            x,y,label=click
            previous=state.probabilities
            component,cc=clicked_error_component(previous.detach()[0,0].cpu().numpy(),truth,valid,
                dict(x=x,y=y,label=label),sample['mask'].shape)
            gate=torch.as_tensor(component,device=target.device)[None,None]
            state=model.refine(state,[[[x,y]]],[[label]])
            raw=captured.pop()
            term=strength_supervised_losses(state,previous,target,valid_tensor,gate,label,raw,variant,weights)
            terms.append(term)
            harmful=harmful_strength_region(previous,target,valid_tensor,state.direction)
            support=dict(activation_pixels=int(component.sum()) if label==1 and variant!='S0' else 0,
                         suppression_pixels=int(harmful.sum()) if variant=='S2' else 0,
                         harmful_pixels=int(harmful.sum()))
            active=dict(mask=True,direction=True,preserve=True,strength_activate=support['activation_pixels']>0,
                        strength_suppress=support['suppression_pixels']>0)
            activity.append(active)
            output_grads=get_gradients(objectives(term,weights),[raw],active)
            outputs={group:vector_report({k:(v[0][:,1:2] if group=='strength_logit' else v[0]).reshape(-1) for k,v in output_grads.items()})
                     for group in ('raw_head','strength_logit')}
            stats=distribution(state.direction.detach()[0,0].cpu().numpy(),state.strength.detach()[0,0].cpu().numpy(),
                previous.detach()[0,0].cpu().numpy(),truth,valid)
            if phase=='eval':
                extra=diagnostic_regions(state,previous,raw,truth,valid,component,cc,output_grads)
        with torch.no_grad():
            probability=model.sam_initializer.postprocess(state.logits,state.image_sizes)[0].sigmoid()[0,0].cpu().numpy()
            native=state.probabilities.detach()[0,0].cpu().numpy()
            prediction=binary_for_clicker(probability)
        if first_prediction is None:
            first_prediction=prediction.copy()
        cumulative=transition_counts(first_prediction,prediction,sample['mask'])
        incremental=transition_counts(previous_prediction,prediction,sample['mask']) if previous_prediction is not None else cumulative
        row=dict(phase=phase,epoch=epoch,index=index,id=sample['id'],round=step,x=x,y=y,label=label,
            iou=mask_iou(prediction,sample['mask']),native_iou=mask_iou((native>.5)&valid,truth&valid),
            iou_delta=mask_iou(prediction,sample['mask'])-trace[-1]['iou'] if trace else None,
            losses={k:float(v.detach()) for k,v in term.items()} if term else None,
            output_gradient_details=outputs,distributions=stats,strength_regions=extra,clicked_component=cc,support=support,
            cumulative_counts=cumulative,cumulative_rates=transition_rates(cumulative),
            incremental_counts=incremental,incremental_rates=transition_rates(incremental))
        round_stream.write(json.dumps(row,allow_nan=False)+'\n')
        trace.append({k:row[k] for k in ('round','iou','native_iou')})
        previous_prediction=prediction
    details=legacy=None
    if terms:
        averaged={key:torch.stack([objectives(t,weights)[key] for t in terms]).mean() for key in OBJECTIVES}
        active={key:any(a[key] for a in activity) for key in OBJECTIVES}
        grads=get_gradients(averaged,parameters,active)
        details=parameter_report(grads,names)
        legacy={group:details[group]['pairs']['mask__direction'] for group in ('all','memory','head')}
        del grads
        loss=torch.stack([t['loss'] for t in terms]).mean()
        if phase=='train':
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
    metrics={key:float(torch.stack([t[key].detach() for t in terms]).mean()) if terms else 0. for key in LOSS_KEYS}
    sequence=dict(phase=phase,epoch=epoch,index=index,id=sample['id'],actual_clicks=len(trace),losses=metrics,
        parameter_gradient=legacy,parameter_gradient_details=details,trace=trace,
        **{f'IoU@{t}':trace[min(t,len(trace))-1]['iou'] for t in (1,3,5)},native_IoU5=trace[-1]['native_iou'])
    sequence_stream.write(json.dumps(sequence,allow_nan=False)+'\n')
    return sequence,bool(terms)


def run_variant(args):
    reference = Path(args.reference).resolve()
    fixed_manifest = reference / 'fixed_train16.jsonl'
    source_config = json.loads((reference / 'config.json').read_text())
    args.manifest = source_config['source_manifest']['path']
    model_factory = LargeDirectionalPIMSNet
    stop_early = False
    experiment = dict(name='Strength_BCE_ablation', strength_variant=args.variant,
        reference=str(reference), initialization='Fresh identical seed42 model; no warm start.',
        S0='Unmodified original L0: global BCE+Dice+class-balanced Direction+0.1 GT-correct Preserve.',
        S1='S0 plus 0.05 BCEWithLogits(z_strength,1) only in positive-click FN component.',
        S2='S1 plus 0.01 BCEWithLogits(z_strength,0) only on harmful updates within old GT-correct pixels: TP & D<0 or TN & D>0, for both click signs.',
        component='Detached old P>0.5 same-sign error 8-connected component at mapped latest click; no snapping; empty skips activation.',
        suppression='All selection conditions detached. Never suppress unrepaired FN/FP. Empty harmful region skipped.',
        normalization='Activation and suppression separately mean over their own selected pixels.',
        inference='Unmodified original model, no GT gates or sign/Gaussian constraints.')
    extra_watched = list((RESEARCH_ROOT/'models').glob('*.py')) + list((RESEARCH_ROOT/'training').glob('*.py'))
    extra_watched += [reference/name for name in ('best.pt','last.pt','config.json','fixed_train16.jsonl')]
    extra_watched += [OUTPUT_ROOT/'large_loss_ablation16_seed42_20261019'/v/name for v in ('L0','L1','L2') for name in ('best.pt','last.pt')]
    extra_watched += [RESEARCH_ROOT/'overfit_directional.py',RESEARCH_ROOT/'overfit_directional_losses.py']
    output = Path(args.output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve() or output.exists():
        raise ValueError('Use a new run directory below pims_net/outputs')
    if args.epochs < 1 or args.eval_every < 1:
        raise ValueError('Positive epoch/evaluation interval required')
    output.mkdir(parents=True)
    if fixed_manifest is None:
        manifest, selected = prepare_fixed_manifest(args.manifest, output, args.seed)
    else:
        manifest = output / 'fixed_train16.jsonl'
        manifest.write_bytes(Path(fixed_manifest).read_bytes())
        selected = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        assert len(selected) == 16 and len({r['image'] for r in selected}) == 16
    original = json.loads((OUTPUT_ROOT / 'directional_implementation_20261018/original_hashes.json').read_text())
    for path, sha in original.items():
        assert file_identity(path)['sha256'] == sha, f'Original A3 source/weights changed: {path}'
    watched = {str(checkpoint_path()): file_identity(checkpoint_path())['sha256'], **original}
    watched.update({str(path): file_identity(path)['sha256'] for path in (extra_watched or [])})
    weights = DirectionalLossWeights()
    config = dict(variant='directional', run_type='train16_overfit_not_generalization', manifest=str(manifest),
                  source_manifest=file_identity(args.manifest), fixed_manifest=file_identity(manifest), seed=args.seed,
                  epochs=args.epochs, batch_size=2, max_clicks=5, lr=1e-4, weight_decay=1e-4, grad_clip=1.,
                  threshold=.5, threshold_applies_to=['click_selection', 'direction_target', 'preserve_region', 'original_iou', 'native_iou'],
                  loss_weights=asdict(weights), eval_every=args.eval_every, sam_frozen=True,
                  click_protocol='RITM half-max-distance sampled errors, binary P>0.5 input; same object RNG reset every rollout.',
                  gradient_protocol='Every round: own weighted Mask vs Direction gradient on 2-channel raw head logits. Every object: mean-round full-BPTT parameter gradients; before clip/AdamW.',
                  distributions='Native valid pixels; all/TP/TN/FP/FN by old P>0.5. Quantiles 0/.01/.1/.5/.9/.99/1; 20-bin D[-1,1],S[0,1] histograms.',
                  initialization='Fresh research weights, seed42; frozen official SAM only.',
                  stop='Maximum epochs, or original mean IoU@5 >=0.95 at three consecutive fixed-weight evaluations.',
                  evaluation='Same 16 training instances and fixed per-object RNG, adaptive later clicks. No heldout claim.',
                  baseline_sha256=watched[str(checkpoint_path())], device=args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model = model_factory().to(args.device).train()
    config['variant'] = model.variant
    config['loss_variant'] = args.variant
    config['strength_loss_weights'] = dict(activate=.05 if args.variant!='S0' else 0., suppress=.01 if args.variant=='S2' else 0.)
    config['gradient_protocol'] = 'Every round weighted objective raw-head gradients; every sequence full-BPTT gradients for Mask/Direction/Preserve/Strength_activate/Strength_suppress on all/shared/StrengthHead/memory/head. Record norms, total projections, pairwise cosine. Zero norms excluded.'
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    config['initial_research_sha256'] = digest.hexdigest()
    if experiment:
        config['experiment'] = experiment
    if not stop_early:
        config['stop'] = 'Fixed full epoch budget; no early stopping.'
    config['parameter_counts'] = dict(decoder=sum(p.numel() for p in model.directional_decoder.parameters()),
                                    trainable=sum(p.numel() for p in model.trainable_parameters()))
    dump(output / 'config.json', config)
    dump(output / 'status.json', dict(status='running', epoch=0, optimizer_updates=0))
    parameters = list(model.trainable_parameters())
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    dataset = InteractiveDataset(manifest)
    cache = []
    for index in range(16):
        sample = dataset[index]
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
        first = RITMClickSimulator(sample['mask'], rng).next_click(np.zeros(sample['mask'].shape, bool))
        x, y, label = first
        initial = model.initialize(sample['image'], [[[x, y]]], [[label]])
        target, valid = prepare_target(sample['mask'], args.device)
        cache.append((sample, initial, target, valid, first))
    captured = []
    handle = model.directional_decoder.head.register_forward_hook(lambda module, inputs, result: captured.append(result))
    start, updates, best, streak = time.monotonic(), 0, -1., 0
    summaries = []
    with (output / 'rounds.jsonl').open('w', encoding='utf-8', buffering=1) as rounds, \
         (output / 'sequences.jsonl').open('w', encoding='utf-8', buffering=1) as sequences, \
         (output / 'epochs.jsonl').open('w', encoding='utf-8', buffering=1) as epochs:
        for epoch in range(args.epochs + 1):
            phases = (['train'] if epoch else []) + (['eval'] if epoch % args.eval_every == 0 or epoch == args.epochs else [])
            for phase in phases:
                model.train(phase == 'train')
                order = np.random.default_rng(np.random.SeedSequence([args.seed, epoch])).permutation(16) if phase == 'train' else np.arange(16)
                results = []
                for batch in range(0, 16, 2):
                    optimizer.zero_grad(set_to_none=True)
                    trained = 0
                    for index in order[batch:batch + 2]:
                        row, trainable = rollout(model, cache[int(index)], int(index), epoch, phase, args.seed, weights,
                                                 parameters, names, captured, rounds, sequences, args.variant)
                        results.append(row)
                        trained += int(trainable)
                    if phase == 'train' and trained:
                        for p in parameters:
                            if p.grad is not None:
                                p.grad.div_(trained)
                        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                        optimizer.step()
                        updates += 1
                    assert all(p.grad is None and not p.requires_grad for p in model.sam_initializer.parameters())
                summary = epoch_summary(results, phase, epoch, updates, time.monotonic() - start)
                summary['gradient_pairs'] = summarize_pairs(results)
                summary['gradient_norms'] = summarize_norms(results)
                summaries.append(summary)
                epochs.write(json.dumps(summary, allow_nan=False) + '\n')
                print(f'{phase} epoch={epoch} updates={updates} IoU3={summary["IoU@3"]:.4f} IoU5={summary["IoU@5"]:.4f} loss={summary["losses"]["loss"]:.4f} cosine={summary["parameter_gradient_cosine_mean"]}', flush=True)
                if phase == 'eval':
                    config['optimizer_updates'] = updates
                    if summary['IoU@5'] > best:
                        best = summary['IoU@5']
                        save_training_checkpoint(output / 'best.pt', model, optimizer, epoch, updates, config)
                    streak = streak + 1 if summary['IoU@5'] >= .95 else 0
            config['optimizer_updates'] = updates
            save_training_checkpoint(output / 'last.pt', model, optimizer, epoch, updates, config)
            dump(output / 'status.json', dict(status='running', epoch=epoch, optimizer_updates=updates,
                best_training_set_IoU5=best, elapsed_seconds=time.monotonic() - start))
            if stop_early and streak >= 3:
                break
    handle.remove()
    assert not captured
    for path, sha in watched.items():
        assert file_identity(path)['sha256'] == sha, path
    outcome = dict(status='completed', epochs=epoch, optimizer_updates=updates, best_training_set_IoU5=best,
                   achieved_095_three_evaluations=streak >= 3, a3_and_baseline_unchanged=True,
                   elapsed_seconds=time.monotonic() - start, run_type=config['run_type'])
    dump(output / 'status.json', outcome)
    dump(output / 'verification.json', dict(original_hashes=watched, unchanged=True, fixed_instances=selected, threshold=.5))
    dump(output / 'summary.json', summaries)
    print(json.dumps(outcome, indent=2), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default=str(OUTPUT_ROOT/'directional_large_overfit16_seed42_20261019'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--variant',choices=('S0','S1','S2'),required=True)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--eval-every',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda')
    run_variant(parser.parse_args())


