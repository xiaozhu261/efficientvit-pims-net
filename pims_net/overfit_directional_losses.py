"""Independent L0/L1/L2 training on fixed train16; unchanged large architecture."""

import argparse
import hashlib
import json
from dataclasses import asdict
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
from pims_net.overfit_directional import (THRESHOLD, binary_for_clicker, distribution, dump,
    epoch_summary, gradient_pair, prepare_fixed_manifest)
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT, checkpoint_path
from pims_net.training.checkpoint import save_training_checkpoint
from pims_net.training.click_simulator import RITMClickSimulator
from pims_net.training.directional_losses import DirectionalLossWeights
from pims_net.training.interactive_losses import prepare_target
from pims_net.training.local_directional_losses import local_directional_losses

PAIRS = (('mask_direction', 'mask', 'direction'), ('mask_preserve', 'mask', 'preserve'),
         ('direction_preserve', 'direction', 'preserve'))


def loss_gradients(losses, parameters, active_direction):
    gradients = {}
    for key, loss in losses.items():
        if key == 'direction' and not active_direction:
            continue
        raw = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        gradients[key] = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, raw))
    return gradients


def paired(gradients, selection=None):
    result = {}
    for name, left, right in PAIRS:
        if left not in gradients or right not in gradients:
            result[name] = None
            continue
        a, b = gradients[left], gradients[right]
        if selection is not None:
            a, b = [a[i] for i in selection], [b[i] for i in selection]
        pair = gradient_pair(a, b)
        result[name] = dict(cosine=pair['cosine'], left_norm=pair['mask_norm'], right_norm=pair['direction_norm'])
    return result


def summarize_pairs(sequences):
    result = {}
    for name, _, _ in PAIRS:
        values = [r['parameter_gradient_pairs']['all'][name]['cosine'] for r in sequences
                  if r['parameter_gradient_pairs'] is not None and r['parameter_gradient_pairs']['all'][name] is not None]
        values = [v for v in values if v is not None]
        result[name] = dict(valid_cosines=len(values), mean_cosine=float(np.mean(values)) if values else None,
                           conflict_fraction=float(np.mean(np.asarray(values)<0)) if values else None)
    return result


def rollout(model, cached, index, epoch, phase, seed, weights, parameters, names, captured,
            round_stream, sequence_stream, variant):
    sample, initial, target, valid_tensor, first = cached
    truth, valid = target[0,0].cpu().numpy().astype(bool), valid_tensor[0,0].cpu().numpy().astype(bool)
    clicker = RITMClickSimulator(sample['mask'], np.random.default_rng(np.random.SeedSequence([seed,index])))
    assert clicker.next_click(np.zeros(sample['mask'].shape,bool)) == first
    state, terms, trace = initial, [], []
    first_prediction = previous_prediction = None
    for step in range(1,6):
        term = stats = output_pairs = cc = None
        if step == 1:
            x,y,label = first
        else:
            click = clicker.next_click(binary_for_clicker(probability))
            if click is None:
                break
            x,y,label = click
            previous = state.probabilities
            component, cc = clicked_error_component(previous.detach()[0,0].cpu().numpy(),truth,valid,
                dict(x=x,y=y,label=label),sample['mask'].shape)
            gate = torch.as_tensor(component,device=target.device)[None,None]
            state = model.refine(state,[[[x,y]]],[[label]])
            raw = captured.pop()
            term = local_directional_losses(state,previous,target,valid_tensor,gate,label,variant,weights)
            terms.append(term)
            objectives = dict(mask=weights.bce*term['bce']+weights.dice*term['dice'],
                direction=weights.direction*term['direction'],preserve=weights.preserve*term['preserve'])
            output_pairs = paired(loss_gradients(objectives,[raw],variant!='L1'))
            stats = distribution(state.direction.detach()[0,0].cpu().numpy(),state.strength.detach()[0,0].cpu().numpy(),
                                 previous.detach()[0,0].cpu().numpy(),truth,valid)
        with torch.no_grad():
            probability = model.sam_initializer.postprocess(state.logits,state.image_sizes)[0].sigmoid()[0,0].cpu().numpy()
            native = state.probabilities.detach()[0,0].cpu().numpy()
            prediction = binary_for_clicker(probability)
        if first_prediction is None:
            first_prediction = prediction.copy()
        cumulative = transition_counts(first_prediction,prediction,sample['mask'])
        incremental = transition_counts(previous_prediction,prediction,sample['mask']) if previous_prediction is not None else cumulative
        row = dict(phase=phase,epoch=epoch,index=index,id=sample['id'],round=step,x=x,y=y,label=label,
            iou=mask_iou(prediction,sample['mask']),native_iou=mask_iou((native>.5)&valid,truth&valid),
            iou_delta=mask_iou(prediction,sample['mask'])-trace[-1]['iou'] if trace else None,
            losses={k:float(v.detach()) for k,v in term.items()} if term else None,
            output_gradient_pairs=output_pairs,distributions=stats,clicked_component=cc,
            cumulative_counts=cumulative,cumulative_rates=transition_rates(cumulative),
            incremental_counts=incremental,incremental_rates=transition_rates(incremental))
        round_stream.write(json.dumps(row,allow_nan=False)+'\n')
        trace.append({k:row[k] for k in ('round','iou','native_iou')})
        previous_prediction=prediction
    parameter_pairs = legacy_gradient = None
    if terms:
        objectives=dict(mask=torch.stack([weights.bce*t['bce']+weights.dice*t['dice'] for t in terms]).mean(),
            direction=weights.direction*torch.stack([t['direction'] for t in terms]).mean(),
            preserve=weights.preserve*torch.stack([t['preserve'] for t in terms]).mean())
        gradients=loss_gradients(objectives,parameters,variant!='L1')
        parameter_pairs={'all':paired(gradients)}
        for group,prefix in (('memory','spatial_memory.'),('head','directional_decoder.head.')):
            selected=[i for i,name in enumerate(names) if name.startswith(prefix)]
            parameter_pairs[group]=paired(gradients,selected)
        legacy_gradient={group:pairs['mask_direction'] for group,pairs in parameter_pairs.items()} if variant!='L1' else None
        del gradients
        loss=torch.stack([t['loss'] for t in terms]).mean()
        if phase=='train':
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
    metrics={key:float(torch.stack([t[key].detach() for t in terms]).mean()) if terms else 0.
             for key in ('loss','bce','dice','direction','preserve')}
    sequence=dict(phase=phase,epoch=epoch,index=index,id=sample['id'],actual_clicks=len(trace),losses=metrics,
        parameter_gradient=legacy_gradient,parameter_gradient_pairs=parameter_pairs,trace=trace,
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
    experiment = dict(name='large_loss_ablation', loss_variant=args.variant,
        reference=str(reference), initialization='Fresh identical seed42 model; no checkpoint warm start.',
        component='Current-click same-sign 8-connected GT error, detached old P>0.5, native256; empty stays empty.',
        L0='Original global BCE+Dice+class-balanced Direction+0.1 Preserve on GT-correct pixels.',
        L1='BCE+Dice mean over clicked error component; 0.1 Preserve over valid pixels outside it; no Direction loss.',
        L2='L1 plus mean SmoothL1 Direction inside component, target click sign; weight1.',
        inference='Unmodified full model; no GT gate or click constraint at inference.')
    extra_watched = list((RESEARCH_ROOT/'models').glob('*.py')) + list((RESEARCH_ROOT/'training').glob('*.py'))
    extra_watched += [reference/name for name in ('best.pt','last.pt','config.json','fixed_train16.jsonl')]
    extra_watched += [RESEARCH_ROOT/'overfit_directional.py']
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
    config['loss_weights']['direction'] = 0. if args.variant == 'L1' else 1.
    config['gradient_protocol'] = 'Weighted Mask/Direction/Preserve pairwise gradients; per-step own raw head and full-BPTT per-sequence parameters. Zero norm or absent loss has null cosine; excludes from conflict denominator.'
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


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default=str(OUTPUT_ROOT/'directional_large_overfit16_seed42_20261019'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--variant',choices=('L0','L1','L2'),required=True)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--eval-every',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda')
    run_variant(parser.parse_args())


