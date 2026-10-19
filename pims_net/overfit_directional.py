"""Fixed-16 COCO memorization experiment, uniformly thresholded at 0.5."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluation.experiment import file_identity
from pims_net.evaluation.metrics import mask_iou
from pims_net.models.directional_pims_net import DirectionalPIMSNet
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT, checkpoint_path
from pims_net.training.checkpoint import save_training_checkpoint
from pims_net.training.click_simulator import RITMClickSimulator
from pims_net.training.directional_losses import DirectionalLossWeights, directional_losses
from pims_net.training.interactive_losses import prepare_target


THRESHOLD = .5


def binary_for_clicker(probability):
    # Existing RITM internally compares to .49; boolean input makes the sole
    # probability decision here, at .5, without changing original A3 code.
    return np.asarray(probability) > THRESHOLD


def describe(values, low, high):
    values = np.asarray(values, np.float64)
    if not values.size:
        return None
    return dict(count=int(values.size), mean=float(values.mean()), std=float(values.std()),
                quantiles=np.quantile(values, [0, .01, .1, .5, .9, .99, 1]).tolist(),
                histogram=np.histogram(values, bins=20, range=(low, high))[0].tolist())


def distribution(direction, strength, old, truth, valid):
    old_fg = old > THRESHOLD
    regions = dict(all=valid, TP=valid & truth & old_fg, TN=valid & ~truth & ~old_fg,
                   FP=valid & ~truth & old_fg, FN=valid & truth & ~old_fg)
    return {name: dict(direction=describe(direction[selection], -1, 1), strength=describe(strength[selection], 0, 1),
                      direction_near_zero=float((np.abs(direction[selection]) <= .05).mean()) if selection.any() else None,
                      direction_positive=float((direction[selection] > 0).mean()) if selection.any() else None)
            for name, selection in regions.items()}


def gradient_pair(left, right):
    left = torch.cat([g.detach().reshape(-1) for g in left]).double()
    right = torch.cat([g.detach().reshape(-1) for g in right]).double()
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    lm, ld = float(left.norm()), float(right.norm())
    return dict(cosine=float(torch.dot(left, right)) / (lm * ld) if lm * ld else None,
                mask_norm=lm, direction_norm=ld, direction_to_mask_norm=ld / lm if lm else None)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def prepare_fixed_manifest(source, output, seed):
    source = Path(source).resolve()
    records = [json.loads(line) for line in source.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    indices = np.random.default_rng(seed).choice(len(records), 16, replace=False)
    selected = []
    for i in indices:
        record = dict(records[int(i)], source_index=int(i))
        for key in ('image', 'mask'):
            record[key] = str((source.parent / record[key]).resolve())
        selected.append(record)
    assert len({r['image'] for r in selected}) == 16
    path = output / 'fixed_train16.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in selected), encoding='utf-8')
    return path, selected


def rollout(model, cached, index, epoch, phase, seed, weights, parameters, names, captured, round_stream, sequence_stream):
    sample, initial, target, valid_tensor, first = cached
    truth, valid = target[0, 0].cpu().numpy().astype(bool), valid_tensor[0, 0].cpu().numpy().astype(bool)
    clicker = RITMClickSimulator(sample['mask'], np.random.default_rng(np.random.SeedSequence([seed, index])))
    assert clicker.next_click(np.zeros(sample['mask'].shape, bool)) == first
    state, terms, trace = initial, [], []
    original_prediction = None
    for step in range(1, 6):
        term = None
        stats = output_gradient = None
        if step == 1:
            x, y, label = first
        else:
            click = clicker.next_click(binary_for_clicker(probability))
            if click is None:
                break
            x, y, label = click
            previous = state.probabilities
            state = model.refine(state, [[[x, y]]], [[label]])
            raw = captured.pop()
            term = directional_losses(state, previous, target, valid_tensor, weights, threshold=THRESHOLD)
            terms.append(term)
            gm = torch.autograd.grad(weights.bce * term['bce'] + weights.dice * term['dice'], raw, retain_graph=True)[0]
            gd = torch.autograd.grad(weights.direction * term['direction'], raw, retain_graph=True)[0]
            output_gradient = gradient_pair([gm], [gd])
            stats = distribution(state.direction.detach()[0, 0].cpu().numpy(), state.strength.detach()[0, 0].cpu().numpy(),
                                 previous.detach()[0, 0].cpu().numpy(), truth, valid)
        with torch.no_grad():
            probability = model.sam_initializer.postprocess(state.logits, state.image_sizes)[0].sigmoid()[0, 0].cpu().numpy()
            native = state.probabilities.detach()[0, 0].cpu().numpy()
            prediction = binary_for_clicker(probability)
        row = dict(phase=phase, epoch=epoch, index=index, id=sample['id'], round=step, x=x, y=y, label=label,
                   iou=mask_iou(prediction, sample['mask']), native_iou=mask_iou((native > .5) & valid, truth & valid),
                   iou_delta=mask_iou(prediction, sample['mask']) - trace[-1]['iou'] if trace else None,
                   native_fg_band_049_051=float(((native[truth & valid] >= .49) & (native[truth & valid] <= .51)).mean()) if (truth & valid).any() else None,
                   correct_pixels_damaged=int(((original_prediction == sample['mask']) & (prediction != sample['mask'])).sum()) if original_prediction is not None else 0,
                   losses={k: float(v.detach()) for k, v in term.items()} if term else None,
                   output_gradient=output_gradient, distributions=stats)
        round_stream.write(json.dumps(row, allow_nan=False) + '\n')
        trace.append({k: row[k] for k in ('round', 'iou', 'native_iou')})
        original_prediction = prediction
    parameter_gradient = None
    loss = None
    if terms:
        mask = torch.stack([weights.bce * t['bce'] + weights.dice * t['dice'] for t in terms]).mean()
        direction = weights.direction * torch.stack([t['direction'] for t in terms]).mean()
        gm = torch.autograd.grad(mask, parameters, retain_graph=True)
        gd = torch.autograd.grad(direction, parameters, retain_graph=phase == 'train')
        parameter_gradient = {'all': gradient_pair(gm, gd)}
        for group, prefix in (('memory', 'spatial_memory.'), ('head', 'directional_decoder.head.')):
            selected = [i for i, name in enumerate(names) if name.startswith(prefix)]
            parameter_gradient[group] = gradient_pair([gm[i] for i in selected], [gd[i] for i in selected])
        loss = torch.stack([t['loss'] for t in terms]).mean()
        if phase == 'train':
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
    metrics = {key: float(torch.stack([t[key].detach() for t in terms]).mean()) if terms else 0.
               for key in ('loss', 'bce', 'dice', 'direction', 'preserve')}
    sequence = dict(phase=phase, epoch=epoch, index=index, id=sample['id'], actual_clicks=len(trace),
                    losses=metrics, parameter_gradient=parameter_gradient, trace=trace,
                    **{f'IoU@{t}': trace[min(t, len(trace)) - 1]['iou'] for t in (1, 3, 5)},
                    native_IoU5=trace[-1]['native_iou'])
    sequence_stream.write(json.dumps(sequence, allow_nan=False) + '\n')
    return sequence, bool(terms)


def epoch_summary(sequences, phase, epoch, updates, elapsed):
    cosine = [r['parameter_gradient']['all']['cosine'] for r in sequences if r['parameter_gradient'] is not None]
    cosine = [v for v in cosine if v is not None]
    result = dict(phase=phase, epoch=epoch, optimizer_updates=updates, elapsed_seconds=elapsed,
                  **{key: float(np.mean([r[key] for r in sequences])) for key in ('IoU@1', 'IoU@3', 'IoU@5', 'native_IoU5')},
                  losses={key: float(np.mean([r['losses'][key] for r in sequences])) for key in sequences[0]['losses']},
                  parameter_gradient_cosine_mean=float(np.mean(cosine)) if cosine else None,
                  parameter_gradient_conflict_fraction=float(np.mean(np.asarray(cosine) < 0)) if cosine else None)
    return result


def run(args, *, model_factory=DirectionalPIMSNet, fixed_manifest=None, experiment=None,
        extra_watched=None, stop_early=True):
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
                                                 parameters, names, captured, rounds, sequences)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(OUTPUT_ROOT / 'coco_subset_20261017/data/train.jsonl'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
