"""Read-only supervision and full-BPTT gradient audit on saved heldout trajectories.

No optimizer, parameter mutation, training-mode switch, or model monkey-patching.
The head hook only observes raw intent logits. Gradients are obtained by autograd.grad.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluation.experiment import file_identity
from pims_net.models import PIMSNet
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.interactive_losses import LossWeights, interactive_losses, prepare_target


CLASSES = ('ADD', 'REMOVE', 'KEEP')


def labels_from_old(old, truth, threshold=.49):
    foreground = old > threshold
    return np.where(truth & ~foreground, 0, np.where(~truth & foreground, 1, 2))


def click_regions(clicks, image_shape, radius=8):
    """Disks in native 256 canvas; SAM's rounded 1024 prompt coordinates / 4.

    Radius 8 is approximately one sigma of the 64-grid click Gaussian. Historical
    same-sign disks are unioned; positive/negative regions may overlap each other.
    The nearest pixel is reported separately; it need not retain the original GT.
    """
    h, w = image_shape
    rh, rw = int(h * 1024 / max(h, w) + .5), int(w * 1024 / max(h, w) + .5)
    yy, xx = np.mgrid[:256, :256]
    result = {key: np.zeros((256, 256), bool) for key in
              ('latest_positive', 'latest_negative', 'history_positive', 'history_negative',
               'center_positive', 'center_negative')}
    for i, click in enumerate(clicks):
        x, y = click['x'] * rw / w / 4, click['y'] * rh / h / 4
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        sign = 'positive' if click['label'] else 'negative'
        result[f'history_{sign}'] |= disk
        if i == len(clicks) - 1:
            result[f'latest_{sign}'] = disk
            result[f'center_{sign}'][min(255, int(y + .5)), min(255, int(x + .5))] = True
    return result


def pair_statistics(left, right):
    left, right = np.asarray(left, np.float64).ravel(), np.asarray(right, np.float64).ravel()
    dot = float(left @ right)
    nl, nr = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    return dict(dot=dot, mask_norm=nl, edit_norm=nr, cosine=dot / (nl * nr) if nl * nr > 0 else None,
                edit_to_mask_norm=nr / nl if nl else None)


def local_statistics(q, labels, alternative, old, truth, selected, mask_grad, edit_grad):
    n = int(selected.sum())
    if not n:
        return None
    values = q[:, selected].astype(np.float64)
    actual = labels[selected]
    prediction = values.argmax(0)
    gm, ge = mask_grad[:, selected].astype(np.float64), edit_grad[:, selected].astype(np.float64)
    dot = (gm * ge).sum(0)
    nm, ne = np.linalg.norm(gm, axis=0), np.linalg.norm(ge, axis=0)
    nonzero = nm * ne > 0
    cosine = dot[nonzero] / (nm[nonzero] * ne[nonzero])
    return dict(pixels=n, q_sum=values.sum(1).tolist(),
                confusion=np.bincount(3 * actual + prediction, minlength=9).reshape(3, 3).tolist(),
                threshold_label_mismatch=int((actual != alternative[selected]).sum()),
                gt_foreground=int(truth[selected].sum()), old_probability_sum=float(old[selected].sum(dtype=np.float64)),
                gradient_dot=float(dot.sum()), mask_gradient_sq=float((gm * gm).sum()),
                edit_gradient_sq=float((ge * ge).sum()), gradient_nonzero=int(nonzero.sum()),
                gradient_conflict=int((cosine < -1e-8).sum()), gradient_cosine_sum=float(cosine.sum()))


def combine_local(rows):
    result = {}
    for key in rows[0]:
        result[key] = np.asarray([row[key] for row in rows]).sum(axis=0).tolist()
    n, c = result['pixels'], np.asarray(result['confusion'])
    result.update(q_mean=(np.asarray(result['q_sum']) / n).tolist(),
                  true_distribution=(c.sum(1) / n).tolist(), predicted_distribution=(c.sum(0) / n).tolist(),
                  confusion_row_normalized=np.divide(c, c.sum(1, keepdims=True),
                      out=np.zeros((3, 3), float), where=c.sum(1, keepdims=True) != 0).tolist(),
                  threshold_mismatch_fraction=result['threshold_label_mismatch'] / n,
                  gradient_conflict_fraction=result['gradient_conflict'] / result['gradient_nonzero']
                      if result['gradient_nonzero'] else None,
                  pooled_gradient_cosine=result['gradient_dot'] /
                      (result['mask_gradient_sq'] * result['edit_gradient_sq']) ** .5
                      if result['mask_gradient_sq'] * result['edit_gradient_sq'] else None)
    return result


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def parameter_groups(names):
    return {'all': list(range(len(names))),
            'memory': [i for i, n in enumerate(names) if n.startswith('spatial_memory.')],
            'decoder_trunk': [i for i, n in enumerate(names) if n.startswith('edit_intent_decoder.') and '.head.' not in n],
            'head': [i for i, n in enumerate(names) if n.startswith('edit_intent_decoder.head.')]}


def run(args):
    reference, output = Path(args.reference).resolve(), Path(args.output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError('Output must be a new subdirectory of pims_net/outputs')
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((reference / 'protocol.json').read_text(encoding='utf-8'))
    dataset = InteractiveDataset(protocol['manifest']['path'])
    references = [json.loads(line) for line in (reference / 'per_object.jsonl').read_text(encoding='utf-8').splitlines()]
    watched = [Path(x['path']) for x in (protocol['manifest'], protocol['baseline'], *protocol['checkpoints'].values())]
    watched += sorted((RESEARCH_ROOT / 'models').glob('*.py')) + sorted((RESEARCH_ROOT / 'training').glob('*.py'))
    before = {str(path): file_identity(path)['sha256'] for path in watched}
    for identity in (protocol['manifest'], protocol['baseline'], *protocol['checkpoints'].values()):
        assert file_identity(identity['path']) == identity
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    if count < 1:
        raise ValueError('Need at least one object')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    output.mkdir(parents=True)
    dump(output / 'protocol.json', dict(reference=str(reference), checkpoint_identities=protocol['checkpoints'],
        manifest=protocol['manifest'], baseline=protocol['baseline'], objects_per_variant=count, variants=['a2', 'a3'],
        labels=list(CLASSES), neighborhood_radius_native_pixels=8, neighborhood_geometry=click_regions.__doc__,
        true_label_threshold=.49, alternative_label_threshold=.5,
        gradient_scope='Mean of rounds 2-5, full BPTT, no detach of hidden or masks; per-object and mean parameter gradients.',
        output_gradient_scope='Each round own loss vs current raw 3-channel intent logits, not future loss.',
        trajectory='Each variant replays its own saved evaluation clicks; no new sampling. Same heldout objects.',
        limitation='Current checkpoint heldout diagnostic, not recorded historical training gradients. Training random clicks use 0.49; saved evaluation uses deterministic 0.5.',
        aggregation='Local regions: pooled valid pixel occurrences; parameter gradients: per object plus mean gradient over objects.',
        optimization='No optimizer; autograd.grad only; unscaled float32, before clipping/AdamW.', device=args.device))
    local_rows, gradient_rows, loss_rows, gradient_summary = [], [], [], []
    verification = dict(max_probability_replay_error=0., max_intent_replay_error=0., max_edit_formula_error=0.,
                        max_total_gradient_linearity_error=0., model_hashes={}, sam_frozen=True, parameter_grad_buffers_empty=True,
                        finite_gradients=True, trainable_parameters_without_gradient=[], nll_clamped_valid_pixels=0)
    with (output / 'local_per_round.jsonl').open('w', encoding='utf-8', buffering=1) as local_stream, \
         (output / 'gradients_per_object.jsonl').open('w', encoding='utf-8', buffering=1) as grad_stream, \
         (output / 'losses_per_round.jsonl').open('w', encoding='utf-8', buffering=1) as loss_stream:
        for variant in ('a2', 'a3'):
            model = PIMSNet(variant=variant).to(args.device).eval()
            checkpoint = load_training_checkpoint(protocol['checkpoints'][variant]['path'], model)
            weights = LossWeights(**checkpoint['config']['loss_weights'])
            del checkpoint
            digest_before = model_digest(model)
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            names, parameters = zip(*named)
            groups = parameter_groups(names)
            sums = {key: [np.zeros(p.numel(), np.float64) for p in parameters] for key in ('mask', 'edit')}
            assert all(not p.requires_grad for p in model.sam_initializer.parameters())
            refs = [row for row in references if row['method'] == variant]
            assert len(refs) == len(dataset)
            captured = []
            handle = model.edit_intent_decoder.head.register_forward_hook(lambda module, inputs, result: captured.append(result))
            for index in range(count):
                sample, ref = dataset[index], refs[index]
                assert ref['id'] == sample['id'] and ref['index'] == index
                clicks = [{k: step[k] for k in ('x', 'y', 'label')} for step in ref['trace'][:5]]
                assert len(clicks) == 5
                first = clicks[0]
                state = model.initialize(sample['image'], [[[first['x'], first['y']]]], [[first['label']]])
                target, valid_tensor = prepare_target(sample['mask'], args.device)
                truth = target[0, 0].cpu().numpy().astype(bool)
                valid = valid_tensor[0, 0].cpu().numpy().astype(bool)
                mask_losses, edit_losses = [], []
                for step, click in enumerate(clicks[1:], 2):
                    old_tensor = state.probabilities
                    old = old_tensor.detach()[0, 0].cpu().numpy()
                    state = model.refine(state, [[[click['x'], click['y']]]], [[click['label']]])
                    z = captured.pop()
                    terms = interactive_losses(state, old_tensor, target, valid_tensor, weights)
                    mask_loss = weights.bce * terms['bce'] + weights.dice * terms['dice']
                    edit_loss = weights.edit * terms['edit']
                    mask_losses.append(mask_loss)
                    edit_losses.append(edit_loss)
                    gm = torch.autograd.grad(mask_loss, z, retain_graph=True)[0]
                    ge = torch.autograd.grad(edit_loss, z, retain_graph=True)[0]
                    if index == 0:  # Verify weighted total-gradient identity for every round/variant.
                        gp = torch.autograd.grad(weights.preserve * terms['preserve'], z, retain_graph=True)[0]
                        gt = torch.autograd.grad(terms['loss'], z, retain_graph=True)[0]
                        error = float((gt - gm - ge - gp).abs().max())
                        verification['max_total_gradient_linearity_error'] = max(verification['max_total_gradient_linearity_error'], error)
                        assert error < 1e-5
                    q = state.edit_probabilities.detach()[0].cpu().numpy()
                    p = state.probabilities.detach()[0, 0].cpu().numpy()
                    assert np.isfinite(q).all() and np.isfinite(p).all()
                    with np.load(reference / 'diagnostics' / f'object_{index:04d}' / variant / f'round_{step:02d}' / 'native.npz') as saved:
                        pe = float(np.abs(p - saved['probabilities'][0]).max())
                        qe = float(np.abs(q - saved['intents']).max())
                    verification['max_probability_replay_error'] = max(verification['max_probability_replay_error'], pe)
                    verification['max_intent_replay_error'] = max(verification['max_intent_replay_error'], qe)
                    assert pe <= 1e-6 and qe <= 1e-6, (variant, index, step, pe, qe)
                    labels, alternative = labels_from_old(old, truth), labels_from_old(old, truth, .5)
                    counts = np.bincount(labels[valid], minlength=3)
                    nll = -np.log(np.maximum(q, 1e-7))[labels, *np.indices(labels.shape)]
                    present = counts > 0
                    means = [float(nll[valid & (labels == i)].mean()) if counts[i] else None for i in range(3)]
                    error = abs(float(terms['edit']) - np.mean([x for x in means if x is not None]))
                    verification['max_edit_formula_error'] = max(verification['max_edit_formula_error'], float(error))
                    assert error < 2e-5
                    verification['nll_clamped_valid_pixels'] += int(((q[labels, *np.indices(labels.shape)] < 1e-7) & valid).sum())
                    info = dict(variant=variant, index=index, round=step, **{k: float(v.detach()) for k, v in terms.items()},
                                mask=float(mask_loss.detach()), class_counts=counts.tolist(), class_mean_nll=means,
                                per_pixel_edit_weight=[float(weights.edit / present.sum() / n) if n else None for n in counts])
                    loss_rows.append(info)
                    loss_stream.write(json.dumps(info, allow_nan=False) + '\n')
                    regions = click_regions(clicks[:step], sample['mask'].shape)
                    regions.update(all=valid, TP=truth & (old > .49), TN=~truth & (old <= .49),
                                   FP=~truth & (old > .49), FN=truth & (old <= .49))
                    gm_np, ge_np = gm.detach()[0].cpu().numpy(), ge.detach()[0].cpu().numpy()
                    assert np.isfinite(gm_np).all() and np.isfinite(ge_np).all()
                    for region, selection in regions.items():
                        stats = local_statistics(q, labels, alternative, old, truth, selection & valid, gm_np, ge_np)
                        if stats is not None:
                            row = dict(variant=variant, index=index, round=step, region=region, **stats)
                            local_rows.append(row)
                            local_stream.write(json.dumps(row, allow_nan=False) + '\n')
                gmask = torch.autograd.grad(torch.stack(mask_losses).mean(), parameters, retain_graph=True, allow_unused=True)
                gedit = torch.autograd.grad(torch.stack(edit_losses).mean(), parameters, allow_unused=True)
                missing = [n for n, m, e in zip(names, gmask, gedit) if m is None or e is None]
                assert not missing, missing
                gm_cpu = [g.detach().cpu().numpy().ravel() for g in gmask]
                ge_cpu = [g.detach().cpu().numpy().ravel() for g in gedit]
                assert all(np.isfinite(g).all() for g in gm_cpu + ge_cpu)
                for i in range(len(parameters)):
                    sums['mask'][i] += gm_cpu[i]
                    sums['edit'][i] += ge_cpu[i]
                for name, indices in groups.items():
                    row = dict(variant=variant, index=index, group=name,
                        **pair_statistics(np.concatenate([gm_cpu[i] for i in indices]), np.concatenate([ge_cpu[i] for i in indices])))
                    gradient_rows.append(row)
                    grad_stream.write(json.dumps(row, allow_nan=False) + '\n')
                assert all(p.grad is None for p in model.parameters())
                del state, old_tensor, terms, z, mask_losses, edit_losses, gmask, gedit, gm, ge
                if (index + 1) % 16 == 0 or index + 1 == count:
                    print(f'{variant}: {index + 1}/{count} objects, full BPTT audited', flush=True)
            handle.remove()
            for group, indices in groups.items():
                selected = [r for r in gradient_rows if r['variant'] == variant and r['group'] == group]
                cosines = [r['cosine'] for r in selected if r['cosine'] is not None]
                gradient_summary.append(dict(variant=variant, group=group, objects=len(selected),
                    conflict_objects=sum(x < -1e-8 for x in cosines), mean_object_cosine=float(np.mean(cosines)),
                    median_object_cosine=float(np.median(cosines)),
                    median_edit_to_mask_norm=float(np.median([r['edit_to_mask_norm'] for r in selected])),
                    mean_gradient=pair_statistics(np.concatenate([sums['mask'][i] / count for i in indices]),
                                                  np.concatenate([sums['edit'][i] / count for i in indices]))))
            digest_after = model_digest(model)
            assert digest_before == digest_after
            verification['model_hashes'][variant] = dict(before=digest_before, after=digest_after)
            del model, parameters, named, sums, captured
            torch.cuda.empty_cache()
    groups = defaultdict(list)
    for row in local_rows:
        stats = {k: v for k, v in row.items() if k not in ('variant', 'index', 'round', 'region')}
        for step in (row['round'], 'all'):
            groups[row['variant'], step, row['region']].append(stats)
    local_summary = [dict(variant=v, round=t, region=r, occurrences=len(rows), **combine_local(rows))
                     for (v, t, r), rows in groups.items()]
    verification.update(objects_per_variant=count, total_refinement_rounds=len(loss_rows),
                        source_and_weight_hashes_unchanged=before == {str(path): file_identity(path)['sha256'] for path in watched}, hashes=before)
    assert verification['source_and_weight_hashes_unchanged']
    for name, data in (('local_summary', local_summary), ('gradient_summary', gradient_summary), ('verification', verification)):
        dump(output / f'{name}.json', data)
    print(json.dumps({k: v for k, v in verification.items() if k not in ('hashes', 'model_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    run(parser.parse_args())
