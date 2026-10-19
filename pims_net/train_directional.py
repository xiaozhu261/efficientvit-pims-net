"""Independently train DirectionalPIMSNet; never load or overwrite A3 research weights."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluation.experiment import file_identity
from pims_net.models.directional_pims_net import DirectionalPIMSNet
from pims_net.paths import OUTPUT_ROOT, checkpoint_path
from pims_net.training.checkpoint import save_training_checkpoint
from pims_net.training.directional_losses import DirectionalLossWeights
from pims_net.training.directional_rollout import directional_rollout


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True, help='New directory below pims_net/outputs')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-clicks', type=int, choices=(2, 3, 4, 5), default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.)
    parser.add_argument('--bce-weight', type=float, default=1.)
    parser.add_argument('--dice-weight', type=float, default=1.)
    parser.add_argument('--direction-weight', type=float, default=1.)
    parser.add_argument('--preserve-weight', type=float, default=.1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--max-steps', type=int, help='Limit optimizer batches for a smoke run')
    return parser


def train(args):
    if min(args.epochs, args.batch_size, args.save_every) < 1 or args.seed < 0:
        raise ValueError('Positive epoch/batch/save counts and nonnegative seed required')
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError('max-steps must be positive')
    if not all(math.isfinite(x) for x in (args.lr, args.weight_decay, args.grad_clip)) or args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError('Invalid optimizer configuration')
    weights = DirectionalLossWeights(args.bce_weight, args.dice_weight, args.direction_weight, args.preserve_weight)
    dataset = InteractiveDataset(args.manifest)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError(f'Unavailable device: {device}')
    output = Path(args.output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError('Output must be a run subdirectory of pims_net/outputs')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite a nonempty run directory')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(**vars(args), variant='directional', manifest_sha256=file_identity(dataset.manifest)['sha256'],
                  loss_weights=asdict(weights), samples=len(dataset), baseline_sha256=file_identity(checkpoint_path())['sha256'],
                  click_protocol='ritm_train_random_half_max_distance_threshold_0.49_v1', sam_frozen=True,
                  direction_target='FN:+1 FP:-1 correct:0, detached P_old>0.49',
                  direction_loss='SmoothL1(beta=1), equally average present classes -1/0/+1',
                  update='P+S*where(D>=0,D*(1-P),D*P); D=tanh, S=sigmoid; exact zero KEEP logits',
                  initialization='New research parameters; frozen official SAM only; no A3 warm start',
                  run_type='smoke_test_not_a_benchmark' if args.max_steps else 'training', torch_version=str(torch.__version__))
    config.update(manifest=str(dataset.manifest), output=str(output), device=str(device))
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = DirectionalPIMSNet().to(device).train()
    parameters = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    step = updates = 0
    stop = False
    with (output / 'metrics.jsonl').open('w', encoding='utf-8', buffering=1) as metrics, \
         (output / 'train.log').open('w', encoding='utf-8', buffering=1) as log:
        def record(value):
            metrics.write(json.dumps(value, allow_nan=False) + '\n')

        for epoch in range(args.epochs):
            model.train()
            order = np.random.default_rng(np.random.SeedSequence([args.seed, epoch])).permutation(len(dataset))
            for start in range(0, len(order), args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                totals = {key: 0. for key in ('loss', 'bce', 'dice', 'direction', 'preserve')}
                trained = 0
                for index in order[start:start + args.batch_size]:
                    sample = dataset[int(index)]
                    rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch, int(index)]))
                    result = directional_rollout(model, sample['image'], sample['mask'], args.max_clicks, rng, weights)
                    if result['loss'] is not None:
                        if not torch.isfinite(result['loss']):
                            raise FloatingPointError(f'Nonfinite loss: {sample["id"]}')
                        result['loss'].backward()
                        trained += 1
                        for key in totals:
                            totals[key] += result['metrics'][key]
                    record(dict(type='sample', epoch=epoch + 1, step=step + 1, id=sample['id'],
                                optimized=result['loss'] is not None, clicks=result['clicks'], **result['metrics']))
                    del result
                if trained:
                    for p in parameters:
                        if p.grad is not None:
                            p.grad.div_(trained)
                    torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                assert all(p.grad is None and not p.requires_grad for p in model.sam_initializer.parameters())
                step += 1
                averages = {k: v / max(1, trained) for k, v in totals.items()}
                record(dict(type='batch', epoch=epoch + 1, step=step, trained_samples=trained, optimizer_updates=updates, **averages))
                line = f'{datetime.now().isoformat()} epoch={epoch + 1} step={step} trained={trained} ' + ' '.join(f'{k}={v:.6f}' for k, v in averages.items())
                log.write(line + '\n')
                print(line, flush=True)
                if args.max_steps is not None and step >= args.max_steps:
                    stop = True
                    break
            config['optimizer_updates'] = updates
            save_training_checkpoint(output / 'last.pt', model, optimizer, epoch + 1, step, config)
            if (epoch + 1) % args.save_every == 0:
                save_training_checkpoint(output / f'epoch_{epoch + 1:04d}.pt', model, optimizer, epoch + 1, step, config)
            if stop:
                break
    result = dict(variant='directional', steps=step, optimizer_updates=updates, output=str(output), run_type=config['run_type'])
    (output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    torch.set_num_threads(4)
    train(build_parser().parse_args())
