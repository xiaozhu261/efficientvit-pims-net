"""Full-budget train16 capacity trial; preserve both earlier models/weights."""

import argparse
import json
from pathlib import Path

from pims_net.models.directional_large import LargeDirectionalPIMSNet
from pims_net.overfit_directional import run
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT


def main(args):
    reference = Path(args.reference).resolve()
    previous = Path(args.previous).resolve()
    config = json.loads((reference / 'config.json').read_text())
    for root in (reference, previous):
        status = json.loads((root / 'status.json').read_text())
        assert status['status'] == 'completed' and status['epochs'] == 100 and status['optimizer_updates'] == 800
    assert (reference / 'fixed_train16.jsonl').read_bytes() == (previous / 'fixed_train16.jsonl').read_bytes()
    assert config['seed'] == 42 and config['threshold'] == .5
    args.manifest = config['source_manifest']['path']
    args.seed, args.epochs, args.eval_every = config['seed'], 100, config['eval_every']
    watched = [root / name for root in (reference, previous) for name in
               ('last.pt', 'best.pt', 'fixed_train16.jsonl', 'config.json', 'epochs.jsonl')]
    watched += list((RESEARCH_ROOT / 'models').glob('*.py')) + list((RESEARCH_ROOT / 'training').glob('*.py'))
    run(args, model_factory=LargeDirectionalPIMSNet, fixed_manifest=reference / 'fixed_train16.jsonl',
        experiment=dict(name='decoder_width4_residual4', reference=str(reference), previous=str(previous),
            widths_original=[64,32,16], widths_previous=[128,64,32], widths_new=[256,128,64],
            residual_blocks=4, residual_channels=128, residual_grid=[64,64],
            residual_formula='x + Conv3x3(GELU(Conv3x3(x))); no normalization',
            initialization='Fresh decoder, no warm start; identical seed42 non-decoder initial weights.',
            constraints='No inference sign or Gaussian constraints; original Directional editor and loss.',
            comparison='Same train16 adaptive RITM protocol and 100epoch/800update budget; later clicks may differ.'),
        extra_watched=watched, stop_early=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default=str(OUTPUT_ROOT/'directional_overfit16_seed42_20261019'))
    parser.add_argument('--previous',default=str(OUTPUT_ROOT/'directional_wide_overfit16_seed42_20261019'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    main(parser.parse_args())
