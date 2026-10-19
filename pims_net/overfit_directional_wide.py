"""Same train16 recipe/budget, independently initialized wider residual decoder."""

import argparse
import json
from pathlib import Path

from pims_net.models.directional_wide import WideDirectionalPIMSNet
from pims_net.overfit_directional import run
from pims_net.paths import OUTPUT_ROOT, RESEARCH_ROOT


def main(args):
    reference = Path(args.reference).resolve()
    config = json.loads((reference / 'config.json').read_text())
    status = json.loads((reference / 'status.json').read_text())
    assert status['status'] == 'completed' and status['epochs'] == 100 and status['optimizer_updates'] == 800
    assert config['threshold'] == .5 and config['seed'] == 42
    args.manifest = config['source_manifest']['path']
    args.seed, args.epochs, args.eval_every = config['seed'], 100, config['eval_every']
    watched = [reference / name for name in ('last.pt', 'best.pt', 'fixed_train16.jsonl', 'config.json', 'epochs.jsonl')]
    watched += list((RESEARCH_ROOT / 'models').glob('*.py'))
    watched += list((RESEARCH_ROOT / 'training').glob('*.py'))
    run(args, model_factory=WideDirectionalPIMSNet, fixed_manifest=reference / 'fixed_train16.jsonl',
        experiment=dict(name='decoder_width2_residual2', reference=str(reference),
            widths_original=[64, 32, 16], widths_new=[128, 64, 32],
            residual_blocks=2, residual_channels=64, residual_grid=[64, 64],
            residual_formula='x + Conv3x3(GELU(Conv3x3(x))); no normalization',
            initialization='Fresh decoder, no warm start; identical seed42 initial ConvGRU to original.',
            comparison='Same 16 training instances and adaptive RITM protocol/budget; not frozen later click coordinates.'),
        extra_watched=watched, stop_early=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default=str(OUTPUT_ROOT / 'directional_overfit16_seed42_20261019'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    main(parser.parse_args())
