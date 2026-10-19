"""Train A1, A2, A3 independently under the same data, click and training budget."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pims_net.paths import OUTPUT_ROOT
from pims_net.train_interactive import build_parser, train


def main():
    parser = build_parser(include_variant=False)
    parser.description = __doc__
    args = parser.parse_args()
    output = Path(args.output).resolve() if args.output else OUTPUT_ROOT / datetime.now().strftime("ablations_%Y%m%d_%H%M%S")
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError("Use a run subdirectory beneath pims_net/outputs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Ablation run directory must be empty: {output}")
    for variant in ("a1", "a2", "a3"):
        variant_args = argparse.Namespace(**{**vars(args), "variant": variant, "output": str(output / variant)})
        train(variant_args)


if __name__ == "__main__":
    main()
