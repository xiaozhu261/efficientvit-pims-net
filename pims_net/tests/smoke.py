"""Check local resources; optionally load the actual pretrained baseline."""

import argparse
import hashlib
import json

from pims_net.paths import MODEL_NAMES, REPO_ROOT, RESEARCH_ROOT, checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-baseline", action="store_true", help="Hash original files, including weights")
    parser.add_argument("--load-model", action="store_true", help="Import dependencies and load pretrained L0")
    args = parser.parse_args()

    for name in MODEL_NAMES:
        path = checkpoint_path(name)
        print(f"Full checkpoint: {path} ({path.stat().st_size} bytes)")
        try:
            path = checkpoint_path(name, distilled=True)
        except FileNotFoundError:
            print(f"Distilled checkpoint unavailable: {name}")
        else:
            print(f"Distilled checkpoint: {path} ({path.stat().st_size} bytes)")

    if args.verify_baseline:
        manifest = json.loads((RESEARCH_ROOT / "docs" / "baseline_manifest.json").read_text(encoding="utf-8"))
        for relative_path, expected in manifest.items():
            with (REPO_ROOT / relative_path).open("rb") as stream:
                actual = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    actual.update(chunk)
            if actual.hexdigest() != expected:
                raise RuntimeError(f"Original file changed: {relative_path}")
        print(f"Baseline unchanged: {len(manifest)} original files verified")

    if args.load_model:
        from pims_net.models import build_baseline

        model = build_baseline().eval()
        print(f"Pretrained L0 loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    else:
        print("Resource checks only; model import, inference, and training were not tested")


if __name__ == "__main__":
    main()
