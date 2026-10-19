"""Background-capable fixed-budget COCO subset experiment with three seeds."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from pims_net.paths import OUTPUT_ROOT, REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--test-count", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError("Use an experiment subdirectory beneath pims_net/outputs")
    if len(set(args.seeds)) != len(args.seeds) or args.epochs < 1:
        raise ValueError("Seeds must be unique; epochs must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "experiment.json").exists():
        raise FileExistsError("Refusing to overwrite an existing experiment")
    config = {**vars(args), "output": str(output), "split_seed": 20261017,
              "train_clicks": 5, "eval_clicks": 20, "batch_size": 2, "lr": 1e-4,
              "checkpoint_selection": "final epoch, no heldout tuning", "pid": os.getpid(),
              "python": sys.executable, "started_utc": datetime.now(timezone.utc).isoformat()}
    (output / "experiment.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def status(phase, **details):
        value = {"phase": phase, "pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(), **details}
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "status.json")

    def run_stage(name, arguments):
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
               "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        with (output / f"{name}.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-B", "-m", *arguments], cwd=REPO_ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT, env=env)
            status(name, child_pid=process.pid, command=arguments)
            code = process.wait()
        if code:
            raise RuntimeError(f"{name} failed with exit code {code}; see {name}.log")

    try:
        status("preparing_subset")
        from pims_net.data.prepare_coco_subset import prepare_subset
        prepare_subset(args.coco_root, output / "data", args.train_count, args.test_count, config["split_seed"])
        for seed in args.seeds:
            run_root = output / f"seed_{seed}"
            run_stage(f"seed_{seed}_training", ["pims_net.train_ablations", "--manifest", str(output / "data/train.jsonl"),
                      "--output", str(run_root / "training"), "--epochs", str(args.epochs), "--batch-size", "2",
                      "--max-clicks", "5", "--lr", "0.0001", "--device", "cuda", "--seed", str(seed),
                      "--save-every", str(args.epochs)])
            run_stage(f"seed_{seed}_evaluation", ["pims_net.evaluate_interactive", "--manifest", str(output / "data/heldout.jsonl"),
                      "--output", str(run_root / "evaluation"), "--device", "cuda", "--seed", str(seed),
                      "--max-clicks", "20", "--a1-checkpoint", str(run_root / "training/a1/last.pt"),
                      "--a2-checkpoint", str(run_root / "training/a2/last.pt"),
                      "--a3-checkpoint", str(run_root / "training/a3/last.pt")])
        status("aggregating")
        from pims_net.evaluation.report_multiseed import build_report
        build_report(output, args.seeds)
        status("completed", report=str(output / "comparison.md"))
    except Exception:
        status("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
