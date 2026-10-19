"""Unified EV-SAM, direct reprediction, memory ablation, and PIMS-Net evaluation."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluation.methods import create_sessions
from pims_net.evaluation.experiment import file_identity, validate_experiment
from pims_net.evaluation.protocol import EvaluationProtocol, aggregate_results, evaluate_session
from pims_net.models import PIMSNet
from pims_net.paths import OUTPUT_ROOT, checkpoint_path
from pims_net.training.checkpoint import load_training_checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    for variant in ("a1", "a2", "a3"):
        parser.add_argument(f"--{variant}-checkpoint", required=True, help=f"Independently trained {variant.upper()} checkpoint")
    parser.add_argument("--output", help="New directory beneath pims_net/outputs")
    parser.add_argument("--max-clicks", type=int, default=20, help="NoC budget, at least 5")
    parser.add_argument("--boundary-ratio", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Label synthetic plumbing results non-benchmark; trained checkpoints still required")
    parser.add_argument("--diagnostics", action="store_true", help="Save every actual round's masks, clicks, GT, native arrays and edit heatmaps")
    return parser


def evaluate(args):
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    protocol = EvaluationProtocol(max_clicks=args.max_clicks, boundary_ratio=args.boundary_ratio)
    dataset = InteractiveDataset(args.manifest)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Unavailable device: {device}")
    torch.manual_seed(args.seed)
    baseline = file_identity(checkpoint_path())
    models, checkpoints, identities = {}, {}, {}
    for variant in ("a1", "a2", "a3"):
        path = getattr(args, f"{variant}_checkpoint")
        identities[variant] = file_identity(path)
        models[variant] = PIMSNet(variant=variant).to(device).eval()
        checkpoints[variant] = load_training_checkpoint(path, models[variant])
    validate_experiment(checkpoints, baseline["sha256"])
    output = Path(args.output).resolve() if args.output else OUTPUT_ROOT / datetime.now().strftime("evaluation_%Y%m%d_%H%M%S")
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError("Evaluation output must be a run subdirectory beneath pims_net/outputs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    method_definitions = {
        "a0": "Frozen official L0; cumulative points AND previous low-resolution mask logits; best predicted-IoU mask",
        "a1": "Frozen first-click SAM; ConvGRU and independently trained direct full-mask decoder",
        "a2": "Frozen first-click SAM; independently trained intent editor; zero ConvGRU history in training and inference",
        "a3": "Frozen first-click SAM; independently trained full PIMS-Net with recurrent history and intent editor",
    }
    metadata = {"run_type": "smoke_test_not_a_benchmark" if args.smoke_test else "evaluation",
                "diagnostics_enabled": getattr(args, "diagnostics", False),
                "manifest": file_identity(args.manifest), "baseline": baseline,
                "seed": args.seed, "device": str(device), "protocol": protocol.metadata(),
                "methods": method_definitions,
                "checkpoints": identities,
                "training": {name: {key: info[key] for key in ("config", "epoch", "global_step")}
                             for name, info in checkpoints.items()}}
    # The model has loaded its weights; optimizer tensors are not needed during evaluation.
    del checkpoints
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    records = {method: [] for method in method_definitions}
    diagnostics = None
    if getattr(args, "diagnostics", False):
        from pims_net.evaluation.diagnostics import DiagnosticWriter
        diagnostics = DiagnosticWriter(output)
    with (output / "per_object.jsonl").open("w", encoding="utf-8", buffering=1) as stream, torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            if diagnostics is not None:
                diagnostics.begin_object(index, sample)
            features, sizes = models["a3"].sam_initializer.encode(sample["image"])
            sessions = create_sessions(models, features, sizes)
            for name, session in sessions.items():
                result = evaluate_session(session, sample["mask"], protocol,
                                          on_step=None if diagnostics is None else diagnostics.callback(name))
                records[name].append(result)
                stream.write(json.dumps({"index": index, "id": sample["id"], "method": name,
                                         "run_type": metadata["run_type"], **result}, allow_nan=False) + "\n")
            if diagnostics is not None:
                diagnostics.end_object()
            print(f"Evaluated object {index + 1}/{len(dataset)}: {sample['id']}", flush=True)
    summary = [{"method": name, "run_type": metadata["run_type"], **aggregate_results(values)} for name, values in records.items()]
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    if diagnostics is not None:
        diagnostics.finalize(records, protocol.max_clicks)
        from pims_net.evaluation.diagnostic_report import write_report
        write_report(output)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
