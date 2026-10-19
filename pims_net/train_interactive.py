"""Train PIMS-Net with up to five online RITM-style clicks per object."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import hashlib
import logging
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluation.experiment import file_identity
from pims_net.models import PIMSNet
from pims_net.paths import OUTPUT_ROOT, checkpoint_path
from pims_net.training.checkpoint import save_training_checkpoint
from pims_net.training.interactive_losses import LossWeights
from pims_net.training.interactive_rollout import interactive_rollout


def build_parser(include_variant=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL image/mask pairs; paths relative to the manifest")
    if include_variant:
        parser.add_argument("--variant", choices=("a1", "a2", "a3"), default="a3")
    parser.add_argument("--output", help="New run directory beneath pims_net/outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1, help="Images per optimizer update, accumulated sequentially")
    parser.add_argument("--max-clicks", type=int, choices=(2, 3, 4, 5), default=5, help="Includes the frozen first click")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--edit-weight", type=float, default=1.0)
    parser.add_argument("--preserve-weight", type=float, default=0.1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save an epoch snapshot every N epochs")
    parser.add_argument("--max-steps", type=int, help="Optional limit on optimizer batches for smoke runs")
    return parser


def train(args):
    if min(args.epochs, args.batch_size, args.save_every) < 1 or args.seed < 0:
        raise ValueError("epochs, batch-size, save-every must be positive; seed must be nonnegative")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if not all(math.isfinite(value) for value in (args.lr, args.weight_decay, args.grad_clip)):
        raise ValueError("Optimizer settings must be finite")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("lr/grad-clip must be positive and weight-decay nonnegative")
    weights = LossWeights(args.bce_weight, args.dice_weight, args.edit_weight, args.preserve_weight)
    if args.variant == "a1" and weights.bce + weights.dice + weights.preserve == 0:
        raise ValueError("A1 has no edit-class head; at least one applicable loss weight must be positive")
    dataset = InteractiveDataset(args.manifest)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Unavailable training device: {device}")
    output = Path(args.output).resolve() if args.output else OUTPUT_ROOT / datetime.now().strftime("interactive_%Y%m%d_%H%M%S")
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError("Use a run subdirectory beneath pims_net/outputs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty; select a new output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "manifest": str(dataset.manifest), "output": str(output),
              "device": str(device), "loss_weights": asdict(weights), "samples": len(dataset),
              "manifest_sha256": hashlib.sha256(dataset.manifest.read_bytes()).hexdigest(),
              "baseline_sha256": file_identity(checkpoint_path())["sha256"],
              "click_protocol": "ritm_train_random_half_max_distance_threshold_0.49_v1",
              "sam_frozen": True, "torch_version": str(torch.__version__)}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    logger = logging.getLogger("pims_net.interactive")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(output / "train.log", encoding="utf-8"), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    try:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = PIMSNet(variant=args.variant).to(device).train()
        parameters = list(model.trainable_parameters())
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        logger.info("device=%s samples=%d trainable_parameters=%d max_clicks=%d", device, len(dataset),
                    sum(parameter.numel() for parameter in parameters), args.max_clicks)
        global_step = updates = 0
        stop = False
        with (output / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as metrics_file:
            def record(value):
                metrics_file.write(json.dumps(value, allow_nan=False) + "\n")

            for epoch in range(args.epochs):
                model.train()
                order = np.random.default_rng(np.random.SeedSequence([args.seed, epoch])).permutation(len(dataset))
                for start in range(0, len(order), args.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    totals = {key: 0.0 for key in ("loss", "bce", "dice", "edit", "preserve")}
                    trained = 0
                    for index in order[start:start + args.batch_size]:
                        sample = dataset[int(index)]
                        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch, int(index)]))
                        result = interactive_rollout(model, sample["image"], sample["mask"], args.max_clicks, rng, weights)
                        if result["loss"] is not None:
                            if not torch.isfinite(result["loss"]):
                                raise FloatingPointError(f"Nonfinite loss for sample {sample['id']}")
                            result["loss"].backward()
                            trained += 1
                            for key in totals:
                                totals[key] += result["metrics"][key]
                        record({"type": "sample", "epoch": epoch + 1, "step": global_step + 1,
                                "id": sample["id"], "optimized": result["loss"] is not None,
                                "clicks": result["clicks"], **result["metrics"]})
                        del result  # Release this image's recurrent graph before the next image.
                    if trained:
                        for parameter in parameters:
                            if parameter.grad is not None:
                                parameter.grad.div_(trained)
                        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
                        optimizer.step()
                        updates += 1
                    global_step += 1
                    averages = {key: value / max(trained, 1) for key, value in totals.items()}
                    record({"type": "batch", "epoch": epoch + 1, "step": global_step,
                            "trained_samples": trained, "optimizer_updates": updates, **averages})
                    logger.info("epoch=%d step=%d trained=%d loss=%.5f bce=%.5f dice=%.5f edit=%.5f preserve=%.5f",
                                epoch + 1, global_step, trained, *(averages[key] for key in totals))
                    if args.max_steps is not None and global_step >= args.max_steps:
                        stop = True
                        break
                config["optimizer_updates"] = updates
                save_training_checkpoint(output / "last.pt", model, optimizer, epoch + 1, global_step, config)
                if (epoch + 1) % args.save_every == 0:
                    save_training_checkpoint(output / f"epoch_{epoch + 1:04d}.pt", model, optimizer,
                                             epoch + 1, global_step, config)
                if stop:
                    break
        summary = {"steps": global_step, "optimizer_updates": updates, "output": str(output)}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logger.info("Finished: %s", summary)
        return summary
    except Exception:
        logger.exception("Training failed")
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def main():
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
