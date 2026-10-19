"""Read-only model audit: KEEP/GT-oracle controls on the evaluated COCO subset."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pims_net.data.interactive_dataset import InteractiveDataset
from pims_net.evaluation.experiment import file_identity
from pims_net.evaluation.metrics import correct_region_damage, mask_iou
from pims_net.models import PIMSNet
from pims_net.paths import OUTPUT_ROOT, checkpoint_path
from pims_net.tests.test_mask_editor_audit import oracle_intents, pure_keep
from pims_net.training.checkpoint import load_training_checkpoint
from pims_net.training.interactive_losses import prepare_target


def original_prediction(model, state):
    # Match evaluation exactly. In float32, tiny positive logits can round to
    # sigmoid==0.5, so logits>0 is not bit-equivalent to sigmoid(logits)>0.5.
    return (model.sam_initializer.postprocess(state.logits, state.image_sizes)[0][0, 0].sigmoid() > .5).cpu().numpy()


def run(args):
    output = Path(args.output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError("Use a dedicated research output subdirectory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(args.experiment).resolve()
    dataset = InteractiveDataset(root / "data/heldout.jsonl")
    evaluation = root / "seed_42/evaluation"
    protocol = json.loads((evaluation / "protocol.json").read_text())
    rows = [json.loads(line) for line in (evaluation / "per_object.jsonl").read_text().splitlines()]
    reference = {row["id"]: row for row in rows if row["method"] == "a3"}
    checkpoint = root / "seed_42/training/a3/last.pt"
    identities = {"baseline": file_identity(checkpoint_path()), "checkpoint": file_identity(checkpoint),
                  "manifest": file_identity(dataset.manifest)}
    if identities["baseline"] != protocol["baseline"] or identities["checkpoint"] != protocol["checkpoints"]["a3"] or identities["manifest"] != protocol["manifest"]:
        raise ValueError("Audit inputs differ from saved evaluation")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = PIMSNet().to(args.device).eval()
    load_training_checkpoint(checkpoint, model)
    records = []
    with (output / "per_object.jsonl").open("w", encoding="utf-8", buffering=1) as stream, torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            ref = reference[sample["id"]]
            clicks = ref["trace"][:5]
            first = clicks[0]
            state = model.initialize(sample["image"], [[[first["x"], first["y"]]]], [[first["label"]]])
            initial = state
            initial_mask = original_prediction(model, initial)
            initial_iou = mask_iou(initial_mask, sample["mask"])
            if abs(initial_iou - ref["IoU@1"]) > 1e-7:
                raise AssertionError("First-click replay differs from original evaluation")
            target, valid = prepare_target(sample["mask"], args.device)
            keep_first, later_flips = None, 0
            hidden_finite = True
            # Forced four KEEP rounds: a control, not a new adaptive benchmark.
            for step in range(1, 5):
                click = clicks[min(step, len(clicks) - 1)]
                previous = state
                with patch.object(model.edit_intent_decoder, "forward", return_value=pure_keep(state.probabilities)):
                    state = model.refine(state, [[[click["x"], click["y"]]]], [[click["label"]]])
                if not torch.equal(state.probabilities, initial.probabilities):
                    raise AssertionError("Probability-domain pure KEEP is not identical")
                if state.points.shape[1] != previous.points.shape[1] + 1 or state.hidden.shape != (1, 128, 64, 64):
                    raise AssertionError("State/click shape contract violated")
                hidden_finite &= bool(torch.isfinite(state.hidden).all())
                current_mask = original_prediction(model, state)
                if keep_first is None:
                    keep_first = current_mask
                else:
                    later_flips += int(np.count_nonzero(current_mask != keep_first))
            damage = correct_region_damage(initial_mask, keep_first, sample["mask"])
            record = {"id": sample["id"], "index": index, "original_shape": list(sample["mask"].shape),
                      "initial_iou": initial_iou, "keep_original_iou": mask_iou(keep_first, sample["mask"]),
                      "keep_original_flipped_pixels": int(np.count_nonzero(keep_first != initial_mask)),
                      "keep_damaged_pixels": damage["damaged_pixels"], "keep_damage_rate": damage["damage_rate"],
                      "keep_later_round_flipped_pixels": later_flips, "keep_probability_max_error": 0.,
                      "keep_logit_max_error": float((state.logits - initial.logits).abs().max()),
                      "probability_logit_roundtrip_max_error": float((state.logits.sigmoid() - state.probabilities).abs().max()),
                      "initial_logit_min": float(initial.logits.min()), "initial_logit_max": float(initial.logits.max()),
                      "clipped_lowres_pixels": int(((initial.probabilities < 1e-6) | (initial.probabilities > 1 - 1e-6)).sum()),
                      "hidden_finite": hidden_finite}
            for threshold, name in ((.5, "oracle_eval"), (.49, "oracle_train")):
                intents = oracle_intents(initial.probabilities, target, threshold)
                with patch.object(model.edit_intent_decoder, "forward", return_value=intents):
                    edited = model.refine(initial, [[[first["x"], first["y"]]]], [[first["label"]]])
                record[name + "_lowres_errors"] = int((((edited.probabilities > .5) != target.bool()) & valid.bool()).sum())
                record[name + "_original_iou"] = mask_iou(original_prediction(model, edited), sample["mask"])
                previously_correct = (initial.probabilities > threshold) == target.bool()
                if not torch.equal(edited.probabilities[previously_correct], initial.probabilities[previously_correct]):
                    raise AssertionError("Oracle changed an already-correct pixel despite KEEP")
            # Separate geometry control: force the entire 256 canvas to binary GT.
            hard_state = replace(initial, probabilities=target, logits=torch.logit(target, eps=1e-6))
            record["hard_gt_reprojection_iou"] = mask_iou(original_prediction(model, hard_state), sample["mask"])
            # Direct original-grid oracle has no resize or logit conversion.
            original_p = model.sam_initializer.postprocess(initial.logits, initial.image_sizes)[0].sigmoid()
            original_gt = torch.as_tensor(sample["mask"], device=args.device)[None, None]
            direct = model.mask_editor(original_p, oracle_intents(original_p, original_gt))
            record["original_grid_oracle_errors"] = int(((direct > .5) != original_gt).sum())
            # Replay the learned path separately to check unmodified recurrent inference.
            learned = initial
            norm_error = 0.
            replay_error = 0.
            threshold_mismatch = 0
            for click in clicks[1:]:
                previous = learned
                learned = model.refine(previous, [[[click["x"], click["y"]]]], [[click["label"]]])
                norm_error = max(norm_error, float((learned.edit_probabilities.sum(1) - 1).abs().max()))
                replay_error = max(replay_error, abs(mask_iou(original_prediction(model, learned), sample["mask"]) - click["iou"]))
                original_logits = model.sam_initializer.postprocess(learned.logits, learned.image_sizes)[0]
                threshold_mismatch += int(((original_logits > 0) != (original_logits.sigmoid() > .5)).sum())
            record["learned_iou5"] = mask_iou(original_prediction(model, learned), sample["mask"])
            record["intent_normalization_max_error"] = norm_error
            record["learned_replay_iou_max_error"] = replay_error
            record["predictor_evaluator_threshold_mismatch_pixels"] = threshold_mismatch
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            if (index + 1) % 32 == 0:
                print(f"Audited {index + 1}/{len(dataset)} objects", flush=True)
    summary = {"objects": len(records), "identities": identities, "architecture_modified": False,
               "control_note": "KEEP rounds are forced controls; oracle uses GT and is not a model performance score"}
    for key in ("initial_iou", "keep_original_iou", "oracle_eval_original_iou", "oracle_train_original_iou",
                "hard_gt_reprojection_iou", "learned_iou5", "keep_damage_rate"):
        summary[key + "_mean"] = float(np.mean([row[key] for row in records]))
    for key in ("keep_original_flipped_pixels", "keep_damaged_pixels", "keep_later_round_flipped_pixels",
                "oracle_eval_lowres_errors", "oracle_train_lowres_errors", "original_grid_oracle_errors",
                "predictor_evaluator_threshold_mismatch_pixels"):
        summary[key + "_total"] = sum(row[key] for row in records)
        summary[key + "_affected_objects"] = sum(row[key] > 0 for row in records)
    for key in ("keep_probability_max_error", "keep_logit_max_error", "probability_logit_roundtrip_max_error",
                "intent_normalization_max_error", "learned_replay_iou_max_error"):
        summary[key] = max(row[key] for row in records)
    differences = [row["keep_original_iou"] - row["initial_iou"] for row in records]
    summary["keep_iou_delta_min"] = min(differences)
    summary["keep_iou_delta_max"] = max(differences)
    summary["all_hidden_finite"] = all(row["hidden_finite"] for row in records)
    summary["checkpoint_unchanged"] = file_identity(checkpoint) == identities["checkpoint"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())
