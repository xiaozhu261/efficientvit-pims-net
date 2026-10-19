"""One deterministic adaptive click protocol shared by every evaluated method."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .metrics import boundary_iou, correct_region_damage, mask_iou, segmentation_statistics


@dataclass(frozen=True)
class EvaluationProtocol:
    max_clicks: int = 20
    threshold: float = 0.5
    target_iou: float = 0.9
    boundary_ratio: float = 0.02

    def __post_init__(self):
        if self.max_clicks < 5:
            raise ValueError("max_clicks must be >= 5 for IoU@1/3/5")
        if not 0 < self.threshold < 1 or not 0 < self.target_iou <= 1:
            raise ValueError("Invalid probability or IoU threshold")
        if self.target_iou != 0.9:
            raise ValueError("NoC@90 requires target_iou=0.9")
        if not np.isfinite(self.boundary_ratio) or self.boundary_ratio <= 0:
            raise ValueError("boundary_ratio must be finite and positive")

    def metadata(self):
        return {**asdict(self), "click_policy": "deterministic_largest_error_distance",
                "click_coordinates": "original_image_xy", "threshold_rule": "probability > threshold",
                "click_trajectory": "adaptive per method, identical first click and selection rule",
                "noc_failure": "cap at max_clicks; also report failure_rate@90",
                "boundary_at_click": 5, "damage_transitions": "2..5, pixel-weighted per image",
                "aggregation": "macro mean over the same image/object manifest"}


class EvaluationClicker:
    def __init__(self, target):
        self.target = np.asarray(target, dtype=bool)
        if self.target.ndim != 2 or min(self.target.shape) == 0 or not self.target.any():
            raise ValueError("Evaluation target must be a nonempty foreground object")
        self.available = np.ones_like(self.target)

    def next_click(self, prediction):
        prediction = np.asarray(prediction, dtype=bool)
        if prediction.shape != self.target.shape:
            raise ValueError("Prediction and target shapes must match")
        distances = []
        for error in (self.target & ~prediction, ~self.target & prediction):
            distance = cv2.distanceTransform(np.pad(error.astype(np.uint8), 1), cv2.DIST_L2, 0)[1:-1, 1:-1]
            distances.append(np.where(self.available, distance, 0))
        positive = distances[0].max() > distances[1].max()
        chosen = distances[0 if positive else 1]
        if chosen.max() <= 0:
            return None
        y, x = np.unravel_index(np.argmax(chosen), chosen.shape)  # Row-major tie breaking.
        self.available[y, x] = False
        return int(x), int(y), int(positive)


def evaluate_session(session, target, protocol=EvaluationProtocol(), on_step=None):
    """session.add_click returns an original-resolution foreground probability map."""
    target = np.asarray(target, dtype=bool)
    clicker = EvaluationClicker(target)
    prediction = np.zeros_like(target)
    trace = []
    first_success = None
    stop_reason = "budget"
    for index in range(1, protocol.max_clicks + 1):
        click = clicker.next_click(prediction)
        if click is None:
            stop_reason = "no_unclicked_errors"
            break
        probabilities = np.asarray(session.add_click(*click))
        if probabilities.shape != target.shape or not (
            np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)
        ).all():
            raise ValueError("Method must return finite original-resolution probabilities in [0,1]")
        current = probabilities > protocol.threshold
        iou = mask_iou(current, target)
        item = {"click": index, "x": click[0], "y": click[1], "label": click[2],
                "iou": iou, "boundary_iou": boundary_iou(current, target, protocol.boundary_ratio)}
        item.update(segmentation_statistics(current, target))
        item["iou_delta"] = iou - trace[-1]["iou"] if trace else None
        item["tp_lost"] = int(np.count_nonzero(prediction & target & ~current)) if trace else 0
        item["tn_lost"] = int(np.count_nonzero(~prediction & ~target & current)) if trace else 0
        item["fn_fixed"] = int(np.count_nonzero(~prediction & target & current)) if trace else 0
        item["fp_fixed"] = int(np.count_nonzero(prediction & ~target & ~current)) if trace else 0
        item["latest_click_satisfied"] = bool(current[click[1], click[0]] == bool(click[2]))
        item.update(correct_region_damage(prediction, current, target) if index > 1 else
                    {"damaged_pixels": 0, "previous_correct_pixels": 0, "damage_rate": 0.0})
        trace.append(item)
        if on_step is not None:
            on_step(session=session, probabilities=probabilities, current_prediction=current,
                    previous_prediction=prediction, item=item, trace=trace)
        prediction = current
        if first_success is None and iou >= protocol.target_iou:
            first_success = index
        # Always measure click 5, even if IoU already crossed 0.90 earlier.
        if index >= 5 and first_success is not None:
            stop_reason = "target_reached_and_five_clicks_measured"
            break
    if not trace:
        raise RuntimeError("No initial click was generated for a foreground object")
    at = lambda number: trace[min(number, len(trace)) - 1]
    transitions = trace[1:5]
    correct = sum(item["previous_correct_pixels"] for item in transitions)
    damaged = sum(item["damaged_pixels"] for item in transitions)
    return {"IoU@1": at(1)["iou"], "IoU@3": at(3)["iou"], "IoU@5": at(5)["iou"],
            "NoC@90": first_success if first_success is not None else protocol.max_clicks,
            "reached_90": first_success is not None, "first_success_click": first_success,
            "BoundaryIoU@5": at(5)["boundary_iou"],
            "correct_region_damage_rate@2:5": damaged / correct if correct else 0.0,
            "clicks_used": len(trace), "stop_reason": stop_reason, "trace": trace}


def aggregate_results(records):
    keys = ("IoU@1", "IoU@3", "IoU@5", "NoC@90", "BoundaryIoU@5", "correct_region_damage_rate@2:5")
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    return {"objects": len(records), **{key: float(np.mean([record[key] for record in records])) for key in keys},
            "failure_rate@90": float(np.mean([not record["reached_90"] for record in records]))}
