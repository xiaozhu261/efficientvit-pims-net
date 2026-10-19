"""Original-resolution binary-mask metrics for interactive segmentation."""

import cv2
import numpy as np


def _pair(first, second):
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if first.ndim != 2 or first.shape != second.shape or min(first.shape) == 0:
        raise ValueError("Metrics require matching nonempty 2D masks")
    return first, second


def mask_iou(prediction, target):
    prediction, target = _pair(prediction, target)
    union = np.count_nonzero(prediction | target)
    return float(np.count_nonzero(prediction & target) / union) if union else 1.0


def segmentation_statistics(prediction, target):
    """Original-grid counts; zero-denominator precision/recall are defined as 0."""
    prediction, target = _pair(prediction, target)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "foreground_ratio": float(prediction.mean()), "gt_foreground_ratio": float(target.mean()),
            "pred_to_gt_area_ratio": (tp + fp) / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.}


def mask_boundary(mask, ratio=0.02):
    """Interior band using the official Boundary IoU diagonal-scaled erosion."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) == 0 or not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Boundary requires a nonempty 2D mask and a positive finite ratio")
    iterations = max(1, int(round(ratio * np.hypot(*mask.shape))))
    padded = np.pad(mask.astype(np.uint8), 1)
    eroded = cv2.erode(padded, np.ones((3, 3), dtype=np.uint8), iterations=iterations)[1:-1, 1:-1]
    return mask & ~eroded.astype(bool)


def boundary_iou(prediction, target, ratio=0.02):
    prediction, target = _pair(prediction, target)
    return mask_iou(mask_boundary(prediction, ratio), mask_boundary(target, ratio))


def correct_region_damage(previous, current, target):
    """Count previously correct pixels that became wrong (both FG and BG)."""
    previous, target = _pair(previous, target)
    current, _ = _pair(current, target)
    correct = previous == target
    count = int(np.count_nonzero(correct))
    damaged = int(np.count_nonzero(correct & (current != target)))
    return {"damaged_pixels": damaged, "previous_correct_pixels": count,
            "damage_rate": float(damaged / count) if count else 0.0}
