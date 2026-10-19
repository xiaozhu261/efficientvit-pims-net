"""Segmentation, class-balanced edit supervision, and correct-region preservation."""

from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    bce: float = 1.0
    dice: float = 1.0
    edit: float = 1.0
    preserve: float = 0.1

    def __post_init__(self):
        values = (self.bce, self.dice, self.edit, self.preserve)
        if not all(math.isfinite(value) and value >= 0 for value in values) or sum(values) == 0:
            raise ValueError("Loss weights must be finite, nonnegative, and not all zero")


def prepare_target(mask, device):
    """Match SAM's 1024 canvas, then nearest-neighbor resize to 256; exclude padding."""
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2 or min(mask.shape) == 0:
        raise ValueError("mask must be a nonempty 2D array")
    height, width = mask.shape
    scale = 1024 / max(height, width)
    resized_h, resized_w = int(height * scale + 0.5), int(width * scale + 0.5)
    if min(resized_h, resized_w) < 1:
        raise ValueError("Image aspect ratio is too extreme for SAM preprocessing")
    target = np.zeros((1024, 1024), dtype=np.uint8)
    valid = np.zeros_like(target)
    target[:resized_h, :resized_w] = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    valid[:resized_h, :resized_w] = 1
    tensors = [torch.as_tensor(cv2.resize(value, (256, 256), interpolation=cv2.INTER_NEAREST),
                               dtype=torch.float32, device=device)[None, None] for value in (target, valid)]
    return tuple(tensors)


def interactive_losses(state, previous_probabilities, target, valid, weights=LossWeights()):
    """Edit targets use detached old predictions: FN=ADD, FP=REMOVE, correct=KEEP.

    Average NLL separately for each present class, then average those class
    means; absent classes do not divide by zero. Preserve is mean absolute
    probability change only on previously correct, non-padding pixels.
    """
    probabilities = state.probabilities
    if probabilities.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Mask, target, and valid-region shapes must match")
    valid = valid.bool()
    if not valid.any():
        raise ValueError("No valid target pixels")
    old = previous_probabilities.detach()
    truth = target > 0.5
    old_foreground = old > 0.49
    add = truth & ~old_foreground
    remove = ~truth & old_foreground
    edit_target = torch.full_like(target, 2, dtype=torch.long)
    edit_target[add] = 0
    edit_target[remove] = 1
    correct = (edit_target == 2) & valid

    bce = F.binary_cross_entropy_with_logits(state.logits, target, reduction="none")[valid].mean()
    intersection = (probabilities * target * valid).sum()
    dice = 1 - (2 * intersection + 1) / ((probabilities * valid).sum() + (target * valid).sum() + 1)
    if state.edit_probabilities is None:  # A1 directly predicts a mask, without edit classes.
        edit = probabilities.sum() * 0
    else:
        nll = -state.edit_probabilities.clamp_min(1e-7).log().gather(1, edit_target)
        class_means = [nll[selected].mean() for label in range(3) if (selected := ((edit_target == label) & valid)).any()]
        edit = torch.stack(class_means).mean()
    preserve = (probabilities - old).abs()[correct].mean() if correct.any() else probabilities.sum() * 0
    total = weights.bce * bce + weights.dice * dice + weights.edit * edit + weights.preserve * preserve
    return {"loss": total, "bce": bce, "dice": dice, "edit": edit, "preserve": preserve}
