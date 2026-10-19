"""Independent direction supervision with original A3 mask/preserve semantics."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

@dataclass(frozen=True)
class DirectionalLossWeights:
    bce: float = 1.
    dice: float = 1.
    direction: float = 1.
    preserve: float = .1

    def __post_init__(self):
        values = (self.bce, self.dice, self.direction, self.preserve)
        if not all(math.isfinite(x) and x >= 0 for x in values) or not any(values):
            raise ValueError('Loss weights must be finite, nonnegative, and not all zero')


def direction_target(previous, target, threshold=.49):
    """Detached FN=+1, FP=-1, correct=0. Default retains the original recipe."""
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('threshold must be finite and in (0,1)')
    old_foreground, truth = previous.detach() > threshold, target.detach() > .5
    return (truth & ~old_foreground).to(previous.dtype) - (~truth & old_foreground).to(previous.dtype)


def directional_losses(state, previous, target, valid, weights=DirectionalLossWeights(), threshold=.49):
    if state.direction is None or state.direction.shape != target.shape:
        raise ValueError('Direction prediction must align with the target')
    if state.edit_probabilities is not None:
        raise ValueError('Directional supervision must not use an ADD/REMOVE/KEEP head')
    probabilities = state.probabilities
    if probabilities.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Mask, target, and valid-region shapes must match')
    valid = valid.bool()
    if not valid.any():
        raise ValueError('No valid target pixels')
    desired = direction_target(previous, target, threshold)
    bce = F.binary_cross_entropy_with_logits(state.logits, target, reduction='none')[valid].mean()
    intersection = (probabilities * target * valid).sum()
    dice = 1 - (2 * intersection + 1) / ((probabilities * valid).sum() + (target * valid).sum() + 1)
    correct = (desired == 0) & valid
    preserve = (probabilities - previous.detach()).abs()[correct].mean() if correct.any() else probabilities.sum() * 0
    pixel_loss = F.smooth_l1_loss(state.direction, desired, reduction='none', beta=1.)
    valid = valid.bool()
    class_means = [pixel_loss[selected].mean() for label in (-1, 0, 1)
                   if (selected := ((desired == label) & valid)).any()]
    direction = torch.stack(class_means).mean()
    total = weights.bce * bce + weights.dice * dice + weights.direction * direction + weights.preserve * preserve
    return dict(loss=total, bce=bce, dice=dice, direction=direction, preserve=preserve)
