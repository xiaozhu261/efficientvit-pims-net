"""Loss-only ablations; no change to forward or recurrent state updates."""

import torch
from torch.nn import functional as F

from .directional_losses import DirectionalLossWeights, directional_losses


def local_directional_losses(state, previous, target, valid, component, label,
                             variant, weights=DirectionalLossWeights()):
    if variant == 'L0':
        return directional_losses(state, previous, target, valid, weights, threshold=.5)
    if variant not in ('L1', 'L2'):
        raise ValueError('Expected L0, L1, or L2')
    if any(t.shape != target.shape for t in (state.probabilities, state.direction, previous, valid, component)):
        raise ValueError('All masks must align')
    if label not in (0, 1):
        raise ValueError('One positive or negative click per sample is required')
    valid = valid.bool()
    if not valid.any():
        raise ValueError('No valid pixels')
    inside = component.detach().bool() & valid
    outside = valid & ~inside
    probability = state.probabilities
    zero = probability.sum() * 0
    if inside.any():
        bce = F.binary_cross_entropy_with_logits(state.logits, target, reduction='none')[inside].mean()
        intersection = (probability * target)[inside].sum()
        dice = 1 - (2 * intersection + 1) / (probability[inside].sum() + target[inside].sum() + 1)
    else:
        bce = dice = zero
    preserve = (probability - previous.detach()).abs()[outside].mean() if outside.any() else zero
    direction = F.smooth_l1_loss(state.direction[inside], torch.full_like(state.direction[inside], 2 * label - 1), beta=1.) \
        if variant == 'L2' and inside.any() else state.direction.sum() * 0
    loss = weights.bce * bce + weights.dice * dice + weights.direction * direction + weights.preserve * preserve
    return dict(loss=loss, bce=bce, dice=dice, direction=direction, preserve=preserve)
