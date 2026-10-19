"""Residual clicked-FN supervision, after the frozen S0 proposal."""

import torch
from torch.nn import functional as F

from pims_net.evaluate_oracle import clicked_error_component


def gate_target(base_probability, target, valid, click, image_shape):
    if click['label'] == 1:
        component, metadata = clicked_error_component(
            base_probability.detach()[0, 0].cpu().numpy(),
            target.detach()[0, 0].cpu().numpy().astype(bool),
            valid.detach()[0, 0].cpu().numpy().astype(bool), click, image_shape)
        selected = torch.as_tensor(component, device=target.device)[None, None]
    else:
        selected, metadata = torch.zeros_like(valid, dtype=torch.bool), None
    return selected, metadata


def positive_gate_losses(state, target, valid, selected, variant):
    if variant not in ('G1', 'G2'):
        raise ValueError('Only G1/G2 train a gate')
    valid = valid.bool()
    if not valid.any() or (selected & ~valid).any():
        raise ValueError('Invalid supervision support')
    pixel = F.binary_cross_entropy_with_logits(state.gate_logits, selected.float(), reduction='none')
    regions = [pixel[region].mean() for region in (selected & valid, ~selected & valid) if region.any()]
    gate = torch.stack(regions).mean()
    bce = F.binary_cross_entropy_with_logits(state.logits, target, reduction='none')[valid].mean()
    p = state.probabilities
    dice = 1 - (2 * (p * target * valid).sum() + 1) / ((p * valid).sum() + (target * valid).sum() + 1)
    total = gate + (bce + dice if variant == 'G2' else 0)
    return dict(loss=total, gate=gate, bce=bce, dice=dice)
