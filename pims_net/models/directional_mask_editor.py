"""Independent bounded signed probability editor; no ADD/REMOVE competition."""

import torch
from torch import nn


class DirectionalMaskEditor(nn.Module):
    """D in [-1,1], S in [0,1]; all inputs [B,1,H,W].

    P_new = P + S * where(D>=0, D*(1-P), D*P).
    The available distance to 0/1 bounds the step without a hard clamp.
    D=0 (or S=0) is exact KEEP. This module consumes activated D/S.
    """

    def forward(self, previous, direction, strength):
        if previous.ndim != 4 or previous.shape[1] != 1 or min(previous.shape) < 1:
            raise ValueError('Expected previous probabilities [B,1,H,W]')
        for name, value, lower in (('previous', previous, 0), ('direction', direction, -1), ('strength', strength, 0)):
            if value.shape != previous.shape or value.device != previous.device:
                raise ValueError(f'{name} must match previous shape and device')
            if not value.is_floating_point() or not (torch.isfinite(value) & (value >= lower) & (value <= 1)).all():
                raise ValueError(f'{name} must be finite floating point in [{lower},1]')
        signed_step = torch.where(direction >= 0, direction * (1 - previous), direction * previous)
        return previous + strength * signed_step

    def update(self, previous, previous_logits, direction, strength):
        """Return probabilities and logits, preserving raw logits on exact KEEP.

        The equality branch intentionally has no mask-loss gradient to D/S at
        exact zero; explicit direction supervision can move D away from zero.
        No detached straight-through estimator is used.
        """
        if previous_logits.shape != previous.shape or previous_logits.device != previous.device:
            raise ValueError('previous_logits must align with previous probabilities')
        if not torch.isfinite(previous_logits).all():
            raise ValueError('previous_logits must be finite')
        probability = self(previous, direction, strength)
        keep = (direction == 0) | (strength == 0)
        logits = torch.where(keep, previous_logits, torch.logit(probability, eps=1e-6))
        return probability, logits
