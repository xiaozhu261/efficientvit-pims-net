"""Parameter-free, differentiable ADD / REMOVE / KEEP probability editing."""

import torch
from torch import nn


class MaskEditor(nn.Module):
    """Apply P_new = KEEP * P_old + ADD, returning probabilities without clipping.

    previous_mask: [B,1,H,W] probabilities in [0,1].
    edit_probabilities: [B,3,H,W], ordered ADD, REMOVE, KEEP and summing to 1.
    REMOVE contributes zero foreground probability, via the softmax competition.
    """

    ADD, REMOVE, KEEP = 0, 1, 2

    def forward(self, previous_mask, edit_probabilities):
        if previous_mask.ndim != 4 or previous_mask.shape[1] != 1 or min(previous_mask.shape) < 1:
            raise ValueError("previous_mask must have nonempty shape [B,1,H,W]")
        if tuple(edit_probabilities.shape) != (previous_mask.shape[0], 3, *previous_mask.shape[-2:]):
            raise ValueError("edit_probabilities must have shape [B,3,H,W] matching previous_mask")
        if not previous_mask.is_floating_point() or not edit_probabilities.is_floating_point():
            raise TypeError("MaskEditor expects floating-point probabilities")
        if previous_mask.device != edit_probabilities.device:
            raise ValueError("Mask and edit probabilities must share a device")
        for value in (previous_mask, edit_probabilities):
            if not (torch.isfinite(value) & (value >= 0) & (value <= 1)).all():
                raise ValueError("Mask and edit values must be finite probabilities in [0,1]")
        total = edit_probabilities.sum(dim=1, keepdim=True)
        tolerance = max(1e-5, 4 * torch.finfo(edit_probabilities.dtype).eps)
        if not torch.allclose(total, torch.ones_like(total), rtol=0, atol=tolerance):
            raise ValueError("ADD, REMOVE, KEEP probabilities must sum to 1 at each pixel")
        return edit_probabilities[:, self.KEEP:self.KEEP + 1] * previous_mask + edit_probabilities[:, self.ADD:self.ADD + 1]
