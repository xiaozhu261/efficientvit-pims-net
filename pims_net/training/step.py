"""Differentiable supervised training interface; optimizer ownership stays with caller."""

import torch
from torch.nn import functional as F


def training_step(model, images, points, labels, targets, image_sizes=None):
    """BCE + soft Dice on steps 2..T, averaged equally across samples and steps.

    targets: float/binary [B,1,256,256] on the official padded mask canvas.
    Click sequences need at least two valid clicks; no automatic click sampling.
    Caller performs optimizer.zero_grad(), loss.backward(), optimizer.step().
    """
    if torch.as_tensor(points).ndim != 3 or torch.as_tensor(points).shape[1] < 2:
        raise ValueError("Training requires at least two clicks; the first-click path is frozen")
    model.train()
    logits = model(images, points, labels, image_sizes)
    targets = torch.as_tensor(targets, device=logits.device, dtype=logits.dtype)
    if tuple(targets.shape) != (logits.shape[0], 1, 256, 256):
        raise ValueError("targets must have shape [B,1,256,256] on the padded canvas")
    if not (torch.isfinite(targets) & (targets >= 0) & (targets <= 1)).all():
        raise ValueError("targets must contain finite values in [0,1]")
    edited = logits[:, 1:]
    expected = targets.unsqueeze(1).expand_as(edited)
    bce = F.binary_cross_entropy_with_logits(edited, expected)
    probabilities = edited.sigmoid()
    dims = (-3, -2, -1)
    intersection = (probabilities * expected).sum(dim=dims)
    dice = (1 - (2 * intersection + 1) / (probabilities.sum(dim=dims) + expected.sum(dim=dims) + 1)).mean()
    return {"loss": bce + dice, "bce": bce, "dice": dice, "logits": logits}
