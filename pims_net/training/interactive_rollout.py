"""Closed-loop supervision: recompute the next click from each actual prediction."""

import numpy as np
import torch

from .click_simulator import RITMClickSimulator
from .interactive_losses import LossWeights, interactive_losses, prepare_target


def interactive_rollout(model, image, mask, max_clicks=5, rng=None, weights=LossWeights()):
    """One object, up to five total clicks, full BPTT over all trainable edits.

    The first mask is produced by the frozen official SAM decoder. Sampling
    and edit targets are detached, but recurrent probabilities/state are not.
    If no correction is needed after click 1, loss is None (no optimizer update).
    """
    if not 1 <= max_clicks <= 5:
        raise ValueError("max_clicks must be between 1 and 5")
    mask = np.asarray(mask, dtype=bool)
    if image.shape[:2] != mask.shape or not mask.any():
        raise ValueError("Image/target dimensions must match and foreground must be nonempty")
    clicker = RITMClickSimulator(mask, rng)
    prediction = np.zeros(mask.shape, dtype=np.float32)
    state, terms, trace = None, [], []
    for index in range(max_clicks):
        click = clicker.next_click(prediction)
        if click is None:
            break
        x, y, label = click
        if state is None:
            state = model.initialize(image, [[[x, y]]], [[label]])
            target, valid = prepare_target(mask, state.logits.device)
        else:
            old_probabilities = state.probabilities
            state = model.refine(state, [[[x, y]]], [[label]])
            terms.append(interactive_losses(state, old_probabilities, target, valid, weights))
        with torch.no_grad():
            original_logits = model.sam_initializer.postprocess(state.logits, state.image_sizes)[0]
            prediction = original_logits.sigmoid()[0, 0].cpu().numpy()
        foreground = prediction > 0.49
        union = np.logical_or(mask, foreground).sum()
        iou = float(np.logical_and(mask, foreground).sum() / max(1, union))
        trace.append({"click": index + 1, "x": x, "y": y, "label": label, "iou": iou})
    loss = torch.stack([term["loss"] for term in terms]).mean() if terms else None
    metrics = {name: float(torch.stack([term[name].detach() for term in terms]).mean()) if terms else 0.0
               for name in ("loss", "bce", "dice", "edit", "preserve")}
    return {"loss": loss, "metrics": metrics, "clicks": trace, "state": state}
