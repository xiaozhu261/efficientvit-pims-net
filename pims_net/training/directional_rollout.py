"""RITM closed-loop training for the independent directional experiment."""

import numpy as np
import torch

from .click_simulator import RITMClickSimulator
from .directional_losses import DirectionalLossWeights, directional_losses
from .interactive_losses import prepare_target


def directional_rollout(model, image, mask, max_clicks=5, rng=None, weights=DirectionalLossWeights()):
    if not 1 <= max_clicks <= 5:
        raise ValueError('max_clicks must be in [1,5]')
    mask = np.asarray(mask, dtype=bool)
    if image.shape[:2] != mask.shape or not mask.any():
        raise ValueError('Image and nonempty GT must align')
    clicker = RITMClickSimulator(mask, rng)
    prediction = np.zeros(mask.shape, np.float32)
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
            previous = state.probabilities
            state = model.refine(state, [[[x, y]]], [[label]])
            terms.append(directional_losses(state, previous, target, valid, weights))
        with torch.no_grad():
            prediction = model.sam_initializer.postprocess(state.logits, state.image_sizes)[0].sigmoid()[0, 0].cpu().numpy()
        foreground = prediction > .49
        iou = float((mask & foreground).sum() / max(1, (mask | foreground).sum()))
        trace.append(dict(click=index + 1, x=x, y=y, label=label, iou=iou))
    loss = torch.stack([term['loss'] for term in terms]).mean() if terms else None
    metrics = {name: float(torch.stack([term[name].detach() for term in terms]).mean()) if terms else 0.
               for name in ('loss', 'bce', 'dice', 'direction', 'preserve')}
    return dict(loss=loss, metrics=metrics, clicks=trace, state=state)
