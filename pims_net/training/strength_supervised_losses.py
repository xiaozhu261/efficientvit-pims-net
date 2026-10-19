"""S0/S1/S2 preserve the original L0 loss and add only raw Strength-logit BCE."""

import torch
from torch.nn import functional as F

from .directional_losses import DirectionalLossWeights, directional_losses


def harmful_strength_region(previous,target,valid,direction):
    old_fg=previous.detach()>.5
    truth=target.detach()>.5
    d=direction.detach()
    return valid.detach().bool() & ((truth&old_fg&(d<0)) | (~truth&~old_fg&(d>0)))


def strength_supervised_losses(state, previous, target, valid, component, label, raw,
                               variant, weights=DirectionalLossWeights(), activation_weight=.05,
                               suppression_weight=.01):
    if variant not in ('S0','S1','S2'):
        raise ValueError('Expected S0, S1 or S2')
    if raw.shape!=(target.shape[0],2,*target.shape[2:]) or component.shape!=target.shape:
        raise ValueError('Raw two-channel head and component must align with GT')
    if label not in (0,1):
        raise ValueError('Expected one binary click sign')
    if activation_weight<0 or suppression_weight<0:
        raise ValueError('Nonnegative Strength loss weights required')
    base=directional_losses(state,previous,target,valid,weights,threshold=.5)
    z=raw[:,1:2]
    gate=component.detach().bool()&valid.bool()
    harmful=harmful_strength_region(previous,target,valid,state.direction)
    zero=z.sum()*0
    activate=F.binary_cross_entropy_with_logits(z[gate],torch.ones_like(z[gate])) \
        if variant in ('S1','S2') and label==1 and gate.any() else zero
    suppress=F.binary_cross_entropy_with_logits(z[harmful],torch.zeros_like(z[harmful])) \
        if variant=='S2' and harmful.any() else zero
    total=base['loss']+activation_weight*activate+suppression_weight*suppress
    return dict(base,loss=total,strength_activate=activate,strength_suppress=suppress)
