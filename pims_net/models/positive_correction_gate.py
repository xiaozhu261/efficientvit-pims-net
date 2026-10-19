"""Independent, additive positive correction on a completely frozen S0 model."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .directional_pims_net import DirectionalState
from .directional_wide import DecoderResidualBlock


class PositiveCorrectionGate(nn.Module):
    """Latest click, S0 image features/updated memory, and S0 proposal -> ADD logit."""

    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(387, 128, 1), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            DecoderResidualBlock(64), DecoderResidualBlock(64))
        self.decoder = nn.Sequential(
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False))
        self.head = nn.Conv2d(32, 1, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.constant_(self.head.bias, -4.)

    def forward(self, image_features, latest_click, hidden, base_probability):
        mask = F.interpolate(base_probability, size=(64, 64), mode='bilinear', align_corners=False)
        return self.head(self.decoder(self.fusion(torch.cat((image_features, latest_click, hidden, mask), 1))))


def positive_update(base_probability, base_logits, gate_logits, labels, enabled=True):
    """Negative clicks / disabled gate preserve original P AND raw logits bitwise."""
    positive = labels.reshape(-1, 1, 1, 1).bool()
    gate = torch.sigmoid(gate_logits) * positive if enabled else torch.zeros_like(gate_logits)
    candidate = base_probability + gate * (1 - base_probability)
    changed = positive & (candidate != base_probability) if enabled else torch.zeros_like(candidate, dtype=torch.bool)
    probability = torch.where(changed, candidate, base_probability)
    # Clipping logit(P) near one must never lower a positive S0 logit. This
    # also preserves monotonicity through official bilinear postprocessing.
    proposed_logits = torch.maximum(torch.logit(probability, eps=1e-6), base_logits)
    logits = torch.where(changed, proposed_logits, base_logits)
    return probability, logits, gate


@dataclass(frozen=True)
class PositiveGateState(DirectionalState):
    gate: torch.Tensor | None = None
    gate_logits: torch.Tensor | None = None
    base_probabilities: torch.Tensor | None = None
    base_logits: torch.Tensor | None = None


class PositiveCorrectionPIMSNet(nn.Module):
    """Frozen S0 with Gate-only optimization; corrected state feeds the next click.

    The first frozen SAM result is retained. S0 is evaluated under no_grad at
    every later click, so Gate training is truncated at the frozen transition.
    There are no GT-dependent operations in this inference module.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.gate = PositiveCorrectionGate()

    @property
    def sam_initializer(self):
        return self.base.sam_initializer

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def initialize(self, *args, **kwargs):
        return self.base.initialize(*args, **kwargs)

    def refine(self, state, points, labels, enabled=True):
        with torch.no_grad():
            base = self.base.refine(state, points, labels)
            latest = self.base.click_encoder(base.points[:, -1:], base.labels[:, -1:], base.image_sizes)
        raw = self.gate(base.image_features, latest, base.hidden, base.probabilities)
        probability, logits, gate = positive_update(base.probabilities, base.logits, raw, base.labels[:, -1], enabled)
        return PositiveGateState(**{name: getattr(base, name) for name in DirectionalState.__dataclass_fields__
                                   if name not in ('probabilities', 'logits')},
            probabilities=probability, logits=logits, gate=gate, gate_logits=raw,
            base_probabilities=base.probabilities, base_logits=base.logits)
