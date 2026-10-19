"""Independent decoder-capacity ablation; shared recurrence/editor unchanged."""

import torch
from torch import nn

from .directional_pims_net import DirectionalEditDecoder, DirectionalPIMSNet


class DecoderResidualBlock(nn.Module):
    """Two 3x3 convolutions, GELU between them, identity skip; no normalization."""

    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, value):
        return value + self.branch(value)


class WideDirectionalEditDecoder(DirectionalEditDecoder):
    """Double hidden channels 64/32/16 -> 128/64/32; add 2 blocks at 64x64."""

    def __init__(self):
        nn.Module.__init__(self)
        self.fusion = nn.Sequential(nn.Conv2d(385, 128, 1), nn.GELU(),
                                    nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
                                    DecoderResidualBlock(64), DecoderResidualBlock(64))
        self.decoder = nn.Sequential(
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False))
        self.head = nn.Conv2d(32, 2, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[1] = -2.


class WideDirectionalPIMSNet(DirectionalPIMSNet):
    def __init__(self):
        # Same seed gives exactly the original initial memory weights. Only
        # replace the decoder; inherit refinement, first-click and editor.
        super().__init__()
        self.variant = 'directional_wide'
        self.directional_decoder = WideDirectionalEditDecoder()
