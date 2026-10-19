"""Second independent capacity step: width4 with four residual blocks."""

import torch
from torch import nn

from .directional_pims_net import DirectionalEditDecoder, DirectionalPIMSNet
from .directional_wide import DecoderResidualBlock


class LargeDirectionalEditDecoder(DirectionalEditDecoder):
    """Hidden widths 256/128/64; four 128-channel residual blocks at 64x64."""

    def __init__(self):
        nn.Module.__init__(self)
        self.fusion = nn.Sequential(
            nn.Conv2d(385, 256, 1), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
            *[DecoderResidualBlock(128) for _ in range(4)])
        self.decoder = nn.Sequential(
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False))
        self.head = nn.Conv2d(64, 2, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[1] = -2.


class LargeDirectionalPIMSNet(DirectionalPIMSNet):
    def __init__(self):
        super().__init__()
        self.variant = 'directional_large'
        self.directional_decoder = LargeDirectionalEditDecoder()
