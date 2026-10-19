"""Decode recurrent features into a residual edit to previous mask logits."""

import torch
from torch import nn


class EditDecoder(nn.Module):
    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.layers = nn.Sequential(
            nn.Conv2d(hidden_channels, 16, 3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(128, 128), mode="bilinear", align_corners=False),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 1),
        )
        # Small, nonzero edits preserve initial SAM predictions while allowing
        # gradients to reach every recurrent layer from the first optimizer step.
        nn.init.normal_(self.layers[-1].weight, std=1e-3)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, hidden: torch.Tensor, previous_logits: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or tuple(hidden.shape[1:]) != (self.hidden_channels, 64, 64):
            raise ValueError(f"hidden must have shape [B, {self.hidden_channels}, 64, 64]")
        if tuple(previous_logits.shape) != (hidden.shape[0], 1, 256, 256):
            raise ValueError("previous_logits must have shape [B, 1, 256, 256]")
        return previous_logits + self.layers(hidden)
