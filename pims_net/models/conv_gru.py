"""Spatial recurrent state for interactive mask refinement."""

import torch
from torch import nn


class ConvGRU(nn.Module):
    def __init__(self, input_channels: int = 35, hidden_channels: int = 32):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(f"inputs must have shape [B, {self.input_channels}, H, W]")
        shape = (inputs.shape[0], self.hidden_channels, *inputs.shape[-2:])
        if hidden is None:
            hidden = inputs.new_zeros(shape)
        if tuple(hidden.shape) != shape:
            raise ValueError(f"hidden must have shape {shape}")
        reset, update = self.gates(torch.cat((inputs, hidden), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = self.candidate(torch.cat((inputs, reset * hidden), dim=1)).tanh()
        return (1 - update) * hidden + update * candidate
