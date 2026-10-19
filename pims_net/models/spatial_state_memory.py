"""Spatial memory with explicit per-image state, shared safely across sessions."""

import torch
from torch import nn
from torch.nn import functional as F

from .conv_gru import ConvGRU


class SpatialStateMemory(nn.Module):
    """Fuse image features, signed clicks, and previous logits into 128 channels.

    Inputs: image_features [B,256,64,64], click_maps [B,2,64,64],
    previous_logits [B,1,256,256], optional state [B,128,64,64].
    The caller owns the returned state and passes it to the next interaction.
    Use reset_state(features), or omit state, for a new image batch. No hidden
    state is stored on this shared module, so separate sessions cannot leak
    memory into one another. Training retains the full recurrent graph.
    """

    hidden_channels = 128

    def __init__(self):
        super().__init__()
        self.feature_projection = nn.Sequential(nn.Conv2d(256, 32, 1), nn.GELU())
        self.conv_gru = ConvGRU(input_channels=35, hidden_channels=self.hidden_channels)

    @staticmethod
    def _validate_features(image_features):
        if image_features.ndim != 4 or image_features.shape[0] == 0 or tuple(image_features.shape[1:]) != (256, 64, 64):
            raise ValueError("image_features must have shape [B,256,64,64] with B > 0")
        if not image_features.is_floating_point():
            raise TypeError("image_features must be floating point")

    def reset_state(self, image_features: torch.Tensor) -> torch.Tensor:
        """Allocate fresh zeros on the new image's device and dtype."""
        self._validate_features(image_features)
        return image_features.new_zeros((image_features.shape[0], self.hidden_channels, 64, 64))

    def forward(self, image_features, click_maps, previous_logits, state=None):
        self._validate_features(image_features)
        batch_size = image_features.shape[0]
        if tuple(click_maps.shape) != (batch_size, 2, 64, 64):
            raise ValueError("click_maps must have shape [B,2,64,64]")
        if tuple(previous_logits.shape) != (batch_size, 1, 256, 256):
            raise ValueError("previous_logits must have shape [B,1,256,256]")
        if not click_maps.is_floating_point() or not previous_logits.is_floating_point():
            raise TypeError("click_maps and previous_logits must be floating point")
        if state is None:
            state = self.reset_state(image_features)
        if tuple(state.shape) != (batch_size, self.hidden_channels, 64, 64):
            raise ValueError("state must have shape [B,128,64,64]")
        if state.device != image_features.device or state.dtype != image_features.dtype:
            raise ValueError("state must match image_features device and dtype; reset for new images")
        features = self.feature_projection(image_features)
        clicks = click_maps.to(device=features.device, dtype=features.dtype)
        previous = previous_logits.to(device=features.device, dtype=features.dtype)
        previous = F.interpolate(previous.sigmoid(), size=(64, 64), mode="bilinear", align_corners=False)
        return self.conv_gru(torch.cat((features, clicks, previous), dim=1), state)
