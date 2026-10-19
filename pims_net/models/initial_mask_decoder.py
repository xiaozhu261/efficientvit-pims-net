"""Lightweight trainable CNN for image embeddings and signed click maps."""

import torch
from torch import nn


class InitialMaskDecoder(nn.Module):
    """Fuse [B,256,64,64] embeddings and [B,2,64,64] maps into mask logits.

    Returns [B,1,256,256] logits on the padded image canvas (no sigmoid).
    All decoder weights are newly initialized and trainable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(258, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(size=(128, 128), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, image_features: torch.Tensor, click_maps: torch.Tensor) -> torch.Tensor:
        if image_features.ndim != 4 or image_features.shape[0] == 0 or tuple(image_features.shape[1:]) != (256, 64, 64):
            raise ValueError("image_features must have shape [B, 256, 64, 64] with B > 0")
        if tuple(click_maps.shape) != (image_features.shape[0], 2, 64, 64):
            raise ValueError("click_maps must have shape [B, 2, 64, 64] matching the feature batch")
        if not image_features.is_floating_point() or not click_maps.is_floating_point():
            raise TypeError("Image features and click maps must be floating-point tensors")
        click_maps = click_maps.to(device=image_features.device, dtype=image_features.dtype)
        return self.decoder(self.fusion(torch.cat((image_features, click_maps), dim=1)))
