"""Positive/negative Gaussian click maps aligned with the official L0 transform."""

import math

import torch
from torch import nn


class ClickEncoder(nn.Module):
    """Encode clicks as [B, 2, 64, 64]; channels are positive, then negative.

    Coordinates are original-image pixel (x, y), with labels 1=positive,
    0=negative, -1=padding. image_size is (height, width), either a shared
    pair or a [B, 2] tensor. The default describes the padded 512x512 canvas.
    Original-image coordinates follow L0's longest-side resize to 512 and
    bottom/right padding, rather than stretching rectangular images.
    Multiple clicks of the same sign combine by maximum. sigma is measured
    in 64x64 feature-grid pixels. Missing signs produce an all-zero channel.
    """

    def __init__(self, sigma: float = 2.0) -> None:
        super().__init__()
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        self.sigma = float(sigma)
        y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        self.register_buffer("grid_x", x.float(), persistent=False)
        self.register_buffer("grid_y", y.float(), persistent=False)

    def forward(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        image_size: tuple[int, int] | torch.Tensor = (512, 512),
    ) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 2 or points.shape[0] == 0:
            raise ValueError("points must have shape [B, N, 2] with B > 0")
        if labels.shape != points.shape[:2]:
            raise ValueError("labels must have shape [B, N] matching points")
        labels = labels.to(device=points.device)
        if not ((labels == -1) | (labels == 0) | (labels == 1)).all():
            raise ValueError("labels must be -1 (padding), 0 (negative), or 1 (positive)")
        sizes = torch.as_tensor(image_size, device=points.device, dtype=torch.float32)
        if sizes.shape == (2,):
            sizes = sizes.expand(points.shape[0], -1)
        if sizes.shape != (points.shape[0], 2):
            raise ValueError("image_size must be (H, W) or a [B, 2] tensor")
        if not (torch.isfinite(sizes) & (sizes > 0) & (sizes == sizes.round())).all():
            raise ValueError("Image dimensions must be finite positive integers")

        valid = labels != -1
        xy = points.to(dtype=torch.float32)
        bounds = sizes.flip(-1).unsqueeze(1)
        inside = (torch.isfinite(xy) & (xy >= 0) & (xy < bounds)).all(dim=-1)
        if (valid & ~inside).any():
            raise ValueError("Valid clicks must lie inside their original image")
        if points.shape[1] == 0:
            return xy.new_zeros((points.shape[0], 2, 64, 64))
        # Match SamResize.get_preprocess_shape rounding, then map to the /8 grid.
        resized_hw = torch.floor(sizes * (512.0 / sizes.amax(dim=-1, keepdim=True)) + 0.5)
        if (resized_hw < 1).any():
            raise ValueError("Image aspect ratio is too extreme for the official 512 resize")
        scale_xy = (resized_hw / sizes).flip(-1) / 8.0
        xy = torch.where(valid.unsqueeze(-1), xy, 0.0) * scale_xy.unsqueeze(1)
        grid_x = self.grid_x.to(device=points.device, dtype=torch.float32)
        grid_y = self.grid_y.to(device=points.device, dtype=torch.float32)
        distance_sq = (grid_x - xy[..., 0, None, None]).square() + (grid_y - xy[..., 1, None, None]).square()
        gaussians = torch.exp(-distance_sq / (2.0 * self.sigma**2))
        return torch.stack(
            [gaussians.masked_fill((labels != sign)[..., None, None], 0).amax(dim=1) for sign in (1, 0)],
            dim=1,
        )
