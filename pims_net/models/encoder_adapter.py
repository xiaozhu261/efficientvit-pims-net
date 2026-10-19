"""Frozen official EfficientViT-SAM-L0 image encoder with its original transform."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from .baseline import build_baseline


class EncoderAdapter(nn.Module):
    """Produce [B, 256, 64, 64] embeddings using local official L0 weights.

    Raw inputs are RGB uint8 HWC numpy images (one image or a sequence with
    potentially different sizes). Tensor inputs must already be preprocessed
    float tensors of shape [B, 3, 512, 512], e.g. from ``preprocess``.
    Only the complete image_encoder is retained: backbone, neck, and final norm.
    Move this module with .to(device); inputs follow its device and dtype.
    """

    def __init__(self) -> None:
        super().__init__()
        sam = build_baseline("efficientvit-sam-l0")
        self.image_encoder = sam.image_encoder
        self.transform = sam.transform
        self.input_size = sam.image_size[1]
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        """Keep BatchNorm statistics fixed, including under parent.train()."""
        return super().train(False)

    def preprocess(self, images: np.ndarray | Sequence[np.ndarray]) -> torch.Tensor:
        """Apply the exact official resize, ToTensor, normalize, and corner pad.

        Returns a CPU float32 batch. No BGR conversion or extra scaling occurs.
        """
        if isinstance(images, np.ndarray):
            images = [images]
        if not isinstance(images, Sequence):
            raise TypeError("Expected an RGB uint8 numpy image or a sequence of images")
        if len(images) == 0:
            raise ValueError("The image batch must not be empty")
        processed = []
        for image in images:
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                raise TypeError("Raw images must be RGB uint8 numpy arrays")
            if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
                raise ValueError("Raw images must have nonempty shape [H, W, 3]")
            processed.append(self.transform(image))
        return torch.stack(processed)

    @torch.no_grad()
    def forward(self, images: torch.Tensor | np.ndarray | Sequence[np.ndarray]) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            batch = images
            if batch.ndim != 4 or batch.shape[0] == 0 or tuple(batch.shape[1:]) != (
                3, self.input_size, self.input_size
            ):
                raise ValueError(f"Preprocessed tensors must have shape [B, 3, {self.input_size}, {self.input_size}]")
            if not batch.is_floating_point():
                raise TypeError("Tensor inputs must be preprocessed floating-point images")
        else:
            batch = self.preprocess(images)
        parameter = next(self.image_encoder.parameters())
        batch = batch.to(device=parameter.device, dtype=parameter.dtype)
        return self.image_encoder(batch)
