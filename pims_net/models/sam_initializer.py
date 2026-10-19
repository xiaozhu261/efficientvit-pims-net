"""Frozen official L0 image, prompt, and mask modules for the first click."""

import numpy as np
import torch
from torch import nn

from .baseline import build_baseline


class FrozenSAMInitializer(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam = build_baseline("efficientvit-sam-l0")
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, images, image_sizes=None):
        """Raw RGB uint8 images, or official-preprocessed [B,3,512,512].

        Tensor inputs require original (H,W) sizes, as click coordinates and
        output cropping cannot be inferred from already padded tensors.
        """
        parameter = next(self.sam.parameters())
        if isinstance(images, torch.Tensor):
            if images.ndim != 4 or tuple(images.shape[1:]) != (3, 512, 512) or images.shape[0] == 0:
                raise ValueError("Preprocessed images must have shape [B, 3, 512, 512]")
            if not images.is_floating_point():
                raise TypeError("Preprocessed images must be floating point")
            if image_sizes is None:
                raise ValueError("Original image_sizes are required for preprocessed tensors")
            batch = images
        else:
            images = [images] if isinstance(images, np.ndarray) else list(images)
            if not images:
                raise ValueError("Image batch must not be empty")
            for image in images:
                if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                    raise TypeError("Raw images must be RGB uint8 numpy arrays")
                if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
                    raise ValueError("Raw images must have nonempty shape [H, W, 3]")
            if image_sizes is not None:
                raise ValueError("Raw image sizes are inferred; omit image_sizes")
            image_sizes = [image.shape[:2] for image in images]
            batch = torch.stack([self.sam.transform(image) for image in images])
        sizes = torch.as_tensor(image_sizes, device=parameter.device, dtype=torch.float32)
        if sizes.shape == (2,):
            sizes = sizes.expand(batch.shape[0], -1)
        if sizes.shape != (batch.shape[0], 2) or not (
            torch.isfinite(sizes) & (sizes > 0) & (sizes == sizes.round())
        ).all():
            raise ValueError("image_sizes must be positive integer (H,W) or [B,2]")
        features = self.sam.image_encoder(batch.to(device=parameter.device, dtype=parameter.dtype))
        return features, sizes

    @torch.no_grad()
    def forward(self, features, points, labels, image_sizes, mask_input=None):
        """Decode point prompts and optional previous logits; return the best-IoU mask."""
        # SAM prompt coordinates use the 1024 canvas, even though L0's image
        # encoder consumes 512. This matches EfficientViTSamPredictor exactly.
        # Official numpy apply_coords computes in float64 before converting to
        # float32. Match it to avoid accumulating prompt rounding differences
        # when previous logits are fed back over multiple interactions.
        sizes64 = image_sizes.double()
        resized_hw = torch.floor(sizes64 * (1024.0 / sizes64.amax(-1, keepdim=True)) + 0.5)
        coordinates = (points.double() * (resized_hw / sizes64).flip(-1).unsqueeze(1)).float()
        masks = []
        for index in range(features.shape[0]):
            sparse, dense = self.sam.prompt_encoder(
                points=(coordinates[index:index + 1], labels[index:index + 1]), boxes=None,
                masks=None if mask_input is None else mask_input[index:index + 1]
            )
            candidates, scores = self.sam.mask_decoder(
                image_embeddings=features[index:index + 1],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
            )
            # Official decoder repeats a single image for its prompts; invoking
            # per image avoids the B*B expansion of a multi-image input batch.
            best = scores.argmax(dim=1)
            masks.append(candidates[torch.arange(1, device=features.device), best].unsqueeze(1))
        return torch.cat(masks)

    @torch.no_grad()
    def postprocess(self, logits, image_sizes):
        result = []
        for index, size in enumerate(image_sizes):
            original = tuple(int(value) for value in size.tolist())
            resized = tuple(int(value) for value in torch.floor(size * (1024 / size.max()) + 0.5).tolist())
            result.append(self.sam.postprocess_masks(logits[index:index + 1], resized, original))
        return result
