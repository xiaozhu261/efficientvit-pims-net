"""Decode image, recurrent state, and previous mask into signed edit intent."""

import torch
from torch import nn
from torch.nn import functional as F


class EditIntentDecoder(nn.Module):
    """Return [B,3,256,256] probabilities in ADD, REMOVE, KEEP channel order.

    Inputs are image features [B,256,64,64], ConvGRU state [B,128,64,64],
    and previous mask *probabilities* [B,1,256,256], not logits or binary labels.
    """

    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(385, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(size=(128, 128), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
        )
        self.head = nn.Conv2d(16, 3, 1)
        # Start near KEEP, with nonzero weights so every layer can learn immediately.
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[2] = 4.0

    def decode_logits(self, image_features, hidden, previous_mask):
        if image_features.ndim != 4 or image_features.shape[0] == 0 or tuple(image_features.shape[1:]) != (256, 64, 64):
            raise ValueError("image_features must have shape [B,256,64,64] with B > 0")
        batch_size = image_features.shape[0]
        if tuple(hidden.shape) != (batch_size, 128, 64, 64):
            raise ValueError("hidden must have shape [B,128,64,64]")
        if tuple(previous_mask.shape) != (batch_size, 1, 256, 256):
            raise ValueError("previous_mask must have shape [B,1,256,256]")
        if not all(value.is_floating_point() for value in (image_features, hidden, previous_mask)):
            raise TypeError("Decoder inputs must be floating-point tensors")
        if hidden.device != image_features.device or previous_mask.device != image_features.device:
            raise ValueError("Decoder inputs must share a device")
        if not (torch.isfinite(previous_mask) & (previous_mask >= 0) & (previous_mask <= 1)).all():
            raise ValueError("previous_mask must contain probabilities in [0,1], not logits")
        previous = F.interpolate(previous_mask, size=(64, 64), mode="bilinear", align_corners=False)
        inputs = torch.cat((image_features, hidden.to(image_features.dtype), previous.to(image_features.dtype)), dim=1)
        return self.head(self.decoder(self.fusion(inputs)))

    def forward(self, image_features, hidden, previous_mask):
        logits = self.decode_logits(image_features, hidden, previous_mask)
        # Compute normalized probabilities in float32 under mixed precision too.
        return logits.softmax(dim=1, dtype=torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype)
