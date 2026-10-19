"""Single-image interactive inference with cached features and recurrent state."""

import numpy as np
import torch


class PIMSPredictor:
    def __init__(self, model):
        self.model = model.eval()
        self.reset_image()

    def reset_image(self):
        self.features = None
        self.image_sizes = None
        self.state = None

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        self.reset_image()
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("set_image expects a single RGB uint8 HWC numpy image")
        self.model.eval()
        self.features, self.image_sizes = self.model.sam_initializer.encode(image)

    @torch.no_grad()
    def add_click(self, x: float, y: float, label: int = 1):
        if self.features is None:
            raise RuntimeError("Call set_image before adding clicks")
        self.model.eval()
        points, labels = [[[x, y]]], [[label]]
        if self.state is None:
            self.state = self.model.initial_state(self.features, self.image_sizes, points, labels)
        else:
            self.state = self.model.refine(self.state, points, labels)
        original_logits = self.model.sam_initializer.postprocess(self.state.logits, self.image_sizes)[0]
        return {
            "low_res_logits": self.state.logits,
            "low_res_probabilities": self.state.probabilities,
            "edit_probabilities": self.state.edit_probabilities,
            "logits": original_logits,
            "mask": original_logits > self.model.sam_initializer.sam.mask_threshold,
            "click_count": self.state.points.shape[1],
        }
