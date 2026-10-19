"""RITM-style online error-region clicks, expressed in original-image (x,y)."""

import cv2
import numpy as np


class RITMClickSimulator:
    """Choose FN/FP by maximum distance, sample where distance > half maximum.

    Uses the RITM training sampler's 0.49 threshold and padded L2 transform
    (maskSize=5). Adds duplicate exclusion and explicit early stopping.
    Reference: original isegm/engine/trainer.py:get_next_points, mirrored at
    https://github.com/ggsonic/ritm_interactive_segmentation
    """

    def __init__(self, target, rng=None):
        self.target = np.asarray(target, dtype=bool)
        if self.target.ndim != 2 or min(self.target.shape) == 0:
            raise ValueError("target must be a nonempty 2D binary mask")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.available = np.ones_like(self.target, dtype=bool)

    def next_click(self, prediction):
        prediction = np.asarray(prediction)
        if prediction.shape != self.target.shape or not np.isfinite(prediction).all():
            raise ValueError("prediction must be finite and match target dimensions")
        foreground = prediction > 0.49
        errors = (self.target & ~foreground, ~self.target & foreground)
        distances = []
        for error in errors:
            padded = np.pad(error.astype(np.uint8), 1)
            distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
            distances.append(np.where(self.available, distance, 0))
        positive = distances[0].max() > distances[1].max()
        distance = distances[0 if positive else 1]
        maximum = distance.max()
        if maximum <= 0:
            return None
        candidates = np.argwhere(distance > maximum / 2)
        y, x = candidates[self.rng.integers(len(candidates))]
        self.available[y, x] = False
        return int(x), int(y), int(positive)
