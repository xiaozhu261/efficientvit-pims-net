"""First-click frozen SAM, followed by recurrent ADD/REMOVE/KEEP mask edits."""

from dataclasses import dataclass

import torch
from torch import nn

from .click_encoder import ClickEncoder
from .direct_mask_decoder import DirectMaskDecoder
from .edit_intent_decoder import EditIntentDecoder
from .mask_editor import MaskEditor
from .sam_initializer import FrozenSAMInitializer
from .spatial_state_memory import SpatialStateMemory


@dataclass(frozen=True)
class InteractionState:
    image_features: torch.Tensor
    image_sizes: torch.Tensor
    points: torch.Tensor
    labels: torch.Tensor
    logits: torch.Tensor
    hidden: torch.Tensor
    probabilities: torch.Tensor
    edit_probabilities: torch.Tensor | None = None


class PIMSNet(nn.Module):
    def __init__(self, variant="a3"):
        super().__init__()
        if variant not in ("a1", "a2", "a3"):
            raise ValueError("variant must be a1 (direct), a2 (no history), or a3 (full)")
        self.variant = variant
        self.sam_initializer = FrozenSAMInitializer()
        self.click_encoder = ClickEncoder()
        self.spatial_memory = SpatialStateMemory()
        if variant == "a1":
            self.direct_mask_decoder = DirectMaskDecoder()
        else:
            self.edit_intent_decoder = EditIntentDecoder()
        self.mask_editor = MaskEditor()

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _click(self, features, points, labels, sizes):
        points = torch.as_tensor(points, device=features.device, dtype=torch.float32)
        labels = torch.as_tensor(labels, device=features.device)
        if tuple(points.shape) != (features.shape[0], 1, 2) or tuple(labels.shape) != (features.shape[0], 1):
            raise ValueError("Each step requires points [B,1,2] and labels [B,1]")
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError("Each image must have one valid click (label 0 or 1) per step")
        self.click_encoder(points, labels, sizes)  # Validate coordinates before SAM or refinement.
        return points, labels

    def initial_state(self, features, image_sizes, points, labels):
        """Initialize from cached frozen image embeddings; bypass all edit modules."""
        points, labels = self._click(features, points, labels, image_sizes)
        logits = self.sam_initializer(features, points, labels, image_sizes)
        hidden = self.spatial_memory.reset_state(features)
        return InteractionState(features, image_sizes, points, labels, logits, hidden, logits.sigmoid())

    def initialize(self, images, points, labels, image_sizes=None):
        features, sizes = self.sam_initializer.encode(images, image_sizes)
        return self.initial_state(features, sizes, points, labels)

    def refine(self, state: InteractionState, points, labels) -> InteractionState:
        """Append exactly one click per image. Keep cross-step gradients for BPTT."""
        points, labels = self._click(state.image_features, points, labels, state.image_sizes)
        points = torch.cat((state.points, points), dim=1)
        labels = torch.cat((state.labels, labels), dim=1)
        clicks = self.click_encoder(points, labels, state.image_sizes)
        history = torch.zeros_like(state.hidden) if self.variant == "a2" else state.hidden
        hidden = self.spatial_memory(state.image_features, clicks, state.logits, history)
        if self.variant == "a1":
            logits = self.direct_mask_decoder(state.image_features, hidden, state.probabilities)
            return InteractionState(state.image_features, state.image_sizes, points, labels,
                                    logits, hidden, logits.sigmoid(), None)
        intents = self.edit_intent_decoder(state.image_features, hidden, state.probabilities)
        probabilities = self.mask_editor(state.probabilities, intents)
        # Preserve the existing logits-based loss/postprocessing API. Only this
        # conversion clips endpoints; the stored probability update is exact.
        logits = torch.logit(probabilities, eps=1e-6)
        return InteractionState(state.image_features, state.image_sizes, points, labels, logits, hidden, probabilities, intents)

    def forward(self, images, points, labels, image_sizes=None):
        """Teacher-provided click sequence [B,T,2]/[B,T] -> [B,T,1,256,256].

        Original-image coordinates, one real click per sample per step (T>=1).
        No padding or click simulation: batches must use a common click count.
        Step 1 is frozen; steps 2..T retain the recurrent training graph.
        """
        points = torch.as_tensor(points)
        labels = torch.as_tensor(labels)
        if points.ndim != 3 or points.shape[-1] != 2 or points.shape[1] < 1 or labels.shape != points.shape[:2]:
            raise ValueError("Expected nonempty points [B,T,2] and labels [B,T]")
        state = self.initialize(images, points[:, :1], labels[:, :1], image_sizes)
        outputs = [state.logits]
        for index in range(1, points.shape[1]):
            state = self.refine(state, points[:, index:index + 1], labels[:, index:index + 1])
            outputs.append(state.logits)
        return torch.stack(outputs, dim=1)
