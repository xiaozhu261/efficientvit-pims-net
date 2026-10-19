"""Independent directional branch. Original PIMSNet/A3 remains unchanged."""

from dataclasses import dataclass

import torch
from torch import nn

from .click_encoder import ClickEncoder
from .directional_mask_editor import DirectionalMaskEditor
from .edit_intent_decoder import EditIntentDecoder
from .pims_net import InteractionState, PIMSNet
from .sam_initializer import FrozenSAMInitializer
from .spatial_state_memory import SpatialStateMemory


class DirectionalEditDecoder(EditIntentDecoder):
    """Reuse A3 trunk/geometry, independently train a 2-channel D/S head.

    Channel 0 -> tanh direction, channel 1 -> sigmoid strength. Both return
    [B,1,256,256]. No A3 research weights are loaded or copied.
    """

    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(16, 2, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[1] = -2.0  # Small initial strength; near-zero D, not a dead zero head.

    def forward(self, image_features, hidden, previous_mask):
        logits = self.decode_logits(image_features, hidden, previous_mask)
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        return logits[:, :1].tanh(), logits[:, 1:2].sigmoid()


@dataclass(frozen=True)
class DirectionalState(InteractionState):
    direction: torch.Tensor | None = None
    strength: torch.Tensor | None = None


class DirectionalPIMSNet(PIMSNet):
    """Same frozen first click and recurrent memory, separate signed editor.

    Inherit only first-click, prompt validation, and sequence-forward plumbing.
    A fresh initial_state/initialize always resets memory for a new image.
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.variant = 'directional'
        self.sam_initializer = FrozenSAMInitializer()
        self.click_encoder = ClickEncoder()
        self.spatial_memory = SpatialStateMemory()
        self.directional_decoder = DirectionalEditDecoder()
        self.directional_editor = DirectionalMaskEditor()

    def refine(self, state, points, labels):
        points, labels = self._click(state.image_features, points, labels, state.image_sizes)
        points = torch.cat((state.points, points), dim=1)
        labels = torch.cat((state.labels, labels), dim=1)
        clicks = self.click_encoder(points, labels, state.image_sizes)
        hidden = self.spatial_memory(state.image_features, clicks, state.logits, state.hidden)
        direction, strength = self.directional_decoder(state.image_features, hidden, state.probabilities)
        probabilities, logits = self.directional_editor.update(state.probabilities, state.logits, direction, strength)
        return DirectionalState(state.image_features, state.image_sizes, points, labels, logits,
                                hidden, probabilities, None, direction, strength)
