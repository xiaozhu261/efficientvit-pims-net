"""A1: predict a complete mask with the same fusion trunk as the edit-intent head."""

from torch import nn

from .edit_intent_decoder import EditIntentDecoder


class DirectMaskDecoder(EditIntentDecoder):
    """Image + ConvGRU state + previous probabilities -> [B,1,256,256] logits.

    The new mask is predicted directly; no residual or ADD/REMOVE/KEEP formula.
    This head is independently trained as part of the A1 variant.
    """

    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, image_features, hidden, previous_mask):
        return self.decode_logits(image_features, hidden, previous_mask)
