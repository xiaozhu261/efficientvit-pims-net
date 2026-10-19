"""Thin entry point to the original model factory; no architecture changes."""

from pims_net.paths import checkpoint_path


def build_baseline(name: str = "efficientvit-sam-l0", *, distilled: bool = False):
    """Load local pretrained weights on CPU; caller selects device and train/eval mode."""
    weights = checkpoint_path(name, distilled=distilled)
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model

    return create_efficientvit_sam_model(name=name, pretrained=True, weight_url=str(weights))
