from .baseline import build_baseline

__all__ = ["build_baseline", "EncoderAdapter", "ClickEncoder", "InitialMaskDecoder", "ConvGRU", "EditDecoder", "SpatialStateMemory", "EditIntentDecoder", "DirectMaskDecoder", "MaskEditor", "PIMSNet"]


def __getattr__(name):
    # Keep resource-only imports independent of PyTorch and the model dependencies.
    if name == "EncoderAdapter":
        from .encoder_adapter import EncoderAdapter

        return EncoderAdapter
    if name == "ClickEncoder":
        from .click_encoder import ClickEncoder

        return ClickEncoder
    if name == "InitialMaskDecoder":
        from .initial_mask_decoder import InitialMaskDecoder

        return InitialMaskDecoder
    if name == "ConvGRU":
        from .conv_gru import ConvGRU

        return ConvGRU
    if name == "EditDecoder":
        from .edit_decoder import EditDecoder

        return EditDecoder
    if name == "PIMSNet":
        from .pims_net import PIMSNet

        return PIMSNet
    if name == "SpatialStateMemory":
        from .spatial_state_memory import SpatialStateMemory

        return SpatialStateMemory
    if name == "EditIntentDecoder":
        from .edit_intent_decoder import EditIntentDecoder

        return EditIntentDecoder
    if name == "DirectMaskDecoder":
        from .direct_mask_decoder import DirectMaskDecoder

        return DirectMaskDecoder
    if name == "MaskEditor":
        from .mask_editor import MaskEditor

        return MaskEditor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
