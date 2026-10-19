"""Research-only checkpoints: reuse the shared frozen SAM checkpoint on load."""

from pathlib import Path

import torch


def research_state_dict(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if not name.startswith("sam_initializer.")}


def save_training_checkpoint(path, model, optimizer, epoch, global_step, config):
    path = Path(path)
    payload = {"format_version": 2, "model": "pims_net_memory128",
               "variant": getattr(model, "variant", "a3"),
               "baseline": "efficientvit-sam-l0", "research_state_dict": research_state_dict(model),
               "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
               "global_step": global_step, "config": config}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    legacy = checkpoint.get("format_version") == 1 and checkpoint.get("model") == "pims_net_intent_memory128"
    current = checkpoint.get("format_version") == 2 and checkpoint.get("model") == "pims_net_memory128"
    if not (legacy or current) or checkpoint.get("baseline") != "efficientvit-sam-l0":
        raise ValueError("Unsupported PIMS-Net checkpoint format")
    if checkpoint.get("variant", "a3") != getattr(model, "variant", "a3"):
        raise ValueError("Checkpoint variant does not match the requested model variant")
    expected = set(research_state_dict(model))
    actual = checkpoint["research_state_dict"]
    if set(actual) != expected:
        raise ValueError("Research checkpoint keys do not match this model")
    model.load_state_dict(actual, strict=False)  # Only frozen SAM keys are omitted by the check above.
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
