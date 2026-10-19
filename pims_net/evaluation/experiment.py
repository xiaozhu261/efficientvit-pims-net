"""Provenance and matched-budget checks for independently trained ablations."""

import hashlib
from pathlib import Path


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


MATCHED_CONFIG = ("manifest", "manifest_sha256", "baseline_sha256", "samples", "seed",
                  "epochs", "batch_size", "max_clicks", "max_steps", "lr", "weight_decay",
                  "grad_clip", "loss_weights", "click_protocol", "sam_frozen")


def validate_experiment(checkpoints, baseline_sha256):
    """Reject untrained, misidentified, or unmatched-budget research checkpoints.

    Global step counts attempted optimizer batches. Actual updates may differ
    when an entire batch has perfect frozen first-click predictions.
    """
    if set(checkpoints) != {"a1", "a2", "a3"}:
        raise ValueError("All three independently trained checkpoints are required")
    reference = checkpoints["a3"]
    for variant, checkpoint in checkpoints.items():
        config = checkpoint.get("config", {})
        if checkpoint.get("format_version") != 2 or checkpoint.get("variant") != variant or config.get("variant") != variant:
            raise ValueError(f"{variant}: expected a version-2 checkpoint trained as {variant}")
        missing = [key for key in MATCHED_CONFIG if key not in config]
        if missing:
            raise ValueError(f"{variant}: missing training provenance: {missing}")
        if config.get("optimizer_updates", 0) < 1:
            raise ValueError(f"{variant}: checkpoint has no recorded optimizer updates")
        if config["sam_frozen"] is not True or config["baseline_sha256"] != baseline_sha256:
            raise ValueError(f"{variant}: frozen baseline differs from the evaluation baseline")
        for key in MATCHED_CONFIG:
            if config[key] != reference.get("config", {}).get(key):
                raise ValueError(f"Training configuration mismatch for {variant}: {key}")
        for key in ("epoch", "global_step"):
            if checkpoint.get(key, 0) < 1 or checkpoint[key] != reference.get(key):
                raise ValueError(f"Training budget mismatch for {variant}: {key}")
