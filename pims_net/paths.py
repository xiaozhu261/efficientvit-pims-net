"""Shared resource locations resolved relative to this checkout, not the shell cwd."""

from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RESEARCH_ROOT.parent
CHECKPOINT_ROOT = REPO_ROOT / "assets" / "checkpoints" / "efficientvit_sam"
OUTPUT_ROOT = RESEARCH_ROOT / "outputs"
MODEL_NAMES = tuple(f"efficientvit-sam-{variant}" for variant in ("l0", "l1", "l2", "xl0", "xl1"))


def checkpoint_path(name: str = "efficientvit-sam-l0", *, distilled: bool = False) -> Path:
    """Return an existing local checkpoint; never download or copy weights."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    directory = CHECKPOINT_ROOT / "distilled_model" if distilled else CHECKPOINT_ROOT
    path = directory / (name.replace("-", "_") + ".pt")
    if not path.is_file():
        raise FileNotFoundError(f"Local checkpoint is missing: {path}")
    return path
