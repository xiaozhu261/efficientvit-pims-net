"""Image/binary-mask pairs from a JSONL manifest, with original pixel geometry."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class InteractiveDataset(Dataset):
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.records = []
        for number, line in enumerate(self.manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for key in ("image", "mask"):
                if key not in record:
                    raise ValueError(f"Manifest line {number} requires {key!r}")
            paths = [self.manifest.parent / record[key] for key in ("image", "mask")]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
            self.records.append((str(record.get("id", number)), *paths))
        if not self.records:
            raise ValueError("Manifest contains no samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample_id, image_path, mask_path = self.records[index]
        with Image.open(image_path) as source:
            image = np.array(source.convert("RGB"))
        with Image.open(mask_path) as source:
            mask = np.array(source)
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError(f"{sample_id}: mask must be 2D and match the original image size")
        values = set(np.unique(mask).tolist())
        if not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError(f"{sample_id}: expected a binary 0/1 or 0/255 mask; split instance IDs first")
        mask = mask != 0
        if not mask.any():
            raise ValueError(f"{sample_id}: interactive object training requires foreground")
        return {"id": sample_id, "image": image, "mask": mask}
