"""Fixed, image-disjoint COCO train2017 subset; never modifies source data."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from pims_net.evaluation.experiment import file_identity


def prepare_subset(coco_root, output, train_count=1024, test_count=256, seed=20261017):
    root, output = Path(coco_root).resolve(), Path(output).resolve()
    annotation = root / "annotations_trainval2017/annotations/instances_train2017.json"
    image_root = root / "train2017/train2017"
    if not annotation.is_file() or not image_root.is_dir():
        raise FileNotFoundError("Expected instances_train2017.json and train2017/train2017 images")
    if min(train_count, test_count) < 1:
        raise ValueError("Both splits must be nonempty")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(annotation))
    rng = np.random.default_rng(seed)
    image_ids = rng.permutation(sorted(coco.imgs))
    records = {"train": [], "heldout": []}
    selected_images = set()
    identities = []
    skipped = 0
    for image_id in image_ids:
        image_id = int(image_id)
        info = coco.imgs[image_id]
        path = image_root / info["file_name"]
        candidates = sorted((ann for ann in coco.imgToAnns[image_id]
                             if not ann.get("iscrowd", 0) and ann.get("area", 0) >= 256), key=lambda ann: ann["id"])
        if not candidates or not path.is_file():
            skipped += 1
            continue
        ann = candidates[int(rng.integers(len(candidates)))]
        mask = coco.annToMask(ann).astype(bool)
        if mask.sum() < 256:
            skipped += 1
            continue
        with Image.open(path) as source:
            if source.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"Image/annotation geometry mismatch: {path}")
            source.verify()
        split = "train" if len(records["train"]) < train_count else "heldout"
        masks = output / split
        masks.mkdir(exist_ok=True)
        mask_path = masks / f"{image_id}_{ann['id']}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        records[split].append({"id": f"coco_{image_id}_{ann['id']}", "image": str(path),
                               "mask": str(mask_path), "image_id": image_id, "annotation_id": ann["id"],
                               "category_id": ann["category_id"], "area": int(mask.sum())})
        selected_images.add(image_id)
        identities.append({"image": file_identity(path), "mask": file_identity(mask_path)})
        if len(records["train"]) == train_count and len(records["heldout"]) == test_count:
            break
    if len(selected_images) != train_count + test_count:
        raise ValueError("Not enough eligible images for requested disjoint subsets")
    for split, items in records.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    metadata = {"source": "COCO train2017", "annotation": file_identity(annotation),
                "split_seed": seed, "train_images": train_count, "heldout_images": test_count,
                "split_policy": "fixed image-disjoint random split of train2017; NOT official val2017",
                "selection": "one random non-crowd instance per image, annotation and decoded area >=256 pixels",
                "skipped_images": skipped, "source_data_modified": False, "files": identities}
    (output / "subset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
