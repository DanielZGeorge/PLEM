"""
Joint SpaceNet + Potsdam training-source glue.

Neither dataset alone supplies all three PLEM feature types: SpaceNet
(datasets/spacenet.py) has road+building but no point class (no point-object
class exists at SpaceNet's ~0.3-0.5m/px GSD); Potsdam (datasets/potsdam.py,
with `extract_buildings=True`) has building+point but no road/linear class at
all (its 6-class palette has no road color). A genuinely joint 0D/1D/2D
training run needs to combine both, with each sample supervised only on the
classes its *source* actually annotates -- this module is the single source
of truth for that mapping, used both by the joint tile loader here and by
losses/multitask.py::PLEMMultiTaskLoss's caller (the `class_mask` argument).

Pure numpy-in/numpy-out at its boundary, consistent with datasets/'s existing
"decoupled from metrics/" contract -- this module doesn't import torch.
"""

from pathlib import Path

import numpy as np

SOURCE_CLASSES = {
    "spacenet": [1, 2],   # road, building -- no point annotations exist
    "potsdam":  [2, 3],   # building, point -- no road annotations exist
}

NUM_CLASSES = 4  # background, road, building, point


def class_mask_for_source(source: str, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    (num_classes,) float32 0/1 array: 1 for background (class 0, always) +
    `SOURCE_CLASSES[source]`, 0 elsewhere.
    """
    if source not in SOURCE_CLASSES:
        raise ValueError(f"Unknown source {source!r}; expected one of {list(SOURCE_CLASSES)}")
    mask = np.zeros(num_classes, dtype=np.float32)
    mask[0] = 1.0
    for c in SOURCE_CLASSES[source]:
        mask[c] = 1.0
    return mask


def load_joint_tiles(spacenet_dir="data/spacenet", potsdam_dir="data/potsdam") -> list:
    """
    Loads every cached SpaceNet tile (`data/spacenet/<city>/*.npz`, built by
    `spacenet_data_prep.ipynb`) and every cached Potsdam tile
    (`data/potsdam/*.npz`, built by `potsdam_data_prep.ipynb` with
    `extract_buildings=True`), tagging each with its `source` and
    `class_mask`. Mirrors `train_unet.ipynb`'s existing `all_tiles`
    list-of-dicts loading pattern exactly, extended across two sources.
    Missing/empty directories are skipped rather than raising, so this can
    be called before either cache exists without special-casing the caller.

    Returns a list of dicts: `{"tile", "image", "label", "source",
    "class_mask"}` (plus `"city"` for SpaceNet tiles, `None` for Potsdam).
    """
    spacenet_dir = Path(spacenet_dir)
    potsdam_dir = Path(potsdam_dir)
    tiles = []

    if spacenet_dir.is_dir():
        for city_dir in sorted(spacenet_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for p in sorted(city_dir.glob("*.npz")):
                d = np.load(p)
                tiles.append({
                    "city": city_dir.name, "tile": p.stem,
                    "image": d["image"], "label": d["label"],
                    "source": "spacenet", "class_mask": class_mask_for_source("spacenet"),
                })

    if potsdam_dir.is_dir():
        for p in sorted(potsdam_dir.glob("*.npz")):
            d = np.load(p)
            tiles.append({
                "city": None, "tile": p.stem,
                "image": d["image"], "label": d["label"],
                "source": "potsdam", "class_mask": class_mask_for_source("potsdam"),
            })

    return tiles
