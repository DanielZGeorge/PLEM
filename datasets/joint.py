"""
Joint SpaceNet + Potsdam + SpaceNet6 training-source glue.

Each sample is supervised only on the classes its *source* annotates.
SpaceNet (datasets/spacenet.py) has road + building, plus the point feature:
road intersections derived from its own vector road labels
(datasets/intersections.py), stored as a separate per-tile `points` layer
because an intersection lies on a road pixel. Potsdam (datasets/potsdam.py,
with `extract_buildings=True`) is used for buildings only -- it has no road
class, and its tree/car "point" labels (class 3 in its cached label maps) are
dropped at load time: they were the previous point feature and proved
unlearnable (30-60 px blobs at 6 cm/px, unlabelled on the other sources).
SpaceNet6 (datasets/spacenet6.py, real SAR imagery -- the illumination-
invariant half of PLEM's night/dark-image robustness approach) has building
only -- this module is the single source of truth
for that mapping, used both by the joint tile loader here and by
losses/multitask.py::PLEMMultiTaskLoss's caller (the `class_mask` argument).

Pure numpy-in/numpy-out at its boundary, consistent with datasets/'s existing
"decoupled from metrics/" contract -- this module doesn't import torch.
"""

from pathlib import Path

import numpy as np

SOURCE_CLASSES = {
    "spacenet":  [1, 2, 3],  # road, building, point (road intersections)
    "potsdam":   [2],        # building only -- no roads; tree/car points dropped
    "spacenet6": [2],     # building only -- SAR source has no road/point annotations
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


def load_joint_tiles(
    spacenet_dir="data/spacenet", potsdam_dir="data/potsdam", spacenet6_dir="data/spacenet6",
) -> list:
    """
    Loads every cached SpaceNet tile (`data/spacenet/<city>/*.npz`, built by
    `spacenet_data_prep.ipynb`), every cached Potsdam tile
    (`data/potsdam/*.npz`, built by `potsdam_data_prep.ipynb` with
    `extract_buildings=True`), and every cached SpaceNet6 tile
    (`data/spacenet6/*.npz`, built by
    `datasets/spacenet6.py::build_spacenet6_sample`), tagging each with its
    `source` and `class_mask`. Mirrors `train_unet.ipynb`'s existing
    `all_tiles` list-of-dicts loading pattern exactly, extended across three
    sources. Missing/empty directories are skipped rather than raising, so
    this can be called before any cache exists without special-casing the
    caller.

    Returns a list of dicts: `{"tile", "image", "label", "points", "source",
    "class_mask"}` (plus `"city"` for SpaceNet tiles, `None` otherwise).
    `"label"` holds classes 0-2 only. `"points"` is an `(N, 2)` float32
    (row, col) array of road intersections -- empty for Potsdam/SpaceNet6, and
    for a SpaceNet tile cached before intersections existed (one warning is
    printed; run `datasets.spacenet.add_intersections_to_cache` to backfill).
    """
    no_points = np.zeros((0, 2), dtype=np.float32)
    n_missing_points = 0
    spacenet_dir = Path(spacenet_dir)
    potsdam_dir = Path(potsdam_dir)
    spacenet6_dir = Path(spacenet6_dir)
    tiles = []

    if spacenet_dir.is_dir():
        # Expected layout is data/spacenet/<city>/*.npz (one level deep).
        sn_paths = sorted(
            p for city_dir in spacenet_dir.iterdir() if city_dir.is_dir()
            for p in city_dir.glob("*.npz")
        )
        if not sn_paths:
            # Fall back to a recursive scan so a cache written one level too
            # deep (e.g. cache_dir passed with the city already appended, which
            # build_spacenet_sample appends again -> <city>/<city>/*.npz) still
            # loads instead of silently yielding zero SpaceNet tiles.
            sn_paths = sorted(spacenet_dir.rglob("*.npz"))
        for p in sn_paths:
            d = np.load(p)
            if "points" in d.files:
                points = d["points"].astype(np.float32).reshape(-1, 2)
            else:
                points = no_points
                n_missing_points += 1
            tiles.append({
                "city": p.parent.name, "tile": p.stem,
                "image": d["image"], "label": d["label"], "points": points,
                "source": "spacenet", "class_mask": class_mask_for_source("spacenet"),
            })

        if n_missing_points:
            print(f"load_joint_tiles: {n_missing_points}/{len(sn_paths)} SpaceNet tiles have no "
                  f"cached `points` (road intersections) -- run "
                  f"datasets.spacenet.add_intersections_to_cache() to backfill them.")

    if potsdam_dir.is_dir():
        for p in sorted(potsdam_dir.glob("*.npz")):
            d = np.load(p)
            label = d["label"].copy()
            label[label == 3] = 0  # drop the old tree/car point class (see module docstring)
            tiles.append({
                "city": None, "tile": p.stem,
                "image": d["image"], "label": label, "points": no_points,
                "source": "potsdam", "class_mask": class_mask_for_source("potsdam"),
            })

    if spacenet6_dir.is_dir():
        for p in sorted(spacenet6_dir.glob("*.npz")):
            d = np.load(p)
            tiles.append({
                "city": None, "tile": p.stem,
                "image": d["image"], "label": d["label"], "points": no_points,
                "source": "spacenet6", "class_mask": class_mask_for_source("spacenet6"),
            })

    return tiles
