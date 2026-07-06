"""
ISPRS Potsdam 2D Semantic Labeling -- point-feature (tree/car) data source.

SpaceNet has no point-object class, and at its ~0.3-0.5 m/px GSD, objects
like lamp posts or manhole covers are sub-pixel anyway (verified during
planning). ISPRS Potsdam (6cm/px aerial ortho) has real `Tree` and `Car`
classes as compact multi-pixel blobs -- a real-world stand-in for PLEM's
point-feature class.

The official ISPRS host requires a manual data-request form (not
scriptable); a Kaggle mirror is used instead, which needs a one-time,
user-side `~/.kaggle/kaggle.json` API token -- this module cannot set that
up on your behalf; see `load_kaggle_dataset`'s docstring.

Potsdam's label rasters are flat colored PNGs/TIFFs with a fixed
class -> RGB palette, not vector data, so turning them into point ground
truth means: threshold to a target class's color, connected-component label
each blob, drop implausible sizes, and (in metrics/point_f1.py, not here)
reduce each remaining blob to a centroid. Keeping that centroid step out of
this module keeps datasets/ decoupled from metrics/ -- this module's only
contract is to produce a plain H×W integer label map.
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label, sum as ndi_sum

# ISPRS Potsdam 2D Semantic Labeling official palette (RGB). Verify against
# the specific Kaggle mirror in use -- some mirrors ship the "eroded
# boundaries" label variant, which uses the same 6-class palette.
PALETTE = {
    "impervious_surface": (255, 255, 255),
    "building":           (0, 0, 255),
    "low_vegetation":     (0, 255, 255),
    "tree":               (0, 255, 0),
    "car":                (255, 255, 0),
    "clutter":            (255, 0, 0),
}

# Trees and cars both count as PLEM "point features" by default, sharing one
# class id. Callers who want them scored separately should call
# label_raster_to_point_mask once per class with different class_id values.
DEFAULT_POINT_CLASSES = ("tree", "car")
POINT_CLASS_ID = 3


def load_kaggle_dataset(dataset: str = "jahidhasan66/isprs-potsdam", dest="data/potsdam_raw") -> Path:
    """
    Download the ISPRS Potsdam mirror from Kaggle.

    Requires a one-time, user-side Kaggle API token: create a free account
    at kaggle.com, go to Account -> Create New API Token, and save the
    downloaded kaggle.json to ~/.kaggle/kaggle.json. This is a manual
    prerequisite this function cannot perform on your behalf -- it raises if
    no token is configured.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi  # local import: optional dep until used
    api = KaggleApi()
    api.authenticate()  # raises if ~/.kaggle/kaggle.json is missing/invalid
    api.dataset_download_files(dataset, path=str(dest), unzip=True)
    return dest


def label_raster_to_point_mask(
    label_rgb: np.ndarray,
    target_classes=DEFAULT_POINT_CLASSES,
    min_area: int = 4,
    max_area: int = 2000,
    class_id: int = POINT_CLASS_ID,
) -> np.ndarray:
    """
    Threshold a Potsdam colored label raster to the given target classes,
    connected-component label each color's blobs, drop blobs outside
    [min_area, max_area] px (speckle noise / merged clumps), and return a
    single H×W uint8 point-class label map (kept blobs marked `class_id`).

    Known limitation: touching instances of the same class merge into one
    connected component and are not split (no watershed) -- a dense cluster
    of adjacent trees/cars may be undercounted as a single larger blob.
    """
    h, w = label_rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for cls in target_classes:
        color = np.array(PALETTE[cls])
        mask = np.all(label_rgb == color, axis=-1)
        if not mask.any():
            continue
        labeled, n = cc_label(mask)
        if n == 0:
            continue
        ids = np.arange(1, n + 1)
        sizes = ndi_sum(np.ones_like(labeled), labeled, index=ids)
        keep_ids = ids[(sizes >= min_area) & (sizes <= max_area)]
        out[np.isin(labeled, keep_ids)] = class_id
    return out


def tile_potsdam(
    image: np.ndarray,
    label_map: np.ndarray,
    tile_size: int = 256,
    stride: int = 256,
    min_instances: int = 1,
) -> list:
    """
    Crop a large Potsdam image + derived point-class label map into
    PLEM-scale (tile_size x tile_size) chips, keeping only crops with at
    least `min_instances` point-class pixels. Returns a list of
    (image_crop, label_crop) tuples.
    """
    h, w = label_map.shape
    crops = []
    for top in range(0, h - tile_size + 1, stride):
        for left in range(0, w - tile_size + 1, stride):
            label_crop = label_map[top:top + tile_size, left:left + tile_size]
            if (label_crop > 0).sum() < min_instances:
                continue
            image_crop = image[top:top + tile_size, left:left + tile_size]
            crops.append((image_crop, label_crop))
    return crops


def _find_subdir(raw_dir: Path, name_substrs) -> Path:
    """Find the first directory under raw_dir whose name contains any of the
    given substrings (tried in order) -- different Potsdam mirrors use
    different folder names (official ISPRS layout vs. this repo's verified
    Kaggle mirror, which ships pre-tiled 300x300 patches under
    patches/Images/ and patches/Labels/)."""
    if not raw_dir.exists():
        return None
    if isinstance(name_substrs, str):
        name_substrs = [name_substrs]
    dirs = [p for p in raw_dir.rglob("*") if p.is_dir()]
    for substr in name_substrs:
        for p in dirs:
            if substr.lower() in p.name.lower():
                return p
    return None


def _matching_rgb_path(label_path: Path, rgb_dir: Path):
    # Official ISPRS naming: "top_potsdam_2_10_label.tif" / "..._RGB.tif".
    # This repo's verified Kaggle mirror instead names patches "Label_{N}.tif"
    # / "Image_{N}.tif" -- try both conventions.
    stem = label_path.stem
    base = stem.replace("_label", "").replace("_Label", "")
    for ext in (".tif", ".png", ".jpg"):
        for suffix in ("_RGB", ""):
            candidate = rgb_dir / f"{base}{suffix}{ext}"
            if candidate.exists():
                return candidate

    m = re.search(r"(\d+)$", stem)
    if m:
        num = m.group(1)
        for ext in (".tif", ".png", ".jpg"):
            for prefix in ("Image_", "image_"):
                candidate = rgb_dir / f"{prefix}{num}{ext}"
                if candidate.exists():
                    return candidate

    matches = list(rgb_dir.glob(f"{base}*"))
    return matches[0] if matches else None


def build_potsdam_sample(
    n_crops: int = 15,
    seed: int = 0,
    cache_dir="data/potsdam",
    raw_dir="data/potsdam_raw",
    tile_size: int = 256,
    target_classes=DEFAULT_POINT_CLASSES,
    min_area: int = 4,
    max_area: int = 2000,
) -> list:
    """
    Build a curated local sample of up to `n_crops` real Potsdam point-feature
    scenes, cached under `cache_dir`. Requires `load_kaggle_dataset()` (or an
    equivalent manual download into `raw_dir`) to have been run first.
    """
    raw_dir = Path(raw_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rgb_dir = _find_subdir(raw_dir, ["2_Ortho_RGB", "RGB", "Images"])
    label_dir = _find_subdir(raw_dir, ["5_Labels_all", "Labels"])
    if rgb_dir is None or label_dir is None:
        raise FileNotFoundError(
            f"Could not find Potsdam RGB/label folders under {raw_dir} -- "
            "run load_kaggle_dataset() first (requires a Kaggle API token, "
            "see its docstring)."
        )

    rng = np.random.default_rng(seed)
    label_paths = sorted(label_dir.glob("*.tif")) + sorted(label_dir.glob("*.png"))
    if not label_paths:
        raise FileNotFoundError(f"No label rasters found under {label_dir}")

    all_crops = []
    for label_path in label_paths:
        rgb_path = _matching_rgb_path(label_path, rgb_dir)
        if rgb_path is None:
            continue
        label_rgb = np.array(Image.open(label_path).convert("RGB"))
        image = np.array(Image.open(rgb_path).convert("RGB"))
        point_map = label_raster_to_point_mask(
            label_rgb, target_classes=target_classes, min_area=min_area, max_area=max_area,
        )
        crops = tile_potsdam(image, point_map, tile_size=tile_size, stride=tile_size)
        for img_c, lbl_c in crops:
            all_crops.append((label_path.stem, img_c, lbl_c))
        if len(all_crops) >= n_crops * 4:  # enough pool to subsample from
            break

    if not all_crops:
        raise RuntimeError("No point-class crops found -- check min_area/max_area thresholds")

    n_take = min(n_crops, len(all_crops))
    idx = rng.choice(len(all_crops), size=n_take, replace=False)

    samples = []
    for i in idx:
        tile_name, image_crop, label_crop = all_crops[i]
        cache_path = cache_dir / f"{tile_name}_crop{i}.npz"
        np.savez_compressed(cache_path, image=image_crop, label=label_crop)
        samples.append({
            "tile": tile_name, "path": str(cache_path),
            "image": image_crop, "label": label_crop,
        })
    return samples
