"""
SpaceNet 6 (SN6) -- Capella Space SAR building-footprint data, the real
illumination-invariant half of PLEM's two-pronged night/dark-image
robustness approach (the augmentation half is `datasets/augment.py`).

SAR is an active sensor (it illuminates its own target with radar and
measures the reflection), so unlike optical imagery it is genuinely
illumination-invariant -- it works identically whether it's day, night, or
overcast. SN6 pairs Capella SAR strips (0.5m, intensity + Pauli-decomposition
bands) with building footprints over Rotterdam, on the same public,
anonymous, unauthenticated `spacenet-dataset` S3 bucket that
`datasets/spacenet.py` already uses for SN2/SN3 -- no registration needed.
SN6 has building-footprint labels only; it has no road/linear annotations at
all, so this module never rasterizes class 1.

Unlike SN2/SN3 (individually listable per-tile S3 objects), SN6 ships as one
large tarball, not per-tile objects -- so acquisition here is
download-then-scan rather than SN2/SN3's list-then-fetch-per-tile pattern.

VERIFIED against a real downloaded tarball on CARC (Sep 2026): the internal
layout is `train/AOI_11_Rotterdam/{SAR-Intensity/*.tif, geojson_buildings/
*.geojson}` -- `_TILE_KEY_RE` / `_SAR_DIR_HINTS` / `_BUILDING_DIR_HINTS`
match it as written. The one thing that bit us: SN6 building geojsons carry
an explicit `crs` member and are in **EPSG:32631 (UTM 31N)**, the same CRS
as the SAR rasters -- NOT the lon/lat that `datasets/common.py` defaults to
for SN2/SN3. `extract_sn6_tile` now reads that CRS (`read_geojson_crs`) and
`rasterize_sn6_tile_labels` threads it through as `src_crs`; without this,
`common.py` reprojected UTM northings as degrees and PROJ raised
"utm: Invalid latitude" on the first footprint. SAR band order/count is
still only used via `mode="replicate"` (band 0 as intensity), so `pauli`
mode stays unverified / `NotImplementedError`.
"""

import json
import re
import tarfile
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors

from datasets.common import (
    download_file,
    load_geojson_features,
    rasterize_polygons,
    read_geojson_crs,
)

BASE_URL = "https://spacenet-dataset.s3.amazonaws.com"
SN6_TARBALL_URL = f"{BASE_URL}/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz"

BUILDING_CLASS_ID = 2

# Directory-name substrings distinguishing SAR-intensity rasters from
# building-footprint geojsons within the tarball -- see this module's
# docstring re: verifying against a real download.
_SAR_DIR_HINTS = ("SAR-Intensity",)
_BUILDING_DIR_HINTS = ("geojson_buildings", "Buildings")

# Matches the tile-identifying suffix of a member's filename once its
# type-specific marker ("SAR-Intensity_" / "Buildings_") is stripped, so SAR
# and building members can be paired up without assuming a specific id format
# (SN6 tile ids are longer/non-numeric, unlike SN2/SN3's simple img{N}).
_TILE_KEY_RE = re.compile(r"(?:SAR-Intensity|Buildings)_(?P<key>.+)\.(?:tif|geojson)$")


def download_sn6_tarball(dest="data/spacenet6_raw/SN6_buildings_AOI_11_Rotterdam_train.tar.gz") -> Path:
    """
    Download the SN6 training tarball via the same plain-anonymous-HTTPS
    mechanism `datasets/common.py::download_file` already uses for SN2/SN3 --
    just a single large object instead of many small ones. This tarball is
    expected to be several GB; only downloaded once (skipped if `dest`
    already exists).
    """
    dest = Path(dest)
    if not dest.exists():
        print(f"Downloading SN6 tarball to {dest} (multi-GB, one-time)...")
    return download_file(SN6_TARBALL_URL, dest)


def _tile_key(name: str):
    m = _TILE_KEY_RE.search(name)
    return m.group("key") if m else None


def list_sn6_members(tarball_path, cache_json=None) -> list:
    """
    Scan a downloaded SN6 tarball once (gzip has no random-access central
    directory, so a full sequential pass is unavoidable -- but this is
    against the local file, not the network, and only needs to happen once
    per tarball) and pair up SAR-intensity rasters with building-footprint
    geojsons by their shared tile key.

    Returns a list of {"tile_key", "sar_member", "building_member"} dicts,
    one per tile with both a matched SAR raster and building geojson.
    Optionally caches the parsed member-name list to `cache_json` so re-runs
    can skip re-scanning the tarball.
    """
    cache_json = Path(cache_json) if cache_json else None
    if cache_json and cache_json.exists():
        import json
        return json.loads(cache_json.read_text())

    sar_members = {}
    building_members = {}
    with tarfile.open(tarball_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if any(h in name for h in _SAR_DIR_HINTS) and name.endswith(".tif"):
                key = _tile_key(name)
                if key:
                    sar_members[key] = name
            elif any(h in name for h in _BUILDING_DIR_HINTS) and name.endswith(".geojson"):
                key = _tile_key(name)
                if key:
                    building_members[key] = name

    common_keys = sorted(set(sar_members) & set(building_members))
    if not common_keys:
        raise RuntimeError(
            "No matching SAR/building tile pairs found in the SN6 tarball -- "
            "the real internal layout likely differs from this module's "
            "assumptions (see datasets/spacenet6.py's module docstring). "
            "Inspect `tarfile.open(tarball_path, 'r:gz').getnames()[:50]` "
            "and adjust `_SAR_DIR_HINTS`/`_BUILDING_DIR_HINTS`/`_TILE_KEY_RE`."
        )

    pairs = [
        {"tile_key": k, "sar_member": sar_members[k], "building_member": building_members[k]}
        for k in common_keys
    ]
    if cache_json:
        import json
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        cache_json.write_text(json.dumps(pairs))
    return pairs


def extract_members_streaming(tarball_path, member_names, tmp_dir) -> None:
    """
    Extract many named members in a SINGLE forward pass over the gzip tarball.

    `.tar.gz` is a non-seekable stream: `tarfile.open(..., "r:gz").extract(m)`
    for a member deep in the archive silently re-decompresses from offset 0
    every call, so extracting N tiles one-open-per-tile is O(N * archive_size)
    -- the cause of the multi-hour `build_spacenet6_sample` runtime on CARC.
    Iterating the archive once and extracting matches as they stream past is
    O(archive_size) total regardless of N.
    """
    wanted = set(member_names)
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as tf:
        for member in tf:  # forward-only streaming iterator, not getmembers()
            if member.name in wanted:
                tf.extract(member, path=tmp_dir)
                wanted.discard(member.name)
                if not wanted:
                    break


def extract_sn6_tile(tarball_path, sar_member: str, building_member: str, tmp_dir) -> tuple:
    """
    Read one tile's SAR raster (bands + geotransform, via rasterio) and its
    building geojson ((geometry, properties) tuples + CRS) from `tmp_dir`,
    extracting the two members from the tarball first if they aren't already
    on disk. `build_spacenet6_sample` bulk-extracts every chosen tile up front
    via `extract_members_streaming` (one archive pass), so in that path the
    files are already present and no per-tile tar access happens here.
    Extracted raw files are the caller's responsibility to clean up.

    Returns (sar_bands, crs, transform, shape, building_geoms, geom_crs) where
    `sar_bands` is a (bands, H, W) float array (rasterio's native band-first
    layout, converted to pseudo-RGB by `sar_bands_to_pseudo_rgb`) and
    `geom_crs` is the CRS the building geometries are actually in -- SN6's
    footprints are in EPSG:32631 (UTM 31N, matching the raster), NOT the
    lon/lat that `datasets/common.py` defaults to for SN2/SN3.
    """
    tmp_dir = Path(tmp_dir)
    sar_path = tmp_dir / sar_member
    building_path = tmp_dir / building_member

    missing = [m for m, p in ((sar_member, sar_path), (building_member, building_path))
               if not p.exists()]
    if missing:
        extract_members_streaming(tarball_path, missing, tmp_dir)

    with rasterio.open(sar_path) as src:
        sar_bands = src.read().astype(np.float64)
        crs, transform, shape = src.crs, src.transform, (src.height, src.width)

    building_geoms = load_geojson_features(building_path)
    geom_crs = read_geojson_crs(building_path, default=str(crs))
    return sar_bands, crs, transform, shape, building_geoms, geom_crs


def sar_bands_to_pseudo_rgb(sar_array: np.ndarray, mode: str = "replicate") -> np.ndarray:
    """
    Convert a SAR raster's raw bands (band-first, e.g. (bands, H, W), any
    numeric dtype) into a 3-channel uint8 pseudo-RGB image, so it can flow
    through PLEM's unmodified `in_ch=3` `SmallUNet` with zero architecture
    change. This trades away SAR-specific phase/polarimetric structure a
    dedicated encoder branch could exploit -- an accepted, documented
    trade-off given this project's moderate scope and its "reuse one
    mechanism across sources" design convention (see `datasets/joint.py`).

    mode="replicate" (default, recommended, band-order-agnostic): takes the
    first band as an intensity image, 2nd/98th-percentile-clips it (SAR
    intensity is heavy-tailed, unlike optical -- a plain min/max normalize
    like `datasets/common.py::read_image_rgb` uses would wash out most
    tiles), normalizes to uint8, and replicates across all 3 channels.

    mode="pauli": classic false-color composite (R=|HH-VV|, G=|HV|,
    B=|HH+VV|) -- raises NotImplementedError until SN6's real band order is
    confirmed against a downloaded tile (see this module's docstring); an
    opt-in future enhancement, not the default.
    """
    if mode == "pauli":
        raise NotImplementedError(
            "mode='pauli' requires SN6's real polarimetric band order/count "
            "verified against a downloaded tile (see datasets/spacenet6.py's "
            "module docstring) -- use mode='replicate' until then."
        )
    if mode != "replicate":
        raise ValueError(f"Unknown mode {mode!r}, expected 'replicate' or 'pauli'")

    intensity = sar_array[0]
    lo, hi = np.percentile(intensity, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((intensity - lo) / (hi - lo), 0, 1)
    band_uint8 = (normalized * 255).astype(np.uint8)
    return np.repeat(band_uint8[:, :, None], 3, axis=-1)


def rasterize_sn6_tile_labels(shape, crs, transform, building_geoms: list,
                              value: int = BUILDING_CLASS_ID, geom_crs=None) -> np.ndarray:
    """
    Rasterize SN6 building-footprint labels onto a tile's pixel grid.
    Thin wrapper around `datasets/common.py::rasterize_polygons` -- never
    writes class 1 (road), since SN6 has no road/linear annotations at all.

    `geom_crs` is the CRS `building_geoms` are in (from
    `read_geojson_crs`); defaults to the raster `crs` when None, since SN6
    footprints and rasters share EPSG:32631. Passing it through is what
    avoids `common.py`'s lon/lat default reprojecting UTM eastings/northings
    as degrees ("PROJ: utm: Invalid latitude").
    """
    return rasterize_polygons(building_geoms, shape, crs, transform, value=value,
                              src_crs=str(geom_crs) if geom_crs is not None else str(crs))


def build_spacenet6_sample(
    n_tiles: int = 300,
    cache_dir="data/spacenet6",
    tarball_path="data/spacenet6_raw/SN6_buildings_AOI_11_Rotterdam_train.tar.gz",
    seed: int = 0,
    pseudo_rgb_mode: str = "replicate",
) -> list:
    """
    Build a curated local sample of up to `n_tiles` real SN6 scenes, cached
    under `cache_dir`. Same list-of-dicts return shape as
    `build_spacenet_sample`/`build_potsdam_sample` -- `{"tile_key", "path",
    "image", "label"}`. Downloads the tarball if missing, lists tile pairs,
    randomly samples `n_tiles` of them, bulk-extracts every chosen tile's two
    members in ONE streaming pass over the archive (see
    `extract_members_streaming` -- per-tile extraction from a `.tar.gz` is
    O(n * archive_size)), then for each: converts SAR to pseudo-RGB,
    rasterizes building-only labels, skips tiles with no building pixels
    landed (mirrors SN2/SN3's own skip rule in
    `datasets/spacenet.py::build_spacenet_sample`), and deletes the raw
    extracted files after each tile (whether it was cached or skipped) so the
    tmp dir never accumulates -- only the tarball itself stays cached under
    gitignored `data/`, so re-runs reopen the local tarball rather than
    re-downloading. Per-tile errors (unreadable raster, malformed geojson,
    un-reprojectable footprint) are logged and skipped, not raised, so one bad
    tile can't abort a build that has already paid for a multi-GB download; the
    skipped-tile count is printed at the end so a systemic failure is visible
    rather than silently returning an empty list.
    """
    tarball_path = Path(tarball_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tarball_path.parent / "_extract_tmp"

    download_sn6_tarball(tarball_path)
    pairs = list_sn6_members(tarball_path, cache_json=tarball_path.parent / "sn6_members.json")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pairs), size=min(n_tiles, len(pairs)), replace=False)

    chosen_pairs = [pairs[i] for i in chosen]
    pending = [
        m for p in chosen_pairs for m in (p["sar_member"], p["building_member"])
        if not (tmp_dir / m).exists()
    ]
    if pending:
        print(f"Extracting {len(pending)} SN6 members in one archive pass...")
        extract_members_streaming(tarball_path, pending, tmp_dir)

    # A bad tile should skip, not crash a build that has already paid for the
    # download. RasterioError covers the CRS-reprojection failure class
    # ("PROJ: utm: Invalid latitude"); the others cover malformed geojson and
    # degenerate geometry.
    _skippable = (
        KeyError, ValueError, json.JSONDecodeError,
        rasterio.errors.RasterioError, rasterio.RasterioIOError,
    )

    samples = []
    n_skipped = 0
    for pair in chosen_pairs:
        cache_path = cache_dir / f"sn6_{pair['tile_key']}.npz"
        try:
            sar_bands, crs, transform, shape, building_geoms, geom_crs = extract_sn6_tile(
                tarball_path, pair["sar_member"], pair["building_member"], tmp_dir,
            )
            label = rasterize_sn6_tile_labels(
                shape, crs, transform, building_geoms, geom_crs=geom_crs,
            )
            if not (label == BUILDING_CLASS_ID).any():
                continue
            image = sar_bands_to_pseudo_rgb(sar_bands, mode=pseudo_rgb_mode)

            np.savez_compressed(cache_path, image=image, label=label)
            samples.append({
                "tile_key": pair["tile_key"], "path": str(cache_path),
                "image": image, "label": label,
            })
        except _skippable as e:
            n_skipped += 1
            print(f"  SN6 tile {pair['tile_key']} skipped: {type(e).__name__}: {e}")
            cache_path.unlink(missing_ok=True)  # drop any half-written cache
        finally:
            for member in (pair["sar_member"], pair["building_member"]):
                extracted = tmp_dir / member
                if extracted.exists():
                    extracted.unlink()

    if n_skipped:
        print(f"SN6: {len(samples)} tiles cached, {n_skipped} skipped.")
    return samples
