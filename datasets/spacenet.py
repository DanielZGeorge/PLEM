"""
SpaceNet (SN2 buildings + SN3 roads) data acquisition and rasterization.

Both datasets are pulled from the public, unauthenticated `spacenet-dataset`
S3 bucket via plain anonymous HTTPS GET (verified: no AWS credentials,
boto3, or --no-sign-request needed -- individual per-tile files are
directly fetchable, not just multi-GB tarballs).

Important caveat (verified): SN2 building tiles (650x650px) and SN3 road
tiles (1300x1300px) use different tiling grids even for the same city, so
`img{N}` does NOT mean the same geographic tile between the two datasets.
This module resolves that by treating each SN3 road tile's own raster as the
canonical pixel grid, and spatially matching every SN2 building tile whose
true geographic extent overlaps it (`build_building_tile_index`).

Second caveat (found on the fifth real scaled run): building labels do not
cover every road tile completely. A 1300 px road tile spans a 2x2 block of
650 px building tiles, and some of those blocks are simply not in the public
SN2 training set. Each cached tile therefore carries a `building_valid`
mask -- 1 where building labels are known, 0 where they are not -- so the
unknown area is never treated as "no buildings here".
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds
from shapely.geometry import box

from datasets.common import (
    download_file,
    load_geojson_features,
    rasterize_polygons,
    rasterize_lines,
    read_image_rgb,
)
from datasets.intersections import road_intersections, intersections_to_pixels

BASE_URL = "https://spacenet-dataset.s3.amazonaws.com"
AOI_INDEX = {"Vegas": 2, "Paris": 3, "Shanghai": 4, "Khartoum": 5}

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_IMG_ID_RE = re.compile(r"_img(\d+)\.(?:tif|geojson)$")


def _list_keys(prefix: str) -> list:
    """List all object keys under `prefix` in the public bucket (paginated, anonymous)."""
    keys = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(f"{BASE_URL}/", params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall("s3:Contents", _S3_NS):
            keys.append(c.find("s3:Key", _S3_NS).text)
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=_S3_NS) == "true"
        if not truncated:
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
        if not token:
            break
    return keys


def list_tile_ids(city: str, dataset: str = "roads") -> list:
    """
    List all available img{N} tile ids for a city's SN3 roads (geojson_roads)
    or SN2 buildings (geojson_buildings) via anonymous S3 listing.
    """
    aoi = AOI_INDEX[city]
    if dataset == "roads":
        prefix = f"spacenet/SN3_roads/train/AOI_{aoi}_{city}/geojson_roads/"
    elif dataset == "buildings":
        prefix = f"spacenet/SN2_buildings/train/AOI_{aoi}_{city}/geojson_buildings/"
    else:
        raise ValueError(f"Unknown dataset {dataset!r}, use 'roads' or 'buildings'")

    keys = _list_keys(prefix)
    ids = set()
    for k in keys:
        m = _IMG_ID_RE.search(k)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def _road_tif_url(city, aoi, img_id):
    return (f"{BASE_URL}/spacenet/SN3_roads/train/AOI_{aoi}_{city}/PS-RGB/"
            f"SN3_roads_train_AOI_{aoi}_{city}_PS-RGB_img{img_id}.tif")


def _road_geojson_url(city, aoi, img_id):
    return (f"{BASE_URL}/spacenet/SN3_roads/train/AOI_{aoi}_{city}/geojson_roads/"
            f"SN3_roads_train_AOI_{aoi}_{city}_geojson_roads_img{img_id}.geojson")


def _building_geojson_url(city, aoi, img_id):
    return (f"{BASE_URL}/spacenet/SN2_buildings/train/AOI_{aoi}_{city}/geojson_buildings/"
            f"SN2_buildings_train_AOI_{aoi}_{city}_geojson_buildings_img{img_id}.geojson")


def get_tile_geometry(tif_path) -> dict:
    """Read a GeoTIFF's CRS, affine transform, geographic bounds, and pixel shape."""
    with rasterio.open(tif_path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "shape": (src.height, src.width),
        }


def _bounds_lonlat(bounds, crs) -> tuple:
    return tuple(transform_bounds(crs, "EPSG:4326", *bounds))


def _building_tif_url(city, aoi, img_id):
    return (f"{BASE_URL}/spacenet/SN2_buildings/train/AOI_{aoi}_{city}/PS-RGB/"
            f"SN2_buildings_train_AOI_{aoi}_{city}_PS-RGB_img{img_id}.tif")


def _remote_tif_bounds_lonlat(url: str, n_bytes: int = 65536, timeout: float = 30.0) -> tuple:
    """
    Lon/lat bounds of a remote GeoTIFF from its first `n_bytes` only (one HTTP
    Range request, ~0.7 s) -- the georeferencing tags sit at the start of
    SpaceNet's tifs, so the multi-MB pixel data never has to be downloaded.
    """
    r = requests.get(url, headers={"Range": f"bytes=0-{n_bytes - 1}"}, timeout=timeout)
    r.raise_for_status()
    with MemoryFile(r.content) as mf, mf.open() as src:
        return _bounds_lonlat(src.bounds, src.crs)


def build_building_tile_index(city: str, raw_dir, max_workers: int = 16) -> dict:
    """
    `{tile_id: (minx, miny, maxx, maxy)}` lon/lat extent of EVERY SN2 building
    tile in `city`'s public training set, cached as
    `<raw_dir>/building_tile_bounds.json`.

    Extents come from each tile's own tif header, not from its geojson: an
    empty geojson (a labelled tile that simply contains no buildings -- about
    45% of Paris) has no geometry to take a bbox from, and a non-empty one
    only spans its buildings, not the tile. This replaces an earlier coarse
    nearest-neighbour search over tile ids, which missed overlapping tiles
    (e.g. it matched 3 of the building tiles around `Paris_img88` but not the
    one covering its south-west quarter).
    """
    aoi = AOI_INDEX[city]
    raw_dir = Path(raw_dir)
    cache = raw_dir / "building_tile_bounds.json"
    index = {}
    if cache.exists():
        with open(cache, "r") as f:
            index = {int(k): tuple(v) for k, v in json.load(f).items()}
    ids = list_tile_ids(city, "buildings")
    missing = [i for i in ids if i not in index]
    if not missing:
        return index

    def fetch(tid):
        for attempt in range(3):
            try:
                return tid, _remote_tif_bounds_lonlat(_building_tif_url(city, aoi, tid))
            except (requests.RequestException, rasterio.errors.RasterioError):
                if attempt == 2:
                    return tid, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(fetch, missing))
    n_failed = sum(1 for _, b in results if b is None)
    index.update({tid: b for tid, b in results if b is not None})
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({str(k): list(v) for k, v in sorted(index.items())}, f)
    print(f"build_building_tile_index({city}): {len(index)} building-tile extents "
          f"({len(missing) - n_failed} read now, {n_failed} failed)")
    return index


def overlapping_building_tiles(target_bbox, index: dict, min_overlap_frac: float = 0.01) -> list:
    """
    Ids of building tiles whose extent genuinely overlaps `target_bbox`: the
    overlap must be at least `min_overlap_frac` of the building tile's own
    area, so tiles that merely touch along an edge are excluded.
    """
    tx0, ty0, tx1, ty1 = target_bbox
    out = []
    for tid, (x0, y0, x1, y1) in index.items():
        w = min(tx1, x1) - max(tx0, x0)
        h = min(ty1, y1) - max(ty0, y0)
        if w > 0 and h > 0 and w * h >= min_overlap_frac * (x1 - x0) * (y1 - y0):
            out.append(tid)
    return sorted(out)


def building_valid_mask(shape, crs, transform, tile_bboxes: list) -> np.ndarray:
    """`H x W` uint8: 1 inside the union of the given building-tile extents
    (building labels are known there), 0 elsewhere (unknown, NOT "no building")."""
    if not tile_bboxes:
        return np.zeros(shape, dtype=np.uint8)
    return rasterize_polygons([(box(*b), {}) for b in tile_bboxes], shape, crs, transform, value=1)


def tile_building_labels(city: str, geom_info: dict, index: dict, building_geojson_dir) -> tuple:
    """
    Every building footprint overlapping one road tile, plus that tile's
    `building_valid` mask. Returns `(building_geoms, valid_mask, tile_ids)`.
    """
    aoi = AOI_INDEX[city]
    road_bbox = _bounds_lonlat(geom_info["bounds"], geom_info["crs"])
    ids = overlapping_building_tiles(road_bbox, index)
    geoms, covered = [], []
    for bid in ids:
        dest = Path(building_geojson_dir) / f"img{bid}.geojson"
        try:
            download_file(_building_geojson_url(city, aoi, bid), dest)
        except requests.RequestException:
            continue  # labels unavailable -> leave this tile's area marked unknown
        geoms.extend(load_geojson_features(dest))
        covered.append(index[bid])
    valid = building_valid_mask(geom_info["shape"], geom_info["crs"], geom_info["transform"], covered)
    return geoms, valid, ids


def rasterize_tile_labels(
    shape, crs, transform,
    road_geoms: list, building_geoms: list,
    lane_width_m: float = 3.0, gsd_m: float = 0.3,
) -> np.ndarray:
    """
    Rasterize road (class 1) and building (class 2) vector labels onto a
    tile's pixel grid. Buildings are drawn after (on top of) roads, since
    building footprints are more precisely surveyed than lane-buffered road
    polygons -- on overlap, the building class wins.
    """
    label = np.zeros(shape, dtype=np.uint8)
    if road_geoms:
        road_mask = rasterize_lines(
            road_geoms, shape, crs, transform,
            lane_width_m=lane_width_m, gsd_m=gsd_m, value=1,
        )
        label[road_mask > 0] = 1
    if building_geoms:
        building_mask = rasterize_polygons(building_geoms, shape, crs, transform, value=2)
        label[building_mask > 0] = 2
    return label


def build_spacenet_sample(
    city: str = "Vegas",
    n_tiles: int = 8,
    cache_dir="data/spacenet",
    seed: int = 0,
    n_road_candidates: int = 30,
) -> list:
    """
    Build a curated local sample of up to `n_tiles` real SpaceNet scenes for
    `city`, each with an RGB image and a rasterized 0/1/2 (bg/road/building)
    label map, cached under `cache_dir/<city>/`.

    Each SN3 road tile defines the pixel grid. Every SN2 building tile whose
    extent overlaps it is found through an exact per-city extent index
    (`build_building_tile_index`) and rasterized onto it. Only road tiles
    with real building pixels are kept. Each `.npz` also stores
    `building_valid` (where building labels are known -- see the module
    docstring) and `points` (road intersections).
    """
    if city not in AOI_INDEX:
        raise ValueError(f"Unknown city {city!r}, choose from {list(AOI_INDEX)}")
    aoi = AOI_INDEX[city]

    cache_dir = Path(cache_dir) / city
    raw_dir = cache_dir / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    road_ids_all = list_tile_ids(city, "roads")
    building_ids_all = list_tile_ids(city, "buildings")
    if not road_ids_all or not building_ids_all:
        raise RuntimeError(f"No SpaceNet tiles found for city={city!r} -- check bucket layout")

    road_candidate_ids = rng.choice(
        road_ids_all, size=min(n_road_candidates, len(road_ids_all)), replace=False
    ).tolist()

    building_geojson_dir = raw_dir / "buildings_geojson"
    building_index = build_building_tile_index(city, raw_dir)

    samples = []
    for rid in road_candidate_ids:
        if len(samples) >= n_tiles:
            break

        tif_dest = raw_dir / "roads_tif" / f"img{rid}.tif"
        geojson_dest = raw_dir / "roads_geojson" / f"img{rid}.geojson"
        try:
            download_file(_road_tif_url(city, aoi, rid), tif_dest)
            download_file(_road_geojson_url(city, aoi, rid), geojson_dest)
        except requests.RequestException:
            continue

        geom_info = get_tile_geometry(tif_dest)
        building_geoms, building_valid, _ = tile_building_labels(
            city, geom_info, building_index, building_geojson_dir,
        )

        if not building_geoms:
            continue  # no matched building coverage for this road tile -- skip

        road_geoms = load_geojson_features(geojson_dest)

        label = rasterize_tile_labels(
            geom_info["shape"], geom_info["crs"], geom_info["transform"],
            road_geoms, building_geoms,
        )
        if not (label == 2).any():
            continue  # overlapping building tiles exist but none puts a footprint in this tile
        image = read_image_rgb(tif_dest)
        points = tile_intersections(road_geoms, geom_info)

        cache_path = cache_dir / f"{city}_img{rid}.npz"
        np.savez_compressed(cache_path, image=image, label=label, points=points,
                            building_valid=building_valid)
        samples.append({
            "tile_id": rid, "city": city, "path": str(cache_path),
            "image": image, "label": label, "points": points,
            "building_valid": building_valid,
        })

    return samples


def tile_intersections(road_geoms: list, geom_info: dict, merge_px: float = 20.0) -> np.ndarray:
    """`(N, 2)` float32 (row, col) road-intersection points for one SN3 road
    tile -- PLEM's point feature (see `datasets/intersections.py`)."""
    return intersections_to_pixels(
        road_intersections(road_geoms), geom_info["crs"], geom_info["transform"],
        geom_info["shape"], merge_px=merge_px,
    )


def add_intersections_to_cache(cache_dir="data/spacenet", overwrite: bool = False) -> dict:
    """
    Backfill the `points` array into SpaceNet `.npz` tiles cached before
    intersections existed, without rebuilding the sample: each tile's raw
    road geojson + tif are already under `<cache_dir>/<city>/raw/` (and
    `download_file` re-fetches either one if it is missing). Tiles that
    already carry `points` are skipped unless `overwrite`.

    Returns `{"updated", "skipped", "failed", "n_points"}` counts.
    """
    cache_dir = Path(cache_dir)
    stats = {"updated": 0, "skipped": 0, "failed": 0, "n_points": 0}
    for npz_path in sorted(cache_dir.rglob("*.npz")):
        m = re.fullmatch(r"(\w+)_img(\d+)", npz_path.stem)
        if m is None or m.group(1) not in AOI_INDEX:
            continue
        city, rid = m.group(1), int(m.group(2))
        with np.load(npz_path) as d:
            arrays = {k: d[k] for k in d.files}
        if "points" in arrays and not overwrite:
            stats["skipped"] += 1
            stats["n_points"] += len(arrays["points"])
            continue
        raw_dir = npz_path.parent / "raw"
        tif_dest = raw_dir / "roads_tif" / f"img{rid}.tif"
        geojson_dest = raw_dir / "roads_geojson" / f"img{rid}.geojson"
        try:
            aoi = AOI_INDEX[city]
            download_file(_road_tif_url(city, aoi, rid), tif_dest)
            download_file(_road_geojson_url(city, aoi, rid), geojson_dest)
            points = tile_intersections(load_geojson_features(geojson_dest), get_tile_geometry(tif_dest))
        except (requests.RequestException, rasterio.errors.RasterioError, ValueError, KeyError) as e:
            print(f"add_intersections_to_cache: {npz_path.name} failed ({type(e).__name__}: {e})")
            stats["failed"] += 1
            continue
        arrays["points"] = points
        np.savez_compressed(npz_path, **arrays)
        stats["updated"] += 1
        stats["n_points"] += len(points)
    return stats


def add_building_coverage_to_cache(cache_dir="data/spacenet", overwrite: bool = False) -> dict:
    """
    Repair the building labels of SpaceNet `.npz` tiles cached before the
    exact building-tile index existed, without rebuilding the sample: for
    each tile, re-rasterize its label map from ALL overlapping building tiles
    (roads unchanged) and add the `building_valid` mask. Images and `points`
    are kept. Tiles that already carry `building_valid` are skipped unless
    `overwrite`.

    Returns `{"updated", "skipped", "failed", "valid_frac", "building_px_before",
    "building_px_after"}`; `valid_frac` is the mean share of tile area with
    known building labels.
    """
    cache_dir = Path(cache_dir)
    stats = {"updated": 0, "skipped": 0, "failed": 0,
             "building_px_before": 0, "building_px_after": 0}
    valid_fracs, indexes = [], {}
    for npz_path in sorted(cache_dir.rglob("*.npz")):
        m = re.fullmatch(r"(\w+)_img(\d+)", npz_path.stem)
        if m is None or m.group(1) not in AOI_INDEX:
            continue
        city, rid = m.group(1), int(m.group(2))
        with np.load(npz_path) as d:
            arrays = {k: d[k] for k in d.files}
        if "building_valid" in arrays and not overwrite:
            stats["skipped"] += 1
            valid_fracs.append(float(arrays["building_valid"].mean()))
            continue
        raw_dir = npz_path.parent / "raw"
        tif_dest = raw_dir / "roads_tif" / f"img{rid}.tif"
        geojson_dest = raw_dir / "roads_geojson" / f"img{rid}.geojson"
        try:
            aoi = AOI_INDEX[city]
            if city not in indexes:
                indexes[city] = build_building_tile_index(city, raw_dir)
            download_file(_road_tif_url(city, aoi, rid), tif_dest)
            download_file(_road_geojson_url(city, aoi, rid), geojson_dest)
            geom_info = get_tile_geometry(tif_dest)
            building_geoms, valid, _ = tile_building_labels(
                city, geom_info, indexes[city], raw_dir / "buildings_geojson",
            )
            label = rasterize_tile_labels(
                geom_info["shape"], geom_info["crs"], geom_info["transform"],
                load_geojson_features(geojson_dest), building_geoms,
            )
        except (requests.RequestException, rasterio.errors.RasterioError, ValueError, KeyError) as e:
            print(f"add_building_coverage_to_cache: {npz_path.name} failed ({type(e).__name__}: {e})")
            stats["failed"] += 1
            continue
        stats["building_px_before"] += int((arrays["label"] == 2).sum())
        stats["building_px_after"] += int((label == 2).sum())
        arrays["label"] = label
        arrays["building_valid"] = valid
        np.savez_compressed(npz_path, **arrays)
        stats["updated"] += 1
        valid_fracs.append(float(valid.mean()))
    stats["valid_frac"] = float(np.mean(valid_fracs)) if valid_fracs else float("nan")
    return stats
